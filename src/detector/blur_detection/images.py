"""
Reading, writing and preparing images. Training, prediction and the web demo all prepare images the
same way (see `prepare_image`), so the model always sees what it was trained on.
"""
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from .config import INPUT_SIZE, WINDOW_FRACTION, WINDOW_OVERLAP

PathLike = Union[str, Path]

#: Encoder settings per file type for `write_image`: PNGs get the smallest (still lossless) files.
WRITE_PARAMS = {'.png': [cv2.IMWRITE_PNG_COMPRESSION, 9]}


def read_image(path: PathLike) -> np.ndarray:
    """Read an image file as an 8-bit BGR array, like ``cv2.imread``.

    Unlike ``cv2.imread`` it also works with non-ASCII paths on Windows, and it raises an error that
    names the file instead of returning ``None`` when the file is not an image.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'{path} is not an image file that OpenCV can read.')
    return img


def write_image(path: PathLike, img: np.ndarray) -> Path:
    """Write ``img`` to ``path`` (format from the extension), creating the folder when needed."""
    path = Path(path)
    try:
        ok, encoded = cv2.imencode(path.suffix, img, WRITE_PARAMS.get(path.suffix.lower(), []))
    except cv2.error:  # OpenCV 5 raises instead of returning False for an unknown extension
        ok = False
    if not ok:
        raise ValueError(f'OpenCV cannot write {path.suffix!r} images ({path}).')
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return path


def check_normal(img: np.ndarray) -> np.ndarray:
    """Scale an image with 0-255 values to 0-1; an image already in 0-1 is returned unchanged."""
    if np.issubdtype(img.dtype, np.integer) or img.max() > 1.:
        img = img / 255.
    return img


def equalize_hist(img: np.ndarray) -> np.ndarray:
    """Equalize the histogram of each channel of an 8-bit image separately."""
    return cv2.merge([cv2.equalizeHist(channel) for channel in cv2.split(img)])


def preprocessing(img: np.ndarray, input_shape: Tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)) -> np.ndarray:
    """Resize to ``input_shape`` (width, height) with bilinear interpolation and scale to 0-1."""
    img = cv2.resize(img, input_shape)
    return check_normal(img)


def prepare_image(img: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """Turn an 8-bit BGR image (as read by `read_image`) into one model input.

    Steps, in this order: equalize the histogram of each channel, BGR to RGB, resize to
    ``input_size`` x ``input_size``, scale to 0-1. Returns a float32 array of shape
    ``(input_size, input_size, 3)``. The web demo repeats exactly these steps in JavaScript.
    """
    img = equalize_hist(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocessing(img, (input_size, input_size)).astype(np.float32)


def sliding_windows(width: int, height: int, fraction: float = WINDOW_FRACTION,
                    overlap: float = WINDOW_OVERLAP) -> List[Tuple[int, int, int]]:
    """Square windows ``(x, y, side)`` that cover a width x height image, like the web heat map.

    The side is ``fraction`` of the shorter image side; neighbours overlap by ``overlap`` of a window;
    the last row and column are moved in to end exactly at the image edge. (The web demo's
    windowGrid() in src/web/public/js/heatmap.js lays out its windows the same way.)
    """
    side = max(1, min(width, height, round(min(width, height) * fraction)))
    stride = max(1, round(side * (1 - overlap)))

    def starts(length):
        positions = list(range(0, length - side + 1, stride))
        if positions[-1] + side < length:
            positions.append(length - side)
        return positions

    return [(x, y, side) for y in starts(height) for x in starts(width)]


def random_crops(img: np.ndarray, count: int, rng: np.random.Generator,
                 min_fraction: float = WINDOW_FRACTION) -> List[np.ndarray]:
    """``count`` random square crops of ``img``, each side between ``min_fraction`` and all of its shorter side.

    Every other crop on average is mirrored left to right: blur looks the same in a mirror, so the model
    gets more variety for free.
    """
    height, width = img.shape[:2]
    short = min(height, width)
    crops = []
    for _ in range(count):
        side = max(1, int(rng.uniform(min_fraction, 1.0) * short))
        y = int(rng.integers(0, height - side + 1))
        x = int(rng.integers(0, width - side + 1))
        crop = img[y:y + side, x:x + side]
        crops.append(np.ascontiguousarray(crop[:, ::-1]) if rng.random() < 0.5 else crop)
    return crops
