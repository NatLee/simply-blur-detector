"""
Blurring with blur-generator: the random blur that builds the training data, and the growing blur of
the comparison figure.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from blurgenerator import gaussian_blur, lens_blur, motion_blur
from loguru import logger

from .config import (
    BLUR_FOLDER,
    DEFAULT_BLUR_MODE,
    FIGURE_LENS_COMPONENTS,
    FIGURE_LENS_EXPOSURE_GAMMA,
    GENERATE_GAUSSIAN_KERNELS,
    GENERATE_LENS_RADII,
    IMAGE_EXTENSIONS,
    SHARP_FOLDER,
)
from .images import read_image, write_image


@dataclass(frozen=True)
class BlurMode:
    """One kind of blur from blur-generator.

    ``blur`` is the blur-generator function. ``random_params`` picks a random strength for the training
    data. ``ladder_params(row, col)`` gives the settings of one tile of the comparison figure, where the
    blur grows to the right and downwards.
    """
    blur: Callable[..., np.ndarray]
    random_params: Callable[[], Dict[str, float]]
    ladder_params: Callable[[int, int], Dict[str, float]]


def _step(row: int, col: int) -> int:
    """How strong a figure tile is: 1, 3, 5, 7 along the first row, then 2, 6, 10, 14, ..."""
    return (row + 1) * (col * 2 + 1)


#: Every blur mode by name. Strengths are drawn from the global `random` module, so ``random.seed()``
#: makes a run repeatable (blur-generator's motion blur picks its random length and angle from it too).
BLURS: Dict[str, BlurMode] = {
    'lens': BlurMode(
        blur=lens_blur,
        random_params=lambda: {'radius': random.choice(GENERATE_LENS_RADII)},
        ladder_params=lambda row, col: {'radius': _step(row, col) * 2,
                                        'components': FIGURE_LENS_COMPONENTS,
                                        'exposure_gamma': FIGURE_LENS_EXPOSURE_GAMMA},
    ),
    'gaussian': BlurMode(
        blur=gaussian_blur,
        random_params=lambda: {'kernel': random.choice(GENERATE_GAUSSIAN_KERNELS)},
        ladder_params=lambda row, col: {'kernel': _step(row, col) * 3},
    ),
    'motion': BlurMode(
        blur=motion_blur,
        random_params=dict,
        ladder_params=lambda row, col: {'size': _step(row, col) * 10, 'angle': (row + 1) * (col * 2)},
    ),
}

#: Names accepted wherever a blur mode is asked for (``None`` means DEFAULT_BLUR_MODE).
BLUR_MODES = tuple(BLURS)


def get_blur_mode(mode: Optional[str]) -> BlurMode:
    """Look up a blur mode by name; ``None`` gives the default (Gaussian)."""
    name = mode or DEFAULT_BLUR_MODE
    if name not in BLURS:
        raise ValueError(f'Unknown blur mode {mode!r}; use one of {", ".join(BLUR_MODES)} or None.')
    return BLURS[name]


def random_blur(img: np.ndarray) -> np.ndarray:
    """Blur ``img`` with a randomly chosen blur mode at a random strength."""
    mode = random.choice(list(BLURS.values()))
    return mode.blur(img, **mode.random_params())


def ladder_blur(img: np.ndarray, mode: Optional[str], row: int, col: int) -> np.ndarray:
    """Blur ``img`` like tile (``row``, ``col``) of the comparison figure."""
    blur_mode = get_blur_mode(mode)
    return blur_mode.blur(img, **blur_mode.ladder_params(row, col))


def generate_blur_images(data_path: str, seed: Optional[int] = None) -> List[Path]:
    """Write a blurred copy of every image in a ``nonblur`` folder to the ``blur`` folder next to it.

    Existing files in ``blur`` with the same name are replaced. Returns the written paths.
    ``seed`` makes the choice of blurs repeatable; ``None`` picks new ones every time.
    """
    if seed is not None:
        random.seed(seed)
    sharp_images = sorted(
        p for p in Path(data_path).rglob('*')
        if p.parent.name == SHARP_FOLDER and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not sharp_images:
        logger.warning(f'No images found in any "{SHARP_FOLDER}" folder under {data_path}.')
    written = []
    for path in sharp_images:
        target = path.parent.parent / BLUR_FOLDER / path.name
        logger.info(f'{path.parent} -> {target}')
        write_image(target, random_blur(read_image(path)))
        written.append(target)
    return written
