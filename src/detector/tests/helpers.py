"""Synthetic test images (no third-party pictures needed)."""
import cv2
import numpy as np


def sharp_image(seed: int, size=(280, 300)) -> np.ndarray:
    """A synthetic sharp BGR image: flat shapes with dark outlines, a bit like line art."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w, 3), rng.integers(150, 256, 3), dtype=np.uint8)
    for _ in range(12):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        radius = int(rng.integers(10, 60))
        cv2.circle(img, center, radius, color, -1)
        cv2.circle(img, center, radius, (20, 20, 20), 2)
    cv2.putText(img, f'#{seed}', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


def blurry_image(seed: int) -> np.ndarray:
    return cv2.GaussianBlur(sharp_image(seed), (31, 31), 8)
