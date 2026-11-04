import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from blur_detection.images import write_image  # noqa: E402

from .helpers import blurry_image, sharp_image  # noqa: E402


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A tiny data folder in the expected layout: 3 sharp + 3 blurry training images, 2 + 2 test images."""
    root = tmp_path / 'data'
    for split, seeds in (('train', range(3)), ('test', range(10, 12))):
        for seed in seeds:
            write_image(root / split / 'nonblur' / f'img{seed}.png', sharp_image(seed))
            write_image(root / split / 'blur' / f'img{seed}.png', blurry_image(seed))
    return root
