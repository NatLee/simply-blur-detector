from pathlib import Path

import numpy as np
import pytest

from blur_detection.config import (
    DATASET_PREFIX,
    DATASET_SIDE_RANGE,
    DATASET_SIZES,
    DEMO_IMAGE,
)
from blur_detection.images import read_image
from blur_detection.scenes import draw_dataset, random_scene

from .helpers import laplacian_variance

def test_random_scenes_are_repeatable_and_differ_by_seed():
    a = random_scene(np.random.default_rng(1), 300, 260).img
    b = random_scene(np.random.default_rng(1), 300, 260).img
    c = random_scene(np.random.default_rng(2), 300, 260).img
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (260, 300, 3)


@pytest.mark.parametrize('seed', range(8))
def test_random_scenes_are_crisp_line_art(seed):
    # Sharp edges everywhere: far above the usual "blurry" threshold of the Laplacian variance (100).
    assert laplacian_variance(random_scene(np.random.default_rng(seed), 420, 400).img) > 300


def test_draw_dataset_writes_the_layout_that_training_expects(tmp_path):
    written = draw_dataset(str(tmp_path), {'train': 3, 'test': 2}, seed=5)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in written)
    assert names == [Path(DEMO_IMAGE).name] + [f'test/nonblur/{DATASET_PREFIX}-{n:03}.png' for n in (3, 4)] + [
        f'train/nonblur/{DATASET_PREFIX}-{n:03}.png' for n in (0, 1, 2)]
    low, high = DATASET_SIDE_RANGE
    for path in written:
        h, w = read_image(path).shape[:2]
        assert low <= w <= high and low <= h <= high


def test_the_drawn_demo_picture_does_not_depend_on_the_size_of_the_data_set(tmp_path):
    # It is drawn first, so a drawn data set of any size comes with the same demo.png. (The repository's
    # data/demo.png is a picture of its own since 2026-11-05; test_pictures.py checks it.)
    name = Path(DEMO_IMAGE).name
    draw_dataset(str(tmp_path / 'small'), {'train': 1, 'test': 1})
    draw_dataset(str(tmp_path / 'large'), {'train': 4, 'test': 2})
    np.testing.assert_array_equal(read_image(tmp_path / 'small' / name), read_image(tmp_path / 'large' / name))
