import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from blur_detection.blur import generate_blur_images
from blur_detection.config import (
    BLUR_FOLDER,
    DATASET_BLUR_SEED,
    DATASET_PREFIX,
    DATASET_SIDE_RANGE,
    DATASET_SIZES,
    DEMO_IMAGE,
    IMAGE_EXTENSIONS,
)
from blur_detection.images import read_image
from blur_detection.scenes import NEAR, SAMPLE_SEED, draw_dataset, random_scene, robot_city_scene

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / 'data'
SAMPLES = REPO / 'src' / 'web' / 'public' / 'samples'


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


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
    assert sharpness(random_scene(np.random.default_rng(seed), 420, 400).img) > 300


def test_the_robot_city_scene_marks_the_robot_as_near():
    scene = robot_city_scene(np.random.default_rng(SAMPLE_SEED))
    assert scene.img.shape == (480, 640, 3)
    assert scene.depth[400, 200] == NEAR      # the robot's body
    assert scene.depth[20, 600] != NEAR       # the sky


def test_draw_dataset_writes_the_layout_that_training_expects(tmp_path):
    written = draw_dataset(str(tmp_path), {'train': 3, 'test': 2}, seed=5)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in written)
    assert names == [Path(DEMO_IMAGE).name] + [f'test/nonblur/{DATASET_PREFIX}-{n:03}.png' for n in (3, 4)] + [
        f'train/nonblur/{DATASET_PREFIX}-{n:03}.png' for n in (0, 1, 2)]
    low, high = DATASET_SIDE_RANGE
    for path in written:
        h, w = read_image(path).shape[:2]
        assert low <= w <= high and low <= h <= high


@pytest.mark.skipif(not (DATA / 'train').is_dir(), reason='the repository data folder is not next to the detector')
def test_the_repository_data_is_what_draw_and_generate_make(tmp_path):
    # data/README.md promises that `draw` and `generate --seed=1` give back the committed pictures:
    # the drawn ones bit for bit, the blurred ones to within one level, because the blurs use floating
    # point and CPUs round a few values differently (seen on GitHub's x86 and ARM runners).
    draw_dataset(str(tmp_path), DATASET_SIZES)
    generate_blur_images(str(tmp_path), seed=DATASET_BLUR_SEED)
    committed = sorted(p.relative_to(DATA).as_posix() for p in DATA.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    fresh = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    assert committed == fresh
    for name in committed:
        old, new = read_image(DATA / name).astype(int), read_image(tmp_path / name).astype(int)
        if Path(name).parent.name == BLUR_FOLDER:
            assert np.abs(old - new).max() <= 1, name
        else:
            np.testing.assert_array_equal(old, new, err_msg=name)


@pytest.mark.skipif(not SAMPLES.is_dir(), reason='the web demo is not next to the detector')
def test_the_web_samples_come_from_the_robot_city_scene():
    # tools/make_samples.py draws the samples with the same module the training data comes from.
    assert 'sharp.png' in {s['file'] for s in json.loads((SAMPLES / 'samples.json').read_text(encoding='utf-8'))['samples']}
    np.testing.assert_array_equal(read_image(SAMPLES / 'sharp.png'),
                                  robot_city_scene(np.random.default_rng(SAMPLE_SEED)).img)
