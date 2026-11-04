import cv2
import numpy as np
import pytest

from blur_detection import check_normal, equalize_hist, prepare_image, preprocessing
from blur_detection.config import INPUT_SIZE
from blur_detection.images import random_crops, read_image, sliding_windows, write_image

from .helpers import sharp_image


def test_write_then_read_gives_the_same_pixels_even_with_a_non_ascii_path(tmp_path):
    img = sharp_image(1)
    path = write_image(tmp_path / '模糊' / 'ミカ.png', img)  # the folder is created
    np.testing.assert_array_equal(read_image(path), img)


def test_read_image_names_the_file_when_it_is_not_an_image(tmp_path):
    path = tmp_path / 'notes.png'
    path.write_text('not an image')
    with pytest.raises(ValueError, match='notes.png'):
        read_image(path)


def test_write_image_rejects_an_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="'.xyz'"):
        write_image(tmp_path / 'out.xyz', sharp_image(1))


@pytest.mark.parametrize('img, expected_max', [
    (np.array([[0, 255]], dtype=np.uint8), 1.0),        # 8-bit: scaled
    (np.array([[0., 128., 255.]]), 1.0),                # floats in 0-255: scaled
    (np.array([[0., 0.5, 1.]]), 1.0),                   # already 0-1: unchanged
    (np.array([[0., 2., 4.]]), 4 / 255),                # dark float image in 0-255: scaled (was left as is)
])
def test_check_normal(img, expected_max):
    assert check_normal(img).max() == pytest.approx(expected_max)


def test_equalize_hist_equalizes_every_channel_on_its_own():
    img = sharp_image(2)
    out = equalize_hist(img)
    for c in range(3):
        np.testing.assert_array_equal(out[:, :, c], cv2.equalizeHist(img[:, :, c]))


def test_preprocessing_resizes_and_scales():
    out = preprocessing(sharp_image(3))
    assert out.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert 0 <= out.min() and out.max() <= 1


def test_prepare_image_gives_a_float32_rgb_model_input():
    bgr = np.zeros((40, 50, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # pure blue in OpenCV's BGR order; a flat channel stays flat when equalized
    out = prepare_image(bgr)
    assert out.shape == (INPUT_SIZE, INPUT_SIZE, 3) and out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [0, 0, 1])  # blue is the last channel in RGB


def test_sliding_windows_cover_the_image_like_the_web_heat_map():
    # Same layout as windowGrid() in src/web/tests/heatmap.test.js: 640 x 480 -> 240 px windows.
    windows = sliding_windows(640, 480)
    assert {side for _, _, side in windows} == {240}
    assert sorted({x for x, _, _ in windows}) == [0, 120, 240, 360, 400]
    assert sorted({y for _, y, _ in windows}) == [0, 120, 240]


def test_random_crops_are_squares_between_half_and_all_of_the_shorter_side():
    img = sharp_image(4, size=(100, 160))
    crops = random_crops(img, 50, np.random.default_rng(0))
    assert len(crops) == 50
    for crop in crops:
        assert crop.shape[0] == crop.shape[1] and 50 <= crop.shape[0] <= 100


def test_some_random_crops_are_mirrored_copies_of_the_picture():
    img = sharp_image(5, size=(120, 120))
    crops = random_crops(img, 40, np.random.default_rng(1), min_fraction=1.0)  # whole picture, so easy to compare
    mirrored = sum(np.array_equal(c, img[:, ::-1]) for c in crops)
    straight = sum(np.array_equal(c, img) for c in crops)
    assert mirrored + straight == 40 and 10 < mirrored < 30
