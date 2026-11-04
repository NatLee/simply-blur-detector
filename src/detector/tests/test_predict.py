import keras
import numpy as np
import pytest

from blur_detection import DetectionModel, prepare_image, predict_blur_img
from blur_detection.blur import ladder_blur
from blur_detection.config import FIGURE_GRID
from blur_detection.images import read_image, write_image

from .helpers import sharp_image


@pytest.fixture
def detector():
    keras.utils.set_random_seed(5)
    return DetectionModel()


@pytest.fixture
def image_path(tmp_path):
    return write_image(tmp_path / 'demo.png', sharp_image(8, size=(120, 100)))


@pytest.mark.parametrize('blur_mode', [None, 'lens', 'gaussian', 'motion'])
def test_predict_blur_img_draws_the_comparison_figure(detector, image_path, tmp_path, blur_mode):
    figure = tmp_path / 'out' / 'results.png'
    grid = predict_blur_img(detector, str(image_path), blur_mode=blur_mode, save_fig_path=str(figure))
    assert read_image(figure).shape[2] == 3  # a real PNG, in a folder that did not exist yet
    assert [len(row) for row in grid] == [FIGURE_GRID[1]] * FIGURE_GRID[0]
    for label, p_blur in (tile for row in grid for tile in row):
        assert label in ('Clear', 'Blur') and 0 <= p_blur <= 1


def test_every_tile_is_blurred_first_and_prepared_like_training_data(detector, image_path, tmp_path):
    # Regression: the figure equalized the histogram before blurring, while training images are blurred
    # first and equalized afterwards, so the model was shown images unlike the ones it learned from.
    grid = predict_blur_img(detector, str(image_path), blur_mode='gaussian', save_fig=False)
    img = read_image(image_path)
    for row, col in [(0, 0), (2, 3)]:
        expected = np.asarray(detector.predict(prepare_image(ladder_blur(img, 'gaussian', row, col)))[0])[0, 1]
        assert grid[row][col][1] == pytest.approx(float(expected), abs=1e-6)
