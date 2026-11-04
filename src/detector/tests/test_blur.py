import numpy as np
import pytest
from blurgenerator import gaussian_blur, lens_blur, motion_blur

from blur_detection.blur import BLUR_MODES, generate_blur_images, get_blur_mode, ladder_blur
from blur_detection.images import read_image, write_image

from .helpers import sharp_image

#: Settings of the figure tiles as the original predict_blur_img computed them (row, col -> params).
ORIGINAL_LADDER = {
    'motion': lambda r, c: {'size': (r + 1) * (c * 2 + 1) * 10, 'angle': (r + 1) * (c * 2)},
    'lens': lambda r, c: {'radius': (r + 1) * (c * 2 + 1) * 2, 'components': 4, 'exposure_gamma': 2},
    'gaussian': lambda r, c: {'kernel': (r + 1) * (c * 2 + 1) * 3},
}


def test_the_three_blurs_of_blur_generator_are_available():
    assert set(BLUR_MODES) == {'lens', 'gaussian', 'motion'}
    assert get_blur_mode('lens').blur is lens_blur
    assert get_blur_mode('motion').blur is motion_blur
    assert get_blur_mode(None).blur is gaussian_blur  # None means Gaussian, as before


@pytest.mark.parametrize('mode', sorted(ORIGINAL_LADDER))
def test_figure_tiles_use_the_original_blur_strengths(mode):
    for row in range(4):
        for col in range(4):
            assert get_blur_mode(mode).ladder_params(row, col) == ORIGINAL_LADDER[mode](row, col)


def test_ladder_blur_matches_calling_blur_generator_directly():
    img = sharp_image(1, size=(64, 64))
    np.testing.assert_array_equal(ladder_blur(img, 'gaussian', 1, 2), gaussian_blur(img, kernel=30))


def test_unknown_blur_mode_is_explained():
    with pytest.raises(ValueError, match='lens, gaussian, motion'):
        get_blur_mode('zoom')


def test_generate_writes_a_blurred_copy_next_to_every_sharp_image(tmp_path):
    for split in ('train', 'test'):
        write_image(tmp_path / split / 'nonblur' / 'a.png', sharp_image(1, size=(64, 80)))
    write_image(tmp_path / 'train' / 'nonblur' / 'b.JPG', sharp_image(2, size=(64, 80)))

    written = generate_blur_images(str(tmp_path), seed=3)

    # The blur folders did not exist; they are created (cv2.imwrite used to fail silently).
    assert sorted(p.relative_to(tmp_path).as_posix() for p in written) == [
        'test/blur/a.png', 'train/blur/a.png', 'train/blur/b.JPG']
    for path in written:
        sharp = read_image(path.parent.parent / 'nonblur' / path.name)
        blurred = read_image(path)
        assert blurred.shape == sharp.shape
        assert not np.array_equal(blurred, sharp)


def test_generate_with_a_seed_is_repeatable(tmp_path):
    write_image(tmp_path / 'train' / 'nonblur' / 'a.png', sharp_image(1, size=(64, 80)))
    first = read_image(generate_blur_images(str(tmp_path), seed=4)[0])
    second = read_image(generate_blur_images(str(tmp_path), seed=4)[0])
    np.testing.assert_array_equal(first, second)
