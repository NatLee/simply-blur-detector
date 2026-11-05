import numpy as np
import pytest

from blur_detection import find_images, get_data
from blur_detection.config import INPUT_SIZE
from blur_detection.images import write_image

from .helpers import sharp_image


def test_find_images_labels_by_folder(data_dir):
    found = find_images(data_dir)
    assert [len(found[split]) for split in ('train', 'test')] == [6, 4]
    labels = {path.parent.name: label for path, label in found['train']}
    assert labels == {'nonblur': 0, 'blur': 1}


def test_find_images_skips_other_folders_and_matches_extensions_in_any_case(data_dir):
    # Regression: an image in an unknown folder used to be added without a label, so the image and
    # label counts no longer matched and building the dataset failed. Upper-case extensions were skipped.
    write_image(data_dir / 'train' / 'misc' / 'stray.png', sharp_image(5))
    write_image(data_dir / 'demo.png', sharp_image(6))
    write_image(data_dir / 'train' / 'nonblur' / 'UPPER.PNG', sharp_image(7))
    names = [path.name for path, _ in find_images(data_dir)['train']]
    assert 'stray.png' not in names and 'demo.png' not in names
    assert 'UPPER.PNG' in names


def test_find_images_leaves_a_data_set_in_a_subfolder_alone(data_dir):
    # Regression: find_images searched every subfolder, so a second data set kept inside the data
    # folder (say data/drawn/train/nonblur from `draw --data_path=data/drawn`) was trained on too.
    write_image(data_dir / 'drawn' / 'train' / 'nonblur' / 'nested.png', sharp_image(8))
    write_image(data_dir / 'train' / 'nonblur' / 'deeper' / 'nested.png', sharp_image(9))
    assert all(path.name != 'nested.png' for split in find_images(data_dir).values() for path, _ in split)


def test_get_data_gives_batches_of_prepared_images_and_one_hot_labels(data_dir):
    train_ds, test_ds = get_data(str(data_dir), batch_size=4)
    x, y = next(iter(train_ds))
    assert tuple(x.shape) == (4, INPUT_SIZE, INPUT_SIZE, 3)
    assert tuple(y.shape) == (4, 2)
    assert sum(int(b[0].shape[0]) for b in train_ds) == 6
    assert sum(int(b[0].shape[0]) for b in test_ds) == 4
    np.testing.assert_array_equal(np.sum(y, axis=1), 1)


def test_get_data_shuffles_every_training_image(tmp_path):
    # Regression: shuffle(seed) used the seed (87) as the shuffle buffer size. With 101 images the
    # last file could then never be among the first 14 of an epoch (101 - 87 = 14).
    root = tmp_path / 'data'
    black, white = (np.full((8, 8, 3), v, dtype=np.uint8) for v in (0, 255))
    for i in range(100):
        write_image(root / 'train' / 'nonblur' / f'{i:03}.png', black)
    write_image(root / 'train' / 'nonblur' / 'zzz.png', white)  # sorted last
    write_image(root / 'test' / 'nonblur' / 't.png', black)
    write_image(root / 'test' / 'blur' / 't.png', black)

    train_ds, _ = get_data(str(root), batch_size=101, seed=0)
    positions = [int(np.argmax(np.max(next(iter(train_ds))[0], axis=(1, 2, 3)))) for _ in range(30)]
    assert min(positions) < 101 - 87


def test_get_data_explains_an_empty_split(tmp_path):
    write_image(tmp_path / 'train' / 'nonblur' / 'a.png', sharp_image(1))
    with pytest.raises(ValueError, match='test'):
        get_data(str(tmp_path))


def test_get_data_with_crops_adds_crops_of_the_training_images_only(data_dir):
    train_ds, test_ds = get_data(str(data_dir), crops=2, seed=1)
    assert sum(int(b[0].shape[0]) for b in train_ds) == 6 * 3  # every image plus 2 crops
    assert sum(int(b[0].shape[0]) for b in test_ds) == 4       # the test images stay whole
    labels = np.concatenate([np.argmax(b[1], axis=1) for b in train_ds])
    assert sorted(labels.tolist()) == [0] * 9 + [1] * 9        # crops keep their image's label
