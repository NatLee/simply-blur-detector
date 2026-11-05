"""data/pictures.json lists the repository's own pictures: each must be there, unchanged and listed once."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from blur_detection.config import BLUR_FOLDER, DEMO_IMAGE, IMAGE_EXTENSIONS, SHARP_FOLDER, SPLIT_FOLDERS
from blur_detection.images import read_image

ROOT = Path(__file__).resolve().parents[3]
LIST = ROOT / 'data' / 'pictures.json'
SAMPLES = ROOT / 'src' / 'web' / 'public' / 'samples' / 'samples.json'
#: How many pictures each use has; data/README.md promises these numbers.
COUNTS = {'train': 100, 'test': 20, 'web': 5, 'demo': 1, 'cover': 1}
#: Longer side of the training and test pictures: the pixel-sized blurs of `generate` (config.py) were
#: chosen for pictures of about this size.
DATA_SIDE = 512

pytestmark = pytest.mark.skipif(not LIST.is_file(), reason='the repository data folder is not next to the detector')


def pictures():
    return json.loads(LIST.read_text(encoding='utf-8'))['pictures']


def test_every_listed_picture_is_there_unchanged():
    for picture in pictures():
        path = ROOT / picture['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == picture['sha256'], picture['file']
        height, width = read_image(path).shape[:2]
        assert (width, height) == (picture['width'], picture['height']), picture['file']


def test_each_use_has_its_number_of_pictures_and_each_file_is_listed_once():
    assert Counter(p['use'] for p in pictures()) == COUNTS
    files = [p['file'] for p in pictures()]
    assert len(set(files)) == len(files)


def test_training_and_test_pictures_are_in_their_split_at_the_data_size_with_a_blurred_copy():
    for picture in (p for p in pictures() if p['use'] in SPLIT_FOLDERS):
        path = ROOT / picture['file']
        assert path.parent == ROOT / 'data' / picture['use'] / SHARP_FOLDER, picture['file']
        assert max(picture['width'], picture['height']) == DATA_SIDE, picture['file']
        assert (path.parent.parent / BLUR_FOLDER / path.name).is_file(), f'{picture["file"]} has no blurred copy'


def test_nothing_unlisted_is_trained_on():
    listed = {ROOT / p['file'] for p in pictures()}
    for split in SPLIT_FOLDERS:
        for folder in (SHARP_FOLDER, BLUR_FOLDER):
            for path in (ROOT / 'data' / split / folder).iterdir():
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    assert path.parent.parent / SHARP_FOLDER / path.name in listed, f'{path} is not in {LIST.name}'


def test_the_demo_web_and_cover_pictures():
    by_use = {}
    for picture in pictures():
        by_use.setdefault(picture['use'], []).append(picture)
    assert [ROOT / p['file'] for p in by_use['demo']] == [ROOT / 'data' / Path(DEMO_IMAGE).name]
    for picture in by_use['web'] + by_use['cover']:
        assert picture['width'] == picture['height'], f'{picture["file"]} is not square'
    samples = json.loads(SAMPLES.read_text(encoding='utf-8'))['samples']
    assert sorted(SAMPLES.parent / s['file'] for s in samples) == sorted(ROOT / p['file'] for p in by_use['web'])
