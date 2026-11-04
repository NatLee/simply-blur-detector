"""data/danbooru.json, the list of training pictures, must stay well formed (no network needed)."""
import json
import re
from pathlib import Path

import pytest

from blur_detection.config import DANBOORU_LIST_NAME, DANBOORU_RATING, SPLIT_FOLDERS
from blur_detection.danbooru import read_list

LIST = Path(__file__).resolve().parents[3] / 'data' / DANBOORU_LIST_NAME
MD5 = re.compile(r'^[0-9a-f]{32}$')

pytestmark = pytest.mark.skipif(not LIST.is_file(), reason='the repository data folder is not next to the detector')


def test_every_listed_post_can_be_checked_and_credited():
    posts = read_list(str(LIST))
    for post in posts:
        assert isinstance(post['id'], int) and post['id'] > 0
        assert MD5.match(post['md5']) and MD5.match(post['large_md5']), post['id']
        assert post['rating'] == DANBOORU_RATING, post['id']
        assert post['split'] in SPLIT_FOLDERS, post['id']
        assert post['artist'] and ' ' not in post['artist'], post['id']  # one named artist to credit


def test_posts_are_listed_once_and_both_splits_are_used():
    posts = read_list(str(LIST))
    assert len({p['id'] for p in posts}) == len(posts)
    counts = {split: sum(p['split'] == split for p in posts) for split in SPLIT_FOLDERS}
    assert counts == {'train': 103, 'test': 20}  # data/README.md promises these numbers


def test_the_list_says_where_it_comes_from():
    listing = json.loads(LIST.read_text(encoding='utf-8'))
    assert listing['site'].startswith('https://danbooru.donmai.us')
    assert f'rating:{DANBOORU_RATING}' in listing['query']
