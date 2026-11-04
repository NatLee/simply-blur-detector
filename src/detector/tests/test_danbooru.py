"""The Danbooru client and `fetch`, against a fake server: no test talks to the real Danbooru."""
import io
import json
import urllib.error
import urllib.parse

import cv2
import numpy as np
import pytest

from blur_detection.config import DANBOORU_USER_AGENT, FETCH_MAX_SIDE
from blur_detection.danbooru import (
    DanbooruBlocked,
    DanbooruClient,
    DanbooruError,
    fetch_listed,
    md5_of,
    target_path,
    unusable_reason,
)
from blur_detection.images import read_image

from .helpers import sharp_image

PICTURE = cv2.imencode('.jpg', sharp_image(1, size=(600, 900)))[1].tobytes()  # 900 x 600
CDN = 'https://cdn.example/sample.jpg'


def post(post_id, **changes):
    base = {'id': post_id, 'md5': f'md5-{post_id}', 'rating': 'g', 'file_ext': 'jpg', 'is_deleted': False,
            'is_banned': False, 'tag_string_general': '1girl solo smile', 'tag_string_meta': 'highres',
            'tag_string_artist': 'someone', 'large_file_url': f'{CDN}?{post_id}'}
    return {**base, **changes}


def entry(post_id, split='train', large_md5=None):
    return {'id': post_id, 'md5': f'md5-{post_id}', 'large_md5': large_md5 or md5_of(PICTURE), 'split': split,
            'artist': 'someone', 'source': '', 'rating': 'g'}


class FakeResponse(io.BytesIO):
    def __init__(self, body, content_type):
        super().__init__(body)
        self.headers = {'Content-Type': content_type}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeServer:
    """Answers the client's requests; `answers` holds what to send back, in order, for each kind."""

    def __init__(self, posts=(), picture=PICTURE, errors=()):
        self.posts = {p['id']: p for p in posts}
        self.picture = picture
        self.errors = list(errors)  # (status, headers, body) answered first, one per request
        self.seen = []

    def __call__(self, request, timeout):
        self.seen.append(request)
        if self.errors:
            status, headers, body = self.errors.pop(0)
            raise urllib.error.HTTPError(request.full_url, status, 'error', headers, io.BytesIO(body))
        if request.full_url.startswith(CDN):
            return FakeResponse(self.picture, 'image/jpeg')
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        ids = [int(i) for i in query['tags'][0].removeprefix('id:').split(',')]
        body = json.dumps([self.posts[i] for i in ids if i in self.posts]).encode()
        return FakeResponse(body, 'application/json; charset=utf-8')


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def client_for(server, fake_time=None):
    fake_time = fake_time or FakeTime()
    return DanbooruClient(site='https://danbooru.example', open_url=server, clock=fake_time.clock,
                          sleep=fake_time.sleep)


@pytest.mark.parametrize('changes, reason', [
    ({}, None),
    ({'rating': 's'}, 'rated "s"'),
    ({'is_deleted': True}, 'deleted'),
    ({'file_ext': 'mp4'}, 'not a still picture'),
    ({'tag_string_general': '1girl swimsuit'}, 'tagged swimsuit'),
    ({'tag_string_meta': 'ai-generated highres'}, 'tagged ai-generated'),
    ({'tag_string_general': 'blurry_background solo'}, 'tagged blurry_background'),
    ({'large_file_url': None}, 'no picture'),
])
def test_only_still_general_audience_pictures_without_excluded_tags_are_used(changes, reason):
    got = unusable_reason(post(1, **changes))
    assert got is None if reason is None else reason in got


def test_requests_say_who_is_asking_and_come_at_most_once_a_second():
    server, fake_time = FakeServer(posts=[post(1)]), FakeTime()
    client = client_for(server, fake_time)
    client.posts([1])
    client.get(CDN)
    assert [r.get_header('User-agent') for r in server.seen] == [DANBOORU_USER_AGENT] * 2
    assert 'Mozilla' not in DANBOORU_USER_AGENT and 'github.com/NatLee/simply-blur-detector' in DANBOORU_USER_AGENT
    assert fake_time.sleeps == [pytest.approx(1.0)]
    assert client.requests == 2


def test_429_waits_as_long_as_the_answer_says_and_tries_again():
    server, fake_time = FakeServer(errors=[(429, {'Retry-After': '7'}, b'slow down')]), FakeTime()
    client = client_for(server, fake_time)
    assert client.get(CDN) == PICTURE
    assert 7 in fake_time.sleeps and client.requests == 2


def test_server_errors_give_up_after_the_retries():
    server = FakeServer(errors=[(503, {}, b'down')] * 10)
    client = client_for(server)
    with pytest.raises(DanbooruError):
        client.get(CDN)
    assert client.requests == client.retries + 1


@pytest.mark.parametrize('error', [
    (403, {'cf-mitigated': 'challenge'}, b''),
    (503, {}, b'<html>Just a moment... challenge-platform</html>'),
])
def test_a_bot_check_stops_everything_at_once(error):
    client = client_for(FakeServer(errors=[error]))
    with pytest.raises(DanbooruBlocked):
        client.get(CDN)
    assert client.requests == 1  # no second try, no workaround


def test_html_instead_of_json_counts_as_a_bot_check():
    def html_page(request, timeout):
        return FakeResponse(b'<html>Checking your browser</html>', 'text/html')
    with pytest.raises(DanbooruBlocked):
        client_for(html_page).posts([1])


def test_fetch_saves_checked_pictures_and_lists_what_it_skipped(tmp_path):
    server = FakeServer(posts=[post(1), post(2, rating='q'), post(3, md5='other'), post(5)])
    entries = [entry(1), entry(2), entry(3), entry(4, 'test'), entry(5, 'test', large_md5='0' * 32)]
    report = fetch_listed(str(tmp_path), entries, client_for(server))

    saved = target_path(str(tmp_path), entries[0])
    assert report.saved == [saved] and saved.parent.as_posix().endswith('train/nonblur')
    img = read_image(saved)
    assert max(img.shape[:2]) == FETCH_MAX_SIDE and img.shape[:2] == (341, 512)  # 900 x 600, shrunk
    reasons = dict(report.skipped)
    assert set(reasons) == {2, 3, 4, 5}
    assert 'rated "q"' in reasons[2]
    assert 'replaced' in reasons[3]
    assert 'not found' in reasons[4]
    assert 'MD5' in reasons[5]
    assert report.requests == 1 + 2  # one API request for all posts, then the pictures of 1 and 5


def test_pictures_already_on_disk_are_not_downloaded_again(tmp_path):
    entries = [entry(1)]
    fetch_listed(str(tmp_path), entries, client_for(FakeServer(posts=[post(1)])))
    again = FakeServer(posts=[post(1)])
    report = fetch_listed(str(tmp_path), entries, client_for(again))
    assert report.present == [target_path(str(tmp_path), entries[0])] and not report.saved
    assert again.seen == []


def test_files_of_posts_that_left_the_list_are_removed_and_nothing_else(tmp_path):
    fetch_listed(str(tmp_path), [entry(1), entry(2)], client_for(FakeServer(posts=[post(1), post(2)])))
    old_blur = tmp_path / 'train' / 'blur' / 'danbooru-2.png'
    old_blur.parent.mkdir(parents=True)
    old_blur.write_bytes(b'blurred copy')
    own = tmp_path / 'train' / 'nonblur' / 'my-photo.png'
    own.write_bytes(b'not ours')
    # Post 2 leaves the list and post 1 moves to the test split.
    report = fetch_listed(str(tmp_path), [entry(1, 'test')], client_for(FakeServer(posts=[post(1)])))
    assert sorted(p.relative_to(tmp_path).as_posix() for p in report.removed) == [
        'train/blur/danbooru-2.png', 'train/nonblur/danbooru-1.png', 'train/nonblur/danbooru-2.png']
    assert own.is_file()
    assert report.saved == [target_path(str(tmp_path), entry(1, 'test'))]
