"""
Download the training pictures listed in data/danbooru.json from Danbooru's JSON API.

The client is polite on purpose: it says who it is (DANBOORU_USER_AGENT), sends at most one request
per DANBOORU_MIN_INTERVAL seconds, waits and tries again after "429 Too Many Requests", and stops
with `DanbooruBlocked` when a bot check answers instead of Danbooru: it never tries to get past one.
Every downloaded file is checked against the MD5 in the list before it is used.
"""
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from .config import (
    BLUR_FOLDER,
    DANBOORU_BACKOFF,
    DANBOORU_BATCH,
    DANBOORU_EXCLUDED_TAGS,
    DANBOORU_EXTENSIONS,
    DANBOORU_MIN_INTERVAL,
    DANBOORU_RATING,
    DANBOORU_RETRIES,
    DANBOORU_SITE,
    DANBOORU_TIMEOUT,
    DANBOORU_USER_AGENT,
    FETCH_MAX_SIDE,
    FETCH_PREFIX,
    SHARP_FOLDER,
)
from .images import write_image

#: Post fields asked from the API (the rest of a post is not needed).
POST_FIELDS = ('id,md5,rating,score,file_ext,image_width,image_height,large_file_url,file_url,source,'
               'is_deleted,is_banned,tag_string_general,tag_string_meta,tag_string_artist,'
               'tag_string_copyright')
#: HTTP answers worth asking again for, after a wait.
RETRY_STATUSES = (429, 500, 502, 503, 504)


class DanbooruBlocked(RuntimeError):
    """A bot check (Cloudflare or similar) answered instead of Danbooru. Not to be worked around."""


class DanbooruError(RuntimeError):
    """Danbooru kept failing (or answered something that is not JSON) after all the retries."""


def unusable_reason(post: dict, excluded: Sequence[str] = DANBOORU_EXCLUDED_TAGS) -> Optional[str]:
    """Why a post must not be used, or None when it is fine: general rating, still picture, no excluded tag."""
    if post.get('is_deleted') or post.get('is_banned'):
        return 'deleted or banned on Danbooru'
    if post.get('rating') != DANBOORU_RATING:
        return f'rated "{post.get("rating")}", not "{DANBOORU_RATING}"'
    if str(post.get('file_ext', '')).lower() not in DANBOORU_EXTENSIONS:
        return f'a {post.get("file_ext")} file, not a still picture'
    tags = set(f'{post.get("tag_string_general", "")} {post.get("tag_string_meta", "")}'.split())
    found = sorted(tags.intersection(excluded))
    if found:
        return 'tagged ' + ', '.join(found)
    if not str(post.get('large_file_url', '')).startswith('https://'):
        return 'no picture to download'
    return None


@dataclass
class DanbooruClient:
    """Danbooru's JSON API and picture servers, one polite request at a time.

    ``open_url``, ``clock`` and ``sleep`` can be replaced, which the tests do to run without a network.
    ``requests`` counts every request sent, retries included.
    """
    site: str = DANBOORU_SITE
    user_agent: str = DANBOORU_USER_AGENT
    min_interval: float = DANBOORU_MIN_INTERVAL
    retries: int = DANBOORU_RETRIES
    backoff: float = DANBOORU_BACKOFF
    timeout: float = DANBOORU_TIMEOUT
    open_url: Callable = urllib.request.urlopen
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    requests: int = 0
    _last: Optional[float] = field(default=None, repr=False)

    def _wait_turn(self) -> None:
        if self._last is not None:
            wait = self.min_interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()

    def get(self, url: str, want_json: bool = False) -> bytes:
        """Download ``url``. Raises DanbooruBlocked on a bot check, DanbooruError when retries run out."""
        accept = 'application/json' if want_json else 'image/*'
        delay = self.backoff
        for attempt in range(self.retries + 1):
            self._wait_turn()
            self.requests += 1
            request = urllib.request.Request(url, headers={'User-Agent': self.user_agent, 'Accept': accept})
            try:
                with self.open_url(request, timeout=self.timeout) as response:
                    body = response.read()
                    content_type = response.headers.get('Content-Type', '')
            except urllib.error.HTTPError as err:
                if err.headers.get('cf-mitigated') or (err.code in (403, 503) and b'challenge' in err.read()[:4096].lower()):
                    raise DanbooruBlocked(f'a bot check answered {url} (HTTP {err.code}); stopping, not working around it')
                if err.code in RETRY_STATUSES and attempt < self.retries:
                    wait = float(err.headers.get('Retry-After') or delay)
                    logger.warning(f'HTTP {err.code} for {url}; waiting {wait:.0f} s before trying again')
                    self.sleep(wait)
                    delay *= 2
                    continue
                raise DanbooruError(f'HTTP {err.code} for {url}') from err
            if want_json and 'json' not in content_type:
                raise DanbooruBlocked(f'{url} answered {content_type or "no content type"} instead of JSON; '
                                      'probably a bot check, stopping')
            return body
        raise DanbooruError(f'gave up on {url}')

    def posts(self, ids: Iterable[int], batch: int = DANBOORU_BATCH) -> Dict[int, dict]:
        """The posts with these ids, by id (missing ids are simply not in the result)."""
        ids = list(ids)
        found: Dict[int, dict] = {}
        for start in range(0, len(ids), batch):
            chunk = ids[start:start + batch]
            query = urllib.parse.urlencode({'tags': 'id:' + ','.join(map(str, chunk)), 'limit': len(chunk),
                                            'only': POST_FIELDS})
            for post in json.loads(self.get(f'{self.site}/posts.json?{query}', want_json=True)):
                found[int(post['id'])] = post
        return found

    def search(self, tags: str, page: int, limit: int = 200) -> List[dict]:
        """One page of a tag search (used to pick the list, see tools/pick_danbooru.py)."""
        query = urllib.parse.urlencode({'tags': tags, 'page': page, 'limit': limit, 'only': POST_FIELDS})
        return json.loads(self.get(f'{self.site}/posts.json?{query}', want_json=True))


def md5_of(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def shrink(img: np.ndarray, max_side: int = FETCH_MAX_SIDE) -> np.ndarray:
    """Scale ``img`` down (area averaging) so that its longer side is at most ``max_side``."""
    scale = max_side / max(img.shape[:2])
    if scale >= 1:
        return img
    size = (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def target_path(data_path: str, entry: dict) -> Path:
    """Where a listed post is saved: <data>/<split>/nonblur/danbooru-<id>.png."""
    return Path(data_path) / entry['split'] / SHARP_FOLDER / f'{FETCH_PREFIX}-{entry["id"]}.png'


def read_list(list_path: str) -> List[dict]:
    """The posts of a list file (data/danbooru.json)."""
    return json.loads(Path(list_path).read_text(encoding='utf-8'))['posts']


@dataclass
class FetchReport:
    """What `fetch_listed` did: saved and already present files, skipped posts with the reason, removed
    files of posts that are no longer listed, and the requests sent."""
    saved: List[Path] = field(default_factory=list)
    present: List[Path] = field(default_factory=list)
    skipped: List[Tuple[int, str]] = field(default_factory=list)
    removed: List[Path] = field(default_factory=list)
    requests: int = 0


def fetch_listed(data_path: str, entries: Sequence[dict], client: Optional[DanbooruClient] = None) -> FetchReport:
    """Download every listed post that is not on disk yet, check it and save it for training.

    A post is skipped (and reported) when Danbooru no longer has it, when it is no longer usable (see
    `unusable_reason`), when its file changed, or when the download does not match the list's MD5.
    Files this function saved earlier for posts that are no longer listed (or now belong to the other
    split) are removed, with their blurred copies, so that the folder trains on the list and nothing
    else. Other files are left alone.
    """
    client = client or DanbooruClient()
    report = FetchReport()
    todo = []
    for entry in entries:
        path = target_path(data_path, entry)
        if path.is_file():
            report.present.append(path)
        else:
            todo.append(entry)
    posts = client.posts([entry['id'] for entry in todo]) if todo else {}
    for entry in todo:
        post = posts.get(entry['id'])
        reason = 'not found on Danbooru' if post is None else unusable_reason(post)
        if reason is None and post.get('md5') != entry['md5']:
            reason = 'its file was replaced on Danbooru'
        if reason is None:
            data = client.get(post['large_file_url'])
            if md5_of(data) != entry['large_md5']:
                reason = 'the download does not match the MD5 in the list'
        if reason is not None:
            report.skipped.append((entry['id'], reason))
            logger.warning(f'Skipping post {entry["id"]}: {reason}')
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        report.saved.append(write_image(target_path(data_path, entry), shrink(img)))
    report.removed = remove_unlisted(data_path, entries)
    report.requests = client.requests
    return report


def remove_unlisted(data_path: str, entries: Sequence[dict]) -> List[Path]:
    """Delete ``<split>/{nonblur,blur}/danbooru-*.png`` files whose post is not listed for that split."""
    wanted = {target_path(data_path, entry).relative_to(data_path).as_posix() for entry in entries}
    removed = []
    for path in sorted(Path(data_path).glob(f'*/*/{FETCH_PREFIX}-*.png')):
        sharp = Path(path.parent.parent.name) / SHARP_FOLDER / path.name
        if path.parent.name in (SHARP_FOLDER, BLUR_FOLDER) and sharp.as_posix() not in wanted:
            path.unlink()
            removed.append(path)
    return removed
