"""
How data/danbooru.json was chosen. A maintainer tool, not part of the package: it needs a person.

1. ``search``: page through Danbooru's best-scored general-audience posts (``rating:g order:score``),
   keep the ones `unusable_reason` accepts that are big enough and not too long or wide, at most
   MAX_PER_ARTIST per artist, download each candidate's "large" picture (about 850 px), record its
   MD5 and save it for review.
2. A person looks at every candidate (the pictures, not just the tags) and writes down the ids to
   drop.
3. ``write``: turn the remaining candidates, in score order, into the list. The first ``--test`` go to
   test and the rest to train. A post that is already in the list keeps its split, so the list can grow
   without changing its test set.

Run it in the detector's Docker image, for example:

    python tools/pick_danbooru.py search /review 110
    python tools/pick_danbooru.py write /review /data/danbooru.json --reject=123,456 --test=20

Every request goes through DanbooruClient, so the User-Agent, the one-request-per-second pace and
the "stop at a bot check" rule apply here too. The number of requests is printed at the end.
"""
import json
import sys
from datetime import date
from pathlib import Path

from blur_detection.config import DANBOORU_RATING, DANBOORU_SITE
from blur_detection.danbooru import DanbooruClient, md5_of, unusable_reason

QUERY = f'rating:{DANBOORU_RATING} order:score'
#: Candidates must be at least this big on their shorter side, and not longer than ASPECT : 1.
MIN_SIDE = 800
ASPECT = 2.0
MAX_PER_ARTIST = 2
CANDIDATES = 'candidates.json'
STATE = 'search-state.json'


def entry_of(post: dict, large_md5: str) -> dict:
    return {
        'id': post['id'],
        'md5': post['md5'],
        'large_md5': large_md5,
        'artist': post['tag_string_artist'],
        'copyright': post.get('tag_string_copyright', ''),
        'source': post.get('source', ''),
        'rating': post['rating'],
        'score': post.get('score'),
        'size': [post.get('image_width'), post.get('image_height')],
    }


def search(review_dir: str, count: int) -> None:
    out = Path(review_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = DanbooruClient()
    # Carry on where an earlier search stopped, when there was one.
    picked = json.loads((out / CANDIDATES).read_text(encoding='utf-8')) if (out / CANDIDATES).is_file() else []
    page = json.loads((out / STATE).read_text())['next_page'] if (out / STATE).is_file() else 1
    seen = {c['id'] for c in picked}
    per_artist = {}
    for c in picked:
        per_artist[c['artist']] = per_artist.get(c['artist'], 0) + 1
    while len(picked) < count:
        posts = client.search(QUERY, page)
        if not posts:
            break
        for post in posts:
            w, h = post.get('image_width') or 0, post.get('image_height') or 0
            artist = post.get('tag_string_artist', '')
            # One named artist, so that every picture can be credited.
            if (post['id'] in seen or not artist or ' ' in artist or unusable_reason(post)
                    or min(w, h) < MIN_SIDE or max(w, h) > ASPECT * min(w, h)
                    or per_artist.get(artist, 0) >= MAX_PER_ARTIST):
                continue
            data = client.get(post['large_file_url'])
            ext = post['large_file_url'].rsplit('.', 1)[-1]
            (out / f'{post["id"]}.{ext}').write_bytes(data)
            picked.append(entry_of(post, md5_of(data)))
            per_artist[artist] = per_artist.get(artist, 0) + 1
            print(f'{len(picked):3} post {post["id"]} by {artist} ({w}x{h}), {client.requests} requests so far')
            if len(picked) >= count:
                break
        else:
            page += 1  # every post of this page looked at
    (out / CANDIDATES).write_text(json.dumps(picked, indent=1), encoding='utf-8')
    (out / STATE).write_text(json.dumps({'next_page': page}))
    print(f'{len(picked)} candidates in {out}; {client.requests} requests to Danbooru')


def write(review_dir: str, list_path: str, reject: str = '', test: int = 20) -> None:
    rejected = {int(i) for i in reject.split(',') if i.strip()}
    candidates = [c for c in json.loads((Path(review_dir) / CANDIDATES).read_text(encoding='utf-8'))
                  if c['id'] not in rejected]
    path = Path(list_path)
    kept = {p['id']: p['split'] for p in json.loads(path.read_text(encoding='utf-8'))['posts']} if path.is_file() else {}
    test_left = test - sum(split == 'test' for split in kept.values())
    posts = []
    for candidate in candidates:
        post = {k: v for k, v in candidate.items() if k not in ('score', 'size')}
        if candidate['id'] in kept:
            post['split'] = kept[candidate['id']]
        elif test_left > 0:
            post['split'], test_left = 'test', test_left - 1
        else:
            post['split'] = 'train'
        posts.append(post)
    listing = {
        'site': DANBOORU_SITE,
        'chosen': str(date.today()),
        'query': QUERY,
        'how': 'tools/pick_danbooru.py search, then every picture checked by eye; see data/README.md',
        'posts': posts,
    }
    path.write_text(json.dumps(listing, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    counts = {split: sum(p['split'] == split for p in posts) for split in ('train', 'test')}
    print(f'wrote {len(posts)} posts to {list_path}: {counts["train"]} train, {counts["test"]} test')


if __name__ == '__main__':
    command, *args = sys.argv[1:]
    if command == 'search':
        search(args[0], int(args[1]))
    elif command == 'write':
        options = dict(arg.lstrip('-').split('=', 1) for arg in args[2:])
        write(args[0], args[1], options.get('reject', ''), int(options.get('test', 20)))
    else:
        raise SystemExit(__doc__)
