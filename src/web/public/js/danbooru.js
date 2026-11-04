// Anime pictures from Danbooru, the page's default source: only the general-audience posts listed in
// samples/danbooru.json (each one looked at by a person before it was listed), each with its credit.
// Nothing here touches the page, so the tests run it with a fake fetch and storage.

/** A post the page must not or cannot show: another rating, a video, a missing file. */
export class UnusablePost extends Error {}

/** The API address of one post. */
export function postUrl({ site, postPath }, id) {
  return new URL(`${postPath}${id}.json`, site).href;
}

const extensionOf = (url) => new URL(url).pathname.split('.').pop().toLowerCase();

function isWebAddress(text) {
  try {
    return ['http:', 'https:'].includes(new URL(text).protocol);
  } catch {
    return false;
  }
}

/**
 * What the page needs from a post: the picture to load (Danbooru's sample, about 850 px wide, or the
 * original when there is no sample) and the credit. Throws UnusablePost unless the post has exactly
 * the wanted rating and a still picture in one of `extensions`.
 */
export function pictureOf(post, { site, rating, extensions }) {
  if (!post || typeof post !== 'object' || !Number.isInteger(post.id)) throw new UnusablePost('not a post');
  if (post.rating !== rating) throw new UnusablePost(`post ${post.id} is rated "${post.rating}", not "${rating}"`);
  const url = [post.large_file_url, post.file_url].find((u) => typeof u === 'string' && u.startsWith('https://'));
  if (!url) throw new UnusablePost(`post ${post.id} has no picture to load`);
  const kinds = [String(post.file_ext).toLowerCase(), extensionOf(url)];
  if (!kinds.every((kind) => extensions.includes(kind))) throw new UnusablePost(`post ${post.id} is a ${kinds[0]}, not a still picture`);
  return {
    id: post.id,
    rating: post.rating,
    url,
    artists: String(post.tag_string_artist ?? '').split(/\s+/).filter(Boolean),
    postUrl: new URL(`/posts/${post.id}`, site).href,
    sourceUrl: isWebAddress(post.source) ? post.source : null,
  };
}

/**
 * Ask Danbooru for the listed posts `ids` in turn until one has a usable picture, at most `attempts`
 * requests in all. A post that is gone (HTTP 404) or no longer usable is skipped; other HTTP and
 * network errors are reported at once, as asking again would not help.
 */
export async function fetchListedPicture(fetchFn, settings, ids, { signal } = {}) {
  const skipped = [];
  for (const id of ids.slice(0, settings.attempts)) {
    const response = await fetchFn(postUrl(settings, id), { signal, credentials: 'omit' });
    if (response.status === 404) {
      skipped.push(`post ${id} is gone from Danbooru`);
      continue;
    }
    if (!response.ok) throw new Error(`Danbooru answered HTTP ${response.status}`);
    try {
      const picture = pictureOf(await response.json(), settings);
      if (picture.id !== id) throw new UnusablePost(`asked for post ${id}, got ${picture.id}`);
      return picture;
    } catch (err) {
      if (!(err instanceof UnusablePost)) throw err;
      skipped.push(err.message);
    }
  }
  // A short reason for the page; why each post was skipped goes along for the console.
  throw new Error(`no usable picture in ${Math.min(ids.length, settings.attempts)} posts`, { cause: skipped });
}

/** The picture kept for this browser session, or null when there is none, it is not one of `ids` or it cannot be used. */
export function readCachedPicture(storage, settings, ids) {
  try {
    const picture = JSON.parse(storage?.getItem(settings.cacheKey) ?? 'null');
    if (!picture || picture.rating !== settings.rating || typeof picture.url !== 'string') return null;
    if (!ids.includes(picture.id)) return null; // e.g. a random post kept by an older version of the page
    return picture.url.startsWith('https://') && settings.extensions.includes(extensionOf(picture.url)) ? picture : null;
  } catch {
    return null; // storage blocked, or a value this version cannot read
  }
}

/** Keep `picture` for this browser session; without storage (private mode, full) there is just no cache. */
export function cachePicture(storage, settings, picture) {
  try {
    storage?.setItem(settings.cacheKey, JSON.stringify(picture));
  } catch {
    // no cache then
  }
}

/** A Danbooru artist tag as people read it: `yukichi_(nyarome-busters)` -> `yukichi (nyarome-busters)`. */
export const artistName = (tag) => tag.replaceAll('_', ' ');

export function artistNames(picture) {
  return picture.artists.map(artistName);
}

/** `ids` starting at `first` and going round, so that a gone post falls back to the next one. */
export function inTurn(ids, first) {
  const start = Math.max(0, ids.indexOf(first));
  return [...ids.slice(start), ...ids.slice(0, start)];
}
