// Random anime pictures from Danbooru, the page's default source: general-audience posts only, each
// with its credit. Nothing here touches the page, so the tests run it with a fake fetch and storage.

/** A post the page must not or cannot show: another rating, a video, a missing file. */
export class UnusablePost extends Error {}

/** The API address that answers with one random post carrying `tags`. */
export function randomPostUrl({ site, randomPath, tags }) {
  const url = new URL(randomPath, site);
  url.searchParams.set('tags', tags);
  return url.href;
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
 * Ask Danbooru for one random post, and again only when that post has no usable picture, at most
 * `attempts` requests in all. HTTP and network errors are not retried: asking again would not help.
 */
export async function fetchRandomPicture(fetchFn, settings, { signal } = {}) {
  const url = randomPostUrl(settings);
  const skipped = [];
  for (let i = 0; i < settings.attempts; i++) {
    const response = await fetchFn(url, { signal, credentials: 'omit' });
    if (!response.ok) throw new Error(`Danbooru answered HTTP ${response.status}`);
    try {
      return pictureOf(await response.json(), settings);
    } catch (err) {
      if (!(err instanceof UnusablePost)) throw err;
      skipped.push(err.message);
    }
  }
  // A short reason for the page; why each post was skipped goes along for the console.
  throw new Error(`no usable picture in ${settings.attempts} posts`, { cause: skipped });
}

/** The picture kept for this browser session, or null when there is none or it cannot be used. */
export function readCachedPicture(storage, settings) {
  try {
    const picture = JSON.parse(storage?.getItem(settings.cacheKey) ?? 'null');
    if (!picture || picture.rating !== settings.rating || typeof picture.url !== 'string') return null;
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

/** Danbooru artist tags as people read them: `yukichi_(nyarome-busters)` -> `yukichi (nyarome-busters)`. */
export function artistNames(picture) {
  return picture.artists.map((tag) => tag.replaceAll('_', ' '));
}
