// The Danbooru source, with a fake fetch and a fake storage: no request ever leaves the machine.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { DANBOORU } from '../public/js/config.js';
import {
  UnusablePost,
  artistNames,
  cachePicture,
  fetchRandomPicture,
  pictureOf,
  randomPostUrl,
  readCachedPicture,
} from '../public/js/danbooru.js';

/** A post shaped like Danbooru's /posts/random.json answer (only the fields the page reads). */
function post(overrides = {}) {
  return {
    id: 11404274,
    rating: 'g',
    file_ext: 'jpg',
    file_url: 'https://cdn.donmai.us/original/ab/cd/abcd.jpg',
    large_file_url: 'https://cdn.donmai.us/sample/ab/cd/sample-abcd.jpg',
    tag_string_artist: 'yukichi_(nyarome-busters)',
    source: 'https://x.com/someone/status/1',
    ...overrides,
  };
}

/** A fake fetch that answers with `answers` in turn (a post object, or a number for an HTTP error). */
function fakeFetch(answers) {
  const calls = [];
  const fetchFn = async (url, options) => {
    calls.push({ url, options });
    const answer = answers[Math.min(calls.length - 1, answers.length - 1)];
    if (typeof answer === 'number') return { ok: false, status: answer, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => structuredClone(answer) };
  };
  return { fetchFn, calls };
}

function fakeStorage() {
  const data = new Map();
  return { getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => data.set(k, String(v)), data };
}

test('the page only ever asks for general-audience posts', () => {
  assert.equal(DANBOORU.tags, 'rating:g');
  assert.equal(DANBOORU.rating, 'g');
  const url = new URL(randomPostUrl(DANBOORU));
  assert.equal(url.origin + url.pathname, 'https://danbooru.donmai.us/posts/random.json');
  assert.equal(url.searchParams.get('tags'), 'rating:g');
});

test('a general-audience post gives the sample picture and its credit', () => {
  const picture = pictureOf(post(), DANBOORU);
  assert.deepEqual(picture, {
    id: 11404274,
    rating: 'g',
    url: 'https://cdn.donmai.us/sample/ab/cd/sample-abcd.jpg',
    artists: ['yukichi_(nyarome-busters)'],
    postUrl: 'https://danbooru.donmai.us/posts/11404274',
    sourceUrl: 'https://x.com/someone/status/1',
  });
  assert.deepEqual(artistNames(picture), ['yukichi (nyarome-busters)']);
});

test('the original is used when there is no sample, and several artists are kept apart', () => {
  const picture = pictureOf(post({ large_file_url: null, tag_string_artist: 'a_b  c' }), DANBOORU);
  assert.equal(picture.url, 'https://cdn.donmai.us/original/ab/cd/abcd.jpg');
  assert.deepEqual(picture.artists, ['a_b', 'c']);
  assert.deepEqual(pictureOf(post({ tag_string_artist: '' }), DANBOORU).artists, []);
});

test('a source that is not a web address is not turned into a link', () => {
  assert.equal(pictureOf(post({ source: 'Scanned from a magazine' }), DANBOORU).sourceUrl, null);
  assert.equal(pictureOf(post({ source: 'javascript:alert(1)' }), DANBOORU).sourceUrl, null);
  assert.equal(pictureOf(post({ source: '' }), DANBOORU).sourceUrl, null);
});

test('posts with any other rating are refused', () => {
  for (const rating of ['s', 'q', 'e', undefined, 'G']) {
    assert.throws(() => pictureOf(post({ rating }), DANBOORU), UnusablePost, String(rating));
  }
});

test('videos, animations and posts without a file are refused', () => {
  const video = { file_ext: 'mp4', large_file_url: 'https://cdn.donmai.us/original/ab/cd/abcd.mp4' };
  const ugoira = { file_ext: 'zip', large_file_url: 'https://cdn.donmai.us/sample/ab/cd/sample-abcd.webm' };
  for (const bad of [video, ugoira, { file_ext: 'gif' }, { large_file_url: null, file_url: null },
    { large_file_url: 'http://cdn.donmai.us/x.jpg', file_url: undefined }, { id: 'x' }]) {
    assert.throws(() => pictureOf(post(bad), DANBOORU), UnusablePost, JSON.stringify(bad));
  }
});

test('one request gives one picture, without cookies', async () => {
  const { fetchFn, calls } = fakeFetch([post()]);
  const picture = await fetchRandomPicture(fetchFn, DANBOORU);
  assert.equal(picture.id, 11404274);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, randomPostUrl(DANBOORU));
  assert.equal(calls[0].options.credentials, 'omit');
});

test('an unusable post is skipped, and a post with another rating is never returned', async () => {
  const { fetchFn, calls } = fakeFetch([post({ rating: 'e' }), post({ file_ext: 'mp4' }), post({ id: 7 })]);
  const picture = await fetchRandomPicture(fetchFn, DANBOORU);
  assert.equal(picture.id, 7);
  assert.equal(picture.rating, 'g');
  assert.equal(calls.length, 3);
});

test('it gives up after the configured number of posts', async () => {
  const { fetchFn, calls } = fakeFetch([post({ rating: 's' })]);
  await assert.rejects(fetchRandomPicture(fetchFn, DANBOORU), (err) => {
    assert.equal(err.message, `no usable picture in ${DANBOORU.attempts} posts`);
    assert.equal(err.cause.length, DANBOORU.attempts); // why each post was skipped, for the console
    return true;
  });
  assert.equal(calls.length, DANBOORU.attempts);
});

test('HTTP and network errors are reported at once, without asking again', async () => {
  const { fetchFn, calls } = fakeFetch([503]);
  await assert.rejects(fetchRandomPicture(fetchFn, DANBOORU), /HTTP 503/);
  assert.equal(calls.length, 1);
  const offline = async () => { throw new TypeError('Failed to fetch'); };
  await assert.rejects(fetchRandomPicture(offline, DANBOORU), /Failed to fetch/);
});

test('the session cache gives back the same picture, and nothing it should not', () => {
  const storage = fakeStorage();
  assert.equal(readCachedPicture(storage, DANBOORU), null);
  const picture = pictureOf(post(), DANBOORU);
  cachePicture(storage, DANBOORU, picture);
  assert.deepEqual(readCachedPicture(storage, DANBOORU), picture);
  storage.setItem(DANBOORU.cacheKey, JSON.stringify({ ...picture, rating: 'q' }));
  assert.equal(readCachedPicture(storage, DANBOORU), null);
  storage.setItem(DANBOORU.cacheKey, JSON.stringify({ ...picture, url: 'https://cdn.donmai.us/x.mp4' }));
  assert.equal(readCachedPicture(storage, DANBOORU), null);
  storage.setItem(DANBOORU.cacheKey, '{not json');
  assert.equal(readCachedPicture(storage, DANBOORU), null);
});

test('blocked or missing storage means no cache, not an error', () => {
  const blocked = { getItem() { throw new Error('SecurityError'); }, setItem() { throw new Error('QuotaExceededError'); } };
  assert.equal(readCachedPicture(blocked, DANBOORU), null);
  assert.doesNotThrow(() => cachePicture(blocked, DANBOORU, pictureOf(post(), DANBOORU)));
  assert.equal(readCachedPicture(null, DANBOORU), null);
  assert.doesNotThrow(() => cachePicture(null, DANBOORU, pictureOf(post(), DANBOORU)));
});
