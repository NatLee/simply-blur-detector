// The Danbooru source, with a fake fetch and a fake storage: no request ever leaves the machine.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { DANBOORU } from '../public/js/config.js';
import {
  UnusablePost,
  artistNames,
  cachePicture,
  fetchListedPicture,
  inTurn,
  pictureOf,
  postUrl,
  readCachedPicture,
} from '../public/js/danbooru.js';

const LISTED = [11404274, 7, 8, 9];

/** A post shaped like Danbooru's /posts/<id>.json answer (only the fields the page reads). */
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

test('the page only ever asks for listed general-audience posts', () => {
  assert.equal(DANBOORU.rating, 'g');
  assert.equal(postUrl(DANBOORU, 4000000), 'https://danbooru.donmai.us/posts/4000000.json');
  const { picks } = JSON.parse(readFileSync(new URL('../public/' + DANBOORU.picksUrl, import.meta.url)));
  assert.ok(picks.length >= 3);
  for (const pick of picks) {
    assert.ok(Number.isInteger(pick.id) && pick.title && pick.artist, JSON.stringify(pick)); // credited on the card
  }
});

test('posts are tried from the chosen one, then round the list', () => {
  assert.deepEqual(inTurn(LISTED, 8), [8, 9, 11404274, 7]);
  assert.deepEqual(inTurn(LISTED, 404), LISTED);
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
  const picture = await fetchListedPicture(fetchFn, DANBOORU, LISTED);
  assert.equal(picture.id, 11404274);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, postUrl(DANBOORU, 11404274));
  assert.equal(calls[0].options.credentials, 'omit');
});

test('a gone or re-rated post is skipped for the next listed one, never shown', async () => {
  const { fetchFn, calls } = fakeFetch([404, post({ id: 7, rating: 'e' }), post({ id: 8 })]);
  const picture = await fetchListedPicture(fetchFn, DANBOORU, LISTED);
  assert.equal(picture.id, 8);
  assert.equal(picture.rating, 'g');
  assert.deepEqual(calls.map((c) => c.url), [11404274, 7, 8].map((id) => postUrl(DANBOORU, id)));
});

test('an answer for another post than the one asked for is not used', async () => {
  const { fetchFn } = fakeFetch([post({ id: 999 })]);
  await assert.rejects(fetchListedPicture(fetchFn, DANBOORU, [11404274]), /no usable picture/);
});

test('it gives up after the configured number of posts', async () => {
  const { fetchFn, calls } = fakeFetch([post({ rating: 's' })]);
  await assert.rejects(fetchListedPicture(fetchFn, DANBOORU, Array(10).fill(11404274)), (err) => {
    assert.equal(err.message, `no usable picture in ${DANBOORU.attempts} posts`);
    assert.equal(err.cause.length, DANBOORU.attempts); // why each post was skipped, for the console
    return true;
  });
  assert.equal(calls.length, DANBOORU.attempts);
});

test('HTTP and network errors are reported at once, without asking again', async () => {
  const { fetchFn, calls } = fakeFetch([503]);
  await assert.rejects(fetchListedPicture(fetchFn, DANBOORU, LISTED), /HTTP 503/);
  assert.equal(calls.length, 1);
  const offline = async () => { throw new TypeError('Failed to fetch'); };
  await assert.rejects(fetchListedPicture(offline, DANBOORU, LISTED), /Failed to fetch/);
});

test('the session cache gives back the same picture, and nothing it should not', () => {
  const storage = fakeStorage();
  assert.equal(readCachedPicture(storage, DANBOORU, LISTED), null);
  const picture = pictureOf(post(), DANBOORU);
  cachePicture(storage, DANBOORU, picture);
  assert.deepEqual(readCachedPicture(storage, DANBOORU, LISTED), picture);
  storage.setItem(DANBOORU.cacheKey, JSON.stringify({ ...picture, rating: 'q' }));
  assert.equal(readCachedPicture(storage, DANBOORU, LISTED), null);
  storage.setItem(DANBOORU.cacheKey, JSON.stringify({ ...picture, url: 'https://cdn.donmai.us/x.mp4' }));
  assert.equal(readCachedPicture(storage, DANBOORU, LISTED), null);
  storage.setItem(DANBOORU.cacheKey, JSON.stringify({ ...picture, id: 12345 })); // not a listed post
  assert.equal(readCachedPicture(storage, DANBOORU, LISTED), null);
  storage.setItem(DANBOORU.cacheKey, '{not json');
  assert.equal(readCachedPicture(storage, DANBOORU, LISTED), null);
});

test('blocked or missing storage means no cache, not an error', () => {
  const blocked = { getItem() { throw new Error('SecurityError'); }, setItem() { throw new Error('QuotaExceededError'); } };
  assert.equal(readCachedPicture(blocked, DANBOORU, LISTED), null);
  assert.doesNotThrow(() => cachePicture(blocked, DANBOORU, pictureOf(post(), DANBOORU)));
  assert.equal(readCachedPicture(null, DANBOORU, LISTED), null);
  assert.doesNotThrow(() => cachePicture(null, DANBOORU, pictureOf(post(), DANBOORU)));
});
