// The page in a real browser, with Danbooru answered by the test (no request leaves the machine).
// Runs when BROWSER_PATH points at Chrome, Chromium or Edge; CI sets it, elsewhere it is skipped.
// Needs public/model/ from `blur-detector export`, like the pipeline test.
import assert from 'node:assert/strict';
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { after, before, test } from 'node:test';

import { PNG } from 'pngjs';

import { DANBOORU, MODEL_CARD_URL } from '../public/js/config.js';

const publicDir = fileURLToPath(new URL('../public/', import.meta.url));
const browserPath = process.env.BROWSER_PATH;
const skip = (!browserPath && 'set BROWSER_PATH to run the page in a browser')
  || (!existsSync(join(publicDir, MODEL_CARD_URL)) && 'no exported model in public/model');

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.onnx': 'application/octet-stream' };

/** A picture for the fake Danbooru: stripes, so it is sharp. */
function stripes(width, height) {
  const png = new PNG({ width, height });
  for (let i = 0; i < width * height; i++) {
    const v = Math.floor(i % width / 6) % 2 ? 230 : 30;
    png.data.set([v, 255 - v, 120, 255], i * 4);
  }
  return PNG.sync.write(png);
}

let server;
let base;
let puppeteer;

before(async () => {
  if (skip) return;
  puppeteer = (await import('puppeteer-core')).default;
  server = createServer((req, res) => {
    const path = normalize(join(publicDir, decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/\/$/, '/index.html')));
    if (!path.startsWith(publicDir) || !existsSync(path)) return res.writeHead(404).end();
    res.writeHead(200, { 'content-type': TYPES[extname(path)] ?? 'application/octet-stream' });
    createReadStream(path).pipe(res);
  });
  await new Promise((resolve) => server.listen(Number(process.env.E2E_PORT ?? 0), '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}/`;
});

after(() => server?.close());

/**
 * Open the page with `danbooru(url)` deciding the answer to every request to danbooru.donmai.us or its
 * image server ({status, body, type}, or null to fail the request). Resolves when the page has settled.
 */
async function openPage({ danbooru, picks }) {
  const browser = await puppeteer.launch({ executablePath: browserPath, headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage();
  const asked = [];
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const url = req.url();
    if (picks && url.endsWith(DANBOORU.picksUrl)) return req.respond(picks);
    if (!/donmai\.us/.test(url)) return req.continue();
    asked.push(url);
    const answer = danbooru(url);
    if (!answer) return req.abort();
    return req.respond({ status: answer.status ?? 200, contentType: answer.type, body: answer.body,
      headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto(base, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => /windows of|drop one of your own|Laplacian only/.test(document.getElementById('hud-status').textContent),
    { timeout: 120000 });
  const view = await page.evaluate(() => ({
    hud: document.getElementById('hud-status').textContent,
    file: document.getElementById('hud-file').textContent,
    credit: document.getElementById('credit').innerText,
    links: [...document.querySelectorAll('#credit a')].map((a) => [a.href, a.rel]),
    cards: document.querySelectorAll('.sample.anime').length,
    dropShown: !document.getElementById('drop').hidden,
    modelCard: document.getElementById('model-card-list').innerText,
  }));
  await browser.close();
  return { view, asked };
}

const post = (id) => ({ id, rating: 'g', file_ext: 'png', large_file_url: `https://cdn.donmai.us/sample/test/${id}.png`,
  tag_string_artist: 'some_artist', source: 'https://example.com/art' });

test('a listed post opens with its credit, one API request and one picture', { skip }, async () => {
  const { view, asked } = await openPage({
    danbooru: (url) => (url.includes('/posts/')
      ? { type: 'application/json', body: JSON.stringify(post(Number(url.match(/posts\/(\d+)/)[1]))) }
      : { type: 'image/png', body: stripes(640, 640) }),
  });
  assert.match(view.hud, /windows of/);
  assert.match(view.file, /^danbooru #\d+ · 640 × 640$/);
  assert.match(view.credit, /^Art: some artist · Danbooru #\d+ · original source$/);
  assert.ok(view.links.every(([, rel]) => rel === 'noopener noreferrer'));
  assert.equal(asked.length, 2);
  assert.ok(view.cards >= 3);
});

test('the model card says when a model is published and why', { skip }, async () => {
  const { publish } = JSON.parse(readFileSync(join(publicDir, MODEL_CARD_URL), 'utf8'));
  const { view } = await openPage({ danbooru: () => null });
  const expected = `at least ${Math.round(publish.min_test_accuracy * 100)}% of the test pictures are right`
    + ` (${publish.needed} of `;
  assert.ok(view.modelCard.includes(expected), view.modelCard);
  assert.ok(view.modelCard.includes(`this model: ${publish.correct}). ${publish.note}`), view.modelCard);
});

test('when Danbooru cannot be reached, the page asks for your own picture', { skip }, async () => {
  const { view } = await openPage({ danbooru: () => null });
  assert.ok(view.dropShown, 'the drop zone is shown');
  assert.match(view.hud, /drop one of your own/);
  assert.match(view.credit, /Could not get a picture from Danbooru \(Failed to fetch\)\. Drop, paste or choose a picture of your own/);
});

test('when the list of posts cannot be read, the page says so', { skip }, async () => {
  const { view, asked } = await openPage({ danbooru: () => null, picks: { status: 404, body: 'not found' } });
  assert.ok(view.dropShown);
  assert.equal(view.cards, 0);
  assert.match(view.credit, /the list of anime pictures could not be read \(samples\/danbooru\.json: HTTP 404\)/);
  assert.match(view.credit, /reload the page/);
  assert.deepEqual(asked, []); // nothing is asked of Danbooru without the list
});
