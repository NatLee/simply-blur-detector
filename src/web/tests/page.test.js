// The page in a real browser, served from public/ by the test.
// Runs when BROWSER_PATH points at Chrome, Chromium or Edge; CI sets it, elsewhere it is skipped.
// Needs public/model/ from `blur-detector export`, like the pipeline test.
import assert from 'node:assert/strict';
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { after, before, test } from 'node:test';

import { MODEL_CARD_URL, ORT_URL, SAMPLES_URL } from '../public/js/config.js';

const publicDir = fileURLToPath(new URL('../public/', import.meta.url));
const browserPath = process.env.BROWSER_PATH;
const skip = (!browserPath && 'set BROWSER_PATH to run the page in a browser')
  || (!existsSync(join(publicDir, MODEL_CARD_URL)) && 'no exported model in public/model');
const { samples } = JSON.parse(readFileSync(join(publicDir, SAMPLES_URL), 'utf8'));

/** The only other sites the page may ask anything: ONNX Runtime Web and the fonts. */
const OTHER_HOSTS = new Set([new URL(ORT_URL).host, 'fonts.googleapis.com', 'fonts.gstatic.com']);

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.onnx': 'application/octet-stream', '.jpg': 'image/jpeg', '.png': 'image/png' };

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
 * Open the page at `hash`, with `list` (if given) answering the request for the list of samples.
 * Resolves when the page has settled, with what it shows and the hosts other than this server it asked.
 */
async function openPage({ hash = '', list } = {}) {
  const browser = await puppeteer.launch({ executablePath: browserPath, headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage();
  const hosts = new Set();
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const url = new URL(req.url());
    if (url.origin !== new URL(base).origin && url.protocol.startsWith('http')) hosts.add(url.host);
    if (list && url.pathname.endsWith(SAMPLES_URL)) return req.respond(list);
    return req.continue();
  });
  await page.goto(base + hash, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => /windows of|drop one of your own|Laplacian only/.test(document.getElementById('hud-status').textContent),
    { timeout: 120000 });
  const view = await page.evaluate(() => ({
    hud: document.getElementById('hud-status').textContent,
    file: document.getElementById('hud-file').textContent,
    caption: document.getElementById('caption').hidden ? null : document.getElementById('caption').innerText,
    cards: [...document.querySelectorAll('.sample:not(.own)')].map((b) => ({
      file: b.dataset.file, pressed: b.getAttribute('aria-pressed') === 'true',
      thumbnail: b.querySelector('img').complete && b.querySelector('img').naturalWidth > 0,
    })),
    dropShown: !document.getElementById('drop').hidden,
    modelCard: document.getElementById('model-card-list').innerText,
    hash: location.hash,
  }));
  await browser.close();
  return { view, hosts: [...hosts] };
}

test('the page opens the first sample and shows every sample as a card', { skip }, async () => {
  const { view, hosts } = await openPage();
  assert.match(view.hud, /windows of/);
  assert.match(view.file, new RegExp(`^${samples[0].file.replace('.', '\\.')} · \\d+ × \\d+$`));
  assert.deepEqual(view.cards.map((c) => c.file), samples.map((s) => s.file));
  assert.ok(view.cards.every((c) => c.thumbnail), 'every card shows its picture');
  assert.deepEqual(view.cards.map((c) => c.pressed), samples.map((_, i) => i === 0));
  assert.equal(view.caption, null);
  assert.deepEqual(hosts.filter((h) => !OTHER_HOSTS.has(h)), []); // the pictures come from this site
});

test('a #sample= link opens that sample', { skip }, async () => {
  const wanted = samples[samples.length - 1].file;
  const { view } = await openPage({ hash: `#sample=${encodeURIComponent(wanted)}` });
  assert.ok(view.file.startsWith(`${wanted} · `), view.file);
  assert.deepEqual(view.cards.filter((c) => c.pressed).map((c) => c.file), [wanted]);
  assert.equal(view.hash, `#sample=${encodeURIComponent(wanted)}`);
});

test('when the list of samples cannot be read, the page says so and asks for your own picture', { skip }, async () => {
  const { view } = await openPage({ list: { status: 404, body: 'not found' } });
  assert.ok(view.dropShown);
  assert.deepEqual(view.cards, []);
  assert.match(view.caption, /the list of sample pictures could not be read \(samples\/samples\.json: HTTP 404\)/);
  assert.match(view.caption, /reload the page/);
});

test('the model card says when a model is published and why', { skip }, async () => {
  const { publish } = JSON.parse(readFileSync(join(publicDir, MODEL_CARD_URL), 'utf8'));
  const { view } = await openPage();
  const expected = `at least ${Math.round(publish.min_test_accuracy * 100)}% of the test pictures are right`
    + ` (${publish.needed} of `;
  assert.ok(view.modelCard.includes(expected), view.modelCard);
  assert.ok(view.modelCard.includes(`this model: ${publish.correct}). ${publish.note}`), view.modelCard);
});
