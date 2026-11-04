// End to end: the JavaScript pipeline (decode -> prepare -> ONNX Runtime Web) must give the same
// P(blur) as the Python pipeline did when the model was exported, for the reference pictures the model
// card lists (data/demo.png, drawn by code, so it is in the repository).
// Needs public/model/ from `blur-detector export`; set REQUIRE_MODEL=1 to fail instead of skipping.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';

import * as ort from 'onnxruntime-web';
import { PNG } from 'pngjs';

import { MODEL_CARD_URL, MODEL_URL, ORT_VERSION } from '../public/js/config.js';
import { BlurDetector } from '../public/js/detector.js';

const publicDir = new URL('../public/', import.meta.url);
const dataDir = new URL('../../../data/', import.meta.url);
const cardFile = new URL(MODEL_CARD_URL, publicDir);
const haveModel = existsSync(cardFile);
const skip = !haveModel && !process.env.REQUIRE_MODEL && 'no exported model in public/model (run `blur-detector export`)';

/** Largest difference from the Python result we accept (OpenCV's resize can round a pixel differently). */
const TOLERANCE = 0.02;

const readJson = (url) => JSON.parse(readFileSync(url, 'utf8'));

test('the page loads the ONNX Runtime version that the tests use', () => {
  const pkg = readJson(new URL('../package.json', import.meta.url));
  assert.equal(pkg.devDependencies['onnxruntime-web'], ORT_VERSION);
});

test('the browser pipeline gives the same P(blur) as Python for the reference pictures', { skip }, async () => {
  const card = readJson(cardFile);
  const detector = await BlurDetector.load(ort, readFileSync(new URL(MODEL_URL, publicDir)), card);
  const names = Object.keys(card.references);
  assert.ok(names.length > 0, 'the model card lists no reference pictures (is data/demo.png there?)');
  for (const name of names) {
    const png = PNG.sync.read(readFileSync(new URL(name, dataDir)));
    const [p] = await detector.predict([{ width: png.width, height: png.height, data: png.data }]);
    const expected = card.references[name];
    assert.ok(Math.abs(p - expected) <= TOLERANCE, `${name}: JavaScript ${p.toFixed(4)}, Python ${expected.toFixed(4)}`);
  }
});
