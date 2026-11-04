// The JavaScript image operations must give what OpenCV gives (fixture made by
// tools/make_opencv_fixture.py, which the detector's tests keep in sync with OpenCV).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  crop, equalizeHist, grayscale, laplacian, laplacianVariance, laplacianVariances, resizeBilinear, toModelInput, variance,
} from '../public/js/imageops.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/opencv-parity.json', import.meta.url)));

/** RGB bytes from the fixture -> RGBA image (alpha 255). */
function rgba(width, height, rgb) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data.set(rgb.slice(i * 3, i * 3 + 3), i * 4);
    data[i * 4 + 3] = 255;
  }
  return { width, height, data };
}

function rgbOf(img) {
  const out = [];
  for (let i = 0; i < img.width * img.height; i++) out.push(...img.data.slice(i * 4, i * 4 + 3));
  return out;
}

const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

for (const c of fixture.cases) {
  const img = rgba(c.width, c.height, c.rgb);

  test(`${c.name}: equalizeHist matches cv2.equalizeHist on every channel`, () => {
    assert.deepEqual(rgbOf(equalizeHist(img)), c.equalized);
  });

  test(`${c.name}: resizeBilinear matches cv2.resize (INTER_LINEAR) within one level`, () => {
    for (const r of c.resized) {
      const out = rgbOf(resizeBilinear(img, r.width, r.height));
      assert.equal(out.length, r.rgb.length);
      assert.ok(maxDiff(out, r.rgb) <= 1, `${r.width}x${r.height}: off by ${maxDiff(out, r.rgb)}`);
    }
  });

  test(`${c.name}: grayscale matches cv2.cvtColor(RGB2GRAY)`, () => {
    assert.deepEqual([...grayscale(img)], c.gray);
  });

  test(`${c.name}: laplacianVariance matches cv2.Laplacian(gray, CV_64F).var()`, () => {
    const v = laplacianVariance(img);
    assert.ok(Math.abs(v - c.laplacian_variance) <= 1e-9 * c.laplacian_variance, `${v} vs ${c.laplacian_variance}`);
  });

  test(`${c.name}: toModelInput matches prepare_image`, () => {
    const out = toModelInput(img, c.model_input.size);
    assert.equal(out.length, c.model_input.values.length);
    assert.ok(maxDiff(out, c.model_input.values) <= 1.01 / 255);
  });
}

test('crop copies a rectangle', () => {
  const c = fixture.cases[0];
  const img = rgba(c.width, c.height, c.rgb);
  const part = crop(img, 2, 1, 3, 2);
  assert.equal(part.width, 3);
  assert.equal(part.height, 2);
  assert.deepEqual([...part.data.slice(0, 4)], [...img.data.slice((1 * c.width + 2) * 4, (1 * c.width + 2) * 4 + 4)]);
});

test('laplacianVariances of the whole image equals laplacianVariance', () => {
  const c = fixture.cases[2];
  const img = rgba(c.width, c.height, c.rgb);
  const [whole] = laplacianVariances(img, [{ x: 0, y: 0, w: c.width, h: c.height }]);
  assert.ok(Math.abs(whole - c.laplacian_variance) < 1e-6 * c.laplacian_variance);
});

test('laplacianVariances of a window uses the real neighbours of the window', () => {
  const c = fixture.cases[2];
  const img = rgba(c.width, c.height, c.rgb);
  const lap = laplacian(grayscale(img), c.width, c.height);
  const r = { x: 5, y: 4, w: 10, h: 9 };
  const values = [];
  for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) values.push(lap[y * c.width + x]);
  const [v] = laplacianVariances(img, [r]);
  assert.ok(Math.abs(v - variance(values)) < 1e-6 * v);
});

test('a flat image has no Laplacian variance and equalizes to itself', () => {
  const img = { width: 4, height: 3, data: new Uint8ClampedArray(48).fill(77) };
  assert.equal(laplacianVariance(img), 0);
  assert.deepEqual([...equalizeHist(img).data], [...img.data]);
});
