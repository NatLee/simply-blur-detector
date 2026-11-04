import assert from 'node:assert/strict';
import { test } from 'node:test';

import { heatField, laplacianScore, scoreColor, windowGrid } from '../public/js/heatmap.js';

test('windows cover the image edge to edge, half overlapping', () => {
  const grid = windowGrid(640, 480, { windowFraction: 1 / 2, overlap: 0.5 });
  assert.equal(grid.size, 240);
  assert.deepEqual([grid.cols, grid.rows], [5, 3]); // x: 0 120 240 360 400(edge); y: 0 120 240
  const xs = [...new Set(grid.rects.map((r) => r.x))];
  assert.deepEqual(xs, [0, 120, 240, 360, 400]);
  for (const r of grid.rects) {
    assert.ok(r.x + r.w <= 640 && r.y + r.h <= 480);
  }
});

test('a tiny image still gets one window', () => {
  const grid = windowGrid(3, 2, { windowFraction: 1 / 3, overlap: 0.5 });
  assert.equal(grid.rects.length >= 1, true);
  for (const r of grid.rects) assert.ok(r.w >= 1 && r.x + r.w <= 3 && r.y + r.h <= 2);
});

test('the heat field blends the windows and leaves unscored cells empty', () => {
  const rects = [{ x: 0, y: 0, w: 10, h: 10 }, { x: 10, y: 0, w: 10, h: 10 }];
  const field = heatField(rects, [0, 1], 20, 10, 4, 2);
  assert.ok(field[0] < 0.01 && field[3] > 0.99); // left cells follow the left window, right the right one
  const half = heatField(rects, [0.2], 20, 10, 4, 2); // the second window has no score yet
  assert.ok(Math.abs(half[0] - 0.2) < 1e-6);
  assert.ok(Number.isNaN(half[3]));
});

test('scores near 0.5 fade out, confident scores take their colour', () => {
  const sharp = [0, 0, 255];
  const blurry = [255, 0, 0];
  assert.deepEqual(scoreColor(0, sharp, blurry, 1), [0, 0, 255, 255]);
  assert.deepEqual(scoreColor(1, sharp, blurry, 1).slice(0, 3), blurry);
  assert.ok(scoreColor(0.55, sharp, blurry, 1)[3] < scoreColor(0.95, sharp, blurry, 1)[3]);
  assert.equal(scoreColor(Number.NaN, sharp, blurry, 1)[3], 0);
});

test('the Laplacian score is 0.5 at the threshold and falls as the variance rises', () => {
  assert.equal(laplacianScore(100, 100), 0.5);
  assert.ok(Math.abs(laplacianScore(1000, 100) - 1 / 11) < 1e-12);
  assert.ok(laplacianScore(10, 100) > 0.9);
});
