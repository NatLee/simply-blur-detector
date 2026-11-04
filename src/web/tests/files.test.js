// Every file the page reads must be committed: a file that only exists on a developer's machine works
// locally and is missing on GitHub Pages (that is how samples/danbooru.json once went missing).
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import * as cfg from '../public/js/config.js';

const publicDir = new URL('../public/', import.meta.url);

/** Files `blur-detector export` writes next to the page when the site is built; they are not committed. */
const BUILT = new Set([cfg.MODEL_URL, cfg.MODEL_CARD_URL]);

/** Every file the page fetches from its own site: settings named *_URL or *Url without a host, however deep. */
function pathsIn(settings) {
  return Object.entries(settings).flatMap(([name, value]) => {
    if (value && typeof value === 'object') return pathsIn(value);
    return typeof value === 'string' && /(URL|Url)$/.test(name) && !value.includes('://') ? [value] : [];
  });
}

function isCommitted(path) {
  const file = fileURLToPath(new URL(path, publicDir));
  try {
    execFileSync('git', ['ls-files', '--error-unmatch', file], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

test('every file named in the settings is committed, except what the build makes', () => {
  const paths = pathsIn({ ...cfg });
  assert.ok(paths.includes(cfg.SAMPLES_URL) && paths.includes(cfg.DANBOORU.picksUrl));
  for (const path of paths.filter((p) => !BUILT.has(p))) assert.ok(isCommitted(path), `${path} is not in git`);
});

test('every built-in sample picture is committed', () => {
  const { samples } = JSON.parse(readFileSync(new URL(cfg.SAMPLES_URL, publicDir)));
  for (const s of samples) assert.ok(isCommitted(cfg.SAMPLES_DIR + s.file), `${s.file} is not in git`);
});
