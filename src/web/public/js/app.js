// The page: load a picture, run both detectors over it, draw the meters, heat map and stress test.
import * as cfg from './config.js';
import { BLURS } from './blur.js';
import {
  artistName, artistNames, cachePicture, fetchListedPicture, inTurn, readCachedPicture, unavailableMessage,
} from './danbooru.js';
import { BlurDetector } from './detector.js';
import { heatField, laplacianScore, scoreColor, windowGrid } from './heatmap.js';
import { crop, laplacianVariance, laplacianVariances } from './imageops.js';

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
/** The sample card of a listed Danbooru post. */
const danbooruCard = (id) => `danbooru-${id}`;

const state = {
  detector: null,
  card: null,
  image: null, // full-size RGBA: what the CNN sees
  work: null, // copy with the longer side <= LAPLACIAN_WORK_SIZE: what the Laplacian sees
  bitmap: null,
  name: '',
  grid: null,
  cnnScores: [],
  lapVariances: [],
  whole: { cnn: null, variance: null },
  token: 0,
  threshold: cfg.LAPLACIAN_THRESHOLD.initial,
  stress: null,
  picks: [], // the listed Danbooru posts (samples/danbooru.json)
  picksError: null, // why that list could not be read, shown instead of a silent fallback
  choice: 0, // counts picture choices, so a slow download never replaces a newer choice
};

// ---------------------------------------------------------------- loading

async function loadModel() {
  const response = await fetch(cfg.MODEL_CARD_URL);
  if (!response.ok) throw new Error('no model card');
  const card = await response.json();
  const ort = await import(cfg.ORT_URL);
  state.detector = await BlurDetector.load(ort, cfg.MODEL_URL, card);
  state.card = card;
  renderModelCard(card);
}

/** A JSON file next to the page; the error names the file, so a missing one is easy to spot. */
async function readJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

/** The listed Danbooru posts, or [] with `state.picksError` set when the list cannot be read. */
async function loadPicks() {
  try {
    const { picks } = await readJson(cfg.DANBOORU.picksUrl);
    state.picks = picks.map((pick) => pick.id);
    return picks;
  } catch (err) {
    state.picksError = err.message;
    console.error('The list of Danbooru pictures could not be read:', err);
    return [];
  }
}

/** Builds the picture cards: one per listed Danbooru post, and one for your own picture. */
async function loadCards() {
  const picks = await loadPicks();
  const box = $('samples');
  for (const pick of picks) {
    const anime = document.createElement('button');
    anime.className = 'sample anime';
    anime.type = 'button';
    anime.dataset.file = danbooruCard(pick.id);
    anime.setAttribute('aria-pressed', 'false');
    anime.innerHTML = '<span class="plus" aria-hidden="true">&#9734;</span><span><b></b><small></small></span>';
    anime.querySelector('b').textContent = pick.title;
    anime.querySelector('small').textContent = `by ${artistName(pick.artist)} · Danbooru #${pick.id}`;
    anime.addEventListener('click', () => openDanbooru({ id: pick.id }));
    box.append(anime);
  }
  const own = document.createElement('label');
  own.className = 'sample own';
  own.innerHTML = '<span class="plus" aria-hidden="true">+</span><span><b>Your picture</b><small>drop, paste or pick a file</small></span><input type="file" accept="image/*" hidden>';
  own.querySelector('input').addEventListener('change', (e) => takeFile(e.target.files[0]));
  box.append(own);
}

function markCard(file) {
  document.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.file === file)));
}

/** Load a picture from another site so that its pixels stay readable (the site must allow CORS). */
function loadCorsImage(url, timeoutMs) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = url;
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('the picture took too long')), timeoutMs));
  return Promise.race([img.decode().then(() => img), timeout]);
}

/**
 * Show one of the listed Danbooru posts with its credit: `id`, or on page load the session's last
 * one (or a random one). When Danbooru cannot be reached, say so and ask for the visitor's own picture.
 */
async function openDanbooru({ id = null } = {}) {
  const settings = cfg.DANBOORU;
  const choice = ++state.choice;
  history.replaceState(null, '', location.pathname);
  caption(null);
  status('<span class="dot"></span>asking Danbooru for a picture…', true);
  let storage = null;
  try {
    storage = window.sessionStorage;
  } catch {
    // storage blocked: no session cache
  }
  try {
    if (state.picksError) throw new Error(`the list of anime pictures could not be read (${state.picksError})`);
    let picture = id === null ? readCachedPicture(storage, settings, state.picks) : null;
    if (!picture) {
      const first = id ?? state.picks[Math.floor(Math.random() * state.picks.length)];
      markCard(danbooruCard(first));
      picture = await fetchListedPicture(fetch, settings, inTurn(state.picks, first), { signal: AbortSignal.timeout(settings.timeoutMs) });
      cachePicture(storage, settings, picture);
    }
    markCard(danbooruCard(picture.id));
    const img = await loadCorsImage(picture.url, settings.timeoutMs);
    if (choice !== state.choice) return;
    caption(creditOf(picture));
    await openImage(img, `danbooru #${picture.id}`);
  } catch (err) {
    if (choice !== state.choice) return;
    console.warn('No Danbooru picture:', err, err?.cause ?? '');
    const reason = err?.name === 'TimeoutError' ? 'it took too long' : err?.message || 'unknown error';
    markCard(null);
    caption(notice(unavailableMessage(reason, { hasPicks: state.picks.length > 0 })));
    if (!state.image) {
      $('drop').hidden = false;
      status('no picture yet · drop one of your own');
    }
  }
}

function link(text, href) {
  const a = document.createElement('a');
  a.textContent = text;
  a.href = href;
  a.target = '_blank';
  a.rel = 'noopener noreferrer';
  return a;
}

/** "Art: artist · Danbooru #123 · original source", with links to the post and the source. */
function creditOf(picture) {
  const names = artistNames(picture);
  const parts = [`Art: ${names.length ? names.join(', ') : 'artist unknown'}`, link(`Danbooru #${picture.id}`, picture.postUrl)];
  if (picture.sourceUrl) parts.push(link('original source', picture.sourceUrl));
  return parts.flatMap((part, i) => (i ? [' · ', part] : [part]));
}

function notice(text) {
  const span = document.createElement('span');
  span.className = 'notice';
  span.textContent = text;
  return [span];
}

/** The line under the picture: its credit, a notice, or nothing (`parts` null). */
function caption(parts) {
  const el = $('credit');
  el.replaceChildren(...(parts ?? []));
  el.hidden = !parts;
}

function pixelsOf(bitmap, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

/** `source` is anything createImageBitmap takes: a Blob, or an image whose pixels may be read. */
async function openImage(source, name) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(source);
  } catch {
    status(`could not read ${name} as an image`);
    return;
  }
  state.bitmap = bitmap;
  state.name = name;
  state.image = pixelsOf(bitmap, bitmap.width, bitmap.height);
  const scale = Math.min(1, cfg.LAPLACIAN_WORK_SIZE / Math.max(bitmap.width, bitmap.height));
  state.work = pixelsOf(bitmap, Math.max(1, Math.round(bitmap.width * scale)), Math.max(1, Math.round(bitmap.height * scale)));
  $('drop').hidden = true;
  $('hud-file').textContent = `${name} · ${bitmap.width} × ${bitmap.height}`;
  layoutStage();
  $('stress-run').disabled = !state.detector;
  await analyze();
}

// ---------------------------------------------------------------- analysis

function windowLayout() {
  const detail = document.querySelector('input[name=detail]:checked').value;
  const fraction = state.card?.windows?.fraction ?? 0.5;
  return windowGrid(state.image.width, state.image.height, { windowFraction: fraction, ...cfg.HEATMAP_DETAIL[detail] });
}

/** Serpentine order: the reticle sweeps like a scanner instead of jumping back every row. */
function scanOrder(grid) {
  return grid.rects.map((r, i) => ({ r, i })).sort((a, b) => a.r.row - b.r.row || (a.r.row % 2 ? b.r.col - a.r.col : a.r.col - b.r.col));
}

async function analyze() {
  const token = ++state.token;
  const started = performance.now();
  state.grid = windowLayout();
  const { rects } = state.grid;
  state.cnnScores = new Array(rects.length);

  // The Laplacian is instant: the whole picture and every window, on the working copy.
  const scale = state.work.width / state.image.width;
  state.whole.variance = laplacianVariance(state.work);
  state.lapVariances = laplacianVariances(state.work, rects.map((r) => ({
    x: Math.floor(r.x * scale), y: Math.floor(r.y * scale),
    w: Math.max(1, Math.round(r.w * scale)), h: Math.max(1, Math.round(r.h * scale)),
  })).map((r) => ({ ...r, w: Math.min(r.w, state.work.width - r.x), h: Math.min(r.h, state.work.height - r.y) })));
  showLaplacian();
  drawOverlay();

  if (!state.detector) {
    showCnn(null);
    return;
  }
  status('<span class="dot"></span>looking at the whole picture…', true);
  state.whole.cnn = null;
  showCnn(null);
  await nextFrame();
  const [whole] = await state.detector.predict([state.image]);
  if (token !== state.token) return;
  state.whole.cnn = whole;
  showCnn(whole);

  const reticle = $('reticle');
  reticle.hidden = false;
  const order = scanOrder(state.grid);
  for (let n = 0; n < order.length; n++) {
    const { r, i } = order[n];
    placeReticle(r);
    status(`<span class="dot"></span>scanning window ${n + 1} / ${order.length}`, true);
    await nextFrame();
    const [p] = await state.detector.predict([crop(state.image, r.x, r.y, r.w, r.h)]);
    if (token !== state.token) return;
    state.cnnScores[i] = p;
    drawOverlay();
  }
  reticle.hidden = true;
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  status(`${order.length} windows of ${state.grid.size} px · ${seconds} s`);
}

// ---------------------------------------------------------------- stage drawing

function layoutStage() {
  const stage = $('stage');
  const { width, height } = state.image;
  stage.style.aspectRatio = `${width} / ${height}`;
  stage.style.width = `min(100%, calc(74vh * ${(width / height).toFixed(4)}))`;
  stage.style.marginInline = 'auto';
  drawPhoto();
}

function sizeCanvas(canvas) {
  const stage = $('stage');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(stage.clientWidth * dpr);
  canvas.height = Math.round(stage.clientHeight * dpr);
  return canvas.getContext('2d');
}

function drawPhoto() {
  if (!state.bitmap) return;
  const canvas = $('photo');
  const ctx = sizeCanvas(canvas);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(state.bitmap, 0, 0, canvas.width, canvas.height);
  drawOverlay();
}

function currentScores() {
  const mode = document.querySelector('input[name=overlay]:checked').value;
  if (mode === 'cnn') return state.cnnScores;
  if (mode === 'laplacian') return state.lapVariances.map((v) => laplacianScore(v, state.threshold));
  return null;
}

function drawOverlay() {
  const canvas = $('overlay');
  const ctx = sizeCanvas(canvas);
  const scores = state.grid && currentScores();
  if (!scores) return;
  const { width, height } = state.image;
  const cols = width >= height ? cfg.HEATMAP_FIELD_CELLS : Math.max(1, Math.round(cfg.HEATMAP_FIELD_CELLS * width / height));
  const rows = height > width ? cfg.HEATMAP_FIELD_CELLS : Math.max(1, Math.round(cfg.HEATMAP_FIELD_CELLS * height / width));
  const field = heatField(state.grid.rects, scores, width, height, cols, rows);
  const small = new ImageData(cols, rows);
  const opacity = Number($('opacity').value);
  field.forEach((score, i) => small.data.set(scoreColor(score, cfg.HEATMAP_COLORS.sharp, cfg.HEATMAP_COLORS.blurry, opacity), i * 4));
  const tile = document.createElement('canvas');
  tile.width = cols;
  tile.height = rows;
  tile.getContext('2d').putImageData(small, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(tile, 0, 0, canvas.width, canvas.height);
}

function placeReticle(r) {
  const k = $('stage').clientWidth / state.image.width;
  const reticle = $('reticle');
  reticle.style.width = `${r.w * k}px`;
  reticle.style.height = `${r.h * k}px`;
  reticle.style.transform = `translate(${r.x * k}px, ${r.y * k}px)`;
}

function status(text, html = false) {
  const hud = $('hud-status');
  if (html) hud.innerHTML = text;
  else hud.textContent = text;
}

// ---------------------------------------------------------------- meters

const percent = (p) => `${(p * 100).toFixed(p > 0.005 && p < 0.995 ? 0 : 1)}%`;
const verdictOf = (blurry) => (blurry ? 'blurry' : 'sharp');

function setVerdict(el, blurry) {
  el.textContent = blurry === null ? '' : verdictOf(blurry);
  el.dataset.state = blurry === null ? '' : verdictOf(blurry);
}

function showCnn(p) {
  $('cnn-value').textContent = p === null ? '–' : percent(p);
  $('cnn-needle').style.transform = `rotate(${-90 + (p ?? 0.5) * 180}deg)`;
  setVerdict($('cnn-verdict'), p === null ? null : p >= 0.5);
  showAgreement();
}

const logPosition = (v) => Math.min(1, Math.max(0, Math.log10(Math.max(v, 1)) / 4)); // 1 .. 10k

function showLaplacian() {
  const v = state.whole.variance;
  $('lap-threshold-mark').style.left = `${logPosition(state.threshold) * 100}%`;
  if (v === null) return;
  $('lap-value').textContent = v >= 100 ? Math.round(v).toLocaleString('en') : v.toFixed(1);
  const pin = $('lap-pin');
  pin.classList.add('on');
  pin.style.left = `${logPosition(v) * 100}%`;
  setVerdict($('lap-verdict'), v < state.threshold);
  showAgreement();
}

function showAgreement() {
  const el = $('agreement');
  const { cnn, variance } = state.whole;
  if (variance === null) { el.textContent = ''; return; }
  const lap = verdictOf(variance < state.threshold);
  if (!state.detector) {
    el.innerHTML = `The Laplacian says <b>${lap}</b>. The CNN is not available: no exported model was found.`;
    return;
  }
  if (cnn === null) { el.textContent = 'The CNN is thinking…'; return; }
  const net = verdictOf(cnn >= 0.5);
  el.innerHTML = net === lap
    ? `Both detectors say <b>${net}</b>.`
    : `They <b>disagree</b>: the CNN says ${net}, the Laplacian says ${lap}.`;
}

function buildDialTicks() {
  const g = document.querySelector('.dial .ticks');
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    const a = Math.PI * (1 - p);
    const [x1, y1, x2, y2] = [100 + 66 * Math.cos(a), 100 - 66 * Math.sin(a), 100 + 60 * Math.cos(a), 100 - 60 * Math.sin(a)];
    const line = document.createElementNS(SVG_NS, 'line');
    Object.entries({ x1, y1, x2, y2 }).forEach(([k, v]) => line.setAttribute(k, v.toFixed(1)));
    const text = document.createElementNS(SVG_NS, 'text');
    text.setAttribute('x', (100 + 50 * Math.cos(a)).toFixed(1));
    text.setAttribute('y', (100 - 50 * Math.sin(a) + 3).toFixed(1));
    text.textContent = `${p * 100}`;
    g.append(line, text);
  }
}

// ---------------------------------------------------------------- threshold slider (log scale)

const SLIDER_STEPS = 1000;
const { min: T_MIN, max: T_MAX } = cfg.LAPLACIAN_THRESHOLD;
const toThreshold = (v) => T_MIN * (T_MAX / T_MIN) ** (v / SLIDER_STEPS);
const toSlider = (t) => Math.round((Math.log(t / T_MIN) / Math.log(T_MAX / T_MIN)) * SLIDER_STEPS);

function setupThreshold() {
  const slider = $('threshold');
  slider.min = 0;
  slider.max = SLIDER_STEPS;
  slider.value = toSlider(state.threshold);
  const update = () => {
    state.threshold = toThreshold(Number(slider.value));
    $('threshold-out').textContent = Math.round(state.threshold);
    showLaplacian();
    drawOverlay();
    if (state.stress) drawStress();
  };
  slider.addEventListener('input', update);
  update();
}

// ---------------------------------------------------------------- stress test

function setupStress() {
  const seg = $('stress-kind');
  Object.entries(BLURS).forEach(([key, blur], i) => {
    const label = document.createElement('label');
    label.innerHTML = `<input type="radio" name="stress-kind" value="${key}"${i === 0 ? ' checked' : ''}><span>${blur.label}</span>`;
    seg.append(label);
  });
  $('stress-run').addEventListener('click', runStress);
  $('stress-legend').innerHTML = '<span><i class="swatch cnn"></i>CNN · P(blur)</span><span><i class="swatch lap"></i>Laplacian score</span>';
}

async function runStress() {
  const kind = document.querySelector('input[name=stress-kind]:checked').value;
  const blur = BLURS[kind];
  const steps = Array.from({ length: cfg.STRESS_STEPS }, (_, i) => (cfg.STRESS_MAX[kind] * i) / (cfg.STRESS_STEPS - 1));
  const button = $('stress-run');
  button.disabled = true;
  state.stress = { kind, unit: blur.unit, points: [] };
  $('stress-empty').hidden = true;
  $('stress-strip').replaceChildren();
  for (const [n, strength] of steps.entries()) {
    $('stress-status').textContent = `step ${n + 1} / ${steps.length}…`;
    await nextFrame();
    const img = strength ? blur.apply(state.work, strength) : state.work;
    const [p] = await state.detector.predict([img]);
    state.stress.points.push({ strength, cnn: p, variance: laplacianVariance(img), img });
    drawStress();
    addStripFrame(img, strength, blur.unit, p);
  }
  $('stress-status').textContent = `${blur.label} blur, ${steps.length} steps`;
  button.disabled = false;
}

function addStripFrame(img, strength, unit, p) {
  const figure = document.createElement('figure');
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(img.data), img.width, img.height), 0, 0);
  const caption = document.createElement('figcaption');
  caption.innerHTML = `<span>${strength.toFixed(1)} ${unit}</span><span>CNN ${percent(p)}</span>`;
  figure.append(canvas, caption);
  $('stress-strip').append(figure);
}

function drawStress() {
  const svg = $('stress-chart');
  const { points, unit } = state.stress;
  const width = svg.clientWidth || 800;
  const height = svg.clientHeight || 280;
  const pad = { l: 44, r: 92, t: 12, b: 38 };
  const xMax = cfg.STRESS_MAX[state.stress.kind];
  const x = (s) => pad.l + (s / xMax) * (width - pad.l - pad.r);
  const y = (v) => pad.t + (1 - v) * (height - pad.t - pad.b);
  const series = [
    { key: 'cnn', name: 'CNN', value: (pt) => pt.cnn },
    { key: 'lap', name: 'Laplacian', value: (pt) => laplacianScore(pt.variance, state.threshold) },
  ];
  const parts = [];
  parts.push('<g class="grid">', ...[0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="${pad.l}" x2="${width - pad.r}" y1="${y(v)}" y2="${y(v)}"/>`), '</g>');
  parts.push(`<line class="mid-line" x1="${pad.l}" x2="${width - pad.r}" y1="${y(0.5)}" y2="${y(0.5)}"/>`);
  parts.push('<g class="axis">', ...[0, 0.5, 1].map((v) => `<text x="${pad.l - 10}" y="${y(v) + 4}" text-anchor="end">${v}</text>`));
  for (let i = 0; i <= 4; i++) {
    const s = (xMax * i) / 4;
    parts.push(`<text x="${x(s)}" y="${height - pad.b + 18}" text-anchor="middle">${+s.toFixed(1)}</text>`);
  }
  parts.push(`<text class="title" x="${width - pad.r}" y="${height - 4}" text-anchor="end">blur strength (${unit}) →</text>`, '</g>');
  for (const s of series) {
    const d = points.map((pt, i) => `${i ? 'L' : 'M'}${x(pt.strength).toFixed(1)},${y(s.value(pt)).toFixed(1)}`).join(' ');
    parts.push(`<path class="series ${s.key}" d="${d}"/>`);
    parts.push(...points.map((pt) => `<circle class="marker ${s.key}" cx="${x(pt.strength)}" cy="${y(s.value(pt))}" r="4.5"/>`));
  }
  // Direct labels at the end of each line, pushed apart when the lines end close together.
  const last = points[points.length - 1];
  if (last) {
    const labels = series.map((s) => ({ name: s.name, y: y(s.value(last)) + 4 })).sort((a, b) => a.y - b.y);
    if (labels[1].y - labels[0].y < 14) {
      const mid = (labels[0].y + labels[1].y) / 2;
      labels[0].y = mid - 7;
      labels[1].y = mid + 7;
    }
    parts.push(...labels.map((l) => `<text class="direct" x="${x(last.strength) + 10}" y="${l.y}">${l.name}</text>`));
  }
  parts.push('<line class="crosshair" id="stress-cross" y1="0" y2="0" hidden/>');
  const half = points.length > 1 ? (x(points[1].strength) - x(points[0].strength)) / 2 : 20;
  parts.push(...points.map((pt, i) => `<rect class="hit" data-i="${i}" x="${x(pt.strength) - half}" y="${pad.t}" width="${half * 2}" height="${height - pad.t - pad.b}"/>`));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = parts.join('');

  const tip = $('stress-tip');
  const cross = svg.querySelector('#stress-cross');
  svg.querySelectorAll('.hit').forEach((rect) => {
    rect.addEventListener('pointerenter', () => {
      const pt = points[Number(rect.dataset.i)];
      cross.removeAttribute('hidden');
      ['x1', 'x2'].forEach((a) => cross.setAttribute(a, x(pt.strength)));
      cross.setAttribute('y1', pad.t);
      cross.setAttribute('y2', height - pad.b);
      tip.hidden = false;
      tip.innerHTML = `<div><span class="k">blur</span><span>${pt.strength.toFixed(1)} ${unit}</span></div>`
        + `<div><span class="k"><i class="swatch cnn"></i>CNN</span><span>${percent(pt.cnn)}</span></div>`
        + `<div><span class="k"><i class="swatch lap"></i>Laplacian</span><span>${pt.variance.toFixed(1)} → ${laplacianScore(pt.variance, state.threshold).toFixed(2)}</span></div>`;
      const left = Math.min(x(pt.strength) + 18, width - 170);
      tip.style.left = `${left}px`;
      tip.style.top = `${pad.t + 30}px`;
    });
    rect.addEventListener('pointerleave', () => { tip.hidden = true; cross.setAttribute('hidden', ''); });
  });

  $('stress-table').innerHTML = `<thead><tr><th>blur (${unit})</th><th>CNN P(blur)</th><th>Laplacian variance</th><th>Laplacian score</th></tr></thead><tbody>${
    points.map((pt) => `<tr><td>${pt.strength.toFixed(1)}</td><td>${pt.cnn.toFixed(3)}</td><td>${pt.variance.toFixed(1)}</td><td>${laplacianScore(pt.variance, state.threshold).toFixed(3)}</td></tr>`).join('')}</tbody>`;
}

// ---------------------------------------------------------------- model card

function renderModelCard(card) {
  const list = $('model-card-list');
  const rows = [];
  const p = card.provenance;
  const link = (href, text) => `<a href="${href}">${text}</a>`;
  rows.push(['trained', p
    ? `by GitHub Actions, ${link(p.run_url, `run ${p.run_id}`)}, commit ${link(`https://github.com/${p.repository}/commit/${p.commit}`, p.commit.slice(0, 7))}`
    : 'on a local machine']);
  rows.push(['exported', new Date(card.created_at).toISOString().replace('T', ' ').slice(0, 16) + ' UTC']);
  const m = card.metrics;
  if (m.test) rows.push(['test accuracy', `${percent(m.test.accuracy)} on ${m.test.count} pictures`]);
  const gate = card.publish;
  if (gate) {
    rows.push(['published if', `at least ${percent(gate.min_test_accuracy)} of the test pictures are right `
      + `(${gate.needed} of ${m.test?.count}; this model: ${gate.correct}).${gate.note ? ` ${gate.note}` : ''}`]);
  }
  if (m.test_windows) rows.push(['window accuracy', `${percent(m.test_windows.accuracy)} on ${m.test_windows.count} heat map windows of the test pictures`]);
  if (m.train) rows.push(['training set', `${m.train.count} pictures (${percent(m.train.accuracy)} right)`]);
  rows.push(['input', `${card.input.size} × ${card.input.size} ${card.input.channels}, ${card.input.layout}, ${card.input.range.join('–')}`]);
  rows.push(['ONNX vs Keras', `largest difference ${card.onnx_max_difference.toExponential(1)}`]);
  rows.push(['built with', Object.entries(card.versions).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(' · ')]);
  rows.push(['runs on', `onnxruntime-web ${cfg.ORT_VERSION} (WebAssembly)`]);
  list.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ---------------------------------------------------------------- input

function takeFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  ++state.choice;
  history.replaceState(null, '', location.pathname);
  caption(null);
  markCard(null);
  openImage(file, file.name || 'pasted image');
}

function setupInput() {
  const stage = $('stage');
  const take = takeFile;
  $('file').addEventListener('change', (e) => take(e.target.files[0]));
  ['dragenter', 'dragover'].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach((t) => stage.addEventListener(t, () => stage.classList.remove('dragging')));
  stage.addEventListener('drop', (e) => { e.preventDefault(); take(e.dataTransfer.files[0]); });
  document.addEventListener('paste', (e) => take([...e.clipboardData.files].find((f) => f.type.startsWith('image/'))));
  stage.addEventListener('dblclick', () => $('file').click());
  stage.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } });
  document.querySelectorAll('input[name=overlay]').forEach((el) => el.addEventListener('change', drawOverlay));
  document.querySelectorAll('input[name=detail]').forEach((el) => el.addEventListener('change', () => state.image && analyze()));
  $('opacity').addEventListener('input', drawOverlay);
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { drawPhoto(); if (state.stress) drawStress(); }, 120);
  });
}

// ---------------------------------------------------------------- start

async function main() {
  document.querySelectorAll('[data-repo]').forEach((a) => { a.href = cfg.REPO_URL; });
  buildDialTicks();
  setupThreshold();
  setupStress();
  setupInput();
  const [model] = await Promise.allSettled([loadModel(), loadCards()]);
  if (model.status === 'rejected') {
    console.warn('CNN unavailable:', model.reason);
    $('model-card-list').innerHTML = '<dt>status</dt><dd>no model found next to this page. Run <code>blur-detector export</code> (see the README) to make one.</dd>';
    status('no CNN model · Laplacian only');
  }
  await openDanbooru();
}

main();
