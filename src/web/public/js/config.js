// Settings of the web demo. The model's own facts (input size, class order) come from the model card,
// not from here.

/** ONNX Runtime Web, loaded from jsDelivr. Keep in step with devDependencies in ../../package.json (a test checks). */
export const ORT_VERSION = '1.30.0';
export const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.bundle.min.mjs`;

/** Files written by `blur-detector export` and `tools/make_samples.py`, relative to this page. */
export const MODEL_URL = 'model/blur-detector.onnx';
export const MODEL_CARD_URL = 'model/model-card.json';
export const SAMPLES_URL = 'samples/samples.json';
export const SAMPLES_DIR = 'samples/';

export const REPO_URL = 'https://github.com/NatLee/simply-blur-detector';

/**
 * The page's default picture: a random anime picture from Danbooru, loaded by the browser at run time
 * (nothing from Danbooru is stored in the repository or the site). Only general-audience posts are
 * asked for, and every answer's rating is checked again before its picture is loaded. One API request
 * per picture; the current one is kept for the browser session, so reloading asks for nothing new.
 */
export const DANBOORU = {
  site: 'https://danbooru.donmai.us',
  randomPath: '/posts/random.json',
  tags: 'rating:g',
  rating: 'g',
  /** Still pictures the browser can decode; videos, Ugoira zips and GIF animations are skipped. */
  extensions: ['jpg', 'jpeg', 'png', 'webp'],
  /** Posts to try in a row when one has no usable picture (a video, a removed file). */
  attempts: 3,
  timeoutMs: 15000,
  cacheKey: 'simply-blur-detector.danbooru',
};

/** The Laplacian is measured on a copy whose longer side is at most this, so big photos are comparable. */
export const LAPLACIAN_WORK_SIZE = 512;
/** Laplacian variance below this counts as blurry (a common starting point; the page has a slider). */
export const LAPLACIAN_THRESHOLD = { initial: 100, min: 5, max: 2000 };

/**
 * How much neighbouring heat map windows overlap. The window size itself comes from the model card
 * (the model was trained on crops down to that size), so finer detail means more overlap.
 */
export const HEATMAP_DETAIL = {
  coarse: { overlap: 0.5 },
  fine: { overlap: 0.75 },
};
export const HEATMAP_DEFAULT_DETAIL = 'coarse';
/** Cells of the blended heat field along the longer side (the overlay is smoothed when drawn). */
export const HEATMAP_FIELD_CELLS = 48;
/** Overlay colours: [r, g, b] for sharp and for blurry. */
export const HEATMAP_COLORS = { sharp: [45, 226, 230], blurry: [255, 84, 54] };

/** Stress test: how many blur steps, and the largest blur (in pixels of the working copy). */
export const STRESS_STEPS = 8;
export const STRESS_MAX = { gaussian: 5, motion: 30, lens: 8 };
