// Blurs for the stress test, done in the browser: Gaussian, motion and lens, like the three kinds
// blur-generator adds to the training data. They are quick approximations, not ports of
// blur-generator, so the numbers are close to but not the same as the Python figure.

const CHANNELS = 4;

/** Mirror an index into 0..n-1 without repeating the edge (BORDER_REFLECT_101). */
function reflect(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** Run `fn(src, dst, index, step, count)` over every row and then every column of float planes. */
function separable(img, pass) {
  const { width, height } = img;
  const planes = [0, 1, 2].map((c) => Float32Array.from({ length: width * height }, (_, i) => img.data[i * CHANNELS + c]));
  const tmp = new Float32Array(width * height);
  for (const plane of planes) {
    for (let y = 0; y < height; y++) pass(plane, tmp, y * width, 1, width);
    for (let x = 0; x < width; x++) pass(tmp, plane, x, width, height);
  }
  return toImage(img, (i, c) => planes[c][i]);
}

function toImage(img, value) {
  const out = new Uint8ClampedArray(img.data.length);
  for (let i = 0; i < img.width * img.height; i++) {
    for (let c = 0; c < 3; c++) out[i * CHANNELS + c] = value(i, c);
    out[i * CHANNELS + 3] = img.data[i * CHANNELS + 3];
  }
  return { width: img.width, height: img.height, data: out };
}

/** Gaussian blur with standard deviation `sigma` pixels. */
export function gaussianBlur(img, sigma) {
  if (sigma <= 0) return img;
  const radius = Math.ceil(sigma * 3);
  const kernel = Array.from({ length: radius * 2 + 1 }, (_, k) => Math.exp(-((k - radius) ** 2) / (2 * sigma * sigma)));
  const total = kernel.reduce((a, b) => a + b, 0);
  return separable(img, (src, dst, start, step, count) => {
    for (let i = 0; i < count; i++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) acc += src[start + reflect(i + k, count) * step] * kernel[k + radius];
      dst[start + i * step] = acc / total;
    }
  });
}

/** Motion blur: the average along a straight line of `length` pixels at `angle` degrees. */
export function motionBlur(img, length, angle) {
  if (length <= 1) return img;
  const { width, height, data } = img;
  const dx = Math.cos((angle * Math.PI) / 180);
  const dy = -Math.sin((angle * Math.PI) / 180);
  const offsets = Array.from({ length: Math.round(length) }, (_, k) => k - (Math.round(length) - 1) / 2);
  return toImage(img, (i, c) => {
    const x = i % width;
    const y = (i - x) / width;
    let acc = 0;
    for (const t of offsets) {
      const sx = reflect(Math.round(x + t * dx), width);
      const sy = reflect(Math.round(y + t * dy), height);
      acc += data[(sy * width + sx) * CHANNELS + c];
    }
    return acc / offsets.length;
  });
}

/**
 * Lens blur: every pixel becomes the average of a disc of `radius` pixels. Bright spots are boosted
 * before and restored after (`gamma`), so highlights bloom into round bokeh like a real lens.
 */
export function lensBlur(img, radius, gamma = 3) {
  if (radius < 0.5) return img;
  const { width, height, data } = img;
  const r = Math.ceil(radius);
  const spans = Array.from({ length: r * 2 + 1 }, (_, k) => Math.floor(Math.sqrt(Math.max(0, radius * radius - (k - r) ** 2))));
  const area = spans.reduce((a, s) => a + 2 * s + 1, 0);
  const out = new Uint8ClampedArray(data.length);
  const prefix = new Float64Array(width + 1);
  const lit = new Float64Array(width * height);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < width * height; i++) lit[i] = (data[i * CHANNELS + c] / 255) ** gamma;
    const acc = new Float64Array(width * height);
    for (let k = 0; k <= 2 * r; k++) {
      const span = spans[k];
      for (let y = 0; y < height; y++) {
        // Row sums of the source row this disc slice reads, with mirrored edges.
        const row = reflect(y + k - r, height) * width;
        prefix[0] = 0;
        for (let x = 0; x < width; x++) prefix[x + 1] = prefix[x] + lit[row + x];
        for (let x = 0; x < width; x++) {
          const lo = x - span;
          const hi = x + span;
          let sum;
          if (lo >= 0 && hi < width) sum = prefix[hi + 1] - prefix[lo];
          else {
            sum = 0;
            for (let t = lo; t <= hi; t++) sum += lit[row + reflect(t, width)];
          }
          acc[y * width + x] += sum;
        }
      }
    }
    for (let i = 0; i < width * height; i++) out[i * CHANNELS + c] = 255 * (acc[i] / area) ** (1 / gamma);
  }
  for (let i = 0; i < width * height; i++) out[i * CHANNELS + 3] = data[i * CHANNELS + 3];
  return { width, height, data: out };
}

/** The stress-test blurs by name: each takes the image and a strength from 0 (none) to `max`. */
export const BLURS = {
  gaussian: { label: 'Gaussian', unit: 'σ px', apply: (img, s) => gaussianBlur(img, s) },
  motion: { label: 'Motion', unit: 'px', apply: (img, s) => motionBlur(img, s, 18) },
  lens: { label: 'Lens', unit: 'radius px', apply: (img, s) => lensBlur(img, s) },
};
