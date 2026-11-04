// Image operations on plain pixel arrays, ported from OpenCV so that the browser prepares images
// exactly like the Python side (blur_detection.images.prepare_image) did when the model was trained.
// tests/fixtures/opencv-parity.json pins the results against OpenCV itself.
//
// An image is { width, height, data } with RGBA bytes, the layout of canvas ImageData.

/** @typedef {{ width: number, height: number, data: Uint8ClampedArray | Uint8Array }} RGBAImage */

const CHANNELS = 4; // RGBA
const COLOR_CHANNELS = 3; // R, G, B; alpha is ignored like cv2.IMREAD_COLOR does

/** Copy the rectangle (x, y, w, h) out of an image. */
export function crop(img, x, y, w, h) {
  const data = new Uint8ClampedArray(w * h * CHANNELS);
  for (let row = 0; row < h; row++) {
    const start = ((y + row) * img.width + x) * CHANNELS;
    data.set(img.data.subarray(start, start + w * CHANNELS), row * w * CHANNELS);
  }
  return { width: w, height: h, data };
}

/** Round half to even, like OpenCV's cvRound / saturate_cast on x86 and ARM. */
function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Lookup table of cv2.equalizeHist for one channel's histogram (float32 maths, as in OpenCV). */
function equalizeLut(hist, total) {
  const lut = new Uint8Array(256);
  let first = 0;
  while (first < 256 && hist[first] === 0) first++;
  if (first === 256 || hist[first] === total) {
    lut.fill(first === 256 ? 0 : first); // a flat channel keeps its value
    return lut;
  }
  const scale = Math.fround(255 / (total - hist[first]));
  let sum = 0;
  for (let i = first + 1; i < 256; i++) {
    sum += hist[i];
    lut[i] = Math.min(255, Math.max(0, roundHalfEven(Math.fround(sum * scale))));
  }
  return lut;
}

/** Equalize the histogram of R, G and B separately (cv2.equalizeHist per channel). */
export function equalizeHist(img) {
  const n = img.width * img.height;
  const out = new Uint8ClampedArray(img.data.length);
  for (let c = 0; c < CHANNELS; c++) {
    if (c === COLOR_CHANNELS) {
      for (let i = 0; i < n; i++) out[i * CHANNELS + c] = img.data[i * CHANNELS + c];
      continue;
    }
    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i++) hist[img.data[i * CHANNELS + c]]++;
    const lut = equalizeLut(hist, n);
    for (let i = 0; i < n; i++) out[i * CHANNELS + c] = lut[img.data[i * CHANNELS + c]];
  }
  return { width: img.width, height: img.height, data: out };
}

const COEF_SCALE = 2048; // OpenCV's INTER_RESIZE_COEF_SCALE: 11-bit fixed-point weights

/** Source index and fixed-point weights per destination pixel, like OpenCV's INTER_LINEAR. */
function linearTaps(srcSize, dstSize) {
  const scale = 1 / (dstSize / srcSize); // OpenCV stores the inverse scale and divides
  const index = new Int32Array(dstSize);
  const w0 = new Int32Array(dstSize);
  const w1 = new Int32Array(dstSize);
  for (let d = 0; d < dstSize; d++) {
    let f = Math.fround((d + 0.5) * scale - 0.5); // OpenCV keeps the fraction as a float
    let s = Math.floor(f);
    f = Math.fround(f - s);
    if (s < 0) { s = 0; f = 0; }
    if (s >= srcSize - 1) { s = srcSize - 1; f = 0; }
    index[d] = s;
    w0[d] = roundHalfEven(Math.fround(1 - f) * COEF_SCALE);
    w1[d] = roundHalfEven(f * COEF_SCALE);
  }
  return { index, w0, w1 };
}

/**
 * Resize with bilinear interpolation the way cv2.resize(..., INTER_LINEAR) does for 8-bit images:
 * half-pixel centres, edge pixels repeated, 11-bit fixed-point weights. OpenCV's SIMD code rounds a
 * little differently, so single values can be off by one.
 */
export function resizeBilinear(img, width, height) {
  const xs = linearTaps(img.width, width);
  const ys = linearTaps(img.height, height);
  const out = new Uint8ClampedArray(width * height * CHANNELS);
  const rowA = new Int32Array(width * CHANNELS);
  const rowB = new Int32Array(width * CHANNELS);
  const horizontal = (srcRow, target) => {
    const base = srcRow * img.width * CHANNELS;
    for (let x = 0; x < width; x++) {
      const s = base + xs.index[x] * CHANNELS;
      const t = xs.w1[x] ? s + CHANNELS : s;
      for (let c = 0; c < CHANNELS; c++) {
        target[x * CHANNELS + c] = img.data[s + c] * xs.w0[x] + img.data[t + c] * xs.w1[x];
      }
    }
  };
  for (let y = 0; y < height; y++) {
    const s = ys.index[y];
    horizontal(s, rowA);
    horizontal(ys.w1[y] ? s + 1 : s, rowB);
    const b0 = ys.w0[y];
    const b1 = ys.w1[y];
    for (let i = 0; i < width * CHANNELS; i++) {
      out[y * width * CHANNELS + i] = (rowA[i] * b0 + rowB[i] * b1 + (1 << 21)) >> 22;
    }
  }
  return { width, height, data: out };
}

/**
 * One model input, prepared like blur_detection.images.prepare_image: equalize every channel,
 * resize to size x size, scale to 0-1. Returns size * size * 3 floats in RGB (NHWC) order.
 */
export function toModelInput(img, size) {
  const small = resizeBilinear(equalizeHist(img), size, size);
  const input = new Float32Array(size * size * COLOR_CHANNELS);
  for (let i = 0, j = 0; i < size * size; i++) {
    for (let c = 0; c < COLOR_CHANNELS; c++) input[j++] = small.data[i * CHANNELS + c] / 255;
  }
  return input;
}

/** Luminance like cv2.cvtColor(..., COLOR_RGB2GRAY) in OpenCV 5: (9798 R + 19235 G + 3735 B) / 2^15, rounded. */
export function grayscale(img) {
  const n = img.width * img.height;
  const gray = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * CHANNELS;
    gray[i] = (img.data[p] * 9798 + img.data[p + 1] * 19235 + img.data[p + 2] * 3735 + 16384) >> 15;
  }
  return gray;
}

/** Mirror an index into 0..n-1 without repeating the edge (OpenCV's BORDER_REFLECT_101). */
function reflect101(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** cv2.Laplacian(gray, CV_64F) with the default 3x3 kernel [0 1 0; 1 -4 1; 0 1 0]. */
export function laplacian(gray, width, height) {
  const out = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    const up = reflect101(y - 1, height) * width;
    const down = reflect101(y + 1, height) * width;
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const left = reflect101(x - 1, width);
      const right = reflect101(x + 1, width);
      out[row + x] = gray[up + x] + gray[down + x] + gray[row + left] + gray[row + right] - 4 * gray[row + x];
    }
  }
  return out;
}

/** Variance of an array (population variance, like NumPy's .var()). */
export function variance(values) {
  let mean = 0;
  for (let i = 0; i < values.length; i++) mean += values[i];
  mean /= values.length;
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += (values[i] - mean) ** 2;
  return sum / values.length;
}

/** The classic blur score: variance of the Laplacian of the grayscale image. Low means blurry. */
export function laplacianVariance(img) {
  return variance(laplacian(grayscale(img), img.width, img.height));
}

/**
 * Laplacian variance of many rectangles of one image at once. The Laplacian is taken over the whole
 * image (so window edges see their real neighbours) and summed with integral images.
 * Returns one variance per rectangle {x, y, w, h}.
 */
export function laplacianVariances(img, rects) {
  const { width, height } = img;
  const lap = laplacian(grayscale(img), width, height);
  const stride = width + 1;
  const sum = new Float64Array(stride * (height + 1));
  const sq = new Float64Array(stride * (height + 1));
  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    let rowSq = 0;
    for (let x = 0; x < width; x++) {
      const v = lap[y * width + x];
      rowSum += v;
      rowSq += v * v;
      sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + rowSum;
      sq[(y + 1) * stride + x + 1] = sq[y * stride + x + 1] + rowSq;
    }
  }
  const box = (table, r) => table[(r.y + r.h) * stride + r.x + r.w] - table[r.y * stride + r.x + r.w]
    - table[(r.y + r.h) * stride + r.x] + table[r.y * stride + r.x];
  return rects.map((r) => {
    const n = r.w * r.h;
    const mean = box(sum, r) / n;
    return Math.max(0, box(sq, r) / n - mean * mean);
  });
}
