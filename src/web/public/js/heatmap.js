// Where to look and how to colour it: the sliding windows of the heat maps, and turning one score
// per window into a smooth map.

/**
 * Square windows that cover the whole image. Each side is `windowFraction` of the image's shorter
 * side, and neighbours overlap by `overlap` (0.5 = half a window). The last row and column are moved
 * in so that they end exactly at the image edge.
 */
export function windowGrid(width, height, { windowFraction, overlap }) {
  const size = Math.max(1, Math.min(width, height, Math.round(Math.min(width, height) * windowFraction)));
  const stride = Math.max(1, Math.round(size * (1 - overlap)));
  const starts = (length) => {
    const s = [];
    for (let p = 0; p + size <= length; p += stride) s.push(p);
    if (s[s.length - 1] + size < length) s.push(length - size);
    return s;
  };
  const xs = starts(width);
  const ys = starts(height);
  const rects = ys.flatMap((y, row) => xs.map((x, col) => ({ x, y, w: size, h: size, row, col })));
  return { size, cols: xs.length, rows: ys.length, rects };
}

/**
 * Blend window scores into a field of `cols` x `rows` cells over a width x height image. Every cell
 * takes the average of the windows that cover it, weighted by how close it is to each window's
 * centre, so overlapping windows fade into each other. Cells no window has scored yet are NaN.
 */
export function heatField(rects, scores, width, height, cols, rows) {
  const field = new Float32Array(cols * rows);
  for (let cy = 0; cy < rows; cy++) {
    const py = ((cy + 0.5) / rows) * height;
    for (let cx = 0; cx < cols; cx++) {
      const px = ((cx + 0.5) / cols) * width;
      let total = 0;
      let weight = 0;
      rects.forEach((r, i) => {
        const score = scores[i];
        if (score === undefined || Number.isNaN(score)) return;
        const dx = Math.abs(px - (r.x + r.w / 2)) / (r.w / 2);
        const dy = Math.abs(py - (r.y + r.h / 2)) / (r.h / 2);
        if (dx > 1 || dy > 1) return;
        const w = (1.05 - dx) * (1.05 - dy); // never exactly 0 inside the window
        total += w * score;
        weight += w;
      });
      field[cy * cols + cx] = weight ? total / weight : Number.NaN;
    }
  }
  return field;
}

/**
 * Colour of a blur score (0 = sharp, 1 = blurry) on the overlay: `sharp` and `blurry` are [r, g, b];
 * the colour fades out towards 0.5, where neither detector is sure. Returns [r, g, b, a] (a in 0-255).
 */
export function scoreColor(score, sharp, blurry, opacity) {
  if (Number.isNaN(score)) return [0, 0, 0, 0];
  const base = score >= 0.5 ? blurry : sharp;
  const confidence = Math.min(1, Math.abs(score - 0.5) * 2);
  return [...base, Math.round(255 * opacity * (0.2 + 0.8 * confidence))];
}

/**
 * The Laplacian variance as a 0-1 blur score: threshold / (threshold + variance). It is 0.5 at the
 * threshold, about 0.09 at ten times the threshold (sharp) and about 0.91 at a tenth of it (blurry).
 */
export function laplacianScore(variance, threshold) {
  return threshold / (threshold + variance);
}
