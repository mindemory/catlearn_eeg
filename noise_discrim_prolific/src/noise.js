// 1/f^alpha noise patches, generated in the browser from a seed. Same recipe as
// task_design/noise_patches.py (white noise -> FFT -> amplitude k^-alpha -> inverse FFT ->
// RMS contrast -> clip), plus a fixed upper cutoff in cycles per degree (config.js NOISE).
// tools/noise_field.py reproduces every field exactly (same PRNG, same order of draws).

import { LAYOUT, NOISE } from './config.js';

// Seeded PRNG (mulberry32): floats in [0, 1)
export function makeRng(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal draws, Box-Muller, both values of each pair used (cos first, then sin)
function gaussians(n, rng) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = 1 - rng();                        // (0, 1]: log never sees 0
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

// In-place radix-2 complex FFT of length n (a power of 2); inverse = true for the inverse
// transform (with the 1/n scaling)
function fft1(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const nr = cr * wr - ci * wi;          // no temporary arrays: they pile up garbage
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// 2-D FFT of n x n row-major arrays, in place
function fft2(re, im, n, inverse) {
  const rr = new Float64Array(n);
  const ri = new Float64Array(n);
  for (let pass = 0; pass < 2; pass++) {
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        const idx = pass === 0 ? a * n + b : b * n + a;    // rows, then columns
        rr[b] = re[idx];
        ri[b] = im[idx];
      }
      fft1(rr, ri, inverse);
      for (let b = 0; b < n; b++) {
        const idx = pass === 0 ? a * n + b : b * n + a;
        re[idx] = rr[b];
        im[idx] = ri[b];
      }
    }
  }
}

/**
 * Noise field for one patch: n x n values in [-1, 1] (row-major, n = NOISE.grid).
 * @param {number} alpha   spectral slope (0 = white noise up to the cutoff)
 * @param {number} seed    32-bit seed; the same seed and alpha always give the same field
 */
export function noiseField(alpha, seed) {
  const n = NOISE.grid;
  const kMax = NOISE.cutoffCpd * LAYOUT.patchSize;       // cutoff in cycles per patch
  const re = gaussians(n * n, makeRng(seed));
  const im = new Float64Array(n * n);
  fft2(re, im, n, false);
  for (let y = 0; y < n; y++) {
    const fy = y <= n / 2 ? y : y - n;                   // signed frequency, as numpy fftfreq * n
    for (let x = 0; x < n; x++) {
      const fx = x <= n / 2 ? x : x - n;
      const k = Math.hypot(fx, fy);                      // cycles per patch
      const amp = k === 0 || k > kMax ? 0 : k ** -alpha;
      re[y * n + x] *= amp;
      im[y * n + x] *= amp;
    }
  }
  fft2(re, im, n, true);
  let ss = 0;
  for (let i = 0; i < n * n; i++) ss += re[i] * re[i];
  const mean = re.reduce((s, v) => s + v, 0) / (n * n);
  const sd = Math.sqrt(ss / (n * n) - mean * mean);
  const out = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) out[i] = Math.max(-1, Math.min(1, (re[i] * NOISE.rms) / sd));
  return out;
}

// Circular aperture with a raised-cosine edge: 1 inside, 0 outside (as noise_patches.py)
let apertureCache = null;
export function aperture() {
  const n = NOISE.grid;
  if (apertureCache) return apertureCache;
  const inner = 1 - NOISE.apertureEdge;
  apertureCache = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const cx = -1 + (2 * x) / (n - 1);
      const cy = -1 + (2 * y) / (n - 1);
      const r = Math.hypot(cx, cy);
      apertureCache[y * n + x] = r <= inner ? 1 : r > 1 ? 0 : 0.5 * (1 + Math.cos((Math.PI * (r - inner)) / NOISE.apertureEdge));
    }
  }
  return apertureCache;
}

// Draw a field into a canvas (NOISE.grid pixels square): grey 0.5 + 0.5 * field * aperture,
// opaque, so outside the aperture it is the background grey
export function drawField(canvas, field) {
  const n = NOISE.grid;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(n, n);
  const ap = aperture();
  for (let i = 0; i < n * n; i++) {
    const v = Math.round(255 * (0.5 + 0.5 * field[i] * ap[i]));
    img.data[4 * i] = v;
    img.data[4 * i + 1] = v;
    img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// A field as a PNG data URL (for the static examples in the instructions)
export function fieldDataUrl(field) {
  const canvas = document.createElement('canvas');
  canvas.width = NOISE.grid;
  canvas.height = NOISE.grid;
  drawField(canvas, field);
  return canvas.toDataURL('image/png');
}
