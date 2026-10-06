// HTML for the trial screens. Positions and sizes are in degrees of visual angle (config.js
// LAYOUT) and become CSS lengths through the --deg variable (pixels per degree, set by
// calibration.js), so the layout has the same visual size on every screen.

import { LAYOUT, NOISE } from './config.js';

const len = (v) => `calc(var(--deg) * ${v.toFixed(4)})`;
const at = (x, y) => `left: calc(50% + ${len(x)}); top: calc(50% - ${len(y)});`;

// Bull's eye fixation (Thaler et al., 2013), as in the MATLAB task: black disc, white cross,
// centre dot in `dot` (black, or the feedback colour)
export function fixation(dot = '#000') {
  const { outer, inner, line } = LAYOUT.fixation;
  const r = 50;
  const w = (line / outer) * 2 * r;
  const ri = (inner / outer) * r;
  return `<svg class="fixation" viewBox="-50 -50 100 100" style="${at(0, 0)} width: ${len(outer)}; height: ${len(outer)};">`
    + `<circle r="${r}" fill="#000"/>`
    + `<rect x="${-r}" y="${-w / 2}" width="${2 * r}" height="${w}" fill="#fff"/>`
    + `<rect x="${-w / 2}" y="${-r}" width="${w}" height="${2 * r}" fill="#fff"/>`
    + `<circle r="${ri}" fill="${dot}"/></svg>`;
}

// An empty canvas for a patch at `side`; noise.js drawField() fills it
function patchCanvas(side) {
  const [x, y] = LAYOUT.positions[side];
  const s = len(LAYOUT.patchSize);
  return `<canvas class="patch" id="patch-${side}" width="${NOISE.grid}" height="${NOISE.grid}"`
    + ` style="${at(x, y)} width: ${s}; height: ${s};"></canvas>`;
}

/**
 * One screen: the fixation, plus the two patch canvases (patches = true) or a feedback
 * colour on the fixation (dot).
 */
export function screen({ patches = false, dot = '#000' } = {}) {
  const parts = patches ? [patchCanvas('left'), patchCanvas('right')] : [];
  parts.push(fixation(dot));
  return `<div class="stage">${parts.join('')}</div>`;
}

// Static example of a pair for the instructions: two images (data URLs) side by side,
// drawn at the trial size and spacing, with the fixation between them
export function examplePair(urls, label) {
  const s = len(LAYOUT.patchSize);
  const gap = len(2 * LAYOUT.positions.right[0] - LAYOUT.patchSize);
  return `<figure class="example-pair"><div class="example-patches" style="gap: ${gap};">`
    + `<img src="${urls[0]}" style="width: ${s}; height: ${s};" alt="">`
    + `<img src="${urls[1]}" style="width: ${s}; height: ${s};" alt="">`
    + `</div><figcaption>${label}</figcaption></figure>`;
}
