// HTML for every screen of a trial. All positions and sizes are in degrees of visual angle
// (config.js LAYOUT) and become CSS lengths through the --deg variable (pixels per degree,
// set by calibration.js), so the layout has the same visual size on every screen.

import { LAYOUT, STIMULI } from './config.js';

// Length in CSS: degrees -> calc(var(--deg) * v), or -> px when drawing into a fixed-size box
let unitPx = null;
const len = (v) => (unitPx ? `${(v * unitPx).toFixed(2)}px` : `calc(var(--deg) * ${v.toFixed(4)})`);

function at(x, y) {
  return `left: calc(50% + ${len(x)}); top: calc(50% - ${len(y)});`;
}

function fractalImg(file, loc) {
  const [x, y] = LAYOUT.positions[loc];
  const s = len(LAYOUT.fractalSize * LAYOUT.fractalImageScale);
  return `<img class="fractal" src="${STIMULI.fractalDir}${file}.png" style="${at(x, y)} width: ${s}; height: ${s};" alt="">`;
}

// Feedback: a circle in `color` around a symbol
function outcomeRing(loc, color) {
  const [x, y] = LAYOUT.positions[loc];
  const { size, width } = LAYOUT.outcomeRing;
  return `<div class="ring" style="${at(x, y)} width: ${len(size)}; height: ${len(size)}; border-width: ${len(width)};`
    + ` border-color: ${color};"></div>`;
}

// Bull's eye fixation (Thaler et al., 2013), as drawn by the MATLAB task: black disc,
// white cross spanning it, black centre dot
function fixation() {
  const { outer, inner, line } = LAYOUT.fixation;
  const r = 50;                                  // SVG units: disc radius
  const w = (line / outer) * 2 * r;
  const ri = (inner / outer) * r;
  return `<svg class="fixation" viewBox="-50 -50 100 100" style="${at(0, 0)} width: ${len(outer)}; height: ${len(outer)};">`
    + `<circle r="${r}" fill="#000"/>`
    + `<rect x="${-r}" y="${-w / 2}" width="${2 * r}" height="${w}" fill="#fff"/>`
    + `<rect x="${-w / 2}" y="${-r}" width="${w}" height="${2 * r}" fill="#fff"/>`
    + `<circle r="${ri}" fill="#000"/></svg>`;
}

/**
 * One screen: the fixation (between rounds only), or the pair and optionally its feedback.
 * Participants look freely while the symbols are on, so those screens have no fixation.
 * @param {object} o
 * @param {boolean} [o.fixation]  the bull's eye in the centre (the gap between rounds)
 * @param {object}  [o.trial]     a design trial: shows its two fractals
 * @param {string}  [o.outcome]   feedback: colour of the circles around the symbols
 */
export function screen(o = {}) {
  const parts = [];
  if (o.trial) parts.push(fractalImg(o.trial.fractal_a, 'A'), fractalImg(o.trial.fractal_b, 'B'));
  if (o.outcome) parts.push(outcomeRing('A', o.outcome), outcomeRing('B', o.outcome));
  if (o.fixation) parts.push(fixation());
  return `<div class="stage">${parts.join('')}</div>`;
}

// A static example screen for the instructions, drawn into a box `height` px tall (scaled
// so the whole layout, LAYOUT.extent, fits it). files = [left, right] fractals; symbols =
// false draws the fixation only (the gap between rounds).
export function exampleScreen(files, { height = 160, outcome = null, symbols = true } = {}) {
  const t = symbols ? { fractal_a: files[0], fractal_b: files[1] } : null;
  unitPx = height / LAYOUT.extent.height;
  const html = screen({ trial: t, outcome, fixation: !symbols });
  unitPx = null;
  const width = Math.round(height * (LAYOUT.extent.width / LAYOUT.extent.height));
  return `<div class="example" style="width: ${width}px; height: ${height}px;">${html}</div>`;
}
