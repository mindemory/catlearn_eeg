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

// One symbol: file number `file` of the screen's set (t.stim_set, t.stim_dir) at position loc
function symbolImg(t, file, loc) {
  const [x, y] = LAYOUT.positions[loc];
  const s = len(LAYOUT.fractalSize * STIMULI[t.stim_set].imageScale);
  return `<img class="fractal" src="${t.stim_dir}${file}.png" style="${at(x, y)} width: ${s}; height: ${s};" alt="">`;
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

// Slot-machine frame: a gold border around the symbols (inside stays grey, so the symbols
// look the same as without it), a title plate on top and a lever on the right. With
// pulling = true the lever swings down and back up (the round starting).
function machine(pulling) {
  const { center: [cx, cy], size: [w, h], border, color } = LAYOUT.machine;
  const { length, knob } = LAYOUT.lever;
  const pivot = [cx + w / 2 + 0.9, cy];
  return `<div class="machine" style="${at(cx, cy)} width: ${len(w)}; height: ${len(h)};`
    + ` border: ${len(border)} solid ${color}; border-radius: ${len(1)};`
    + ` box-shadow: 0 0 0 ${len(0.12)} #6b4e00 inset, 0 0 0 ${len(0.12)} #6b4e00;"></div>`
    + `<div class="machine-plate" style="${at(cx, cy + h / 2)} background: ${color};`
    + ` font-size: ${len(0.55)}; padding: ${len(0.12)} ${len(0.7)}; border-radius: ${len(0.3)};">SLOTS</div>`
    + `<div class="lever${pulling ? ' pulling' : ''}" style="${at(...pivot)} width: ${len(knob)}; height: ${len(length + knob / 2)};">`
    + `<div class="lever-arm" style="width: ${len(0.3)}; background: ${color}; border-radius: ${len(0.15)};"></div>`
    + `<div class="lever-knob" style="width: ${len(knob)}; height: ${len(knob)};"></div></div>`
    + `<div class="lever-hub" style="${at(...pivot)} width: ${len(0.7)}; height: ${len(0.7)}; background: #6b4e00;"></div>`;
}

/**
 * Coin bar: fill = coins / max (clamped to 0-1), animated from `from` coins on the
 * feedback screen. Sized in window units (it is not a stimulus).
 * @param {{coins, from?, max, label, paid}} b
 */
export function coinBar(b) {
  const pct = (c) => `${(100 * Math.min(1, Math.max(0, c / b.max))).toFixed(2)}%`;
  const from = pct(b.from ?? b.coins);
  return `<div class="coinbar ${b.paid ? 'paid' : 'practice'}">`
    + `<div class="coinbar-track"><div class="coinbar-fill" style="--from: ${from}; --to: ${pct(b.coins)};"></div></div>`
    + `<div class="coinbar-label">${b.label}</div></div>`;
}

/**
 * One screen. The machine and the fixation are always shown.
 * @param {object} o
 * @param {object}  [o.trial]     a design trial: shows its symbols (symbol_a, _b, _c of stim_set)
 * @param {boolean} [o.pulling]   lever animation (the round starting)
 * @param {string}  [o.outcome]   feedback: colour of the circles around the symbols
 * @param {string}  [o.hint]      small text under the machine (e.g. "press SPACE")
 * @param {string}  [o.bar]       coin bar HTML (coinBar()), at the top of the window
 */
export function screen(o) {
  const parts = [machine(o.pulling)];
  // positions this trial shows a symbol at (practice: A and B; test: A, B and C)
  const shown = o.trial ? Object.keys(LAYOUT.positions).filter((f) => o.trial[`symbol_${f.toLowerCase()}`] != null) : [];
  shown.forEach((f) => parts.push(symbolImg(o.trial, o.trial[`symbol_${f.toLowerCase()}`], f)));
  if (o.outcome) shown.forEach((f) => parts.push(outcomeRing(f, o.outcome)));
  parts.push(fixation());
  if (o.hint) parts.push(`<div class="hint" style="${at(0, LAYOUT.hintY)} font-size: ${len(0.55)};">${o.hint}</div>`);
  if (o.bar) parts.push(o.bar);
  return `<div class="stage">${parts.join('')}</div>`;
}

// A static example machine for the instructions, drawn into a box `height` px tall
// (scaled so the whole layout, LAYOUT.extent, fits it). example = the symbols of one screen
// (design.example.practice or .test); symbols = false draws the empty machine (the waiting
// screen).
export function exampleMachine(example, { height = 220, outcome = null, symbols = true, hint = null } = {}) {
  unitPx = height / LAYOUT.extent.height;
  const html = screen({ trial: symbols ? example : null, outcome, hint });
  unitPx = null;
  const width = Math.round(height * (LAYOUT.extent.width / LAYOUT.extent.height));
  return `<div class="example" style="width: ${width}px; height: ${height}px;">${html}</div>`;
}
