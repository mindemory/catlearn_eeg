// Builds one participant's full design from a seed: the noise condition (alpha), the rule
// of every block, its symbols and the trial order, and which key means YES. Pure functions,
// no jsPsych, so the same seed always gives the same design (and it can be checked in a
// console).

import { BLOCKS, KEYS, LAYOUT, RULES, STIMULI } from './config.js';
import { FRACTAL_GROUPS } from '../stimuli/fractal_groups/groups.js';
import { NOISE_GROUPS } from '../stimuli/noise/groups.js';

// Small seeded PRNG (mulberry32): returns floats in [0, 1)
export function makeRng(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle(arr, rng) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const pick = (arr, rng) => arr[Math.floor(rng() * arr.length)];

export function newSeed() {
  return Math.floor(Math.random() * 2 ** 31);
}

// Positions in order (A, B, C); a block uses the first levels.length of them
export const FEATURES = Object.keys(LAYOUT.positions);

// Levels of every feature for a combination index (mixed radix, first feature slowest)
export function levelsOf(compound, levels) {
  const out = [];
  for (let i = levels.length - 1; i >= 0; i--) {
    out[i] = compound % levels[i];
    compound = Math.floor(compound / levels[i]);
  }
  return out;
}

// A symbol set's folder and pairs/groups (by size) for this participant's noise condition
function symbolSet(name, alpha) {
  if (name === 'fractals') return { dir: STIMULI.fractals.dir, groups: FRACTAL_GROUPS };
  if (name === 'noise') return { dir: `${STIMULI.noise.dir}a${alpha}/`, groups: NOISE_GROUPS[alpha] };
  throw new Error(`unknown symbol set ${name}`);
}

// The symbols of one screen: the set, its folder, and a file number per position (null = empty)
function symbolsOf(set, dir, files) {
  const t = { stim_set: set, stim_dir: dir };
  FEATURES.forEach((f, k) => { t[`symbol_${f.toLowerCase()}`] = files[k] ?? null; });
  return t;
}

/**
 * The participant's design.
 * @param {number} seed
 * @param {{debug?: boolean, alpha?: string}} opts  debug: 1 pass per block (3 for blocks with a
 *   criterion, so it can still be reached); alpha: fix the noise alpha (one of
 *   STIMULI.noise.alphas; otherwise random among them)
 * @returns {{seed, keys: {yes, no}, alpha, testRule, blocks: Array, example}}
 *   each block: {block, phase, blockInPhase, nInPhase, rule, ruleType, levels, features,
 *   labels, reps, criterion, stimSet, stimDir, symbols: {A: [...], B: [...], (C: [...])},
 *   trials: [...]}; example: {practice, test}, symbols for the instruction machines that no
 *   block shows
 */
export function buildDesign(seed, opts = {}) {
  const rng = makeRng(seed);
  // drawn even when fixed by opts, so the rest of the design is the same for a given seed
  const drawnAlpha = pick(STIMULI.noise.alphas, rng);
  const alpha = opts.alpha ?? drawnAlpha;
  if (!STIMULI.noise.alphas.includes(alpha) || !NOISE_GROUPS[alpha]) {
    throw new Error(`alpha ${alpha} is not used in this study (STIMULI.noise.alphas: ${STIMULI.noise.alphas.join(', ')})`);
  }

  const rules = BLOCKS.map((spec) => (Array.isArray(spec.rule) ? pick(spec.rule, rng) : spec.rule));
  rules.forEach((r) => {
    if (!RULES[r]) throw new Error(`unknown rule ${r}`);
    if (RULES[r].levels.length > FEATURES.length) throw new Error(`${r} has more features than LAYOUT.positions`);
  });

  // Each feature of a block gets a whole group of equally distinct symbols (by group size =
  // its number of levels) from the block's set, drawn without replacement, so nothing repeats
  const sets = Object.fromEntries([...new Set(BLOCKS.map((b) => b.stimuli))].map((name) => [name, symbolSet(name, alpha)]));
  const queues = {};
  BLOCKS.forEach((spec, i) => RULES[rules[i]].levels.forEach((k) => {
    const id = `${spec.stimuli}:${k}`;
    if (!queues[id]) queues[id] = { name: spec.stimuli, size: k, n: 0 };
    queues[id].n += 1;
  }));
  Object.values(queues).forEach((q) => {
    const available = sets[q.name].groups[q.size] || [];
    if (q.n > available.length) throw new Error(`design needs ${q.n} ${q.name} groups of ${q.size}, there are ${available.length}`);
    q.groups = shuffle(available, rng);
  });
  const phaseCount = {};
  BLOCKS.forEach((b) => { phaseCount[b.phase] = (phaseCount[b.phase] || 0) + 1; });
  const phaseSeen = {};

  const blocks = BLOCKS.map((spec, i) => {
    const rule = rules[i];
    const { levels, type, labels } = RULES[rule];
    const features = FEATURES.slice(0, levels.length);
    const { dir } = sets[spec.stimuli];
    // symbols[feature][level]: the file showing that level; levels are assigned to the
    // group's members at random
    const symbols = Object.fromEntries(features.map((f, k) => [f, shuffle(queues[`${spec.stimuli}:${levels[k]}`].groups.pop(), rng)]));
    const reps = opts.debug ? (spec.criterion ? 3 : 1) : spec.reps;
    const nCombos = levels.reduce((n, k) => n * k, 1);
    const trials = [];
    for (let rep = 0; rep < reps; rep++) {
      for (const compound of shuffle([...Array(nCombos).keys()], rng)) {
        const lv = levelsOf(compound, levels);
        const t = { rep: rep + 1, compound, category: labels[compound] };
        // one column per position, null where this block shows nothing
        FEATURES.forEach((f, k) => { t[`level_${f.toLowerCase()}`] = k < levels.length ? lv[k] : null; });
        Object.assign(t, symbolsOf(spec.stimuli, dir, features.map((f, k) => symbols[f][lv[k]])));
        trials.push(t);
      }
    }
    trials.forEach((t, k) => { t.trial_in_block = k + 1; });
    phaseSeen[spec.phase] = (phaseSeen[spec.phase] || 0) + 1;
    return {
      block: i + 1, phase: spec.phase, blockInPhase: phaseSeen[spec.phase], nInPhase: phaseCount[spec.phase],
      rule, ruleType: type, levels, features, labels, reps, criterion: spec.criterion ?? null,
      stimSet: spec.stimuli, stimDir: dir, symbols, trials,
    };
  });

  const yes = pick(KEYS.pair, rng);
  const keys = { yes, no: KEYS.pair.find((k) => k !== yes) };
  // Instruction machines: a practice one (2 symbols) and a test one (3), from groups no block uses
  const unused = (set, n) => {
    const used = new Set(blocks.filter((b) => b.stimSet === set).flatMap((b) => Object.values(b.symbols).flat()));
    return shuffle(Object.values(sets[set].groups).flat(2).filter((f) => !used.has(f)), rng).slice(0, n);
  };
  const practice = blocks.find((b) => b.phase === 'practice') ?? blocks[0];
  const test = blocks.find((b) => b.phase === 'test') ?? blocks[blocks.length - 1];
  const example = {
    practice: symbolsOf(practice.stimSet, practice.stimDir, unused(practice.stimSet, practice.features.length)),
    test: symbolsOf(test.stimSet, test.stimDir, unused(test.stimSet, test.features.length)),
  };
  return { seed, keys, alpha, testRule: test.rule, debug: Boolean(opts.debug), blocks, example };
}

// Image files of every symbol on a screen
const screenFiles = (t) => FEATURES.map((f) => t[`symbol_${f.toLowerCase()}`]).filter((x) => x != null)
  .map((x) => `${t.stim_dir}${x}.png`);

// Image files the design will show, examples included (for preloading)
export function designImages(design) {
  const files = new Set([...screenFiles(design.example.practice), ...screenFiles(design.example.test)]);
  design.blocks.forEach((b) => Object.values(b.symbols).flat().forEach((f) => files.add(`${b.stimDir}${f}.png`)));
  return [...files];
}
