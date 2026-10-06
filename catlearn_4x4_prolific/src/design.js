// Builds one participant's full design from a seed: test type and version order, the rule
// of every block, its fractals and the trial order, and which key means YES. Pure functions,
// no jsPsych, so the same seed always gives the same design (and it can be checked in a
// console).

import { BLOCKS, KEYS, RULES, STIMULI, STUDY, TEST_TYPES } from './config.js';
import { FRACTAL_GROUPS } from '../stimuli/fractal_groups/groups.js';

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

/**
 * The participant's design.
 * @param {number} seed
 * @param {{debug?: boolean, type?: string, first?: 'A'|'B'}} opts  debug: 1 pass per block
 *   (3 for practice, so the criterion can still be reached); type / first: fix the test type
 *   and which version comes first (otherwise random, from STUDY.types[STUDY.phase])
 * @returns {{seed, keys: {yes, no}, testType, testFirst: 'A'|'B', blocks: Array, example}}
 *   each block: {block, phase, blockInPhase, nInPhase, rule, ruleType, size, labels, reps,
 *   criterion, fractals: {A: [...size], B: [...size]}, trials: [...]}; example: two
 *   fractals not shown in any block (for the instructions)
 */
export function buildDesign(seed, opts = {}) {
  const rng = makeRng(seed);
  // drawn even when fixed by opts, so the rest of the design is the same for a given seed
  const drawnType = pick(STUDY.types[STUDY.phase], rng);
  const drawnFirst = rng() < 0.5 ? 'A' : 'B';
  const testType = opts.type ?? drawnType;
  const testFirst = opts.first ?? drawnFirst;
  if (!TEST_TYPES[testType]) throw new Error(`unknown test type ${testType}`);
  if (!['A', 'B'].includes(testFirst)) throw new Error(`first must be A or B, not ${testFirst}`);
  const [versionA, versionB] = TEST_TYPES[testType];
  const testRules = testFirst === 'A' ? { first: versionA, second: versionB } : { first: versionB, second: versionA };

  const resolveRule = (rule) => {
    if (Array.isArray(rule)) return pick(rule, rng);
    if (rule.startsWith('test:')) return testRules[rule.slice(5)];
    return rule;
  };
  const rules = BLOCKS.map((spec) => resolveRule(spec.rule));
  rules.forEach((r) => { if (!RULES[r]) throw new Error(`unknown rule ${r}`); });

  // Each side of a block gets a whole group of equally distinct fractals (FRACTAL_GROUPS,
  // by group size), drawn without replacement, so no fractal or group repeats
  const groups = {};
  const nNeeded = {};
  rules.forEach((r) => { nNeeded[RULES[r].size] = (nNeeded[RULES[r].size] || 0) + 2; });
  Object.entries(nNeeded).forEach(([size, n]) => {
    const available = FRACTAL_GROUPS[size] || [];
    if (n > available.length) throw new Error(`design needs ${n} groups of ${size} fractals, there are ${available.length}`);
    groups[size] = shuffle(available, rng);
  });
  const phaseCount = {};
  BLOCKS.forEach((b) => { phaseCount[b.phase] = (phaseCount[b.phase] || 0) + 1; });
  const phaseSeen = {};

  const blocks = BLOCKS.map((spec, i) => {
    const rule = rules[i];
    const { size, type, labels } = RULES[rule];
    // fractals[side][level]: the fractal file showing that level of A (left) or B (right);
    // levels are assigned to the group's fractals at random
    const fractals = { A: shuffle(groups[size].pop(), rng), B: shuffle(groups[size].pop(), rng) };
    const reps = opts.debug ? (spec.criterion ? 3 : 1) : spec.reps;
    const trials = [];
    for (let rep = 0; rep < reps; rep++) {
      for (const compound of shuffle([...Array(size * size).keys()], rng)) {
        const a = Math.floor(compound / size);
        const b = compound % size;
        trials.push({ rep: rep + 1, compound, level_a: a, level_b: b,
                      fractal_a: fractals.A[a], fractal_b: fractals.B[b], category: labels[compound] });
      }
    }
    trials.forEach((t, k) => { t.trial_in_block = k + 1; });
    phaseSeen[spec.phase] = (phaseSeen[spec.phase] || 0) + 1;
    return {
      block: i + 1, phase: spec.phase, blockInPhase: phaseSeen[spec.phase], nInPhase: phaseCount[spec.phase],
      rule, ruleType: type, size, labels, reps, criterion: Boolean(spec.criterion), fractals, trials,
    };
  });

  const yes = pick(KEYS.pair, rng);
  const keys = { yes, no: KEYS.pair.find((k) => k !== yes) };
  // Two fractals this participant never sees in a block, for the instruction examples
  const used = new Set(blocks.flatMap((b) => Object.values(b.fractals).flat()));
  const example = shuffle(Object.values(FRACTAL_GROUPS).flat(), rng)
    .find((g) => g.every((f) => !used.has(f))).slice(0, 2);
  return { seed, keys, testType, testFirst, debug: Boolean(opts.debug), blocks, example };
}

// Image files the design will show, examples included (for preloading)
export function designImages(design) {
  const files = new Set(design.example.map((f) => `${STIMULI.fractalDir}${f}.png`));
  design.blocks.forEach((b) => Object.values(b.fractals).flat().forEach((f) => files.add(`${STIMULI.fractalDir}${f}.png`)));
  return [...files];
}
