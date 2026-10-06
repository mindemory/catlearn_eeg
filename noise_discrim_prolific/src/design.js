// One participant's design from a seed: which key means SAME, and every trial (alpha, same or
// different, and the seeds of its two patches). Pure functions, no jsPsych, so the same seed
// always gives the same design.

import { ALPHAS, BLOCKS, KEYS } from './config.js';
import { makeRng } from './noise.js';

export function shuffle(arr, rng) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function newSeed() {
  return Math.floor(Math.random() * 2 ** 31);
}

const seed32 = (rng) => Math.floor(rng() * 2 ** 32) >>> 0;

/**
 * @param {number} seed
 * @param {{debug?: boolean}} opts  debug: 2 blocks of 20 trials (1 same + 1 different per alpha)
 * @returns {{seed, keys: {same, different}, debug, blocks: Array<Array<trial>>, example}}
 *   trial: {block, trial_in_block, alpha, alpha_level (1-10), pair ('same' | 'different'),
 *   seed_left, seed_right}; example: patch seeds for the instruction examples
 */
export function buildDesign(seed, opts = {}) {
  const rng = makeRng(seed);
  const same = KEYS.pair[Math.floor(rng() * 2)];
  const keys = { same, different: KEYS.pair.find((k) => k !== same) };
  const nBlocks = opts.debug ? 2 : BLOCKS.n;
  const perLevel = opts.debug ? 1 : BLOCKS.perLevel;

  const blocks = [];
  for (let b = 0; b < nBlocks; b++) {
    const cells = [];
    ALPHAS.forEach((alpha, level) => {
      for (let r = 0; r < perLevel; r++) {
        cells.push({ alpha, alpha_level: level + 1, pair: 'same' });
        cells.push({ alpha, alpha_level: level + 1, pair: 'different' });
      }
    });
    blocks.push(shuffle(cells, rng).map((t, k) => {
      const left = seed32(rng);
      const right = t.pair === 'same' ? left : seed32(rng);
      return { block: b + 1, trial_in_block: k + 1, ...t, seed_left: left, seed_right: right };
    }));
  }
  // Instruction examples: a same and a different pair at a clearly visible alpha
  const a = seed32(rng);
  const example = { alpha: ALPHAS[6], same: [a, a], different: [seed32(rng), seed32(rng)] };
  return { seed, keys, debug: Boolean(opts.debug), blocks, example };
}
