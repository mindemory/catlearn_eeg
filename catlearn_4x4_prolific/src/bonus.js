// The performance bonus: what participants are told, what is saved and what is paid all
// come from these functions (and tools/bonus_payments.py repeats the same rule).

import { BONUS } from './config.js';

export function countsForBonus(phase) {
  return BONUS.phases.includes(phase);
}

// Bonus for one block from its proportion correct: the amount of the highest tier reached
// (unrounded, so the total of three top blocks is exactly the maximum)
export function blockBonus(pCorrect) {
  let usd = 0;
  BONUS.tiers.forEach(([min, amount]) => { if (pCorrect >= min) usd = amount; });
  return usd;
}

// Total bonus in dollars, rounded to cents
export function totalBonus(blockAmounts) {
  return Math.round(blockAmounts.reduce((s, x) => s + x, 0) * 100) / 100;
}

export const maxBonus = (nBlocks) => totalBonus(Array(nBlocks).fill(BONUS.tiers[BONUS.tiers.length - 1][1]));
export const formatDollars = (usd) => `$${usd.toFixed(2)}`;
