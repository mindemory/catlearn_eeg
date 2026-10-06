// Coins and bonus: every payout and the final bonus come from these functions, so what
// participants see, what is saved and what is paid always agree.

import { REWARD } from './config.js';

// Coins for one trial
export function coinsFor({ correct, timeout }) {
  if (timeout) return REWARD.late;
  return correct ? REWARD.correct : REWARD.wrong;
}

export function countsForBonus(phase) {
  return REWARD.bonusPhases.includes(phase);
}

// Bonus in dollars (rounded to cents) for a number of bonus coins
export function bonusDollars(coins) {
  let usd = coins / REWARD.coinsPerDollar;
  usd = Math.max(REWARD.minBonus, usd);
  if (REWARD.maxBonus !== null) usd = Math.min(REWARD.maxBonus, usd);
  return Math.round(usd * 100) / 100;
}

export const formatCoins = (n) => `${n < 0 ? '−' : ''}${Math.abs(n).toLocaleString('en-US')}`;
export const formatDollars = (usd) => `$${usd.toFixed(2)}`;
export const signed = (n) => (n >= 0 ? `+${n}` : `−${Math.abs(n)}`);
