// The task: for every block an intro screen, the rounds and a summary screen. A round:
//   wait      machine + fixation; SPACE pulls the lever, or the round starts by itself
//             after TIMING.wait
//   pull      the lever swings down and back (TIMING.pull)
//   response  the symbols (2 in practice, 3 in the test), until F / J or TIMING.response
//   feedback  green / red circles around the symbols (TIMING.feedback)
// A coin bar at the top of every screen tracks the coins. Blocks with a criterion end as soon
// as it is met; in a paid block with creditRemaining, the rounds left are then paid as correct. One data row per round carries everything needed for analysis
// (part = 'response'); the other rows are kept for timing checks.

import { scaleData } from './calibration.js';
import { FEEDBACK, GRADES, KEYS, REWARD, SPACE_HINT_PHASES, TIMING } from './config.js';
import { attention } from './data.js';
import { coinBar, screen } from './display.js';
import { coinsFor, countsForBonus, formatCoins, signed } from './reward.js';

const OUTCOME = ['lose', 'win'];
const RECENT = 10;   // recent_correct window in blocks without a criterion

// Coin bar for a phase: paid coins fill toward the most the participant could earn (every
// paid round correct); practice coins are grey and labelled "practice". `from` (coins before
// this round) makes the feedback screen animate the change.
function bar(phase, state, from = undefined) {
  const paid = countsForBonus(phase);
  const coins = paid ? state.bonusCoins : state.practiceCoins;
  return coinBar({ coins, from, paid, max: paid ? state.maxPaidCoins : state.maxPracticeCoins,
                   label: paid ? `${formatCoins(coins)} coins` : `practice: ${formatCoins(coins)}` });
}

function blockIntro(block, keys, isFirstOfPhase, isFirstBlock, debug) {
  const phaseName = block.phase === 'practice' ? 'Practice' : 'Test';
  let begin = '';
  if (isFirstOfPhase) {
    begin = `${phaseName} blocks begin.\n`;
    begin += countsForBonus(block.phase)
      ? `From now on your coins count toward your bonus (${formatCoins(REWARD.coinsPerDollar)} coins = $1).\n\n`
      : 'Practice coins are not paid.\n\n';
  }
  const n = block.features.length === 2 ? 'two' : 'three';
  const learn = (isFirstBlock ? `This machine shows ${n} symbols at a time. `
    : `This is a NEW machine with new symbols. It shows ${n} symbols at a time. `)
    + 'Learn which combinations of symbols win and which lose.';
  const c = block.criterion;
  let goal = `\n${block.trials.length} rounds.`;
  if (c && block.phase === 'practice') goal = `\nThis practice ends once ${c.minCorrect} of your last ${c.window} answers are correct.`;
  else if (c) {
    goal = `\nUp to ${block.trials.length} rounds. The machine stops early once ${c.minCorrect} of your last `
      + `${c.window} answers are correct`
      + (c.creditRemaining && countsForBonus(block.phase) ? ', and the rounds left are then paid as if you got them all right.' : '.');
  }
  const text = `${begin}${phaseName} block ${block.blockInPhase} of ${block.nInPhase}\n\n${learn}${goal}\n\n`
    + 'Press SPACE to pull the lever for each round.\n'
    + `Will the symbols win? Press ${keys.yes.toUpperCase()} for YES and ${keys.no.toUpperCase()} for NO.\n\n`
    + 'Press SPACE to start!';
  return {
    type: jsPsychHtmlKeyboardResponse,
    save_trial_parameters: { stimulus: false },   // screen HTML is rebuilt from the design
    // A debug run (?debug=1) has shortened blocks: say so, so it is never mistaken for a real run
    stimulus: (debug ? `<div class="debug-banner">DEBUG RUN: shortened blocks (remove ?debug=1 for the real study)</div>` : '')
      + `<div class="page preline center">${text}</div>`,
    choices: [KEYS.start],
    data: { part: 'block_intro', block: block.block },
  };
}

/**
 * Timeline of one block.
 * @param {object} block   a block of buildDesign()
 * @param {object} keys    {yes, no}: this participant's response keys
 * @param {object} state   shared across blocks: coin banks and test accuracy
 */
function blockTimeline(block, keys, state, isFirstOfPhase, isFirstBlock, debug) {
  const local = { correct: 0, late: 0, n: 0, coins: 0, recent: [], done: false, criterionMet: false, wait: null,
                  creditRounds: 0, creditCoins: 0 };
  const paid = countsForBonus(block.phase);
  const crit = block.criterion;
  const recentWindow = crit?.window ?? RECENT;
  const hint = SPACE_HINT_PHASES.includes(block.phase) ? 'press SPACE to play' : null;
  const blockInfo = {
    block: block.block, phase: block.phase, block_in_phase: block.blockInPhase, rule: block.rule,
    rule_type: block.ruleType, levels: block.levels.join('x'), coins_paid: paid,
  };
  let last = null;   // the latest response row, for its feedback screen

  const round = (t) => ({
    // blocks with a criterion: skip the remaining rounds once it is met
    conditional_function: () => !local.done,
    timeline: [
      {
        type: jsPsychHtmlKeyboardResponse,
        save_trial_parameters: { stimulus: false },
        stimulus: () => screen({ hint, bar: bar(block.phase, state) }),
        choices: [KEYS.start],
        trial_duration: TIMING.wait,
        response_ends_trial: true,
        data: { part: 'wait', ...blockInfo, trial_in_block: t.trial_in_block },
        on_finish: (data) => {
          data.started_by = data.response === null ? 'timeout' : 'space';
          local.wait = { started_by: data.started_by, wait_ms: data.rt ?? TIMING.wait };
        },
      },
      {
        type: jsPsychHtmlKeyboardResponse,
        save_trial_parameters: { stimulus: false },
        stimulus: () => screen({ pulling: true, bar: bar(block.phase, state) }),
        choices: 'NO_KEYS',
        trial_duration: TIMING.pull,
        data: { part: 'pull', ...blockInfo, trial_in_block: t.trial_in_block },
      },
      {
        type: jsPsychHtmlKeyboardResponse,
        save_trial_parameters: { stimulus: false },
        stimulus: () => screen({ trial: t, bar: bar(block.phase, state) }),
        choices: [keys.yes, keys.no],                      // either case (jsPsych default)
        trial_duration: TIMING.response,
        response_ends_trial: true,
        data: { part: 'response', ...blockInfo, ...t, outcome: OUTCOME[t.category],
                correct_key: t.category === 1 ? keys.yes : keys.no },
        on_finish: (data) => {
          const key = data.response === null ? null : String(data.response).toLowerCase();   // null on a timeout
          data.response = key;
          data.timeout = key === null;
          data.choice = key === null ? null : (key === keys.yes ? 1 : 0);   // 1 = answered YES (predicted win)
          data.correct = data.choice === t.category;
          data.coins_delta = coinsFor(data);
          data.round_started_by = local.wait.started_by;   // 'space' or 'timeout'
          data.wait_ms = local.wait.wait_ms;
          local.n += 1;
          local.correct += data.correct ? 1 : 0;
          local.late += data.timeout ? 1 : 0;
          local.coins += data.coins_delta;
          local.recent = [...local.recent, data.correct].slice(-recentWindow);
          if (paid) state.bonusCoins += data.coins_delta;
          else state.practiceCoins += data.coins_delta;
          data.coins_block = local.coins;
          data.coins_bonus_total = state.bonusCoins;        // paid coins so far
          data.coins_practice_total = state.practiceCoins;
          data.recent_correct = local.recent.filter(Boolean).length;   // of the last recentWindow rounds
          data.blur_count = attention.blur;                // tab/window switches so far
          data.fullscreen_exit_count = attention.fullscreenExit;
          Object.assign(data, scaleData());                // px per degree, layout scale, window size
          if (crit && local.recent.length === recentWindow && data.recent_correct >= crit.minCorrect) {
            local.done = true;
            local.criterionMet = true;
            // paid block: the rounds left count as correct (credited on the summary screen)
            if (paid && crit.creditRemaining) {
              local.creditRounds = block.trials.length - local.n;
              local.creditCoins = local.creditRounds * REWARD.correct;
            }
          }
          last = data;
        },
      },
      {
        type: jsPsychHtmlKeyboardResponse,
        save_trial_parameters: { stimulus: false },
        stimulus: () => {
          const color = last.timeout ? FEEDBACK.late : last.correct ? FEEDBACK.correct : FEEDBACK.wrong;
          const before = (paid ? state.bonusCoins : state.practiceCoins) - last.coins_delta;
          return screen({ trial: t, outcome: color, bar: bar(block.phase, state, before) });
        },
        choices: 'NO_KEYS',
        trial_duration: TIMING.feedback,
        data: { part: 'feedback', ...blockInfo, trial_in_block: t.trial_in_block },
      },
    ],
  });

  // Block results, computed once when the summary screen is built (jsPsych evaluates the
  // stimulus before on_start, so this cannot live in on_start)
  function summarize() {
    if (local.summary) return local.summary;
    if (local.creditCoins) {
      state.bonusCoins += local.creditCoins;
      local.coins += local.creditCoins;
    }
    const p = local.correct / Math.max(1, local.n);
    if (block.phase === 'test') {
      state.testCorrect += local.correct;
      state.testTrials += local.n;
      state.totalPCorrect = state.testCorrect / state.testTrials;
    }
    const [, grade, comment] = GRADES.find(([bound]) => p < bound);
    local.summary = { block_pcorrect: p, block_grade: grade, block_n_late: local.late, block_coins: local.coins,
                      block_rounds: local.n, test_pcorrect_so_far: state.totalPCorrect,
                      coins_bonus_total: state.bonusCoins, comment,
                      ...(crit ? { criterion_met: local.criterionMet } : {}),
                      ...(crit && paid ? { block_credit_rounds: local.creditRounds, block_credit_coins: local.creditCoins } : {}) };
    return local.summary;
  }

  const summary = {
    type: jsPsychHtmlKeyboardResponse,
    save_trial_parameters: { stimulus: false },
    stimulus: () => {
      const s = summarize();
      let text = '';
      if (crit) {
        text += s.criterion_met
          ? `Well done: ${crit.minCorrect} of your last ${crit.window} answers were correct.\n\n`
          : `This ${block.phase === 'practice' ? 'practice block' : 'block'} is over.\n\n`;
        if (s.block_credit_rounds) {
          text += `You mastered this machine: the ${s.block_credit_rounds} rounds left are paid in full `
            + `(${signed(s.block_credit_coins)} coins).\n\n`;
        }
      }
      text += `Proportion correct in this block = ${s.block_pcorrect.toFixed(2)}\n`
        + `Coins in this block: ${signed(s.block_coins)}${paid ? '' : ' (practice, not paid)'}`;
      if (paid) text += `\nYour coins so far: ${formatCoins(s.coins_bonus_total)}`;
      if (block.phase !== 'practice') text += `\n\nYour score in this block: ${s.block_grade}\n\n${s.comment}`;
      text += '\n\nPress SPACE to continue!';
      return `<div class="page preline center">${text}</div>`;
    },
    choices: [KEYS.start],
    data: { part: 'block_summary', ...blockInfo },
    on_finish: (data) => {
      const { comment, ...s } = summarize();
      Object.assign(data, s);
    },
  };

  return { timeline: [blockIntro(block, keys, isFirstOfPhase, isFirstBlock, debug), ...block.trials.map(round), summary] };
}

export function taskTimeline(design) {
  const maxCoins = (paid) => design.blocks.filter((b) => countsForBonus(b.phase) === paid)
    .reduce((n, b) => n + b.trials.length * REWARD.correct, 0);
  const state = { testCorrect: 0, testTrials: 0, totalPCorrect: null, bonusCoins: 0, practiceCoins: 0,
                  maxPaidCoins: maxCoins(true), maxPracticeCoins: maxCoins(false) };
  const seenPhase = new Set();
  const blocks = design.blocks.map((b, i) => {
    const first = !seenPhase.has(b.phase);
    seenPhase.add(b.phase);
    return blockTimeline(b, design.keys, state, first, i === 0, design.debug);
  });
  return { timeline: blocks, state };
}
