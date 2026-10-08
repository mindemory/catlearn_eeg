// The task: for every block an intro screen, the rounds and a summary screen. A round:
//   iti       fixation only, jittered per trial (TIMING.iti; the trial's iti_ms): every round
//             starts from the centre
//   response  the two symbols, no fixation (free viewing), until F / J or TIMING.response
//   feedback  green / red circles around the symbols (TIMING.feedback)
// Practice blocks with a criterion end as soon as it is met; after each practice block the
// screening checks (SCREENING) may end the study for this participant. Each test block's
// break screen shows its bonus (BONUS tiers). One data row per round carries
// everything needed for analysis (part = 'response'); the other rows are kept for timing
// checks.

import { scaleData } from './calibration.js';
import { FEEDBACK, GRADES, KEYS, practiceCriterion, SCREENING, TIMING } from './config.js';
import { attention } from './data.js';
import { screen } from './display.js';
import { blockBonus, countsForBonus, formatDollars, totalBonus } from './bonus.js';

function blockIntro(block, keys, isFirstOfPhase, isFirstBlock, debug) {
  const { window: WINDOW, minCorrect: MIN_CORRECT } = practiceCriterion(block.ruleType);
  const phaseName = block.phase === 'practice' ? 'Practice' : 'Test';
  let begin = '';
  if (isFirstOfPhase) {
    begin = `${phaseName} blocks begin.\n`;
    begin += countsForBonus(block.phase)
      ? 'From now on your answers count toward your bonus.\n\n'
      : 'Practice answers do not count toward your bonus.\n\n';
  }
  const learn = isFirstBlock ? 'Learn which pairs of symbols go with F and which go with J.'
    : 'This block has NEW symbols. Learn which of its pairs go with F and which go with J.';
  const goal = block.criterion
    ? `\nThis practice ends once ${MIN_CORRECT} of your last ${WINDOW} answers are correct.`
    : `\n${block.trials.length} rounds.`;
  const text = `${begin}${phaseName} block ${block.blockInPhase} of ${block.nInPhase}\n\n${learn}${goal}\n\n`
    + `Press ${keys.cat1.toUpperCase()} or ${keys.cat0.toUpperCase()} for each pair.\n\n`
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
 * @param {object} keys    {cat1, cat0}: the keys for category 1 and category 0
 * @param {object} state   shared across blocks: test accuracy, bonuses, practice timeouts
 * @param {object} opts    {debug, screening: run the screening checks, onScreenOut(reason)}
 */
function blockTimeline(block, keys, state, isFirstOfPhase, isFirstBlock, opts) {
  const { debug } = opts;
  const { window: WINDOW, minCorrect: MIN_CORRECT } = practiceCriterion(block.ruleType);
  const local = { correct: 0, late: 0, n: 0, recent: [], done: false, criterionMet: false };
  const blockInfo = {
    block: block.block, phase: block.phase, block_in_phase: block.blockInPhase, rule: block.rule,
    rule_type: block.ruleType, size: block.size, counts_for_bonus: countsForBonus(block.phase),
    ...(block.criterion ? { criterion_window: WINDOW, criterion_min_correct: MIN_CORRECT } : {}),
  };
  let last = null;   // the latest response row, for its feedback screen

  const round = (t) => ({
    // practice with a criterion: skip the remaining rounds once it is met
    conditional_function: () => !local.done,
    timeline: [
      {
        type: jsPsychHtmlKeyboardResponse,
        save_trial_parameters: { stimulus: false },
        stimulus: () => screen({ fixation: true }),
        choices: 'NO_KEYS',
        trial_duration: t.iti_ms,
        data: { part: 'iti', ...blockInfo, trial_in_block: t.trial_in_block, iti_ms: t.iti_ms },
      },
      {
        type: jsPsychHtmlKeyboardResponse,
        save_trial_parameters: { stimulus: false },
        stimulus: () => screen({ trial: t }),
        choices: [keys.cat1, keys.cat0],                   // either case (jsPsych default)
        trial_duration: TIMING.response,
        response_ends_trial: true,
        data: { part: 'response', ...blockInfo, ...t, correct_key: t.category === 1 ? keys.cat1 : keys.cat0 },
        on_finish: (data) => {
          const key = data.response === null ? null : String(data.response).toLowerCase();   // null on a timeout
          data.response = key;
          data.timeout = key === null;
          data.choice = key === null ? null : (key === keys.cat1 ? 1 : 0);   // 1 = answered category 1 (F)
          data.correct = data.choice === t.category;
          local.n += 1;
          local.correct += data.correct ? 1 : 0;
          local.late += data.timeout ? 1 : 0;
          local.recent = [...local.recent, data.correct].slice(-WINDOW);
          data.recent_correct = local.recent.filter(Boolean).length;   // of the last WINDOW rounds
          data.blur_count = attention.blur;                // tab/window switches so far
          data.fullscreen_exit_count = attention.fullscreenExit;
          Object.assign(data, scaleData());                // px per degree, layout scale, window size
          if (block.criterion && local.recent.length === WINDOW && data.recent_correct >= MIN_CORRECT) {
            local.done = true;
            local.criterionMet = true;
          }
          last = data;
        },
      },
      {
        type: jsPsychHtmlKeyboardResponse,
        save_trial_parameters: { stimulus: false },
        stimulus: () => {
          const color = last.timeout ? FEEDBACK.late : last.correct ? FEEDBACK.correct : FEEDBACK.wrong;
          return screen({ trial: t, outcome: color });
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
    const p = local.correct / Math.max(1, local.n);
    let bonus = {};
    if (countsForBonus(block.phase)) {
      state.testCorrect += local.correct;
      state.testTrials += local.n;
      state.totalPCorrect = state.testCorrect / state.testTrials;
      state.blockBonuses.push(blockBonus(p));
      bonus = { block_bonus_usd: Math.round(blockBonus(p) * 100) / 100, bonus_so_far_usd: totalBonus(state.blockBonuses) };
    }
    if (block.phase === 'practice') {
      state.practiceRounds += local.n;
      state.practiceLate += local.late;
    }
    const [, grade, comment] = GRADES.find(([bound]) => p < bound);
    local.summary = { block_pcorrect: p, block_grade: grade, block_n_late: local.late, block_rounds: local.n,
                      test_pcorrect_so_far: state.totalPCorrect, comment, ...bonus,
                      ...(block.criterion ? { practice_criterion_met: local.criterionMet } : {}) };
    return local.summary;
  }

  const summary = {
    type: jsPsychHtmlKeyboardResponse,
    save_trial_parameters: { stimulus: false },
    stimulus: () => {
      const s = summarize();
      let text = '';
      if (block.criterion) {
        text += s.practice_criterion_met
          ? `Well done: ${MIN_CORRECT} of your last ${WINDOW} answers were correct.\n\n`
          : 'This practice block is over.\n\n';
      }
      text += `Proportion correct in this block = ${s.block_pcorrect.toFixed(2)}`;
      if (s.block_bonus_usd !== undefined) {
        text += `\nBonus for this block: ${formatDollars(s.block_bonus_usd)} (so far: ${formatDollars(s.bonus_so_far_usd)})`;
      }
      if (!block.criterion) text += `\n\nYour score in this block: ${s.block_grade}\n\n${s.comment}`;
      text += `\n\nTake a short break if you like (up to ${TIMING.breakMax / 1000} seconds).\n`
        + 'Please sit at about the same distance from the screen as at the start.\n'
        + `Press SPACE to continue, or wait: the study moves on in <span id="break-countdown">${TIMING.breakMax / 1000}</span> s.`;
      return `<div class="page preline center">${text}</div>`;
    },
    choices: [KEYS.start],
    // the break is capped: after TIMING.breakMax the next screen (the next block's intro,
    // which waits for SPACE) comes up on its own
    trial_duration: TIMING.breakMax,
    on_load: () => {
      const end = performance.now() + TIMING.breakMax;
      local.countdown = setInterval(() => {
        const el = document.getElementById('break-countdown');
        if (el) el.textContent = String(Math.max(0, Math.ceil((end - performance.now()) / 1000)));
      }, 250);
    },
    data: { part: 'block_summary', ...blockInfo },
    on_finish: (data) => {
      clearInterval(local.countdown);
      const { comment, ...s } = summarize();
      Object.assign(data, s);
      data.break_timed_out = data.response === null;
    },
  };

  // Screening after a practice block: the type-I practice must be passed, and practice
  // timeouts must stay under SCREENING.practiceTimeouts (all practice rounds so far)
  const screenCheck = {
    type: jsPsychCallFunction,
    func: () => {
      if (!opts.screening || block.phase !== 'practice') return;
      if (SCREENING.practiceTypeI && block.ruleType === 'I' && !local.criterionMet) {
        opts.onScreenOut('practice_type_I');
      } else if (SCREENING.practiceTimeouts !== null && state.practiceRounds > 0
                 && state.practiceLate / state.practiceRounds > SCREENING.practiceTimeouts) {
        opts.onScreenOut('practice_timeouts');
      }
    },
    data: { part: 'screen_check', block: block.block },
  };

  return { timeline: [blockIntro(block, keys, isFirstOfPhase, isFirstBlock, debug), ...block.trials.map(round), summary,
                      ...(block.phase === 'practice' ? [screenCheck] : [])] };
}

/**
 * @param {object} design
 * @param {{screening: boolean, onScreenOut: function(string)}} opts
 */
export function taskTimeline(design, opts) {
  const state = { testCorrect: 0, testTrials: 0, totalPCorrect: null, blockBonuses: [], practiceRounds: 0,
                  practiceLate: 0 };
  const seenPhase = new Set();
  const blocks = design.blocks.map((b, i) => {
    const first = !seenPhase.has(b.phase);
    seenPhase.add(b.phase);
    return blockTimeline(b, design.keys, state, first, i === 0, { ...opts, debug: design.debug });
  });
  return { timeline: blocks, state };
}
