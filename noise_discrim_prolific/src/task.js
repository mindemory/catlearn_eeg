// The task: blocks of trials with a break screen between them. A trial (plugin-noise-pair.js):
//   fixation   the bull's eye alone (TIMING.fixation), the patches already drawn but hidden
//   stimulus   both patches for TIMING.stimulus, switched on and off in display frames
//   response   the fixation alone until an answer or TIMING.response after the patches
//   feedback   the fixation's centre turns green (correct) or red (wrong or late)
// One data row per trial (part = 'response') carries everything needed for analysis.

import { scaleData } from './calibration.js';
import { FEEDBACK, KEYS, TIMING } from './config.js';
import { attention } from './data.js';
import { screen } from './display.js';
import { noiseField } from './noise.js';
import NoisePairPlugin from './plugin-noise-pair.js';

const key = (k) => `<span class="key">${k.toUpperCase()}</span>`;

function trialTimeline(t, keys, state) {
  let last = null;
  const stimulus = {
    type: NoisePairPlugin,
    html: screen({ patches: true }),
    // generated at trial start, before the fixation period
    fields: () => {
      const left = noiseField(t.alpha, t.seed_left);
      return { left, right: t.seed_right === t.seed_left ? left : noiseField(t.alpha, t.seed_right) };
    },
    choices: [keys.same, keys.different],                // either case (jsPsych default)
    fixation_ms: TIMING.fixation,
    stimulus_ms: TIMING.stimulus,
    response_ms: TIMING.response,
    save_trial_parameters: { html: false, fields: false },
    data: { part: 'response', ...t },
    on_finish: (data) => {
      const k = data.response === null ? null : String(data.response).toLowerCase();
      data.response = k;
      data.timeout = k === null;
      data.answer = k === null ? null : k === keys.same ? 'same' : 'different';
      data.correct = data.answer === t.pair;
      data.said_different = k === null ? null : data.answer === 'different';
      data.blur_count = attention.blur;
      data.fullscreen_exit_count = attention.fullscreenExit;
      Object.assign(data, scaleData());
      state.n += 1;
      state.correct += data.correct ? 1 : 0;
      state.blockCorrect += data.correct ? 1 : 0;
      state.blockN += 1;
      last = data;
    },
  };

  const feedback = {
    timeline: [{
      type: jsPsychHtmlKeyboardResponse,
      stimulus: () => screen({ dot: last.timeout ? FEEDBACK.late : last.correct ? FEEDBACK.correct : FEEDBACK.wrong }),
      choices: 'NO_KEYS',
      trial_duration: TIMING.feedback,
      save_trial_parameters: { stimulus: false },
      data: { part: 'feedback', block: t.block, trial_in_block: t.trial_in_block },
    }],
    conditional_function: () => FEEDBACK.show,
  };

  return { timeline: [stimulus, feedback] };
}

function blockStart(b, nBlocks, keys, state, debug) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    save_trial_parameters: { stimulus: false },
    stimulus: () => {
      let text = '';
      if (b > 0) {
        const p = state.blockCorrect / Math.max(1, state.blockN);
        text += `Block ${b} of ${nBlocks} done: ${Math.round(100 * p)}% correct.\n\nTake a short break if you like.\n\n`;
      }
      text += `Block ${b + 1} of ${nBlocks}\n\n`
        + `${key(keys.same)} = SAME picture      ${key(keys.different)} = DIFFERENT pictures\n\n`
        + 'Keep your eyes on the centre.\n\nPress SPACE to start.';
      return (debug ? '<div class="debug-banner">DEBUG RUN: shortened (remove ?debug=1 for the real study)</div>' : '')
        + `<div class="page preline center">${text}</div>`;
    },
    choices: [KEYS.start],
    data: { part: 'block_start', block: b + 1 },
    on_finish: (data) => {
      if (b > 0) data.prev_block_pcorrect = state.blockCorrect / Math.max(1, state.blockN);
      state.blockCorrect = 0;
      state.blockN = 0;
    },
  };
}

export function taskTimeline(design) {
  const state = { n: 0, correct: 0, blockCorrect: 0, blockN: 0 };
  const timeline = design.blocks.flatMap((trials, b) => [
    blockStart(b, design.blocks.length, design.keys, state, design.debug),
    ...trials.map((t) => trialTimeline(t, design.keys, state)),
  ]);
  return { timeline, state };
}
