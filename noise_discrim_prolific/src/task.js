// The task: a practice block (same mix, not analysed), then blocks of trials with a break
// screen between them. A trial (plugin-noise-pair.js):
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
      if (t.phase === 'main') {                         // the final score leaves out the practice
        state.n += 1;
        state.correct += data.correct ? 1 : 0;
      }
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

const keyLine = (keys) => `${key(keys.same)} = SAME picture      ${key(keys.different)} = DIFFERENT pictures`;
const debugBanner = (debug) => (debug
  ? '<div class="debug-banner">DEBUG RUN: shortened (remove ?debug=1 for the real study)</div>' : '');

function practiceStart(nTrials, keys, debug) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    save_trial_parameters: { stimulus: false },
    stimulus: debugBanner(debug) + '<div class="page preline center">Practice\n\n'
      + `First, ${nTrials} practice trials to get used to the task. They work exactly like the real task`
      + ' and are not scored.\n\n'
      + `${keyLine(keys)}\n\nKeep your eyes on the centre.\n\nPress SPACE to start.</div>`,
    choices: [KEYS.start],
    data: { part: 'practice_start' },
  };
}

function blockStart(b, nBlocks, keys, state, debug) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    save_trial_parameters: { stimulus: false },
    stimulus: () => {
      let text = '';
      const p = Math.round((100 * state.blockCorrect) / Math.max(1, state.blockN));
      if (b > 0) {
        text += `Block ${b} of ${nBlocks} done: ${p}% correct.\n\nTake a short break if you like.\n\n`;
      } else if (state.blockN > 0) {
        text += `Practice done: ${p}% correct.\n\nNow the real task: ${nBlocks} blocks, with a short break after`
          + ' each.\n\n';
      }
      text += `Block ${b + 1} of ${nBlocks}\n\n${keyLine(keys)}\n\nKeep your eyes on the centre.\n\nPress SPACE to start.`;
      return debugBanner(debug) + `<div class="page preline center">${text}</div>`;
    },
    choices: [KEYS.start],
    data: { part: 'block_start', block: b + 1 },
    on_finish: (data) => {
      if (state.blockN > 0) data.prev_block_pcorrect = state.blockCorrect / state.blockN;   // block b, or the practice
      state.blockCorrect = 0;
      state.blockN = 0;
    },
  };
}

export function taskTimeline(design) {
  const state = { n: 0, correct: 0, blockCorrect: 0, blockN: 0 };
  const practice = design.practice.length
    ? [practiceStart(design.practice.length, design.keys, design.debug),
       ...design.practice.map((t) => trialTimeline(t, design.keys, state))]
    : [];
  const timeline = practice.concat(design.blocks.flatMap((trials, b) => [
    blockStart(b, design.blocks.length, design.keys, state, design.debug),
    ...trials.map((t) => trialTimeline(t, design.keys, state)),
  ]));
  return { timeline, state };
}
