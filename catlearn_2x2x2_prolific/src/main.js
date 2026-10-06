// Entry point: builds the participant's design, assembles the timeline and runs it.
//   browser check -> consent -> full screen -> screen calibration -> instructions + quiz
//   -> blocks -> bonus -> save -> Prolific

import { calibrationTimeline } from './calibration.js';
import { BROWSER, PROLIFIC } from './config.js';
import { completionUrl, dataFilename, isProlific, onInteraction, saveData, sessionProperties, urlParams } from './data.js';
import { buildDesign, designImages, newSeed } from './design.js';
import { consentTrial, instructionsWithCheck } from './instructions.js';
import { bonusDollars, formatCoins, formatDollars } from './reward.js';
import { taskTimeline } from './task.js';

const params = urlParams();
const seed = params.seed ?? newSeed();
const design = buildDesign(seed, { debug: params.debug, alpha: params.alpha });

const saved = { ok: false, mode: null, error: null, declined: false };

const jsPsych = initJsPsych({
  on_interaction_data_update: onInteraction,
  on_finish: () => showEnd(),
});
jsPsych.data.addProperties({ ...sessionProperties(params, seed), key_yes: design.keys.yes, key_no: design.keys.no,
                             test_rule: design.testRule, noise_alpha: design.alpha });
window.catlearn = { jsPsych, design, params };   // for inspection in the browser console

const browserCheck = {
  type: jsPsychBrowserCheck,
  minimum_width: BROWSER.minWidth,
  minimum_height: BROWSER.minHeight,
  inclusion_function: (d) => BROWSER.allowMobile || !d.mobile,
  exclusion_message: () => '<p>Sorry, this study needs a computer with a keyboard. Please return it on Prolific.</p>',
  data: { part: 'browser_check' },
  on_finish: (data) => { data.low_refresh = data.vsync_rate !== null && data.vsync_rate < BROWSER.minRefreshHz; },
};

const preload = {
  type: jsPsychPreload,
  images: designImages(design),
  data: { part: 'preload' },
};

const declined = () => {
  saved.declined = true;
  const go = isProlific(params) && !PROLIFIC.noConsentCode.startsWith('REPLACE');
  jsPsych.abortExperiment(
    '<p>You did not consent, so the study ends here. Thank you for your time.</p>'
    + (go ? '<p>Returning you to Prolific...</p>' : '<p>Please return the study on Prolific.</p>'),
  );
  if (go) setTimeout(() => { window.location.href = completionUrl(PROLIFIC.noConsentCode); }, 3000);
};

const fullscreenOn = {
  type: jsPsychFullscreen,
  fullscreen_mode: true,
  message: '<p>The study runs in full screen. Please keep it that way until the end.</p>',
  button_label: 'Enter full screen',
  data: { part: 'fullscreen_on' },
};

// Simulated runs cannot do the card / blind-spot measurement, so they use the assumed values
const calibration = calibrationTimeline({ skip: Boolean(params.simulate) || params.skipCalibration });

const task = taskTimeline(design);

const designRow = {
  type: jsPsychCallFunction,
  func: () => {},
  data: {
    part: 'design',
    design_json: JSON.stringify(design.blocks.map(({ trials, ...b }) => b)),   // rules, order, symbols
  },
};

const finale = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    const { totalPCorrect: p, bonusCoins } = task.state;
    return '<div class="page preline center">'
      + (p === null ? '' : `Proportion correct in the test = ${p.toFixed(2)}\n\n`)
      + `You won <b>${formatCoins(bonusCoins)} coins</b> in the test.\n`
      + `They become your bonus of ${formatDollars(bonusDollars(bonusCoins))},\n`
      + 'paid through Prolific after we review your submission.\n\n'
      + 'Thanks for your participation!\n\nPress the space bar to finish and save your answers.</div>';
  },
  choices: [' '],
  data: { part: 'final' },
  on_finish: (data) => {
    data.test_pcorrect = task.state.totalPCorrect;
    data.bonus_coins = task.state.bonusCoins;
    data.bonus_usd = bonusDollars(task.state.bonusCoins);
    data.practice_coins = task.state.practiceCoins;
    data.interaction_log = JSON.stringify(jsPsych.data.getInteractionData().values());
    data.finished_at = new Date().toISOString();
  },
};

const fullscreenOff = { type: jsPsychFullscreen, fullscreen_mode: false, delay_after: 0, data: { part: 'fullscreen_off' } };

// Saving waits until the data are stored (or fail) before the experiment ends
const save = {
  type: jsPsychCallFunction,
  async: true,
  func: (done) => {
    document.querySelector('.jspsych-content').innerHTML = '<p>Saving your data, please do not close this window...</p>';
    saveData(jsPsych, dataFilename(params))
      .then((result) => { Object.assign(saved, { ok: true, mode: result.mode }); done(result); })
      .catch((err) => { console.error(err); Object.assign(saved, { ok: false, error: err.message }); done({ error: err.message }); });
  },
  data: { part: 'save' },
};

function showEnd() {
  if (saved.declined) return;                     // abortExperiment already shows its message
  let message;
  if (!saved.ok) {
    message = '<p>Saving failed. Please contact the researcher on Prolific and do not close this window.</p>'
      + `<p class="small">${saved.error}</p>`;
  } else if (isProlific(params)) {
    message = '<p>Your answers are saved. Returning you to Prolific...</p>';
    setTimeout(() => { window.location.href = completionUrl(PROLIFIC.completionCode); }, 2000);
  } else {
    message = `<p>Pilot finished. Data saved (${saved.mode}).</p>`;
  }
  jsPsych.getDisplayElement().innerHTML = `<div class="page center">${message}</div>`;
}

const consent = params.skipIntro ? [] : [consentTrial(declined)];
const instructions = params.skipIntro ? [] : [instructionsWithCheck(design, design.example)];
const timeline = [browserCheck, preload, ...consent, fullscreenOn, calibration, ...instructions, designRow, task,
                  finale, fullscreenOff, save];

if (params.simulate) {
  jsPsych.simulate(timeline, params.simulate);
} else {
  jsPsych.run(timeline);
}
