// Entry point: builds the participant's design, assembles the timeline and runs it.
//   browser check -> consent -> full screen -> screen calibration -> instructions + quiz
//   -> blocks -> bonus -> save -> Prolific
// Saving: online runs use DataPipe's jsPsych extension, which stages trials as the session
// runs (so a participant who quits partway still leaves a .partial.json) and uploads the
// whole CSV when the session ends. Local runs (DATA.save = 'local', ?save=local, simulations)
// download the CSV instead.

import { calibrationTimeline } from './calibration.js';
import { BONUS, BROWSER, CONTACT, DATA, PROLIFIC, QUESTIONNAIRE, SCREENING } from './config.js';
import { completionUrl, dataFilename, isProlific, onInteraction, saveLocal, sessionProperties, urlParams } from './data.js';
import { buildDesign, designImages, newSeed } from './design.js';
import { consentTrial, instructionsWithCheck } from './instructions.js';
import { formatDollars, totalBonus } from './bonus.js';
import { questionnaireTimeline } from './questionnaire.js';
import { taskTimeline } from './task.js';

const params = urlParams();
const seed = params.seed ?? newSeed();
const design = buildDesign(seed, { debug: params.debug, version: params.version });

const saved = { ok: false, mode: null, error: null, backupFile: null, declined: false, screenedOut: null,
                finished: false };   // finished: reached the final screen (only then the completion code)
const filename = dataFilename();                 // no participant label: prefix, start time, random tag
const online = DATA.save === 'datapipe' && !params.simulate && params.save !== 'local';

// The DataPipe extension owns the online upload (no save trial then). on_save runs when the
// final upload has finished, before showEnd(); if it failed, a copy downloads as a backup.
const pipe = {
  type: jsPsychExtensionPipe,
  params: {
    experiment_id: DATA.datapipeId,
    filename,
    // a participant who declines consent leaves only that decision, nothing else
    data_string: () => (saved.declined
      ? jsPsych.data.get().filter({ part: 'consent' }).ignore('user_agent').csv()
      : jsPsych.data.get().csv()),
    wait_message: '<div class="page center"><p>Saving your answers, please do not close this window...</p></div>',
    on_save: (result) => {
      if (result.ok) {
        Object.assign(saved, { ok: true, mode: 'datapipe' });
      } else {
        saveLocal(jsPsych, filename);
        Object.assign(saved, { ok: false, error: `DataPipe answered ${result.status}`, backupFile: filename });
      }
    },
  },
};

const jsPsych = initJsPsych({
  on_interaction_data_update: onInteraction,
  on_finish: () => showEnd(),
  extensions: online ? [pipe] : [],
});
jsPsych.data.addProperties({ ...sessionProperties(params, seed), session_file: filename,
                             key_cat1: design.keys.cat1, key_cat0: design.keys.cat0,
                             version: design.version, sequence: design.sequence,
                             // 'url' on Prolific: the study's URL fixes the version, so its completion code matches
                             version_source: params.version ? 'url' : 'random' });
window.catlearn = { jsPsych, design, params, screenOut: (reason) => screenOut(reason) };   // for inspection and testing in the browser console

const browserCheck = {
  type: jsPsychBrowserCheck,
  minimum_width: BROWSER.minWidth,
  minimum_height: BROWSER.minHeight,
  inclusion_function: (d) => BROWSER.allowMobile || !d.mobile,
  // Excluded (window too small after the resize prompt, or a phone / tablet): out through the
  // screen-out path, never the completion code. An empty message keeps showEnd()'s page.
  exclusion_message: () => { markScreenedOut('browser_check'); return ''; },
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
  const noConsentCode = PROLIFIC.noConsentCodes[design.version] ?? 'REPLACE';
  const go = isProlific(params) && !noConsentCode.startsWith('REPLACE');
  jsPsych.abortExperiment(
    '<p>You did not consent, so the study ends here. Thank you for your time.</p>'
    + (go ? '<p>Returning you to Prolific...</p>' : '<p>Please return the study on Prolific.</p>'),
  );
  if (go) setTimeout(() => { window.location.href = completionUrl(noConsentCode); }, 3000);
};

const fullscreenOn = {
  type: jsPsychFullscreen,
  fullscreen_mode: true,
  message: '<p>The study runs in full screen. Please keep it that way until the end.</p>',
  button_label: 'Enter full screen',
  data: { part: 'fullscreen_on' },
};

// Screening (SCREENING): a participant who fails a check leaves now, with the fixed screen-out
// payment, through Prolific's custom screening path. Their data are saved as usual (online:
// the DataPipe extension uploads when the experiment ends; locally: downloaded here).
const screening = params.screen ?? !(params.debug || params.simulate);

function markScreenedOut(reason) {
  if (saved.screenedOut) return false;
  saved.screenedOut = reason;
  jsPsych.data.addProperties({ screened_out: reason });
  if (!online) {
    saveLocal(jsPsych, filename);
    Object.assign(saved, { ok: true, mode: 'local' });
  }
  return true;
}

function screenOut(reason) {
  if (!markScreenedOut(reason)) return;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  // no end message: jsPsych would write it after on_finish, over showEnd()'s screen-out page
  jsPsych.getDisplayElement().innerHTML = '<div class="page center"><p>Saving your answers, please do not close this window...</p></div>';
  jsPsych.abortExperiment();
}

// Simulated runs cannot do the card / blind-spot measurement, so they use the assumed values
const calibration = calibrationTimeline({ skip: Boolean(params.simulate) || params.skipCalibration });

const task = taskTimeline(design, { screening, onScreenOut: screenOut });

const designRow = {
  type: jsPsychCallFunction,
  func: () => {},
  data: {
    part: 'design',
    design_json: JSON.stringify(design.blocks.map(({ trials, ...b }) => b)),   // rules, order, fractals
  },
};

const finale = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    const { totalPCorrect: p } = task.state;
    return '<div class="page preline center">'
      + (p === null ? '' : `Proportion correct in the test blocks = ${p.toFixed(2)}\n\n`)
      + `Your performance bonus: <b>${formatDollars(totalBonus(task.state.blockBonuses))}</b>,\n`
      + 'paid through Prolific after we review your submission.\n\n'
      + 'Thanks for your participation!\n\nPress the space bar to finish and save your answers.</div>';
  },
  choices: [' '],
  data: { part: 'final' },
  on_finish: (data) => {
    data.test_pcorrect = task.state.totalPCorrect;
    data.test_correct = task.state.testCorrect;
    data.test_rounds = task.state.testTrials;
    data.block_bonuses_usd = JSON.stringify(task.state.blockBonuses.map((x) => Math.round(x * 100) / 100));
    data.bonus_usd = totalBonus(task.state.blockBonuses);
    data.bonus_tiers = JSON.stringify(BONUS.tiers);       // the tiers this session was paid by
    data.interaction_log = JSON.stringify(jsPsych.data.getInteractionData().values());
    data.finished_at = new Date().toISOString();
    saved.finished = true;
  },
};

const fullscreenOff = { type: jsPsychFullscreen, fullscreen_mode: false, delay_after: 0, data: { part: 'fullscreen_off' } };

// Local runs only: download the CSV (online runs are saved by the DataPipe extension)
const save = {
  type: jsPsychCallFunction,
  func: () => {
    saveLocal(jsPsych, filename);
    Object.assign(saved, { ok: true, mode: 'local' });
  },
  data: { part: 'save' },
};

function showEnd() {
  if (saved.declined) return;                     // abortExperiment already shows its message
  let message;
  if (!saved.ok) {
    message = saved.backupFile
      ? '<p>Saving online failed, so a copy of your answers was downloaded to your computer as '
        + `<b>${saved.backupFile}</b> (usually in your Downloads folder).</p>`
        + `<p>Please send that file through Prolific or to ${CONTACT.email}, so we can pay you. Thank you!</p>`
      : '<p>Saving failed. Please contact the researcher on Prolific and do not close this window.</p>';
    message += `<p class="small">${saved.error}</p>`;
  } else if (saved.screenedOut) {
    const code = PROLIFIC.screenOutCodes[design.version] ?? 'REPLACE';
    message = saved.screenedOut === 'browser_check'
      ? '<p>Thank you for your interest. This study needs a desktop or laptop computer with a browser window of at '
        + `least ${BROWSER.minWidth} × ${BROWSER.minHeight} pixels, so it ends here. `
        + `You will receive ${formatDollars(SCREENING.payUsd)} for your time.</p>`
      : '<p>Thank you for your time. Based on your answers so far, this study is not a good match for you, '
        + `so it ends here. You will receive ${formatDollars(SCREENING.payUsd)} for taking part.</p>`;
    if (isProlific(params) && !code.startsWith('REPLACE')) {
      message += '<p>Returning you to Prolific...</p>';
      setTimeout(() => { window.location.href = completionUrl(code); }, 3000);
    } else if (isProlific(params)) {
      message += `<p>Please return to Prolific and contact the researcher (${CONTACT.email}).</p>`;
    } else {
      message += `<p class="small">Pilot: screened out (${saved.screenedOut}). Data saved (${saved.mode}).</p>`;
    }
  } else if (!saved.finished) {
    // ended early for any other reason: never send the completion code
    message = '<p>The study ended before its end. Please return it on Prolific ("Stop without completing"), '
      + `and contact the researcher (${CONTACT.email}) if this was unexpected.</p>`;
  } else if (isProlific(params)) {
    message = '<p>Your answers are saved. Returning you to Prolific...</p>';
    setTimeout(() => { window.location.href = completionUrl(PROLIFIC.completionCodes[design.version]); }, 2000);
  } else {
    message = `<p>Pilot finished. Data saved (${saved.mode}).</p>`;
  }
  jsPsych.getDisplayElement().innerHTML = `<div class="page center">${message}</div>`;
}

const consent = params.skipIntro ? [] : [consentTrial(declined)];
const instructions = params.skipIntro ? [] : [instructionsWithCheck(design, design.example,
  { onFail: screening && SCREENING.quiz ? () => screenOut('quiz') : null })];
const questionnaire = QUESTIONNAIRE.enabled && !params.skipQuestionnaire ? [questionnaireTimeline(design)] : [];
const timeline = [browserCheck, preload, ...consent, fullscreenOn, calibration, ...instructions, designRow, task,
                  ...questionnaire, finale, fullscreenOff, ...(online ? [] : [save])];

if (params.simulate) {
  jsPsych.simulate(timeline, params.simulate);
} else {
  jsPsych.run(timeline);
}
