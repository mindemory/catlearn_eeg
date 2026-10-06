// Participant IDs, attention-loss tracking and saving.

import { DATA, PROLIFIC, VERSION } from './config.js';

// ---------------------------------------------------------------- URL / Prolific
export function urlParams() {
  const q = new URLSearchParams(window.location.search);
  const flag = (k) => q.has(k) && q.get(k) !== '0' && q.get(k) !== 'false';
  return {
    prolificPid: q.get('PROLIFIC_PID') || '',
    studyId: q.get('STUDY_ID') || '',
    sessionId: q.get('SESSION_ID') || '',
    debug: flag('debug'),
    simulate: q.has('simulate') ? (q.get('simulate') === 'visual' ? 'visual' : 'data-only') : null,
    seed: q.has('seed') ? parseInt(q.get('seed'), 10) : null,
    // fix the noise condition (e.g. ?alpha=2 or ?alpha=2.00)
    alpha: q.get('alpha') ? Number(q.get('alpha')).toFixed(2) : null,
    skipIntro: (q.get('skip') || '').split(',').includes('intro'),
    skipCalibration: (q.get('skip') || '').split(',').includes('calibration'),
  };
}

export function isProlific(params) {
  return params.prolificPid !== '';
}

// ---------------------------------------------------------------- attention loss
// jsPsych logs every time the participant leaves the tab/window (blur) or full screen.
// We keep running counts so each trial row says how often this happened so far.
export const attention = { blur: 0, fullscreenExit: 0 };

export function onInteraction(event) {
  if (event.event === 'blur') attention.blur += 1;
  if (event.event === 'fullscreenexit') attention.fullscreenExit += 1;
}

// ---------------------------------------------------------------- session-level data
export function sessionProperties(params, seed) {
  return {
    participant: params.prolificPid || 'pilot',
    prolific_pid: params.prolificPid,
    study_id: params.studyId,
    session_id: params.sessionId,
    seed,
    task_version: VERSION,
    debug: params.debug,
    simulated: Boolean(params.simulate),
    user_agent: navigator.userAgent,
    started_at: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------- saving
export function dataFilename(params) {
  const id = params.prolificPid || 'pilot';
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
  return `${DATA.filePrefix}_${id}_${stamp}.csv`;
}

/**
 * Save all data. 'local' downloads the CSV (for pilots). To run on Prolific, add the
 * chosen backend here and set DATA.save in config.js, e.g.
 *   DataPipe (OSF):  POST to https://pipe.jspsych.org/api/data/ with
 *                    {experimentID, filename, data: csv}
 *   JATOS:           jatos.submitResultData(csv)
 *   Pavlovia:        use the jsPsych-Pavlovia plugin's 'finish' command
 * Resolves when the data are safely stored; rejects otherwise.
 */
export async function saveData(jsPsych, filename) {
  const csv = jsPsych.data.get().csv();
  switch (DATA.save) {
    case 'local':
      jsPsych.data.get().localSave('csv', filename);
      return { mode: 'local', bytes: csv.length };
    default:
      throw new Error(`DATA.save = '${DATA.save}' is not implemented yet (see data.js)`);
  }
}

export function completionUrl(code) {
  return `${PROLIFIC.completeUrl}${code}`;
}
