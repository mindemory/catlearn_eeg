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
    save: q.get('save') === 'local' ? 'local' : null,   // ?save=local: download instead of uploading
    // screening (SCREENING): ?screen=1 forces it on, ?screen=0 off (default: on, except in debug
    // and simulated runs)
    screen: q.get('screen') === '1' ? true : (q.get('screen') === '0' ? false : null),
    // fix the test sequence version (?version=A or B)
    version: q.get('version') ? q.get('version').toUpperCase() : null,
    skipIntro: (q.get('skip') || '').split(',').includes('intro'),
    skipCalibration: (q.get('skip') || '').split(',').includes('calibration'),
    skipQuestionnaire: (q.get('skip') || '').split(',').includes('questionnaire'),
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
// File name without any participant label (the Prolific ID is inside the data): prefix, start
// time (UTC) and a random tag, e.g. catlearn_online_20261006-195146_756c.csv
export function dataFilename() {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
  const tag = Math.random().toString(36).slice(2, 6);   // two runs in the same second never collide
  return `${DATA.filePrefix}_${stamp}_${tag}.csv`;
}

// Download the CSV to this computer (local runs, and the backup when an upload fails)
export function saveLocal(jsPsych, filename) {
  jsPsych.data.get().localSave('csv', filename);
}

export function completionUrl(code) {
  return `${PROLIFIC.completeUrl}${code}`;
}
