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
    // a label for runs outside Prolific (e.g. ?id=alex in a link shared with a friend)
    id: (q.get('id') || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40),
    save: q.get('save') === 'local' ? 'local' : null,   // ?save=local: download instead of uploading
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
    participant: params.prolificPid || params.id || 'pilot',
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
  const id = params.prolificPid || params.id || 'pilot';
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
  const tag = Math.random().toString(36).slice(2, 6);   // two runs in the same second never collide
  return `${DATA.filePrefix}_${id}_${stamp}_${tag}.csv`;
}

// DataPipe (pipe.jspsych.org) stores each file in the OSF project linked to the experiment
async function sendToDataPipe(filename, csv) {
  if (DATA.datapipeId.startsWith('REPLACE')) throw new Error('DATA.datapipeId is not set in config.js');
  let lastError = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetch('https://pipe.jspsych.org/api/data/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: '*/*' },
        body: JSON.stringify({ experimentID: DATA.datapipeId, filename, data: csv }),
      });
      if (res.ok) return { mode: 'datapipe', bytes: csv.length, attempts: attempt };
      lastError = new Error(`DataPipe answered ${res.status}: ${(await res.text()).slice(0, 200)}`);
      if (res.status >= 400 && res.status < 500) break;   // a setup problem: retrying won't help
    } catch (err) {
      lastError = err;                                     // network problem: try again
    }
    await new Promise((r) => setTimeout(r, 1500 * attempt));
  }
  throw lastError;
}

/**
 * Save all data (DATA.save in config.js). Resolves when the data are safely stored; rejects
 * otherwise. If an online upload fails, a copy is downloaded first, so nothing is lost
 * (err.backupFile names it).
 */
export async function saveData(jsPsych, filename, params = {}) {
  const csv = jsPsych.data.get().csv();
  const mode = params.simulate || params.save === 'local' ? 'local' : DATA.save;
  switch (mode) {
    case 'local':
      jsPsych.data.get().localSave('csv', filename);
      return { mode: 'local', bytes: csv.length };
    case 'datapipe':
      try {
        return await sendToDataPipe(filename, csv);
      } catch (err) {
        jsPsych.data.get().localSave('csv', filename);
        err.backupFile = filename;
        throw err;
      }
    default:
      throw new Error(`DATA.save = '${mode}' is not implemented (see data.js)`);
  }
}

export function completionUrl(code) {
  return `${PROLIFIC.completeUrl}${code}`;
}
