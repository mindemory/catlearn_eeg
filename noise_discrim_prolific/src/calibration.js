// Screen calibration: how many pixels make one degree of visual angle for this participant.
//
// jsPsych's virtual chinrest measures pixels per mm (resizing a credit card on screen) and the
// viewing distance (the blind spot sits ~13.5 deg from fixation). An implausible measurement is
// repeated once; if it is still implausible, the assumed values in config.js CALIBRATION are
// used and the participant is flagged (calibration_source).
//
// The result is applied as the CSS variable --deg (pixels per degree). If the whole layout
// (LAYOUT.extent) would not fit the window, --deg is shrunk so it does (layout_scale < 1).

import { CALIBRATION, LAYOUT } from './config.js';

export const scale = {
  pxPerMm: CALIBRATION.assumedPxPerMm,
  distanceMm: CALIBRATION.assumedDistanceMm,
  pxPerDeg: null,
  layoutScale: 1,
  source: 'not_run',      // measured | assumed_distance | assumed_card | assumed_both | skipped | not_run
  attempts: 0,
};

const inRange = (v, [lo, hi]) => Number.isFinite(v) && v >= lo && v <= hi;

function pxPerDegree(pxPerMm, distanceMm) {
  return 2 * distanceMm * Math.tan((0.5 * Math.PI) / 180) * pxPerMm;
}

// Set --deg for the current window; called after calibration and on every resize
export function applyScale() {
  if (scale.pxPerDeg === null) scale.pxPerDeg = pxPerDegree(scale.pxPerMm, scale.distanceMm);
  const { width, height } = LAYOUT.extent;
  const { innerWidth: w, innerHeight: h } = window;
  const fit = !w || !h ? 1    // window size unknown (e.g. hidden tab): don't shrink to nothing
    : Math.min(1, (CALIBRATION.fitMargin * h) / (height * scale.pxPerDeg), (CALIBRATION.fitMargin * w) / (width * scale.pxPerDeg));
  scale.layoutScale = fit;
  document.documentElement.style.setProperty('--deg', `${(scale.pxPerDeg * fit).toFixed(3)}px`);
}
window.addEventListener('resize', () => { if (scale.pxPerDeg !== null) applyScale(); });

// What every trial row records about the screen
export function scaleData() {
  return {
    px_per_deg: scale.pxPerDeg,
    layout_scale: scale.layoutScale,
    window_width: window.innerWidth,
    window_height: window.innerHeight,
  };
}

function finalize(last) {
  const cardOk = inRange(last?.px2mm, CALIBRATION.plausiblePxPerMm);
  const distOk = inRange(last?.view_dist_mm, CALIBRATION.plausibleDistanceMm);
  scale.pxPerMm = cardOk ? last.px2mm : CALIBRATION.assumedPxPerMm;
  scale.distanceMm = distOk ? last.view_dist_mm : CALIBRATION.assumedDistanceMm;
  scale.source = cardOk && distOk ? 'measured'
    : cardOk ? 'assumed_distance' : distOk ? 'assumed_card' : 'assumed_both';
  scale.pxPerDeg = pxPerDegree(scale.pxPerMm, scale.distanceMm);
  applyScale();
}

/**
 * Calibration timeline. With skip = true (simulation, ?skip=calibration) nothing is measured
 * and the assumed values are used.
 */
export function calibrationTimeline({ skip = false } = {}) {
  const summary = {
    type: jsPsychCallFunction,
    func: () => {},
    data: { part: 'calibration_summary' },
    on_finish: (data) => {
      Object.assign(data, {
        calibration_source: scale.source, calibration_attempts: scale.attempts, px_per_mm: scale.pxPerMm,
        view_dist_mm: scale.distanceMm, ...scaleData(),
      });
    },
  };
  if (skip) {
    return { timeline: [{ type: jsPsychCallFunction, func: () => { scale.source = 'skipped'; applyScale(); },
                          data: { part: 'calibration_skipped' } }, summary] };
  }

  let last = null;
  const plausible = () => inRange(last?.px2mm, CALIBRATION.plausiblePxPerMm)
    && inRange(last?.view_dist_mm, CALIBRATION.plausibleDistanceMm);

  const intro = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div class="page"><h2>Screen setup</h2>
      <p class="center">So that the pictures have the same size for everyone, we first measure your screen
      and how far you sit from it. This takes about 2 minutes.</p>
      <p class="center">You need a <b>card the size of a credit card</b> (any bank, ID or loyalty card).
      Sit as you will sit during the whole study, about an arm's length from the screen.</p></div>`,
    choices: ['Start'],
    data: { part: 'calibration_intro' },
  };
  const chinrest = {
    type: jsPsychVirtualChinrest,
    blindspot_reps: CALIBRATION.blindspotReps,
    resize_units: 'none',
    // The blind spot only works if the eye is pointed AT the square (not just attending to it)
    blindspot_prompt: `
      <p>Now we measure how far away you are sitting.</p>
      <div style="text-align: left">
        <ol>
          <li>Put your left hand on the <b>space bar</b>.</li>
          <li>Cover your <b>right eye</b> with your right hand.</li>
          <li><b>Look straight at the black square</b> with your left eye: move your eye onto it and keep it
            there. Do not follow the red ball.</li>
          <li>The <span style="color: red; font-weight: bold;">red ball</span> moves to the left, away from
            the square. At some point it seems to vanish (it falls in your blind spot). Press the space bar the
            moment it disappears.</li>
        </ol>
      </div>
      <p>Press the space bar when you are ready to begin.</p>`,
    data: { part: 'calibration' },
    on_finish: (data) => {
      scale.attempts += 1;
      last = data;
      data.calibration_attempt = scale.attempts;
      data.calibration_plausible = plausible();
    },
  };
  const retry = {
    timeline: [{
      type: jsPsychHtmlButtonResponse,
      stimulus: `<div class="page center"><p>That measurement looked unusual. Let's do it once more.</p>
        <p>Make sure the card on the screen matches your real card exactly. Then cover your right eye, keep your
        left eye pointed straight at the black square (don't follow the ball) and press the space bar the
        moment the red ball disappears.</p></div>`,
      choices: ['Measure again'],
      data: { part: 'calibration_retry' },
    }],
    conditional_function: () => !plausible() && scale.attempts < 2,
  };
  const finish = { type: jsPsychCallFunction, func: () => finalize(last), data: { part: 'calibration_apply' } };

  return {
    timeline: [
      intro,
      { timeline: [chinrest, retry], loop_function: () => !plausible() && scale.attempts < 2 },
      finish,
      summary,
    ],
  };
}
