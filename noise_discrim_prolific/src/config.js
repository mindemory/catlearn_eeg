// Every design choice of the noise discrimination study lives here.
//
// Design: on each trial two 1/f^alpha noise patches flash for 100 ms, 6 deg left and right
// of fixation. Both have the same alpha; they are either the very same image (same) or two
// different images of that alpha (different). The participant answers same or different
// (F / J, which is which randomised per participant) within 2 s. Method of constant
// stimuli: ALPHAS (10 levels, 0 to 4) are randomly interleaved, every level equally often
// with same and different trials in every block, so sensitivity (d') and bias (c) can be
// computed at every alpha without assuming the shape of the curve.
// Patches are generated in the browser (noise.js) from a seed per trial, so every
// stimulus can be regenerated exactly afterwards (tools/noise_field.py).

export const VERSION = '1.2.0';   // 1.1: 100 ms, 5 blocks, frame-accurate plugin; 1.2: 20 + 20 per alpha

// ---------------------------------------------------------------- design
// Spectral slopes: 0 to 4 in 10 equal steps (4/9 = 0.444)
export const ALPHAS = Array.from({ length: 10 }, (_, i) => Number(((4 * i) / 9).toFixed(4)));

// `n` blocks; each has `perLevel` same and `perLevel` different trials of every alpha, in
// random order: 10 x 4 x 2 = 80 trials per block, 400 in all (20 same + 20 different per alpha)
export const BLOCKS = { n: 5, perLevel: 4 };

// ---------------------------------------------------------------- stimuli
// Noise patches as in task_design/noise_patches.py (the learning task's patches): white
// noise filtered to a 1/f^alpha amplitude spectrum, scaled to RMS contrast `rms` (of the
// [-1, 1] field, over the whole square), clipped to [-1, 1], in a circular aperture with a
// raised-cosine edge, shown as grey 0.5 + 0.5 * field * aperture on the grey background.
// One difference: the spectrum stops at cutoffCpd cycles per degree. Without a cutoff the
// finest detail would depend on each participant's screen resolution, which matters for low
// alphas (alpha 0 is white noise); 8 c/deg is about the finest detail resolvable at 6 deg
// eccentricity. Of the variance a full-band patch would have, the cutoff removes 80% at
// alpha 0, 25% at 0.89, 1% at 1.5 and < 0.1% from alpha 2 on.
export const NOISE = {
  grid: 128,              // samples across the patch (a power of 2, for the FFT); Nyquist 16 c/deg at 4 deg
  rms: 0.3,
  cutoffCpd: 8,
  apertureEdge: 0.1,      // raised-cosine edge, fraction of the radius
};

// ---------------------------------------------------------------- layout (degrees)
// x to the right, y up, (0, 0) = fixation at the screen centre. Pixels per degree come
// from the calibration (calibration.js); if the layout does not fit the window, everything
// shrinks by the same factor (layout_scale).
export const LAYOUT = {
  positions: { left: [-6, 0], right: [6, 0] },
  patchSize: 4,                                       // aperture diameter
  fixation: { outer: 0.6, inner: 0.2, line: 0.15 },   // bull's eye as in the MATLAB task
  extent: { width: 18, height: 6 },                   // everything drawn (centred on fixation), for the fit check
  background: '#808080',
};

// ---------------------------------------------------------------- timing (ms) and keys
export const TIMING = {
  fixation: 500,      // fixation alone before the patches
  stimulus: 100,      // patches on screen (6 frames at 60 Hz)
  response: 2000,     // after the patches go off; answers during the patches count too
  feedback: 300,      // the fixation turns green / red
};

// Response keys (either case). Which one means SAME is randomised per participant (from the
// seed). SPACE continues after instructions and breaks.
export const KEYS = { pair: ['f', 'j'], start: ' ' };

// Feedback: colour of the fixation's centre disc after each answer (late = no answer in time)
export const FEEDBACK = { show: true, correct: '#2ecc40', wrong: '#ff4136', late: '#ff4136' };

// ---------------------------------------------------------------- participants and platform
export const BROWSER = {
  minWidth: 1000,          // px; smaller windows are asked to enlarge, phones/tablets are excluded
  minHeight: 650,
  allowMobile: false,
  minRefreshHz: 50,        // flagged in the data, not excluded
};

// Screen calibration (virtual chinrest): credit-card sizing (pixels per mm), then the
// blind-spot task (viewing distance). An implausible measurement is repeated once; if it is
// still implausible we assume the values below and flag the participant.
export const CALIBRATION = {
  blindspotReps: 5,
  plausibleDistanceMm: [300, 1000],
  plausiblePxPerMm: [1.5, 12],
  assumedDistanceMm: 600,
  assumedPxPerMm: 96 / 25.4,      // a standard 96-dpi screen
  fitMargin: 0.95,                // the layout may fill at most this fraction of the window
};

export const COMPREHENSION = { maxAttempts: 3 };   // instruction re-reads before continuing anyway (flagged)

export const PROLIFIC = {
  completionCode: 'REPLACE_WITH_COMPLETION_CODE',   // from the Prolific study page
  noConsentCode: 'REPLACE_WITH_NO_CONSENT_CODE',    // optional "returned" code for declined consent
  completeUrl: 'https://app.prolific.com/submissions/complete?cc=',
};

export const CONTACT = {
  name: 'Mrugank Dake, PhD',
  address: 'HB 6207, Moore Hall, Hanover, NH 03755',
  email: 'mrugank.dake@dartmouth.edu',
};

// Where data go at the end:
//   'local'     the CSV downloads to the participant's computer (piloting on your own machine)
//   'datapipe'  the CSV is sent through DataPipe (pipe.jspsych.org) to the storage linked to
//               the experiment (here a Google Drive folder), once, at the end of the session.
//               If the upload fails, a copy downloads instead and the participant is asked to
//               email it (CONTACT).
// Simulated runs (?simulate=1) always save locally, so they never reach the dataset; ?save=local
// does the same for any run (e.g. testing the hosted page without adding a session).
export const DATA = { save: 'datapipe', filePrefix: 'noise_discrim', datapipeId: 'A6UGkxMHa4mJ' };

// URL options (for piloting): ?debug=1 shortens the study to 2 blocks of 20 trials,
// ?simulate=1 (or =visual) lets jsPsych play the whole experiment, ?seed=123 fixes the design,
// ?id=name labels a run outside Prolific (e.g. a friend's link),
// ?skip=intro skips consent, instructions and the quiz, ?skip=calibration the screen setup.
