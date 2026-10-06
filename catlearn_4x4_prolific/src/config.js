// Every design choice of the online experiment lives here. Nothing else needs editing to
// change blocks, rules, timing, keys, sizes or the Prolific / data settings.
//
// Design: a slot machine shows two symbols (fractals), one left (A) and one right (B) of
// fixation. A rule says which pairs win (1) and which lose (0); participants predict win or
// lose and learn from the payout. Correct predictions earn coins, wrong or late ones cost
// coins, and test-block coins become a bonus. Every block is a new machine with new fractals.
//   Practice (2x2: 2 fractals per side): type I (A or B, random), then XOR; each runs until
//     the participant is accurate enough (PRACTICE_CRITERION).
//   Test (4x4: 4 fractals per side): 3 blocks of one type per participant (TEST_TYPES). Blocks
//     1-2 use one version (A- or B-dominant, random), block 3 the other.
// Each trial is self-paced: the participant presses SPACE to pull the lever (or the round
// starts by itself after TIMING.wait), the symbols appear, they answer, feedback follows.

export const VERSION = '3.1.0';   // 3.1: equidistant fractal groups

// ---------------------------------------------------------------- rules
// Outcome (1 = win, 0 = lose) of every pair: compound = size * a + b, where a and b (0 ..
// size-1) say which fractal is shown left (A) and right (B). Generated from the task
// definitions in kernel_model/kernel_modes.py (Design('2x2'), Design('4x4')), the same ones
// as in the task-type figures. Within a block the fractals are assigned to levels at random.
export const RULES = {
  x2_I_A: { type: 'I', size: 2, labels: [1, 1, 0, 0] },                      // A
  x2_I_B: { type: 'I', size: 2, labels: [1, 0, 1, 0] },                      // B
  x2_XOR: { type: 'XOR', size: 2, labels: [1, 0, 0, 1] },                    // AB
  x4_I_A: { type: 'I', size: 4, labels: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] },     // A
  x4_I_B: { type: 'I', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] },     // B
  x4_VI_A: { type: 'VI', size: 4, labels: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0] },   // A+AB
  x4_VI_B: { type: 'VI', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0] },   // B+AB
  x4_X_A: { type: 'X', size: 4, labels: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] },     // A+AB
  x4_X_B: { type: 'X', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0] },     // B+AB
  x4_XII_A: { type: 'XII', size: 4, labels: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }, // AB+A
  x4_XII_B: { type: 'XII', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1] }, // AB+B
  x4_XIV_A: { type: 'XIV', size: 4, labels: [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1] }, // AB+A
  x4_XIV_B: { type: 'XIV', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1] }, // AB+B
  x4_XV_A: { type: 'XV', size: 4, labels: [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1] },   // AB+A
  x4_XV_B: { type: 'XV', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1] },   // AB+B
  x4_XVI: { type: 'XVI', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1] },   // AB (no A/B version)
};

// Test types: each participant gets one (at random) as [A-dominant, B-dominant] versions
export const TEST_TYPES = {
  I: ['x4_I_A', 'x4_I_B'],
  VI: ['x4_VI_A', 'x4_VI_B'],
  X: ['x4_X_A', 'x4_X_B'],
  XII: ['x4_XII_A', 'x4_XII_B'],
  XIV: ['x4_XIV_A', 'x4_XIV_B'],
  XV: ['x4_XV_A', 'x4_XV_B'],
  XVI: ['x4_XVI', 'x4_XVI'],
};

// Which test types this run uses: the pilot uses 4 types, 10 participants each; the main
// study all 7, 30 each. Each participant gets one of STUDY.types[STUDY.phase] at random, unless
// the URL fixes it (?type=XV; ?first=A or B fixes which version comes first). For exact
// counts, run one Prolific study per type with ?type= in its URL. Saved as study_phase.
export const STUDY = {
  phase: 'pilot',
  types: {
    pilot: ['VI', 'X', 'XV', 'XVI'],
    main: ['I', 'VI', 'X', 'XII', 'XIV', 'XV', 'XVI'],
  },
};

// ---------------------------------------------------------------- blocks
// One entry per block, in order. Fields:
//   phase      'practice' | 'test'
//   rule       a RULES key; a list of keys (one picked at random per participant); or
//              'test:first' / 'test:second' = this participant's test type, in the version
//              shown first (blocks 1-2) or second (block 3)
//   reps       passes through all pairs (each pass in a new random order)
//   criterion  practice: end the block once PRACTICE_CRITERION is met (reps = the maximum)
export const BLOCKS = [
  { phase: 'practice', rule: ['x2_I_A', 'x2_I_B'], reps: 20, criterion: true },
  { phase: 'practice', rule: 'x2_XOR', reps: 20, criterion: true },
  { phase: 'test', rule: 'test:first', reps: 8 },
  { phase: 'test', rule: 'test:first', reps: 8 },
  { phase: 'test', rule: 'test:second', reps: 8 },
];

// Practice ends when at least `minCorrect` of the last `window` answers are correct (75%);
// at the latest after `reps` passes (flagged in the data if never met)
export const PRACTICE_CRITERION = { window: 10, minCorrect: 8 };

// ---------------------------------------------------------------- stimuli
// Fractals matched in brightness, colourfulness and size, in groups whose members are all
// equally distinct from each other (DreamSim distance; ../fractal_stimuli/make_fractal_sets.py):
// 18 groups of 4 for the test, 12 pairs for practice. Each side of a block shows one whole
// group, never used again for this participant. The groups are in fractalDir/groups.js.
export const STIMULI = {
  fractalDir: 'stimuli/fractal_groups/',
};

// Layout in degrees of visual angle (dva), as in the MATLAB EEG task: 4 deg symbols centred
// 6 deg left and right of fixation. x to the right, y up, (0, 0) = the fixation point at the
// screen centre. Pixels per degree come from the calibration (calibration.js); if the layout
// is larger than the window, everything shrinks by the same factor to fit (layout_scale).
export const LAYOUT = {
  positions: { A: [-6, 0], B: [6, 0] },
  fractalSize: 4,
  // The PNGs (500 px) have transparent margins. In the EEG task's fractals the square
  // enclosing every fractal is 368 px (as cropped by the MATLAB task); drawing the image
  // 500/368 larger makes that square fractalSize degrees. The new fractals have the same
  // median extent and area as those, so the same scale shows them at the same size (the
  // spikiest reach 4.3 deg across).
  fractalImageScale: 500 / 368,
  fixation: { outer: 0.6, inner: 0.2, line: 0.15 },   // bull's eye as in the MATLAB task
  machine: { center: [0, 0], size: [20, 8.5], border: 0.5, color: '#c9a227' },
  lever: { length: 3.5, knob: 1 },                    // on the right of the machine
  outcomeRing: { size: 5.4, width: 0.3 },             // feedback circle around each symbol
  hintY: -5.6,                                        // "press SPACE" hint below the machine
  extent: { width: 24, height: 12.5 },                // everything drawn, for the fit check
  background: '#808080',
};

// ---------------------------------------------------------------- timing (ms) and keys
export const TIMING = {
  wait: 3000,         // fixation only; SPACE starts the round, or it starts by itself after this
  pull: 400,          // lever animation before the symbols appear
  response: 4000,     // symbols on screen until a key press or this deadline
  feedback: 2000,     // feedback circles
};

// Response keys (either case). Which one means YES ("this pair wins") is randomised per
// participant (from the seed), so left/right hand is not confounded with the answer.
export const KEYS = { pair: ['f', 'j'], start: ' ' };

// Show "press SPACE" under the machine during these phases
export const SPACE_HINT_PHASES = ['practice'];

// Feedback: circles around the symbols in one of these colours (no text). Late answers cost
// coins like wrong ones, so they are red too by default.
export const FEEDBACK = { correct: '#2ecc40', wrong: '#ff4136', late: '#ff4136' };

// ---------------------------------------------------------------- coins and bonus
// Coins per trial; only coins from `bonusPhases` blocks are paid. The bonus is paid on top
// of the base pay set on Prolific (as a Prolific bonus payment, see README).
export const REWARD = {
  correct: 10,
  wrong: -5,
  late: -5,
  bonusPhases: ['test'],
  coinsPerDollar: 500,
  minBonus: 0,            // dollars; the bonus never goes below this
  maxBonus: null,         // dollars, or null for no cap
};

// Block-end grades: shown for proportion correct below each bound
export const GRADES = [
  [0.55, 'Insufficient', 'The task seems to have been quite difficult.\nPlease review the instructions carefully.'],
  [0.60, 'Fair', 'Your performance was modest.\nWith a bit more focus and effort, you can definitely improve.'],
  [0.70, 'Good', "Solid performance!\nYou're getting the hang of it - keep up the consistency."],
  [0.80, 'Excellent', 'Outstanding work!\nYour performance is among the best.\nStay sharp and keep going strong!'],
  [Infinity, 'Perfect', 'Flawless performance!\nYou mastered this block.\nKeep up this exceptional work!'],
];

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

// Where data go at the end: 'local' downloads a CSV (pilots). Add a backend in data.js
// (DataPipe, JATOS or Pavlovia) before running on Prolific.
export const DATA = { save: 'local', filePrefix: 'catlearn_online' };

// URL options (for piloting): ?debug=1 shortens every block to 1 pass (practice: 3 passes),
// ?simulate=1 (or =visual) lets jsPsych play the whole experiment, ?seed=123 fixes the design,
// ?skip=intro skips consent, instructions and the quiz, ?skip=calibration the screen setup.
