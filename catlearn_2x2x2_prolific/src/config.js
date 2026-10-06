// Every design choice of the online experiment lives here. Nothing else needs editing to
// change blocks, rules, timing, keys, sizes or the Prolific / data settings.
//
// Design: a slot machine shows symbols at up to three positions, the corners of an
// equilateral triangle around fixation: A bottom left, B bottom right, C top. A rule says
// which combinations win (1) and which lose (0); participants predict win or lose and learn
// from the payout. Correct predictions earn coins, wrong or late ones cost coins, and
// test-block coins become a bonus. Every block is a new machine with new symbols.
//   Practice (2x2: A and B only, a pair of fractals each): type I (A or B, random), then XOR;
//     each runs until the participant is accurate enough (PRACTICE_CRITERION).
//   Test (2x2x2: A, B and C, a pair of noise patches each): two blocks of type III, version
//     A+B+AC+BC, for everyone; the second is a new machine (new patches) with the same rule.
//     Each ends at TEST_CRITERION or after its last round. The noise patches' spectral slope
//     (alpha) is set in STIMULI.noise: for now everyone gets alpha 2.00.
// Each trial is self-paced: the participant presses SPACE to pull the lever (or the round
// starts by itself after TIMING.wait), the symbols appear, they answer, feedback follows.

export const VERSION = '2.0.0';

// ---------------------------------------------------------------- rules
// Outcome (1 = win, 0 = lose) of every combination. `levels` = levels per feature (A, B, C
// in order); combination index = levels in mixed radix with A slowest: 2x2 -> 2a + b,
// 2x2x2 -> 4a + 2b + c. Generated from the task definitions in kernel_model/kernel_modes.py
// (Design('2x2'), Design('2x2x2')), the same ones as in the task-type figures. Within a
// block the fractals are assigned to levels at random.
export const RULES = {
  x2_I_A: { type: 'I', levels: [2, 2], labels: [1, 1, 0, 0] },                   // A
  x2_I_B: { type: 'I', levels: [2, 2], labels: [1, 0, 1, 0] },                   // B
  x2_XOR: { type: 'XOR', levels: [2, 2], labels: [1, 0, 0, 1] },                 // AB
  // type III has three versions (by the two features with linear modes); the test uses this one
  x8_III_AB: { type: 'III', levels: [2, 2, 2], labels: [1, 1, 1, 0, 0, 1, 0, 0] },   // A+B+AC+BC
};

// ---------------------------------------------------------------- blocks
// Practice ends when at least `minCorrect` of the last `window` answers are correct (80%);
// at the latest after `reps` passes (flagged in the data if never met)
export const PRACTICE_CRITERION = { window: 10, minCorrect: 8 };

// The test ends early once `minCorrect` of the last `window` answers are correct (92%). In
// this rule any one or two features alone give 75% correct, so the criterion must be well
// above 75%: in simulations, 22 of 24 is reached by no guesser and by 2% of participants who
// know only A, but by everyone who knows the whole rule (even with 10% lapses).
// creditRemaining: the rounds left are paid as correct, so learning fast never costs bonus.
export const TEST_CRITERION = { window: 24, minCorrect: 22, creditRemaining: true };

// One entry per block, in order. Fields:
//   phase      'practice' | 'test'
//   rule       a RULES key, or a list of keys (one picked at random per participant)
//   stimuli    'fractals' | 'noise' (STIMULI)
//   reps       passes through all combinations (each pass in a new random order)
//   criterion  end the block once this criterion is met (reps = the maximum)
export const BLOCKS = [
  { phase: 'practice', rule: ['x2_I_A', 'x2_I_B'], stimuli: 'fractals', reps: 20, criterion: PRACTICE_CRITERION },
  { phase: 'practice', rule: 'x2_XOR', stimuli: 'fractals', reps: 20, criterion: PRACTICE_CRITERION },
  // two machines with the same rule and new patches, up to 152 rounds each (304 in all)
  { phase: 'test', rule: 'x8_III_AB', stimuli: 'noise', reps: 19, criterion: TEST_CRITERION },
  { phase: 'test', rule: 'x8_III_AB', stimuli: 'noise', reps: 19, criterion: TEST_CRITERION },
];

// ---------------------------------------------------------------- stimuli
// Symbol sets. Each feature of a block shows one whole pair, never used again for this
// participant. imageScale: how much larger than LAYOUT.fractalSize the image is drawn.
//   fractals  matched in brightness, colourfulness and size; in every pair the DreamSim
//             distance is within 0.0025 of 0.256 (../fractal_stimuli, the 2x2x2 set: 48 pairs).
//             The PNGs have transparent margins: see LAYOUT.fractalSize.
//   noise     1/f^alpha noise patches in a circular aperture, the image filling fractalSize.
//             A participant sees 6 pairs (2 test blocks x 3 positions).
//             Higher alpha = smoother blobs = easier to tell apart. In every pair the DreamSim
//             distance is within 0.005 of that alpha's median (30 pairs per alpha;
//             tools/make_noise_stimuli.py). Pairs: stimuli/noise/groups.js.
//             alphas: the alphas participants are assigned to (one each, at random). For now
//             only 2.00; stimuli for 1.50, 1.75 and 2.25 exist too, so adding them here makes
//             alpha a between-participant condition again.
export const STIMULI = {
  fractals: { dir: 'stimuli/fractal_groups/', imageScale: 500 / 368 },
  noise: { dir: 'stimuli/noise/', imageScale: 1, alphas: ['2.00'] },
};

// Layout in degrees of visual angle (dva). x to the right, y up, (0, 0) = the fixation point
// at the screen centre. The three positions are the corners of an equilateral triangle
// centred on fixation, each 6 deg from it (6 cos 30 = 5.196); practice blocks use A and B.
// Pixels per degree come from the calibration (calibration.js); if the layout is larger
// than the window, everything shrinks by the same factor to fit (layout_scale).
export const LAYOUT = {
  positions: { A: [-5.196, -3], B: [5.196, -3], C: [0, 6] },
  fractalSize: 4,                                     // symbol size (noise: aperture diameter)
  // The fractal PNGs (500 px) have transparent margins. In the EEG task's fractals the square
  // enclosing every fractal is 368 px (as cropped by the MATLAB task); drawing the image
  // 500/368 larger makes that square fractalSize degrees. The new fractals have the same
  // median extent and area as those, so the same scale shows them at the same size (the
  // spikiest reach 4.3 deg across).
  fractalImageScale: 500 / 368,
  fixation: { outer: 0.6, inner: 0.2, line: 0.15 },   // bull's eye as in the MATLAB task
  machine: { center: [0, 1.5], size: [19, 16.5], border: 0.5, color: '#c9a227' },
  lever: { length: 3.5, knob: 1 },                    // on the right of the machine
  outcomeRing: { size: 5.4, width: 0.3 },             // feedback circle around each symbol
  hintY: -7.8,                                        // "press SPACE" hint below the machine
  extent: { width: 24, height: 21 },                  // everything drawn (centred on fixation), for the fit check
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
  fitMargin: 0.85,                // the layout may fill at most this fraction of the window (leaves the
                                  // top free for the coin bar: the machine reaches high above fixation)
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
export const DATA = { save: 'local', filePrefix: 'catlearn_2x2x2_online' };

// URL options (for piloting): ?debug=1 shortens every block to 1 pass (3 for blocks with a criterion),
// ?simulate=1 (or =visual) lets jsPsych play the whole experiment, ?seed=123 fixes the design,
// ?alpha=2.00 fixes the noise alpha (one of STIMULI.noise.alphas),
// ?skip=intro skips consent, instructions and the quiz, ?skip=calibration the screen setup.
