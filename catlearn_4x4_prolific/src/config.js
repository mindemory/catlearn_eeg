// Every design choice of the online experiment lives here. Nothing else needs editing to
// change blocks, rules, timing, keys, sizes or the Prolific / data settings.
//
// Design: two symbols (fractals) appear, one left (A) and one right (B) of the centre. Every
// pair belongs to one of two categories; participants learn to press F for one category and
// J for the other (the same keys for everyone), from feedback after every answer. A
// performance bonus (BONUS) is paid for accuracy above chance in the test blocks; there are
// no points or coins on screen. Every block has new fractals.
//   Practice (2x2: 2 fractals per side): type I (A or B, random), then XOR; each runs until
//     the participant is accurate enough (PRACTICE_CRITERION).
//   Test (4x4: 4 fractals per side): three blocks, type VI -> type X -> type II (SEQUENCES).
//     Half the participants get the A versions (A carries the main effect), half the B
//     versions; type II weighs A and B equally, so its block is the same for both.
// Each trial: fixation (TIMING.iti), then the pair with no fixation (participants look freely)
// until F / J or TIMING.response, then feedback.

export const VERSION = '1.1.0';   // the first study: F/J categorisation, VI -> X -> II, free viewing.
                                  // 1.0.1: practice criterion 9 of the last 10 (was 8).
                                  // 1.1: post-task questionnaire; type-I practice criterion 11 of
                                  // the last 12. The earlier slot-machine pilot builds saved
                                  // task_version 3.x.

// ---------------------------------------------------------------- rules
// Category (1 = F, 0 = J) of every pair: compound = size * a + b, where a and b (0 .. size-1)
// say which fractal is shown left (A) and right (B). Generated from the task definitions in
// kernel_model/kernel_modes.py (Design('2x2'), Design('4x4')), the same ones as in the
// task-type figures. Within a block the fractals are assigned to levels at random.
export const RULES = {
  x2_I_A: { type: 'I', size: 2, labels: [1, 1, 0, 0] },                      // A
  x2_I_B: { type: 'I', size: 2, labels: [1, 0, 1, 0] },                      // B
  x2_XOR: { type: 'XOR', size: 2, labels: [1, 0, 0, 1] },                    // AB
  x4_II: { type: 'II', size: 4, labels: [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] },     // A+B+AB (no A/B version)
  x4_VI_A: { type: 'VI', size: 4, labels: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0] },   // A+AB
  x4_VI_B: { type: 'VI', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0] },   // B+AB
  x4_X_A: { type: 'X', size: 4, labels: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] },     // A+AB
  x4_X_B: { type: 'X', size: 4, labels: [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0] },     // B+AB
};

// Test sequences: each participant gets one version, A or B (at random, or ?version=A / B in
// the URL; for exact halves run one Prolific study per version). Blocks 'seq:1' .. 'seq:3'.
export const SEQUENCES = {
  A: ['x4_VI_A', 'x4_X_A', 'x4_II'],
  B: ['x4_VI_B', 'x4_X_B', 'x4_II'],
};

// ---------------------------------------------------------------- blocks
// One entry per block, in order. Fields:
//   phase      'practice' | 'test'
//   rule       a RULES key; a list of keys (one picked at random per participant); or
//              'seq:<k>' = the k-th rule of this participant's SEQUENCES version
//   reps       passes through all pairs (each pass in a new random order)
//   criterion  practice: end the block once PRACTICE_CRITERION is met (reps = the maximum)
export const BLOCKS = [
  { phase: 'practice', rule: ['x2_I_A', 'x2_I_B'], reps: 20, criterion: true },
  { phase: 'practice', rule: 'x2_XOR', reps: 20, criterion: true },
  { phase: 'test', rule: 'seq:1', reps: 13 },    // 208 rounds each
  { phase: 'test', rule: 'seq:2', reps: 13 },
  { phase: 'test', rule: 'seq:3', reps: 13 },
];

// Practice ends when at least `minCorrect` of the last `window` answers are correct, at the
// latest after `reps` passes (flagged in the data if never met). Per rule type: type I, the
// screening gate (SCREENING.practiceTypeI), is the strictest. Within 80 trials a guesser
// reaches 11 of 12 8% of the time (9 of 10: 25%, 8 of 10: 69%); a participant at 80% correct
// reaches it 99% of the time.
export const PRACTICE_CRITERION = {
  I: { window: 12, minCorrect: 11 },
  XOR: { window: 10, minCorrect: 9 },
};
export const practiceCriterion = (ruleType) => PRACTICE_CRITERION[ruleType] ?? PRACTICE_CRITERION.XOR;

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
  fixation: { outer: 0.6, inner: 0.2, line: 0.15 },   // bull's eye as in the MATLAB task, between rounds only
  outcomeRing: { size: 5.4, width: 0.3 },             // feedback circle around each symbol
  extent: { width: 19, height: 7 },                   // everything drawn, for the fit check
  background: '#808080',
};

// ---------------------------------------------------------------- timing (ms) and keys
export const TIMING = {
  iti: 1000,          // fixation only before each pair (no fixation while the pair is on)
  response: 4000,     // symbols on screen until a key press or this deadline
  feedback: 2000,     // feedback circles
};

// Response keys (either case): the same for every participant. cat1 answers category 1, cat0
// category 0 (RULES labels). start: continue from block screens.
export const KEYS = { cat1: 'f', cat0: 'j', start: ' ' };

// Feedback: circles around the symbols in one of these colours, no text
export const FEEDBACK = { correct: '#2ecc40', wrong: '#ff4136', late: '#ff4136' };

// ---------------------------------------------------------------- performance bonus
// Paid on top of the Prolific base pay, per block of `phases` (the 3 test blocks), from that
// block's proportion correct (late answers count as wrong): the amount of the highest tier
// reached. Up to $2 per block, $6 in all; no penalties. Participants are told the rule in
// the instructions and see each block's bonus on its break screen; nothing during rounds.
export const BONUS = {
  phases: ['test'],
  tiers: [            // [minimum proportion correct, dollars for the block]
    [0.5, 0.5],
    [0.6, 1.0],
    [0.7, 2.0],
  ],
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
  minHeight: 500,          // (the layout is only 7 deg tall; 650 excluded ordinary laptop windows)
                           // Excluded participants leave through the screen-out path (main.js)
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

// Post-task questionnaire (src/questionnaire.js): after the last test block, before the
// bonus screen, ~4 min. ?skip=questionnaire leaves it out of a run.
export const QUESTIONNAIRE = { enabled: true, minutes: 4 };

export const COMPREHENSION = { maxAttempts: 2 };   // quiz attempts; failing all of them screens out (SCREENING)

// Custom screening (Prolific "custom screening" completion path): participants who are not a
// good fit leave early with a fixed payment (payUsd, set on Prolific too) and do not use up a
// place. Following Prolific's attention and comprehension check policy:
//   quiz             the comprehension quiz failed COMPREHENSION.maxAttempts times (~5 min in)
//   practiceTypeI    the type-I practice (one side decides) not passed within its rounds
//   practiceTimeouts more than this share of practice rounds unanswered (checked after each
//                    practice block)
// The XOR practice is not a screen: it is genuinely hard, and screening on it would select
// good learners. Debug and simulated runs skip screening unless the URL has ?screen=1.
export const SCREENING = { quiz: true, practiceTypeI: true, practiceTimeouts: 0.2, payUsd: 1.0 };

// One Prolific study per test version: the same code and page, with &version=A or &version=B
// in each study's URL. Every Prolific study has its own codes, so they are listed per version
// and picked by the participant's version (from the URL).
export const PROLIFIC = {
  // B is a duplicate of A's study, so Prolific kept A's codes (a code is checked against the
  // participant's own study, so sharing them is fine)
  completionCodes: { A: 'CCP8AO6I', B: 'CCP8AO6I' },        // "Completion paths" on each study's page
  screenOutCodes: { A: 'CASM8Y0V', B: 'CASM8Y0V' },         // custom screening path
  noConsentCodes: { A: 'REPLACE_WITH_NO_CONSENT_CODE', B: 'REPLACE_WITH_NO_CONSENT_CODE' },   // optional "returned" path
  completeUrl: 'https://app.prolific.com/submissions/complete?cc=',
};

export const CONTACT = {
  name: 'Mrugank Dake, PhD',
  address: 'HB 6207, Moore Hall, Hanover, NH 03755',
  email: 'mrugank.dake@dartmouth.edu',
};

// Where data go:
//   'local'     the CSV downloads to the participant's computer (piloting on your own machine)
//   'datapipe'  DataPipe (pipe.jspsych.org, experiment catlearn_4x4) through its jsPsych
//               extension, into the linked Google Drive folder: trials are staged as the session
//               runs, so a participant who quits partway still leaves a <file>.partial.json (it
//               does not count as a session), and the whole CSV is uploaded at the end. If the
//               final upload fails, a copy downloads and the participant is asked to email it.
// File names carry no participant label: catlearn_online_<start time>_<random tag>.csv (the
// Prolific ID is inside the data). Simulated runs (?simulate=1) always save locally; ?save=local
// does the same for any run (testing the hosted page without adding a session).
export const DATA = { save: 'datapipe', filePrefix: 'catlearn_online', datapipeId: 'PbuWRFfUHDfo' };

// URL options (for piloting): ?debug=1 shortens every block to 1 pass (practice: 3 passes),
// ?simulate=1 (or =visual) lets jsPsych play the whole experiment, ?seed=123 fixes the design,
// ?version=A or B fixes the test sequence,
// ?skip=intro skips consent, instructions and the quiz, ?skip=calibration the screen setup.
