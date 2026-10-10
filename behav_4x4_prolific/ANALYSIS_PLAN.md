# Analysis plan: 4x4 study, following Peng, Ehrlich, Lee & Murray (2025)

A checklist for this study's analyses, based on Hui's thesis chapter / bioRxiv paper on
inductive-bias adaptation (`~/Documents/Readings/Hui_s Thesis - PengMurray(InductiveBaisAdaptation_bioArxiv2025).pdf`).
`[x]` = done (with the script), `[~]` = partly done, `[ ]` = open.

## How the paper maps onto this study

| | Peng et al. (2025) | This study |
|---|---|---|
| Stimulus | 3 features: T top (3 values), A and B left and right (2 values each); 12 conditions | 2 features: A left, B right (4 values each); 16 pairs |
| Blocks | 3 × 360 trials, new fractals each block | 3 × 208 trials (13 passes), new fractals each block |
| Manipulation | Blocks 1–2: T informative (group T) or B informative (group B); block 3: task X, T and B equally informative | VI → X: A informative (version A) or B (version B, the mirror image); block 3: II, A and B equally informative |
| Key result | In block 3 each group learns the previously informative feature faster | Each version ahead on its previously relevant side in II (B07 side index) |

What our design adds:
- [ ] **Mirror groups.** A and B are identical except for side, so a plain left/right preference can be separated from history (their T differs from A/B in place and number of values, which they note makes it more salient).
- [ ] **A large interaction.** AB has 9 dimensions here (their 3–4), so learning combinations matters far more.
- [ ] **Free viewing.** Where people look is a direct readout of attention (see Phase 4).

## Phase 1: repeat their behavioural analyses (descriptive)

- [~] **1.1 Accuracy per pair in II by version, with "exceptions".** Mark each II pair that a left-only or a right-only rule gets wrong. Prediction: version A worse on the left-only exceptions, version B on the right-only ones. Test with a logistic mixed model, correct ~ version × pair + (1 | participant). Their Fig. 3B–C.
  - done: per-pair grids and per-row tests (`B08_levels_a_vs_b.py`: `pair_grids.png`, `levels_tests.csv`)
  - open: the exception labels and the mixed model
- [~] **1.2 Response weights per mode.** Their squared projection of the choice pattern onto each mode, per pass of 16 trials; the same quantity the kernel model fits.
  - done: mode accuracy (sign agreement) over trials (`B02_mode_heatmaps.py`, `B07_a_vs_b.py`)
  - open: squared projections ("response weights") per pass, per version
- [~] **1.3 The same for VI and X.** Exceptions to the relevant side (their Fig. S2).
  - done: A vs B with B's sides swapped (`B07`, `B08`); no difference beyond a small left tilt in X
  - open: exception labels as in 1.1
- [ ] **1.4 Learning across blocks.** Learning speed in II compared with VI (e.g. trials to a criterion, or slope of the mode curves), per version.

## Phase 2: the kernel model (the core of the paper)

- [~] **2.1 Parameter recovery first.** Simulate learners with known weights through the exact design (16 pairs, 208 trials, 13 passes) and refit them. Which parameters can the data pin down (especially the 9-dimensional AB weight), and how many participants are needed?
  - model 1 (weights cst/A/B/AB per block, shared lr and beta; maximum likelihood): `helpers/kernel.py`, `B12_kernel_recovery.py`
  - open: model 2 (each mode split into the rule's direction and the rest), and a group-level power check for the version difference
- [ ] **2.2 Their model.** Mode weights (constant, A, B, AB) at the start and end of each block, per version, with a shared learning rate and decision noise; Bayesian (PyMC), hierarchical for per-participant weights.
- [ ] **2.3 Model comparison** (WAIC or cross-validation):
  - [ ] fixed: one kernel per block
  - [ ] linear drift: weights move linearly from start to end of a block (theirs)
  - [ ] adaptive: weights move toward the current response weights at rate α (as in their network); α = a per-person "meta-learning rate"
  - [ ] adaptive with partial reset: at a new block the weights drift back toward neutral by ρ (hinted at in their Fig. S3D)
  - [ ] side bias: a shared left/right preference plus a version-specific history term (only the mirror design can separate these)
- [ ] **2.4 Key numbers:**
  - [ ] the A:B weight ratio at the start of II by version (history) and at its end (convergence)
  - [ ] kernel–task alignment within each block
  - [ ] carry-over from the end of X to the start of II
- [ ] **2.5 Predictions.** Simulate alternative third blocks (an A-only block, an AB-only block) from the fitted model; the paper predicts slower adaptation for interactions than for single features. Pick the most informative block for the next study.

## Phase 3: beyond the paper

- [ ] **3.1 Individual learning curves** (the margin question). Step-change or hidden-state models on each person's mode curves: is single-feature learning sudden (finding a rule) and interaction learning gradual? An RT drop at the step would support insight.
- [ ] **3.2 Memorising exceptions vs generalising interactions.** Is AB learning spread over all the relevant pairs (kernel-like) or concentrated on a few memorised pairs (exemplar-like)? When is each pair learned?
- [~] **3.3 Colour as a hidden feature.** Participants say they sorted by colour, and colour drives the fractals' perceived similarity.
  - done: step 1, participant level, whether colour/shape similarity aligned with the rule predicts mode learning (`B11_feature_alignment.py`): no relationship so far
  - open: step 1b, within each person, pair by pair: do errors on a pair lean toward the answers of its colour-similar pairs?
  - open: a stimulus-similarity term in the kernel model (2.x), so colour similarity isn't mistaken for feature generalisation (the same confound would affect EEG decoding)
- [~] **3.4 Individual differences.** Per-person weights and α vs the questionnaire (reported strategy, Need for Cognition) and practice speed.
  - done: questionnaire by version, ratings vs accuracy, reported strategy vs data (`B10_questionnaire.py`)
  - open: against the fitted weights (needs Phase 2)
- [ ] **3.5 A free natural experiment.** The 2×2 type-I practice randomly makes the left or the right symbol relevant. Does that small dose shift where VI starts?
- [ ] **3.6 Yoking** (weaker with two features): do people with a strong single-feature weight also have a strong AB weight?

## Phase 4: mechanistic hypotheses for the gaze / EEG study

| Hypothesis | Gaze (free viewing) | EEG |
|---|---|---|
| [ ] **H1.** Attention acts as gain on a feature and follows the fitted weights, lagging behind performance (slow α) | Share of looking time on the left vs right symbol, during the stimulus and the feedback: shifts toward the relevant side, follows error reduction, and favours the previously relevant side at the start of II | Alpha lateralization or N2pc toward the attended side (needs fixation: covert attention) |
| [ ] **H2.** Learning credit goes to the attended feature | Where people look at feedback predicts which mode updates more | Feedback-locked FRN / P3 carry feature-specific prediction errors |
| [ ] **H3.** The brain's representational "kernel" aligns with the task | | Covariance of EEG patterns across the 16 pairs, projected onto the modes: the A or B weight rises in its own version, AB less |
| [ ] **H4.** Partial reset at a new block | Gaze bias at the start of a block; pupil (eye tracker only) | EEG attention weights at block starts |
| [ ] **H5.** Sudden insight | Sudden gaze shifts at learning steps | Frontal theta or P3b at the step |

- [ ] **The strongest single test:** a kernel model whose trial-by-trial weights come from gaze, against the same model without gaze. If gaze improves the fit, attention is setting the generalisation.

## Design decisions for the next study

- [ ] **Webcam gaze online first?** The symbols are 12° apart and webcam gaze (2–4°) can tell left from right: tests H1 and H4 cheaply, same task, free viewing. Then the lab EEG study (fixation, EyeLink) tests H1–H5 with the same model.
- [ ] **Fixation for EEG.** Central fixation with the symbols at ±6°; free viewing would need fixation-locked EEG, which is much harder.
- [ ] **A fourth block in the lab**, chosen from the 2.5 simulations.
- [ ] **Sample size and trials** from simulations of the fitted model, not rules of thumb. 13 trials per pair per block is too few for pair-level EEG patterns: pool at single-symbol values (~52 trials per block) or modes.

## Order of work

1. [ ] Phase 1 and the 2.1 recovery check (cheap, current data).
2. [ ] Group-level kernel fits and model comparison (now that each version has 15–20 kept participants).
3. [ ] Phase 3, and the gaze / EEG predictions written down in advance from the fitted models.

## Caveats to keep in mind

- Fewer participants than the paper (theirs ~55 per group): group-level, hierarchical fits are realistic; precise individual fits need more.
- Fewer repetitions (13 per pair per block vs their 30): pair-level analyses stay noisy.

## Done so far (outside the paper's list)

- [x] Data pipeline, one dataset, exclusions with reasons (`params.py`, `helpers/`, `exclusions_manual.csv`; repeat sessions of a participant are left out)
- [x] Learning curves, mode and level heatmaps, screening (`B01`–`B04`)
- [x] Participant details, session outcomes and screen-out reasons (`B05`, `B06`)
- [x] Version A vs B: modes, levels and pairs, sanity measures (`B07`–`B09`)
- [x] Questionnaire (`B10`); colour/shape alignment step 1 (`B11`)
