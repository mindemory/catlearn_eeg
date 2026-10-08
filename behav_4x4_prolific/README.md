# behav_4x4_prolific

Behavioral analyses of the F/J category-learning study
([`../catlearn_4x4_prolific`](../catlearn_4x4_prolific), task_version 1.x).

Everything lives in `~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/`:

| Folder | What it holds |
|---|---|
| `data/` | finished sessions (`catlearn_online_*.csv`), copied from the DataPipe Drive folder (`~/My Drive/DataPipe/catlearn_4x4/`) at every run; `--no-sync` skips that. Unfinished sessions (`.partial.json`) stay on Drive only. |
| `analysis/` | figures and tables: `<participant>/`, `average_version_<A|B>/`, `exclusions.csv` |
| `prolific/` | Prolific's demographic export(s) for B05, saved by hand (Submissions page, "Download demographic data") |
| `bonus/` | the Prolific bonus list from `../catlearn_4x4_prolific/tools/bonus_payments.py` |

Run with the kernelbehav env:

```bash
~/miniforge3/envs/kernelbehav/bin/python B01_learning_curves.py
```

```bash
~/miniforge3/envs/kernelbehav/bin/python B02_mode_heatmaps.py --no-sync
```

```bash
~/miniforge3/envs/kernelbehav/bin/python B03_screening.py --no-sync
```

```bash
~/miniforge3/envs/kernelbehav/bin/python B04_level_heatmaps.py --no-sync
```

```bash
~/miniforge3/envs/kernelbehav/bin/python B05_participants.py --no-sync
```

| File | What it does |
|---|---|
| `load_data.py` | copies new files from Drive and reads every finished session into one row per trial; see below for what is left out |
| `B01_learning_curves.py` | learning curves for the test blocks: accuracy, F1 and RT, per participant and averaged per version; early vs late |
| `B02_mode_heatmaps.py` | accuracy and F1 per task mode (A, B, AB) over trials, as heatmaps, per participant and averaged per version |
| `B03_screening.py` | screening metrics over participants (box plots, average folder only): key bias, late and fast answers, key repetition, and P(F) per level and pair against the rule |
| `B04_level_heatmaps.py` | accuracy per level (a1–a4, b1–b4) and per pair over trials, and F1 per level, as heatmaps, per participant and averaged per version |
| `B05_participants.py` | per-participant details (average folder, `sanity_tests/`): device, calibration, quiz attempts, time per block and break, practice, attention, accuracy, bonus, and demographics from a Prolific export |
| `style.py` | the dark plot theme (`behav_analyses/plot_style.py`) under its own module name |

`load_data.py` leaves out, and lists:
- sessions with no trials (e.g. excluded by the browser check);
- debug runs;
- the slot-machine pilot builds (task_version 3.x).

## B01: learning curves

Each test block's trials in order, with a centred moving window of 16 trials
(`--window`; one pass through the 16 pairs). Test blocks are numbered 1–3, the session's
blocks 3–5 after the two practice blocks.

- **Accuracy:** proportion correct. Late answers count as wrong.
- **F1:** 2·TP / (2·TP + FP + FN). Category 1, the F key, is the positive class; late
  answers count as not F.
- **RT:** the mean RT of the answered trials in the window.

`learning_curves.png` has one column per test block: accuracy (mango) and F1 (violet) on
0–1 in the top row, RT (sky blue) in the bottom row, on one range for all blocks.

**Average:** the mean over the participants of each version, trial by trial, with ± 1 SEM
bands. Only participants with complete test blocks are included.

**Early vs late:** accuracy, F1 and RT over the first and the last 48 trials of each test
block (`--edge`). The figure uses standard seaborn box plots over participants (box: 25th
to 75th percentile, line: median, whiskers: seaborn's default), and the dashed lines join
each participant's early and late values.

**Outputs:**
- `learning_curves.png` and `learning_curves.csv` for each participant;
- the same for each average, plus `early_late.png`, `early_late.csv` and
  `participants.txt` (who is in it).

## B02: accuracy and F1 per task mode

**How it's computed:**
1. **Choice function.** In the same centred 16-trial window as B01, the choices over the 16
   pairs give y(s) = 2·P(F | s) − 1. Late trials are left out, and a pair with no answered
   trial in the window counts as 0.
2. **Target.** The rule gives y*(s) = +1 for F pairs and −1 for J pairs.
3. **Projection.** Both are projected onto modes A, B and AB with the projectors of
   `kernel_model/kernel_modes.py`.
4. **Scores per mode**, over the pairs where the rule's component isn't 0:
   - **Accuracy:** the share of pairs where the signs of the choices' component and the
     rule's component agree (a 0 counts as half).
   - **F1:** the rule's positive side of the mode is the positive class.

**Reading it:** a participant who has learned the whole rule scores 1 in every mode the rule
uses. A mode the rule doesn't use (B in types VI and X) has no target and is grey.

**Figure** (`mode_heatmaps.png`): rows are accuracy and F1, columns are test blocks, and
each heatmap is modes × trials. The colour map is RdBu_r from 0 to 1, with 0.5 at chance.

**Average:** the cell-wise mean over participants with complete test blocks.
`mode_heatmaps.csv` has every value.

## B03: screening metrics

These are box plots over participants, saved in `average_version_<A|B>/sanity_tests/`. In
`screening.png`, each participant is a set of dots joined by a faint dashed line. Each is computed per test block and over all three ("All"):

| Metric | Expected | Flags |
|---|---|---|
| P(F): share of answered trials pressed F | ≈ 0.5 (every rule has 8 F and 8 J pairs) | key bias |
| Late: no answer within 4 s | ≈ 0 | disengagement |
| Fast: RT < 250 ms (`--fast`) | ≈ 0 | anticipating or mashing keys |
| P(same key as the previous answer) | ≈ 0.5 | perseverating (high) or alternating (low) |

These four are in `screening.png` and `screening.csv`.

`associations.png` and `associations.csv` show P(F) per level (a1–a4, b1–b4) and per pair,
for each test block. Each is set against the rule's share of F pairs for that level, or the
pair's category, drawn as a white diamond.

## B04: accuracy per level and pair

The same idea as B02, with the design's levels and pairs as rows instead of modes: a1–a4
(left symbol), b1–b4 (right symbol), then the 16 pairs a1b1 … a4b4. Each row is labelled
with the rule's share of F pairs (levels) or the pair's category (pairs). Each block draws
new fractals and has its own rule, so the labels are per block, and a row doesn't follow a
fractal from one block to the next.

The window is wider than in B01 and B02: a centred 48 trials (`--window`, three passes),
so a level has about 12 trials in it and a pair about 3. With 16 trials a pair would appear
about once, and its cell would be just right or wrong.

- **Accuracy** (top), for levels and pairs: the share correct, with late answers counted as
  wrong.
- **F1** (bottom), for levels only, with F as the positive class. A pair is always F or
  always J, so its F1 says nothing. A level with no F pair (e.g. a4 at 0/4 F) has no F1 and
  is grey.

**Reading it:** pairs that stay blue are the exceptions to whatever simpler rule the
participant is using. For example, a participant who answers by "a1 or b1 → F" in type II
gets every pair right except a2b2.

**Outputs:** `level_heatmaps.png` and `level_heatmaps.csv`, per participant and in each
`average_version_<A|B>/` (kept participants with complete test blocks, as in B01).

## B05: participant details

One row per participant in `average_version_<A|B>/sanity_tests/participants.csv`, and the
same as readable tables in `participants.md`:

- **Session:** task version, start time (UTC), browser, OS, screen size, refresh rate.
- **Setup and attention:** viewing distance, calibration attempts, px per degree, quiz
  attempts, window/tab switches, fullscreen exits.
- **Time (minutes):** setup (start to the first block), each practice and test block (end of
  its intro to its last trial), each break (last trial of the previous block to the end of
  the next intro), and the total.
- **Practice:** trials to criterion and accuracy per practice block.
- **Test and bonus:** accuracy per test block and overall, whether overall accuracy is above
  chance (one-sided binomial, p < .05; information only, not an exclusion), P(F), the
  key-bias exclusion, and the bonus the task promised per block and in total.
- **Demographics:** from any Prolific export CSV in `prolific/`, merged on the participant
  ID. Without one, these columns are left out.

`timing.png` shows the minutes per part of the session as box plots, with each participant
as dashed lines (red: excluded).
