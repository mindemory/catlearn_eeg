# behav_4x4_prolific

Behavioral analyses of the F/J category-learning study
([`../catlearn_4x4_prolific`](../catlearn_4x4_prolific), task_version 1.x). Every session of
the study (both versions, all Prolific studies) is analysed as one dataset.

## Layout

| Path | What it is |
|---|---|
| `params.py` | every parameter: paths, which sessions count, exclusion rules, moving windows, statistics, colours |
| `helpers/` | the shared code (below) |
| `B01`–`B12` | one script per analysis; each reads `params.py`, uses `helpers/`, and writes into `analysis/` |
| `ANALYSIS_PLAN.md` | the checklist of analyses planned after Peng, Ehrlich, Lee & Murray (2025), with what is done |

| Helper | What it holds |
|---|---|
| `helpers/data.py` | the Drive sync, loading every kept session into one row per trial, exclusions |
| `helpers/study.py` | `Study`: one run's trials, exclusions, version groups (kept participants with complete test blocks) and output folders; `parser()` for the shared `--no-sync` option |
| `helpers/measures.py` | accuracy, F1, RT, screening checks, per-mode and per-level/pair measures, and the A ↔ B alignment |
| `helpers/sessions.py` | session details, how each session ended, the questionnaire |
| `helpers/stats.py` | permutation tests (seeded), Benjamini–Hochberg, mean ± SEM |
| `helpers/plots.py` | the dark theme (`../behav_analyses/plot_style.py`) and the shared plot pieces |
| `helpers/report.py` | markdown tables |
| `helpers/features.py` | the fractals' colour, shape and DreamSim similarity per block, and its alignment with the rule's modes |
| `helpers/kernel.py` | the kernel learner (Peng et al. 2025): likelihood (JAX), simulation, maximum-likelihood fits per participant and per group |

## Data

Everything lives in `~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/`:

| Folder or file | What it holds |
|---|---|
| `data/` | every finished session (`catlearn_online_*.csv`), copied from the DataPipe Drive folder (`~/My Drive/DataPipe/catlearn_4x4/`) at every run; `--no-sync` skips that. Unfinished sessions (`.partial.json`) stay on Drive only. `data/old_versions/` holds files the sync never brings back (the slot-machine builds, task_version 3.x). |
| `analysis/` | figures and tables (below) |
| `exclusions_manual.csv` | participants excluded by hand, with the reason and date |
| `prolific/` | Prolific's demographic export(s) for B05, saved by hand (Submissions page, "Download demographic data") |
| `bonus/` | the Prolific bonus list from `../catlearn_4x4_prolific/tools/bonus_payments.py`, and `paid.csv`, the ledger of bonuses already paid |

Sessions are kept if they are task_version 1.x, not debug runs, and have at least one trial.
A participant with several sessions (e.g. someone who took both the A and the B study) is
analysed on the first only: the later ones come from someone who already knew the task. The
loader lists what it leaves out.

**Exclusions** (`helpers.data.exclusions`), left out of every average and marked in each
participant's own figures:
- **key bias** (automatic): P(F) over the answered test trials outside 0.25–0.75
  (`params.BIAS_LIMITS`);
- **rushing** (automatic): in some test block at least 20% of the answers faster than 250 ms
  with that block at chance, and the whole test at chance too ("at chance": accuracy not above
  0.5, one-sided binomial p ≥ .05; `params.RUSH_FAST_SHARE`, `RUSH_CHANCE_ALPHA`). Someone who
  rushed one block but learned the others is kept;
- **manual**: participants listed in `exclusions_manual.csv` (participant, reason,
  decided_on). It sits with the data, not in this repository, because it holds Prolific IDs.
  Add a row to exclude someone by hand.

`analysis/exclusions.csv` lists every participant with version, P(F), whether they are
excluded, by which rule, why, and when a manual exclusion was decided.

## Running

With the kernelbehav env, from this folder. The first script syncs from Drive; the rest can
skip it:

```bash
~/miniforge3/envs/kernelbehav/bin/python B01_learning_curves.py
```

```bash
for s in B02_mode_heatmaps B03_screening B04_level_heatmaps B05_participants B06_sessions B07_a_vs_b B08_levels_a_vs_b B09_sanity_a_vs_b B10_questionnaire B11_feature_alignment; do ~/miniforge3/envs/kernelbehav/bin/python $s.py --no-sync; done
```

| Script | What it does | Outputs (in `analysis/`) |
|---|---|---|
| `B01_learning_curves.py` | accuracy, F1 and RT over trials per test block; early vs late | `<participant>/`, `average_version_<A\|B>/` |
| `B02_mode_heatmaps.py` | accuracy and F1 per task mode (A, B, AB) over trials | `<participant>/`, `average_version_<A\|B>/` |
| `B03_screening.py` | key bias, late and fast answers, key repetition; P(F) per level and pair | `average_version_<A\|B>/sanity_tests/` |
| `B04_level_heatmaps.py` | accuracy per level and pair, F1 per level, over trials | `<participant>/`, `average_version_<A\|B>/` |
| `B05_participants.py` | one row per participant: device, calibration, time per block and break, practice, accuracy, bonus, demographics | `average_version_<A\|B>/sanity_tests/` |
| `B06_sessions.py` | every session and how it ended (finished, screened out and why, unfinished, test run) | `sessions.csv`, `sessions.md` |
| `B07_a_vs_b.py` | version A vs B: accuracy and per-mode accuracy, the II side index | `a_vs_b/` |
| `B08_levels_a_vs_b.py` | version A vs B per level and pair, with B's sides swapped in VI and X | `a_vs_b/` |
| `B09_sanity_a_vs_b.py` | version A vs B on accuracy, F1, RT, response habits and session measures | `a_vs_b/` |
| `B10_questionnaire.py` | the questionnaire: ratings, strategies, free text, and how they match the data | `questionnaire/` |
| `B11_feature_alignment.py` | whether the fractals' colour / shape similarity, aligned with each mode of the rule, predicts learning of that mode (per test type, versions pooled) | `features/` |
| `B12_kernel_recovery.py` | parameter recovery for the kernel learner (model 1: mode weights per block, learning rate, choice sensitivity) on synthetic participants running real sessions | `kernel_recovery/` |

Every script's docstring describes its measures in full. The main ones in brief:

**Learning curves (B01).** Centred moving window of 16 trials (one pass through the 16
pairs). Accuracy counts late answers as wrong; F1 takes category 1 (the F key) as positive
and late answers as not F; RT is the mean of the answered trials. Averages are trial by trial
with ± 1 SEM bands, over kept participants with complete test blocks. Early vs late: the
first and last 48 trials of each block.

**Modes (B02, B07).** In each window the participant's choice function over the 16 pairs,
y = 2·P(F) − 1, and the rule's y* = ±1 are projected onto the modes of
`kernel_model/kernel_modes.py`: A (left symbol), B (right symbol), AB (their interaction).
Mode accuracy is the share of pairs where the signs agree; a mode the rule doesn't use is
grey.

**Levels and pairs (B04, B08).** Rows a1–a4 (left symbol), b1–b4 (right symbol) and the 16
pairs, labelled by each block's rule. Pairs that stay wrong show which simpler rule a
participant is using.

**Version A vs B (B07–B09).** The B rules of VI and X are the A rules with left and right
swapped; II is the same for both and symmetric. So in VI and X, A's mode A is compared with
B's mode B (and B's levels are shown in A's terms); in II any A − B pattern reflects what
was learned before. Differences of per-participant means are tested by permuting the version
labels (p uncorrected unless stated; B08 also gives BH-adjusted q). A − B heatmaps use their
own diverging scale (orange: A higher).

**Kernel model (B12, `helpers/kernel.py`).** The learner keeps a value per pair, reset at each
new block; P(F) = sigmoid(β·value); after feedback the shown pair's value moves toward the
correct answer by the learning rate, and the kernel K = Σₑ (16 wₑ / dₑ) Pₑ spreads that to the
other pairs by mode (cst, A, B, AB). Recovery (B12, run on its own since it doesn't depend on
new data): single participants' parameters are not recoverable (learning rate and β trade off),
but parameters shared by a group of about 20 are, including the II A:B balance (r ≈ 0.9), so
the model is fit per version.
