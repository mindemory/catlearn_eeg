# behav_noise_discrim

Analysis of the same/different noise study
([`../noise_discrim_prolific`](../noise_discrim_prolific)).

- **Data:** uploads land in `~/My Drive/DataPipe/noise_discrim_thresholding/` (DataPipe).
  `D01_dprime.py` first copies new or changed files from there into
  `~/Documents/data/catlearn_eeg/noise_discrim_prolific/` (`--no-sync` skips this), then
  analyses everything in that folder.
  - **Kinds of file:** `noise_discrim_*.csv` are finished sessions. `*.partial.json` are the
    staged trials of sessions that never finished, used only when there is no CSV for that
    session.
  - **Debug runs** are skipped unless `--include-debug`.
- **Output:** the `analysis/` subfolder there

```bash
~/miniforge3/envs/kernelbehav/bin/python D01_dprime.py
```

| File | What it does |
|---|---|
| `load_data.py` | copies new files from the Drive folder, and reads every data file (CSV or partial JSON) into a per-trial table (signal-detection columns, exposure check) and a per-participant table |
| `D01_dprime.py` | hit and false-alarm rates, d' and c at every alpha, per participant and for the group, plus exclusion flags. Practice trials are left out. Saves `dprime_by_alpha.csv`, `participants.csv` and `dprime_group.png`, and per participant `dprime.png`, `curves.png` (accuracy and d' by alpha) and `learning.png` (accuracy by block; d' in the first vs second half of the blocks) |
| `style.py` | the dark plot theme (`behav_analyses/plot_style.py`) under its own module name |

**Measures.** A *different* pair is the signal and *different* is the positive answer.
Rates use the log-linear correction (k + 0.5) / (n + 1). Late trials are left out of the
rates, as are trials with an unknown exposure.

| Measure | What it is | Use it for |
|---|---|---|
| `d_yn` = z(H) − z(F) | model-free sensitivity | whether an alpha is at chance |
| `d_sd` | d' under the differencing model of same/different (Macmillan & Creelman, 2005) | the size of sensitivity; never below 0, so biased upward near chance |
| `c` = −(z(H) + z(F)) / 2 | bias: > 0 = tends to say "same" | see the caution below |

**Caution about c.** With a fixed criterion on how different the two patches look, yes/no
c still changes with d'. So a c that changes with alpha does not by itself mean that
participants moved their criterion; see the test below.

**Tested** on synthetic observers with a known d'(alpha) and fixed differencing criteria:
- `d_sd` recovers the true d' from about 1 upward;
- `d_yn` reads about 0.1–0.3 near chance;
- yes/no c falls with alpha although the observers' criteria were fixed.
