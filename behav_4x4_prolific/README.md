# behav_4x4_prolific

Behavioral analyses of the online slot-machine task
([`../catlearn_4x4_prolific`](../catlearn_4x4_prolific)).

- **Data:** sessions upload to `~/My Drive/DataPipe/catlearn_4x4/` (DataPipe). The scripts
  first copy new or changed files from there into
  `~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/`, then analyse the
  `catlearn_online_*.csv` files in that folder.
  - **`--no-sync`** skips the copy.
  - **Archived files:** anything archived in `old_versions/` there is not copied again.
  - **Unfinished sessions** (`.partial.json` files) are listed but not analysed.
- **Output:** the `analysis/` subfolder there, one folder per participant plus tables for
  all participants

Run with the kernelbehav env, in order:

```bash
~/miniforge3/envs/kernelbehav/bin/python B01_learning_curves.py
```

```bash
~/miniforge3/envs/kernelbehav/bin/python B02_task_modes.py
```

| File | What it does |
|---|---|
| `load_data.py` | reads every data file into a per-round table, adds the signal-detection outcomes, and builds a per-participant session table |
| `B01_learning_curves.py` | per block: running accuracy, hit / miss / false-alarm rates and F1, plus RT per round. Saves `learning_curves.png`, `block_summary.csv`, `block_summary_all.csv` and `sessions.csv` |
| `B02_task_modes.py` | kernel-framework decomposition of the choices into constant / A / B / A×B modes, per block and over windows within blocks. Saves `task_modes.png`, `task_modes_time.png`, `task_modes.csv` and `task_modes_all.csv` |
| `style.py` | the dark plot theme (`behav_analyses/plot_style.py`) under its own module name |

Signal-detection measures:

- **Signal and answer:** a **win** pair is the signal, and **YES** is the positive answer.
- **Hit rate:** P(YES | win). **Miss rate:** 1 − hit rate.
- **False-alarm rate:** P(YES | lose).
- **F1:** 2·hits / (2·hits + misses + false alarms).
- **Late rounds:** they contain no YES, so they count as misses (win pairs) or correct
  rejections (lose pairs). They are also reported separately as the late rate.

Mode decomposition (as in `kernel_model/kernel_modes.py`):

- **Choice function:** y(s) = 2·P(YES | s) − 1 over the pairs, leaving out late rounds.
- **Share:** how y's power splits across A, B and A×B, compared with the rule's own split
  (its kernel loadings).
- **Progress:** the learned fraction of the rule's component in each mode (1 = learned,
  0 = none, < 0 = reversed).
- **Off-rule power:** power in a mode the rule doesn't use means choices follow a feature
  that doesn't matter, e.g. the old side after the A → B switch.
