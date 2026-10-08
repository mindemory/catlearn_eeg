"""Accuracy and F1 per task mode (A, B, AB) over trials in the test blocks, as heatmaps.

In a centred moving window of --window trials (default 16, as in B01) the participant's
choice function over the 16 pairs is y(s) = 2 P(F | s) - 1 (late trials left out; a pair
without an answered trial in the window counts as 0). The rule's target is y*(s) = +1 for
category-1 (F) pairs, -1 for J pairs. Both are projected onto each mode with the projectors
of kernel_model/kernel_modes.py (Design('4x4')): P_A y, P_B y, P_AB y.

Per mode, over the pairs where the rule's component P_e y* is not 0:
  accuracy   share of pairs where sign(P_e y) == sign(P_e y*) (a 0 counts as half)
  F1         the rule's positive side of the mode as the positive class:
             TP = both > 0, FP = choices > 0 but rule < 0, FN = choices <= 0 but rule > 0
A mode the rule doesn't use (B in types VI and X) has no target: undefined (grey). A
participant who learned the whole rule scores 1 in every mode the rule uses.

Outputs, in ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/analysis/:
  <participant>/mode_heatmaps.png, mode_heatmaps.csv
  average_version_<A|B>/mode_heatmaps.png, mode_heatmaps.csv (mean over participants with
      complete test blocks who are not excluded: load_data.exclusions, key bias)

  ~/miniforge3/envs/kernelbehav/bin/python B02_mode_heatmaps.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kernel_model"))
from B01_learning_curves import block_name, test_blocks  # noqa: E402
from kernel_modes import Design  # noqa: E402
from load_data import OUT_DIR, load_trials, save_exclusions  # noqa: E402
from style import apply_dark_theme  # noqa: E402

DESIGN = Design("4x4")
MODES = ["A", "B", "AB"]
PROJ = {m: DESIGN.projectors[DESIGN.mode_names.index(m)] for m in MODES}
METRICS = {"accuracy": "Accuracy", "f1": "F1"}
EPS = 1e-9


def mode_scores(y, ystar):
    """Accuracy and F1 per mode of a choice function y against the target y* (16 pairs)."""
    out = {}
    for m in MODES:
        c, t = PROJ[m] @ y, PROJ[m] @ ystar
        use = np.abs(t) > EPS
        if not use.any():
            out[m] = {"accuracy": np.nan, "f1": np.nan}
            continue
        c, t = c[use], t[use]
        agree = np.where(np.abs(c) < EPS, 0.5, (np.sign(c) == np.sign(t)).astype(float))
        tp = np.sum((c > EPS) & (t > 0))
        fp = np.sum((c > EPS) & (t < 0))
        fn = np.sum((c <= EPS) & (t > 0))
        out[m] = {"accuracy": agree.mean(), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}
    return out


def block_heat(b, window):
    """{metric: array (modes, trials)} over centred windows of `window` trials."""
    ystar = 2.0 * b.groupby("compound")["category"].first().reindex(range(16)).to_numpy() - 1
    n = len(b)
    heat = {k: np.full((len(MODES), n), np.nan) for k in METRICS}
    half = window // 2
    for i in range(n):
        w = b.iloc[max(0, i - half):min(n, i - half + window)]
        w = w[~w["timeout"]]
        p = w.groupby("compound")["choice"].mean().reindex(range(16))
        y = np.nan_to_num(2 * p.to_numpy() - 1)
        scores = mode_scores(y, ystar)
        for r, m in enumerate(MODES):
            for k in METRICS:
                heat[k][r, i] = scores[m][k]
    return heat


def participant_heat(r, window):
    return {k: (rule, block_heat(b, window)) for k, rule, b in test_blocks(r)}


def plot(data, title, path):
    """Rows: accuracy, F1; columns: test blocks; each heatmap: modes x trials."""
    blocks = list(data)
    fig, axes = plt.subplots(len(METRICS), len(blocks), figsize=(4.6 * len(blocks) + 0.8, 4.2), squeeze=False,
                             constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#3a3a3a")
    for k, blk in enumerate(blocks):
        rule, heat = data[blk]
        for r, (key, label) in enumerate(METRICS.items()):
            ax = axes[r, k]
            vals = heat[key]
            im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=1, aspect="auto",
                           interpolation="nearest", extent=(0.5, vals.shape[1] + 0.5, len(MODES) - 0.5, -0.5))
            ax.set_yticks(range(len(MODES)), MODES if k == 0 else [""] * len(MODES))
            if r == 0:
                ax.set_title(block_name(blk, rule), fontsize=10)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Trial")
            if k == 0:
                ax.set_ylabel(label)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="0.5 = chance")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def records(data, who):
    rows = []
    for blk, (rule, heat) in data.items():
        for key in METRICS:
            for r, m in enumerate(MODES):
                for i, v in enumerate(heat[key][r]):
                    rows.append({"participant": who, "test_block": blk, "rule": rule, "metric": key, "mode": m,
                                 "trial": i + 1, "value": v})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=16, help="trials in the centred moving window")
    parser.add_argument("--no-sync", action="store_true", help="don't copy new files from the Drive folder")
    args = parser.parse_args()
    apply_dark_theme()

    trials = load_trials(sync=not args.no_sync)
    excluded = save_exclusions(trials)
    per = []
    for pid, r in trials.groupby("participant", sort=False):
        if r[r["phase"] == "test"].empty:
            continue
        data = participant_heat(r, args.window)
        version = r["version"].iloc[0]
        out = OUT_DIR / pid
        out.mkdir(parents=True, exist_ok=True)
        note = " - excluded (key bias)" if pid in excluded else ""
        plot(data, f"Participant {pid[:8]} (version {version}){note}", out / "mode_heatmaps.png")
        pd.DataFrame(records(data, pid)).to_csv(out / "mode_heatmaps.csv", index=False)
        print(f"{pid} (version {version}) -> {out}/mode_heatmaps.png")
        per.append((pid, version, data))

    for version in sorted({v for _, v, _ in per}):
        group = [(pid, d) for pid, v, d in per if v == version and pid not in excluded]
        shape = max(sum(h["accuracy"].shape[1] for _, h in d.values()) for _, d in group)
        group = [(pid, d) for pid, d in group if sum(h["accuracy"].shape[1] for _, h in d.values()) == shape]
        if len(group) < 2:
            continue
        avg = {}
        for blk in group[0][1]:
            # cells undefined for everyone (a mode not in the rule) stay NaN, without warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                avg[blk] = (group[0][1][blk][0],
                            {k: np.nanmean(np.stack([d[blk][1][k] for _, d in group]), 0) for k in METRICS})
        out = OUT_DIR / f"average_version_{version}"
        out.mkdir(parents=True, exist_ok=True)
        plot(avg, f"Mean of {len(group)} participants (version {version})", out / "mode_heatmaps.png")
        pd.DataFrame(records(avg, f"mean_version_{version}")).to_csv(out / "mode_heatmaps.csv", index=False)
        print(f"average of {len(group)} (version {version}) -> {out}/mode_heatmaps.png")


if __name__ == "__main__":
    main()
