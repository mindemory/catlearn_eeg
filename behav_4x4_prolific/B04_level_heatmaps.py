"""Accuracy and F1 per symbol level and pair over trials in the test blocks, as heatmaps.

Rows: the levels a1-a4 (left symbol) and b1-b4 (right symbol), and the 16 pairs a1b1 ...
a4b4; each labelled with the rule's share of F pairs (levels) or its category (pairs).
Levels are positions in the design (which fractal plays a1 is random), so rows line up
across participants of one version.

For each trial of a block, a row's value pools that row's trials in a centred window of
--window trials (default 48, three passes: ~12 trials per level, ~3 per pair):
  accuracy   share correct (late answers count as wrong)            levels and pairs
  F1         category 1 (F) positive, late answers count as not F   levels only (a pair is
             always F or always J, so its F1 says nothing); undefined (grey) for a level
             with no F pair
Each block draws new fractals and has its own rule, so a row is a position in that
block's design, labelled per block; rows don't follow a fractal from one block to the next.
Averages: the mean over the participants of each version who are not excluded
(load_data.exclusions, key bias) and have complete test blocks.

Outputs, in ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/analysis/:
  <participant>/level_heatmaps.png, level_heatmaps.csv
  average_version_<A|B>/level_heatmaps.png, level_heatmaps.csv

  ~/miniforge3/envs/kernelbehav/bin/python B04_level_heatmaps.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from B01_learning_curves import block_name, test_blocks  # noqa: E402
from load_data import OUT_DIR, load_trials, save_exclusions  # noqa: E402
from style import apply_dark_theme  # noqa: E402

N_LEVELS = 8          # a1-a4, b1-b4; then the 16 pairs


def rows(b):
    """(label, mask) for every level and pair of a block."""
    cat = b.groupby("compound")["category"].first()
    out = []
    for i in range(4):
        out.append((f"a{i + 1} ({int(cat[[4 * i + j for j in range(4)]].sum())}/4 F)", b["level_a"].eq(i)))
    for j in range(4):
        out.append((f"b{j + 1} ({int(cat[[4 * i + j for i in range(4)]].sum())}/4 F)", b["level_b"].eq(j)))
    for i in range(4):
        for j in range(4):
            c = 4 * i + j
            out.append((f"a{i + 1}b{j + 1} ({'F' if cat[c] == 1 else 'J'})", b["compound"].eq(c)))
    return out


def block_heat(b, window):
    """{'accuracy': (24, trials), 'f1': (8, trials)} and the row labels."""
    roll = lambda s: s.rolling(window, center=True, min_periods=1).sum().to_numpy()   # noqa: E731
    f = b["choice"].eq(1).astype(float)
    pos = b["category"].eq(1).astype(float)
    ok = b["correct"].astype(float)
    labels, acc, f1 = [], [], []
    for k, (label, mask) in enumerate(rows(b)):
        m = mask.astype(float)
        n = roll(m)
        with np.errstate(all="ignore"):
            acc.append(np.where(n > 0, roll(ok * m) / n, np.nan))
            if k < N_LEVELS:
                tp, fp, fn = roll(f * pos * m), roll(f * (1 - pos) * m), roll((1 - f) * pos * m)
                d = 2 * tp + fp + fn
                has_f = (pos * m).any()                  # a level with no F pair has no positive class
                f1.append(np.where((d > 0) & has_f, 2 * tp / d, np.nan))
        labels.append(label)
    return {"accuracy": np.array(acc), "f1": np.array(f1)}, labels


def participant_heat(r, window):
    return {k: (rule, *block_heat(b, window)) for k, rule, b in test_blocks(r)}


def plot(data, title, window, path):
    blocks = list(data)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#3a3a3a")
    fig, axes = plt.subplots(2, len(blocks), figsize=(4.6 * len(blocks) + 1.2, 10.5), squeeze=False,
                             gridspec_kw={"height_ratios": [24, 8]}, constrained_layout=True)
    for k, blk in enumerate(blocks):
        rule, heat, labels = data[blk]
        for r, (key, lab) in enumerate((("accuracy", "Accuracy"), ("f1", "F1 (levels)"))):
            ax = axes[r, k]
            vals = heat[key]
            labs = labels[: vals.shape[0]]
            im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=1, aspect="auto",
                           interpolation="nearest", extent=(0.5, vals.shape[1] + 0.5, len(labs) - 0.5, -0.5))
            ax.set_yticks(range(len(labs)), labs, fontsize=7)      # every block: its own rule's labels
            for y in (3.5, 7.5)[: 2 if len(labs) > N_LEVELS else 1]:
                ax.axhline(y, color="0.9", lw=0.8)
            if r == 0:
                ax.set_title(block_name(blk, rule), fontsize=10)
            else:
                ax.set_xlabel("Trial")
            if k == 0:
                ax.set_ylabel(lab)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label=f"0.5 = chance ({window}-trial window)")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def records(data, who):
    out = []
    for blk, (rule, heat, labels) in data.items():
        for key, vals in heat.items():
            for r in range(vals.shape[0]):
                out += [{"participant": who, "test_block": blk, "rule": rule, "metric": key, "row": labels[r],
                         "trial": i + 1, "value": v} for i, v in enumerate(vals[r])]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=48, help="trials in the centred moving window")
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
        plot(data, f"Participant {pid[:8]} (version {version}){note}", args.window, out / "level_heatmaps.png")
        pd.DataFrame(records(data, pid)).to_csv(out / "level_heatmaps.csv", index=False)
        print(f"{pid} (version {version}) -> {out}/level_heatmaps.png")
        per.append((pid, version, data))

    for version in sorted({v for _, v, _ in per}):
        group = [(pid, d) for pid, v, d in per if v == version and pid not in excluded]
        size = max(sum(h["accuracy"].shape[1] for _, h, _ in d.values()) for _, d in group)
        group = [(pid, d) for pid, d in group if sum(h["accuracy"].shape[1] for _, h, _ in d.values()) == size]
        if len(group) < 2:
            continue
        avg = {}
        for blk, (rule, _, labels) in group[0][1].items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # cells undefined for everyone stay NaN
                avg[blk] = (rule, {key: np.nanmean(np.stack([d[blk][1][key] for _, d in group]), 0)
                                   for key in ("accuracy", "f1")}, labels)
        out = OUT_DIR / f"average_version_{version}"
        out.mkdir(parents=True, exist_ok=True)
        plot(avg, f"Mean of {len(group)} participants (version {version})", args.window, out / "level_heatmaps.png")
        pd.DataFrame(records(avg, f"mean_version_{version}")).to_csv(out / "level_heatmaps.csv", index=False)
        print(f"average of {len(group)} (version {version}) -> {out}/level_heatmaps.png")


if __name__ == "__main__":
    main()
