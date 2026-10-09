"""Accuracy and F1 per symbol level and pair over trials in the test blocks, as heatmaps.

Rows: the levels a1-a4 (left symbol) and b1-b4 (right symbol), and the 16 pairs a1b1 ...
a4b4; each labelled with the rule's share of F pairs (levels) or its category (pairs).
Levels are positions in the design (which fractal plays a1 is random), so rows line up
across participants of one version. Each block draws new fractals and has its own rule, so a
row is a position in that block's design, labelled per block; rows don't follow a fractal
from one block to the next.

For each trial of a block, a row's value pools that row's trials in a centred window of
--window trials (default params.LEVEL_WINDOW = 48, three passes: ~12 trials per level, ~3 per
pair):
  accuracy   share correct (late answers count as wrong)            levels and pairs
  F1         category 1 (F) positive, late answers count as not F   levels only (a pair is
             always F or always J, so its F1 says nothing); undefined (grey) for a level
             with no F pair
Averages: the mean over the kept participants of each version with complete test blocks.

Outputs, in analysis/:
  <participant>/level_heatmaps.png, level_heatmaps.csv
  average_version_<A|B>/level_heatmaps.png, level_heatmaps.csv

  ~/miniforge3/envs/kernelbehav/bin/python B04_level_heatmaps.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.measures import N_LEVELS, block_name, level_heat, test_blocks
from helpers.plots import accuracy_cmap, apply_dark_theme, heatmap, save
from helpers.stats import nanmean
from helpers.study import Study, parser
from params import LEVEL_WINDOW, VERSIONS


def participant_heat(r, window):
    return {k: (rule, *level_heat(b, window)) for k, rule, b in test_blocks(r)}


def plot(data, title, window, path):
    blocks = list(data)
    fig, axes = plt.subplots(2, len(blocks), figsize=(4.6 * len(blocks) + 1.2, 10.5), squeeze=False,
                             gridspec_kw={"height_ratios": [24, 8]}, constrained_layout=True)
    for k, blk in enumerate(blocks):
        rule, heat, labels = data[blk]
        for r, (key, lab) in enumerate((("accuracy", "Accuracy"), ("f1", "F1 (levels)"))):
            ax = axes[r, k]
            vals = heat[key]
            labs = labels[: vals.shape[0]]
            im = heatmap(ax, vals, accuracy_cmap(), (0, 1), labs, fontsize=7)   # every block: its own rule's labels
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
    save(fig, path, dpi=130)


def records(data, who):
    return [{"participant": who, "test_block": blk, "rule": rule, "metric": key, "row": labels[r], "trial": i + 1,
             "value": v}
            for blk, (rule, heat, labels) in data.items() for key, vals in heat.items()
            for r in range(vals.shape[0]) for i, v in enumerate(vals[r])]


def main():
    p = parser(__doc__)
    p.add_argument("--window", type=int, default=LEVEL_WINDOW, help="trials in the centred moving window")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    excluded = study.save_exclusions()

    per = {}
    for pid, version, r in study.participants():
        data = participant_heat(r, args.window)
        out = study.out(pid)
        note = " - excluded" if pid in excluded else ""
        plot(data, f"Participant {pid[:8]} (version {version}){note}", args.window, out / "level_heatmaps.png")
        pd.DataFrame(records(data, pid)).to_csv(out / "level_heatmaps.csv", index=False)
        print(f"{pid} (version {version}) -> {out}/level_heatmaps.png")
        per[pid] = data

    groups = study.groups()
    for version in VERSIONS:
        group = groups[version]
        if len(group) < 2:
            continue
        avg = {blk: (rule, {key: nanmean(np.stack([per[p][blk][1][key] for p in group]), 0)
                            for key in ("accuracy", "f1")}, labels)
               for blk, (rule, _, labels) in per[group[0]].items()}
        out = study.out(f"average_version_{version}")
        plot(avg, f"Mean of {len(group)} participants (version {version})", args.window, out / "level_heatmaps.png")
        pd.DataFrame(records(avg, f"mean_version_{version}")).to_csv(out / "level_heatmaps.csv", index=False)
        print(f"average of {len(group)} (version {version}) -> {out}/level_heatmaps.png")


if __name__ == "__main__":
    main()
