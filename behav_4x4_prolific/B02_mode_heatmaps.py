"""Accuracy and F1 per task mode (A, B, AB) over trials in the test blocks, as heatmaps.

In a centred moving window of --window trials (default params.WINDOW = 16, as in B01) the
participant's choice function over the 16 pairs is y(s) = 2 P(F | s) - 1 (late trials left
out; a pair without an answered trial in the window counts as 0). The rule's target is
y*(s) = +1 for category-1 (F) pairs, -1 for J pairs. Both are projected onto each mode with the
projectors of kernel_model/kernel_modes.py (Design('4x4')): P_A y, P_B y, P_AB y.

Per mode, over the pairs where the rule's component P_e y* is not 0 (helpers.measures):
  accuracy   share of pairs where sign(P_e y) == sign(P_e y*) (a 0 counts as half)
  F1         the rule's positive side of the mode as the positive class
A mode the rule doesn't use (B in version A's VI and X, A in version B's) is grey. A
participant who learned the whole rule scores 1 in every mode the rule uses.

Outputs, in analysis/:
  <participant>/mode_heatmaps.png, mode_heatmaps.csv
  average_version_<A|B>/mode_heatmaps.png, mode_heatmaps.csv (mean over kept participants with
      complete test blocks)

  ~/miniforge3/envs/kernelbehav/bin/python B02_mode_heatmaps.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.measures import block_name, mode_heat, test_blocks
from helpers.plots import accuracy_cmap, apply_dark_theme, heatmap, save
from helpers.stats import nanmean
from helpers.study import Study, parser
from params import MODES, VERSIONS, WINDOW

METRICS = {"accuracy": "Accuracy", "f1": "F1"}


def participant_heat(r, window):
    return {k: (rule, mode_heat(b, window)) for k, rule, b in test_blocks(r)}


def plot(data, title, path):
    """Rows: accuracy, F1; columns: test blocks; each heatmap: modes x trials."""
    blocks = list(data)
    fig, axes = plt.subplots(len(METRICS), len(blocks), figsize=(4.6 * len(blocks) + 0.8, 4.2), squeeze=False,
                             constrained_layout=True)
    for k, blk in enumerate(blocks):
        rule, heat = data[blk]
        for r, (key, label) in enumerate(METRICS.items()):
            ax = axes[r, k]
            im = heatmap(ax, heat[key], accuracy_cmap(), (0, 1), MODES if k == 0 else [""] * len(MODES), fontsize=10)
            if r == 0:
                ax.set_title(block_name(blk, rule), fontsize=10)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Trial")
            if k == 0:
                ax.set_ylabel(label)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="0.5 = chance")
    fig.suptitle(title, fontsize=11)
    save(fig, path)


def records(data, who):
    return [{"participant": who, "test_block": blk, "rule": rule, "metric": key, "mode": m, "trial": i + 1, "value": v}
            for blk, (rule, heat) in data.items() for key in METRICS for r, m in enumerate(MODES)
            for i, v in enumerate(heat[key][r])]


def main():
    p = parser(__doc__)
    p.add_argument("--window", type=int, default=WINDOW, help="trials in the centred moving window")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    excluded = study.save_exclusions()

    per = {}
    for pid, version, r in study.participants():
        data = participant_heat(r, args.window)
        out = study.out(pid)
        note = " - excluded" if pid in excluded else ""
        plot(data, f"Participant {pid[:8]} (version {version}){note}", out / "mode_heatmaps.png")
        pd.DataFrame(records(data, pid)).to_csv(out / "mode_heatmaps.csv", index=False)
        print(f"{pid} (version {version}) -> {out}/mode_heatmaps.png")
        per[pid] = data

    groups = study.groups()
    for version in VERSIONS:
        group = groups[version]
        if len(group) < 2:
            continue
        first = per[group[0]]
        avg = {blk: (rule, {k: nanmean(np.stack([per[p][blk][1][k] for p in group]), 0) for k in METRICS})
               for blk, (rule, _) in first.items()}
        out = study.out(f"average_version_{version}")
        plot(avg, f"Mean of {len(group)} participants (version {version})", out / "mode_heatmaps.png")
        pd.DataFrame(records(avg, f"mean_version_{version}")).to_csv(out / "mode_heatmaps.csv", index=False)
        print(f"average of {len(group)} (version {version}) -> {out}/mode_heatmaps.png")


if __name__ == "__main__":
    main()
