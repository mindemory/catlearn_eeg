"""Task-mode decomposition of choices, as in the kernel framework (kernel_model/kernel_modes.py).

For each block (and, with enough rounds, each window of --passes passes within a block)
the participant's choice function over the size x size pairs is
    y(s) = 2 P(YES | s) - 1                 (late rounds left out)
and the rule's target is y*(s) = +1 for win pairs, -1 for lose pairs. Both are split into
the design's modes (constant, A, B, AB) with the projectors P_e of Design('2x2' / '4x4'):
  share     y' P_e y / sum over A, B, AB    how the choice function's stimulus-dependent
                                            power splits across modes (the rule has its own
                                            split, the task's "loadings")
  progress  y . P_e y* / (y*' P_e y*)       learned fraction of the rule's component in mode
                                            e: 1 = fully learned, 0 = none, < 0 = reversed
                                            (the same measure as simulate_blocks' progress)
Off-rule power (share in a mode the rule does not use) means responses follow a feature
that does not matter, e.g. the old rule's side after the A -> B switch.

Outputs per participant in ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/analysis/<participant>/:
  task_modes.png        per block: choice P(YES) per pair next to the rule's win map; mode
                        shares of choices vs rule; progress per mode
  task_modes_time.png   progress per mode across windows within blocks (when blocks have
                        at least 2 windows)
  task_modes.csv        every block / window: share and progress per mode
and task_modes_all.csv for all participants.

  ~/miniforge3/envs/kernelbehav/bin/python B02_task_modes.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "kernel_model"))
from B01_learning_curves import rule_label  # noqa: E402
from kernel_modes import Design  # noqa: E402
from load_data import DATA_DIR, OUT_DIR, load_all  # noqa: E402
from style import apply_dark_theme  # noqa: E402

MODE_COLORS = {"A": "#ffa630", "B": "#c77dff", "AB": "#2ec4b6"}   # as in the kernel / design figures
DESIGNS = {2: Design("2x2"), 4: Design("4x4")}


def choice_function(rounds, size):
    """y(s) = 2 P(YES | s) - 1 per pair (NaN where a pair has no answered round)."""
    answered = rounds[~rounds["timeout"]]
    p = answered.groupby("compound")["yes"].mean().reindex(range(size * size))
    return 2 * p.to_numpy() - 1, answered.groupby("compound").size().reindex(range(size * size), fill_value=0)


def target(rounds, size):
    lab = rounds.groupby("compound")["category"].first().reindex(range(size * size))
    return 2 * lab.to_numpy() - 1


def decompose(y, ystar, design):
    """Mode shares of y and of the target (among the stimulus-dependent modes) and progress."""
    P = design.projectors
    names = design.mode_names
    y = np.nan_to_num(y)                      # unseen pairs count as indifferent (P(YES) = 0.5)
    power = np.einsum("n,enm,m->e", y, P, y)
    tpower = np.einsum("n,enm,m->e", ystar, P, ystar)
    stim = [i for i, n in enumerate(names) if n != "cst"]
    out = {}
    for i, n in enumerate(names):
        out[f"power_{n}"] = power[i]
        if n == "cst":
            continue
        out[f"share_{n}"] = power[i] / power[stim].sum() if power[stim].sum() > 0 else np.nan
        out[f"rule_share_{n}"] = tpower[i] / tpower[stim].sum()
        out[f"progress_{n}"] = (y @ P[i] @ ystar) / tpower[i] if tpower[i] > 1e-9 else np.nan
    out["power_total"] = power.sum()
    return out


def block_table(r, passes):
    """Decomposition for every block (window = 'all') and every window of `passes` passes."""
    rows = []
    for (pid, blk), b in r.groupby(["participant", "block"]):
        size = int(b["size"].iloc[0])
        design = DESIGNS[size]
        ystar = target(b, size)
        info = {"participant": pid, "block": blk, "phase": b["phase"].iloc[0], "rule": b["rule"].iloc[0], "size": size}
        y, _ = choice_function(b, size)
        rows.append({**info, "window": "all", "first_round": int(b["trial_in_block"].min()),
                     "last_round": int(b["trial_in_block"].max()), **decompose(y, ystar, design)})
        n_pairs = size * size
        win = passes * n_pairs
        n_windows = int(np.ceil(len(b) / win))
        if n_windows >= 2:
            for w in range(n_windows):
                part = b[(b["trial_in_block"] > w * win) & (b["trial_in_block"] <= (w + 1) * win)]
                if len(part) < n_pairs:          # too few rounds to cover the pairs
                    continue
                y, _ = choice_function(part, size)
                rows.append({**info, "window": w + 1, "first_round": int(part["trial_in_block"].min()),
                             "last_round": int(part["trial_in_block"].max()), **decompose(y, ystar, design)})
    return pd.DataFrame(rows)


def pair_grid(values, size):
    return np.asarray(values, dtype=float).reshape(size, size)   # rows: level of A, columns: level of B


def plot_blocks(r, table, session, out_dir):
    blocks = sorted(r["block"].unique())
    fig, axes = plt.subplots(4, len(blocks), figsize=(3.3 * len(blocks) + 1, 12),
                             gridspec_kw={"height_ratios": [1, 1, 1.1, 1.1]}, squeeze=False)
    for j, blk in enumerate(blocks):
        b = r[r["block"] == blk]
        size = int(b["size"].iloc[0])
        t = table[(table["block"] == blk) & (table["window"] == "all")].iloc[0]
        y, counts = choice_function(b, size)
        p_yes = (y + 1) / 2
        ystar = target(b, size)

        ax = axes[0, j]
        im = ax.imshow(pair_grid((ystar + 1) / 2, size), cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"block {blk}: {rule_label(t['rule'])}\nrule: green = win", fontsize=9)
        ax = axes[1, j]
        im = ax.imshow(pair_grid(p_yes, size), cmap="RdYlGn", vmin=0, vmax=1)
        for (a, bb), c in np.ndenumerate(pair_grid(counts.to_numpy(), size)):
            ax.text(bb, a, f"{int(c)}", ha="center", va="center", fontsize=6, color="black")
        ax.set_title("choices: P(YES)  (n per pair)", fontsize=9)
        for ax in axes[:2, j]:
            ax.set_xticks(range(size), [f"b{k + 1}" for k in range(size)], fontsize=7)
            ax.set_yticks(range(size), [f"a{k + 1}" for k in range(size)], fontsize=7)

        modes = ["A", "B", "AB"]
        x = np.arange(len(modes))
        ax = axes[2, j]
        ax.bar(x - 0.18, [t[f"rule_share_{m}"] for m in modes], 0.34, color="none",
               edgecolor=[MODE_COLORS[m] for m in modes], lw=1.5, label="rule")
        ax.bar(x + 0.18, [t[f"share_{m}"] for m in modes], 0.34, color=[MODE_COLORS[m] for m in modes],
               label="choices")
        ax.set_xticks(x, modes)
        ax.set_ylim(0, 1.05)
        if j == 0:
            ax.set_ylabel("share of stimulus-dependent power")
            ax.legend(frameon=False, fontsize=7, loc="upper right")

        ax = axes[3, j]
        prog = [t[f"progress_{m}"] for m in modes]
        ax.bar(x, np.nan_to_num(prog), 0.5, color=[MODE_COLORS[m] for m in modes])
        for k, v in enumerate(prog):
            if np.isnan(v):
                ax.text(k, 0.03, "not in\nrule", ha="center", va="bottom", fontsize=7, color="0.6")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhline(1, color="0.4", lw=0.8, ls=":")
        ax.set_xticks(x, modes)
        ax.set_ylim(-1.1, 1.1)
        if j == 0:
            ax.set_ylabel("progress: learned fraction\nof the rule in each mode")
    fig.colorbar(im, ax=axes[:2, :].ravel().tolist(), fraction=0.015, pad=0.01, label="P(YES) / win")
    title = f"{session['participant']}: task-mode decomposition of choices (kernel framework)"
    if session["debug"]:
        title += "   [DEBUG RUN: ~1 round per pair, estimates very noisy]"
    fig.suptitle(title, fontsize=10)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "task_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time(table, session, out_dir):
    t = table[table["window"] != "all"]
    if t.empty:
        return False
    blocks = sorted(t["block"].unique())
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.3 * len(blocks) + 1, 3.4), sharey=True, squeeze=False)
    for j, blk in enumerate(blocks):
        ax = axes[0, j]
        b = t[t["block"] == blk]
        x = (b["first_round"] + b["last_round"]) / 2
        for m in ("A", "B", "AB"):
            if b[f"progress_{m}"].notna().any():
                ax.plot(x, b[f"progress_{m}"], marker="o", color=MODE_COLORS[m], label=m)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhline(1, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"block {blk}: {rule_label(b['rule'].iloc[0])}", fontsize=9)
        ax.set_xlabel("round in block")
    axes[0, 0].set_ylabel("progress per mode")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{session['participant']}: learning in each mode over the block", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "task_modes_time.png", dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-sync", action="store_true", help="skip copying new files from the DataPipe Drive folder")
    parser.add_argument("--passes", type=int, default=2, help="passes through the pairs per time window")
    args = parser.parse_args()
    apply_dark_theme()

    rounds, sessions = load_all(args.data_dir, sync=not args.no_sync)
    table = block_table(rounds, args.passes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "task_modes_all.csv", index=False)
    for _, session in sessions.iterrows():
        pid = session["participant"]
        out = OUT_DIR / pid
        r, t = rounds[rounds["participant"] == pid], table[table["participant"] == pid]
        plot_blocks(r, t, session, out)
        timed = plot_time(t, session, out)
        t.to_csv(out / "task_modes.csv", index=False)
        cols = ["block", "rule"] + [f"{k}_{m}" for k in ("share", "rule_share", "progress") for m in ("A", "B", "AB")]
        print(f"{pid}{' (debug run)' if session['debug'] else ''}"
              f"{'' if timed else ' (no within-block windows: blocks too short)'}")
        print(t.loc[t["window"] == "all", cols].round(2).to_string(index=False))
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
