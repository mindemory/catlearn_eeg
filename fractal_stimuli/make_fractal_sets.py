"""Fractal sets for any factorial design (2x2, 2x2x2, 2x2x2x2, 4x4, 3x2x2, ...).

A design such as 3x2x2 has 3 features (positions) with 3, 2 and 2 levels. In a block each
feature shows one fractal per level, so a block needs one group of 3 and two groups of 2
fractals. Within every group all fractals must be equally distinct (else one level is
easier to tell apart than another), and equally so in every group, block and design.
From the candidate pool (generate_fractals.py, distances from embed_dreamsim.py) this
script picks:

  groups   every pairwise DreamSim distance within a group is within --tol of one target
           distance (default: the median of all candidate pairs, 0.256), the same target
           for every group size and every design. Each group size uses the tightest
           tolerance (1/4, 1/2, 3/4 or all of --tol) at which enough groups fit
  spacing  no two fractals of the set (any groups) are closer than --d-min (default: the
           5th percentile of candidate pairs, 0.162), so there are no near-duplicates
  size     about --n fractals per design: n // (fractals per block) blocks' worth of groups

Search: all groups within --tol are listed (cliques of the "distance close to the target"
graph), then taken greedily, largest group size first, skipping any that shares a fractal
with, or comes closer than --d-min to, the fractals already chosen. --order conflict (the
default) takes first the groups whose fractals have the fewest close neighbours in the
pool, which fits far more groups (e.g. 34 rather than 24 groups of 4); --order deviation
takes the groups closest to the target first (how the online task's set was made).

Writes to ~/Documents/data/catlearn_eeg/fractal_sets/<design>/ (or --out):
  <n>.png          the fractals, renumbered 1..N (larger groups first)
  groups.json      {"design", "levels", "groups": {size: [[files]...]}, "blocks": [[group per
                   feature]...], "target_distance", ...}; blocks = the groups assigned to
                   blocks at random (seed --seed), feature by feature
  groups.csv       one row per fractal: file, group, size, block, feature, candidate, L*, C*
  groups.js        the groups as an ES module (FRACTAL_GROUPS = {size: [[files]...]}) for
                   web tasks
  contact_sheet.png  every block on the task's grey: its features left to right
  report.png       distances (all candidates, within groups, between fractals of the set)
                   and brightness / chroma of the set

  ~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/make_fractal_sets.py 2x2 2x2x2 2x2x2x2 4x4 3x2x2
  ~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/make_fractal_sets.py 4x4 --n 50
  # custom group counts (the online task: 18 groups of 4 and 12 pairs)
  ~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/make_fractal_sets.py --groups 4:18,2:12 \\
      --order deviation --name online_4x4 --out catlearn_4x4_prolific/stimuli/fractal_groups
"""

import argparse
import itertools
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DATA = Path.home() / "Documents" / "data" / "catlearn_eeg"
POOL = DATA / "fractal_pool"
SETS = DATA / "fractal_sets"
BACKGROUND = (128, 128, 128, 255)
TOL_STEPS = [0.25, 0.5, 0.75, 1.0]   # fractions of --tol tried, tightest first


# ---------------------------------------------------------------- groups
def find_groups(D, size, target, tol):
    """All groups of `size` fractals whose pairwise distances are all within tol of target,
    as (worst deviation, members) sorted by deviation."""
    close = np.abs(D - target) < tol
    np.fill_diagonal(close, False)
    if size == 2:
        iu = np.triu_indices(len(D), 1)
        dev = np.abs(D[iu] - target)
        return [(dev[o], (iu[0][o], iu[1][o])) for o in np.argsort(dev) if dev[o] < tol]
    groups = []

    def extend(members, candidates):
        if len(members) == size:
            dev = max(abs(D[a, b] - target) for a, b in itertools.combinations(members, 2))
            groups.append((dev, tuple(members)))
            return
        for c in candidates:
            extend(members + [c], candidates[close[c, candidates] & (candidates > c)])

    for i in range(len(D)):
        extend([i], np.flatnonzero(close[i, i + 1:]) + i + 1)
    groups.sort()
    return groups


def take_groups(candidates, n_wanted, D, d_min, chosen):
    """Greedy, in the given order: no shared fractals, all >= d_min from those chosen."""
    picked = []
    for _, g in candidates:
        if len(picked) == n_wanted:
            break
        if any(m in chosen for m in g):
            continue
        if chosen and D[np.ix_(list(g), sorted(chosen))].min() < d_min:
            continue
        picked.append(g)
        chosen.update(g)
    return picked


def select(D, counts, target, tol, d_min, order):
    """counts {size: number of groups} -> ({size: [groups of candidate indices]}, {size: tol used}).
    Each size uses the tightest tolerance (TOL_STEPS of tol) at which its groups still fit."""
    near = D < d_min
    np.fill_diagonal(near, False)
    n_near = near.sum(1)
    chosen, picked, tol_used = set(), {}, {}
    for size in sorted(counts, reverse=True):
        all_candidates = find_groups(D, size, target, tol)
        if order == "conflict":
            all_candidates.sort(key=lambda c: (n_near[list(c[1])].sum(), c[0]))
        for step in TOL_STEPS:
            trial = set(chosen)
            groups = take_groups([c for c in all_candidates if c[0] < step * tol], counts[size], D, d_min, trial)
            if len(groups) == counts[size]:
                break
        if len(groups) < counts[size]:
            raise SystemExit(f"only {len(groups)} of {counts[size]} groups of {size} fit "
                             f"({len(all_candidates)} candidates): loosen --tol or --d-min, ask for fewer, or "
                             "generate more candidate fractals")
        picked[size], tol_used[size] = groups, step * tol
        chosen = trial
    return picked, tol_used


# ---------------------------------------------------------------- output
def on_grey(path, tile):
    im = Image.open(path).convert("RGBA").resize((tile, tile), Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", (tile, tile), BACKGROUND)
    bg.alpha_composite(im)
    return bg


def contact_sheet(blocks, folder, path, tile=110, gap=12, cols=3, label="block"):
    """Each block as one labelled strip: its features' groups left to right, separated by gaps."""
    width = max(sum(len(g) for g in b) * tile + (len(b) - 1) * gap for b in blocks)
    rows = int(np.ceil(len(blocks) / cols))
    pad, head = 80, 34
    sheet = Image.new("RGBA", (cols * (width + pad) + pad, rows * (tile + head + 20) + 20), (18, 18, 18, 255))
    draw = ImageDraw.Draw(sheet)
    for bi, b in enumerate(blocks):
        x = pad + (bi % cols) * (width + pad)
        y = 20 + (bi // cols) * (tile + head + 20)
        draw.text((x, y), f"{label} {bi + 1}", fill=(220, 220, 220, 255), font_size=22)
        y += head
        for g in b:
            for f in g:
                sheet.alpha_composite(on_grey(folder / f"{f}.png", tile), (x, y))
                x += tile
            x += gap
    sheet.convert("RGB").save(path)


def report(D, groups, params, target, title, path):
    iu = np.triu_indices(len(D), 1)
    within = [D[a, b] for g in groups for a, b in itertools.combinations(g, 2)]
    member = {m: gi for gi, g in enumerate(groups) for m in g}
    sel = sorted(member)
    between = [D[a, b] for a, b in itertools.combinations(sel, 2) if member[a] != member[b]]
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": "#121212", "axes.facecolor": "#121212", "savefig.facecolor": "#121212"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.hist(D[iu], bins=80, density=True, color="0.5", label="all candidate pairs")
    ax.hist(between, bins=50, density=True, color="#ffa630", alpha=0.6, label="between groups of the set")
    ax.axvspan(min(within), max(within), color="#2ec4b6", alpha=0.8, label="within groups (range)")
    ax.axvline(target, color="white", lw=0.8, ls=":")
    ax.set_xlabel("DreamSim distance")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax = axes[1]
    p = params.set_index("k").loc[[m + 1 for m in sel]]
    ax.scatter(p["L_mean"], p["C_mean"], s=10, color="#c77dff")
    ax.set_xlabel("mean L* (background 53.6)")
    ax.set_ylabel("mean chroma")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return within, between


def write_set(name, levels, picked, tol_used, D, params, target, args, d_min, out):
    """Copy, renumber, assign groups to blocks, and write the tables and figures."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    file_of, files_by_size = {}, {}
    for size in sorted(picked, reverse=True):
        files_by_size[size] = []
        for g in picked[size]:
            for m in g:
                file_of[m] = len(file_of) + 1
                shutil.copy(args.pool / "candidates" / f"{m + 1}.png", out / f"{file_of[m]}.png")
            files_by_size[size].append([file_of[m] for m in g])

    # groups -> blocks: one group per feature, at random
    rng = np.random.default_rng(args.seed)
    blocks = []
    if levels:
        queue = {s: [files_by_size[s][i] for i in rng.permutation(len(files_by_size[s]))] for s in files_by_size}
        n_blocks = min(len(queue[s]) // levels.count(s) for s in set(levels))
        blocks = [[queue[s].pop() for s in levels] for _ in range(n_blocks)]

    groups = [g for s in sorted(picked, reverse=True) for g in picked[s]]
    rows = []
    where = {tuple(g): (bi + 1, fi + 1) for bi, b in enumerate(blocks) for fi, g in enumerate(b)}
    for gi, g in enumerate(groups):
        files = [file_of[m] for m in g]
        block, feature = where.get(tuple(files), (None, None))
        for m, f in zip(g, files):
            p = params.loc[params["k"] == m + 1].iloc[0]
            rows.append({"file": f, "group": gi + 1, "size": len(g), "block": block, "feature": feature,
                         "candidate": m + 1, "L_mean": p["L_mean"], "C_mean": p["C_mean"], "area": p["area"]})
    pd.DataFrame(rows).to_csv(out / "groups.csv", index=False)

    meta = {"design": name, "levels": levels, "groups": {str(s): files_by_size[s] for s in files_by_size},
            "blocks": blocks, "target_distance": target,
            "tolerance": {str(s): round(t, 5) for s, t in tol_used.items()}, "min_distance_between_any": d_min,
            "order": args.order, "seed": args.seed,
            "metric": "DreamSim ensemble (1 - cosine), fractals on grey RGB 128",
            "candidates": {str(file_of[m]): int(m + 1) for m in file_of},
            "source": "fractal_stimuli/generate_fractals.py -> embed_dreamsim.py -> make_fractal_sets.py"}
    (out / "groups.json").write_text(json.dumps(meta, indent=1))
    (out / "groups.js").write_text(
        "// Written by fractal_stimuli/make_fractal_sets.py: do not edit. Equally distinct fractal groups\n"
        f"// (DreamSim distance {target:.3f} +- {max(tol_used.values()):.4f} within a group, >= {d_min:.3f} between any two).\n"
        "// Keys are the group size: the fractals that show the levels of one side of a block.\n"
        "export const FRACTAL_GROUPS = {\n"
        + "".join(f"  {s}: {json.dumps(files_by_size[s])},\n" for s in sorted(files_by_size))
        + "};\n")
    if blocks:
        contact_sheet(blocks, out, out / "contact_sheet.png")
    else:
        contact_sheet([[g] for s in sorted(files_by_size, reverse=True) for g in files_by_size[s]], out,
                      out / "contact_sheet.png", cols=6, label="group")
    within, between = report(D, groups, params, target,
                             f"{name}: {len(file_of)} fractals, {len(blocks) or len(groups)} "
                             f"{'blocks' if blocks else 'groups'}", out / "report.png")
    print(f"{name}: {len(file_of)} fractals in "
          + ", ".join(f"{len(files_by_size[s])} groups of {s} (+-{tol_used[s]:.4f})" for s in files_by_size)
          + (f" = {len(blocks)} blocks" if blocks else "") + f" -> {out}")
    print(f"  within groups {min(within):.3f}-{max(within):.3f}; between groups min {min(between):.3f}, "
          f"median {np.median(between):.3f}; L* {pd.DataFrame(rows)['L_mean'].min():.1f}-"
          f"{pd.DataFrame(rows)['L_mean'].max():.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("designs", nargs="*", help="designs as levels per feature, e.g. 2x2 4x4 3x2x2")
    parser.add_argument("--n", type=int, default=100, help="about this many fractals per design")
    parser.add_argument("--groups", help="instead of designs: group counts as size:count,... (e.g. 4:18,2:12)")
    parser.add_argument("--name", help="set name with --groups")
    parser.add_argument("--out", type=Path, help="output folder (default: fractal_sets/<design>)")
    parser.add_argument("--target", type=float, default=None, help="within-group distance (default: median pair)")
    parser.add_argument("--tol", type=float, default=0.01, help="max |distance - target| within a group")
    parser.add_argument("--d-min", type=float, default=None,
                        help="min distance between any two fractals of a set (default: 5th percentile of pairs)")
    parser.add_argument("--order", choices=["conflict", "deviation"], default="conflict")
    parser.add_argument("--seed", type=int, default=0, help="assignment of groups to blocks")
    parser.add_argument("--pool", type=Path, default=POOL)
    args = parser.parse_args()
    if bool(args.designs) == bool(args.groups):
        parser.error("give designs (e.g. 2x2 4x4) or --groups, not both")
    if args.out and len(args.designs) > 1:
        parser.error("--out works with one design at a time")

    D = np.load(args.pool / "dreamsim_distances.npy").astype(float)
    params = pd.read_csv(args.pool / "candidates" / "params.csv")
    off = D[np.triu_indices(len(D), 1)]
    target = args.target if args.target is not None else float(np.median(off))
    d_min = args.d_min if args.d_min is not None else float(np.percentile(off, 5))
    print(f"{len(D)} candidates; within-group distance {target:.3f} +- {args.tol}; "
          f"min distance between any two {d_min:.3f}; order {args.order}")

    if args.groups:
        counts = {int(s): int(c) for s, c in (item.split(":") for item in args.groups.split(","))}
        name = args.name or "custom"
        picked, tol_used = select(D, counts, target, args.tol, d_min, args.order)
        write_set(name, None, picked, tol_used, D, params, target, args, d_min, args.out or SETS / name)
        return
    for design in args.designs:
        levels = [int(v) for v in design.lower().split("x")]
        n_blocks = args.n // sum(levels)
        counts = {s: levels.count(s) * n_blocks for s in set(levels)}
        picked, tol_used = select(D, counts, target, args.tol, d_min, args.order)
        write_set(design, levels, picked, tol_used, D, params, target, args, d_min, args.out or SETS / design)


if __name__ == "__main__":
    main()
