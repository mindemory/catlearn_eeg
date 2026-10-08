# fractal_stimuli

Fractal symbols for the category-learning tasks.

- **Matched:** every fractal has the same brightness, colourfulness and size.
- **Grouped:** each set is cut into groups whose members all look equally different from
  each other (DreamSim distance).
- **Designs:** one set per factorial design (2×2, 2×2×2, 2×2×2×2, 4×4, 3×2×2, …).

In a block, each feature (position) shows one group, with one fractal per level. A 3×2×2
block therefore needs one group of 3 and two groups of 2.

## Make the sets

Run the steps in order; steps 1–2 need running only once.

**1. Candidates.** 1,500 fractals in `~/Documents/data/catlearn_eeg/fractal_pool/candidates/`:

```bash
~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/generate_fractals.py --n 1500
```

**2. Distances.** DreamSim distances between all candidates, in its own env:

```bash
~/miniforge3/envs/dreamsim/bin/python fractal_stimuli/embed_dreamsim.py
```

**3. Sets.** One set per design, about `--n` fractals each (default 100), in
`~/Documents/data/catlearn_eeg/fractal_sets/<design>/`:

```bash
~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/make_fractal_sets.py 2x2 2x2x2 2x2x2x2 4x4 3x2x2
```

Any design works, e.g. `3x3`, `2x2x2x2x2` or `4x2`.

| File | What it does |
|---|---|
| `generate_fractals.py` | makes the candidate fractals (below) |
| `embed_dreamsim.py` | computes DreamSim distances between all candidates (`dreamsim` env; weights in `~/Documents/data/catlearn_eeg/models/dreamsim`, 3 GB) |
| `make_fractal_sets.py` | picks groups of equally distinct fractals for each design |
| `similarity_matrices.py` | colour, shape and perceptual (DreamSim) similarity between all candidates, as heatmaps sorted by clustering, plus example fractals from the largest clusters (`<pool>/similarity/`) |

## How the fractals are matched

Shapes follow Miyashita et al. (1997): 3 nested polygons with deflected edges. The old
fractals (`catlearn_task/fractals`) used fully saturated colours, so some were much
brighter or more colourful than others:

|   | Old | New |
|---|---|---|
| L* (lightness) | 35–93 | 54–65 per fractal; every fractal uses the same layer profile, L* 62 / 48 / 68 (outer / middle / inner); background grey is 53.6 |
| Chroma | 60–129 | 36–40: chroma 40 for every layer, lowered only for cyan-blue hues that a screen can't show at 40 (down to about 29) |
| Hues | random | 3 of 12 equally spaced hues per fractal |
| Area | 15–33% of the image | 22% for every fractal |

### The vivid pool (October 2026)

The fractals above looked dull and close to the background grey. A second pool raises
both lightness and colour strength, still with one profile for every fractal:

|   | Original pool | Vivid pool |
|---|---|---|
| Layer lightness (outer / middle / inner) | L* 62 / 48 / 68 | L* 70 / 56 / 78 |
| Mean L* per fractal (background 53.6) | 57–63 | 62–74 |
| Chroma | 40 (29–40 after the screen limit) | 60, lowered per hue to what the screen can show: 33–60 per layer, median 54 (cyan-blue lowest) |

Above about L* 72, blues, purples and reds lose most of their available chroma (at L* 85
some reach only 22). That is why the lightness stays moderate and the colour does the work.
The shapes and hues are the same as in the original pool (same seed); only the colours
differ, so the size and the tasks' `fractalImageScale` are unchanged.

These are the fractals used from October 2026 on: both online tasks'
`stimuli/fractal_groups/` hold the vivid `online_4x4` and `2x2x2` sets.

**What drives the perceptual distances** (`similarity_matrices.py`, all 1,500 vivid
candidates):
- **Colour:** DreamSim similarity correlates 0.46 with colour similarity (same colours in
  the same proportions).
- **Shape:** it correlates 0.03 with shape similarity (overlap of the silhouettes).
- **Clusters:** the perceptual clusters are colour schemes, e.g. cyan/blue/purple fractals
  vs pink/orange ones.

So groups matched in DreamSim distance are mostly matched in how different their colours
are; the shapes add little.

Commands (the same steps as above, into separate folders):

```bash
~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/generate_fractals.py --n 1500 --lightness 70 56 78 --chroma 60 --out ~/Documents/data/catlearn_eeg/fractal_pool_vivid/candidates
```

```bash
~/miniforge3/envs/dreamsim/bin/python fractal_stimuli/embed_dreamsim.py --pool ~/Documents/data/catlearn_eeg/fractal_pool_vivid
```

```bash
~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/make_fractal_sets.py 4x4 --pool ~/Documents/data/catlearn_eeg/fractal_pool_vivid --out ~/Documents/data/catlearn_eeg/fractal_sets_vivid/4x4
```

Sets made: `online_4x4` (the 4×4 task's 18 groups of 4 and 12 pairs), `2x2`, `2x2x2`,
`2x2x2x2`, `4x4` and `3x2x2`, in `~/Documents/data/catlearn_eeg/fractal_sets_vivid/`. The
within-group target is the vivid pool's own median distance.

## How the groups are picked

- **Distance:** DreamSim (Fu et al., 2023) is an ensemble of DINO, CLIP and OpenCLIP
  features, tuned on human similarity judgements. Distance = 1 − cosine of the
  embeddings, computed with each fractal on the task's grey. Over all candidate pairs the
  median is 0.256 and the 5–95% range 0.162–0.357.
- **Within a group:** every pairwise distance is within `--tol` (0.01) of one target, the
  median 0.256.
  - The target is the same for every group size and every design, so a level in a 2×2 is
    as easy to tell apart as one in a 4×4.
  - Each group size uses the tightest tolerance that still fits. Pairs and triples come
    out within ±0.0025, groups of 4 within ±0.0075.
- **Between groups:** no two fractals of a set are closer than `--d-min`, 0.162 (the 5th
  percentile), so a set has no near-duplicates. The median distance between groups is
  0.29–0.31.
- **Search:** every qualifying group is listed, then groups are taken greedily, largest
  first.
  - `--order conflict` (the default) takes first the groups whose fractals have the
    fewest near neighbours in the pool. This fits many more groups: 34 groups of 4,
    against 24 when the closest-to-target groups go first (`--order deviation`).

The sets made so far:

| Design | Fractals | Groups | Blocks |
|---|---|---|---|
| 2×2 | 100 | 50 pairs | 25 |
| 2×2×2 | 96 | 48 pairs | 16 |
| 2×2×2×2 | 96 | 48 pairs | 12 |
| 4×4 | 96 | 24 groups of 4 | 12 |
| 3×2×2 | 98 | 14 triples + 28 pairs | 14 |

The pair-only designs (2×2, 2×2×2, 2×2×2×2) are picked the same way, so they largely
share pairs. Within one set, every fractal and group is used once.

## Output, per set

| File | What it holds |
|---|---|
| `<n>.png` | the fractals, 500 × 500 RGBA with a transparent background, numbered 1..N (larger groups first) |
| `groups.json` | the design, its levels, `groups` (by size), `blocks` (one group per feature, assigned at random with `--seed`), the target, tolerances and minimum distance, and the candidate behind each file |
| `groups.csv` | one row per fractal: file, group, size, block, feature, candidate, L*, chroma, area (for MATLAB `readtable`) |
| `groups.js` | the groups as an ES module, for web tasks |
| `contact_sheet.png` | every block on the task's grey, features left to right |
| `report.png` | distance histograms (all candidates, within groups, between groups) and the set's brightness and chroma |

## The online task's set

The set in `catlearn_4x4_prolific/stimuli/fractal_groups/` (18 groups of 4 and 12
pairs) was made with:

```bash
~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/make_fractal_sets.py --groups 4:18,2:12 --order deviation --name online_4x4 --out catlearn_4x4_prolific/stimuli/fractal_groups
```
