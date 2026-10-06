"""Noise-patch stimuli for the test block: pairs of equally distinct 1/f^alpha patches per alpha.

Each alpha condition (ALPHAS) is one difficulty level. Within a condition every pair should
be equally hard to tell apart, so that no feature of the 2x2x2 machine is easier to see than
another. For each alpha this script:
  1. generates --n candidate patches (task_design/noise_patches.py: RMS-contrast matched,
     circular aperture with a soft edge) as RGBA PNGs: grey values inside, the aperture as
     the alpha channel, so on the task's grey background they look as in generate_stimuli.py.
     Candidates and params.csv go to ~/Documents/data/catlearn_eeg/noise_pool/a<alpha>/.
  2. computes DreamSim distances between all candidates (fractal_stimuli/embed_dreamsim.py,
     in the dreamsim env).
  3. picks --pairs pairs whose distance is within --tol of that alpha's median pair distance,
     no two patches closer than the 5th percentile (fractal_stimuli/make_fractal_sets.py),
     into stimuli/noise/a<alpha>/.
Then it writes stimuli/noise/groups.js, which the task imports: NOISE_GROUPS = {alpha: {2:
[[file, file], ...]}}.

The median distance, and so the difficulty, rises with alpha (task_design/noise_similarity.py:
DreamSim 0.056 at alpha 1.5 to 0.126 at 2.25; fractal pairs are 0.256).

  ~/miniforge3/envs/kernelbehav/bin/python tools/make_noise_stimuli.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent
sys.path.insert(0, str(REPO / "task_design"))
from noise_patches import circular_aperture, make_noise_patch  # noqa: E402

POOL = Path.home() / "Documents" / "data" / "catlearn_eeg" / "noise_pool"
OUT = TASK / "stimuli" / "noise"
ALPHAS = ["1.50", "1.75", "2.00", "2.25"]
PY_KERNEL = Path.home() / "miniforge3" / "envs" / "kernelbehav" / "bin" / "python"
PY_DREAMSIM = Path.home() / "miniforge3" / "envs" / "dreamsim" / "bin" / "python"
SIZE = 256
GREY = 0.5


def generate(alpha, n, seed, folder):
    """n candidates at this alpha: <k>.png (RGBA) and params.csv."""
    cand = folder / "candidates"
    cand.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, int(round(float(alpha) * 100))])
    mask = circular_aperture(SIZE)
    inside = mask > 0.5
    rows = []
    for k in range(1, n + 1):
        field = make_noise_patch(SIZE, rng, float(alpha))
        grey = np.round(255 * (GREY + 0.5 * field)).astype(np.uint8)
        rgba = np.dstack([grey, grey, grey, np.round(255 * mask).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(cand / f"{k}.png")
        shown = GREY + 0.5 * field * mask                      # as seen on the grey background
        L = color.rgb2lab(np.repeat(shown[..., None], 3, -1))[..., 0]
        rows.append({"k": k, "alpha": float(alpha), "L_mean": L[inside].mean(), "C_mean": 0.0,
                     "area": inside.mean(), "rms_contrast": (0.5 * field[inside]).std() / GREY})
    pd.DataFrame(rows).to_csv(cand / "params.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alphas", nargs="+", default=ALPHAS)
    parser.add_argument("--n", type=int, default=500, help="candidates per alpha")
    parser.add_argument("--pairs", type=int, default=30, help="pairs per alpha (a participant sees 3)")
    parser.add_argument("--tol", type=float, default=0.005, help="max |distance - median| within a pair")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    groups = {}
    for alpha in args.alphas:
        folder = POOL / f"a{alpha}"
        print(f"== alpha {alpha}: {args.n} candidates -> {folder}", flush=True)
        generate(alpha, args.n, args.seed, folder)
        subprocess.run([str(PY_DREAMSIM), str(REPO / "fractal_stimuli" / "embed_dreamsim.py"), "--pool", str(folder)],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([str(PY_KERNEL), str(REPO / "fractal_stimuli" / "make_fractal_sets.py"),
                        "--pool", str(folder), "--groups", f"2:{args.pairs}", "--tol", str(args.tol),
                        "--order", "deviation", "--name", f"noise_a{alpha}", "--out", str(OUT / f"a{alpha}")],
                       check=True)
        meta = json.loads((OUT / f"a{alpha}" / "groups.json").read_text())
        groups[alpha] = meta["groups"]["2"]
        (OUT / f"a{alpha}" / "groups.js").unlink()           # the task reads the combined module

    (OUT / "groups.js").write_text(
        "// Written by tools/make_noise_stimuli.py: do not edit. Pairs of equally distinct noise\n"
        "// patches per alpha (DreamSim distance within --tol of that alpha's median pair distance).\n"
        "// Files: stimuli/noise/a<alpha>/<n>.png. Keys: alpha, then group size.\n"
        "export const NOISE_GROUPS = {\n"
        + "".join(f"  '{a}': {{ 2: {json.dumps(g)} }},\n" for a, g in groups.items())
        + "};\n")
    print(f"wrote {OUT / 'groups.js'}")


if __name__ == "__main__":
    main()
