"""How discriminable are 1/f^alpha noise patches from each other, as a function of alpha?

For each alpha (0.5 to 3 in steps of 0.25) many patches are generated as in
generate_stimuli.py (RMS-contrast matched, circular aperture, mid-grey background), and the
distance between every pair is measured four ways:

  dreamsim     DreamSim ensemble (DINO + CLIP + OpenCLIP, tuned on human similarity
               judgements): 1 - cosine of the embeddings. The same measure as the fractal
               sets (fractal_stimuli/), so the two are on one scale.
  dino_global  plain DINO ViT-B/16 CLS token: 1 - cosine. A whole-image summary, largely
               blind to where things are; for noise, close to "same texture or not".
  dino_local   plain DINO patch tokens compared location by location (16 x 16 px cells
               inside the aperture): 1 - mean cosine. Sensitive to the layout of blobs.
  pixel        1 - cosine of the contrast images (intensity - grey), as generate_stimuli.py
               uses to pick dissimilar patches. Pure layout.

Two references:
  floor        each patch against itself shifted by --shift px (the noise is periodic, so the
               shift wraps): a distance that does not mean a different stimulus.
  fractals     the online task's fractal groups (catlearn_4x4_prolific/stimuli/fractal_groups),
               within-group distances, on every measure.

These are foveal, image-computable proxies. At 6 deg eccentricity peripheral vision pools
texture statistics, so same-alpha patches are likely harder to tell apart than any of these
measures says; a short same/different test at 6 deg is the real check.

Two steps, in two envs:
  ~/miniforge3/envs/dreamsim/bin/python task_design/noise_similarity.py embed
  ~/miniforge3/envs/kernelbehav/bin/python task_design/noise_similarity.py plot

Outputs in ~/Documents/data/catlearn_eeg/task_design/noise_similarity/:
  distances.npz           distance matrices (patches + fractals), floors, labels
  summary.csv             per alpha and measure: within-alpha distance percentiles, floor,
                          fractal reference
  within_alpha.png        within-alpha distance vs alpha, per measure, with example patches
  between_alpha.png       median distance between patches of two alphas, per measure
"""

import argparse
import json
from pathlib import Path

import numpy as np

from noise_patches import circular_aperture, make_noise_patch

DATA = Path.home() / "Documents" / "data" / "catlearn_eeg"
OUT = DATA / "task_design" / "noise_similarity"
CACHE = DATA / "models" / "dreamsim"
FRACTALS = Path(__file__).resolve().parents[1] / "catlearn_4x4_prolific" / "stimuli" / "fractal_groups"
ALPHAS = np.round(np.arange(0.5, 3.01, 0.25), 2)
MEASURES = ["dreamsim", "dino_global", "dino_local", "pixel"]
LABELS = {"dreamsim": "DreamSim ensemble", "dino_global": "DINO, whole image (CLS)",
          "dino_local": "DINO, location by location", "pixel": "pixels (contrast)"}
SIZE = 256
GREY = 0.5


# ---------------------------------------------------------------- stimuli
def patches_for(alpha_index, n, shift, seed):
    """n patches at ALPHAS[alpha_index], and each shifted by `shift` px; values in [0, 1]."""
    rng = np.random.default_rng([seed, alpha_index])
    mask = circular_aperture(SIZE)
    fields = [make_noise_patch(SIZE, rng, ALPHAS[alpha_index]) for _ in range(n)]
    patches = np.array([GREY + 0.5 * f * mask for f in fields], dtype=np.float32)
    shifted = np.array([GREY + 0.5 * np.roll(f, (shift, shift), (0, 1)) * mask for f in fields], dtype=np.float32)
    return patches, shifted


def fractal_images():
    """The online task's fractals on the grey background (PIL, as in fractal_stimuli/embed_dreamsim.py),
    and their groups as indices into that list."""
    from PIL import Image
    groups = json.loads((FRACTALS / "groups.json").read_text())["groups"]
    files = sorted(int(p.stem) for p in FRACTALS.glob("*.png") if p.stem.isdigit())
    imgs = []
    for f in files:
        im = Image.open(FRACTALS / f"{f}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
        bg.alpha_composite(im)
        imgs.append(bg.convert("RGB"))
    index = {f: i for i, f in enumerate(files)}
    return imgs, [[index[f] for f in g] for size in groups.values() for g in size]


# ---------------------------------------------------------------- embed (dreamsim env)
def embed(args):
    import torch
    from dreamsim import dreamsim
    from PIL import Image
    from dreamsim.model import PerceptualModel
    from torch.nn import functional as F

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ds_model, preprocess = dreamsim(pretrained=True, device=device, cache_dir=str(CACHE), dreamsim_type="ensemble")
    dino = PerceptualModel(model_type="dino_vitb16", feat_type="cls_patch", stride="16", baseline=True,
                           load_dir=str(CACHE), device=device).to(device).eval()

    # images: patches (all alphas), their shifted copies, then the fractals; grey -> RGB
    patches, shifted, alpha_of = [], [], []
    for ai in range(len(ALPHAS)):
        p, s = patches_for(ai, args.n, args.shift, args.seed)
        patches.append(p)
        shifted.append(s)
        alpha_of += [ALPHAS[ai]] * args.n
    patches, shifted = np.concatenate(patches), np.concatenate(shifted)
    frac, frac_groups = fractal_images()
    images = [Image.fromarray(np.round(255 * p).astype(np.uint8)).convert("RGB")
              for p in np.concatenate([patches, shifted])] + frac
    n_patch = len(patches)

    # aperture cells: 14 x 14 tokens of a 224 px input; keep the cells mostly inside the circle
    cell = circular_aperture(224).reshape(14, 16, 14, 16).mean((1, 3)).ravel() > 0.9

    ds, cls, local, pix = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(images), args.batch):
            x = torch.cat([preprocess(im) for im in images[i:i + args.batch]]).to(device)   # (B, 3, 224, 224)
            pix.append((x.mean(1) - GREY).flatten(1).cpu())
            ds.append(ds_model.embed(x).float().cpu())
            desc = dino.extractor_list[0].extract_descriptors(dino._preprocess(x, "dino_vitb16"), 11)
            desc = desc[:, 0, 0].float().cpu()                         # (B, 1 + 196, 768)
            cls.append(desc[:, 0])
            local.append(F.normalize(desc[:, 1:][:, torch.from_numpy(cell)], dim=-1).half())
            print(f"  embedded {min(i + args.batch, len(images))} / {len(images)}", flush=True)

    def cosine_dist(e):
        e = F.normalize(torch.cat(e).float(), dim=-1)
        return (1 - e @ e.T).numpy()

    D = {"dreamsim": cosine_dist(ds), "dino_global": cosine_dist(cls)}
    tok = torch.cat(local).float()                                     # (N, cells, 768), unit tokens
    flat = tok.reshape(len(tok), -1) / np.sqrt(tok.shape[1])
    D["dino_local"] = (1 - flat @ flat.T).numpy()
    D["pixel"] = cosine_dist(pix)

    # keep patches + fractals in the matrices; floors = patch vs its own shifted copy
    keep = np.r_[0:n_patch, 2 * n_patch:len(images)]
    out = {f"D_{m}": D[m][np.ix_(keep, keep)].astype(np.float32) for m in MEASURES}
    out.update({f"floor_{m}": D[m][np.arange(n_patch), n_patch + np.arange(n_patch)].astype(np.float32)
                for m in MEASURES})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "distances.npz", alpha_of=np.array(alpha_of), alphas=ALPHAS, n_patch=n_patch,
                        fractal_groups=np.array(frac_groups, dtype=object), shift=args.shift, seed=args.seed,
                        **out)
    print(f"saved {OUT / 'distances.npz'}: {n_patch} patches ({args.n} per alpha) + {len(frac)} fractals")


# ---------------------------------------------------------------- plot (kernelbehav env)
def plot(args):
    import matplotlib.pyplot as plt
    import pandas as pd
    from plot_style import apply_dark_theme

    z = np.load(OUT / "distances.npz", allow_pickle=True)
    alpha_of, n_patch = z["alpha_of"], int(z["n_patch"])
    groups = [np.array(g) + n_patch for g in z["fractal_groups"]]
    apply_dark_theme()

    rows = []
    for m in MEASURES:
        D = z[f"D_{m}"]
        frac_within = np.concatenate([D[np.ix_(g, g)][np.triu_indices(len(g), 1)] for g in groups])
        for a in ALPHAS:
            idx = np.flatnonzero(alpha_of == a)
            d = D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
            floor = z[f"floor_{m}"][idx]
            rows.append({"measure": m, "alpha": a, "median": np.median(d), "p5": np.percentile(d, 5),
                         "p25": np.percentile(d, 25), "p75": np.percentile(d, 75), "p95": np.percentile(d, 95),
                         "floor_median": np.median(floor), "fractal_within_median": np.median(frac_within)})
    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(OUT / "summary.csv", index=False)

    # within-alpha distances vs alpha, with an example patch per alpha on top
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, len(ALPHAS), height_ratios=[1, 4.2], hspace=0.25)
    for ai, a in enumerate(ALPHAS):
        ax = fig.add_subplot(gs[0, ai])
        ax.imshow(patches_for(ai, 1, 0, z["seed"].item())[0][0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"α {a:g}", fontsize=9)
        ax.axis("off")
    sub = gs[1, :].subgridspec(1, len(MEASURES), wspace=0.3)
    for k, m in enumerate(MEASURES):
        s = summary[summary.measure == m]
        ax = fig.add_subplot(sub[0, k])
        ax.fill_between(s.alpha, s.p5, s.p95, color="#2ec4b6", alpha=0.2, lw=0, label="pairs, 5–95%")
        ax.fill_between(s.alpha, s.p25, s.p75, color="#2ec4b6", alpha=0.4, lw=0, label="pairs, 25–75%")
        ax.plot(s.alpha, s["median"], color="#2ec4b6", lw=2, label="pairs, median")
        ax.plot(s.alpha, s.floor_median, color="#ff6b6b", lw=1.5, ls="--",
                label=f"same patch shifted {int(z['shift'])} px")
        ax.axhline(s.fractal_within_median.iloc[0], color="#ffa630", lw=1.5, ls=":",
                   label="fractal groups (online task)")
        ax.set_title(LABELS[m], fontsize=10)
        ax.set_xlabel("α (spectral slope)")
        if k == 0:
            ax.set_ylabel("distance between two patches of the same α")
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Noise patches: how different two patches of the same α look, by measure", fontsize=13)
    fig.savefig(OUT / "within_alpha.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # between-alpha median distances
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(4.6 * len(MEASURES), 4.4))
    for ax, m in zip(axes, MEASURES):
        D = z[f"D_{m}"]
        M = np.zeros((len(ALPHAS), len(ALPHAS)))
        for i, a in enumerate(ALPHAS):
            for j, b in enumerate(ALPHAS):
                ia, ib = np.flatnonzero(alpha_of == a), np.flatnonzero(alpha_of == b)
                block = D[np.ix_(ia, ib)]
                M[i, j] = np.median(block[np.triu_indices(len(ia), 1)] if i == j else block)
        im = ax.imshow(M, cmap="magma", origin="lower")
        ticks = range(0, len(ALPHAS), 2)
        ax.set_xticks(ticks, [f"{ALPHAS[t]:g}" for t in ticks])
        ax.set_yticks(ticks, [f"{ALPHAS[t]:g}" for t in ticks])
        ax.set_xlabel("α")
        ax.set_ylabel("α")
        ax.set_title(LABELS[m], fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Median distance between patches of two α values (diagonal: within one α)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "between_alpha.png", dpi=140)
    plt.close(fig)

    show = summary.pivot(index="alpha", columns="measure", values="median")[MEASURES]
    print("median within-alpha distance:\n" + show.round(3).to_string())
    print("shift floor (median):\n" + summary.pivot(index="alpha", columns="measure",
                                                     values="floor_median")[MEASURES].round(3).to_string())
    print("fractal groups (within, median): " + ", ".join(
        f"{m} {summary[summary.measure == m].fractal_within_median.iloc[0]:.3f}" for m in MEASURES))
    print(f"saved {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["embed", "plot"])
    parser.add_argument("--n", type=int, default=150, help="patches per alpha")
    parser.add_argument("--shift", type=int, default=4, help="px shift for the floor (of a 256 px patch)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    embed(args) if args.step == "embed" else plot(args)


if __name__ == "__main__":
    main()
