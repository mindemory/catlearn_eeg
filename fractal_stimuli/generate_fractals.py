"""Generate a pool of candidate fractals with controlled colour, brightness and size.

Shapes follow Miyashita et al. (1997), as in the original generate_fractals.py: each fractal
is 3 nested polygons (5-10 edges, sizes 25 / 20 / 15) whose edges are deflected inward 3-4
times. What changed, so that no fractal is brighter, more colourful or larger than another:

  colour   each layer's colour is set in CIELCh (perceptual lightness L*, chroma C, hue h),
           not as fully saturated HLS. All fractals share one lightness profile
           (LIGHTNESS, outer / middle / inner layer), so brightness is matched exactly, and
           one chroma (CHROMA). Where a hue cannot reach CHROMA at a layer's lightness on an
           sRGB screen (cyan-blue hues, 180-240 deg), its chroma is lowered to the most the
           screen can show (down to ~29) rather than clipping the colour, which would change
           its brightness. Hues are 3 distinct values from HUES, 12 equally spaced hues.
  size     every fractal is rescaled so its filled area is the same (TARGET_AREA of the
           canvas).

Candidates go to ~/Documents/data/catlearn_eeg/fractal_pool/candidates/
as <k>.png (500 x 500 RGBA, transparent background) plus params.csv (shape and colour
parameters, measured area, brightness and chroma of each). make_fractal_sets.py then
picks fractal sets from them using DreamSim distances (embed_dreamsim.py).

  ~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/generate_fractals.py --n 1500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage import color

OUT_DIR = Path.home() / "Documents" / "data" / "catlearn_eeg" / "fractal_pool" / "candidates"

SIZE = 500                      # output image, px
SUPERSAMPLE = 4                 # draw larger, then downsample (smooth edges)
EXTENT = 30                     # fractal units shown across half the canvas (as the original's axes, +-30)
EDGE_SIZE = [25, 20, 15]        # outer, middle, inner layer
LIGHTNESS = [62, 48, 68]        # L* of the outer, middle, inner layer (background grey is L* 53.6)
CHROMA = 40                     # C*; lowered per hue where the sRGB gamut forces it (see lch_to_rgb255)
HUES = np.arange(0, 360, 30)    # 12 equally spaced hues
TARGET_AREA = 0.22              # filled fraction of the canvas, for every fractal
MAX_RADIUS = 0.46               # reject fractals reaching beyond this fraction of the canvas from its centre


def deflectMP(a, b, GA):
    """One recursion: insert a point between each pair of vertices, displaced by GA."""
    numDots = len(a)
    a1 = np.zeros((2 * numDots, 1))
    a2 = np.zeros((2 * numDots, 1))
    for i in range(numDots):
        a1[2 * i + 1] = a[i]
        a2[2 * i + 1] = b[i]
        if i == 0:
            ma = (a[-1] + a[0]) / 2
            mb = (b[-1] + b[0]) / 2
            da = a[0] - a[-1]
            db = b[0] - b[-1]
        else:
            ma = (a[i] + a[i - 1]) / 2
            mb = (b[i] + b[i - 1]) / 2
            da = a[i] - a[i - 1]
            db = b[i] - b[i - 1]
        if abs(db) < 1e-3:
            theta = np.pi / 2 * 3
        else:
            theta = np.arctan(da / db)
            if db < 0:
                theta = theta + np.pi
        a1[2 * i] = ma + GA * np.sin(theta - np.pi / 2)
        a2[2 * i] = mb + GA * np.cos(theta - np.pi / 2)
    return a1, a2


def layer_polygon(n_edges, depth, edge_size, ga):
    """Vertices (x, y) of one layer in fractal units."""
    angles = 2 * np.pi * np.linspace(1 / n_edges, 1, n_edges)
    x, y = np.sin(angles) * edge_size, np.cos(angles) * edge_size
    for _ in range(depth):
        x, y = deflectMP(x, y, ga)
    return np.column_stack((-np.ravel(x), -np.ravel(y)))


def _lab(L, C, h):
    return np.array([[[L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))]]])


def _in_gamut(L, C, h):
    lab = _lab(L, C, h)
    return np.abs(color.rgb2lab(color.lab2rgb(lab)) - lab).max() < 0.3


def gamut_chroma(L, h, C):
    """C, or the largest chroma below it that an sRGB screen can show at this L* and hue."""
    if _in_gamut(L, C, h):
        return C
    lo, hi = 0.0, C
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _in_gamut(L, mid, h) else (lo, mid)
    return lo


def lch_to_rgb255(L, C, h):
    """sRGB colour at lightness L*, hue h and chroma C (lowered to the gamut if needed:
    keeps L* and h exact). Returns ((r, g, b), chroma used)."""
    c = gamut_chroma(L, h, C)
    rgb = color.lab2rgb(_lab(L, c, h))[0, 0]
    return tuple(int(round(255 * v)) for v in rgb), c


def render(layers, colors, scale):
    """RGBA image of the layers (outer first), coordinates scaled by `scale`."""
    S = SIZE * SUPERSAMPLE
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for poly, rgb in zip(layers, colors):
        px = (poly * scale + EXTENT) / (2 * EXTENT) * S
        draw.polygon([tuple(p) for p in px], fill=rgb + (255,))
    return img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def measure(img):
    a = np.asarray(img, dtype=float) / 255
    mask = a[..., 3] > 0.5
    lab = color.rgb2lab(a[..., :3])[mask]
    ys, xs = np.nonzero(mask)
    r = np.hypot(xs - SIZE / 2, ys - SIZE / 2).max() / SIZE
    return {"area": mask.mean(), "L_mean": lab[:, 0].mean(), "C_mean": np.hypot(lab[:, 1], lab[:, 2]).mean(),
            "max_radius": r}


def make_candidate(rng):
    n_edges = rng.integers(5, 10, 3, endpoint=True)
    depth = rng.integers(3, 4, 3, endpoint=True)
    ga = rng.integers(-5, -2, 3, endpoint=True)
    hues = rng.choice(HUES, 3, replace=False)
    layers = [layer_polygon(n_edges[i], depth[i], EDGE_SIZE[i], ga[i]) for i in range(3)]
    colors, chromas = zip(*(lch_to_rgb255(LIGHTNESS[i], CHROMA, hues[i]) for i in range(3)))
    # equalise the filled area: area scales with the square of the coordinates
    area = measure(render(layers, colors, 1.0))["area"]
    scale = np.sqrt(TARGET_AREA / area)
    img = render(layers, colors, scale)
    params = {"n_edges": "-".join(map(str, n_edges)), "depth": "-".join(map(str, depth)), "ga": "-".join(map(str, ga)),
              "hues": "-".join(str(int(h)) for h in hues), "layer_chroma": "-".join(f"{c:.0f}" for c in chromas),
              "scale": scale}
    return img, params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1500, help="number of candidates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, k, rejected = [], 0, 0
    while k < args.n:
        img, params = make_candidate(rng)
        m = measure(img)
        if m["max_radius"] > MAX_RADIUS:          # too spiky to fit the canvas at the common area
            rejected += 1
            continue
        k += 1
        img.save(args.out / f"{k}.png")
        rows.append({"k": k, **params, **m})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "params.csv", index=False)
    print(f"{args.n} candidates in {args.out} ({rejected} rejected as too large)")
    print(df[["area", "L_mean", "C_mean", "max_radius"]].describe().round(3).loc[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
