"""Regenerate the task's noise patches in Python (same PRNG, draws and recipe as src/noise.js).

Every trial row has `alpha`, `seed_left` and `seed_right`; image(alpha, seed) returns the
patch the participant saw (grey levels 0-255, NOISE.grid pixels square, aperture included),
equal to the browser's to floating-point precision. Settings are read from src/config.js.

  python3 tools/noise_field.py 2.6667 123456789 --out patch.png
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / "src" / "config.js"
M32 = 0xFFFFFFFF


def read_config():
    """NOISE and LAYOUT.patchSize from config.js"""
    text = CONFIG.read_text()
    noise = re.search(r"export const NOISE = \{(.*?)\};", text, re.S).group(1)
    get = lambda name: float(re.search(rf"\b{name}:\s*([\d.]+)", noise).group(1))  # noqa: E731
    size = float(re.search(r"patchSize:\s*([\d.]+)", text).group(1))
    return {"grid": int(get("grid")), "rms": get("rms"), "cutoffCpd": get("cutoffCpd"),
            "apertureEdge": get("apertureEdge"), "patchSize": size}


NOISE = read_config()


def mulberry32(seed):
    """The task's PRNG: floats in [0, 1), identical to makeRng() in src/noise.js"""
    t = seed & M32

    def nxt():
        nonlocal t
        t = (t + 0x6D2B79F5) & M32
        r = ((t ^ (t >> 15)) * (1 | t)) & M32
        r = ((r + (((r ^ (r >> 7)) * (61 | r)) & M32)) & M32) ^ r
        return ((r ^ (r >> 14)) & M32) / 2**32

    return nxt


def gaussians(n, rng):
    out = np.empty(n)
    for i in range(0, n, 2):
        u1 = 1 - rng()
        u2 = rng()
        r = math.sqrt(-2 * math.log(u1))
        out[i] = r * math.cos(2 * math.pi * u2)
        if i + 1 < n:
            out[i + 1] = r * math.sin(2 * math.pi * u2)
    return out


def field(alpha, seed, cutoff_cpd=None):
    """Noise field in [-1, 1] (grid x grid), as noiseField() in src/noise.js"""
    n = NOISE["grid"]
    k_max = (NOISE["cutoffCpd"] if cutoff_cpd is None else cutoff_cpd) * NOISE["patchSize"]
    white = gaussians(n * n, mulberry32(seed)).reshape(n, n)
    f = np.fft.fftfreq(n) * n
    k = np.hypot(f[None, :], f[:, None])
    amp = np.zeros_like(k)
    keep = (k > 0) & (k <= k_max)
    amp[keep] = k[keep] ** -float(alpha)
    x = np.real(np.fft.ifft2(np.fft.fft2(white) * amp))
    return np.clip(x * NOISE["rms"] / x.std(), -1, 1)


def aperture():
    n = NOISE["grid"]
    c = np.linspace(-1, 1, n)
    r = np.hypot(c[None, :], c[:, None])
    inner = 1 - NOISE["apertureEdge"]
    a = np.ones_like(r)
    ramp = (r > inner) & (r <= 1)
    a[ramp] = 0.5 * (1 + np.cos(np.pi * (r[ramp] - inner) / NOISE["apertureEdge"]))
    a[r > 1] = 0
    return a


def image(alpha, seed):
    """Grey levels 0-255 as drawn on screen (drawField() in src/noise.js)"""
    return np.floor(255 * (0.5 + 0.5 * field(alpha, seed) * aperture()) + 0.5).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("alpha", type=float)
    parser.add_argument("seed", type=int)
    parser.add_argument("--out", type=Path, help="save the patch as a PNG")
    parser.add_argument("--json", action="store_true", help="print the field as JSON")
    args = parser.parse_args()
    if args.out:
        from PIL import Image
        Image.fromarray(image(args.alpha, args.seed)).save(args.out)
        print(f"saved {args.out}")
    if args.json:
        print(json.dumps(field(args.alpha, args.seed).round(10).ravel().tolist()))


if __name__ == "__main__":
    main()
