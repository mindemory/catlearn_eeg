import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import os


def deflectMP(a, b, GA):
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


def getFractal(nEdges, depthRecur, edgeSize, GA, hues, offsets, gains):
    """
    Algorithm adapted from Miyashita et al., 1997, Generation of fractal patterns for probing the visual memory
    Returns a patch collection of 3 polygon objects corresponding to 3 layers of a fractal
    """
    patches = []
    for i in range(len(nEdges)):
        angles = 2 * np.pi * np.linspace(1 / nEdges[i], 1, nEdges[i])
        x = np.sin(angles) * edgeSize[i]
        y = np.cos(angles) * edgeSize[i]

        ga = GA[i]
        for j in range(depthRecur[i]):
            [x, y] = deflectMP(x, y, ga)

        coordinates = np.column_stack(
            (-x * gains[0] + offsets[0], -y * gains[1] + offsets[1])
        )
        patches.append(Polygon(coordinates, color=hls_to_rgb(hues[i] / 360, 0.5, 1)))
    return patches


# NEdges = 5~10, Depth = 3~4, GA = -5~-3 (negative means inward deflection)
# EdgeSize = 20 - (5~6)*(i-1) where i = ith superposition
# Color in HSL space, S = 255, L = 255*0.5


class Fractal:
    def __init__(self, n_layers=3, hues=None, rng=None) -> None:
        if rng is None:
            self.rng = np.random.default_rng()
        self.nEdges = rng.integers(5, 10, n_layers, endpoint=True)
        self.depthRecur = rng.integers(3, 4, n_layers, endpoint=True)
        self.GA = rng.integers(-5, -2, n_layers, endpoint=True)
        self.edgeSize = [25, 20, 15]
        # self.edgeSize = np.random.randint([25, 20, 14], [27, 21, 16], n_layers)
        self.hues = rng.uniform(0, 360, n_layers) if hues is None else hues

    def save(self, filename):
        patches = getFractal(
            self.nEdges,
            self.depthRecur,
            self.edgeSize,
            self.GA,
            self.hues,
            [0, 0],
            [1, 1],
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        p = PatchCollection(patches, alpha=1, match_original=True)
        ax.add_collection(p)
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.savefig(
            filename,
        )
        plt.close()


def generate_fractals(N, seed=0, folder=None):
    """
    params: N = number of sets (each set contains 3T, 2A, 2B)
    """
    n_layers = 3
    rng = np.random.default_rng(seed)
    for i in range(1, N + 1):
        huesT1 = rng.uniform(0, 360, 3)
        # one color for each layer
        huesT = [huesT1, (huesT1 + 120) % 360, (huesT1 + 240) % 360]
        # instances within each cue are maximally different
        # TODO: control distance between cues A, B, T
        c = 1
        for t in range(3):
            fractal = Fractal(n_layers, hues=huesT[t], rng=rng)
            fractal.save(os.path.join(folder, f"{c}.svg"))
            c += 1
        for cue in ["A", "B"]:
            hues1 = rng.uniform(0, 360, 3)
            for j in range(2):
                fractal = Fractal(n_layers, hues=(hues1 + j * 180) % 360, rng=rng)
                fractal.save(os.path.join(folder, f"{c}.svg"))
                c += 1


if __name__ == "__main__":
    import sys

    generate_fractals(5, folder="fractals")
