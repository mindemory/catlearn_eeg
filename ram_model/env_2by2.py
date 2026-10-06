"""Display rendering and glimpse sensor for the 2x2 task, matching task_2by2's geometry.

The display is a square 7x7 grid of cells (as task_2by2 fits it to the screen height),
fixation in the center cell, and two noise patches filling 0.9 of a cell at the two
locations set by the spatial configuration (numbering from
task_design/plot_spatial_configs.py, same as task_2by2's `config` input). Location A
shows one of two dimension-A patches, location B one of two dimension-B patches, so a
block has 4 compound displays. Stimulus index s = 2 * levelA + levelB, the ordering the
kernel model's task labels use.

Pixel values are luminance minus the patches' 128/255 background, so the empty screen
is exactly 0.

The glimpse sensor returns a stack of square crops centered on the fixation point,
each downsampled to the same size -- a sharp fovea and progressively blurrier
surround -- plus a low-resolution view of the whole display:

    scales = (16, 32, 64) px with out = 16  ->  1x, 2x, 4x downsampled, + full display

With 16-px cells, the fovea covers exactly one cell at full resolution.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "task_design"))
sys.path.insert(0, str(REPO / "kernel_model"))
from generate_stimuli import circular_aperture, make_noise_patch  # noqa: E402
from kernel_modes import DATA_ROOT, Design  # noqa: E402
from plot_spatial_configs import CONFIGS  # noqa: E402

GRID = 7
STIM_DIR = DATA_ROOT / "task_design" / "stimuli" / "png"
ALIEN_DIR = DATA_ROOT / "task_design" / "stimuli_aliens"
FRACTAL_DIR = DATA_ROOT / "task_design" / "stimuli_fractals"
N_POOL = 20
N_ALIENS = 19
N_FRACTALS = 72


def spatial_config(config):
    """(name, [[rowA, colA], [rowB, colB]]) for config 1-18."""
    items = list(CONFIGS.items())
    if not 1 <= config <= len(items):
        raise ValueError(f"config must be 1-{len(items)}, got {config}")
    name, cells = items[config - 1]
    return name, np.array(cells)


def _to_tile(img_l, size):
    """8-bit grayscale PIL image -> size x size tile by area averaging, background -> 0."""
    img_l = img_l.resize((size, size), Image.Resampling.BOX)
    return np.asarray(img_l, dtype=np.float32) / 255.0 - 128.0 / 255.0


def load_patch(k, size):
    """Noise patch k (1-20) from the stimulus folder, as a size x size tile."""
    return _to_tile(Image.open(STIM_DIR / f"noisepatch_{k:02d}.png").convert("L"), size)


def load_object(path, size, color=False):
    """An RGBA object image (transparent background) as a size x size tile on the grey
    background (0 = grey).

    Cropped to the object (alpha > 0) -> centered on a mid-grey (128) square -> resized by
    area averaging. color=False: luminance tile (size, size); color=True: RGB tile
    (size, size, 3), each channel centered so the grey background is 0.
    """
    im = Image.open(path).convert("RGBA")
    bbox = im.getchannel("A").point(lambda v: 255 if v > 0 else 0).getbbox()
    im = im.crop(bbox)
    side = max(im.size)
    canvas = Image.new("RGBA", (side, side), (128, 128, 128, 255))
    canvas.alpha_composite(im, ((side - im.width) // 2, (side - im.height) // 2))
    if not color:
        return _to_tile(canvas.convert("L"), size)
    rgb = canvas.convert("RGB").resize((size, size), Image.Resampling.BOX)
    return np.asarray(rgb, dtype=np.float32) / 255.0 - 128.0 / 255.0


def load_alien(k, size, color=False):
    """Alien k (1-19) from task_design/stimuli_aliens (see load_object)."""
    return load_object(ALIEN_DIR / f"alien{k:02d}.png", size, color)


def load_fractal(k, size, color=False):
    """Fractal k (1-72) from task_design/stimuli_fractals (see load_object)."""
    return load_object(FRACTAL_DIR / f"{k}.png", size, color)


def load_stimulus_set(name, size, color=False):
    """(n, size, size[, 3]) tiles of a whole pool: 'noise' (20, grayscale), 'aliens' (19),
    'fractals' (72)."""
    if name == "noise":
        if color:
            raise ValueError("the noise patches are grayscale")
        return np.stack([load_patch(k, size) for k in range(1, N_POOL + 1)])
    if name == "aliens":
        return np.stack([load_alien(k, size, color) for k in range(1, N_ALIENS + 1)])
    if name == "fractals":
        return np.stack([load_fractal(k, size, color) for k in range(1, N_FRACTALS + 1)])
    raise ValueError(f"unknown stimulus set {name}")


def make_patch_bank(n, size, seed=1, alpha=1.5, full_px=256):
    """n fresh noise patches made exactly like the real ones (generate_stimuli.py: 1/f^alpha
    noise, RMS contrast 0.3, soft circular aperture, saved as 8-bit), as size x size tiles.

    The 20 real patches were drawn with seed 0; any other seed gives patches the agent
    never meets in the experiment, for pretraining.
    """
    if seed == 0:
        raise ValueError("seed 0 regenerates the real stimulus set; use another seed for pretraining patches")
    rng = np.random.default_rng(seed)
    mask = circular_aperture(full_px)
    tiles = []
    for _ in range(n):
        patch = 0.5 + 0.5 * make_noise_patch(full_px, rng, alpha) * mask
        img8 = np.round(np.clip(patch, 0, 1) * 255).astype(np.uint8)
        tiles.append(_to_tile(Image.fromarray(img8, mode="L"), size))
    return np.stack(tiles)


def cell_to_loc(cell, grid=GRID):
    """Grid cell (row, col) -> glimpse location (x, y) in [-1, 1], y downward."""
    row, col = cell
    return np.array([(col + 0.5) / grid * 2 - 1, (row + 0.5) / grid * 2 - 1], dtype=np.float32)


def tile_size(cell_px=16, stim_fraction=0.9):
    return int(round(stim_fraction * cell_px))


def render_block(config, patches, cell_px=16, stim_fraction=0.9):
    """The 4 compound displays of one block from real patches.

    patches: 4 pool indices [a1, a2, b1, b2]
    Returns images (4, H, W) with s = 2 * levelA + levelB, and the glimpse locations of
    the centers of location A and location B.
    """
    size = tile_size(cell_px, stim_fraction)
    return render_tiles(config, [load_patch(k, size) for k in patches], cell_px)


def render_tiles(config, tiles, cell_px=16):
    """Like render_block, but from 4 ready-made tiles [a1, a2, b1, b2] (e.g. a patch bank).
    Tiles may be luminance (size, size) or color (size, size, 3); images follow suit."""
    _, cells = spatial_config(config)
    H = GRID * cell_px
    size = tiles[0].shape[0]
    images = np.zeros((4, H, H) + tuple(tiles[0].shape[2:]), dtype=np.float32)   # (4, H, H[, 3])
    for s in range(4):
        level_a, level_b = divmod(s, 2)
        for tile, (row, col) in ((tiles[level_a], cells[0]), (tiles[2 + level_b], cells[1])):
            top = int(round((row + 0.5) * cell_px - size / 2))
            left = int(round((col + 0.5) * cell_px - size / 2))
            images[s, top:top + size, left:left + size] = tile
    locs = np.stack([cell_to_loc(cells[0]), cell_to_loc(cells[1])])
    return images, locs


def render_single_everywhere(tiles, cell_px=16):
    """Every grid position for each tile: a display with one patch in one cell.

    tiles: (k, size, size). Returns images (k * GRID**2, H, W), ordered tile-major, and
    labels (k * GRID**2,) giving the tile index -- for the rule-naive perceptual
    pretraining task ("which patch is this?"), with the patch equally often in every
    cell, the central fixation cell included.
    """
    k, size = tiles.shape[0], tiles.shape[1]
    H = GRID * cell_px
    images = np.zeros((k, GRID, GRID, H, H), dtype=np.float32)
    for row in range(GRID):
        for col in range(GRID):
            top = int(round((row + 0.5) * cell_px - size / 2))
            left = int(round((col + 0.5) * cell_px - size / 2))
            images[:, row, col, top:top + size, left:left + size] = tiles
    labels = np.repeat(np.arange(k, dtype=np.int32), GRID * GRID)
    return images.reshape(k * GRID * GRID, H, H), labels


def block_labels(design, task_name):
    """Class (0/1) of each of the 4 compounds under a task, e.g. 'A', 'B', 'AB'."""
    y = design.task_labels[design.task_index(task_name)]
    return (y > 0).astype(np.int32)


def glimpse(img, loc, scales=(16, 32, 64), out=16, full_view=True):
    """Glimpse stack (len(scales) + full_view, out, out) at loc = (x, y) in [-1, 1].

    Crops that run off the display see empty screen (zeros). full_view adds a
    low-resolution view of the whole display (the coarsest periphery).
    """
    H = img.shape[0]
    pad = max(scales) // 2
    padded = jnp.pad(img, pad)
    cx = (loc[0] + 1) / 2 * H
    cy = (loc[1] + 1) / 2 * H
    chans = []
    for s in scales:
        x0 = jnp.round(cx - s / 2).astype(jnp.int32) + pad
        y0 = jnp.round(cy - s / 2).astype(jnp.int32) + pad
        crop = lax.dynamic_slice(padded, (y0, x0), (s, s))
        f = s // out
        chans.append(crop.reshape(out, f, out, f).mean(axis=(1, 3)))
    if full_view:
        f = H // out
        chans.append(img.reshape(out, f, out, f).mean(axis=(1, 3)))
    return jnp.stack(chans)


__all__ = ["DATA_ROOT", "Design", "GRID", "N_POOL", "spatial_config", "render_block", "render_tiles",
           "make_patch_bank", "tile_size", "load_patch", "load_alien", "load_fractal", "load_object",
           "load_stimulus_set", "N_ALIENS", "N_FRACTALS", "render_single_everywhere", "block_labels", "glimpse", "cell_to_loc"]
