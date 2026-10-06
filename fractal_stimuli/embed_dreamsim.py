"""DreamSim distances between all candidate fractals (generate_fractals.py output).

DreamSim (Fu et al., 2023) is an ensemble of DINO, CLIP and OpenCLIP ViT-B/16 features,
tuned on human similarity judgements; distance = 1 - cosine similarity of the embeddings
(0 = identical, larger = more different). Each fractal is shown on the task's grey
background (RGB 128) before embedding, as participants see it.

Runs in the separate `dreamsim` env (PyTorch; kept apart from kernelbehav). The weights
(~3 GB) are cached in ~/Documents/data/catlearn_eeg/models/dreamsim.

Writes to the candidates' parent folder (fractal_pool/):
  dreamsim_embeddings.npy   (N, D) float32, row k-1 = candidate k
  dreamsim_distances.npy    (N, N) float32

  ~/miniforge3/envs/dreamsim/bin/python fractal_stimuli/embed_dreamsim.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from dreamsim import dreamsim
from PIL import Image

POOL = Path.home() / "Documents" / "data" / "catlearn_eeg" / "fractal_pool"
CACHE = Path.home() / "Documents" / "data" / "catlearn_eeg" / "models" / "dreamsim"
BACKGROUND = (128, 128, 128, 255)


def on_grey(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, BACKGROUND)
    bg.alpha_composite(im)
    return bg.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    files = sorted((args.pool / "candidates").glob("*.png"), key=lambda p: int(p.stem))
    assert [int(p.stem) for p in files] == list(range(1, len(files) + 1)), "candidates must be 1.png ... N.png"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True, device=device, cache_dir=str(CACHE), dreamsim_type="ensemble")
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(files), args.batch):
            batch = torch.cat([preprocess(on_grey(f)) for f in files[i:i + args.batch]]).to(device)
            embeddings.append(model.embed(batch).float().cpu())
            print(f"  embedded {min(i + args.batch, len(files))} / {len(files)}", flush=True)
    e = torch.cat(embeddings)
    e = e / e.norm(dim=1, keepdim=True)
    dist = (1 - e @ e.T).clamp(min=0).numpy().astype(np.float32)
    np.fill_diagonal(dist, 0)
    np.save(args.pool / "dreamsim_embeddings.npy", e.numpy().astype(np.float32))
    np.save(args.pool / "dreamsim_distances.npy", dist)
    off = dist[np.triu_indices(len(files), 1)]
    print(f"{len(files)} candidates, embedding dim {e.shape[1]}; distances: median {np.median(off):.3f}, "
          f"5-95% {np.percentile(off, 5):.3f}-{np.percentile(off, 95):.3f}")


if __name__ == "__main__":
    main()
