"""1/f^alpha noise patches in a circular aperture (numpy only, so any env can import it)."""

import numpy as np


def make_noise_patch(size, rng, alpha, rms_contrast=0.3):
    """Noise with a 1/f^alpha amplitude spectrum, clipped to [-1, 1].

    alpha = 0 is white noise; alpha ~ 1 matches natural images; larger alpha
    gives smoother, blobbier patches. Patches are RMS-contrast matched so they
    differ only in spectral slope.
    """
    white = rng.standard_normal((size, size))

    fx = np.fft.fftfreq(size)
    fy = np.fft.fftfreq(size)
    radius = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
    radius[0, 0] = 1  # avoid divide-by-zero at DC
    amplitude = radius ** -alpha
    amplitude[0, 0] = 0  # zero mean

    filtered = np.real(np.fft.ifft2(np.fft.fft2(white) * amplitude))
    filtered *= rms_contrast / filtered.std()
    return np.clip(filtered, -1, 1)


def circular_aperture(size, edge_width=0.1):
    """Circular mask with a raised-cosine edge (1 inside, 0 outside)."""
    coords = np.linspace(-1, 1, size)
    r = np.sqrt(coords[None, :] ** 2 + coords[:, None] ** 2)
    inner = 1 - edge_width
    mask = np.ones_like(r)
    ramp = (r > inner) & (r <= 1)
    mask[ramp] = 0.5 * (1 + np.cos(np.pi * (r[ramp] - inner) / edge_width))
    mask[r > 1] = 0
    return mask
