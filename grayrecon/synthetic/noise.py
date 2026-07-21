"""Noise models for synthetic grayscale reconstruction problems.

All functions return new arrays and do not modify the input image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def add_gaussian_noise(
    image: FloatArray,
    *,
    sigma: float = 0.05,
    seed: int | None = None,
    clip: tuple[float, float] | None = None,
) -> FloatArray:
    """Add independent Gaussian noise to a grayscale image.

    Parameters
    ----------
    image
        Two-dimensional grayscale image.

    sigma
        Standard deviation of the Gaussian noise.

    seed
        Random seed for reproducibility.

    clip
        Optional ``(minimum, maximum)`` range applied after adding noise.

    Returns
    -------
    numpy.ndarray
        Noisy grayscale image.
    """

    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")

    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional array.")

    rng = np.random.default_rng(seed)

    noisy = image + rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    )

    if clip is not None:
        lower, upper = clip

        if lower >= upper:
            raise ValueError(
                "The lower clipping bound must be less than the upper bound."
            )

        noisy = np.clip(noisy, lower, upper)

    return noisy


def add_poisson_noise(
    image: FloatArray,
    *,
    peak_counts: float = 100.0,
    seed: int | None = None,
    clip: tuple[float, float] | None = None,
) -> FloatArray:
    """Add signal-dependent Poisson counting noise.

    The image is interpreted as a nonnegative relative intensity field.
    ``peak_counts`` specifies the expected count at intensity one.

    Parameters
    ----------
    image
        Two-dimensional nonnegative grayscale image.

    peak_counts
        Expected count corresponding to unit intensity.

    seed
        Random seed for reproducibility.

    clip
        Optional output clipping range.

    Returns
    -------
    numpy.ndarray
        Poisson-noisy image on the original intensity scale.
    """

    if peak_counts <= 0:
        raise ValueError("peak_counts must be positive.")

    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional array.")

    if np.any(image < 0):
        raise ValueError(
            "Poisson noise requires a nonnegative image."
        )

    rng = np.random.default_rng(seed)

    counts = rng.poisson(image * peak_counts)
    noisy = counts.astype(float) / peak_counts

    if clip is not None:
        lower, upper = clip

        if lower >= upper:
            raise ValueError(
                "The lower clipping bound must be less than the upper bound."
            )

        noisy = np.clip(noisy, lower, upper)

    return noisy
