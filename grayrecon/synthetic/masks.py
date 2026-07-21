"""Masks for grayscale reconstruction problems.

Mask convention
---------------
``True`` means that a pixel is observed.
``False`` means that a pixel is missing and should be reconstructed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def central_rectangle_mask(
    shape: tuple[int, int],
    *,
    size: tuple[int, int],
    center: tuple[float, float] | None = None,
) -> BoolArray:
    """Create a mask with a missing central rectangle.

    Parameters
    ----------
    shape
        Image shape as ``(height, width)``.

    size
        Missing-region size as ``(height, width)``.

    center
        Center of the missing region as ``(y, x)``.
        By default, the image center is used.

    Returns
    -------
    numpy.ndarray
        Boolean mask where ``True`` denotes observed pixels.
    """

    height, width = shape
    missing_height, missing_width = size

    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    if missing_height <= 0 or missing_width <= 0:
        raise ValueError("Missing-region dimensions must be positive.")

    if missing_height > height or missing_width > width:
        raise ValueError(
            "The missing region cannot be larger than the image."
        )

    if center is None:
        center_y = height / 2.0
        center_x = width / 2.0
    else:
        center_y, center_x = center

    y_start = int(round(center_y - missing_height / 2.0))
    x_start = int(round(center_x - missing_width / 2.0))

    y_start = min(max(y_start, 0), height - missing_height)
    x_start = min(max(x_start, 0), width - missing_width)

    y_stop = y_start + missing_height
    x_stop = x_start + missing_width

    mask = np.ones(shape, dtype=bool)
    mask[y_start:y_stop, x_start:x_stop] = False

    return mask


def central_square_mask(
    shape: tuple[int, int],
    *,
    size: int,
) -> BoolArray:
    """Create a mask with a missing square at the image center."""

    return central_rectangle_mask(
        shape,
        size=(size, size),
    )


def random_missing_mask(
    shape: tuple[int, int],
    *,
    missing_fraction: float,
    seed: int | None = None,
) -> BoolArray:
    """Create a mask with independently missing random pixels.

    Parameters
    ----------
    shape
        Image shape as ``(height, width)``.

    missing_fraction
        Fraction of pixels to mark as missing. Must lie between zero and one.

    seed
        Random seed for reproducibility.
    """

    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError(
            "missing_fraction must lie between zero and one."
        )

    rng = np.random.default_rng(seed)

    return rng.random(shape) >= missing_fraction


def outpainting_mask(
    target_shape: tuple[int, int],
    *,
    observed_shape: tuple[int, int],
    offset: tuple[int, int] | None = None,
) -> BoolArray:
    """Create a mask for outpainting into a larger target canvas.

    Parameters
    ----------
    target_shape
        Shape of the complete reconstruction canvas.

    observed_shape
        Shape of the observed image region placed inside the target canvas.

    offset
        Top-left location of the observed image as ``(y, x)``.
        By default, the observed image is centered.

    Returns
    -------
    numpy.ndarray
        Boolean mask with ``True`` inside the observed region and ``False``
        in the outpainting region.
    """

    target_height, target_width = target_shape
    observed_height, observed_width = observed_shape

    if target_height <= 0 or target_width <= 0:
        raise ValueError("Target dimensions must be positive.")

    if observed_height <= 0 or observed_width <= 0:
        raise ValueError("Observed dimensions must be positive.")

    if (
        observed_height > target_height
        or observed_width > target_width
    ):
        raise ValueError(
            "The observed image cannot be larger than the target canvas."
        )

    if offset is None:
        y_start = (target_height - observed_height) // 2
        x_start = (target_width - observed_width) // 2
    else:
        y_start, x_start = offset

    y_stop = y_start + observed_height
    x_stop = x_start + observed_width

    if (
        y_start < 0
        or x_start < 0
        or y_stop > target_height
        or x_stop > target_width
    ):
        raise ValueError(
            "The observed region must lie inside the target canvas."
        )

    mask = np.zeros(target_shape, dtype=bool)
    mask[y_start:y_stop, x_start:x_stop] = True

    return mask


def apply_mask(
    image: FloatArray,
    mask: BoolArray,
    *,
    fill_value: float = 0.0,
) -> FloatArray:
    """Apply an observed-pixel mask to a grayscale image.

    The input image is not modified.
    """

    image = np.asarray(image, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional array.")

    if mask.shape != image.shape:
        raise ValueError(
            "mask must have the same shape as image."
        )

    observation = image.copy()
    observation[~mask] = fill_value

    return observation


def embed_observation(
    image: FloatArray,
    target_shape: tuple[int, int],
    *,
    offset: tuple[int, int] | None = None,
    fill_value: float = 0.0,
) -> tuple[FloatArray, BoolArray]:
    """Place an observed image inside a larger outpainting canvas.

    Returns the expanded observation and its corresponding mask.
    """

    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional array.")

    observed_height, observed_width = image.shape
    target_height, target_width = target_shape

    mask = outpainting_mask(
        target_shape,
        observed_shape=image.shape,
        offset=offset,
    )

    if offset is None:
        y_start = (target_height - observed_height) // 2
        x_start = (target_width - observed_width) // 2
    else:
        y_start, x_start = offset

    observation = np.full(
        target_shape,
        fill_value,
        dtype=float,
    )

    observation[
        y_start:y_start + observed_height,
        x_start:x_start + observed_width,
    ] = image

    return observation, mask
