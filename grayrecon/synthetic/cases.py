"""Reusable synthetic GrayRecon benchmark cases.

This module combines structures, masks, and noise into complete
``GrayReconCase`` objects that can be used by field, motif, and quasi-MD
reconstruction methods.
"""

from __future__ import annotations

from typing import Literal

from grayrecon.types import GrayReconCase

from .masks import (
    apply_mask,
    central_rectangle_mask,
    outpainting_mask,
)
from .noise import add_gaussian_noise
from .structures import make_lattice_image


LatticeName = Literal[
    "square",
    "triangular",
    "honeycomb",
]


def make_lattice_inpainting_case(
    *,
    shape: tuple[int, int] = (128, 128),
    lattice: LatticeName = "square",
    spacing: float = 8.0,
    rotation_degrees: float = 0.0,
    peak_sigma: float = 1.0,
    missing_size: tuple[int, int] = (32, 32),
    noise_sigma: float = 0.0,
    seed: int | None = None,
    fill_value: float = 0.0,
) -> GrayReconCase:
    """Create a grayscale lattice inpainting problem.

    The same case can be passed to field, motif, and quasi-MD
    reconstruction methods.

    Parameters
    ----------
    shape
        Complete image shape as ``(height, width)``.

    lattice
        Lattice type: ``"square"``, ``"triangular"``, or ``"honeycomb"``.

    spacing
        Lattice spacing in pixels.

    rotation_degrees
        Counterclockwise lattice rotation.

    peak_sigma
        Gaussian width used to render each unlabeled lattice point.

    missing_size
        Size of the central missing rectangle as ``(height, width)``.

    noise_sigma
        Standard deviation of Gaussian noise applied before masking.

    seed
        Random seed controlling synthetic noise.

    fill_value
        Value assigned to missing pixels in the observation.
    """

    ground_truth, points = make_lattice_image(
        shape=shape,
        lattice=lattice,
        spacing=spacing,
        rotation_degrees=rotation_degrees,
        sigma=peak_sigma,
    )

    if noise_sigma > 0:
        measured_image = add_gaussian_noise(
            ground_truth,
            sigma=noise_sigma,
            seed=seed,
        )
    else:
        measured_image = ground_truth.copy()

    mask = central_rectangle_mask(
        shape,
        size=missing_size,
    )

    observation = apply_mask(
        measured_image,
        mask,
        fill_value=fill_value,
    )

    return GrayReconCase(
        observation=observation,
        mask=mask,
        ground_truth=ground_truth,
        auxiliary={
            "points": points,
        },
        metadata={
            "case_type": "lattice_inpainting",
            "lattice": lattice,
            "shape": shape,
            "spacing": spacing,
            "rotation_degrees": rotation_degrees,
            "peak_sigma": peak_sigma,
            "missing_size": missing_size,
            "noise_sigma": noise_sigma,
            "seed": seed,
            "fill_value": fill_value,
        },
    )


def make_lattice_outpainting_case(
    *,
    target_shape: tuple[int, int] = (160, 160),
    observed_shape: tuple[int, int] = (96, 96),
    lattice: LatticeName = "square",
    spacing: float = 8.0,
    rotation_degrees: float = 0.0,
    peak_sigma: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int | None = None,
    fill_value: float = 0.0,
) -> GrayReconCase:
    """Create a grayscale lattice outpainting problem.

    The complete lattice is generated over the target canvas. Only the
    centered observed region is retained in the input image.
    """

    ground_truth, points = make_lattice_image(
        shape=target_shape,
        lattice=lattice,
        spacing=spacing,
        rotation_degrees=rotation_degrees,
        sigma=peak_sigma,
    )

    mask = outpainting_mask(
        target_shape,
        observed_shape=observed_shape,
    )

    if noise_sigma > 0:
        measured_image = add_gaussian_noise(
            ground_truth,
            sigma=noise_sigma,
            seed=seed,
        )
    else:
        measured_image = ground_truth.copy()

    observation = apply_mask(
        measured_image,
        mask,
        fill_value=fill_value,
    )

    return GrayReconCase(
        observation=observation,
        mask=mask,
        ground_truth=ground_truth,
        auxiliary={
            "points": points,
        },
        metadata={
            "case_type": "lattice_outpainting",
            "lattice": lattice,
            "target_shape": target_shape,
            "observed_shape": observed_shape,
            "spacing": spacing,
            "rotation_degrees": rotation_degrees,
            "peak_sigma": peak_sigma,
            "noise_sigma": noise_sigma,
            "seed": seed,
            "fill_value": fill_value,
        },
    )
