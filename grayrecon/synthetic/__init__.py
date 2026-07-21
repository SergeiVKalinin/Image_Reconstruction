"""Synthetic grayscale reconstruction problems.

This subpackage provides common two-dimensional structures, masks,
noise models, and benchmark cases used by all GrayRecon methods.
"""

from .cases import (
    make_lattice_inpainting_case,
    make_lattice_outpainting_case,
)
from .masks import (
    apply_mask,
    central_rectangle_mask,
    central_square_mask,
    embed_observation,
    outpainting_mask,
    random_missing_mask,
)
from .noise import (
    add_gaussian_noise,
    add_poisson_noise,
)
from .structures import (
    honeycomb_lattice_points,
    make_lattice_image,
    render_points,
    square_lattice_points,
    triangular_lattice_points,
)


__all__ = [
    "add_gaussian_noise",
    "add_poisson_noise",
    "apply_mask",
    "central_rectangle_mask",
    "central_square_mask",
    "embed_observation",
    "honeycomb_lattice_points",
    "make_lattice_image",
    "make_lattice_inpainting_case",
    "make_lattice_outpainting_case",
    "outpainting_mask",
    "random_missing_mask",
    "render_points",
    "square_lattice_points",
    "triangular_lattice_points",
]
