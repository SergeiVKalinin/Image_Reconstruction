"""Synthetic two-dimensional grayscale structures.

The functions in this module generate scalar grayscale images without
assigning chemical or atomic identities. Point coordinates may be returned
as auxiliary construction information.

Coordinate convention
---------------------
Point coordinates are stored as ``(x, y)`` pairs.

Image arrays use NumPy indexing:

    image[y, x]
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def render_points(
    points: FloatArray,
    shape: tuple[int, int],
    *,
    sigma: float = 1.0,
    amplitude: float = 1.0,
    normalize: bool = True,
) -> FloatArray:
    """Render unlabeled points as Gaussian peaks in a grayscale image.

    Parameters
    ----------
    points
        Array with shape ``(number_of_points, 2)`` containing ``(x, y)``
        coordinates.

    shape
        Output image shape as ``(height, width)``.

    sigma
        Standard deviation of each Gaussian peak, in pixels.

    amplitude
        Peak amplitude assigned to every point.

    normalize
        If ``True``, divide the image by its maximum value.

    Returns
    -------
    numpy.ndarray
        Two-dimensional grayscale image.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (number_of_points, 2).")

    image = np.zeros((height, width), dtype=float)

    cutoff = max(1, int(np.ceil(4.0 * sigma)))

    for x_center, y_center in points:
        x_min = max(0, int(np.floor(x_center)) - cutoff)
        x_max = min(width, int(np.floor(x_center)) + cutoff + 1)

        y_min = max(0, int(np.floor(y_center)) - cutoff)
        y_max = min(height, int(np.floor(y_center)) + cutoff + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        x = np.arange(x_min, x_max, dtype=float)
        y = np.arange(y_min, y_max, dtype=float)

        xx, yy = np.meshgrid(x, y)

        local_peak = amplitude * np.exp(
            -(
                (xx - x_center) ** 2
                + (yy - y_center) ** 2
            )
            / (2.0 * sigma**2)
        )

        image[y_min:y_max, x_min:x_max] += local_peak

    if normalize and image.max() > 0:
        image = image / image.max()

    return image


def _rotation_matrix(angle_degrees: float) -> FloatArray:
    """Return a two-dimensional rotation matrix."""

    angle = np.deg2rad(angle_degrees)

    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=float,
    )


def _crop_points(
    points: FloatArray,
    shape: tuple[int, int],
    *,
    margin: float = 0.0,
) -> FloatArray:
    """Keep points lying inside the requested image region."""

    height, width = shape

    keep = (
        (points[:, 0] >= -margin)
        & (points[:, 0] < width + margin)
        & (points[:, 1] >= -margin)
        & (points[:, 1] < height + margin)
    )

    return points[keep]


def square_lattice_points(
    shape: tuple[int, int],
    *,
    spacing: float = 8.0,
    rotation_degrees: float = 0.0,
    offset: tuple[float, float] | None = None,
) -> FloatArray:
    """Generate points on a rotated square lattice."""

    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    height, width = shape

    if offset is None:
        offset = (width / 2.0, height / 2.0)

    extent = np.hypot(height, width)
    index_limit = int(np.ceil(extent / spacing)) + 3

    indices = np.arange(-index_limit, index_limit + 1)

    ii, jj = np.meshgrid(indices, indices)

    points = np.column_stack(
        [
            spacing * ii.ravel(),
            spacing * jj.ravel(),
        ]
    )

    rotation = _rotation_matrix(rotation_degrees)
    points = points @ rotation.T
    points += np.asarray(offset, dtype=float)

    return _crop_points(points, shape)


def triangular_lattice_points(
    shape: tuple[int, int],
    *,
    spacing: float = 8.0,
    rotation_degrees: float = 0.0,
    offset: tuple[float, float] | None = None,
) -> FloatArray:
    """Generate points on a triangular lattice.

    This is the two-dimensional Bravais lattice commonly associated with
    hexagonal ordering.
    """

    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    height, width = shape

    if offset is None:
        offset = (width / 2.0, height / 2.0)

    extent = np.hypot(height, width)
    index_limit = int(np.ceil(extent / spacing)) + 3

    indices = np.arange(-index_limit, index_limit + 1)

    ii, jj = np.meshgrid(indices, indices)

    basis_1 = np.array([spacing, 0.0])
    basis_2 = np.array(
        [0.5 * spacing, np.sqrt(3.0) * spacing / 2.0]
    )

    points = (
        ii.ravel()[:, None] * basis_1
        + jj.ravel()[:, None] * basis_2
    )

    rotation = _rotation_matrix(rotation_degrees)
    points = points @ rotation.T
    points += np.asarray(offset, dtype=float)

    return _crop_points(points, shape)


def honeycomb_lattice_points(
    shape: tuple[int, int],
    *,
    nearest_neighbor: float = 6.0,
    rotation_degrees: float = 0.0,
    offset: tuple[float, float] | None = None,
) -> FloatArray:
    """Generate an unlabeled honeycomb lattice.

    The two sublattices are combined into one grayscale point set. They do
    not represent different atomic identities.
    """

    if nearest_neighbor <= 0:
        raise ValueError("nearest_neighbor must be positive.")

    height, width = shape

    if offset is None:
        offset = (width / 2.0, height / 2.0)

    lattice_constant = np.sqrt(3.0) * nearest_neighbor
    extent = np.hypot(height, width)
    index_limit = int(np.ceil(extent / lattice_constant)) + 4

    indices = np.arange(-index_limit, index_limit + 1)
    ii, jj = np.meshgrid(indices, indices)

    basis_1 = np.array(
        [lattice_constant, 0.0]
    )
    basis_2 = np.array(
        [
            lattice_constant / 2.0,
            3.0 * nearest_neighbor / 2.0,
        ]
    )

    base_points = (
        ii.ravel()[:, None] * basis_1
        + jj.ravel()[:, None] * basis_2
    )

    sublattice_shift = np.array(
        [0.0, nearest_neighbor]
    )

    points = np.vstack(
        [
            base_points,
            base_points + sublattice_shift,
        ]
    )

    rotation = _rotation_matrix(rotation_degrees)
    points = points @ rotation.T
    points += np.asarray(offset, dtype=float)

    return _crop_points(points, shape)


def make_lattice_image(
    shape: tuple[int, int] = (128, 128),
    *,
    lattice: Literal[
        "square",
        "triangular",
        "honeycomb",
    ] = "square",
    spacing: float = 8.0,
    rotation_degrees: float = 0.0,
    sigma: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Generate a grayscale lattice image and its unlabeled coordinates.

    Parameters
    ----------
    shape
        Image shape as ``(height, width)``.

    lattice
        One of ``"square"``, ``"triangular"``, or ``"honeycomb"``.

    spacing
        For square and triangular lattices, the primitive nearest spacing.
        For the honeycomb lattice, the nearest-neighbor distance.

    rotation_degrees
        Counterclockwise lattice rotation.

    sigma
        Gaussian rendering width.

    amplitude
        Common grayscale peak amplitude.

    Returns
    -------
    image, points
        The grayscale image and unlabeled ``(x, y)`` point coordinates.
    """

    if lattice == "square":
        points = square_lattice_points(
            shape,
            spacing=spacing,
            rotation_degrees=rotation_degrees,
        )

    elif lattice == "triangular":
        points = triangular_lattice_points(
            shape,
            spacing=spacing,
            rotation_degrees=rotation_degrees,
        )

    elif lattice == "honeycomb":
        points = honeycomb_lattice_points(
            shape,
            nearest_neighbor=spacing,
            rotation_degrees=rotation_degrees,
        )

    else:
        raise ValueError(
            "lattice must be 'square', 'triangular', or 'honeycomb'."
        )

    image = render_points(
        points,
        shape,
        sigma=sigma,
        amplitude=amplitude,
        normalize=True,
    )

    return image, points
