"""Shared two-dimensional grayscale structure specifications.

This module defines neutral structural descriptions that can be used by
field models, motif models, quasi-MD models, and synthetic-data generators.

The classes in this module describe reference structures only. They do not
contain reconstruction algorithms, inferred orientations, wall geometry,
chemical identities, or method-specific latent variables.

Coordinate conventions
----------------------
Lattice vectors are stored as Cartesian ``(x, y)`` row vectors:

    [[a1_x, a1_y],
     [a2_x, a2_y]]

Motif positions are fractional coordinates ``(u, v)`` inside the unit cell.
The corresponding Cartesian position is

    r = u * a1 + v * a2

Motif weights are scalar grayscale weights. They do not encode chemical or
atomic identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _readonly_float_array(
    value: object,
    *,
    name: str,
) -> FloatArray:
    """Return an independent, read-only float64 array."""

    array = np.array(value, dtype=np.float64, copy=True)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    array.setflags(write=False)
    return array


def _positive_finite(value: float, *, name: str) -> float:
    """Validate and return a positive finite scalar."""

    scalar = float(value)

    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")

    if scalar <= 0:
        raise ValueError(f"{name} must be positive.")

    return scalar


@dataclass(frozen=True, slots=True)
class PeriodicStructure2D:
    """Reference description of a periodic grayscale structure.

    Parameters
    ----------
    name
        Human-readable structure name.
    lattice_vectors
        Array with shape ``(2, 2)``. The rows are the two Cartesian
        primitive lattice vectors ``a1`` and ``a2`` in ``(x, y)`` order.
    motif_positions
        Array with shape ``(number_of_sites, 2)`` containing fractional
        motif coordinates ``(u, v)``.
    motif_weights
        One-dimensional array of scalar grayscale weights, with one value
        per motif position. Weights may be signed because the class is
        intended for general grayscale contrast rather than chemical labels.
    metadata
        Optional descriptive information such as material, zone axis,
        source, or notes. Metadata are copied and exposed as a read-only
        mapping.

    Notes
    -----
    Rotation, translation, deformation, image amplitude, and fitted model
    parameters are deliberately excluded. They describe a particular state
    of the reference structure in an image and belong to the reconstruction
    model or result.
    """

    name: str
    lattice_vectors: FloatArray
    motif_positions: FloatArray
    motif_weights: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("name must be a non-empty string.")

        lattice_vectors = _readonly_float_array(
            self.lattice_vectors,
            name="lattice_vectors",
        )
        motif_positions = _readonly_float_array(
            self.motif_positions,
            name="motif_positions",
        )
        motif_weights = _readonly_float_array(
            self.motif_weights,
            name="motif_weights",
        )

        if lattice_vectors.shape != (2, 2):
            raise ValueError(
                "lattice_vectors must have shape (2, 2)."
            )

        vector_scale = max(
            float(
                np.linalg.norm(lattice_vectors[0])
                * np.linalg.norm(lattice_vectors[1])
            ),
            1.0,
        )
        cell_area = abs(float(np.linalg.det(lattice_vectors)))

        if cell_area <= 1.0e-12 * vector_scale:
            raise ValueError(
                "lattice_vectors must be linearly independent."
            )

        if motif_positions.ndim != 2 or motif_positions.shape[1] != 2:
            raise ValueError(
                "motif_positions must have shape "
                "(number_of_sites, 2)."
            )

        if motif_positions.shape[0] == 0:
            raise ValueError(
                "motif_positions must contain at least one site."
            )

        if motif_weights.ndim != 1:
            raise ValueError(
                "motif_weights must be one-dimensional."
            )

        if motif_weights.shape[0] != motif_positions.shape[0]:
            raise ValueError(
                "motif_weights must contain one value per "
                "motif position."
            )

        if not np.any(motif_weights != 0):
            raise ValueError(
                "At least one motif weight must be nonzero."
            )

        metadata = MappingProxyType(dict(self.metadata))

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "lattice_vectors",
            lattice_vectors,
        )
        object.__setattr__(
            self,
            "motif_positions",
            motif_positions,
        )
        object.__setattr__(
            self,
            "motif_weights",
            motif_weights,
        )
        object.__setattr__(self, "metadata", metadata)

    @property
    def number_of_motif_sites(self) -> int:
        """Return the number of grayscale motif sites."""

        return int(self.motif_positions.shape[0])

    @property
    def cell_area(self) -> float:
        """Return the absolute projected unit-cell area."""

        return abs(float(np.linalg.det(self.lattice_vectors)))


@dataclass(frozen=True, slots=True)
class NonperiodicStructure:
    """Reference description of a nonperiodic grayscale component.

    This lightweight class can represent vacuum, amorphous material,
    contamination, unresolved matrix contrast, or another component that
    does not have a periodic lattice description.
    """

    name: str = "nonperiodic"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("name must be a non-empty string.")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def square_structure(
    spacing: float = 1.0,
    *,
    name: str = "square",
    metadata: Mapping[str, Any] | None = None,
) -> PeriodicStructure2D:
    """Create a one-site square-lattice structure."""

    spacing = _positive_finite(spacing, name="spacing")

    return PeriodicStructure2D(
        name=name,
        lattice_vectors=np.array(
            [
                [spacing, 0.0],
                [0.0, spacing],
            ],
            dtype=float,
        ),
        motif_positions=np.array(
            [
                [0.0, 0.0],
            ],
            dtype=float,
        ),
        motif_weights=np.array([1.0], dtype=float),
        metadata={} if metadata is None else metadata,
    )


def triangular_structure(
    spacing: float = 1.0,
    *,
    name: str = "triangular",
    metadata: Mapping[str, Any] | None = None,
) -> PeriodicStructure2D:
    """Create a one-site triangular Bravais-lattice structure."""

    spacing = _positive_finite(spacing, name="spacing")

    return PeriodicStructure2D(
        name=name,
        lattice_vectors=np.array(
            [
                [spacing, 0.0],
                [
                    0.5 * spacing,
                    np.sqrt(3.0) * spacing / 2.0,
                ],
            ],
            dtype=float,
        ),
        motif_positions=np.array(
            [
                [0.0, 0.0],
            ],
            dtype=float,
        ),
        motif_weights=np.array([1.0], dtype=float),
        metadata={} if metadata is None else metadata,
    )


def honeycomb_structure(
    nearest_neighbor: float = 1.0,
    *,
    name: str = "honeycomb",
    metadata: Mapping[str, Any] | None = None,
) -> PeriodicStructure2D:
    """Create an unlabeled two-site honeycomb structure.

    Both motif sites have the same grayscale weight. They do not represent
    different chemical or atomic species.
    """

    nearest_neighbor = _positive_finite(
        nearest_neighbor,
        name="nearest_neighbor",
    )
    lattice_constant = np.sqrt(3.0) * nearest_neighbor

    return PeriodicStructure2D(
        name=name,
        lattice_vectors=np.array(
            [
                [lattice_constant, 0.0],
                [
                    lattice_constant / 2.0,
                    3.0 * nearest_neighbor / 2.0,
                ],
            ],
            dtype=float,
        ),
        motif_positions=np.array(
            [
                [0.0, 0.0],
                [2.0 / 3.0, 2.0 / 3.0],
            ],
            dtype=float,
        ),
        motif_weights=np.array(
            [
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        metadata={} if metadata is None else metadata,
    )
