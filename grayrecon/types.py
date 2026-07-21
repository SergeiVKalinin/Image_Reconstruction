"""Common data structures for GrayRecon.

The reconstruction families may use different internal models and APIs,
but they share a minimal representation of reconstruction problems and
results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]


@dataclass
class GrayReconCase:
    """A two-dimensional grayscale reconstruction problem.

    Parameters
    ----------
    observation
        Observed grayscale image.

    mask
        Boolean array with the same shape as ``observation``.

        ``True`` means that the pixel is observed.
        ``False`` means that the pixel is missing and should be reconstructed.

        If ``mask`` is ``None``, the full image is treated as observed.

    ground_truth
        Optional complete image used for synthetic validation.

    auxiliary
        Optional method-independent latent information, such as known particle
        coordinates, lattice vectors, wall geometry, or component images.

    metadata
        Description of how the case was generated.
    """

    observation: FloatArray
    mask: BoolArray | None = None
    ground_truth: FloatArray | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array dimensions and shapes."""

        self.observation = np.asarray(self.observation)

        if self.observation.ndim != 2:
            raise ValueError(
                "GrayRecon observations must be two-dimensional grayscale arrays."
            )

        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)

            if self.mask.shape != self.observation.shape:
                raise ValueError(
                    "The mask must have the same shape as the observation."
                )

        if self.ground_truth is not None:
            self.ground_truth = np.asarray(self.ground_truth)

            if self.ground_truth.shape != self.observation.shape:
                raise ValueError(
                    "The ground truth must have the same shape as the observation."
                )


@dataclass
class GrayReconResult:
    """Common top-level output from a GrayRecon reconstruction method.

    Method-specific internal quantities remain available through
    ``parameters``, ``state``, and ``history``.
    """

    reconstruction: FloatArray
    uncertainty: FloatArray | None = None
    samples: FloatArray | None = None
    residual: FloatArray | None = None

    parameters: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate common result-array shapes."""

        self.reconstruction = np.asarray(self.reconstruction)

        if self.reconstruction.ndim != 2:
            raise ValueError(
                "GrayRecon reconstructions must be two-dimensional arrays."
            )

        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty)

            if self.uncertainty.shape != self.reconstruction.shape:
                raise ValueError(
                    "The uncertainty map must match the reconstruction shape."
                )

        if self.residual is not None:
            self.residual = np.asarray(self.residual)

            if self.residual.shape != self.reconstruction.shape:
                raise ValueError(
                    "The residual must match the reconstruction shape."
                )

        if self.samples is not None:
            self.samples = np.asarray(self.samples)

            expected_shape = self.reconstruction.shape

            if (
                self.samples.ndim != 3
                or self.samples.shape[1:] != expected_shape
            ):
                raise ValueError(
                    "Samples must have shape "
                    "(number_of_samples, height, width)."
                )
