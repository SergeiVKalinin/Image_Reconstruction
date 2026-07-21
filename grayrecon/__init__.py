"""GrayRecon: grayscale scientific-image reconstruction.

GrayRecon provides three reconstruction families:

- field models;
- structural motif models;
- quasi-MD models.

Common synthetic problems are defined in ``grayrecon.synthetic``.
Shared visualization and evaluation tools are maintained outside this
package.
"""

from .structures import (
    NonperiodicStructure,
    PeriodicStructure2D,
    honeycomb_structure,
    square_structure,
    triangular_structure,
)


__version__ = "0.1.0"

__all__ = [
    "NonperiodicStructure",
    "PeriodicStructure2D",
    "honeycomb_structure",
    "square_structure",
    "triangular_structure",
]
