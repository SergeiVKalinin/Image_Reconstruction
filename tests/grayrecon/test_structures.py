"""Tests for shared two-dimensional structure specifications."""

from __future__ import annotations

import numpy as np
import pytest

from grayrecon.structures import (
    NonperiodicStructure,
    PeriodicStructure2D,
    honeycomb_structure,
    square_structure,
    triangular_structure,
)


def test_periodic_structure_accepts_general_lattice_and_motif() -> None:
    structure = PeriodicStructure2D(
        name="general",
        lattice_vectors=np.array(
            [
                [4.0, 0.5],
                [1.0, 3.0],
            ]
        ),
        motif_positions=np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
            ]
        ),
        motif_weights=np.array([1.0, 0.4]),
        metadata={"material": "example"},
    )

    assert structure.name == "general"
    assert structure.lattice_vectors.shape == (2, 2)
    assert structure.motif_positions.shape == (2, 2)
    assert structure.motif_weights.shape == (2,)
    assert structure.number_of_motif_sites == 2
    assert structure.cell_area > 0
    assert structure.metadata["material"] == "example"


def test_periodic_structure_copies_input_arrays() -> None:
    lattice_vectors = np.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ]
    )
    motif_positions = np.array([[0.0, 0.0]])
    motif_weights = np.array([1.0])

    structure = PeriodicStructure2D(
        name="copy_test",
        lattice_vectors=lattice_vectors,
        motif_positions=motif_positions,
        motif_weights=motif_weights,
    )

    lattice_vectors[0, 0] = 100.0
    motif_positions[0, 0] = 100.0
    motif_weights[0] = 100.0

    assert structure.lattice_vectors[0, 0] == pytest.approx(2.0)
    assert structure.motif_positions[0, 0] == pytest.approx(0.0)
    assert structure.motif_weights[0] == pytest.approx(1.0)


def test_periodic_structure_arrays_are_read_only() -> None:
    structure = square_structure(spacing=2.0)

    with pytest.raises(ValueError):
        structure.lattice_vectors[0, 0] = 5.0

    with pytest.raises(ValueError):
        structure.motif_positions[0, 0] = 0.5

    with pytest.raises(ValueError):
        structure.motif_weights[0] = 2.0


def test_periodic_structure_metadata_are_copied_and_read_only() -> None:
    metadata = {"material": "test"}

    structure = square_structure(
        metadata=metadata,
    )

    metadata["material"] = "changed"

    assert structure.metadata["material"] == "test"

    with pytest.raises(TypeError):
        structure.metadata["new_key"] = "new_value"


def test_periodic_structure_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="name must be a non-empty string",
    ):
        PeriodicStructure2D(
            name="   ",
            lattice_vectors=np.eye(2),
            motif_positions=np.array([[0.0, 0.0]]),
            motif_weights=np.array([1.0]),
        )


def test_periodic_structure_rejects_invalid_lattice_shape() -> None:
    with pytest.raises(
        ValueError,
        match="lattice_vectors must have shape",
    ):
        PeriodicStructure2D(
            name="bad_lattice",
            lattice_vectors=np.ones((3, 2)),
            motif_positions=np.array([[0.0, 0.0]]),
            motif_weights=np.array([1.0]),
        )


def test_periodic_structure_rejects_dependent_lattice_vectors() -> None:
    with pytest.raises(
        ValueError,
        match="linearly independent",
    ):
        PeriodicStructure2D(
            name="dependent",
            lattice_vectors=np.array(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                ]
            ),
            motif_positions=np.array([[0.0, 0.0]]),
            motif_weights=np.array([1.0]),
        )


def test_periodic_structure_rejects_invalid_motif_shape() -> None:
    with pytest.raises(
        ValueError,
        match="motif_positions must have shape",
    ):
        PeriodicStructure2D(
            name="bad_motif",
            lattice_vectors=np.eye(2),
            motif_positions=np.array([0.0, 0.0]),
            motif_weights=np.array([1.0]),
        )


def test_periodic_structure_rejects_mismatched_weights() -> None:
    with pytest.raises(
        ValueError,
        match="one value per motif position",
    ):
        PeriodicStructure2D(
            name="bad_weights",
            lattice_vectors=np.eye(2),
            motif_positions=np.array(
                [
                    [0.0, 0.0],
                    [0.5, 0.5],
                ]
            ),
            motif_weights=np.array([1.0]),
        )


def test_periodic_structure_rejects_all_zero_weights() -> None:
    with pytest.raises(
        ValueError,
        match="At least one motif weight must be nonzero",
    ):
        PeriodicStructure2D(
            name="zero_weights",
            lattice_vectors=np.eye(2),
            motif_positions=np.array([[0.0, 0.0]]),
            motif_weights=np.array([0.0]),
        )


def test_square_structure() -> None:
    structure = square_structure(spacing=4.0)

    np.testing.assert_allclose(
        structure.lattice_vectors,
        np.array(
            [
                [4.0, 0.0],
                [0.0, 4.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        structure.motif_positions,
        np.array([[0.0, 0.0]]),
    )
    np.testing.assert_allclose(
        structure.motif_weights,
        np.array([1.0]),
    )
    assert structure.cell_area == pytest.approx(16.0)


def test_triangular_structure() -> None:
    structure = triangular_structure(spacing=4.0)

    np.testing.assert_allclose(
        structure.lattice_vectors,
        np.array(
            [
                [4.0, 0.0],
                [2.0, 2.0 * np.sqrt(3.0)],
            ]
        ),
    )
    assert structure.number_of_motif_sites == 1


def test_honeycomb_structure() -> None:
    structure = honeycomb_structure(nearest_neighbor=2.0)

    assert structure.number_of_motif_sites == 2

    np.testing.assert_allclose(
        structure.motif_positions,
        np.array(
            [
                [0.0, 0.0],
                [2.0 / 3.0, 2.0 / 3.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        structure.motif_weights,
        np.array([1.0, 1.0]),
    )


@pytest.mark.parametrize(
    "constructor, keyword",
    [
        (square_structure, "spacing"),
        (triangular_structure, "spacing"),
        (honeycomb_structure, "nearest_neighbor"),
    ],
)
def test_structure_constructors_reject_nonpositive_spacing(
    constructor,
    keyword: str,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        constructor(**{keyword: 0.0})


def test_nonperiodic_structure() -> None:
    structure = NonperiodicStructure(
        name="amorphous",
        metadata={"description": "smooth background"},
    )

    assert structure.name == "amorphous"
    assert structure.metadata["description"] == "smooth background"


def test_nonperiodic_structure_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="name must be a non-empty string",
    ):
        NonperiodicStructure(name=" ")
