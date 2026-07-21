"""Tests for common GrayRecon synthetic data generation."""

import numpy as np

from grayrecon.synthetic import (
    add_gaussian_noise,
    make_lattice_image,
    make_lattice_inpainting_case,
    make_lattice_outpainting_case,
)


def test_square_lattice_image() -> None:
    """The lattice generator should return a valid grayscale image."""

    image, points = make_lattice_image(
        shape=(64, 80),
        lattice="square",
        spacing=8.0,
        sigma=1.0,
    )

    assert image.shape == (64, 80)
    assert image.ndim == 2
    assert points.ndim == 2
    assert points.shape[1] == 2

    assert np.all(np.isfinite(image))
    assert np.all(image >= 0.0)
    assert image.max() <= 1.0
    assert image.max() > 0.0


def test_all_lattice_types() -> None:
    """All initial grayscale lattice types should generate successfully."""

    for lattice in ("square", "triangular", "honeycomb"):
        image, points = make_lattice_image(
            shape=(64, 64),
            lattice=lattice,
            spacing=8.0,
            rotation_degrees=12.0,
        )

        assert image.shape == (64, 64)
        assert len(points) > 0
        assert np.all(np.isfinite(image))


def test_inpainting_mask_convention() -> None:
    """True must mean observed and False must mean missing."""

    case = make_lattice_inpainting_case(
        shape=(64, 64),
        lattice="square",
        missing_size=(16, 20),
        fill_value=0.0,
    )

    assert case.mask is not None
    assert case.mask.shape == (64, 64)

    missing = ~case.mask

    assert missing.sum() == 16 * 20
    assert np.all(case.observation[missing] == 0.0)
    assert np.any(case.observation[case.mask] > 0.0)


def test_ground_truth_is_preserved() -> None:
    """Masking must not modify the complete synthetic image."""

    case = make_lattice_inpainting_case(
        shape=(64, 64),
        missing_size=(16, 16),
    )

    assert case.ground_truth is not None
    assert case.mask is not None

    missing = ~case.mask

    assert np.any(case.ground_truth[missing] > 0.0)
    assert np.all(case.observation[missing] == 0.0)


def test_outpainting_case() -> None:
    """Only the centered measured region should be marked observed."""

    case = make_lattice_outpainting_case(
        target_shape=(80, 96),
        observed_shape=(40, 48),
        lattice="triangular",
    )

    assert case.mask is not None
    assert case.mask.shape == (80, 96)
    assert case.mask.sum() == 40 * 48

    assert np.all(case.observation[~case.mask] == 0.0)
    assert np.any(case.observation[case.mask] > 0.0)


def test_gaussian_noise_is_reproducible() -> None:
    """Identical seeds should generate identical noisy images."""

    image = np.ones((32, 32), dtype=float)

    noisy_1 = add_gaussian_noise(
        image,
        sigma=0.1,
        seed=42,
    )

    noisy_2 = add_gaussian_noise(
        image,
        sigma=0.1,
        seed=42,
    )

    noisy_3 = add_gaussian_noise(
        image,
        sigma=0.1,
        seed=43,
    )

    np.testing.assert_array_equal(noisy_1, noisy_2)
    assert not np.array_equal(noisy_1, noisy_3)


def test_synthetic_metadata() -> None:
    """A generated case should record its construction parameters."""

    case = make_lattice_inpainting_case(
        shape=(64, 64),
        lattice="honeycomb",
        spacing=6.0,
        rotation_degrees=15.0,
        missing_size=(20, 24),
        noise_sigma=0.03,
        seed=7,
    )

    assert case.metadata["case_type"] == "lattice_inpainting"
    assert case.metadata["lattice"] == "honeycomb"
    assert case.metadata["spacing"] == 6.0
    assert case.metadata["rotation_degrees"] == 15.0
    assert case.metadata["missing_size"] == (20, 24)
    assert case.metadata["seed"] == 7
