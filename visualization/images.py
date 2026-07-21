"""Shared image-visualization functions.

These functions are independent of a specific reconstruction method.

Supported array conventions
---------------------------
Grayscale image:
    ``(height, width)``

Multilayer or species-resolved image:
    ``(number_of_channels, height, width)``
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def show_image(
    image: ArrayLike,
    *,
    title: str | None = None,
    channel_names: Sequence[str] | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Display a grayscale image or a channel-first multilayer image.

    Parameters
    ----------
    image
        Either a two-dimensional grayscale array or a three-dimensional
        channel-first array.

    title
        Optional figure title.

    channel_names
        Names assigned to individual channels of a multilayer image.

    cmap
        Matplotlib colormap used for each scalar image.

    vmin, vmax
        Optional shared display range.

    show_colorbar
        If ``True``, display a colorbar for every panel.

    Returns
    -------
    figure, axes
        Matplotlib figure and a one-dimensional array of axes.
    """

    array = np.asarray(image)

    if array.ndim == 2:
        channels = array[np.newaxis, ...]
    elif array.ndim == 3:
        channels = array
    else:
        raise ValueError(
            "image must have shape (height, width) or "
            "(number_of_channels, height, width)."
        )

    number_of_channels = channels.shape[0]

    if channel_names is not None:
        if len(channel_names) != number_of_channels:
            raise ValueError(
                "channel_names must contain one name per channel."
            )
        names = list(channel_names)
    else:
        names = [
            f"Channel {index}"
            for index in range(number_of_channels)
        ]

    figure, axes = plt.subplots(
        1,
        number_of_channels,
        figsize=(4.0 * number_of_channels, 4.0),
        squeeze=False,
    )

    axes_array = axes.ravel()

    for index, axis in enumerate(axes_array):
        displayed = axis.imshow(
            channels[index],
            cmap=cmap,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )

        axis.set_xticks([])
        axis.set_yticks([])

        if number_of_channels > 1:
            axis.set_title(names[index])

        if show_colorbar:
            figure.colorbar(
                displayed,
                ax=axis,
                fraction=0.046,
                pad=0.04,
            )

    if title is not None:
        figure.suptitle(title)

    figure.tight_layout()

    return figure, axes_array


def show_grayrecon_case(
    ground_truth: ArrayLike,
    observation: ArrayLike,
    mask: ArrayLike,
    *,
    title: str | None = None,
    cmap: str = "gray",
) -> tuple[Figure, np.ndarray]:
    """Display the main components of a grayscale reconstruction case.

    The panels show:

    1. complete ground truth;
    2. observed image;
    3. observed-pixel mask.

    The function accepts arrays rather than a ``GrayReconCase`` object so
    that the visualization package remains independent of ``grayrecon``.
    """

    ground_truth_array = np.asarray(ground_truth)
    observation_array = np.asarray(observation)
    mask_array = np.asarray(mask, dtype=bool)

    if ground_truth_array.ndim != 2:
        raise ValueError(
            "ground_truth must be a two-dimensional grayscale image."
        )

    if observation_array.shape != ground_truth_array.shape:
        raise ValueError(
            "observation must have the same shape as ground_truth."
        )

    if mask_array.shape != ground_truth_array.shape:
        raise ValueError(
            "mask must have the same shape as ground_truth."
        )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.0),
        squeeze=False,
    )

    axes_array: np.ndarray = axes.ravel()

    panels = [
        (ground_truth_array, "Ground truth", cmap),
        (observation_array, "Observation", cmap),
        (mask_array, "Observed-pixel mask", "gray"),
    ]

    for axis, (panel, panel_title, panel_cmap) in zip(
        axes_array,
        panels,
        strict=True,
    ):
        axis.imshow(
            panel,
            cmap=panel_cmap,
            origin="upper",
        )
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])

    if title is not None:
        figure.suptitle(title)

    figure.tight_layout()

    return figure, axes_array
