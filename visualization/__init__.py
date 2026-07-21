"""Shared visualization tools.

This package contains plotting utilities intended for reuse across
grayscale, species-resolved, and future multichannel reconstruction
packages.
"""

from .images import (
    show_grayrecon_case,
    show_image,
)


__all__ = [
    "show_grayrecon_case",
    "show_image",
]
