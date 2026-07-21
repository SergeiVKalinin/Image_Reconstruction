# Evaluation

Shared evaluation tools for grayscale, color, and future multichannel reconstruction packages.

This module will provide common metrics for:

- full-image reconstruction accuracy;
- error inside masked or missing regions;
- observed-region data consistency;
- posterior and ensemble uncertainty;
- structural similarity;
- coordinate and occupancy accuracy when structural representations are available;
- comparison of computational cost across reconstruction methods.

Evaluation is kept outside `grayrecon` so the same metrics can later be used by grayscale, color, and multichannel reconstruction packages.
