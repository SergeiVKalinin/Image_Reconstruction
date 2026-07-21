# Visualization

Shared visualization tools for grayscale, color, and future multichannel reconstruction packages.

This module will provide common functions for:

- displaying input, masked, and reconstructed images;
- comparing reconstructions from different methods;
- plotting reconstruction errors and residuals;
- displaying uncertainty maps;
- displaying posterior or ensemble samples;
- overlaying structural information such as particles, motifs, walls, and masks.

Visualization is kept outside `grayrecon` so the same tools can later be used by grayscale, color, and multichannel reconstruction packages.
