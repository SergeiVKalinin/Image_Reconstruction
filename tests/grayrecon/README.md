# GrayRecon tests

Automated tests for the GrayRecon package.

The test suite will verify:

- reproducible generation of synthetic structures, masks, and noise;
- the common input and result formats;
- field-model kernels, latent fields, reconstruction, and sampling;
- motif definition, covariance, matching, adjacency, and reconstruction;
- quasi-MD rendering, force fields, dynamics, capture, and reconstruction;
- compatibility with shared visualization and evaluation tools.

Tests should use small two-dimensional examples and fixed random seeds so they remain fast and reproducible. Larger scientific benchmark calculations will remain in the notebooks.
