"""FastGS: Training 3D Gaussian Splatting in 100 Seconds.

This package bundles two pipelines that share a common FastGS lineage but have
diverged enough to live side by side:

* :mod:`fastgs.rendering` -- standard FastGS, fast 3DGS training / rendering
  (from the upstream ``main`` branch).
* :mod:`fastgs.surface` -- Fast-PGSR, PGSR-based surface reconstruction / mesh
  extraction (from the upstream ``fast-pgsr`` branch).

Both pipelines depend on CUDA extensions (``diff_gaussian_rasterization_fastgs``,
``diff_plane_rasterization``, ``fused_ssim``, ``simple_knn``) that must be compiled
against the locally installed PyTorch / CUDA toolchain. See the project README for
installation details.
"""

__version__ = "0.1.0"

__all__ = ["rendering", "surface", "__version__"]
