"""FastGS rendering pipeline.

Standard FastGS: fast 3D Gaussian Splatting training / rendering (from the
upstream ``main`` branch).

Key API (lazily imported so that ``import fastgs.rendering`` does not require the
CUDA extensions to be built):

    from fastgs.rendering import GaussianModel, Scene, render_fastgs, training, render_sets
"""

import importlib

_LAZY = {
    "GaussianModel": ("fastgs.rendering.scene", "GaussianModel"),
    "Scene": ("fastgs.rendering.scene", "Scene"),
    "render_fastgs": ("fastgs.rendering.gaussian_renderer", "render_fastgs"),
    "training": ("fastgs.rendering.trainer", "training"),
    "render_sets": ("fastgs.rendering.renderer", "render_sets"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
