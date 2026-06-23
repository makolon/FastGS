"""FastGS surface reconstruction pipeline (Fast-PGSR).

PGSR-based surface reconstruction / mesh extraction on top of FastGS (from the
upstream ``fast-pgsr`` branch).

Key API (lazily imported so that ``import fastgs.surface`` does not require the
CUDA extensions to be built):

    from fastgs.surface import GaussianModel, Scene, render, render_fastgs, training, render_sets
"""

import importlib

_LAZY = {
    "GaussianModel": ("fastgs.surface.scene", "GaussianModel"),
    "Scene": ("fastgs.surface.scene", "Scene"),
    "render": ("fastgs.surface.gaussian_renderer", "render"),
    "render_fastgs": ("fastgs.surface.gaussian_renderer", "render_fastgs"),
    "AppModel": ("fastgs.surface.scene.app_model", "AppModel"),
    "training": ("fastgs.surface.trainer", "training"),
    "render_sets": ("fastgs.surface.mesh", "render_sets"),
    "post_process_mesh": ("fastgs.surface.mesh", "post_process_mesh"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
