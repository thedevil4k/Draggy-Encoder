"""Catalog of AI engines decoupled from ``src/dependencies.py``.

Why a separate module?

``src/dependencies.py`` is the registry for program prerequisites
(FFmpeg, runtimes). It is treated as "if these are not present, the
program cannot start" infrastructure. AI engines are different: they
are *optional* quality-restoration building blocks organised by
domain (category). Mixing them into the binary/runtime registry would
overload its schema (binary vs model semantics are different) and
force the existing ``_DepRow`` widget to keep growing.

This module exposes:

    CATALOG              - hand-curated list of known engines.
    split_engine_id(id)  - tuple (category, family, variant).
    get_engine_dir(id)   - path under bin/models/<cat>/<family>-<variant>.
    iter_catalog()       - yields each entry with computed metadata.
    list_installed(cat)  - catalog entries whose state.installed is True.
    list_enabled(cat)    - catalog entries whose state.installed AND state.enabled.
    get_engine_meta(id)  - cheap lookup by engine_id (returns the catalog
                           entry merged with persisted state).
    CATALOG_BY_ID        - dict {engine_id: entry} for O(1) lookups.

The UI consumes this list via ``iter_catalog`` / ``list_enabled``;
the on-disk state (installed+enabled+converted) is read via
``src.ai.engines_state``.
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Optional

from src.config import get_models_dir


# Slug validator: pipeline-safe identifier ``[a-z0-9]+(-[a-z0-9]+)*``.
# Why: ``get_engine_dir`` rejoins the slug as a path leaf under
# ``bin/models/<cat>/...``. If a future user-editable id ever contains
# ``..`` or path separators, it becomes a traversal vector that
# escapes ``bin/models``. Locking the invariant in now while the
# catalog is still a literal set is cheap and prevents that class
# of bug from ever shipping.
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _assert_slug(engine_id) -> str:
    if not isinstance(engine_id, str) or not _SLUG_RE.match(engine_id):
        raise ValueError(
            f"Invalid engine_id {engine_id!r}: must match {_SLUG_RE.pattern}"
        )
    return engine_id


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
# Schema (every field is required for each entry, even if empty):
#
#   engine_id:        str, unique ID "<category>-<family>-<variant>".
#                     Slug-safe: lowercase, [a-z0-9-] only, no spaces.
#   display_name:     str, human label for the row title.
#   category:         str, single word aligned to the workflow node.
#                     Currently: Upscale, Interpolate, Colorize, Denoise,
#                     FaceRestore. Anything else groups under "Other".
#   hf_repo_id:       str, HuggingFace "<owner>/<repo>" pair.
#   hf_url:           str, full HTTPS clone URL.
#   size_estimate:    str, human-readable on-disk size ("~67 MB").
#   sample_shape:     str, declarative input/output shape description
#                     ("1x3xHxW -> 1x3x(4H)x(4W)").
#   license:          str, SPDX-ish identifier (Apache-2.0, MIT, BSD-3,
#                     Custom). Used to surface licensing in the UI.
#   popularity_score: int 0-100, hand-curated proxy for "how many users
#                     rely on this repo on HuggingFace". Static for now;
#                     future: refresh via HF API on tab open.
#   quality_score:    int 0-100, hand-curated proxy for perceptual
#                     quality output. Static for now.
#   is_implemented:   bool, whether the Add/Convert buttons are wired
#                     today. False entries render as COMING SOON with
#                     disabled actions.
#   sentinel:         str, file expected inside the cloned dir to
#                     confirm a successful install. Used by the row to
#                     decide installed=true after a clone finishes.
#   lfs_warning:      bool, True if the repo uses Git LFS (heavy files
#                     like model.safetensors are NOT in the regular
#                     clone). We surface a tip in the install tooltip.
# ---------------------------------------------------------------------------

CATALOG = [
    # --- Upscaling --------------------------------------------------------
    {
        "engine_id": "upscale-realesrgan-x4plus",
        "display_name": "Real-ESRGAN x4+",
        "category": "Upscale",
        "hf_repo_id": "ai-forever/Real-ESRGAN",
        "hf_url": "https://huggingface.co/ai-forever/Real-ESRGAN",
        "size_estimate": "~67 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "BSD-3-Clause",
        "popularity_score": 92,
        "quality_score": 88,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
    },
    {
        "engine_id": "upscale-realcugan-pro",
        "display_name": "Real-CUGAN Pro (anime)",
        "category": "Upscale",
        "hf_repo_id": "smnorini/Real_CUGAN_4x",
        "hf_url": "https://huggingface.co/smnorini/Real_CUGAN_4x",
        "size_estimate": "~5.4 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "MIT",
        "popularity_score": 65,
        "quality_score": 85,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
    },
    {
        "engine_id": "upscale-swinir-real",
        "display_name": "SwinIR (real-world SR)",
        "category": "Upscale",
        "hf_repo_id": "mikestealth/SwinIR",
        "hf_url": "https://huggingface.co/mikestealth/SwinIR",
        "size_estimate": "~136 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "Apache-2.0",
        "popularity_score": 55,
        "quality_score": 82,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
    },
    # --- Upscaling (fast / lightweight) -----------------------------------
    {
        "engine_id": "upscale-realesr-animevideov3",
        "display_name": "RealESR AnimeVideo v3 (fast, x4)",
        "category": "Upscale",
        "hf_repo_id": "leonelhs/realesrgan",
        "hf_url": "https://huggingface.co/leonelhs/realesrgan",
        "size_estimate": "~2.4 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "BSD-3-Clause",
        "popularity_score": 88,
        "quality_score": 78,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
        "similar_models_url": "https://openmodeldb.info/models/4x-realesr-animevideo-v3",
    },
    {
        "engine_id": "upscale-realesr-general-x4v3",
        "display_name": "RealESR General v3 (fast, x4)",
        "category": "Upscale",
        "hf_repo_id": "axhello/upscale",
        "hf_url": "https://huggingface.co/axhello/upscale",
        "size_estimate": "~4.7 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "BSD-3-Clause",
        "popularity_score": 75,
        "quality_score": 76,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
        "similar_models_url": "https://openmodeldb.info/models/4x-realesr-general",
    },
    {
        "engine_id": "upscale-realesrgan-anime-6b",
        "display_name": "RealESRGAN Anime 6B (x4)",
        "category": "Upscale",
        "hf_repo_id": "amd/realesrgan-x4plus-anime-6b",
        "hf_url": "https://huggingface.co/amd/realesrgan-x4plus-anime-6b",
        "size_estimate": "~17 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "BSD-3-Clause",
        "popularity_score": 70,
        "quality_score": 82,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
        "similar_models_url": "https://openmodeldb.info/models/4x-realesrgan-anime-6b",
    },
    {
        "engine_id": "upscale-omnisr-x4",
        "display_name": "OmniSR x4 (ultra-fast)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://github.com/Francis0625/Omni-SR",
        "size_estimate": "~1.7 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "Apache-2.0",
        "popularity_score": 50,
        "quality_score": 74,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
        "similar_models_url": "https://openmodeldb.info/?search=omnisr",
    },
    # --- FSR (FidelityFX Super Resolution) — no model download needed -------
    {
        "engine_id": "upscale-fsr-2x",
        "display_name": "FSR 1.0 x2 (CPU, no download)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (algorithmic)",
        "sample_shape": "1x3xHxW -> 1x3x(2H)x(2W)",
        "license": "MIT",
        "popularity_score": 85,
        "quality_score": 70,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
    },
    {
        "engine_id": "upscale-fsr-3x",
        "display_name": "FSR 1.0 x3 (CPU, no download)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (algorithmic)",
        "sample_shape": "1x3xHxW -> 1x3x(3H)x(3W)",
        "license": "MIT",
        "popularity_score": 70,
        "quality_score": 68,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
    },
    {
        "engine_id": "upscale-fsr-4x",
        "display_name": "FSR 1.0 x4 (CPU, no download)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (algorithmic)",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "MIT",
        "popularity_score": 80,
        "quality_score": 72,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
    },

    # --- FSR 2.0 (via FFmpeg libplacebo) — requires FFmpeg full build -------
    {
        "engine_id": "upscale-fsr-2x-v2",
        "display_name": "FSR 2.0 x2 (GPU via libplacebo)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(2H)x(2W)",
        "license": "MIT",
        "popularity_score": 90,
        "quality_score": 82,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
    },
    {
        "engine_id": "upscale-fsr-3x-v2",
        "display_name": "FSR 2.0 x3 (GPU via libplacebo)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(3H)x(3W)",
        "license": "MIT",
        "popularity_score": 80,
        "quality_score": 80,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
    },
    {
        "engine_id": "upscale-fsr-4x-v2",
        "display_name": "FSR 2.0 x4 (GPU via libplacebo)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "MIT",
        "popularity_score": 85,
        "quality_score": 78,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
    },

# --- FSR 3.1 (FSR 2.0 upscaling + frame interpolation via libplacebo) -------
    # FSR 3.1 decouples FSR upscaling from frame generation (like the real
    # FSR 3.1 API), allowing each feature independently.  Frame generation
    # (optical flow) requires DX12/Vulkan swapchain integration and is NOT
    # available in FFmpeg/libplacebo for batch video processing. These
    # engines provide the closest available approximation: FSR 2.0 temporal
    # upscaling + temporal blending frame rate interpolation.
    {
        "engine_id": "upscale-fsr31-2x",
        "display_name": "FSR 3.1 x2 (Upscale only, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(2H)x(2W)",
        "license": "MIT",
        "popularity_score": 92,
        "quality_score": 85,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "upscale",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-2x-interp",
        "display_name": "FSR 3.1 x2 (Interpolate only, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3xHxW (double FPS)",
        "license": "MIT",
        "popularity_score": 87,
        "quality_score": 80,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "interpolate",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-2x-both",
        "display_name": "FSR 3.1 x2 (Upscale + Interpolate, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(2H)x(2W) (double FPS)",
        "license": "MIT",
        "popularity_score": 95,
        "quality_score": 88,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "both",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-3x",
        "display_name": "FSR 3.1 x3 (Upscale only, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(3H)x(3W)",
        "license": "MIT",
        "popularity_score": 82,
        "quality_score": 80,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "upscale",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-3x-interp",
        "display_name": "FSR 3.1 x3 (Interpolate only, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3xHxW (double FPS)",
        "license": "MIT",
        "popularity_score": 78,
        "quality_score": 75,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "interpolate",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-3x-both",
        "display_name": "FSR 3.1 x3 (Upscale + Interpolate, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(3H)x(3W) (double FPS)",
        "license": "MIT",
        "popularity_score": 86,
        "quality_score": 82,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "both",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-4x",
        "display_name": "FSR 3.1 x4 (Upscale only, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "MIT",
        "popularity_score": 88,
        "quality_score": 78,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "upscale",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-4x-interp",
        "display_name": "FSR 3.1 x4 (Interpolate only, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3xHxW (double FPS)",
        "license": "MIT",
        "popularity_score": 82,
        "quality_score": 72,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "interpolate",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },
    {
        "engine_id": "upscale-fsr31-4x-both",
        "display_name": "FSR 3.1 x4 (Upscale + Interpolate, GPU)",
        "category": "Upscale",
        "hf_repo_id": "",
        "hf_url": "https://gpuopen.com/fsr/",
        "size_estimate": "0 MB (FFmpeg filter)",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W) (double FPS)",
        "license": "MIT",
        "popularity_score": 92,
        "quality_score": 85,
        "is_implemented": True,
        "sentinel": "",
        "lfs_warning": False,
        "mode": "both",
        "similar_models_url": "https://gpuopen.com/fsr/",
    },

    # --- Frame interpolation --------------------------------------------
    {
        "engine_id": "interpolate-rife-v4",
        "display_name": "RIFE v4.6",
        "category": "Interpolate",
        "hf_repo_id": "TensorForger/RIFE-safetensors",
        "hf_url": "https://huggingface.co/TensorForger/RIFE-safetensors",
        "size_estimate": "~22 MB",
        "sample_shape": "(2x3xHxW) -> 1x3xHxW (insert 1 frame)",
        "license": "MIT",
        "popularity_score": 90,
        "quality_score": 80,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
    },
    {
        "engine_id": "interpolate-film",
        "display_name": "FILM (Google)",
        "category": "Interpolate",
        "hf_repo_id": "Comfy-Org/frame_interpolation",
        "hf_url": "https://huggingface.co/Comfy-Org/frame_interpolation",
        "size_estimate": "~69 MB",
        "sample_shape": "(2x3xHxW) -> N frames between",
        "license": "Apache-2.0",
        "popularity_score": 50,
        "quality_score": 91,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
    },

    # --- Colorization ----------------------------------------------------
    {
        "engine_id": "colorize-ddcolor",
        "display_name": "DDColor",
        "category": "Colorize",
        "hf_repo_id": "piddnad/ddcolor_modelscope",
        "hf_url": "https://huggingface.co/piddnad/ddcolor_modelscope",
        "size_estimate": "~310 MB",
        "sample_shape": "1x3xHxW (gray) -> 1x3xHxW (RGB)",
        "license": "Apache-2.0",
        "popularity_score": 70,
        "quality_score": 87,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
    },
    {
        "engine_id": "colorize-deoldify",
        "display_name": "DeOldify (artistic)",
        "category": "Colorize",
        "hf_repo_id": "spensercai/DeOldify",
        "hf_url": "https://huggingface.co/spensercai/DeOldify",
        "size_estimate": "~255 MB",
        "sample_shape": "1x3xHxW (gray) -> 1x3xHxW (RGB)",
        "license": "MIT",
        "popularity_score": 60,
        "quality_score": 78,
        "is_implemented": True,
        "sentinel": "ColorizeArtistic_gen.pth",
        "lfs_warning": False,
    },

    # --- Denoising / face restoration (extras) --------------------------
    {
        "engine_id": "denoise-nafnet",
        "display_name": "NAFNet (denoise)",
        "category": "Denoise",
        "hf_repo_id": "cstr/nafnet-sidd-GGUF",
        "hf_url": "https://huggingface.co/cstr/nafnet-sidd-GGUF",
        "size_estimate": "~17 MB",
        "sample_shape": "1x3xHxW (noisy) -> 1x3xHxW (clean)",
        "license": "MIT",
        "popularity_score": 40,
        "quality_score": 84,
        "is_implemented": True,
        "sentinel": "README.md",
        "lfs_warning": False,
    },
    {
        "engine_id": "face-gfpgan",
        "display_name": "GFPGAN (face restore)",
        "category": "FaceRestore",
        "hf_repo_id": "litert-community/GFPGAN-v1.4-LiteRT",
        "hf_url": "https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT",
        "size_estimate": "~330 MB",
        "sample_shape": "1x3xHxW (face) -> 1x3xHxW (restored)",
        "license": "Apache-2.0",
        "popularity_score": 80,
        "quality_score": 89,
        "is_implemented": False,
        "sentinel": "README.md",
        "lfs_warning": False,
    },
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


# Built lazily on first access. Cheap, but worth being explicit.
CATALOG_BY_ID: dict[str, dict] = {e["engine_id"]: e for e in CATALOG}


def _all_with_state() -> Iterator[tuple[dict, dict]]:
    """Yield ``(entry, state)`` pairs for every catalog entry.

    State is read through ``engines_state.get_state``; if the read
    raises (corrupt engines.json etc.) we yield an empty default so
    the UI never crashes.
    """
    try:
        from src.ai.engines_state import get_state
    except Exception:
        get_state = lambda _id: {}  # noqa: E731
    for entry in CATALOG:
        try:
            state = get_state(entry["engine_id"]) or {}
        except Exception:
            state = {}
        yield entry, state


def _enrich(entry: dict, state: dict) -> dict:
    """Return a shallow copy of ``entry`` with state + install_dir merged."""
    out = dict(entry)
    out["state"] = dict(state or {})
    out["install_dir"] = get_engine_dir(entry["engine_id"])
    return out


def iter_catalog() -> Iterator[dict]:
    """Yield each catalog entry enriched with state + install_dir."""
    for entry, state in _all_with_state():
        yield _enrich(entry, state)


def list_installed(category: Optional[str] = None) -> list[dict]:
    """Enriched catalog entries whose disk state has ``installed=True``.

    ``category`` is matched case-sensitively against ``entry["category"]``;
    pass ``None`` (default) to span all categories.
    """
    out = []
    for entry, state in _all_with_state():
        if not state.get("installed"):
            continue
        if category is not None and entry.get("category") != category:
            continue
        out.append(_enrich(entry, state))
    return out


def list_enabled(category: Optional[str] = None) -> list[dict]:
    """Enriched catalog entries whose state is BOTH installed AND enabled.

    ``category`` filters by workflow category (e.g. ``"Upscale"``).
    Returns enriched dicts (so the workflow node can render the
    display_name without another lookup).
    """
    out = []
    for entry, state in _all_with_state():
        if not (state.get("installed") and state.get("enabled")):
            continue
        if category is not None and entry.get("category") != category:
            continue
        out.append(_enrich(entry, state))
    return out


def get_engine_meta(engine_id: str) -> Optional[dict]:
    """Cheap lookup by ``engine_id`` (used by the pipeline for labels)."""
    entry = CATALOG_BY_ID.get(engine_id)
    if not entry:
        return None
    try:
        from src.ai.engines_state import get_state
        state = get_state(engine_id) or {}
    except Exception:
        state = {}
    return _enrich(entry, state)


def get_scale_factor(engine_id: str) -> int:
    """Parse the scale factor from sample_shape (e.g. 4 for 4x upscale).

    Returns 1 if the scale factor cannot be determined.
    """
    import re
    entry = CATALOG_BY_ID.get(engine_id)
    if not entry:
        return 1
    shape = entry.get("sample_shape", "")
    # Look for pattern like "(4H)x(4W)" or "(2H)x(2W)"
    m = re.search(r"\((\d+)H\)", shape)
    if m:
        return int(m.group(1))
    return 1


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def split_engine_id(engine_id: str):
    """Return ``(category, family, variant)``.

    The convention is ``<category>-<family>-<variant>``, but some
    entries might miss a variant suffix. We always return exactly
    3 strings, padding missing pieces with empty strings. This keeps
    ``get_engine_dir`` predictable.
    """
    parts = (engine_id or "").split("-", 2)
    while len(parts) < 3:
        parts.append("")
    return tuple(parts)


def get_engine_dir(engine_id: str) -> str:
    """Local directory for the engine's cloned weights.

    Layout: ``<models_dir>/<category>/<family>-<variant>``.
    Slug-validated so a malformed id can never escape ``bin/models``.
    """
    _assert_slug(engine_id)
    cat, family, variant = split_engine_id(engine_id)
    if cat:
        leaf = "-".join(x for x in (family, variant) if x) or engine_id
        return os.path.join(get_models_dir(), cat, leaf)
    return os.path.join(get_models_dir(), engine_id or "")


def categories() -> list[str]:
    """Distinct categories preserving the catalog order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in CATALOG:
        cat = entry.get("category") or "Other"
        if cat not in seen:
            seen.add(cat)
            ordered.append(cat)
    return ordered


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Catalog size: {len(CATALOG)} engines across {len(categories())} categories.")
    for entry in _all_with_state():
        eid = entry[0]["engine_id"]
        cat = entry[0]["category"]
        path = get_engine_dir(eid)
        installed = entry[1].get("installed")
        enabled = entry[1].get("enabled")
        print(f"  - {eid:<35} {cat:<14} inst={installed} enabled={enabled} -> {path}")
    print("\nlist_installed():", [e["engine_id"] for e in list_installed()])
    print("list_enabled(Upscale):", [e["engine_id"] for e in list_enabled("Upscale")])
