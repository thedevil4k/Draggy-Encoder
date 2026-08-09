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
        "display_name": "RealESR AnimeVideo v3 (fast, x2/x3/x4)",
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
        # Como en video2x: el modelo admite 2x, 3x y 4x (la escala elegida
        # se comunica a la engine vía _target_scale y el encoder re-encaja).
        "supported_scales": [2, 3, 4],
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
        "hf_repo_id": "Acly/Omni-SR",
        "hf_url": "https://huggingface.co/Acly/Omni-SR",
        "size_estimate": "~1.7 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "Apache-2.0",
        "popularity_score": 50,
        "quality_score": 74,
        "is_implemented": True,
        "sentinel": "OmniSR_X4_DIV2K.safetensors",
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
        "display_name": "FSR 3.1 x2 (Interpolate, GPU)",
        "category": "Interpolate",
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
        "engine_id": "upscale-fsr31-3x",
        "display_name": "FSR 3.1 x3 (Upscale, GPU)",
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
        "display_name": "FSR 3.1 x3 (Interpolate, GPU)",
        "category": "Interpolate",
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
        "engine_id": "upscale-fsr31-4x",
        "display_name": "FSR 3.1 x4 (Upscale, GPU)",
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
        "display_name": "FSR 3.1 x4 (Interpolate, GPU)",
        "category": "Interpolate",
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

    # --- Denoising ------------------------------------------------------
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

    # --- Face restoration ----------------------------------------------
    # Modelos de restauración facial (GFPGAN / CodeFormer / GPEN). Son
    # pesos ``.pth`` descargables; la conversión a ONNX y el wiring de
    # inferencia llegan por motor en un milestone posterior (igual que
    # Interpolate/Colorize/Denoise).
    {
        "engine_id": "facerestore-gfpgan",
        "display_name": "GFPGAN v1.4",
        "category": "FaceRestore",
        "hf_repo_id": "nlightcho/gfpgan_v14",
        "hf_url": "https://huggingface.co/nlightcho/gfpgan_v14",
        "size_estimate": "~333 MB",
        "sample_shape": "1x3xHxW (face) -> 1x3xHxW (restored)",
        "license": "Apache-2.0",
        "popularity_score": 88,
        "quality_score": 90,
        "is_implemented": True,
        "sentinel": "GFPGANv1.4.pth",
        "lfs_warning": True,
    },
    {
        "engine_id": "facerestore-codeformer",
        "display_name": "CodeFormer",
        "category": "FaceRestore",
        "hf_repo_id": "wwy/codeformer",
        "hf_url": "https://huggingface.co/wwy/codeformer",
        "size_estimate": "~360 MB",
        "sample_shape": "1x3xHxW (face) -> 1x3xHxW (restored)",
        "license": "NTU S-Lab License 1.0",
        "popularity_score": 94,
        "quality_score": 92,
        "is_implemented": True,
        "sentinel": "codeformer.pth",
        "lfs_warning": True,
    },
    {
        "engine_id": "facerestore-gpen-512",
        "display_name": "GPEN BFR-512",
        "category": "FaceRestore",
        "hf_repo_id": "akhaliq/GPEN-BFR-512",
        "hf_url": "https://huggingface.co/akhaliq/GPEN-BFR-512",
        "size_estimate": "~270 MB",
        "sample_shape": "1x3xHxW (face) -> 1x3xHxW (restored)",
        "license": "Custom (non-commercial)",
        "popularity_score": 72,
        "quality_score": 86,
        "is_implemented": True,
        "sentinel": "GPEN-BFR-512.pth",
        "lfs_warning": False,
    },
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

# Built lazily on first access. Cheap, but worth being explicit.
# Solo catálogo: los motores de usuario (Hub) se resuelven con ``get_entry``.
CATALOG_BY_ID: dict[str, dict] = {e["engine_id"]: e for e in CATALOG}

# Arquitectura conocida por motor de catálogo. Solo los motores con pesos
# de red neuronal la declaran; los FSR (algorítmicos) no tienen ninguna.
# La UI la usa para deshabilitar "Convert to ONNX" sin cargar pesos en el
# hilo de la GUI (el conversor solo soporta las arquitecturas que
# ``build_model_from_state_dict`` reconoce: rrdb / srvgg / cugan / omnisr).
CATALOG_ARCH: dict[str, str] = {
    "upscale-realesrgan-x4plus": "rrdb",
    "upscale-realcugan-pro": "cugan",
    "upscale-swinir-real": "swinir",
    "upscale-realesr-animevideov3": "srvgg",
    "upscale-realesr-general-x4v3": "srvgg",
    "upscale-realesrgan-anime-6b": "rrdb",
    "upscale-omnisr-x4": "omnisr",
    "interpolate-rife-v4": "rife",
    "interpolate-film": "film",
    "colorize-ddcolor": "ddcolor",
    "colorize-deoldify": "deoldify",
    "denoise-nafnet": "nafnet",
    "facerestore-gfpgan": "gfpgan",
    "facerestore-codeformer": "codeformer",
    "facerestore-gpen-512": "gpen",
}


def get_catalog_arch(engine_id: str) -> Optional[str]:
    """Arquitectura conocida de un motor de catálogo, o None.

    Los motores de usuario (Hub) guardan su ``arch`` detectada en
    ``user_models.json``; el catálogo estático usa ``CATALOG_ARCH``.
    Devuelve None cuando no se conoce (FSR algorítmicos, o pesos sin
    detectar) — los llamadores deben tratar None como "desconocido",
    no como "no convertible".
    """
    if not engine_id:
        return None
    arch = CATALOG_ARCH.get(engine_id)
    if arch is not None:
        return arch
    if engine_id.startswith("user-"):
        try:
            from src.ai.user_models import get_model
            rec = get_model(engine_id) or {}
            return rec.get("arch") or None
        except Exception:
            return None
    return None


def _user_catalog_entries() -> list[dict]:
    """Entradas del registro de motores de usuario en forma de catálogo."""
    try:
        from src.ai.user_models import catalog_entries
        return catalog_entries()
    except Exception:
        return []


def get_entry(engine_id: str) -> Optional[dict]:
    """Resuelve un engine_id contra el catálogo estático y, si no está,
    contra el registro de motores de usuario (Hub)."""
    entry = CATALOG_BY_ID.get(engine_id)
    if entry is not None:
        return entry
    if engine_id and engine_id.startswith("user-"):
        for ue in _user_catalog_entries():
            if ue["engine_id"] == engine_id:
                return ue
    return None


def _all_with_state() -> Iterator[tuple[dict, dict]]:
    """Yield ``(entry, state)`` pairs for every known engine.

    Incluye el catálogo estático Y los motores de usuario registrados desde
    el Hub (``src.ai.user_models``). State is read through
    ``engines_state.get_state``; if the read raises (corrupt engines.json
    etc.) we yield an empty default so the UI never crashes.
    """
    try:
        from src.ai.engines_state import get_state
    except Exception:
        get_state = lambda _id: {}  # noqa: E731
    entries = list(CATALOG) + _user_catalog_entries()
    for entry in entries:
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
    if "arch" not in out:
        try:
            arch = get_catalog_arch(entry["engine_id"])
            if arch:
                out["arch"] = arch
        except Exception:
            pass
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
    """Cheap lookup by ``engine_id`` (used by the pipeline for labels).

    Resuelve contra el catálogo y, si no está, contra los motores de
    usuario registrados desde el Hub.
    """
    entry = get_entry(engine_id)
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
    entry = get_entry(engine_id)
    if not entry:
        return 1
    shape = entry.get("sample_shape", "")
    # Look for pattern like "(4H)x(4W)" or "(2H)x(2W)"
    m = re.search(r"\((\d+)H\)", shape)
    if m:
        return int(m.group(1))
    return 1


def get_supported_scales(engine_id: str) -> list:
    """Factores de upscale que el catálogo declara para una engine.

    Cuando el entry tiene ``supported_scales`` (p.ej. AnimeVideo v3 con
    ``[2, 3, 4]`` como en video2x) se devuelve esa lista; si no, una lista
    con un único factor derivado de ``sample_shape`` (o vacía si no se
    puede determinar).
    """
    entry = get_entry(engine_id)
    if not entry:
        return []
    supported = entry.get("supported_scales")
    if isinstance(supported, list):
        try:
            scales = sorted({int(s) for s in supported if int(s) > 1})
        except (TypeError, ValueError):
            scales = []
        if scales:
            return scales
    native = get_scale_factor(engine_id)
    return [native] if native > 1 else []


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

    Los motores de usuario (``user-<category>-<owner>-<repo>-<hash>``)
    mantienen el mismo layout: el primer segmento tras ``user-`` es la
    categoría real, así que quedan bajo ``bin/models/<category>/...``
    igual que el catálogo (``models/upscale/foo-...``).
    """
    _assert_slug(engine_id)
    if engine_id.startswith("user-"):
        parts = engine_id.split("-", 2)
        cat = parts[1] if len(parts) > 1 else "user"
        leaf = parts[2] if len(parts) > 2 else engine_id
        return os.path.join(get_models_dir(), cat, leaf)
    cat, family, variant = split_engine_id(engine_id)
    if cat:
        leaf = "-".join(x for x in (family, variant) if x) or engine_id
        return os.path.join(get_models_dir(), cat, leaf)
    return os.path.join(get_models_dir(), engine_id or "")


def categories() -> list[str]:
    """Distinct categories preserving the catalog order.

    Incluye categorías presentes solo en motores de usuario, al final.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in CATALOG:
        cat = entry.get("category") or "Other"
        if cat not in seen:
            seen.add(cat)
            ordered.append(cat)
    for entry in _user_catalog_entries():
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
