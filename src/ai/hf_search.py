"""Búsqueda de modelos de HuggingFace y verificación de compatibilidad.

Este módulo proporciona:
  - Detección de motores de inferencia instalados (ONNX Runtime, PyTorch)
    con soporte GPU.
  - Búsqueda de modelos del catálogo en HuggingFace para obtener metadata
    (downloads, tamaño, archivos).
  - Verificación de compatibilidad entre un modelo y un motor de inferencia.

La búsqueda se limita a los modelos declarados en ``src.ai.registry.CATALOG``
para la categoría solicitada. No se devuelven modelos externos al catálogo.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Tipos de archivo que indican compatibilidad con cada motor
_ONNX_WEIGHT_EXTS = (".onnx",)
_TORCH_WEIGHT_EXTS = (".safetensors", ".bin", ".pth", ".pt")
_CONFIG_FILES = ("config.json", "config.yaml", "model_index.json")

# Pipeline tags que indican soporte de entrada de imagen
_IMAGE_INPUT_TAGS = {
    "image-to-image",
    "image-classification",
    "object-detection",
    "pixel-level-moderation",
    "zero-shot-image-classification",
}

# Cache for _query_model_info calls to avoid double API requests when
# search_catalog_models() and check_model_compatibility() both query
# the same repo_id.
_MODEL_INFO_CACHE: dict[str, Optional[dict]] = {}


# ---------------------------------------------------------------------------
# Detección de motores de inferencia
# ---------------------------------------------------------------------------

def _check_onnxruntime() -> dict:
    """Verifica si ONNX Runtime está instalado y si soporta GPU."""
    info = {
        "id": "onnxruntime",
        "name": "ONNX Runtime",
        "installed": False,
        "gpu": False,
        "version": None,
        "detail": "",
    }
    try:
        import onnxruntime as ort
        info["installed"] = True
        info["version"] = ort.__version__
        providers = ort.get_available_providers()
        info["gpu"] = "CUDAExecutionProvider" in providers
        if info["gpu"]:
            info["detail"] = "GPU (CUDA) disponible"
        else:
            info["detail"] = "CPU only (instala onnxruntime-gpu para aceleración GPU)"
    except ImportError:
        info["detail"] = "No instalado. Instálalo desde 'Manage Dependencies'."
    except Exception as exc:
        info["detail"] = f"Error de detección: {exc}"
    return info


def _check_torch() -> dict:
    """Verifica si PyTorch está instalado y si soporta CUDA."""
    info = {
        "id": "torch",
        "name": "PyTorch",
        "installed": False,
        "gpu": False,
        "version": None,
        "detail": "",
    }
    try:
        import torch
        info["installed"] = True
        info["version"] = torch.__version__
        info["gpu"] = torch.cuda.is_available()
        if info["gpu"]:
            gpu_name = torch.cuda.get_device_name(0)
            info["detail"] = f"GPU disponible: {gpu_name}"
        else:
            info["detail"] = "CPU only (instala PyTorch con soporte CUDA)"
    except ImportError:
        info["detail"] = "No instalado. Instálalo desde 'Manage Dependencies'."
    except Exception as exc:
        info["detail"] = f"Error de detección: {exc}"
    return info


def get_inference_engines() -> list[dict]:
    """Devuelve la lista de motores de inferencia detectados.

    Cada entrada incluye: id, name, installed, gpu, version, detail.
    El orden prioriza motores con GPU.
    """
    engines = [_check_onnxruntime(), _check_torch()]
    # Ordenar: GPU primero, luego instalados, luego por nombre
    engines.sort(key=lambda e: (
        not e["gpu"],
        not e["installed"],
        e["name"],
    ))
    return engines


def get_engine_by_id(engine_id: str) -> Optional[dict]:
    """Devuelve el motor de inferencia con el ID dado, o None."""
    for engine in get_inference_engines():
        if engine["id"] == engine_id:
            return engine
    return None


# ---------------------------------------------------------------------------
# Búsqueda de modelos del catálogo en HuggingFace
# ---------------------------------------------------------------------------

def _get_catalog_entries(category: str) -> list[dict]:
    """Devuelve las entradas del catálogo filtradas por categoría."""
    try:
        from src.ai.registry import CATALOG
    except Exception:
        return []
    return [e for e in CATALOG if e.get("category") == category]


def _get_engine_state(engine_id: str) -> dict:
    """Devuelve el estado persistente de un engine (installed/enabled/etc)."""
    try:
        from src.ai.engines_state import get_state
        return get_state(engine_id) or {}
    except Exception:
        return {}


def _query_model_info(repo_id: str) -> Optional[dict]:
    """Consulta HuggingFace para obtener metadata de un modelo.

    Maneja el error 'this model does not support image input' que ocurre
    cuando huggingface_hub intenta parsear el YAML del README y encuentra
    un campo ``image: image.png`` en un modelo cuyo ``pipeline_tag`` no
    indica soporte de imagen.  Si ``model_info`` falla, intentamos una
    consulta mínima via ``list_repo_files`` que no parsea el README.
    """
    if repo_id in _MODEL_INFO_CACHE:
        return _MODEL_INFO_CACHE[repo_id]

    from huggingface_hub import HfApi
    try:
        from src.personal import get_token as _get_token
        _token = _get_token() or None
    except Exception:
        _token = None
    api = HfApi(token=_token)

    try:
        info = api.model_info(repo_id)
        result = {
            "repo_id": repo_id,
            "downloads": getattr(info, "downloads", 0) or 0,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "tags": getattr(info, "tags", []) or [],
            "last_modified": getattr(info, "lastModified", None),
            "siblings": [s.rfilename for s in (getattr(info, "siblings", []) or [])],
        }
        _MODEL_INFO_CACHE[repo_id] = result
        return result
    except Exception as exc:
        err_msg = str(exc).lower()
        if "does not support image input" in err_msg or "image.png" in err_msg:
            logger.info("HF model_info image-error for %s, falling back to list_repo_files", repo_id)
            try:
                files = api.list_repo_files(repo_id)
                result = {
                    "repo_id": repo_id,
                    "downloads": 0,
                    "pipeline_tag": None,
                    "tags": [],
                    "last_modified": None,
                    "siblings": list(files) if files else [],
                }
                _MODEL_INFO_CACHE[repo_id] = result
                return result
            except Exception:
                pass
        logger.warning("HF model_info failed for %s: %s", repo_id, exc)
        _MODEL_INFO_CACHE[repo_id] = None
        return None


def _check_image_input_support(hf_info: dict) -> bool:
    """Verifica si un modelo de HuggingFace soporta entrada de imagen.

    Filtra el error 'this model does not support image input' comprobando:
      - pipeline_tag está en _IMAGE_INPUT_TAGS
      - O algún tag contiene 'image' o 'super-resolution'
    """
    if not hf_info:
        return False
    pipeline_tag = hf_info.get("pipeline_tag")
    tags = hf_info.get("tags", [])
    if pipeline_tag and pipeline_tag in _IMAGE_INPUT_TAGS:
        return True
    for tag in tags:
        tag_lower = tag.lower()
        if "image" in tag_lower or "super-resolution" in tag_lower or "upscale" in tag_lower:
            return True
    return False


def _estimate_model_size_mb(hf_info: dict, catalog_entry: dict) -> Optional[float]:
    """Estima el tamaño del modelo en MB.

    Prioriza el size_estimate del catálogo; si no está disponible,
    intenta calcularlo a partir de los archivos del repo.
    """
    size_str = catalog_entry.get("size_estimate", "")
    if size_str:
        import re
        match = re.search(r"([\d.]+)\s*MB", size_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
    # Fallback: sumar tamaños de archivos de peso
    siblings = hf_info.get("siblings", []) if hf_info else []
    total = 0
    for fname in siblings:
        lower = fname.lower()
        if any(lower.endswith(ext) for ext in _ONNX_WEIGHT_EXTS + _TORCH_WEIGHT_EXTS):
            # No tenemos el tamaño exacto de cada archivo desde model_info,
            # así que devolvemos None y usamos el size_estimate del catálogo.
            pass
    return None


def _get_model_files(hf_info: dict) -> list[str]:
    """Devuelve la lista de archivos del modelo en HuggingFace."""
    if not hf_info:
        return []
    return hf_info.get("siblings", [])


def search_catalog_models(
    category: str = "Upscale",
    query: str = "",
    min_downloads: int = 0,
    max_size_mb: Optional[float] = None,
    engine_id: Optional[str] = None,
) -> list[dict]:
    """Busca modelos del catálogo en HuggingFace y filtra por compatibilidad.

    Args:
        category: Categoría del catálogo (ej: "Upscale").
        query: Texto de búsqueda (filtra por nombre/display_name).
        min_downloads: Filtra modelos con menos de este número de downloads.
        max_size_mb: Filtra modelos mayores a este tamaño en MB.
        engine_id: Si se especifica, solo devuelve modelos compatibles
                   con este motor de inferencia.

    Returns:
        Lista de dicts con metadata enriquecida de cada modelo.
    """
    catalog_entries = _get_catalog_entries(category)
    if not catalog_entries:
        return []

    results = []
    for entry in catalog_entries:
        engine_id_catalog = entry.get("engine_id", "")
        display_name = entry.get("display_name", engine_id_catalog)
        repo_id = entry.get("hf_repo_id", "")

        # Filtrar por query (nombre del modelo)
        if query:
            query_lower = query.lower()
            if query_lower not in display_name.lower() and query_lower not in repo_id.lower():
                continue

        # Engines sin repo HF (FSR, FSR 2.0, etc.) no tienen metadata
        # en HuggingFace. Saltamos la query silenciosamente en lugar de
        # invocar ``_query_model_info("")`` que falla con "Repo id must
        # use alphanumeric chars" en cada refresh del diálogo.
        if not repo_id:
            hf_info = None
        else:
            hf_info = _query_model_info(repo_id)

        # Filtrar modelos que no soportan entrada de imagen
        # (skip for catalog models — they are curated and known to work)
        if hf_info and not _check_image_input_support(hf_info):
            # Only skip if this looks like a non-image model based on
            # the pipeline_tag. Catalog entries are curated, so allow them.
            pipeline = hf_info.get("pipeline_tag")
            if pipeline and pipeline not in (None, "image-to-image", "image-to-text"):
                logger.info("Skipping %s: pipeline_tag=%s", repo_id, pipeline)
                continue

        # Filtrar por downloads
        downloads = hf_info.get("downloads", 0) if hf_info else 0
        if downloads < min_downloads:
            continue

        # Estimar tamaño
        size_mb = _estimate_model_size_mb(hf_info, entry)

        # Filtrar por tamaño
        if max_size_mb is not None and size_mb is not None and size_mb > max_size_mb:
            continue

        # Verificar compatibilidad con motor de inferencia
        compatibility = None
        if engine_id:
            compatibility = check_model_compatibility(repo_id, engine_id, hf_info)
            if not compatibility.get("compatible", False):
                continue

        # Estado del engine
        state = _get_engine_state(engine_id_catalog)

        result = {
            "engine_id": engine_id_catalog,
            "display_name": display_name,
            "category": entry.get("category", ""),
            "hf_repo_id": repo_id,
            "hf_url": entry.get("hf_url", ""),
            "size_estimate": entry.get("size_estimate", ""),
            "size_mb": size_mb,
            "downloads": downloads,
            "license": entry.get("license", ""),
            "quality_score": entry.get("quality_score", 0),
            "popularity_score": entry.get("popularity_score", 0),
            "is_implemented": entry.get("is_implemented", False),
            "lfs_warning": entry.get("lfs_warning", False),
            "last_modified": hf_info.get("last_modified") if hf_info else None,
            "files": _get_model_files(hf_info),
            "installed": bool(state.get("installed", False)),
            "enabled": bool(state.get("enabled", False)),
            "onnx_converted": bool(state.get("onnx_converted", False)),
            "install_path": state.get("install_path"),
            "compatibility": compatibility,
        }
        results.append(result)

    # Ordenar por downloads (descendente) por defecto
    results.sort(key=lambda r: r.get("downloads", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Verificación de compatibilidad
# ---------------------------------------------------------------------------

def check_model_compatibility(
    repo_id: str,
    engine_id: str,
    hf_info: Optional[dict] = None,
) -> dict:
    """Verifica si un modelo es compatible con un motor de inferencia.

    Args:
        repo_id: ID del repo en HuggingFace (ej: "ai-forever/Real-ESRGAN").
        engine_id: ID del motor de inferencia ("onnxruntime", "torch").
        hf_info: Metadata ya consultada de HuggingFace (opcional, para evitar
                 múltiples llamadas a la API).

    Returns:
        dict con: compatible (bool), reason (str), needs_conversion (bool).
    """
    if hf_info is None:
        # Engines sin repo HF (FSR, etc.) — no podemos verificar compat
        # vía HuggingFace, así que devolvemos "compatible" como fallback
        # para no romper el render. La UI mostrará el engine como
        # "no verificado" si quisiéramos ser estrictos.
        if not repo_id:
            return {
                "compatible": True,
                "reason": "Engine sin repo HF (FSR/algorítmico). Compatibilidad asumida.",
                "needs_conversion": False,
            }
        hf_info = _query_model_info(repo_id) or {}

    files = _get_model_files(hf_info)
    files_lower = [f.lower() for f in files]

    if engine_id == "onnxruntime":
        # Compatible si tiene archivos .onnx (ya convertido)
        has_onnx = any(f.endswith(".onnx") for f in files_lower)
        if has_onnx:
            return {
                "compatible": True,
                "reason": "Modelo ya está en formato ONNX",
                "needs_conversion": False,
            }
        # Compatible si tiene archivos de peso que se pueden convertir
        has_weights = any(
            f.endswith(ext) for f in files_lower
            for ext in _TORCH_WEIGHT_EXTS
        )
        has_config = any(
            f.endswith(ext) for f in files_lower
            for ext in _CONFIG_FILES
        )
        if has_weights and has_config:
            return {
                "compatible": True,
                "reason": "Modelo convertible a ONNX (requiere conversión)",
                "needs_conversion": True,
            }
        return {
            "compatible": False,
            "reason": "No se encontraron archivos de peso compatibles",
            "needs_conversion": False,
        }

    elif engine_id == "torch":
        # Compatible si tiene archivos de peso de PyTorch
        has_weights = any(
            f.endswith(ext) for f in files_lower
            for ext in _TORCH_WEIGHT_EXTS
        )
        if has_weights:
            return {
                "compatible": True,
                "reason": "Modelo compatible con PyTorch",
                "needs_conversion": False,
            }
        return {
            "compatible": False,
            "reason": "No se encontraron archivos de peso de PyTorch",
            "needs_conversion": False,
        }

    return {
        "compatible": False,
        "reason": f"Motor de inferencia desconocido: {engine_id}",
        "needs_conversion": False,
    }


# ---------------------------------------------------------------------------
# Cache de resultados (para evitar múltiples llamadas a la API)
# ---------------------------------------------------------------------------

_cache: dict = {}
_cache_keys: dict = {}


def clear_cache() -> None:
    """Limpia el cache de resultados de búsqueda y model_info."""
    _cache.clear()
    _cache_keys.clear()
    _MODEL_INFO_CACHE.clear()


def get_cached_search(
    category: str,
    query: str,
    min_downloads: int,
    max_size_mb: Optional[float],
    engine_id: Optional[str],
) -> Optional[list[dict]]:
    """Intenta obtener resultados del cache."""
    key = (category, query, min_downloads, max_size_mb, engine_id)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    return None


def cache_search(
    category: str,
    query: str,
    min_downloads: int,
    max_size_mb: Optional[float],
    engine_id: Optional[str],
    results: list[dict],
) -> None:
    """Almacena resultados en el cache."""
    key = (category, query, min_downloads, max_size_mb, engine_id)
    _cache[key] = results
    _cache_keys[key] = True


# ---------------------------------------------------------------------------
# API pública de conveniencia
# ---------------------------------------------------------------------------

def search_upscaling_models(
    query: str = "",
    min_downloads: int = 0,
    max_size_mb: Optional[float] = None,
    engine_id: Optional[str] = None,
    use_cache: bool = True,
) -> list[dict]:
    """Búsqueda de modelos de upscaling del catálogo en HuggingFace.

    Args:
        query: Texto de búsqueda.
        min_downloads: Mínimo número de downloads.
        max_size_mb: Tamaño máximo en MB.
        engine_id: Motor de inferencia para filtrar compatibilidad.
        use_cache: Si usar el cache de resultados.

    Returns:
        Lista de modelos compatibles, ordenados por downloads.
    """
    if use_cache:
        cached = get_cached_search("Upscale", query, min_downloads, max_size_mb, engine_id)
        if cached is not None:
            return cached

    results = search_catalog_models(
        category="Upscale",
        query=query,
        min_downloads=min_downloads,
        max_size_mb=max_size_mb,
        engine_id=engine_id,
    )

    if use_cache:
        cache_search("Upscale", query, min_downloads, max_size_mb, engine_id, results)

    return results


def get_model_downloads(repo_id: str) -> int:
    """Obtiene el número de downloads de un modelo en HuggingFace."""
    info = _query_model_info(repo_id)
    if info:
        return info.get("downloads", 0)
    return 0


def is_model_installed(engine_id: str) -> bool:
    """Verifica si un modelo del catálogo está instalado localmente."""
    try:
        from src.ai.engines_state import get_state
        state = get_state(engine_id) or {}
        return bool(state.get("installed", False))
    except Exception:
        return False


def is_model_enabled(engine_id: str) -> bool:
    """Verifica si un modelo del catálogo está habilitado."""
    try:
        from src.ai.engines_state import get_state
        state = get_state(engine_id) or {}
        return bool(state.get("enabled", False))
    except Exception:
        return False
