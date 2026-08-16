"""Búsqueda de modelos de HuggingFace y verificación de compatibilidad.

Este módulo proporciona:
  - Detección de motores de inferencia instalados (ONNX Runtime, PyTorch)
    con soporte GPU.
  - Búsqueda de modelos del catálogo en HuggingFace para obtener metadata
    (downloads, tamaño, archivos).
  - Búsqueda libre en el Hub de HuggingFace (``search_hub_models``) que
    devuelve modelos arbitrarios compatibles con los motores de inferencia.
  - Verificación de compatibilidad entre un modelo y un motor de inferencia.

La búsqueda del catálogo se limita a los modelos declarados en
``src.ai.registry.CATALOG``. La búsqueda del Hub consulta la API pública
``/api/models`` de HuggingFace y filtra por archivos de pesos, soporte de
imagen y heurísticas de categoría (upscale/interpolación/colorización/...).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import re
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
        from src.runtimes.manager import onnx_status
        status = onnx_status()
        info["installed"] = bool(status.get("installed"))
        info["version"] = status.get("version")
        info["gpu"] = bool(status.get("gpu"))
        if info["installed"] and info["gpu"]:
            info["detail"] = "GPU (CUDA) disponible"
        elif info["installed"]:
            info["detail"] = "CPU only (instala ONNX Runtime GPU desde Manage Dependencies)"
        else:
            info["detail"] = "No instalado en el runtime privado de la aplicación."
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
        from src.runtimes.manager import torch_status
        status = torch_status()
        info["installed"] = bool(status.get("installed"))
        info["version"] = status.get("version")
        info["gpu"] = bool(status.get("gpu"))
        if info["installed"] and info["gpu"]:
            info["detail"] = "GPU disponible en el runtime privado"
        elif info["installed"]:
            info["detail"] = "CPU only (instala PyTorch con soporte CUDA desde Manage Dependencies)"
        else:
            info["detail"] = "No instalado en el runtime privado de la aplicación."
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

        # Engines sin repo HF no tienen metadata
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
# Búsqueda libre en el Hub de HuggingFace
# ---------------------------------------------------------------------------

# Heurística de categoría: palabras clave buscadas en el id del repo, sus
# tags, pipeline_tag y library_name. Cada categoría tiene sinónimos amplios;
# el orden importa porque la primera categoría con match gana.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Upscale": (
        "super-resolution", "super resolution", "superresolution", "upscale",
        "upscaling", "esrgan", "realesrgan", "realesr", "swinir", "realcugan",
        "cugan", "srgan", "omnisr", "waifu", "anime4k", "srcnn", "edsr",
        "rcan", "span", "aesr", "carcass", "4x-ultrasharp", "4x-animesharp",
    ),
    "Interpolate": (
        "rife", "frame interpolation", "frame-interpolation", "interpolation",
        "frame interpolation", "vfi", "video interpolation", "flowframe",
        "gmfss", "film",
    ),
    "Colorize": (
        "colorize", "colorization", "colorizer", "deoldify", "ddcolor",
    ),
    "Denoise": (
        "denoise", "denoising", "nafnet", "dncnn", "scunet", "real denoising",
        "denoiser",
    ),
    "FaceRestore": (
        "gfpgan", "codeformer", "gpen", "face restore", "face-restore",
        "face restoration", "face enhancement", "faceenhance", "faceenhencement",
        "biniou-facerestore",
    ),
}

# Requisito mínimo de señal de imagen para incluir un repo "incierto" (sin
# match de categoría). Evita que modelos de texto/LLM (también usan .bin y
# .safetensors) se cuelen en el buscador.
_IMAGE_SIGNAL_TAGS = ("image", "super-resolution", "upscale", "sr")


def _repo_text(repo_id: str, tags: list[str], pipeline_tag: Optional[str],
               library_name: Optional[str]) -> str:
    """Texto plano de un modelo del Hub para buscar keywords de categoría."""
    parts = [repo_id or "", pipeline_tag or "", library_name or ""]
    parts.extend(tags or [])
    return " ".join(parts).lower()


def _detect_category(repo_id: str, tags: list[str],
                     pipeline_tag: Optional[str],
                     library_name: Optional[str] = None) -> Optional[str]:
    """Devuelve la categoría más probable para un repo, o None si no matchea
    ninguna heurística."""
    text = _repo_text(repo_id, tags, pipeline_tag, library_name)
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return None


def user_engine_id(category: str, repo_id: str) -> str:
    """Genera un ``engine_id`` slug-safe y determinista para un modelo del Hub.

    Formato: ``user-<categoría>-<owner>-<name>-<hash8>``. Cumple
    ``_SLUG_RE`` (``[a-z0-9]+(-[a-z0-9]+)*``) para que ``get_engine_dir``,
    ``engines_state.patch`` y el sensor de disco funcionen igual que con el
    catálogo. El hash corto (SHA-1) evita colisiones entre repos con el mismo
    nombre y estabiliza el id entre búsquedas.
    """
    cat = re.sub(r"[^a-z0-9]+", "-", (category or "").lower()).strip("-") or "other"
    owner, _, name = (repo_id or "").lower().strip().partition("/")
    slug_parts = []
    for part in (owner, name):
        cleaned = re.sub(r"[^a-z0-9]+", "-", part).strip("-")[:40]
        if cleaned:
            slug_parts.append(cleaned)
    slug = re.sub(r"-{2,}", "-", "-".join(slug_parts)).strip("-")
    digest = hashlib.sha1((repo_id or "").encode("utf-8")).hexdigest()[:8]
    return f"user-{cat}-{slug}-{digest}"


def _hub_file_list(model) -> list[str]:
    """Lista de nombres de archivo del modelo (vía ``siblings``)."""
    try:
        return [s.rfilename for s in (model.siblings or [])]
    except Exception:
        return []


def _has_weight_files(files: list[str]) -> bool:
    lower = [f.lower() for f in files]
    return any(
        f.endswith(ext) for f in lower
        for ext in _ONNX_WEIGHT_EXTS + _TORCH_WEIGHT_EXTS
    )


def _hub_cache_key(category, query, engine_id, page, limit):
    return ("hub", category, query or "", engine_id, page, limit)


_HUB_CACHE: dict = {}


def search_hub_models(
    category: str = "Upscale",
    query: str = "",
    engine_id: Optional[str] = None,
    page: int = 0,
    limit: int = 30,
    use_cache: bool = True,
) -> list[dict]:
    """Busca modelos arbitrarios en el Hub de HuggingFace.

    Consulta ``/api/models`` (``HfApi.list_models``) con ``search=``,
    ``sort=downloads`` y ``expand=["siblings", "cardData"]``, y filtra:

      - Que el repo tenga archivos de pesos compatibles
        (``.safetensors/.pth/.pt/.onnx``).
      - Compatibilidad con ``engine_id`` (si se indica), vía
        ``check_model_compatibility``.
      - Categoría: la heurística debe matchear la categoría pedida O el repo
        debe tener señal clara de imagen (pipeline/tags) para mostrarse con
        una insignia "categoría no confirmada".

    Args:
        category: Categoría objetivo ("Upscale", "Interpolate", ...) a
            filtrar; ``None`` = cualquier categoría detectada.
        query: Texto de búsqueda (libre, como en la web de HF).
        engine_id: Motor para filtrar compatibilidad (None = todos).
        page: Página (0-based) para paginar; cada página hace una llamada API.
        limit: Modelos por página (máx. HF ~100).

    Returns:
        Lista de dicts con la misma forma que ``search_catalog_models`` más
        los campos ``is_user_model`` (True) y ``category_confidence``.
    """
    key = _hub_cache_key(category, query, engine_id, page, limit)
    if use_cache and key in _HUB_CACHE:
        return list(_HUB_CACHE[key])

    from huggingface_hub import HfApi

    try:
        from src.personal import get_token as _get_token
        _token = _get_token() or None
    except Exception:
        _token = None

    api = HfApi(token=_token)
    try:
        # ``list_models`` returns a lazy generator that pages the server
        # automatically. ``huggingface_hub`` 1.x removed the ``offset``
        # kwarg, so we paginate client-side with ``islice``. We must NOT
        # pass ``limit`` to the SDK either: it caps the TOTAL number of
        # yielded models, which would break every page after the first.
        models = api.list_models(
            search=query or None,
            sort="downloads",
            expand=["siblings", "cardData"],
        )
        start = page * limit
        models = itertools.islice(models, start, start + limit)
    except Exception as exc:
        logger.warning("HF list_models failed: %s", exc)
        raise

    results = []
    for model in (models or []):
        try:
            repo_id = getattr(model, "modelId", None) or getattr(model, "id", None)
            if not repo_id:
                continue
            files = _hub_file_list(model)
            if not _has_weight_files(files):
                continue
            tags = list(getattr(model, "tags", None) or [])
            pipeline_tag = getattr(model, "pipeline_tag", None)
            library_name = getattr(model, "library_name", None)

            detected = _detect_category(repo_id, tags, pipeline_tag, library_name)
            if detected is not None and category is not None and detected != category:
                continue
            if detected is None:
                # Repo sin categoría clara: solo entra si tiene señal de
                # imagen (evita LLM y repos de texto con .bin/.safetensors).
                if not _has_image_signal(pipeline_tag, tags):
                    continue
                category_match = False
                effective_category = category or "Other"
            else:
                category_match = True
                effective_category = detected

            hf_info = {
                "repo_id": repo_id,
                "downloads": getattr(model, "downloads", 0) or 0,
                "pipeline_tag": pipeline_tag,
                "tags": tags,
                "last_modified": getattr(model, "lastModified", None),
                "siblings": files,
            }

            compatibility = None
            if engine_id:
                compatibility = check_model_compatibility(repo_id, engine_id, hf_info)
                if not compatibility.get("compatible", False):
                    continue

            generated_id = user_engine_id(effective_category, repo_id)
            state = _get_engine_state(generated_id)
            try:
                from src.ai.user_models import get_model as _get_user_model
                _arch = (_get_user_model(generated_id) or {}).get("arch") or ""
            except Exception:
                _arch = ""

            card = {
                "engine_id": generated_id,
                "display_name": _humanize_repo_name(repo_id),
                "category": effective_category,
                "hf_repo_id": repo_id,
                "hf_url": f"https://huggingface.co/{repo_id}",
                "size_estimate": "",
                "size_mb": None,
                "downloads": hf_info["downloads"],
                "likes": getattr(model, "likes", 0) or 0,
                "license": _model_license(model),
                "quality_score": 0,
                "popularity_score": _downloads_to_score(hf_info["downloads"]),
                "is_implemented": True,
                "lfs_warning": False,
                "last_modified": hf_info["last_modified"],
                "files": files,
                "installed": bool(state.get("installed", False)),
                "enabled": bool(state.get("enabled", False)),
                "onnx_converted": bool(state.get("onnx_converted", False)),
                "install_path": state.get("install_path"),
                "compatibility": compatibility,
                "is_user": True,
                "arch": _arch,
                "category_confidence": category_match,
                "detected_category": detected,
            }
            results.append(card)
        except Exception as exc:
            logger.warning("Skipping hub model %s: %s", getattr(model, "modelId", "?"), exc)

    if use_cache:
        _HUB_CACHE[key] = results
    return results


def _has_image_signal(pipeline_tag: Optional[str], tags: list[str]) -> bool:
    if pipeline_tag and pipeline_tag in _IMAGE_INPUT_TAGS:
        return True
    lowered = " ".join(tags or []).lower()
    return any(sig in lowered for sig in ("image", "super-resolution", "upscale"))


def _humanize_repo_name(repo_id: str) -> str:
    """Convierte ``owner/4x-UltraSharp`` en ``4x UltraSharp``."""
    _, _, name = (repo_id or "").partition("/")
    cleaned = re.sub(r"[-_]+", " ", name or repo_id or "").strip()
    return cleaned[:60] or (repo_id or "")


def _model_license(model) -> str:
    """Lee la licencia del cardData del modelo si está disponible."""
    try:
        card = getattr(model, "cardData", None)
        lic = getattr(card, "license", None) or getattr(card, "classify_license", None)
        return lic or ""
    except Exception:
        return ""


def _downloads_to_score(downloads: int) -> int:
    """Proxy 0-100 de popularidad desde downloads del repo."""
    try:
        d = max(0, int(downloads))
    except Exception:
        return 0
    if d >= 1_000_000:
        return 100
    if d >= 100_000:
        return 80
    if d >= 10_000:
        return 60
    if d >= 1_000:
        return 40
    if d >= 100:
        return 20
    return 0


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
    _HUB_CACHE.clear()


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
