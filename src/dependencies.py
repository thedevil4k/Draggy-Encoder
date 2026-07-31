"""Registro unificado de dependencias externas del programa.

Por ahora solo FFmpeg es completamente operativo. El resto de entradas son
marcadores (runtimes de IA, ranuras de modelos) que ya tienen una carpeta
dentro de ``bin/`` para que el panel de gestión pueda mostrárselas al usuario
y se conviertan en puntos de integración reales cuando lleguen los modelos.

La capa ``src.dependencies`` está pensada para que añadir una dependencia nueva
sea puramente declarativo: una entrada en ``REGISTRY`` + una función ``status``
que devuelva un dict con los campos expuestos aquí. La UI consume este módulo
a través de ``list_all()`` y ``reinstall()`` exclusivamente.
"""

from __future__ import annotations

import os
import platform
from typing import Callable, Optional

from src.config import (
    ensure_runtime_dirs,
    get_models_dir,
    get_runtimes_dir,
)
from src.ffmpeg import manager as ffmpeg_manager
from src.runtimes import manager as ai_runtimes_manager


# ---------------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------------

KIND_BINARY = "binary"
KIND_RUNTIME = "runtime"
KIND_MODEL = "model"

STATUS_INSTALLED = "installed"
STATUS_MISSING = "missing"
STATUS_COMING_SOON = "coming_soon"
STATUS_ERROR = "error"


# ---------------------------------------------------------------------------
# Factories para runs/modelos placeholder
# ---------------------------------------------------------------------------

def _slot_status(display_name, install_dir, kind):
    """Estado comun a las ranuras todavia no implementadas."""
    ensure_runtime_dirs()
    if not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir, exist_ok=True)
        except Exception:
            pass
    return {
        "id": None,
        "display_name": display_name,
        "kind": kind,
        "status": STATUS_COMING_SOON,
        "version": "—",
        "install_dir": install_dir,
        "source_url": None,
        "is_updatable": False,
        "size_bytes": _safe_dir_size(install_dir),
        "detail": "Coming soon — la integracion con modelos IA se hara en el siguiente hito.",
    }


def _onnxruntime_status():
    """Detect ONNX Runtime via importlib."""
    ensure_runtime_dirs()
    # Asegurar que el runtime local esté en sys.path antes de importar
    try:
        from src.runtimes.manager import inject_runtime_paths
        inject_runtime_paths()
    except Exception:
        pass
    install_dir = os.path.join(get_runtimes_dir(), "onnxruntime")
    try:
        import importlib.metadata as _meta
        ver = _meta.version("onnxruntime")
        gpu = False
        try:
            import onnxruntime as _ort
            provs = _ort.get_available_providers()
            gpu = "CUDAExecutionProvider" in provs
        except Exception:
            pass
        provider_label = "GPU (CUDA)" if gpu else "CPU only"
        return {
            "id": "onnxruntime",
            "display_name": "ONNX Runtime (CPU/GPU)",
            "kind": KIND_RUNTIME,
            "status": STATUS_INSTALLED,
            "version": ver,
            "install_dir": install_dir,
            "source_url": "https://pypi.org/project/onnxruntime/",
            "is_updatable": False,
            "size_bytes": _safe_dir_size(install_dir),
            "detail": f"Installed {provider_label}. Version: {ver}",
            "provider": "CUDAExecutionProvider" if gpu else "CPUExecutionProvider",
        }
    except Exception:
        return {
            "id": "onnxruntime",
            "display_name": "ONNX Runtime (CPU/GPU)",
            "kind": KIND_RUNTIME,
            "status": STATUS_MISSING,
            "version": "—",
            "install_dir": install_dir,
            "source_url": "https://pypi.org/project/onnxruntime/",
            "is_updatable": False,
            "size_bytes": _safe_dir_size(install_dir),
            "detail": "Not installed. Install: pip install onnxruntime-gpu",
        }


def _pytorch_status():
    """Detect PyTorch via importlib."""
    ensure_runtime_dirs()
    # Asegurar que el runtime local esté en sys.path antes de importar
    try:
        from src.runtimes.manager import inject_runtime_paths
        inject_runtime_paths()
    except Exception:
        pass
    install_dir = os.path.join(get_runtimes_dir(), "torch")
    try:
        import importlib.metadata as _meta
        ver = _meta.version("torch")
        gpu = False
        try:
            import torch as _torch
            gpu = _torch.cuda.is_available()
        except Exception:
            pass
        device = "CUDA" if gpu else "CPU"
        return {
            "id": "torch",
            "display_name": "PyTorch runtime",
            "kind": KIND_RUNTIME,
            "status": STATUS_INSTALLED,
            "version": ver,
            "install_dir": install_dir,
            "source_url": "https://pytorch.org/get-started/locally/",
            "is_updatable": False,
            "size_bytes": _safe_dir_size(install_dir),
            "detail": f"Installed ({device}). Version: {ver}",
            "gpu_available": gpu,
        }
    except Exception:
        return {
            "id": "torch",
            "display_name": "PyTorch runtime",
            "kind": KIND_RUNTIME,
            "status": STATUS_MISSING,
            "version": "—",
            "install_dir": install_dir,
            "source_url": "https://pytorch.org/get-started/locally/",
            "is_updatable": False,
            "size_bytes": _safe_dir_size(install_dir),
            "detail": "Not installed. Install: pip install torch torchvision",
        }


def _local_ffmpeg_size_or_none():
    """Tamano del binario FFmpeg local, o None si falta / falla el stat.

    Se usa en el closure ``bytes_fn`` del REGISTRY para alimentar el
    mini-stats banner y el historial persistente. Devolver None
    explicito (vs. explotar) es importante: el controller puede entonces
    omitir el campo ``old_bytes``/``new_bytes`` en la entrada del
    historial sin contaminar el resto de la captura.
    """
    try:
        from src.ffmpeg.manager import get_local_ffmpeg_path
        path = get_local_ffmpeg_path()
        if path and os.path.exists(path):
            return os.path.getsize(path)
    except Exception:
        pass
    return None


def _safe_dir_size(path):
    try:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Estado dinámico de FFmpeg
# ---------------------------------------------------------------------------

def _ffmpeg_status():
    info = ffmpeg_manager.get_install_info()
    status = info["status"]
    if status == STATUS_INSTALLED:
        detail = (
            "Alojado en la propia carpeta del programa. "
            "Usa 'Update' si hay una build más reciente en el upstream."
        )
    elif status == STATUS_MISSING:
        detail = (
            "FFmpeg no está instalado todavía. Pulsa 'Install' para "
            "descargarlo en la propia carpeta del programa."
        )
    else:
        detail = "FFmpeg está instalado pero no responde a -version."
    return {
        "id": info["id"],
        "display_name": info["display_name"],
        "kind": info["kind"],
        "status": status,
        "version": info.get("version") or "unknown",
        "install_dir": info["install_dir"],
        "source_url": info.get("source_url"),
        "is_updatable": True,
        "size_bytes": _safe_dir_size(info["install_dir"]),
        "detail": detail,
        # Datos de upstream (para Latest/Update available en la UI)
        "latest_fingerprint": info.get("latest_fingerprint") or {},
        "update_available": info.get("update_available"),
        "last_check_iso": info.get("last_check_iso"),
        "upstream_error": info.get("upstream_error"),
    }


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

REGISTRY = [
    {
        "id": "ffmpeg",
        "kind": KIND_BINARY,
        "status_fn": _ffmpeg_status,
        "reinstall_fn": ffmpeg_manager.force_redownload,
        "open_label": "Open bin/ffmpeg folder",
        "is_updatable": True,
        # Size/introspection hook: el chain controller lo usa para capturar
        # el tamano del binario ANTES y DESPUES de la reinstalacion, y asi
        # alimentar el mini-stats banner + el historial persistente.
        "bytes_fn": lambda: _local_ffmpeg_size_or_none(),
    },
    {
        "id": "torch",
        "kind": KIND_RUNTIME,
        "display_name": "PyTorch runtime",
        "status_fn": _pytorch_status,
        "reinstall_fn": ai_runtimes_manager.install_all,
        "open_label": "Open bin/runtimes/ai_modules",
        "is_updatable": True,
    },
    {
        "id": "onnxruntime",
        "kind": KIND_RUNTIME,
        "display_name": "ONNX Runtime (CPU/GPU)",
        "status_fn": _onnxruntime_status,
        "reinstall_fn": ai_runtimes_manager.install_all,
        "open_label": "Open bin/runtimes/ai_modules",
        "is_updatable": True,
    },
    {
        "id": "upscale_models",
        "kind": KIND_MODEL,
        "display_name": "Upscaling models",
        "status_fn": lambda: _slot_status(
            "Upscaling models (Real-ESRGAN / SwinIR / …)",
            os.path.join(get_models_dir(), "upscalers"),
            KIND_MODEL,
        ),
        "reinstall_fn": None,
        "open_label": "Open bin/models/upscalers",
    },
    {
        "id": "interp_models",
        "kind": KIND_MODEL,
        "display_name": "Interpolation models",
        "status_fn": lambda: _slot_status(
            "Interpolation models (RIFE / DAIN / …)",
            os.path.join(get_models_dir(), "interpolators"),
            KIND_MODEL,
        ),
        "reinstall_fn": None,
        "open_label": "Open bin/models/interpolators",
    },
    {
        "id": "colorize_models",
        "kind": KIND_MODEL,
        "display_name": "Colorization models",
        "status_fn": lambda: _slot_status(
            "Colorization models (DDColor / DeOldify / …)",
            os.path.join(get_models_dir(), "colorizers"),
            KIND_MODEL,
        ),
        "reinstall_fn": None,
        "open_label": "Open bin/models/colorizers",
    },
]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def list_all() -> list[dict]:
    """Devuelve el estado actual de cada dependencia registrada."""
    ensure_runtime_dirs()
    out = []
    for entry in REGISTRY:
        status_fn: Callable = entry["status_fn"]
        info = status_fn()
        # Asegurar campos canónicos
        info.setdefault("id", entry.get("id"))
        info.setdefault("display_name", entry.get("display_name", entry.get("id", "")))
        info.setdefault("kind", entry.get("kind", "unknown"))
        info.setdefault("status", STATUS_COMING_SOON)
        info.setdefault("version", "—")
        info.setdefault("install_dir", None)
        info.setdefault("source_url", None)
        info.setdefault("is_updatable", bool(entry.get("is_updatable", False)))
        info.setdefault("size_bytes", 0)
        info.setdefault("detail", "")
        info["open_label"] = entry.get("open_label", "Open folder")
        out.append(info)
    return out


def reinstall(dep_id: str, progress_callback: Optional[Callable[[int], None]] = None,
              parent=None) -> bool:
    """Re-descarga / re-instala la dependencia indicada. Lanza excepción si falla."""
    ensure_runtime_dirs()
    for entry in REGISTRY:
        if entry.get("id") != dep_id:
            continue
        fn = entry.get("reinstall_fn")
        if not fn:
            raise RuntimeError(
                f"{dep_id} no tiene instalador automático todavía (coming soon)."
            )
        fn(progress_callback=progress_callback, parent=parent)
        return True
    raise KeyError(f"Dependencia desconocida: {dep_id}")


def get_executable_path(dep_id: str) -> Optional[str]:
    """Devuelve la ruta concreta del binario principal (si aplica)."""
    if dep_id == "ffmpeg":
        path = ffmpeg_manager.get_local_ffmpeg_path()
        return path if os.path.exists(path) else None
    return None


# ---------------------------------------------------------------------------
# Auto-update-on-close: descubre dependencias pendientes de actualizar
# ---------------------------------------------------------------------------

def pending_updatable_dependencies() -> list[dict]:
    """Devuelve las dependencias que cumplen TODAS las condiciones:

    - Declaradas ``updatable=True`` en ``REGISTRY`` (explicit opt-in).
    - Tienen ``reinstall_fn`` no nula (instalador automático disponible).
    - Su ``status_fn()`` reporta ``update_available`` True o un valor
      'truthy' (ej. ``needs_update``); el resto (None/False) las excluimos.

    Cada elemento del resultado incluye el ``entry`` completo + el ``info``
    ya calculado + el índice de ``REGISTRY`` (para preservar orden estable
    cuando se renderizan al usuario).

    Las ranuras no-operativas (``reinstall_fn`` None o ``update_available``
    ausente) **no** aparecen en el resultado: si el usuario activa
    ``auto_update_all_on_close`` pero el registry aún no tiene un actualizador
    automático para otra categoría, esa categoría simplemente se omite
    silenciosamente — sería peor pedirla. Los mensajes sobre categorías
    no-operativas se generan aparte en ``report_skipped_dependencies``.
    """
    out: list[dict] = []
    ensure_runtime_dirs()
    for idx, entry in enumerate(REGISTRY):
        if not entry.get("is_updatable"):
            continue
        reinstall = entry.get("reinstall_fn")
        if not reinstall:
            continue
        status_fn: Callable = entry["status_fn"]
        try:
            info = status_fn()
        except Exception:
            continue
        # Convención: cada status_fn declara update_available en su dict.
        # Solo entran las que digan True. None/False => no hay update.
        needs_update = info.get("update_available")
        if not needs_update:
            continue
        out.append(
            {
                "index": idx,
                "entry": entry,
                "info": info,
                "dep_id": entry.get("id"),
                "display_name": info.get("display_name")
                or entry.get("display_name")
                or entry.get("id", ""),
                "reinstall_fn": reinstall,
                # Introspection hook mas preciso que status_fn()["size_bytes"]:
                # un callable que mide la unidad logica (binario, modelo,
                # runtime) en lugar del tamano completo del install_dir.
                # El chain controller lo prefiere cuando esta disponible.
                "bytes_fn": entry.get("bytes_fn"),
            }
        )
    return out


def report_skipped_dependencies() -> list[dict]:
    """Para diagnóstico en ``notifications.log`` y para mostrar al usuario
    qué categorías están marcadas como ``is_updatable=True`` pero todavía
    no tienen upstream_url / reinstall automático.

    Devuelve la lista de entradas que SÍ declaran ``is_updatable`` pero
    carecen de ``reinstall_fn``.
    """
    skipped: list[dict] = []
    for entry in REGISTRY:
        if not entry.get("is_updatable"):
            continue
        if entry.get("reinstall_fn"):
            continue
        skipped.append(
            {
                "dep_id": entry.get("id"),
                "display_name": entry.get("display_name", entry.get("id", "")),
                "reason": "no_upstream_or_no_reinstall_fn",
            }
        )
    return skipped


def auto_update_all_pref_enabled() -> bool:
    """Lee la preferencia ``auto_update_all_on_close``.

    Reconoce el alias legacy ``auto_update_on_close`` para usuarios que
    ya tenían la versión FFmpeg-only activa — se trata como la misma
    preferencia, así nadie pierde su ajuste al actualizar.
    """
    try:
        import src.preferences as preferences
        new = preferences.get("auto_update_all_on_close", None)
        if new is not None:
            return bool(new)
        legacy = preferences.get("auto_update_on_close", False)
        return bool(legacy)
    except Exception:
        return False
