"""Registro unificado de dependencias externas del programa.

El panel de gestión muestra FFmpeg, FFmpeg full (libplacebo), los runtimes
de IA (PyTorch/ONNX Runtime) y las ranuras de modelos por categoría. El
sensor de estado es *honesto contra disco*: cada ``status_fn`` comprueba
la realidad (binario, store local ``bin/runtimes/ai_modules``, motores bajo
``bin/models/<Category>``) en lugar de adivinar.

La capa ``src.dependencies`` está pensada para que añadir una dependencia
nueva sea puramente declarativo: una entrada en ``REGISTRY`` + una función
``status`` que devuelva un dict con los campos expuestos aquí. La UI consume
este módulo a través de ``list_all()``, ``reinstall()`` y ``uninstall()``
exclusivamente.

El borrado es robusto en Windows: si un archivo (DLL de torch/onnxruntime,
binario FFmpeg en uso) está bloqueado, ``_rmtree_force`` registra la ruta
en ``bin/.session/pending_cleanup.json`` y ``process_pending_cleanup()`` la
completa en el siguiente arranque, antes de importar ningún runtime.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Optional

from src.config import (
    ensure_runtime_dirs,
    get_ffmpeg_full_dir,
    get_models_dir,
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
# Borrado robusto + cleanup pendiente (archivos DLL bloqueados en Windows)
# ---------------------------------------------------------------------------

PENDING_CLEANUP_FILENAME = "pending_cleanup.json"


def _pending_cleanup_path():
    from src.config import get_session_dir
    return os.path.join(get_session_dir(), PENDING_CLEANUP_FILENAME)


def _register_pending_cleanup(path: str) -> None:
    """Registra una ruta para borrarla en el próximo arranque (best-effort).

    Windows bloquea DLLs cargadas (torch/onnxruntime ya importados). No
    podemos borrarlos en caliente; al siguiente arranque, antes de importar
    cualquier runtime, ``process_pending_cleanup()`` completa el borrado.
    """
    try:
        ensure_runtime_dirs()
        p = _pending_cleanup_path()
        data = {}
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("paths", [])
        if path not in data["paths"]:
            data["paths"].append(path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def process_pending_cleanup() -> None:
    """Completa borrados pendientes del arranque anterior.

    DEBE llamarse antes de importar ``torch``/``onnxruntime`` (en ``main.py``
    justo después de ``inject_runtime_paths()``) para que los DLLs no estén
    todavía bloqueados. Idempotente y tolerante a errores de E/S.
    """
    p = _pending_cleanup_path()
    if not os.path.exists(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        data = {}
    paths = list(data.get("paths") or [])
    remaining = []
    for path in paths:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
        if os.path.exists(path):
            remaining.append(path)
    try:
        if remaining:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"paths": remaining}, f, indent=2)
        else:
            os.remove(p)
    except Exception:
        pass


def _rmtree_force(path: str) -> bool:
    """Borra recursivamente ``path``; si queda algo bloqueado, lo programa
    para el próximo arranque. Devuelve True si el borrado quedó completo."""
    if not os.path.isdir(path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return not os.path.exists(path)
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
    if os.path.exists(path):
        _register_pending_cleanup(path)
        return False
    return True


def _dir_has_files(path: str) -> bool:
    """True si el directorio contiene al menos un archivo (recursivo)."""
    try:
        for dirpath, _, filenames in os.walk(path):
            if filenames:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Factories para runtimes / ranuras de modelos
# ---------------------------------------------------------------------------

def _runtime_status(engine_id, display_name, source_url, manager_status_fn):
    """Estado de un runtime IA (torch/onnxruntime) usando la fuente única de
    verdad: ``runtimes.manager``.

    La app es autocontenida ("caracol"): SOLO cuenta el runtime descargado
    por ella misma en ``bin/runtimes/ai_modules``. Una copia del entorno del
    sistema (venv, otros programas) se menciona como diagnóstico en el
    mensaje, pero NO satisface la dependencia: sin store local => MISSING,
    aunque el sistema tenga torch/onnxruntime.

    Antes este sensor consultaba ``importlib.metadata`` sobre el entorno Python
    y reportaba INSTALLED aunque ``bin/runtimes/ai_modules`` estuviera vacío
    (0 bytes). Ahora el badge refleja únicamente el runtime local.
    """
    ensure_runtime_dirs()
    try:
        from src.runtimes.manager import inject_runtime_paths
        inject_runtime_paths()
    except Exception:
        pass
    info = manager_status_fn() or {}
    installed = bool(info.get("installed"))
    version = info.get("version") or "—"
    store = ai_runtimes_manager.get_runtime_modules_dir()
    local_bytes = _safe_dir_size(store)
    gpu = bool(info.get("gpu"))
    system = bool(info.get("system"))

    if installed:
        status = STATUS_INSTALLED
        gpu_text = " · GPU (CUDA) disponible" if gpu else " · solo CPU"
        detail = f"Instalado en la carpeta del programa. Versión: {version}{gpu_text}"
    elif _runtime_package_downloaded(store, engine_id):
        # Descargado en el store local pero el import falla. Dos causas
        # típicas: el store se instaló con OTRA versión de Python (wheels
        # cpXY incompatibles, no inyectados por el guard de versión), o la
        # descarga quedó a medias / faltan DLLs de CUDA.
        if not ai_runtimes_manager.store_is_compatible_with_python():
            status = STATUS_ERROR
            detail = (
                "La descarga anterior se hizo con otra versión de Python. "
                "Pulsa 'Reintentar' para volver a descargarlo con la versión "
                "actual; no hace falta borrar nada."
            )
        else:
            status = STATUS_ERROR
            detail = (
                "La descarga quedó incompleta o interrumpida. "
                "Pulsa 'Reintentar' para volver a descargarlo."
            )
    else:
        status = STATUS_MISSING
        if system:
            detail = (
                "No está instalado en la carpeta del programa. "
                "Pulsa 'Instalar' para descargarlo."
            )
        else:
            detail = "No está instalado. Pulsa 'Instalar' para descargarlo."

    return {
        "id": engine_id,
        "display_name": display_name,
        "kind": KIND_RUNTIME,
        "status": status,
        "version": version,
        # install_dir: el store local. Como la app solo usa su propio
        # runtime, el botón 'Open folder' siempre apunta ahí (la carpeta
        # donde se descargará / vive la copia local).
        "install_dir": store,
        "source_url": source_url,
        "is_updatable": True,
        "size_bytes": local_bytes,
        "detail": detail,
        "local": installed,
        "deletable": installed and local_bytes > 0,
        "gpu_available": gpu,
    }


def _runtime_package_downloaded(store: str, engine_id: str) -> bool:
    """True si pip descargó el paquete del runtime en el store local.

    pip install --target deja un directorio con el nombre del paquete
    (``torch/``, ``onnxruntime/``) + su ``*.dist-info``. Comprobar la
    existencia del directorio distingue 'descargado pero roto' (import
    falla: DLLs CUDA, descarga a medias) de 'ni siquiera intentado'.
    """
    pkg_dir = {
        "torch": "torch",
        "onnxruntime": "onnxruntime",
    }.get(engine_id)
    if not pkg_dir:
        return False
    if not os.path.isdir(os.path.join(store, pkg_dir)):
        return False
    try:
        return any(
            os.path.isdir(os.path.join(store, e)) and e.endswith(".dist-info")
            for e in os.listdir(store)
        )
    except Exception:
        return True


def _pytorch_status():
    return _runtime_status(
        "torch",
        "PyTorch runtime",
        "https://pytorch.org/get-started/locally/",
        ai_runtimes_manager.torch_status,
    )


def _onnxruntime_status():
    return _runtime_status(
        "onnxruntime",
        "ONNX Runtime (CPU/GPU)",
        "https://pypi.org/project/onnxruntime/",
        ai_runtimes_manager.onnx_status,
    )


def _slot_status(display_name, category, install_dir, kind):
    """Estado real de una ranura de modelos: cuenta motores instalados en
    ``bin/models/<Category>`` y reporta el tamaño agregado.

    Antes devolvía siempre ``coming_soon`` sin mirar el disco. Ahora la
    ranura es un sensor honesto: installed si hay al menos un motor con
    archivos, missing si no hay nada.
    """
    ensure_runtime_dirs()
    # Limpieza de carpetas legacy del layout anterior (upscalers/
    # interpolators/ colorizers). Idempotente: solo borra si están vacías.
    for _legacy in ("upscalers", "interpolators", "colorizers"):
        _p = os.path.join(get_models_dir(), _legacy)
        if os.path.isdir(_p):
            try:
                if not any(os.scandir(_p)):
                    os.rmdir(_p)
            except Exception:
                pass
    count = 0
    total = 0
    if os.path.isdir(install_dir):
        try:
            for entry in os.listdir(install_dir):
                sub = os.path.join(install_dir, entry)
                if not os.path.isdir(sub):
                    continue
                if _dir_has_files(sub):
                    count += 1
                    total += _safe_dir_size(sub)
        except Exception:
            pass

    if count:
        status = STATUS_INSTALLED
        version = f"{count} engine(s)"
        detail = (
            f"{count} engine(s) installed in the program folder ({install_dir}). "
            "Manage individual engines from the 'AI Engines' tab."
        )
    else:
        status = STATUS_MISSING
        version = "—"
        detail = (
            "No models installed here yet. Use the 'AI Engines' tab to add engines."
        )

    return {
        "id": None,  # list_all() lo rellena con el id del REGISTRY
        "display_name": display_name,
        "kind": kind,
        "status": status,
        "version": version,
        "install_dir": install_dir,
        "source_url": None,
        "is_updatable": False,
        "size_bytes": total,
        "detail": detail,
        "deletable": count > 0,
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
        "deletable": status == STATUS_INSTALLED,
        # Datos de upstream (para Latest/Update available en la UI)
        "latest_fingerprint": info.get("latest_fingerprint") or {},
        "update_available": info.get("update_available"),
        "last_check_iso": info.get("last_check_iso"),
        "upstream_error": info.get("upstream_error"),
    }


def _ffmpeg_full_status():
    """Estado del binario FFmpeg 'full' (con libplacebo, FSR 3.1)."""
    ensure_runtime_dirs()
    from src.ffmpeg.manager import (
        get_local_ffmpeg_full_path,
        has_libplacebo_support,
    )
    path = get_local_ffmpeg_full_path()
    exists = bool(path) and os.path.exists(path)
    if exists and has_libplacebo_support(path):
        status = STATUS_INSTALLED
        detail = (
            "FFmpeg full (libplacebo) installed in the program folder. "
            "Required for FSR 3.1."
        )
    elif exists:
        status = STATUS_ERROR
        detail = "FFmpeg full present but without libplacebo support."
    else:
        status = STATUS_MISSING
        detail = (
            "Not installed. Download it from the 'AI Engines' tab "
            "(FSR 3.1 rows)."
        )
    return {
        "id": "ffmpeg_full",
        "display_name": "FFmpeg full (libplacebo)",
        "kind": KIND_BINARY,
        "status": status,
        "version": ffmpeg_manager.get_ffmpeg_version(path) or "—",
        "install_dir": get_ffmpeg_full_dir(),
        "source_url": "https://github.com/BtbN/FFmpeg-Builds",
        "is_updatable": False,
        "size_bytes": _safe_dir_size(get_ffmpeg_full_dir()),
        "detail": detail,
        "deletable": exists,
        "latest_fingerprint": {},
        "update_available": False,
        "last_check_iso": None,
        "upstream_error": None,
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
        "uninstall_fn": ffmpeg_manager.uninstall,
        "open_label": "Open bin/ffmpeg folder",
        "is_updatable": True,
        # Size/introspection hook: el chain controller lo usa para capturar
        # el tamano del binario ANTES y DESPUES de la reinstalacion, y asi
        # alimentar el mini-stats banner + el historial persistente.
        "bytes_fn": lambda: _local_ffmpeg_size_or_none(),
    },
    {
        "id": "ffmpeg_full",
        "kind": KIND_BINARY,
        "display_name": "FFmpeg full (libplacebo)",
        "status_fn": _ffmpeg_full_status,
        "reinstall_fn": None,
        "uninstall_fn": ffmpeg_manager.uninstall_full,
        "open_label": "Open bin/ffmpeg-full folder",
        "is_updatable": False,
    },
    {
        "id": "torch",
        "kind": KIND_RUNTIME,
        "display_name": "PyTorch runtime",
        "status_fn": _pytorch_status,
        "reinstall_fn": ai_runtimes_manager.install_pytorch,
        "uninstall_fn": ai_runtimes_manager.uninstall_local_runtimes,
        "open_label": "Open bin/runtimes/ai_modules",
        "is_updatable": True,
    },
    {
        "id": "onnxruntime",
        "kind": KIND_RUNTIME,
        "display_name": "ONNX Runtime (CPU/GPU)",
        "status_fn": _onnxruntime_status,
        "reinstall_fn": ai_runtimes_manager.install_onnxruntime,
        "uninstall_fn": ai_runtimes_manager.uninstall_local_runtimes,
        "open_label": "Open bin/runtimes/ai_modules",
        "is_updatable": True,
    },
    {
        "id": "upscale_models",
        "kind": KIND_MODEL,
        "display_name": "Upscaling models",
        "status_fn": lambda: _slot_status(
            "Upscaling models (Real-ESRGAN / SwinIR / …)",
            "Upscale",
            os.path.join(get_models_dir(), "upscale"),
            KIND_MODEL,
        ),
        "reinstall_fn": None,
        "uninstall_fn": lambda parent=None: _uninstall_category_models("Upscale"),
        "open_label": "Open bin/models/upscale",
    },
    {
        "id": "interp_models",
        "kind": KIND_MODEL,
        "display_name": "Interpolation models",
        "status_fn": lambda: _slot_status(
            "Interpolation models (RIFE / DAIN / …)",
            "Interpolate",
            os.path.join(get_models_dir(), "interpolate"),
            KIND_MODEL,
        ),
        "reinstall_fn": None,
        "uninstall_fn": lambda parent=None: _uninstall_category_models("Interpolate"),
        "open_label": "Open bin/models/interpolate",
    },
    {
        "id": "colorize_models",
        "kind": KIND_MODEL,
        "display_name": "Colorization models",
        "status_fn": lambda: _slot_status(
            "Colorization models (DDColor / DeOldify / …)",
            "Colorize",
            os.path.join(get_models_dir(), "colorize"),
            KIND_MODEL,
        ),
        "reinstall_fn": None,
        "uninstall_fn": lambda parent=None: _uninstall_category_models("Colorize"),
        "open_label": "Open bin/models/colorize",
    },
    {
        "id": "face_restore_models",
        "kind": KIND_MODEL,
        "display_name": "Face restoration models",
        "status_fn": lambda: _slot_status(
            "Face restoration models (GFPGAN / CodeFormer / GPEN / …)",
            "FaceRestore",
            os.path.join(get_models_dir(), "facerestore"),
            KIND_MODEL,
        ),
        "reinstall_fn": None,
        "uninstall_fn": lambda parent=None: _uninstall_category_models("FaceRestore"),
        "open_label": "Open bin/models/facerestore",
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
        # El id canónico es el del REGISTRY (no permitimos que status_fn lo
        # pise con None: rompería reinstall/uninstall por clave).
        info["id"] = entry.get("id")
        # Asegurar campos canónicos
        info.setdefault("display_name", entry.get("display_name", entry.get("id", "")))
        info.setdefault("kind", entry.get("kind", "unknown"))
        info.setdefault("status", STATUS_COMING_SOON)
        info.setdefault("version", "—")
        info.setdefault("install_dir", None)
        info.setdefault("source_url", None)
        info.setdefault("is_updatable", bool(entry.get("is_updatable", False)))
        info.setdefault("size_bytes", 0)
        info.setdefault("detail", "")
        info.setdefault("deletable", info.get("status") == STATUS_INSTALLED)
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


def uninstall(dep_id: str, parent=None) -> bool:
    """Desinstala/borra la dependencia indicada desde la app.

    Lanza excepción si la dependencia no admite borrado o el borrado falla
    (la UI traduce el error). Para runtimes y FFmpeg con archivos bloqueados
    por Windows, el borrado se programa para el próximo arranque
    (``process_pending_cleanup()``).
    """
    ensure_runtime_dirs()
    for entry in REGISTRY:
        if entry.get("id") != dep_id:
            continue
        fn = entry.get("uninstall_fn")
        if not fn:
            raise RuntimeError(
                f"{dep_id} no se puede desinstalar desde la aplicación."
            )
        fn(parent=parent)
        return True
    raise KeyError(f"Dependencia desconocida: {dep_id}")


def remove_local_path(path: str) -> bool:
    """Borra un directorio/archivo local (fuerza + cleanup pendiente).

    Usado por el borrado de motores IA desde la pestaña AI Engines. Devuelve
    True si el borrado quedó completo; si algún archivo queda bloqueado, se
    programa para el próximo arranque.
    """
    return _rmtree_force(path)


def _uninstall_category_models(category: str, parent=None) -> None:
    """Borra todos los motores instalados de una categoría + su estado.

    El directorio en disco es el prefijo del engine_id en minúsculas
    (``bin/models/upscale``, ``bin/models/facerestore``, ...), no el nombre
    de categoría tal cual aparece en la UI ("Upscale", "FaceRestore").
    """
    try:
        from src.ai.engines_state import all_states, mark_uninstalled
        from src.ai.registry import CATALOG
    except Exception:
        return
    cat_dir = os.path.join(get_models_dir(), category.lower())
    if os.path.isdir(cat_dir):
        for sub in os.listdir(cat_dir):
            target = os.path.join(cat_dir, sub)
            if os.path.isdir(target):
                _rmtree_force(target)
    try:
        states = all_states()
    except Exception:
        states = {}
    for entry in CATALOG:
        if entry.get("category") != category:
            continue
        eid = entry.get("engine_id")
        if not eid:
            continue
        state = states.get(eid) or {}
        if state.get("installed"):
            try:
                mark_uninstalled(eid)
            except Exception:
                pass


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
