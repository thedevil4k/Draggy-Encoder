import json
import logging
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

from src.config import (
    FFMPEG_DOWNLOAD_URLS,
    FFMPEG_FULL_DOWNLOAD_URLS,
    ensure_runtime_dirs,
    get_bin_dir,
    get_ffmpeg_dir,
    get_ffmpeg_full_dir,
    get_ffmpeg_executable_name,
    get_ffprobe_executable_name,
    get_logs_dir,
)
from src.ffmpeg.downloader import download_ffmpeg, download_ffmpeg_full
from src.ffmpeg.upstream import (
    get_fingerprint,
    is_update_available,
    read_cache as read_upstream_cache,
)

MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_local_ffmpeg_path():
    """Ruta al binario ffmpeg dentro de la carpeta propia del programa."""
    return os.path.join(get_ffmpeg_dir(), get_ffmpeg_executable_name())


def get_local_ffprobe_path():
    """Ruta al binario ffprobe dentro de la carpeta propia del programa."""
    return os.path.join(get_ffmpeg_dir(), get_ffprobe_executable_name())


def get_local_ffmpeg_full_path():
    """Ruta al binario ffmpeg completo (con libplacebo) dentro de la carpeta propia."""
    return os.path.join(get_ffmpeg_full_dir(), get_ffmpeg_executable_name())


def get_local_ffprobe_full_path():
    """Ruta al binario ffprobe completo (con libplacebo) dentro de la carpeta propia."""
    return os.path.join(get_ffmpeg_full_dir(), get_ffprobe_executable_name())


def get_manifest_path():
    return os.path.join(get_ffmpeg_dir(), MANIFEST_FILENAME)


def find_system_ffmpeg():
    """Busca ffmpeg en el PATH del sistema (sólo diagnóstico, ya no se usa)."""
    return shutil_which("ffmpeg")


def find_system_ffprobe():
    return shutil_which("ffprobe")


def shutil_which(name):
    try:
        import shutil
        return shutil.which(name)
    except Exception:
        return None


def _emit(progress_callback, percent, message=""):
    """Wrapper fail-safe del callback de progreso.

    Acepta callbacks 1-arg (compat legacy de ``codec_tester``) ó 2-args
    (nuevo contrato de ``download_ffmpeg``). El caller decide qué fase está
    corriendo; aquí sólo nos aseguramos de no propagar excepciones del
    callback al flujo de descarga.
    """
    if not progress_callback:
        return
    try:
        try:
            progress_callback(int(percent), message)
        except TypeError:
            # Compat: callback legacy de 1 argumento.
            progress_callback(int(percent))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Manifest (provenance + version)
# ---------------------------------------------------------------------------

def read_manifest():
    """Lee el manifest.json o devuelve {} si no existe / está corrupto."""
    path = get_manifest_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_manifest(data):
    """Escribe el manifest al disco. Silencioso en caso de error de E/S."""
    ensure_runtime_dirs()
    path = get_manifest_path()
    try:
        payload = dict(data)
        payload.setdefault("id", "ffmpeg")
        payload.setdefault("installed_at", datetime.now().isoformat(timespec="seconds"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"ffmpeg version\s+([^\s]+)", re.IGNORECASE)


def _parse_ffmpeg_version_text(text):
    """Extrae la versión del bloque devuelto por ``ffmpeg -version``."""
    if not text:
        return None
    match = _VERSION_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: primera palabra tras el primer salto de línea
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    parts = first_line.split()
    if len(parts) >= 3 and parts[0].lower() == "ffmpeg":
        return parts[2]
    return None


def get_ffmpeg_version(ffmpeg_path=None):
    """Lee la versión del binario ffmpeg ejecutando ``-version``.

    Devuelve la cadena de versión o ``None`` si el binario falta o no responde.
    """
    path = ffmpeg_path or get_local_ffmpeg_path()
    if not path or not os.path.exists(path):
        return None
    try:
        result = subprocess.run(
            [path, "-version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        return _parse_ffmpeg_version_text(text)
    except Exception:
        return None


@lru_cache(maxsize=32)
def _has_libplacebo_at(ffmpeg_path: str, mtime: float) -> bool:
    """Cached probe: (binary path, mtime) identifies a unique FFmpeg build."""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-filters"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return "libplacebo" in output.lower()
    except Exception:
        return False


def has_libplacebo_support(ffmpeg_path=None):
    """Check if FFmpeg has libplacebo support (needed for FSR 3.1).

    Returns True if libplacebo filter is available, False otherwise.

    The probe is cached per (binary path, mtime), so rendering the AI
    Engines tab / refreshing the workflow never re-spawns ffmpeg for
    unchanged binaries, while a freshly downloaded FFmpeg full build
    (new mtime) is re-probed automatically.
    """
    path = ffmpeg_path or get_local_ffmpeg_full_path()
    if not path or not os.path.exists(path):
        # Check standard FFmpeg as fallback
        path = get_local_ffmpeg_path()
        if not path or not os.path.exists(path):
            return False
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return _has_libplacebo_at(path, mtime)


# ---------------------------------------------------------------------------
# Estado público (para dependencias y UI)
# ---------------------------------------------------------------------------

def get_install_info(ffmpeg_path=None):
    """Devuelve un dict con el estado actual de la instalación de FFmpeg."""
    ffmpeg = ffmpeg_path or get_local_ffmpeg_path()
    ffprobe = get_local_ffprobe_path()
    manifest = read_manifest()
    binary_missing = not (ffmpeg and os.path.exists(ffmpeg))
    version = None
    if not binary_missing:
        version = get_ffmpeg_version(ffmpeg)
    if not version:
        version = manifest.get("version")
    if binary_missing:
        status = "missing"
    elif version is None:
        status = "error"
    else:
        status = "installed"

    # Comprobar contra el upstream (lee cache, sin red). El panel se encarga de
    # refrescar la cache cada 24h desde su QTimer; aquí solo leemos.
    fingerprint = read_upstream_cache()
    update_available = is_update_available(manifest, fingerprint)

    # Check if FFmpeg full (with libplacebo) is available
    ffmpeg_full = get_local_ffmpeg_full_path()
    has_full = os.path.exists(ffmpeg_full) if ffmpeg_full else False
    libplacebo_available = has_libplacebo_support() if has_full else False

    return {
        "id": "ffmpeg",
        "display_name": "FFmpeg (encoder suite)",
        "kind": "binary",
        "status": status,
        "version": version or "unknown",
        "install_dir": get_ffmpeg_dir(),
        "ffmpeg_path": ffmpeg,
        "ffprobe_path": ffprobe if os.path.exists(ffprobe) else None,
        "manifest_path": get_manifest_path(),
        "source_url": manifest.get("source_url"),
        "installed_at": manifest.get("installed_at"),
        "system_path": find_system_ffmpeg(),
        "ffmpeg_exists": not binary_missing,
        "ffprobe_exists": os.path.exists(ffprobe) if ffprobe else False,
        # Datos de upstream (para la columna 'Latest' / 'Update available')
        "latest_fingerprint": fingerprint,
        "update_available": update_available,
        "last_check_iso": fingerprint.get("checked_at"),
        "upstream_error": fingerprint.get("last_error"),
        # FFmpeg full (libplacebo) status
        "ffmpeg_full_exists": has_full,
        "ffmpeg_full_path": ffmpeg_full if has_full else None,
        "libplacebo_available": libplacebo_available,
    }


# ---------------------------------------------------------------------------
# FFmpeg selection for FSR 3.1
# ---------------------------------------------------------------------------


def get_ffmpeg_path_for_fsr31():
    """Get the appropriate FFmpeg path for FSR 3.1 processing.

    Returns the path to FFmpeg with libplacebo support if available,
    otherwise falls back to standard FFmpeg.
    """
    # First try FFmpeg full (with libplacebo)
    ffmpeg_full = get_local_ffmpeg_full_path()
    if os.path.exists(ffmpeg_full) and has_libplacebo_support(ffmpeg_full):
        return ffmpeg_full

    # Fallback to standard FFmpeg
    return get_local_ffmpeg_path()


def ensure_ffmpeg_full(parent=None, progress_callback=None, abort_event=None,
                       timeout=None):
    """Ensure FFmpeg full (with libplacebo) is available for FSR 3.1.

    Tries to download FFmpeg full if not already present.
    Returns True if successful, False otherwise.
    ``abort_event`` cancela la descarga; ``timeout`` acota el tiempo total.
    """
    ffmpeg_full = get_local_ffmpeg_full_path()
    if os.path.exists(ffmpeg_full):
        # Already have it, just verify libplacebo support
        if has_libplacebo_support(ffmpeg_full):
            return True

    # Try to download FFmpeg full
    try:
        path = download_ffmpeg_full(
            parent=parent, progress_callback=progress_callback,
            abort_event=abort_event, timeout=timeout,
        )
        if path and os.path.exists(path):
            return has_libplacebo_support(path)
    except Exception as exc:
        logger.warning("Error downloading FFmpeg full: %s", exc)
        return False

    return False


# ---------------------------------------------------------------------------
# Instalación / re-instalación
# ---------------------------------------------------------------------------

def _build_install_manifest(ffmpeg_path, source_label):
    """Construye un payload de manifest con la huella upstream en el momento
    de la instalación. Si la red está caída seguimos siendo capaces de escribir
    un manifest válido con sólo los campos locales.
    """
    ensure_runtime_dirs()
    info = FFMPEG_DOWNLOAD_URLS.get(platform.system(), {})
    # Tomamos una huella fresca para saber qué hay hoy en el upstream; si esto
    # falla (sin red) seguimos con el cache anterior o un dict vacío.
    try:
        fp = get_fingerprint(force=True)
    except Exception:
        fp = read_upstream_cache() or {}

    payload = {
        "id": "ffmpeg",
        "version": get_ffmpeg_version(ffmpeg_path) or "unknown",
        "source_url": info.get("url"),
        "archive_type": info.get("archive_type"),
        "source": source_label,
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "upstream_last_modified": fp.get("last_modified_iso"),
        "upstream_content_length": fp.get("content_length"),
        "upstream_etag": fp.get("etag"),
        "upstream_url": fp.get("url") or info.get("url"),
    }
    return payload


def _migrate_legacy_layout():
    """Migra el layout plano antiguo (``bin/ffmpeg.exe``) al nuevo (``bin/ffmpeg/...``).

    Estrategia copy → verify → delete para que un fallo a mitad de camino
    deje los originales intactos y nunca perdamos una instalación manual del
    usuario. El manifest sólo se escribe si AMBOS binarios están en la nueva
    ruta; en caso contrario la migración se aborta sin tocar nada, dejando
    que ``ensure_ffmpeg`` caiga al camino de descarga limpio.
    """
    try:
        old_ffmpeg = os.path.join(get_bin_dir(), get_ffmpeg_executable_name())
        old_ffprobe = os.path.join(get_bin_dir(), get_ffprobe_executable_name())
        new_ffmpeg = get_local_ffmpeg_path()
        new_ffprobe = get_local_ffprobe_path()
        # Idempotente: si el nuevo layout ya tiene ffmpeg o ffprobe, no tocar.
        if (
            not os.path.exists(old_ffmpeg)
            or os.path.exists(new_ffmpeg)
            or os.path.exists(new_ffprobe)
        ):
            return False
        ensure_runtime_dirs()

        # 1) Copiar ambos a la nueva ubicación
        shutil.copy2(old_ffmpeg, new_ffmpeg)
        if os.path.exists(old_ffprobe):
            shutil.copy2(old_ffprobe, new_ffprobe)

        # 2) Verificar que la copia aterrizó antes de borrar nada
        if not (os.path.exists(new_ffmpeg) and os.path.exists(new_ffprobe)):
            return False

        # 3) Manifest sólo si la copia es completa, con huella upstream
        payload = _build_install_manifest(new_ffmpeg, "migrated-from-flat-layout")
        wrote = write_manifest(payload)
        if not wrote:
            return False

        # 4) Borrar originales (best-effort; el estado "nuevo" ya es válido)
        for p in (old_ffmpeg, old_ffprobe):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

        # 5) Permisos Linux
        if platform.system() != "Windows":
            for p in (new_ffmpeg, new_ffprobe):
                try:
                    if os.path.exists(p):
                        os.chmod(p, 0o755)
                except OSError:
                    pass
        return True
    except Exception as exc:
        logger.warning("Legacy FFmpeg migration failed: %s", exc)
        return False


def ensure_ffmpeg(parent=None, progress_callback=None, prefer_full=True,
                  abort_event=None, timeout=None):
    """Asegura que FFmpeg esté disponible dentro de la carpeta del programa.

    Args:
        parent: QWidget padre (futuras integraciones con QFileDialog).
        progress_callback: callable(percent, message) -> None. Recibe
            updates durante la descarga y la extracción. Compatible con
            callbacks de 1 argumento vía ``_emit``.
        prefer_full: Si es True, intenta descargar FFmpeg completo (con libplacebo)
            primero. Si falla, cae al FFmpeg estándar.
        abort_event: threading.Event opcional para cancelar la descarga.
        timeout: presupuesto total de pared en segundos (opcional).

    Reglas (alineadas con el requisito del usuario):
      1. Si existe el binario local -> se usa, sin fallback al sistema.
      2. Si no existe -> se descarga automáticamente a ``bin/ffmpeg/`` o ``bin/ffmpeg-full/``.
      3. No se consulta nunca el PATH del sistema para el flujo normal; sólo
         se expone a través de ``get_install_info()`` como diagnóstico.

    Devuelve la ruta al ejecutable o ``None`` si no se pudo obtener.
    """
    # Migración puntual del layout antiguo (bin/ffmpeg.exe -> bin/ffmpeg/ffmpeg.exe)
    _migrate_legacy_layout()

    # Si prefer_full, intentar FFmpeg completo primero
    if prefer_full:
        local_full = get_local_ffmpeg_full_path()
        if os.path.exists(local_full) and has_libplacebo_support(local_full):
            _emit(progress_callback, 100, "FFmpeg full ready (cached)")
            return local_full

    local = get_local_ffmpeg_path()
    if os.path.exists(local) and os.path.exists(get_local_ffprobe_path()):
        # Refrescar manifest si no existe (binario traído manualmente).
        if not os.path.exists(get_manifest_path()):
            write_manifest(_manifest_from_binary(local))
        _emit(progress_callback, 100, "FFmpeg ready (cached)")
        return local

    ensure_runtime_dirs()
    try:
        if prefer_full:
            # Intentar descargar FFmpeg completo primero
            _emit(progress_callback, 0, "Downloading FFmpeg full (with libplacebo)...")
            try:
                path = download_ffmpeg_full(
                    parent=parent, progress_callback=progress_callback,
                    abort_event=abort_event, timeout=timeout,
                )
                if path and os.path.exists(path):
                    # Verify libplacebo support
                    if has_libplacebo_support(path):
                        _emit(progress_callback, 100, "FFmpeg full ready")
                        return path
                    else:
                        # FFmpeg full downloaded but no libplacebo, fall back to standard
                        _emit(progress_callback, 0, "FFmpeg full doesn't have libplacebo, trying standard...")
            except Exception as exc:
                _emit(progress_callback, 0, f"FFmpeg full download failed: {exc}, trying standard...")
                logger.warning("FFmpeg full download failed: %s", exc)

        # Fallback to standard FFmpeg
        path = download_ffmpeg(
            parent=parent, progress_callback=progress_callback,
            abort_event=abort_event, timeout=timeout,
        )
    except Exception as exc:
        # Registrar el error en bin/logs/ffmpeg_download.log para diagnóstico.
        try:
            log_dir = get_logs_dir()
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "ffmpeg_download.log"), "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] ensure_ffmpeg download failed: {exc}\n")
        except Exception:
            pass
        logger.error("Error descargando FFmpeg: %s", exc)
        return None
    # Escribir manifest con datos de la descarga + huella upstream
    write_manifest(_build_install_manifest(path, "first-download"))
    # ``download_ffmpeg`` ya emite 100%/Ready al terminar; no re-emitimos.
    return path


def force_redownload(progress_callback=None, parent=None, abort_event=None,
                     timeout=None):
    """Borra la instalación actual y descarga de nuevo.

    Lanza ``RuntimeError`` si la descarga falla o si tras la descarga falta el
    binario (el caller debe traducirlo al UI). ``abort_event`` cancela la
    descarga; ``timeout`` acota el tiempo total.
    """
    ffmpeg_dir = get_ffmpeg_dir()
    # Borrar binarios anteriores (si existen) — conservamos la propia carpeta.
    for fname in (get_ffmpeg_executable_name(), get_ffprobe_executable_name(), MANIFEST_FILENAME):
        try:
            target = os.path.join(ffmpeg_dir, fname)
            if os.path.exists(target):
                os.remove(target)
        except Exception:
            pass

    ensure_runtime_dirs()
    path = download_ffmpeg(
        progress_callback=progress_callback, parent=parent,
        abort_event=abort_event, timeout=timeout,
    )
    write_manifest(_build_install_manifest(path, "user-reinstall"))
    if not path or not os.path.exists(path):
        raise RuntimeError("Re-download finished but FFmpeg binary is missing.")
    return path


def _manifest_from_binary(ffmpeg_path):
    """Genera un manifest mínimo cuando el binario ya estaba presente sin manifest."""
    system = platform.system()
    info = FFMPEG_DOWNLOAD_URLS.get(system, {})
    return {
        "id": "ffmpeg",
        "version": get_ffmpeg_version(ffmpeg_path) or "unknown",
        "source_url": info.get("url"),
        "source": "manual-copy",
    }


def uninstall(parent=None):
    """Borra la instalación local de FFmpeg (binarios + manifest + cache).

    La app lo re-descarga automáticamente en el próximo arranque
    (``app.py`` → ``ensure_ffmpeg``). Si algún archivo está bloqueado
    (render en curso), ``dependencies._rmtree_force`` lo programa para
    el próximo arranque.
    """
    try:
        from src.dependencies import _rmtree_force
    except Exception:
        return
    ffmpeg_dir = get_ffmpeg_dir()
    if os.path.isdir(ffmpeg_dir):
        _rmtree_force(ffmpeg_dir)


def uninstall_full(parent=None):
    """Borra la instalación de FFmpeg 'full' (libplacebo, FSR 3.1)."""
    try:
        from src.dependencies import _rmtree_force
    except Exception:
        return
    full_dir = get_ffmpeg_full_dir()
    if os.path.isdir(full_dir):
        _rmtree_force(full_dir)


# ---------------------------------------------------------------------------
# Compatibilidad con código existente
# ---------------------------------------------------------------------------

def get_ffprobe_path():
    """Devuelve la ruta a ffprobe local; busca en ffmpeg dir, ffmpeg-full, y PATH."""
    # 1. Intentar con el ffmpeg estándar (bin/ffmpeg/ffprobe.exe)
    local = get_local_ffprobe_path()
    if os.path.exists(local):
        return local
    # 2. Intentar con el ffmpeg-full con libplacebo (bin/ffmpeg-full/ffprobe.exe)
    full = get_local_ffprobe_full_path()
    if os.path.exists(full):
        return full
    # 3. Último recurso: buscar en PATH del sistema
    system = find_system_ffprobe()
    if system:
        return system
    return None
