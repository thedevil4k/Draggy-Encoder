"""Pack GPU offline de NVIDIA: importación sin pip ni red.

El instalador de Draggy Encoder NO lleva torch/CUDA (pesan varios GB). En un
equipo sin internet, la app no puede bajar el runtime en el primer arranque.
Este módulo importa un *pack* descargado aparte: un conjunto de ZIPs de menos
de 2 GiB más un ``gpu-pack.json`` con el SHA-256 de cada parte.

Formato del pack
----------------
``gpu-pack.json`` y las partes viven en la MISMA carpeta::

    gpu-pack/
      gpu-pack.json
      gpu-pack-nvidia-cu130-win-x64-cp313-part01.zip
      gpu-pack-nvidia-cu130-win-x64-cp313-part02.zip

Cada ZIP contiene archivos con rutas relativas a la raíz del store
(``torch/...``, ``onnxruntime/...``, ``numpy/...``). Las partes son
independientes: se extraen correlativamente sobre el mismo staging, así que
un asset corrupto se puede volver a descargar solo, sin rearmar un archivo
gigante ni necesitar el doble de disco para concatenarlo.

Por qué no se usa pip
---------------------
``pip install --target`` exige red, un intérprete con pip y resuelve
dependencias en el equipo del usuario. El pack ya trae el cierre completo de
dependencias, compilado para el CPython del bundle (cp313). Importarlo es solo
verificar, extraer e intercambiar el store.

Garantías
---------
- El ``python_tag`` del manifiesto debe coincidir con el intérprete actual:
  un wheel cp314 en un Python 3.13 no importa.
- Cada parte se verifica por tamaño y SHA-256 ANTES de extraer nada.
- La extracción va a ``bin/runtimes/.staging`` (nunca encima del store en uso)
  y rechaza rutas absolutas, ``..`` y symlinks.
- Se valida con una sonda en subproceso limpio que torch y onnxruntime
  importan DESDE el staging y que reportan CUDA. Si la GPU no está
  disponible, no se toca el store que ya funcionaba.
- El intercambio es un ``rename`` del store a ``.previous`` y del staging a su
  sitio; si el segundo falla, se restaura el anterior.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

from src import fsops
from src.runtimes import manager


#: El pack se construye y valida contra este SO. Linux/AppImage todavía no
#: tiene variante: allí el cierre CUDA de ONNX Runtime es distinto (wheels
#: ``nvidia-*`` + LD_LIBRARY_PATH) y no se ha verificado en un equipo Linux.
SUPPORTED_PACK_PLATFORM = "win32"

PACK_FORMAT = "draggy-gpu-pack/1"
MANIFEST_NAME = "gpu-pack.json"
MARKER_NAME = "gpu-pack-installed.json"

#: Carpeta que se busca junto al ejecutable (o en ``bin/``) al arrancar.
PACK_DIR_NAME = "gpu-pack"

logger = logging.getLogger(__name__)

#: Ruta explícita a ``gpu-pack.json`` (o a su carpeta) para el primer arranque.
ENV_PACK = "DRAGGY_GPU_PACK"

#: ``DRAGGY_OFFLINE=1`` prohíbe cualquier instalación por red.
ENV_OFFLINE = "DRAGGY_OFFLINE"

#: GitHub Releases rechaza assets de 2 GiB o más. El builder deja margen.
MAX_ASSET_BYTES = 2 * 1024 ** 3

#: Espacio extra que se exige para no llenar la unidad a mitad del swap.
FREE_SPACE_MARGIN = 512 * 1024 ** 2

#: En Windows el rename puede fallar un instante por antivirus, un subproceso
#: de la sonda o un handle de DLL. Se reintenta solo ese tipo de error, con
#: espera creciente; un error permanente (p. ej. otra unidad) falla igual.
_SWAP_RETRY_DELAYS = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0)

#: Sonda propia (no la del manager) porque el staging no es el store activo y
#: ORT necesita las DLL de CUDA/cuDNN que torch trae en ``torch/lib``. El host
#: de IA hace exactamente esto antes de inferir
#: (``ensure_runtime_dlls_on_path``); la sonda debe medir las mismas
#: condiciones o aprobaría un pack que luego falla en el primer frame.
_PACK_PROBE_SOURCE = r'''
import importlib
import os
import sys

store = os.path.abspath(sys.argv[1])
extra = [store]
for relative in ("torch/lib", "onnxruntime/capi", "onnxruntime/capi/build"):
    path = os.path.join(store, *relative.split("/"))
    if os.path.isdir(path):
        extra.append(path)
nvidia = os.path.join(store, "nvidia")
if os.path.isdir(nvidia):
    for name in os.listdir(nvidia):
        path = os.path.join(nvidia, name, "bin")
        if os.path.isdir(path):
            extra.append(path)

# Keep the DLL-directory handles alive until the import finishes.
handles = []
for path in extra:
    try:
        handles.append(os.add_dll_directory(path))
    except OSError:
        pass
os.environ["PATH"] = ";".join(extra) + ";" + os.environ.get("PATH", "")

sys.path.insert(0, store)
try:
    module = importlib.import_module(sys.argv[2])
except BaseException:
    import traceback
    traceback.print_exc()
    sys.exit(3)

origin = os.path.abspath(getattr(module, "__file__", "") or "")
ok = os.path.normcase(origin).startswith(os.path.normcase(store) + os.sep)
caps = []
try:
    if sys.argv[2] == "torch":
        caps.append("cuda" if module.cuda.is_available() else "cpu")
    elif sys.argv[2] == "onnxruntime":
        caps.append(
            "cuda"
            if "CUDAExecutionProvider" in module.get_available_providers()
            else "cpu"
        )
except BaseException:
    pass

print("DRAGGY-CAPS=" + ",".join(caps))
print("DRAGGY-IMPORT-OK" if ok else "DRAGGY-IMPORT-FOREIGN")
sys.exit(0 if ok else 4)
'''

_PROBE_TIMEOUT_S = 300.0

_CHUNK = 4 * 1024 * 1024

_OFFLINE_VALUES = ("1", "true", "yes", "on")


class OfflinePackError(RuntimeError):
    """El pack no se puede importar (formato, hash, GPU o espacio)."""


# ---------------------------------------------------------------------------
# Modo offline
# ---------------------------------------------------------------------------


def pack_supported(platform_name: str | None = None) -> bool:
    """True solo en Windows x64, la única plataforma con pack offline.

    Se mantiene como función (y no como un ``if sys.platform`` repartido por
    la UI) para que el mensaje que ve el usuario y la decisión de buscar el
    pack no puedan discrepar.
    """
    return (platform_name or sys.platform) == SUPPORTED_PACK_PLATFORM


def is_offline_mode() -> bool:
    """True si el usuario pidió no tocar la red para instalar runtimes.

    No se detecta la conectividad (una sonda de red puede tardar y mentir):
    es una decisión explícita, la misma que toma quien descarga el pack.
    """
    return os.environ.get(ENV_OFFLINE, "").strip().lower() in _OFFLINE_VALUES


# ---------------------------------------------------------------------------
# Localización del manifiesto
# ---------------------------------------------------------------------------


def _manifest_in(path: str) -> str | None:
    """``gpu-pack.json`` a partir de un archivo o de su carpeta."""
    if not path:
        return None
    expanded = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(expanded):
        candidate = os.path.join(expanded, MANIFEST_NAME)
        return candidate if os.path.isfile(candidate) else None
    if os.path.isfile(expanded):
        if os.path.basename(expanded).lower() == MANIFEST_NAME:
            return expanded
        if expanded.lower().endswith(".zip"):
            # El usuario selecciona la primera parte: el manifiesto está al lado.
            sibling = os.path.join(os.path.dirname(expanded), MANIFEST_NAME)
            return sibling if os.path.isfile(sibling) else None
        return expanded
    return None


def find_pack() -> str | None:
    """Ruta del manifiesto del pack, o ``None`` si no hay ninguno.

    Orden: variable de entorno, ``bin/gpu-pack`` y ``gpu-pack`` junto al
    ejecutable/raíz del repo. Todo son comprobaciones de existencia: no se
    lee ni se extrae nada en el arranque si no hay pack.
    """
    if not pack_supported():
        if os.environ.get(ENV_PACK):
            logger.warning(
                "DRAGGY_GPU_PACK está definido, pero el pack GPU offline solo "
                "existe para Windows x64; se ignora en %s",
                sys.platform,
            )
        return None
    candidates: list[str] = []
    env_path = os.environ.get(ENV_PACK)
    if env_path:
        candidates.append(env_path)
    try:
        from src.config import get_base_dir, get_bin_dir

        candidates.append(os.path.join(get_bin_dir(), PACK_DIR_NAME))
        candidates.append(os.path.join(get_base_dir(), PACK_DIR_NAME))
    except Exception:
        pass
    for candidate in candidates:
        manifest = _manifest_in(candidate)
        if manifest:
            return manifest
    return None


def _marker_path() -> str:
    return os.path.join(manager._install_base_dir(), MARKER_NAME)


def installed_pack_info() -> dict:
    """Manifiesto reducido de lo que se importó por última vez (o ``{}``)."""
    try:
        with open(_marker_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Validación del manifiesto
# ---------------------------------------------------------------------------


def _normalize_arch(value: str) -> str:
    text = (value or "").strip().lower().replace("-", "_")
    if text in ("x86_64", "amd64", "x64", "win64"):
        return "amd64"
    if text in ("aarch64", "arm64", "arm64e"):
        return "arm64"
    return text


def _current_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def load_manifest(path: str) -> dict:
    """Lee y valida la estructura del manifiesto (sin tocar las partes)."""
    manifest_path = _manifest_in(path)
    if not manifest_path:
        raise OfflinePackError(
            f"No se encontró {MANIFEST_NAME} en {path!r}. Descarga el "
            "manifiesto junto con las partes del pack."
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise OfflinePackError(
            f"No se pudo leer {manifest_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise OfflinePackError(f"{manifest_path} no contiene un objeto JSON.")
    data["_manifest_path"] = manifest_path
    data["_manifest_dir"] = os.path.dirname(manifest_path)
    return data


def validate_manifest(manifest: dict) -> dict:
    """Comprueba que el pack corresponde a esta app, CPU y plataforma.

    Se ejecuta ANTES de leer las partes: un pack de otra versión de Python o
    de Linux no debe consumir minutos de hash para acabar en un error de
    import.
    """
    if manifest.get("format") != PACK_FORMAT:
        raise OfflinePackError(
            "El manifiesto no es de este formato (se esperaba "
            f"{PACK_FORMAT!r}, hay {manifest.get('format')!r})."
        )

    expected_platform = manifest.get("platform")
    if expected_platform and expected_platform != SUPPORTED_PACK_PLATFORM:
        raise OfflinePackError(
            "El pack GPU offline solo está disponible para Windows x64 y el "
            f"manifiesto declara {expected_platform!r}."
        )

    expected_arch = manifest.get("arch")
    if expected_arch and _normalize_arch(expected_arch) != _normalize_arch(
        platform.machine()
    ):
        raise OfflinePackError(
            f"El pack es para arquitectura {expected_arch!r} y este equipo es "
            f"{platform.machine()!r}."
        )

    tag = str(manifest.get("python_tag") or "")
    current = _current_python_tag()
    if tag != current:
        raise OfflinePackError(
            f"El pack está compilado para Python {tag or '?'} y esta app usa "
            f"{current}. Los wheels nativos de torch/onnxruntime no son "
            "compatibles entre versiones: descarga el pack de esta versión."
        )

    if str(manifest.get("backend") or "").lower() != "cuda":
        raise OfflinePackError(
            "Este pack no es de GPU (backend="
            f"{manifest.get('backend')!r}); para CPU no hace falta el pack."
        )

    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise OfflinePackError("El manifiesto no declara ninguna parte (parts).")

    seen: set[str] = set()
    total_parts = 0
    for index, part in enumerate(parts, start=1):
        if not isinstance(part, dict):
            raise OfflinePackError(f"La parte #{index} no es un objeto JSON.")
        name = str(part.get("name") or "")
        if not name or os.path.basename(name) != name:
            raise OfflinePackError(
                f"La parte #{index} tiene un nombre no válido ({name!r}): "
                "debe ser un archivo de la misma carpeta que el manifiesto."
            )
        if name in seen:
            raise OfflinePackError(f"La parte {name!r} está repetida.")
        seen.add(name)
        if not name.lower().endswith(".zip"):
            raise OfflinePackError(f"La parte {name!r} no es un ZIP.")
        size = int(part.get("bytes") or 0)
        if size <= 0:
            raise OfflinePackError(f"La parte {name!r} no declara tamaño.")
        if size >= MAX_ASSET_BYTES:
            raise OfflinePackError(
                f"La parte {name!r} mide {size} bytes y supera el límite de "
                "2 GiB por asset de GitHub Releases."
            )
        digest = str(part.get("sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise OfflinePackError(
                f"La parte {name!r} no tiene un SHA-256 válido."
            )
        total_parts += size

    installed = int(manifest.get("installed_bytes") or 0)
    if installed <= 0:
        raise OfflinePackError(
            "El manifiesto no declara installed_bytes (tamaño descomprimido)."
        )
    declared_download = int(manifest.get("download_bytes") or 0)
    if declared_download and total_parts > declared_download:
        raise OfflinePackError(
            "download_bytes del manifiesto es menor que la suma de las partes."
        )

    required = manifest.get("min_driver_branch")
    if required:
        driver = manager._nvidia_driver_branch()
        if driver is not None and driver < int(required):
            raise OfflinePackError(
                f"El pack necesita un driver NVIDIA de la rama {required} o "
                f"superior, y este equipo tiene la {driver}. Actualiza el "
                "driver o usa un pack CUDA más antiguo."
            )

    return manifest


# ---------------------------------------------------------------------------
# Verificación de partes
# ---------------------------------------------------------------------------


def _emit(callback, percent: float, message: str) -> None:
    if not callback:
        return
    try:
        callback(int(max(0, min(100, round(percent)))), message)
    except Exception:
        pass


def _check_abort(abort_event) -> None:
    if abort_event is not None and abort_event.is_set():
        raise RuntimeError("cancelled by user")


def _sha256_file(path: str, abort_event=None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            _check_abort(abort_event)
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_parts(
    manifest: dict,
    progress_callback=None,
    abort_event=None,
) -> list[str]:
    """Comprueba tamaño y SHA-256 de cada parte; devuelve sus rutas.

    Un fallo identifica la parte exacta para que el usuario vuelva a bajar
    solo ese asset. No se abre ningún ZIP aquí: primero se verifica todo y
    solo después se extrae.
    """
    base = manifest["_manifest_dir"]
    total = sum(int(p["bytes"]) for p in manifest["parts"])
    done = 0
    paths: list[str] = []
    for part in manifest["parts"]:
        name = part["name"]
        path = os.path.join(base, name)
        _check_abort(abort_event)
        if not os.path.isfile(path):
            raise OfflinePackError(
                f"Falta la parte {name!r}. Descárgala junto al manifiesto "
                f"({base})."
            )
        size = os.path.getsize(path)
        expected_size = int(part["bytes"])
        if size != expected_size:
            raise OfflinePackError(
                f"La parte {name!r} mide {size} bytes y el manifiesto declara "
                f"{expected_size}. La descarga está incompleta o es otro "
                "archivo: vuelve a descargarla."
            )
        _emit(
            progress_callback,
            2 + (done / max(1, total)) * 26,
            f"Verificando {name}...",
        )
        actual = _sha256_file(path, abort_event)
        if actual != part["sha256"]:
            raise OfflinePackError(
                f"El SHA-256 de {name!r} no coincide.\n"
                f"  esperado: {part['sha256']}\n"
                f"  obtenido: {actual}\n"
                "Vuelve a descargar esa parte."
            )
        done += expected_size
        paths.append(path)
        _emit(
            progress_callback,
            2 + (done / max(1, total)) * 26,
            f"Parte verificada: {name}",
        )
    return paths


# ---------------------------------------------------------------------------
# Extracción segura
# ---------------------------------------------------------------------------


def _member_parts(name: str) -> tuple[str, ...]:
    """Segmentos seguros de una entrada ZIP, o error.

    Rechaza rutas absolutas (POSIX y con letra de unidad), ``..`` y entradas
    vacías. Es la guarda contra un ZIP manipulado que escribiría fuera del
    staging.
    """
    normalized = (name or "").replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise OfflinePackError(f"Entrada ZIP con ruta absoluta: {name!r}")
    parts = tuple(p for p in normalized.split("/") if p not in ("", "."))
    if not parts or any(p == ".." for p in parts):
        raise OfflinePackError(f"Entrada ZIP con ruta insegura: {name!r}")
    return parts


def _extract_zip(part_path: str, destination: str, abort_event=None,
                 on_bytes=None, max_bytes: int | None = None) -> int:
    """Extrae una parte sobre ``destination``; devuelve bytes escritos.

    No usa ``ZipFile.extractall``: esa función confía en el nombre de la
    entrada. Aquí cada archivo se valida y se escribe en streaming, y los
    symlinks se rechazan (en Windows se materializarían como archivos raros).
    """
    written = 0
    with zipfile.ZipFile(part_path, "r") as archive:
        for info in archive.infolist():
            _check_abort(abort_event)
            if info.is_dir():
                continue
            parts = _member_parts(info.filename)
            if "__pycache__" in parts:
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise OfflinePackError(
                    f"La parte contiene un symlink ({info.filename!r}); el "
                    "pack debe contener solo archivos regulares."
                )
            target = os.path.join(destination, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.open(info, "r") as source, open(target, "wb") as out:
                while True:
                    _check_abort(abort_event)
                    chunk = source.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if max_bytes and written > max_bytes:
                        raise OfflinePackError(
                            f"La parte {os.path.basename(part_path)!r} contiene "
                            f"más bytes de los que declara el manifiesto "
                            f"({max_bytes}). Descarga de nuevo el pack."
                        )
                    if on_bytes:
                        on_bytes(len(chunk))
    return written


def _probe_dir(directory: str, module_name: str) -> tuple[bool, str, tuple[str, ...]]:
    """Importa ``module_name`` desde ``directory`` en un subproceso limpio.

    Se reutiliza la sonda del manager (misma comprobación de origen "caracol"
    y mismas capacidades), pero apuntando al staging en lugar del store.
    """
    python = manager.get_embed_python_path()
    if python is None:
        if getattr(sys, "frozen", False):
            raise OfflinePackError(
                "No se encontró el Python privado (embed-python) junto al "
                "ejecutable: no se puede validar el pack. Reinstala la app."
            )
        python = sys.executable
    try:
        proc = subprocess.run(
            [python, "-c", _PACK_PROBE_SOURCE, directory, module_name],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            env=manager._isolated_pip_env(),
        )
    except subprocess.TimeoutExpired:
        raise OfflinePackError(
            f"La validación de {module_name!r} agotó {int(_PROBE_TIMEOUT_S)}s. "
            "Puede ser el antivirus examinando las DLL recién extraídas."
        )
    except OSError as exc:
        raise OfflinePackError(
            f"No se pudo ejecutar la validación del pack ({python}): {exc}"
        ) from exc
    stdout = proc.stdout or ""
    caps = manager._parse_probe_caps(stdout)
    if proc.returncode == 0 and "DRAGGY-IMPORT-OK" in stdout:
        return True, "", caps
    detail = (stdout + (proc.stderr or "")).strip()
    return False, detail or f"(sin salida; exit code {proc.returncode})", caps


def _missing_critical(staging: str, manifest: dict) -> list[str]:
    missing = []
    for relative in manifest.get("critical_files") or []:
        try:
            parts = _member_parts(str(relative))
        except OfflinePackError:
            missing.append(str(relative))
            continue
        if not os.path.isfile(os.path.join(staging, *parts)):
            missing.append(str(relative))
    return missing


def _dir_size(path: str) -> int:
    try:
        return fsops.safe_dir_size(path)
    except Exception:
        return 0


def _free_bytes(path: str) -> int:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return int(shutil.disk_usage(probe or ".").free)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Instalación
# ---------------------------------------------------------------------------


def _installed_versions_match(manifest: dict) -> bool:
    """True si el store actual tiene las versiones que declara el pack.

    El marcador solo dice qué pack se importó; no prueba que siga siendo el
    mismo runtime. Si después el usuario instaló otra versión con pip y vuelve
    a importar el pack (para volver a la offline), la importación debe ocurrir
    de nuevo. Comparar las versiones reales del store lo decide sin duplicar
    estado.
    """
    versions = manifest.get("versions") or {}
    expected_torch = versions.get("torch")
    if expected_torch and manager._version_of("torch") != expected_torch:
        return False
    expected_ort = versions.get("onnxruntime") or versions.get("onnxruntime-gpu")
    if expected_ort:
        actual = manager._version_of(
            "onnxruntime", ("onnxruntime-gpu", "onnxruntime_gpu")
        )
        if actual != expected_ort:
            return False
    return True


def _is_transient_swap_error(exc: OSError) -> bool:
    """True para bloqueos de Windows que merece la pena reintentar."""
    if isinstance(exc, PermissionError):
        return True
    if getattr(exc, "winerror", None) in (5, 32, 33):
        return True
    return getattr(exc, "errno", None) in (
        errno.EACCES, errno.EBUSY, errno.EPERM
    )


def _replace_with_retry(source: str, target: str) -> None:
    """Rename con reintentos cortos para bloqueos transitorios de Windows."""
    last: OSError | None = None
    for attempt, delay in enumerate(_SWAP_RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            os.replace(source, target)
            if attempt > 1:
                logger.info(
                    "store swap: rename %s -> %s succeeded on attempt %d/%d",
                    source, target, attempt, len(_SWAP_RETRY_DELAYS),
                )
            return
        except OSError as exc:
            if not _is_transient_swap_error(exc):
                raise
            last = exc
            logger.warning(
                "store swap: rename %s -> %s failed on attempt %d/%d: %s",
                source, target, attempt, len(_SWAP_RETRY_DELAYS), exc,
            )
    if last is not None:
        raise last


def _store_swap(staging: str) -> None:
    """Sustituye el store por el staging, con vuelta atrás si falla.

    El store anterior se aparta a ``.previous`` (rename en la misma unidad, sin
    copiar GB) y solo se borra cuando el nuevo ya está en su sitio. Si el
    segundo rename falla (DLL cargadas, antivirus), se restaura el anterior:
    la app nunca queda sin runtime por un import fallido.
    """
    store = manager.RUNTIME_MODULES_DIR
    aside = manager._aside_dir()
    base = manager._install_base_dir()
    os.makedirs(base, exist_ok=True)
    if os.path.isdir(aside):
        shutil.rmtree(aside, ignore_errors=True)

    moved_old = False
    # Se aparta también un store VACÍO: en Windows ``os.replace`` no puede
    # sustituir un directorio existente (aunque esté vacío) y devuelve
    # WinError 5. El rename del directorio es igual de barato.
    if os.path.isdir(store):
        try:
            _replace_with_retry(store, aside)
            moved_old = True
        except OSError as exc:
            raise OfflinePackError(
                "No se pudo apartar el runtime actual para instalar el pack. "
                "Cierra cualquier trabajo de IA en curso y reinicia la app; si "
                "persiste, el antivirus puede estar bloqueando las DLL.\n"
                f"Detalle: {exc}"
            ) from exc

    try:
        _replace_with_retry(staging, store)
    except OSError as exc:
        if moved_old and not os.path.isdir(store):
            try:
                _replace_with_retry(aside, store)
            except OSError:
                pass
        raise OfflinePackError(
            "No se pudo mover el runtime verificado a su carpeta final; se ha "
            f"conservado el anterior.\nDetalle: {exc}"
        ) from exc

    if moved_old:
        try:
            fsops.rmtree_force(aside)
        except Exception:
            shutil.rmtree(aside, ignore_errors=True)


def import_pack(
    pack_path: str,
    progress_callback=None,
    abort_event=None,
) -> dict:
    """Importa el pack y deja ``bin/runtimes/ai_modules`` listo para usar.

    Devuelve un dict con ``imported`` (True/False), ``already_installed``,
    ``manifest``, ``versions`` y ``parts``. No instala nada si el pack ya está
    importado (mismo ``id``), así que se puede llamar en cada arranque.
    """
    if not pack_supported():
        raise OfflinePackError(
            "El pack GPU offline solo está disponible para Windows x64; esta "
            f"app corre en {sys.platform!r}. En Linux/AppImage, PyTorch y ONNX "
            "Runtime se instalan online desde Manage Dependencies."
        )

    manifest = validate_manifest(load_manifest(pack_path))
    pack_id = str(manifest.get("id") or "")

    marker = installed_pack_info()
    store = manager.RUNTIME_MODULES_DIR
    if (
        pack_id
        and marker.get("id") == pack_id
        and os.path.isfile(os.path.join(store, "torch", "__init__.py"))
        and os.path.isfile(os.path.join(store, "onnxruntime", "__init__.py"))
        and _installed_versions_match(manifest)
    ):
        return {
            "imported": False,
            "already_installed": True,
            "manifest": manifest,
            "versions": marker.get("versions") or manifest.get("versions") or {},
            "parts": marker.get("parts") or [],
        }

    _emit(progress_callback, 0, "Comprobando espacio en disco...")
    installed_bytes = int(manifest["installed_bytes"])
    # El staging convive con el store anterior hasta el swap: hacen falta los
    # dos. Tras el commit, el anterior se borra.
    needed = installed_bytes + _dir_size(store) + FREE_SPACE_MARGIN
    free = _free_bytes(manager._install_base_dir())
    if free and free < needed:
        raise OfflinePackError(
            "No hay espacio para importar el pack. Hacen falta "
            f"{needed / 1024 ** 3:.1f} GB libres en la unidad del programa y "
            f"quedan {free / 1024 ** 3:.1f} GB."
        )

    part_paths = verify_parts(manifest, progress_callback, abort_event)
    _check_abort(abort_event)

    staging = manager._staging_dir()
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    extracted = 0
    try:
        for index, (part_path, part) in enumerate(
            zip(part_paths, manifest["parts"]), start=1
        ):
            _emit(
                progress_callback,
                30,
                f"Extrayendo parte {index}/{len(part_paths)}...",
            )
            declared = int(part.get("uncompressed_bytes") or 0) or None

            def _on_bytes(count: int) -> None:
                nonlocal extracted
                extracted += count
                _emit(
                    progress_callback,
                    30 + (extracted / max(1, installed_bytes)) * 38,
                    f"Extrayendo parte {index}/{len(part_paths)}...",
                )

            written = _extract_zip(
                part_path, staging, abort_event, _on_bytes, declared
            )
            if declared and written != declared:
                raise OfflinePackError(
                    f"La parte {part['name']!r} se extrajo con {written} bytes y "
                    f"el manifiesto declara {declared}. Descarga de nuevo esa "
                    "parte."
                )

        _check_abort(abort_event)
        missing = _missing_critical(staging, manifest)
        if missing:
            raise OfflinePackError(
                "El pack no contiene archivos críticos: "
                + ", ".join(missing)
                + ". Descarga de nuevo el pack completo."
            )

        _emit(progress_callback, 70, "Validando PyTorch en el pack...")
        ok, detail, caps = _probe_dir(staging, "torch")
        if not ok:
            raise OfflinePackError(
                "PyTorch del pack no se puede importar; no se ha tocado el "
                f"runtime actual.\n\n{detail}"
            )
        if "cuda" not in caps:
            raise OfflinePackError(
                "PyTorch del pack importa pero no ve la GPU (CUDA). "
                "Comprueba que el driver NVIDIA está instalado y actualizado "
                "y que la GPU es compatible; el runtime actual sigue intacto."
            )

        _emit(progress_callback, 82, "Validando ONNX Runtime en el pack...")
        ok, detail, caps = _probe_dir(staging, "onnxruntime")
        if not ok:
            raise OfflinePackError(
                "ONNX Runtime del pack no se puede importar; no se ha tocado "
                f"el runtime actual.\n\n{detail}"
            )
        if "cuda" not in caps:
            raise OfflinePackError(
                "ONNX Runtime del pack importa pero no expone "
                "CUDAExecutionProvider. Comprueba el driver NVIDIA; el "
                "runtime actual sigue intacto."
            )

        _check_abort(abort_event)
        _emit(progress_callback, 94, "Instalando el runtime del pack...")
        _store_swap(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manager._invalidate_store_probe_cache()
    manager.inject_runtime_paths()
    manager.ensure_runtime_dlls_on_path()

    marker_payload = {
        "id": pack_id,
        "format": PACK_FORMAT,
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_tag": manifest.get("python_tag"),
        "backend": manifest.get("backend"),
        "cuda": manifest.get("cuda"),
        "versions": manifest.get("versions") or {},
        "parts": [p["name"] for p in manifest["parts"]],
        "source": manifest["_manifest_path"],
    }
    try:
        os.makedirs(manager._install_base_dir(), exist_ok=True)
        with open(_marker_path(), "w", encoding="utf-8") as fh:
            json.dump(marker_payload, fh, indent=2)
    except Exception:
        # El runtime ya está instalado; el marcador solo evita repetir la
        # importación en el próximo arranque.
        pass

    _emit(progress_callback, 100, "Pack GPU instalado")
    return {
        "imported": True,
        "already_installed": False,
        "manifest": manifest,
        "versions": manifest.get("versions") or {},
        "parts": [p["name"] for p in manifest["parts"]],
    }
