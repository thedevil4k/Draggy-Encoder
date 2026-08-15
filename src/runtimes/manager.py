"""Gestión de runtimes de IA descargables.

Los runtimes (PyTorch, ONNX Runtime y futuros motores) se instalan en
``bin/runtimes/ai_modules`` usando ``pip install --target``. Esto mantiene
la aplicación autocontenida y evita tocar el entorno Python del usuario.

La app es autocontenida ("caracol"): SOLO cuenta el runtime descargado por
la propia app en ``bin/runtimes/ai_modules``. Un torch/onnxruntime presente
en el entorno del sistema (venv, Python del sistema, otros programas) se
detecta únicamente como diagnóstico, pero NO satisface la dependencia ni se
usa para inferencia.

Para conseguirlo, ``inject_runtime_paths()`` debe llamarse lo antes posible
en el arranque de la aplicación (main.py) y también antes de importar
``torch`` u ``onnxruntime`` desde src.ai.inference.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from src.config import ensure_runtime_dirs, get_bin_dir


RUNTIME_MODULES_DIR = os.path.join(get_bin_dir(), "runtimes", "ai_modules")

# Variables que pueden hacer que pip vea otro intérprete, otro entorno
# virtual o una configuración global del usuario. El proceso de instalación
# debe ser reproducible y no heredar ninguno de esos atajos.
_PIP_CONTAMINATION_VARS = (
    "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PIP_TARGET", "PIP_PREFIX",
    "PIP_USER", "PIP_CONFIG_FILE", "PIP_FIND_LINKS", "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_NO_INDEX",
)


def _isolated_pip_env() -> dict[str, str]:
    """Construye el entorno del pip privado sin heredar instalaciones ajenas.

    El PATH no se elimina: en Windows permite a los wheels encontrar las DLL
    del driver gráfico del sistema. Lo que se aísla es la resolución de
    Python/pip: no se acepta PYTHONPATH, site de usuario, virtualenv,
    configuración de pip, índices adicionales ni caché global.
    """
    env = os.environ.copy()
    for key in _PIP_CONTAMINATION_VARS:
        env.pop(key, None)
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        # Impide que pip intente instalar en el perfil del usuario si alguna
        # dependencia cambia el destino internamente.
        "PIP_USER": "0",
    })
    return env


def _pip_command_prefix() -> list[str]:
    """Invocación absoluta del pip del runtime privado."""
    return [_pip_python(), "-I", "-m", "pip"]


# Tag de versión de CPython en los wheels (ej. "cp314-cp314-win_amd64").
# Captura el número "majorminor" completo (314, 311, ...).
_WHEEL_CP_RE = re.compile(r"^cp(\d+)-")


# ---------------------------------------------------------------------------
# Path injection
# ---------------------------------------------------------------------------


def get_runtime_modules_dir() -> str:
    return RUNTIME_MODULES_DIR


def get_embed_python_path() -> str | None:
    """Return the private Python used to install AI runtimes in a bundle.

    PyInstaller's executable cannot run ``-m pip`` itself. Windows bundles
    carry ``embed-python/python.exe``; Linux bundles carry a relocatable
    ``runtime-python/bin/python3.13`` tree. Source checkouts use the current
    interpreter directly and therefore return None here.
    """
    if not getattr(sys, "frozen", False):
        return None
    root = os.path.dirname(sys.executable)
    if sys.platform == "win32":
        candidates = (
            os.path.join(get_bin_dir(), "embed-python", "python.exe"),
            os.path.join(root, "embed-python", "python.exe"),
        )
    else:
        candidates = (
            os.path.join(root, "runtime-python", "bin", "python3.13"),
            os.path.join(root, "runtime-python", "bin", "python3"),
            os.path.join(get_bin_dir(), "runtime-python", "bin", "python3.13"),
        )
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _pip_python() -> str:
    """Intérprete que ejecuta los comandos ``pip`` del store de runtimes.

    Frozen: usa el Python embebido del instalador (autocontenido). Si no
    está, falla con un mensaje claro: sin él, la app no puede instalar
    torch/onnxruntime y la inferencia IA quedaría deshabilitada. En modo
    fuente usa el intérprete actual.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    embed = get_embed_python_path()
    if embed is not None:
        return embed
    raise RuntimeError(
        "No se encontró el Python privado para instalar runtimes junto al "
        "ejecutable. Reinstala la aplicación: los runtimes de IA no se "
        "pueden instalar sin él, así que la inferencia IA quedará "
        "deshabilitada hasta entonces."
    )


def _wheel_tag_matches(tag: str) -> bool:
    """True si un tag de wheel (ej. ``cp314-cp314-win_amd64``) es compatible
    con el intérprete en ejecución.

    Los wheels con extensiones compiladas son específicos por versión de
    Python (cp311 vs cp314): inyectar un store compilado para otra versión
    rompe el arranque (ej. numpy cp314 importado desde Python 3.11). Los
    wheels abi3 (``cp311-abi3``) son compatibles hacia delante.
    """
    m = _WHEEL_CP_RE.match(tag)
    if not m:
        return True  # pure-python (py3-none-any) o tag desconocido
    tag_num = int(m.group(1))
    cur_num = int(f"{sys.version_info.major}{sys.version_info.minor}")
    if "abi3" in tag:
        return cur_num >= tag_num
    return cur_num == tag_num


def store_is_compatible_with_python() -> bool:
    """True si los wheels compilados del store local son compatibles con el
    intérprete actual.

    Lee los tags de los ``*.dist-info/WHEEL`` del store. Si algún wheel
    compilado es para otra versión de Python (por ejemplo el store se
    instaló con ``py -3.14`` pero la app corre con 3.11), devuelve False:
    ``inject_runtime_paths`` entonces NO mete el store en ``sys.path`` y la
    app reporta el runtime como missing/error aunque el sistema tenga una
    copia.
    """
    if not os.path.isdir(RUNTIME_MODULES_DIR):
        return True
    try:
        entries = os.listdir(RUNTIME_MODULES_DIR)
    except Exception:
        return True
    checked = False
    for entry in entries:
        if not entry.endswith(".dist-info"):
            continue
        wheel_path = os.path.join(RUNTIME_MODULES_DIR, entry, "WHEEL")
        if not os.path.isfile(wheel_path):
            continue
        try:
            with open(wheel_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Tag:"):
                        for tag in line[4:].split():
                            checked = True
                            if not _wheel_tag_matches(tag):
                                return False
        except Exception:
            continue
    return True


def _purge_foreign_version_wheels() -> int:
    """Elimina del store los wheels compilados para OTRA versión de Python.

    ``pip install --target --upgrade`` deja a veces ``*.dist-info`` huérfanos
    de la instalación previa (p.ej. ``numpy-2.5.1.dist-info`` cp314 cuando se
    reinstala con Python 3.11: pip instala el wheel cp311 pero no borra el
    dist-info viejo). Esos huérfanos hacen que ``store_is_compatible_with_python()``
    devuelva False para siempre y el runtime nunca se considere instalado,
    aunque la instalación real haya sido correcta. Devuelve cuántos paquetes
    purgó (0 si no hay nada que purgar).

    Los directorios de paquete (``numpy/``, ``numpy.libs``) se borran SOLO si
    ningún otro dist-info compatible del mismo paquete los comparte: purgar
    el dist-info cp314 de numpy no debe tumbar el ``numpy/`` del cp311 recién
    instalado.
    """
    if not os.path.isdir(RUNTIME_MODULES_DIR):
        return 0
    try:
        entries = os.listdir(RUNTIME_MODULES_DIR)
    except Exception:
        return 0
    dist_infos = [e for e in entries if e.endswith(".dist-info")]
    foreign: list[str] = []
    for entry in dist_infos:
        if _dist_info_is_foreign(entry):
            foreign.append(entry)
    if not foreign:
        return 0
    # Nombres de paquete con dist-info NO extranjero en el store: sus dirs
    # (numpy/, numpy.libs) pueden ser compartidos -> no borrarlos.
    safe_names = set()
    for entry in dist_infos:
        if entry not in foreign:
            base = entry[: -len(".dist-info")]
            safe_names.add(base.split("-", 1)[0])
    purged = 0
    for entry in foreign:
        dist_dir = os.path.join(RUNTIME_MODULES_DIR, entry)
        base = entry[: -len(".dist-info")]
        name = base.split("-", 1)[0]
        if name in safe_names:
            # El paquete actual (compatible) usa los mismos dirs: solo se
            # retira el dist-info extranjero.
            _rmtree_quiet(dist_dir)
            purged += 1
            continue
        top_level: list[str] = []
        top_path = os.path.join(dist_dir, "top_level.txt")
        if os.path.isfile(top_path):
            try:
                with open(top_path, "r", encoding="utf-8", errors="replace") as f:
                    top_level = [ln.strip() for ln in f if ln.strip()]
            except Exception:
                top_level = []
        if not top_level:
            top_level = [name]
        for tl in top_level:
            _rmtree_quiet(os.path.join(RUNTIME_MODULES_DIR, tl))
            # Datos compilados hermanos (numpy.libs, ...).
            _rmtree_quiet(os.path.join(RUNTIME_MODULES_DIR, tl + ".libs"))
        _rmtree_quiet(dist_dir)
        purged += 1
    return purged


def _dist_info_is_foreign(dist_info_dirname: str) -> bool:
    """True si un directorio ``*.dist-info`` declara un wheel compilado para
    otra versión de Python (WHEEL con Tag cpXY != intérprete actual)."""
    dist_dir = os.path.join(RUNTIME_MODULES_DIR, dist_info_dirname)
    wheel_path = os.path.join(dist_dir, "WHEEL")
    if not os.path.isfile(wheel_path):
        return False
    try:
        with open(wheel_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("Tag:"):
                    continue
                for tag in line[4:].split():
                    if not _wheel_tag_matches(tag):
                        return True
    except Exception:
        pass
    return False


def _rmtree_quiet(path: str) -> None:
    """Borra un dir/archivo sin propagar errores (archivos bloqueados en
    Windows se quedan y el guard de versión los vuelve a excluir)."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def inject_runtime_paths() -> None:
    """Inserta el directorio de módulos locales al principio de sys.path.

    Autocuración: si el store contiene wheels compilados para OTRA versión de
    Python (p.ej. instalado con ``py -3.14`` y la app corre con 3.11), se
    purgan automáticamente para que la instalación con el intérprete actual
    funcione de verdad (pip deja dist-infos huérfanos que mantendrían el
    store 'incompatible' para siempre). Tras purgar, el store queda vacío de
    wheels ajenos: la app reporta MISSING y el botón Install funciona.

    Sin esta guarda, un store instalado con otro intérprete (ej. cp314 vs
    cp311) reventaba el arranque con errores de import de las C-extensions
    (numpy, torch, onnxruntime).
    """
    try:
        os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    except Exception:
        pass
    if not os.path.isdir(RUNTIME_MODULES_DIR):
        return
    _purge_foreign_version_wheels()
    if not store_is_compatible_with_python():
        return
    if RUNTIME_MODULES_DIR not in sys.path:
        sys.path.insert(0, RUNTIME_MODULES_DIR)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _store_abspath() -> str:
    """Ruta normalizada del store local (normcase para comparaciones en
    Windows, donde el case de las rutas no es fiable)."""
    return os.path.normcase(os.path.abspath(RUNTIME_MODULES_DIR))


def _is_path_in_store(path: str | None) -> bool:
    """True solo para una ruta dentro del store, no para un prefijo parecido.

    La frontera debe distinguir ``.../ai_modules_extra`` de
    ``.../ai_modules``; un ``startswith`` simple permitiría confundir ambos.
    """
    if not path:
        return False
    normalized = os.path.normcase(os.path.abspath(path))
    store = _store_abspath()
    return normalized == store or normalized.startswith(store + os.sep)


def _can_import(module_name: str) -> bool:
    """True si el módulo importa de verdad DESDE el store local.

    La app es autocontenida ("caracol"): solo el runtime descargado por la
    propia app en ``bin/runtimes/ai_modules`` satisface la dependencia. Un
    torch/onnxruntime del entorno del sistema (venv, Python del sistema,
    otros programas) NO cuenta, aunque se importe sin problemas.

    Un directorio ``torch/`` vacío o a medio descargar dentro del store
    local se importa como namespace package (``spec.origin is None``): daría
    un falso INSTALLED. Solo cuenta si existe un origin real (paquete
    completo) DENTRO del store.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return False
        if not _is_path_in_store(spec.origin):
            return False
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def import_local_runtime(module_name: str):
    """Importa un runtime únicamente desde ``ai_modules``.

    Es la frontera que deben usar los módulos de IA: ``import torch`` a
    secas puede resolver un paquete instalado en un venv o en el Python del
    usuario. Esta función comprueba origen e importación real y rechaza el
    módulo aunque exista fuera del store.
    """
    inject_runtime_paths()
    if not _can_import(module_name):
        raise ImportError(
            f"{module_name} is not installed in the application's private "
            "AI runtime store"
        )
    module = importlib.import_module(module_name)
    origin = getattr(module, "__file__", None)
    if not _is_path_in_store(origin):
        raise ImportError(
            f"{module_name} resolved outside the application's private store"
        )
    return module


def _can_import_system(module_name: str) -> bool:
    """True si el módulo existe en el entorno del sistema pero NO en el store.

    Solo diagnóstico (para el mensaje de la UI): la app no lo usa, pero el
    usuario debe saber que su sistema tiene una copia. No importa el módulo:
    mirar el spec es barato y evita cargar torch del sistema.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return False
        return not os.path.normcase(spec.origin).startswith(_store_abspath())
    except Exception:
        return False


def _has_nvidia_gpu() -> bool:
    """Heurística rápida: ¿existe nvidia-smi en PATH?"""
    return shutil.which("nvidia-smi") is not None


def _python_short() -> str:
    """Versión corta de Python, e.g. '3.11'."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _is_python_version_supported() -> bool:
    """True si es una versión de Python con wheels oficiales de PyTorch/ONNX.

    PyTorch y ONNX Runtime suelen tardar en publicar wheels para las
    versiones más nuevas de Python. A día de hoy (2026) el soporte
    estable es 3.9–3.14.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    if major != 3:
        return False
    return 9 <= minor <= 14


def get_python_support_info() -> dict:
    """Devuelve información de compatibilidad para mostrar en la UI."""
    supported = _is_python_version_supported()
    short = _python_short()
    if supported:
        return {
            "supported": True,
            "short": short,
            "message": f"Python {short} is supported. Click Install to download AI runtimes.",
        }
    return {
        "supported": False,
        "short": short,
        "message": (
            f"Python {short} is NOT supported by official PyTorch/ONNX wheels yet. "
            "Please use Python 3.11–3.14, or build the PyInstaller bundle with one of those versions. "
            "AI upscaling/interpolation/colorize will be disabled."
        ),
    }


# ---------------------------------------------------------------------------
# Package specs
# ---------------------------------------------------------------------------


def _default_gpu_variant() -> str:
    """Variant CUDA por defecto según la versión de Python en uso.

    PyTorch publica wheels CUDA por versión de Python:
    - cu121: wheels para cp39–cp312 (Windows) / cp39–cp313 (Linux)
    - cu126: wheels para cp39–cp314 (Windows + Linux) desde torch 2.6.0
    - cu130: wheels para cp310–cp314 (Windows + Linux) desde torch 2.9.0

    En Python 3.13+, cu121 NO tiene wheels Windows (solo Linux cp313),
    así que usamos cu126 que tiene soporte completo cp313-win_amd64.
    """
    minor = sys.version_info.minor
    if minor >= 14:
        return "cuda130"
    if minor >= 13:
        return "cuda126"
    return "cuda121"


def _resolve_variant(variant: str) -> str:
    """Resuelve 'auto' a cpu/cuda121/cuda118/cuda126/cuda130."""
    if variant == "auto":
        if _has_nvidia_gpu():
            return _default_gpu_variant()
        return "cpu"
    if variant in ("cpu", "cuda118", "cuda11.8", "cuda121", "cuda126", "cuda130"):
        if variant == "cuda11.8":
            return "cuda118"
        return variant
    return "cpu"


def _build_pytorch_command(variant: str) -> list[str]:
    """Construye el comando pip para instalar torch/torchvision/torchaudio."""
    variant = _resolve_variant(variant)
    if variant == "cpu":
        index_url = "https://download.pytorch.org/whl/cpu"
    elif variant == "cuda118":
        index_url = "https://download.pytorch.org/whl/cu118"
    elif variant == "cuda126":
        index_url = "https://download.pytorch.org/whl/cu126"
    elif variant == "cuda130":
        index_url = "https://download.pytorch.org/whl/cu130"
    else:  # cuda121
        index_url = "https://download.pytorch.org/whl/cu121"

    return [
        *_pip_command_prefix(), "install",
        "--isolated", "--no-cache-dir", "--disable-pip-version-check",
        "--ignore-installed", "--only-binary=:all:",
        "--target", RUNTIME_MODULES_DIR,
        "--no-warn-script-location",
        "--upgrade",
        "--index-url", index_url,
        "torch", "torchvision", "torchaudio", "onnx", "onnxscript",
    ]


def _build_onnx_command(variant: str) -> list[str]:
    """Construye el comando pip para instalar onnxruntime."""
    variant = _resolve_variant(variant)
    pkg = "onnxruntime-gpu" if variant.startswith("cuda") else "onnxruntime"
    return [
        *_pip_command_prefix(), "install",
        "--isolated", "--no-cache-dir", "--disable-pip-version-check",
        "--ignore-installed", "--only-binary=:all:",
        "--target", RUNTIME_MODULES_DIR,
        "--no-warn-script-location",
        "--upgrade",
        pkg,
    ]


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Connectivity pre-flight (antes de descargar runtimes de IA)
# ---------------------------------------------------------------------------

# Hosts que pip necesita para instalar los runtimes. La sonda usa HTTPS y el
# mismo certificado CA que pip (certifi vendido) para medir lo que pip vería
# de verdad, no el store CA (a menudo ausente) del Python embebido.
_CONNECTIVITY_HOSTS = (
    ("download.pytorch.org", "https://download.pytorch.org/whl/cpu/torch/"),
    ("PyPI", "https://pypi.org/simple/onnxruntime/"),
)

_PROBE_TIMEOUT_S = 10

# Caché corta de la sonda de conectividad: reinstalaciones seguidas (p. ej.
# instalar PyTorch y luego ONNX Runtime como dos filas separadas) no deben
# volver a sondear la red cada vez. Solo se cachea un resultado NO fatal: si
# hubo un bloqueo inequívoco (DNS/firewall/SSL/proxy) se re-sondea en el
# siguiente intento, para no penalizar un arreglo rápido de la red.
_PROBE_CACHE_TTL_S = 5 * 60  # 5 minutos
_probe_cache: dict = {"ts": 0.0, "report": None}
_probe_cache_lock = threading.Lock()


def clear_connectivity_cache() -> None:
    """Invalida la caché de la sonda de conectividad (tests y cambios de red)."""
    with _probe_cache_lock:
        _probe_cache["ts"] = 0.0
        _probe_cache["report"] = None


_PROBE_SCRIPT = r'''
import json, os, socket, ssl, urllib.request, urllib.error

hosts = json.loads(os.environ.get("DRAGGY_PROBE_HOSTS", "[]"))
timeout = float(os.environ.get("DRAGGY_PROBE_TIMEOUT", "10"))
proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "")

def _ca_context():
    # Mirror pip: usa el certifi vendido por pip, que es el CA que pip
    # realmente confía; el Python embebido no trae su propio CA bundle.
    try:
        import pip._vendor.certifi as _c
        return ssl.create_default_context(cafile=_c.where())
    except Exception:
        return ssl.create_default_context()

def _classify(exc):
    if isinstance(exc, ssl.SSLCertVerificationError):
        return "ssl"
    if isinstance(exc, socket.gaierror):
        return "dns"
    if isinstance(exc, ConnectionRefusedError):
        return "refused"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, ssl.SSLError):
        return "ssl"
    return "unreachable"

out = {}
for name, url in hosts:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "DraggyEncoder connectivity-probe"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ca_context()) as resp:
            out[name] = {"status": "ok", "code": resp.status}
    except urllib.error.HTTPError as e:
        out[name] = {"status": "proxy" if e.code == 407 else ("http_%d" % e.code),
                     "code": e.code}
    except urllib.error.URLError as e:
        out[name] = {"status": _classify(e.reason), "detail": str(e.reason)}
    except Exception as e:
        out[name] = {"status": _classify(e), "detail": repr(e)}

out["__proxy__"] = proxy
print(json.dumps(out))
'''

_HOST_STATUS_ES = {
    "ok": "accesible",
    "dns": "no se pudo resolver el dominio (fallo de DNS)",
    "refused": "conexión rechazada — posible firewall o antivirus bloqueando la salida",
    "timeout": "tiempo de espera agotado — posible firewall o red muy lenta",
    "ssl": "certificado SSL no verificado — posible proxy/MITM o reloj del sistema incorrecto",
    "proxy": "el proxy pidió autenticación (HTTP 407)",
    "unreachable": "no alcanzable",
    "unknown": "error desconocido",
}


def _friendly_host_line(name: str, res: dict) -> str:
    status = res.get("status", "unknown")
    if status.startswith("http_"):
        msg = f"el servidor respondió HTTP {status[len('http_'):]}"
    else:
        msg = _HOST_STATUS_ES.get(status, "no alcanzable")
    detail = res.get("detail")
    line = f"  - {name}: {msg}"
    if detail and status != "ok":
        line += f" ({detail})"
    return line


def run_connectivity_probe(timeout: float | None = None,
                           use_cache: bool = True) -> dict:
    """Comprueba si el Python que ejecutará pip puede llegar a los servidores
    de descarga de los runtimes de IA (download.pytorch.org y PyPI).

    Ejecuta una sonda HTTPS con el MISMO intérprete y entorno que usará pip
    (el Python embebido en el instalador) y clasifica el fallo: DNS, firewall
    (rechazo/timeout), certificados SSL rotos o proxy con autenticación.

    Devuelve un dict con ``ok``, ``fatal``, ``summary`` (texto legible en
    español), ``hosts``, ``proxy`` y ``error``. ``fatal=True`` significa un
    bloqueo inequívoco de la red: la instalación debe abortarse de inmediato
    con ``summary`` en lugar de esperar el timeout de pip.
    """
    timeout = _PROBE_TIMEOUT_S if timeout is None else timeout

    if use_cache:
        with _probe_cache_lock:
            cached = _probe_cache.get("report")
            if cached is not None and time.monotonic() - _probe_cache["ts"] < _PROBE_CACHE_TTL_S:
                return dict(cached)

    report: dict = {
        "ok": False,
        "fatal": False,
        "summary": "",
        "hosts": {},
        "proxy": "",
        "error": None,
    }

    try:
        python = _pip_python()
    except RuntimeError as exc:
        report["fatal"] = True
        report["error"] = str(exc)
        report["summary"] = str(exc)
        return report

    env = _isolated_pip_env()
    env["DRAGGY_PROBE_HOSTS"] = json.dumps(
        [[name, url] for name, url in _CONNECTIVITY_HOSTS]
    )
    env["DRAGGY_PROBE_TIMEOUT"] = str(timeout)
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            [python, "-I", "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, env=env,
            encoding="utf-8", errors="replace",
            timeout=timeout * len(_CONNECTIVITY_HOSTS) + 20,
        )
    except subprocess.TimeoutExpired:
        report["fatal"] = True
        report["error"] = "La sonda de red no respondió (timeout)."
        report["summary"] = report["error"]
        return report
    except Exception as exc:
        report["fatal"] = True
        report["error"] = f"No se pudo ejecutar la sonda de red: {exc}"
        report["summary"] = report["error"]
        return report

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        report["fatal"] = True
        report["error"] = (
            "La sonda de red no pudo ejecutarse en el Python embebido"
            + (f": {stderr}" if stderr else "")
        )
        report["summary"] = report["error"]
        return report

    try:
        raw = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        report["fatal"] = True
        report["error"] = "La sonda de red devolvió una respuesta ilegible."
        report["summary"] = report["error"]
        return report

    report["proxy"] = raw.pop("__proxy__", "") or ""
    report["hosts"] = raw

    statuses = [h.get("status", "unknown") for h in raw.values()]
    report["ok"] = "ok" in statuses

    # Bloqueo inequívoco: ningún host alcanzable y al menos uno falla de forma
    # definitiva (DNS, rechazo, SSL o proxy). Un simple timeout se considera
    # transitorio y no aborta: pip tiene sus propios reintentos.
    definitive = any(s in ("dns", "refused", "ssl", "proxy") for s in statuses)
    report["fatal"] = (not report["ok"]) and definitive

    if report["fatal"]:
        lines = [
            "No se pudo contactar con los servidores de descarga de los "
            "runtimes de IA (posible firewall, proxy o certificados SSL):",
        ]
    elif report["ok"]:
        lines = ["Red verificada: los servidores de descarga son accesibles."]
    else:
        lines = [
            "La sonda de red no pudo confirmar todos los servidores de "
            "descarga (se continúa con la instalación):",
        ]
    for name, _url in _CONNECTIVITY_HOSTS:
        res = raw.get(name)
        if res is not None:
            lines.append(_friendly_host_line(name, res))
    if report["proxy"]:
        lines.append(f"  - Proxy detectado: {report['proxy']}")
    report["summary"] = "\n".join(lines)

    if use_cache and not report["fatal"]:
        with _probe_cache_lock:
            _probe_cache["ts"] = time.monotonic()
            _probe_cache["report"] = dict(report)
    return report


def _preflight_connectivity() -> dict:
    """Aborta rápido con un mensaje claro si la red bloquea la descarga."""
    report = run_connectivity_probe()
    if report["fatal"]:
        raise RuntimeError(report["summary"])
    return report


# ---------------------------------------------------------------------------
# Disk space pre-flight (antes de descargar runtimes de IA)
# ---------------------------------------------------------------------------


def _required_disk_bytes(kind: str, variant: str) -> tuple[int, int]:
    """Estimación conservadora de disco en bytes: ``(instalado, descarga)``.

    ``kind`` es "pytorch", "onnxruntime" o "all"; ``variant`` es
    cpu/cudaXXX. Los wheels CUDA de torch pesan ~2.5 GB instalados (más la
    descarga comprimida), frente a ~400 MB en CPU; ONNX Runtime es mucho más
    ligero. Se incluye margen para torchvision/torchaudio/onnx y futuros
    updates, de modo que el aviso dispare ANTES de que pip falle a mitad de
    una descarga de gigabytes.
    """
    is_cuda = variant != "cpu"
    if kind == "pytorch":
        if is_cuda:
            return (4_500_000_000, 3_000_000_000)
        return (400_000_000, 250_000_000)
    if kind == "onnxruntime":
        if is_cuda:
            return (400_000_000, 200_000_000)
        return (150_000_000, 60_000_000)
    # "all": suma de ambos paquetes.
    pt = _required_disk_bytes("pytorch", variant)
    onx = _required_disk_bytes("onnxruntime", variant)
    return (pt[0] + onx[0], pt[1] + onx[1])


def _existing_ancestor(path: str) -> str:
    """Sube hasta el ancestro que existe (``shutil.disk_usage`` lo exige)."""
    candidate = os.path.abspath(path)
    while candidate and not os.path.isdir(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return candidate or os.path.abspath(os.sep)


def _free_bytes(path: str) -> int:
    """Bytes libres en la unidad que contiene ``path``; -1 si no se mide."""
    try:
        return shutil.disk_usage(_existing_ancestor(path)).free
    except OSError:
        return -1


def _device_id(path: str) -> str:
    """Identificador del dispositivo de ``path`` ('' si no se mide)."""
    try:
        return str(os.stat(_existing_ancestor(path)).st_dev)
    except OSError:
        return ""


def _human_size(num: float) -> str:
    gb = 1024 ** 3
    mb = 1024 ** 2
    if num >= gb:
        return f"{num / gb:.1f} GB"
    if num >= mb:
        return f"{num / mb:.0f} MB"
    return f"{int(num)} bytes"


def _check_disk_space(kind: str, variant: str) -> None:
    """Aborta ANTES de descargar si no hay espacio libre suficiente.

    El umbral depende de la variante (CPU/CUDA). Revisa la unidad del store
    de instalación (``RUNTIME_MODULES_DIR``) y la de la carpeta temporal de
    descargas (``tempfile.gettempdir()``). Si ambas son la misma unidad (el
    caso habitual en Windows con un solo disco), exige la SUMA de descarga +
    instalación, porque pip pica ambas a la vez en el pico.

    Si no se puede medir el espacio (``-1``), no se aborta: pip ya reportará
    el error de disco si de verdad falta.
    """
    installed, download = _required_disk_bytes(kind, variant)

    target_dir = RUNTIME_MODULES_DIR
    tmp_dir = tempfile.gettempdir()

    target_free = _free_bytes(target_dir)
    tmp_free = _free_bytes(tmp_dir)
    target_dev = _device_id(target_dir)
    tmp_dev = _device_id(tmp_dir)
    same_device = bool(target_dev) and target_dev == tmp_dev

    problems: list[str] = []
    if same_device:
        free = target_free if target_free >= 0 else tmp_free
        need = installed + download
        if free >= 0 and free < need:
            problems.append(
                f"  - Espacio libre en {target_dir}: {_human_size(free)} "
                f"(se necesitan ~{_human_size(need)} para descarga + instalación)."
            )
    else:
        if 0 <= target_free < installed:
            problems.append(
                f"  - Carpeta de instalación ({target_dir}): "
                f"{_human_size(target_free)} libres, "
                f"se necesitan ~{_human_size(installed)}."
            )
        if 0 <= tmp_free < download:
            problems.append(
                f"  - Carpeta temporal de descargas ({tmp_dir}): "
                f"{_human_size(tmp_free)} libres, "
                f"se necesitan ~{_human_size(download)}."
            )

    if not problems:
        return

    label = "CUDA" if variant != "cpu" else "CPU"
    raise RuntimeError(
        "No hay espacio libre en disco para instalar los runtimes de IA "
        f"(variante {label}):\n\n"
        + "\n".join(problems)
        + "\n\nLibera espacio en disco y vuelve a intentarlo."
    )


# Expresiones para traducir la salida cruda de pip a fases amigables para el
# usuario. El usuario final NO debe ver líneas de pip ni términos técnicos.
_DL_FRACTION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*/\s*([\d.]+)\s*(MB|GB)", re.IGNORECASE
)


def _download_fraction(line: str) -> float | None:
    """Fracción (0..1) de descarga si la línea de pip lleva progreso de
    tamaño (``2.4/2.4 GB``); None si no es una línea de progreso."""
    m = _DL_FRACTION_RE.search(line or "")
    if not m:
        return None
    cur = float(m.group(1))
    total = float(m.group(2))
    if m.group(3).upper() == "GB":
        cur *= 1024
        total *= 1024
    if total <= 0:
        return None
    return max(0.0, min(1.0, cur / total))


def _friendly_phase(line: str) -> str:
    """Fase humana para una línea de pip (vacío si no se reconoce)."""
    if "Collecting" in line:
        return "Preparando la descarga..."
    if "Downloading" in line:
        return "Descargando..."
    if "Installing collected" in line or line.startswith("Installing "):
        return "Instalando..."
    if "Successfully installed" in line:
        return "Completando..."
    return ""


# Presupuesto de pared por defecto para una instalación vía pip.
# Un torch CUDA completo puede superar los 2 GB en una conexión lenta, así
# que damos margen; el objetivo no es cortar descargas legítimas, sino
# garantizar que un pip colgado no deje "Install" girando para siempre.
_DEFAULT_PIP_TIMEOUT_S = 2 * 3600


def _stop_process(proc) -> None:
    """Termina un subproceso con gracia y escala a kill si no reacciona."""
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass


# Categorías de fallo de pip traducidas a causas legibles. La sonda de
# conectividad cubre "red caída" ANTES de instalar; aquí se clasifica lo que
# pip reporta cuando la red funciona pero la instalación falla por otro
# motivo (espacio en disco, SSL, proxy, versión incompatible...).
_PIP_FAILURE_TITLES = {
    "disk_space": "No hay espacio libre en disco para instalar los runtimes",
    "ssl": "Fallo de certificados SSL durante la descarga (posible proxy/MITM o reloj del sistema incorrecto)",
    "network": "Problema de red durante la descarga (firewall, caída de conexión o DNS)",
    "proxy": "El proxy bloqueó la descarga (posible autenticación requerida)",
    "no_version": "No se encontró un wheel compatible para tu versión de Python/plataforma (o el índice no es accesible)",
    "pip": "pip falló al instalar los runtimes",
}


def _classify_pip_failure(text: str) -> str:
    """Clasifica la salida cruda de pip en una categoría legible."""
    low = (text or "").lower()
    if any(s in low for s in (
            "no space left", "errno 28", "not enough space",
            "disk full", "enospc", "no queda espacio",
    )):
        return "disk_space"
    if any(s in low for s in (
            "certificate_verify_failed", "certificate verify failed",
            "sslcertverificationerror", "sslerror", "ssl:", "ssl3_",
            "ssl connection", "tls/ssl", "tlsv1", "handshake failure",
            "unable to get local issuer",
    )):
        return "ssl"
    if any(s in low for s in (
            "407 ", "proxyerror", "proxy authentication",
            "tunnel connection failed", "cannot connect to proxy",
    )):
        return "proxy"
    if any(s in low for s in (
            "connection refused", "timed out", "timeout", "getaddrinfo",
            "name or service not known", "temporary failure in name resolution",
            "connection reset", "failed to establish a new connection",
            "network is unreachable", "no route to host", "connection aborted",
    )):
        return "network"
    if any(s in low for s in (
            "no matching distribution", "could not find a version",
            "no versions available", "not a supported wheel",
    )):
        return "no_version"
    return "pip"


def _run_pip(cmd: list[str], progress_callback, start_pct: int, end_pct: int,
             message: str, abort_event=None, timeout: float | None = None):
    """Ejecuta un comando pip y emite progreso interpolado.

    La UI recibe SOLO fases amigables y porcentajes: la salida cruda de pip
    se queda en la cola de diagnóstico (visible únicamente si la instalación
    falla, en el detalle técnico del diálogo de error).

    ``abort_event`` (``threading.Event`` opcional) detiene la instalación en
    cuanto se setea: matamos el proceso pip y lanzamos ``RuntimeError
    ("cancelled by user")`` para que el caller distinga cancelación de fallo.
    ``timeout`` es el presupuesto total de pared en segundos (default
    ``_DEFAULT_PIP_TIMEOUT_S``): pasado ese límite matamos pip y lanzamos
    ``RuntimeError``, de modo que Instalar/Actualizar nunca cuelgue
    indefinidamente aunque la red esté medio muerta. Se lee en un hilo
    dedicado para poder observar el abort/vencimiento mientras ``readline``
    bloquea.
    """
    deadline = time.monotonic() + (
        timeout if timeout is not None else _DEFAULT_PIP_TIMEOUT_S
    )
    if progress_callback:
        progress_callback(start_pct, message)

    env = _isolated_pip_env()

    # Verificar que pip esté disponible en el intérprete privado, no en el
    # Python/venv desde el que se lanzó la aplicación.
    try:
        test_proc = subprocess.run(
            [*_pip_command_prefix(), "--version"],
            capture_output=True, text=True, env=env, timeout=30,
        )
        if test_proc.returncode != 0:
            raise RuntimeError(
                "pip no se encuentra disponible. "
                "Asegúrate de que Python se instaló con pip."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "pip no se encuentra disponible. "
            "Asegúrate de que Python se instaló con pip."
        ) from exc

    proc = None
    last_pct = start_pct
    tail: list[str] = []
    lines: "queue.Queue[str | None]" = queue.Queue()

    def _reader():
        try:
            assert proc is not None and proc.stdout is not None
            for line in proc.stdout:
                lines.put(line)
        except Exception:
            pass
        finally:
            lines.put(None)  # sentinel EOF

    cancelled = False
    timed_out = False
    ret = -1
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        threading.Thread(target=_reader, daemon=True).start()
        while True:
            if abort_event is not None and abort_event.is_set():
                cancelled = True
                _stop_process(proc)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _stop_process(proc)
                break
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break  # EOF
            line = line.strip()
            if not line:
                continue
            # Guardar cola del output para el diagnóstico de errores
            tail.append(line)
            if len(tail) > 15:
                tail.pop(0)
            if not progress_callback or last_pct >= end_pct:
                continue
            # Progreso real cuando pip reporta tamaño descargado
            # (2.4/2.4 GB), para que la barra se mueva de forma honesta
            # durante descargas largas. Nunca retroceder.
            frac = _download_fraction(line)
            if frac is not None:
                pct = start_pct + (end_pct - start_pct) * frac
            else:
                pct = last_pct + 1
            pct = min(end_pct, max(pct, last_pct))
            last_pct = pct
            phase = _friendly_phase(line) or message
            progress_callback(int(pct), phase)
        ret = proc.wait()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se pudo ejecutar pip. Verifica que Python esté en el PATH."
        ) from exc
    finally:
        # Red de seguridad: un pip a medio matar no debe quedar huérfano.
        if proc is not None:
            try:
                still_running = proc.poll() is None
            except Exception:
                still_running = True
            if still_running:
                try:
                    proc.kill()
                except Exception:
                    pass

    if cancelled:
        raise RuntimeError("cancelled by user")
    if timed_out:
        raise RuntimeError(
            f"Installation timed out after {int(timeout or _DEFAULT_PIP_TIMEOUT_S)}s "
            "(the download stalled or the network is too slow)."
        )
    if ret != 0:
        detail = "\n".join(tail) or "(no output)"
        category = _classify_pip_failure(detail)
        raise RuntimeError(
            f"{_PIP_FAILURE_TITLES[category]}\n\n"
            f"pip exited with code {ret}.\n\n{detail}"
        )

    if progress_callback:
        progress_callback(end_pct, message)


def _run_pip_checked(report: dict, cmd, progress_callback, start_pct: int,
                     end_pct: int, message: str, abort_event=None,
                     timeout: float | None = None):
    """Igual que ``_run_pip`` pero, si pip falla, adjunta el resultado de la
    sonda de conectividad para distinguir 'red caída' de 'problema de pip'."""
    try:
        _run_pip(
            cmd, progress_callback, start_pct=start_pct, end_pct=end_pct,
            message=message, abort_event=abort_event, timeout=timeout,
        )
    except RuntimeError as exc:
        if str(exc) == "cancelled by user":
            raise
        raise RuntimeError(f"{exc}\n\n{report['summary']}") from exc


def _raise_unsupported_python() -> None:
    short = _python_short()
    raise RuntimeError(
        f"Python {short} no tiene wheels oficiales de PyTorch/ONNX todavía. "
        "Por favor, usa Python 3.9–3.14 para que la descarga funcione, "
        "o instala torch/onnxruntime manualmente en este entorno."
    )


def install_pytorch(
    variant: str = "auto",
    progress_callback=None,
    parent=None,
    abort_event=None,
    timeout: float | None = None,
) -> str:
    """Instala SOLO PyTorch (torch + torchvision + torchaudio) en
    ``bin/runtimes/ai_modules``.

    Separado de ONNX Runtime: cada fila del panel de dependencias instala
    únicamente su paquete, evitando arrastrar 3 GB de torch cuando el
    usuario solo quiere ONNX (y viceversa).

    ``abort_event`` cancela la descarga; ``timeout`` acota el tiempo total.
    """
    if not _is_python_version_supported():
        _raise_unsupported_python()
    report = _preflight_connectivity()
    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    # Purga previa: si el store tiene wheels de otra versión de Python
    # (cp314 vs cp311), quitarlos para que pip resuelva limpio.
    _purge_foreign_version_wheels()
    used_variant = _resolve_variant(variant)
    _check_disk_space("pytorch", used_variant)
    _run_pip_checked(
        report,
        _build_pytorch_command(used_variant),
        progress_callback,
        start_pct=5,
        end_pct=99,
        message="Descargando PyTorch...",
        abort_event=abort_event,
        timeout=timeout,
    )
    # Purga posterior: pip --target deja dist-infos huérfanos de la versión
    # previa que mantendrían el store 'incompatible' (guard de versión).
    _purge_foreign_version_wheels()
    if progress_callback:
        progress_callback(100, "PyTorch listo")
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    if not _can_import("torch"):
        raise RuntimeError(
            "PyTorch installation finished but the local runtime could not "
            "be imported. The installation may be incomplete or incompatible."
        )
    return used_variant


def install_onnxruntime(
    variant: str = "auto",
    progress_callback=None,
    parent=None,
    abort_event=None,
    timeout: float | None = None,
) -> str:
    """Instala SOLO ONNX Runtime (onnxruntime / onnxruntime-gpu) en
    ``bin/runtimes/ai_modules``.

    ``abort_event`` cancela la descarga; ``timeout`` acota el tiempo total.
    """
    if not _is_python_version_supported():
        _raise_unsupported_python()
    report = _preflight_connectivity()
    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    _purge_foreign_version_wheels()
    used_variant = _resolve_variant(variant)
    _check_disk_space("onnxruntime", used_variant)
    _run_pip_checked(
        report,
        _build_onnx_command(used_variant),
        progress_callback,
        start_pct=5,
        end_pct=99,
        message="Descargando ONNX Runtime...",
        abort_event=abort_event,
        timeout=timeout,
    )
    _purge_foreign_version_wheels()
    if progress_callback:
        progress_callback(100, "ONNX Runtime listo")
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    if not _can_import("onnxruntime"):
        raise RuntimeError(
            "ONNX Runtime installation finished but the local runtime could "
            "not be imported. The installation may be incomplete or incompatible."
        )
    return used_variant


def install_runtimes(
    variant: str = "auto",
    progress_callback=None,
    parent=None,
    abort_event=None,
    timeout: float | None = None,
) -> str:
    """Instala PyTorch y ONNX Runtime en ``bin/runtimes/ai_modules``.

    Args:
        variant: "cpu", "cuda118", "cuda121", "cuda126", "cuda130" o "auto".
        progress_callback: callable(percent, message) -> None.
        parent: QWidget padre (no usado actualmente, reservado).

    Returns:
        El variant real usado ("cpu", "cuda121", "cuda130", ...).
    """
    if not _is_python_version_supported():
        _raise_unsupported_python()

    report = _preflight_connectivity()

    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    _purge_foreign_version_wheels()

    used_variant = _resolve_variant(variant)

    # Umbral combinado: el pico de disco lo marcan los wheels CUDA de torch.
    _check_disk_space("all", used_variant)

    # 1. PyTorch (la parte más grande)
    _run_pip_checked(
        report,
        _build_pytorch_command(used_variant),
        progress_callback,
        start_pct=5,
        end_pct=85,
        message="Descargando PyTorch...",
        abort_event=abort_event,
        timeout=timeout,
    )

    # 2. ONNX Runtime
    _run_pip_checked(
        report,
        _build_onnx_command(used_variant),
        progress_callback,
        start_pct=86,
        end_pct=99,
        message="Descargando ONNX Runtime...",
        abort_event=abort_event,
        timeout=timeout,
    )

    _purge_foreign_version_wheels()

    if progress_callback:
        progress_callback(100, "Runtimes de IA listos")

    # Después de instalar, reinjectar por si acaso y verificar ambos
    # paquetes. Pip puede terminar con código 0 aunque un wheel nativo no
    # sea importable en la versión/arquitectura actual.
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    missing = [
        name for name in ("torch", "onnxruntime")
        if not _can_import(name)
    ]
    if missing:
        raise RuntimeError(
            "Runtime installation finished but these local packages could "
            "not be imported: " + ", ".join(missing)
        )
    return used_variant


# ---------------------------------------------------------------------------
# Per-package status helpers
# ---------------------------------------------------------------------------


def _version_of(module_name: str) -> str:
    """Obtiene la versión exclusivamente desde dist-info del store local."""
    try:
        import importlib.metadata as _meta
        wanted = module_name.replace("_", "-").lower()
        for dist in _meta.distributions(path=[RUNTIME_MODULES_DIR]):
            name = (dist.metadata.get("Name") or "").replace("_", "-").lower()
            if name == wanted:
                return dist.version or "—"
    except Exception:
        pass
    return "—"


def _import_dir(module_name: str) -> str:
    """Directorio desde el que se importaría el módulo (spec.origin)."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None and spec.origin:
            if _is_path_in_store(spec.origin):
                return os.path.dirname(spec.origin)
    except Exception:
        pass
    return RUNTIME_MODULES_DIR


def torch_status() -> dict:
    # Status is also a public boundary (la UI puede llamarlo sin pasar por
    # main.py): asegura que solo el store privado participa en la resolución.
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    installed = _can_import("torch")
    gpu = False
    if installed:
        try:
            _torch = import_local_runtime("torch")
            gpu = _torch.cuda.is_available()
        except Exception:
            pass
    return {
        "id": "torch",
        "display_name": "PyTorch runtime",
        # Autocontenido: "installed" SOLO si el store local lo tiene.
        # Una copia del sistema/venv no satisface la dependencia.
        "installed": installed,
        "local": installed,
        "system": _can_import_system("torch") if not installed else False,
        "version": _version_of("torch") if installed else "—",
        "gpu": gpu,
        "install_dir": _import_dir("torch") if installed else RUNTIME_MODULES_DIR,
    }


def onnx_status() -> dict:
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    installed = _can_import("onnxruntime")
    gpu = False
    if installed:
        try:
            _ort = import_local_runtime("onnxruntime")
            gpu = "CUDAExecutionProvider" in _ort.get_available_providers()
        except Exception:
            pass
    return {
        "id": "onnxruntime",
        "display_name": "ONNX Runtime (CPU/GPU)",
        "installed": installed,
        "local": installed,
        "system": _can_import_system("onnxruntime") if not installed else False,
        "version": _version_of("onnxruntime") if installed else "—",
        "gpu": gpu,
        "install_dir": _import_dir("onnxruntime") if installed else RUNTIME_MODULES_DIR,
    }


# ---------------------------------------------------------------------------
# Entry points from dependencies.py
# ---------------------------------------------------------------------------


def install_all(progress_callback=None, parent=None, variant: str = "auto",
                abort_event=None, timeout: float | None = None) -> None:
    """Instala PyTorch + ONNX en un solo comando."""
    install_runtimes(
        variant=variant,
        progress_callback=progress_callback,
        parent=parent,
        abort_event=abort_event,
        timeout=timeout,
    )


def uninstall_local_runtimes(parent=None) -> None:
    """Borra el store local de runtimes (``bin/runtimes/ai_modules``) completo.

    PyTorch y ONNX Runtime se instalan juntos ahí, así que el borrado es
    conjunto. En Windows, si algún DLL está cargado (runtime ya importado),
    ``dependencies._rmtree_force`` lo programa para el próximo arranque.
    """
    try:
        from src.dependencies import _rmtree_force
    except Exception:
        return
    if os.path.isdir(RUNTIME_MODULES_DIR):
        _rmtree_force(RUNTIME_MODULES_DIR)


def get_status() -> dict:
    """Estado consolidado de los runtimes de IA para dependencies.py."""
    torch = torch_status()
    onnx = onnx_status()

    if torch["installed"] and onnx["installed"]:
        status = "installed"
        detail = (
            f"PyTorch {torch['version']} ({'CUDA' if torch['gpu'] else 'CPU'}), "
            f"ONNX Runtime {onnx['version']} ({'CUDA' if onnx['gpu'] else 'CPU'})"
        )
    else:
        status = "missing"
        missing = []
        if not torch["installed"]:
            missing.append("PyTorch")
        if not onnx["installed"]:
            missing.append("ONNX Runtime")
        detail = f"Missing: {', '.join(missing)}. Click Install to download."

    return {
        "id": "ai_runtimes",
        "display_name": "AI Runtimes (PyTorch + ONNX)",
        "kind": "runtime",
        "status": status,
        "version": torch.get("version") or "—",
        "install_dir": RUNTIME_MODULES_DIR,
        "source_url": "https://pytorch.org/get-started/locally/",
        "is_updatable": True,
        "size_bytes": _safe_dir_size(RUNTIME_MODULES_DIR),
        "detail": detail,
        "torch": torch,
        "onnx": onnx,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
    except Exception:
        pass
    return total


# ---------------------------------------------------------------------------
# Windows DLL path helper
# ---------------------------------------------------------------------------


def ensure_runtime_dlls_on_path() -> None:
    """Hace localizables las DLL nativas de torch/onnx del store.

    Do canales porque en Python ≥3.8 `PATH` solo se usa como fallback para
    DLLs de extensiones; ``os.add_dll_directory`` es el mecanismo fiable
    (y ademas sale antes que un CUDA toolkit viejo del PATH de la máquina,
    que antes de esta fix podía cargar un cudnn desalineado y dar
    WinError 127/126 ilegible en los imports).

    Los invocadores seguros (``install_pytorch`` etc.) la reinvocan tras
    instalar: así una app recién instalada encuentra las DLLs en esa
    misma sesión sin reiniciar.
    """
    if sys.platform != "win32":
        return
    if not store_is_compatible_with_python():
        return
    try:
        os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    except Exception:
        pass

    extra = [RUNTIME_MODULES_DIR]
    nvidia_dir = os.path.join(RUNTIME_MODULES_DIR, "nvidia")
    if os.path.isdir(nvidia_dir):
        for sub in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, sub, "bin")
            if os.path.isdir(bin_dir):
                extra.append(bin_dir)
                for entry in os.listdir(bin_dir):
                    d = os.path.join(bin_dir, entry)
                    if os.path.isdir(d):
                        extra.append(d)

    # Canal fiable: os.add_dll_directory (passed to LoadLibraryExW con
    # LOAD_LIBRARY_SEARCH_USER_DIRS). Se llama de nuevo al重複idempotente
    # (el handle interno vive hasta process exit; no duplica).
    for d in extra:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass

    # Canal legacy: PATH (el viej¿hasta el loader solo entra aquí cuando
    # no se resolvió nada por add_dll_directory).
    current_path = os.environ.get("PATH", "")
    new_parts = [p for p in extra if p not in current_path]
    if new_parts:
        os.environ["PATH"] = ";".join(new_parts) + ";" + current_path
