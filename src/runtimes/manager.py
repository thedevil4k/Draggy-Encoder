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
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

from src.config import ensure_runtime_dirs, get_bin_dir


RUNTIME_MODULES_DIR = os.path.join(get_bin_dir(), "runtimes", "ai_modules")

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
        if not os.path.normcase(spec.origin).startswith(_store_abspath()):
            return False
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


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
        _pip_python(), "-m", "pip", "install",
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
        _pip_python(), "-m", "pip", "install",
        "--target", RUNTIME_MODULES_DIR,
        "--no-warn-script-location",
        "--upgrade",
        pkg,
    ]


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

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

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    # Verificar que pip esté disponible
    try:
        test_proc = subprocess.run(
            [_pip_python(), "-m", "pip", "--version"],
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
        raise RuntimeError(f"pip exited with code {ret}\n\n{detail}")

    if progress_callback:
        progress_callback(end_pct, message)


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
    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    # Purga previa: si el store tiene wheels de otra versión de Python
    # (cp314 vs cp311), quitarlos para que pip resuelva limpio.
    _purge_foreign_version_wheels()
    used_variant = _resolve_variant(variant)
    _run_pip(
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
    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    _purge_foreign_version_wheels()
    used_variant = _resolve_variant(variant)
    _run_pip(
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

    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    _purge_foreign_version_wheels()

    used_variant = _resolve_variant(variant)

    # 1. PyTorch (la parte más grande)
    _run_pip(
        _build_pytorch_command(used_variant),
        progress_callback,
        start_pct=5,
        end_pct=85,
        message="Descargando PyTorch...",
        abort_event=abort_event,
        timeout=timeout,
    )

    # 2. ONNX Runtime
    _run_pip(
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
    try:
        import importlib.metadata as _meta
        return _meta.version(module_name) or "—"
    except Exception:
        return "—"


def _import_dir(module_name: str) -> str:
    """Directorio desde el que se importaría el módulo (spec.origin)."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None and spec.origin:
            return os.path.dirname(spec.origin)
    except Exception:
        pass
    return RUNTIME_MODULES_DIR


def torch_status() -> dict:
    installed = _can_import("torch")
    gpu = False
    if installed:
        try:
            import torch as _torch
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
    installed = _can_import("onnxruntime")
    gpu = False
    if installed:
        try:
            import onnxruntime as _ort
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
