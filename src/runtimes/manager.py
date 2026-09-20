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
import logging
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

logger = logging.getLogger(__name__)

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

    Un tag que NO se reconoce se acepta (True): este guard solo debe rechazar
    lo que demuestra ser de otra CPython. Rechazar por un tag raro convierte
    un WHEEL malformado o un formato futuro en "el store entero es
    incompatible" — y eso desactiva la IA y provoca re-descargas de GB.
    """
    text = (tag or "").strip()
    if not text:
        return True
    first = text.split("-")[0].strip()
    if not first.startswith("cp"):
        return True  # py3-none-any, py2.py3-none-any, ...
    digits = ""
    for char in first[2:]:
        if char.isdigit():
            digits += char
        else:
            break
    if not digits:
        return True
    try:
        tag_num = int(digits)
        cur_num = int(f"{sys.version_info.major}{sys.version_info.minor}")
    except (TypeError, ValueError):  # pragma: no cover - int() sobre dígitos
        return True
    if "abi3" in text:
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


def _purge_stale_module_modules(module_name: str) -> int:
    """Borra ``module`` y todos sus submódulos de ``sys.modules``.

    Tras un import que falla a mitad de ejecución, Python expulsa el módulo
    raíz de ``sys.modules`` pero CONSERVA los submódulos ya cargados. Un
    re-import posterior re-ejecuta el ``__init__`` del paquete y, al
    reutilizar esos submódulos huérfanos, NO re-adjunta los atributos al
    paquete (p. ej. ``torch.autograd``) y re-registra estado global de las
    extensiones C: el segundo import revienta con errores SECUNDARIOS falsos
    ("partially initialized module 'torch' ... has no attribute 'autograd'
    (most likely due to a circular import)", "Only a single TORCH_LIBRARY
    can be used to register the namespace triton") que ocultan la causa
    original. Purgar del todo fuerza una ejecución limpia y reproducible.
    """
    prefix = module_name + "."
    stale = [key for key in sys.modules if key == module_name or key.startswith(prefix)]
    for key in stale:
        try:
            del sys.modules[key]
        except KeyError:  # pragma: no cover - carrera teórica
            pass
    return len(stale)


def _store_has_package_dir(module_name: str) -> bool:
    """True si el store contiene el paquete (``<module>/__init__.py``)."""
    return os.path.isfile(
        os.path.join(RUNTIME_MODULES_DIR, module_name, "__init__.py")
    )


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

    Autocuración: si el import falla (DLL bloqueada un instante por el
    antivirus, fallo transitorio, etc.), el paquete queda medio cargado y
    TODO re-import posterior en este proceso revienta con errores falsos de
    circular import (ver ``_purge_stale_module_modules``). Aquí purgamos y
    reintentamos una vez; si el reintento también falla, purgamos de nuevo
    para no dejar el proceso envenenado para el siguiente intento.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return False
        if not _is_path_in_store(spec.origin):
            return False
    except Exception:
        return False
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        _purge_stale_module_modules(module_name)
        try:
            importlib.import_module(module_name)
            return True
        except Exception:
            _purge_stale_module_modules(module_name)
            return False


def _import_error_detail(module_name: str) -> str:
    """Diagnóstico legible cuando ``_can_import`` falla.

    Intenta importar el módulo y, si falla, devuelve el traceback completo
    (tipo + mensaje + pila). Si el spec ni siquiera apunta al store local,
    explica por qué. Nunca lanza: siempre devuelve una cadena.
    """
    import traceback as _tb

    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:  # pragma: no cover - muy raro
        return f"find_spec('{module_name}') falló: {type(exc).__name__}: {exc}\n{_tb.format_exc()}"

    if spec is None:
        return (
            f"importlib.find_spec('{module_name}') devolvió None — "
            f"el paquete no se encontró en sys.path.\n"
            f"sys.path[0:3]={sys.path[:3]}\n"
            f"store={RUNTIME_MODULES_DIR}\n"
            f"store existe: {os.path.isdir(RUNTIME_MODULES_DIR)}"
        )
    if spec.origin is None:
        # Namespace package (directorio vacío sin __init__.py) -> pip no terminó
        try:
            locs = getattr(spec, "submodule_search_locations", None)
            locs = list(locs) if locs else None
        except Exception:
            locs = None
        return (
            f"spec.origin is None para '{module_name}' (namespace package, "
            f"posible instalación incompleta/corrupta).\n"
            f"spec.submodule_search_locations={locs}\n"
            f"store={RUNTIME_MODULES_DIR}"
        )
    if not _is_path_in_store(spec.origin):
        return (
            f"'{module_name}' se resolvió fuera del store privado:\n"
            f"  spec.origin={spec.origin}\n"
            f"  store={RUNTIME_MODULES_DIR}\n"
            f"  sys.path[0]={sys.path[0] if sys.path else '—'}"
        )
    # Origen correcto pero la importación falla (DLL, ABI, etc.)
    try:
        importlib.import_module(module_name)
        return ""  # no debería llegar aquí: _can_import habría dado True
    except Exception as exc:
        tb = _tb.format_exc()
        # Mensaje corto primero, traceback completo después (útil en
        # 'Mostrar detalles' del diálogo de error).
        return f"{type(exc).__name__}: {exc}\n{tb}"

    return f"Importación de '{module_name}' falló por motivo desconocido."


# ---------------------------------------------------------------------------
# Subprocess store-import validation (proceso limpio, sys.modules vacío)
# ---------------------------------------------------------------------------


# Sonda que ejecuta el subproceso: importa el módulo con el store como
# primer ``sys.path`` y VERIFICA que el paquete cargado vive dentro del
# store (misma regla "caracol" que ``_can_import``: un torch del venv o del
# sistema no cuenta aunque importe sin problemas).
_IMPORT_PROBE_SOURCE = (
    "import importlib, os, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "try:\n"
    "    module = importlib.import_module(sys.argv[2])\n"
    "except BaseException:\n"
    "    import traceback\n"
    "    traceback.print_exc()\n"
    "    sys.exit(3)\n"
    "origin = os.path.abspath(getattr(module, '__file__', '') or '')\n"
    "store = os.path.abspath(sys.argv[1])\n"
    "ok = os.path.normcase(origin).startswith(os.path.normcase(store) + os.sep)\n"
    "print('DRAGGY-IMPORT-OK' if ok else 'DRAGGY-IMPORT-FOREIGN')\n"
    "sys.exit(0 if ok else 4)\n"
)


def _can_probe_subprocess() -> bool:
    """True si existe un intérprete apto para la sonda en subproceso.

    Frozen: el Python privado del instalador (embed-python). Dev: el propio
    ``sys.executable`` vale, porque la sonda verifica el origen y un torch
    del venv no da falso positivo.
    """
    if not getattr(sys, "frozen", False):
        return True
    return get_embed_python_path() is not None


def _validate_store_import_subprocess(
    module_name: str, timeout: float = 180.0
) -> tuple[bool, str]:
    """Valida en un subproceso LIMPIO que ``module_name`` importe del store.

    Un import in-process hereda el estado del proceso: ``sys.modules`` con
    submódulos huérfanos de intentos previos y DLLs ya cargadas. El
    subproceso no hereda nada, así que su resultado es el veredicto sobre
    el STORE (no sobre el proceso actual) y su traceback es la causa real.

    Devuelve ``(ok, detalle)``; ``detalle`` es el stderr/stdout del
    subproceso (causa legible) o la explicación de por qué no se sondeó.
    """
    python = get_embed_python_path()
    if python is None:
        if getattr(sys, "frozen", False):
            return False, (
                "No se encontró el Python privado (embed-python) junto al "
                "ejecutable: no se puede validar el runtime en un subproceso. "
                "Reinstala la aplicación."
            )
        python = sys.executable
    try:
        proc = subprocess.run(
            [python, "-c", _IMPORT_PROBE_SOURCE, RUNTIME_MODULES_DIR, module_name],
            capture_output=True, text=True, timeout=timeout,
            env=_isolated_pip_env(),
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"La sonda de import de '{module_name}' agotó {int(timeout)}s "
            "(el import se colgó; posible antivirus examinando las DLL "
            "recién extraídas o disco muy lento)."
        )
    except OSError as exc:
        return False, f"No se pudo ejecutar la sonda de import ({python}): {exc}"
    if proc.returncode == 0 and "DRAGGY-IMPORT-OK" in (proc.stdout or ""):
        return True, ""
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    detail = output or f"(sin salida; exit code {proc.returncode})"
    return False, detail


_SUBPROBE_CACHE: dict[str, tuple[tuple, bool, str]] = {}


def _store_probe_signature(module_name: str) -> tuple:
    """Firma barata del paquete en el store para invalidar la caché."""
    init_py = os.path.join(RUNTIME_MODULES_DIR, module_name, "__init__.py")
    try:
        stat = os.stat(init_py)
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (0, 0)


def _invalidate_store_probe_cache(module_name: str | None = None) -> None:
    """Invalida la caché de la sonda (tras (re)instalar o borrar el store)."""
    if module_name is None:
        _SUBPROBE_CACHE.clear()
    else:
        _SUBPROBE_CACHE.pop(module_name, None)


def _subprocess_import_ok(module_name: str) -> tuple[bool, str]:
    """``_validate_store_import_subprocess`` con caché por firma del store.

    La sonda cuesta segundos (importa torch de verdad); cachear por
    (mtime, tamaño) del ``__init__.py`` evita repetirla en cada refresco
    de estado mientras el paquete no cambie.
    """
    signature = _store_probe_signature(module_name)
    cached = _SUBPROBE_CACHE.get(module_name)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]
    ok, detail = _validate_store_import_subprocess(module_name)
    _SUBPROBE_CACHE[module_name] = (signature, ok, detail)
    return ok, detail


def _post_install_import_check(module_name: str) -> tuple[bool, str]:
    """Veredicto tras instalar: el subproceso manda, el in-process da contexto.

    - Subproceso OK  -> el store es válido: instalado (aunque el proceso
      actual tarde más en importarlo; ``_can_import`` ya purga y reintenta).
    - Subproceso KO  -> el store no importa de verdad: el detalle es el
      stderr real del paquete, no un artefacto de re-import.
    - Sin subproceso -> fallback in-process con el detalle de siempre.
    """
    if _can_import(module_name):
        return True, ""
    if _can_probe_subprocess():
        ok, detail = _validate_store_import_subprocess(module_name)
        if ok:
            if not _can_import(module_name):
                logger.warning(
                    "%s importa en subproceso pero no in-process: store "
                    "válido, no se reinstala", module_name
                )
            return True, ""
        in_process = _import_error_detail(module_name)
        return False, (
            f"{detail}\n\n"
            f"--- Diagnóstico del proceso actual (contexto) ---\n{in_process}"
        )
    return False, _import_error_detail(module_name)


def _vc_redist_hint(detail: str) -> str:
    """Hint específico de Windows cuando faltan las DLL de VC++."""
    low = (detail or "").lower()
    if any(k in low for k in ("vcruntime", "msvcp", "api-ms-win-crt", "142", "140")):
        return (
            " Parece que falta Microsoft Visual C++ Redistributable "
            "(vcruntime140.dll / msvcp140.dll).\n"
            " Instálalo desde https://aka.ms/vs/17/release/vc_redist.x64.exe "
            "y vuelve a intentar instalar PyTorch.\n"
        )
    if "dll load failed" in low or "errno 126" in low or "errno 127" in low:
        return (
            " Falló la carga de una DLL nativa (posible antivirus bloqueando "
            "torch_cpu.dll / c10.dll o falta de VC++ Redistributable). "
            "Prueba a desactivar el antivirus momentáneamente o instalar la variante CPU.\n"
        )
    return ""


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

    El suelo real lo marca PyTorch: las versiones actuales publican desde
    Python 3.10 (antes llegaron a soportar 3.8/3.9). Si la app corre por
    debajo, ningún índice sirve wheels y seguir sólo alarga el error, así que
    se rechaza de entrada con un mensaje claro.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    if major != 3:
        return False
    minimum = _torch_min_python(_TORCH_MIN_PYTHON[0][0])
    return minor >= minimum[1]


def get_python_support_info() -> dict:
    """Devuelve información de compatibilidad para mostrar en la UI."""
    supported = _is_python_version_supported()
    short = _python_short()
    minimum = _torch_min_python(_TORCH_MIN_PYTHON[0][0])
    floor = f"{minimum[0]}.{minimum[1]}"
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
            f"Python {short} is NOT supported by official PyTorch/ONNX wheels. "
            f"Please use Python {floor} or newer, or build the PyInstaller "
            "bundle with one of those versions. "
            "AI upscaling/interpolation/colorize will be disabled."
        ),
    }


# ---------------------------------------------------------------------------
# Package specs — resolución dinámica de variante y versión
#
# El bug histórico (2026): la variante CUDA se elegía con una heurística
# ("Python >= 3.13 -> cu126, si no cu121") y la versión se fijaba a un pin
# estático (torch==2.13.0). Ambas cosas se decidían SIN consultar los índices
# reales, así que la pareja (índice, versión) podía no existir:
#
#   cu121 (elegido para Python 3.9-3.12 con GPU NVIDIA) no publica NINGUNA
#   versión del pin -> pip responde "Could not find a version that satisfies
#   the requirement torch==2.13.0" y la instalación falla con código 1.
#
# La regla correcta es no adivinar: se consultan los índices de PyTorch y se
# instala la pareja (variante, versión) que DE VERDAD tiene wheels para el
# intérprete, la plataforma y el driver actuales. El pin estático sobrevive
# solo como plan B para cuando no hay red (ver _plan_from_offline_defaults).
# ---------------------------------------------------------------------------

# Tag de CPython del intérprete en ejecución, como prefijo de wheel ("cp314").
_CP_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"

# Fichero de caché disco de la resolución (evita re-sondear los índices en
# cada refresco de estado y sobrevive a reinicios de la app).
_PLAN_CACHE_FILE = "pytorch_plan.json"
_PLAN_CACHE_TTL_S = 12 * 3600
_plan_cache: dict = {"ts": 0.0, "plan": None}

# Candidatos ordenados por preferencia. `driver_any` son versiones de driver
# NVIDIA (rama 5xx, tal como las reporta el propio driver) que sabemos que
# pueden cargar la CUDA embebida de cada wheel. Un driver de rama igual o
# superior es aceptable; si no se puede medir el driver, se acepta el
# candidato (mejor intentar CUDA que degradar a CPU por una sonda fallida).
#
# NOTA sobre cu126: los wheels de PyTorch embeben el runtime CUDA y hablan
# con el driver por la CUDA Driver API, que es retrocompatible dentro de la
# misma rama mayor, pero no entre ramas mayores. Los requisitos de abajo son
# deliberadamente conservadores: solo se exige el driver cuando la rama mayor
# del runtime es MÁS NUEVA que la del toolkit CUDA 12 con el que se publicó.
_VARIANT_CANDIDATES: tuple[dict, ...] = (
    {"variant": "cuda130", "cuda_pref": "13.0", "url": "https://download.pytorch.org/whl/cu130",
     "label": "CUDA 13.0", "driver_any": (580,)},
    {"variant": "cuda126", "cuda_pref": None, "url": "https://download.pytorch.org/whl/cu126",
     "label": "CUDA 12.6", "driver_any": ()},
    {"variant": "cuda121", "cuda_pref": None, "url": "https://download.pytorch.org/whl/cu121",
     "label": "CUDA 12.1", "driver_any": ()},
    {"variant": "cuda118", "cuda_pref": None, "url": "https://download.pytorch.org/whl/cu118",
     "label": "CUDA 11.8", "driver_any": ()},
    {"variant": "cpu", "cuda_pref": None, "url": "https://download.pytorch.org/whl/cpu",
     "label": "CPU", "driver_any": ()},
)
_CANDIDATES_BY_VARIANT = {c["variant"]: c for c in _VARIANT_CANDIDATES}
_INDEX_URLS = {c["variant"]: c["url"] for c in _VARIANT_CANDIDATES}

# Paquetes que forman el set de PyTorch (los tres se publican en el mismo
# índice y deben salir de la MISMA familia de versión para ser consistentes).
_PYTORCH_PACKAGES = ("torch", "torchvision", "torchaudio")

# Versión mínima de Python de cada familia de torch. Consultar la metadata
# de PyPI por versión exacta sería una petición por versión; este mapa cubre
# los saltos reales de `requires-python` publicados por PyTorch y es la parte
# que decide si un Python antiguo puede instalar la familia nueva.
_TORCH_MIN_PYTHON: tuple[tuple[str, tuple[int, int]], ...] = (
    ("2.13", (3, 10)),
    ("2.10", (3, 10)),
    ("2.9", (3, 9)),
    ("2.6", (3, 9)),
    ("2.5", (3, 9)),
    ("2.4", (3, 8)),
)

# Plan B sin red: el set más reciente que conocemos bueno, con el índice que
# lo publica. Si el usuario está offline y no hay caché, se usa esto y pip
# decide; el error final (si lo hay) ya lo explica _pytorch_no_candidate_error.
_TORCH_PIN = "torch==2.13.0"
_TORCHVISION_PIN = "torchvision==0.28.0"
_TORCHAUDIO_PIN = "torchaudio==2.11.0"
_OFFLINE_PLAN_VARIANT = "cuda126"


class PytorchPlanError(RuntimeError):
    """No hay ninguna pareja (variante, versión) instalable.

    Lleva ``attempts`` para que la UI pueda listar qué se probó y por qué
    falló cada candidato, en lugar del escueto "pip exited with code 1".
    """

    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def _default_gpu_variant() -> str:
    """Variante CUDA preferida de forma offline (sin consultar índices).

    Solo se usa como último recurso cuando la red no permite resolver el
    plan. El orden de preferencia lo define :data:`_VARIANT_CANDIDATES`:
    se prefiere la CUDA más nueva disponible y se descarta lo que no pueda
    publicar wheels (cu121 ya no recibe versiones nuevas).
    """
    return _VARIANT_CANDIDATES[0]["variant"]


def _resolve_variant(variant: str) -> str:
    """Normaliza un nombre de variante a cpu/cuda118/cuda121/cuda126/cuda130.

    ``auto`` NO consulta la red aquí (es una función barata y síncrona que
    usan también los tests y la UI): devuelve la variante preferida según el
    hardware. La resolución real con consulta a los índices la hace
    :func:`resolve_pytorch_plan`, que es la que usa el instalador.
    """
    if variant in ("cuda11.8",):
        return "cuda118"
    if variant in _CANDIDATES_BY_VARIANT:
        return variant
    if variant == "auto":
        if _has_nvidia_gpu():
            return _gpu_preferred_variant()
        return "cpu"
    return "cpu"


def _nvidia_driver_branch() -> int | None:
    """Rama mayor del driver NVIDIA ("580" en "580.88"), o None si no se mide.

    ``nvidia-smi`` es el canal fiable y está en PATH en cualquier equipo con
    driver instalado (el mismo supuesto que usa ``_has_nvidia_gpu``). Un
    fallo de la sonda NO se interpreta como "sin GPU": devuelve None.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or "").strip().splitlines()
    if not first:
        return None
    m = re.match(r"\s*(\d+)", first[0])
    return int(m.group(1)) if m else None


def _variant_driver_ok(candidate: dict) -> bool:
    """True si el driver instalado puede cargar la CUDA de este candidato.

    ``driver_any`` vacío significa "sin requisito medible" (CUDA 12.x: la
    retrocompatibilidad dentro de la rama 12 cubre los drivers en uso). Si el
    requisito existe y el driver NO se pudo medir, se acepta igualmente: es
    preferible intentar CUDA y fallar con un mensaje claro que degradar a CPU
    silenciosamente por una sonda que no respondió.
    """
    required = candidate.get("driver_any") or ()
    if not required:
        return True
    driver = _nvidia_driver_branch()
    if driver is None:
        return True
    return driver >= min(required)


def _gpu_preferred_variant() -> str:
    """Primera variante CUDA de la lista de preferencia que el driver acepte.

    Si ninguna CUDA es viable con el driver actual, devuelve "cpu": instalar
    una torch CUDA que no puede cargar deja el runtime inservible, y el
    usuario siempre puede forzar una variante concreta desde la UI.
    """
    for cand in _VARIANT_CANDIDATES:
        if cand["variant"] == "cpu":
            break
        if _variant_driver_ok(cand):
            return cand["variant"]
    return "cpu"


def _version_key(version: str) -> tuple:
    """Clave de orden comparable para versiones ``2.13.0`` / ``0.28.0``.

    Los segmentos no numéricos (local ``+cu126``, sufijos ``rc1``) se
    descartan: para elegir "la más nueva" basta con la parte numérica, y así
    el orden es total y estable (nada de tuple[Any] incomparable).
    """
    numeric: list[int] = []
    for part in re.split(r"[.+-]", version or ""):
        if part.isdigit():
            numeric.append(int(part))
        else:
            break
    return tuple(numeric)


def _torch_min_python(family: str) -> tuple[int, int]:
    """Versión mínima de Python publicada para una familia de torch.

    ``family`` es "major.minor" ("2.13"). Las familias más nuevas que
    conocemos empiezan en 3.10; las antiguas llegaron a soportar 3.8.
    """
    for prefix, minimum in _TORCH_MIN_PYTHON:
        if family == prefix:
            return minimum
    try:
        major, minor = (int(x) for x in family.split(".")[:2])
    except ValueError:
        return (3, 9)
    newest = _version_key(_TORCH_MIN_PYTHON[0][0])
    if (major, minor) >= newest:
        return _TORCH_MIN_PYTHON[0][1]
    oldest = _version_key(_TORCH_MIN_PYTHON[-1][0])
    if (major, minor) <= oldest:
        return _TORCH_MIN_PYTHON[-1][1]
    return (3, 9)


def _platform_compatible_index() -> bool:
    """True si el índice de PyTorch publica wheels para esta plataforma.

    macOS no usa los índices ``whl/<variante>`` (recibe torch por PyPI), así
    que la resolución dinámica se salta en Darwin y se deja el plan offline.
    """
    return not sys.platform.startswith("darwin")


_wheel_list_cache: dict[str, list[tuple[str, str]]] = {}


def _http_get_text(url: str, timeout: float = 30.0) -> str:
    """GET de texto con ``requests`` y repliegue a la stdlib.

    ``requests`` viene con el bundle (lo arrastra huggingface_hub), pero el
    gestor de runtimes también se ejecuta en el intérprete privado desnudo
    (sonda/CLI) y en tests: un helper de red no debe ser el punto único de
    fallo. El repliegue usa ``urllib`` con el CA bundle de certifi si está
    disponible (el mismo que confía pip).
    """
    try:
        import requests

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as primary:
        try:
            import ssl
            import urllib.request

            context = None
            try:
                import certifi  # type: ignore

                context = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                context = ssl.create_default_context()
            req = urllib.request.Request(
                url, headers={"User-Agent": "DraggyEncoder runtime-resolver"}
            )
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as fallback:
            raise RuntimeError(
                f"No se pudo consultar {url}: {primary} / {fallback}"
            ) from fallback


def _index_wheel_list(index_url: str, package: str) -> list[tuple[str, str]]:
    """``[(version, filename)]`` publicados por ``index_url`` para ``package``.

    El índice de PyTorch es un listado HTML estático (el mismo que consume
    pip). Solo se aceptan wheels compatibles con el intérprete y la
    plataforma actuales: de nada sirve una versión que existe solo para otra
    CPython. Cachea en memoria por sesión.
    """
    cache_key = f"{index_url}|{package}"
    if cache_key in _wheel_list_cache:
        return _wheel_list_cache[cache_key]

    url = f"{index_url}/{package}/"
    lookup_error = ""
    try:
        html = _http_get_text(url, timeout=30)
    except RuntimeError as exc:
        lookup_error = str(exc)
        html = ""

    found: list[tuple[str, str]] = []
    if html:
        # El href puede ser relativo o absoluto y puede llevar fragmento
        # ``#sha256=...`` entre el nombre del wheel y la comilla de cierre
        # (los índices modernos lo añaden): hay que tolerar ambos, o páginas
        # como torchaudio devuelven cero wheels y parecen vacías.
        for href in re.findall(r'<a[^>]+href="([^"]+)"', html):
            filename = href.split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1]
            if not filename.endswith(".whl"):
                continue
            if not filename.startswith(package + "-"):
                continue
            if not _filename_matches_python(filename):
                continue
            if not _wheel_matches_platform(filename):
                continue
            found.append((_wheel_version(filename), filename))
    if not found and lookup_error:
        logger.info(
            "resolución de %s en %s sin resultados (%s)", package, url, lookup_error
        )
    _wheel_list_cache[cache_key] = found
    return found


def _filename_matches_python(filename: str) -> bool:
    """True si un wheel sirve para el intérprete actual.

    A diferencia de ``_wheel_matches_cp`` (que exige el tag exacto
    ``cp313-cp313`` y sirve para validar un wheel ya elegido), aquí también
    se aceptan los wheels ``abi3``, que son compatibles hacia delante.
    """
    stem = filename[:-4] if filename.endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 3:
        return False
    py_tag, abi_tag = parts[-3], parts[-2]
    if py_tag == _CP_TAG and abi_tag == _CP_TAG:
        return True
    if "abi3" in abi_tag:
        m = re.match(r"cp(\d+)", py_tag)
        if m:
            return int(_CP_TAG[2:]) >= int(m.group(1))
    return False


def _wheel_version(filename: str) -> str:
    """Versión declarada por el nombre de un wheel, normalizada.

    No se puede trocear por ``-`` a ciegas: las URLs del índice percent-
    codifican el ``+`` del segmento local (``torch-2.13.0%2Bcu130-...``), así
    que ``%2B`` cuenta como un guion extra y desplaza los índices. Se extrae
    la versión estructuralmente: nombre, versión y luego los tres tags
    (python-abi-plataforma) al final.

        torch-2.9.0+cu126-cp313-cp313-win_amd64.whl -> 2.9.0+cu126
        torch-2.13.0-cp313-cp313-win_amd64.whl      -> 2.13.0
    """
    stem = filename[:-4] if filename.endswith(".whl") else filename
    match = re.match(r"^[^-]+-([^-%]+(?:%2B[^-]*)?)-", stem, re.IGNORECASE)
    if not match:
        return ""
    return _normalize_version(match.group(1))


def _normalize_version(version: str) -> str:
    """Deshace el percent-encoding de una versión (``2.13.0%2Bcu130``)."""
    return (version or "").replace("%2B", "+").replace("%2b", "+").replace("%2E", ".").replace("%2e", ".")


def _public_version(version: str) -> str:
    """``2.9.0+cu126`` -> ``2.9.0`` (el pin que viaja a pip)."""
    return version.split("+", 1)[0]


def _pypi_release_requires_python(package: str, version: str) -> tuple[int, int] | None:
    """``requires_python`` de una release cualquiera (cacheado), o None."""
    if not version:
        return None
    cache_key = f"pyfloor|{package}|{version}"
    cached = _wheel_list_cache.get(cache_key)
    if cached is not None:
        return cached[0][0] if cached else None
    floor: tuple[int, int] | None = None
    try:
        import json as _json

        payload = _json.loads(
            _http_get_text(
                f"https://pypi.org/pypi/{package}/{version}/json", timeout=30
            )
        )
        requires = ((payload.get("info") or {}).get("requires_python") or "").strip()
        match = re.search(r">=\s*(\d+)\.(\d+)", requires)
        if match:
            floor = (int(match.group(1)), int(match.group(2)))
    except Exception:
        floor = None
    _wheel_list_cache[cache_key] = [(floor, "")] if floor else []
    return floor


def _pypi_versions_newest_first(package: str) -> list[str]:
    """Versiones publicadas en PyPI, de más nueva a más antigua (cacheado)."""
    cache_key = f"pypiversions|{package}"
    cached = _wheel_list_cache.get(cache_key)
    if cached is not None:
        return [v for v, _f in cached]
    versions: list[str] = []
    try:
        import json as _json

        payload = _json.loads(
            _http_get_text(f"https://pypi.org/pypi/{package}/json", timeout=30)
        )
        releases = payload.get("releases") or {}
        versions = [
            v for v in sorted(releases.keys(), key=_version_key, reverse=True)
            if any(f.get("filename", "").endswith(".whl") for f in releases.get(v) or [])
        ]
    except Exception:
        versions = []
    _wheel_list_cache[cache_key] = [(v, "") for v in versions]
    return versions


def _pypi_newest_for_interpreter(package: str) -> str:
    """Versión más nueva de ``package`` en PyPI instalable en este Python.

    Un wheel ``cp314`` aparece como "compatible" en el listado del índice
    aunque la app corra en 3.13 (el tag no siempre es el del intérprete),
    así que la versión de Python del candidato se confirma contra el
    ``requires_python`` real de la release. Sin esto se elegiría una torch
    publicada solo para 3.14 y pip fallaría con "no matching distribution".
    """
    current = (sys.version_info.major, sys.version_info.minor)
    for version in _pypi_versions_newest_first(package):
        floor = _pypi_release_requires_python(package, version)
        if floor is not None and current < floor:
            continue
        return version
    return ""


def _index_version_usable(package: str, version: str) -> bool:
    """True si la versión del índice es instalable en ESTE intérprete."""
    floor = _pypi_release_requires_python(package, version)
    if floor is None:
        return True  # sin dato: no se descarta lo que el índice ya filtró
    current = (sys.version_info.major, sys.version_info.minor)
    return current >= floor


def _newest_usable_in_index(index_url: str, package: str) -> str:
    """Candidatas del índice de más nueva a más antigua, la primera usable.

    Se recorren TODAS las versiones compatibles en vez de quedarse con la más
    nueva: si la última publicada es solo para un Python más moderno que el
    que ejecuta la app, se baja a la siguiente.
    """
    candidates: list[tuple[tuple, str]] = []
    for version, _filename in _index_wheel_list(index_url, package):
        public = _public_version(version)
        key = _version_key(public)
        if key:
            candidates.append((key, public))
    candidates.sort(reverse=True)
    for _key, public in candidates:
        if _index_version_usable(package, public):
            return public
    return ""


def _pypi_requires_version(package: str, version: str, dependency: str) -> str | None:
    """Versión exacta de ``dependency`` que exige ``package==version``.

    Es la fuente de verdad para emparejar el set: torchaudio/torchvision NO
    comparten número de versión con torch (torchaudio 2.11.0 va con torch
    2.13.0), pero su metadata sí declara qué torch necesitan. Preguntarlo
    evita el bug inverso al original: instalar un set que existe pero cuyas
    piezas no encajan entre sí.
    """
    cache_key = f"req|{package}|{version}|{dependency}"
    cached = _wheel_list_cache.get(cache_key)
    if cached is not None:
        return cached[0][0] if cached else None
    pinned: str | None = None
    try:
        import json as _json

        payload = _json.loads(
            _http_get_text(
                f"https://pypi.org/pypi/{package}/{version}/json", timeout=30
            )
        )
        for requirement in (payload.get("info") or {}).get("requires_dist") or []:
            text = str(requirement)
            if not text.lower().startswith(dependency):
                continue
            match = re.search(r"==\s*([0-9][0-9A-Za-z.+]*)", text)
            if match:
                pinned = match.group(1)
                break
    except Exception:
        pinned = None
    _wheel_list_cache[cache_key] = [(pinned, "")] if pinned else []
    return pinned


def _pypi_newest_for_torch(package: str, torch_version: str) -> str:
    """Versión más nueva de ``package`` (PyPI) que exige exactamente esa torch.

    Repliegue para los índices que no publican torchvision/torchaudio (caso
    real: los wheels CUDA de torch y torchvision viven en el índice de
    PyTorch, pero torchaudio se sirve desde PyPI). El emparejamiento NO puede
    ser por número de versión — torchaudio 2.11.0 acompaña a torch 2.13.0 —
    sino por la dependencia que la propia metadata declara.
    """
    current = (sys.version_info.major, sys.version_info.minor)
    for version in _pypi_versions_newest_first(package):
        floor = _pypi_release_requires_python(package, version)
        if floor is not None and current < floor:
            continue
        pinned = _pypi_requires_version(package, version, "torch")
        if pinned and _version_key(pinned) == _version_key(torch_version):
            return version
    return ""


def _plan_cache_path() -> str:
    return os.path.join(get_bin_dir(), "runtimes", _PLAN_CACHE_FILE)


def _read_plan_cache() -> dict | None:
    """Plan cacheado en disco, solo si sigue siendo válido para este entorno.

    Se descarta si cambió la versión de Python, la plataforma o la variante
    preferida por el hardware: un plan cacheado para otro intérprete sería
    precisamente el bug que estamos arreglando.
    """
    try:
        with open(_plan_cache_path(), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        if float(payload.get("ts", 0)) + _PLAN_CACHE_TTL_S < time.time():
            return None
    except (TypeError, ValueError):
        return None
    environment = payload.get("environment") or {}
    if environment.get("python") != _python_short():
        return None
    if environment.get("platform") != sys.platform:
        return None
    if environment.get("preferred") != _gpu_preferred_variant():
        return None
    return payload.get("plan") if isinstance(payload.get("plan"), dict) else None


def _write_plan_cache(plan: dict) -> None:
    """Guarda el plan resuelto (best effort: la caché nunca debe romper nada)."""
    payload = {
        "ts": time.time(),
        "environment": {
            "python": _python_short(),
            "platform": sys.platform,
            "preferred": _gpu_preferred_variant(),
        },
        "plan": plan,
    }
    try:
        os.makedirs(os.path.dirname(_plan_cache_path()), exist_ok=True)
        with open(_plan_cache_path(), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


def clear_pytorch_plan_cache() -> None:
    """Invalida la caché de resolución (memoria y disco)."""
    with _probe_cache_lock:
        _plan_cache["ts"] = 0.0
        _plan_cache["plan"] = None
    try:
        os.remove(_plan_cache_path())
    except OSError:
        pass
    _wheel_list_cache.clear()


def _plan_from_offline_defaults(reason: str) -> dict:
    """Plan sin red: el set conocido bueno sobre el índice preferido.

    Se usa cuando los índices no se pueden consultar (sin red, proxy caído).
    No es una adivinanza ciega como antes: la variante sale del hardware y el
    resultado de pip se valida igualmente antes de dar la instalación por
    buena, así que un plan equivocado se reporta con su causa real.
    """
    variant = _resolve_variant("auto")
    return {
        "variant": variant,
        "versions": {},
        "source": "offline",
        "note": reason,
        "attempts": [],
    }


def _pytorch_no_candidate_error(attempts: list[dict]) -> PytorchPlanError:
    """Error accionable cuando ningún candidato puede satisfacer la petición."""
    lines = [
        "No se encontró ninguna versión de PyTorch instalable para este equipo.",
        "",
        f"  Python   : {sys.version.split()[0]}",
        f"  Plataforma: {sys.platform}",
        "",
        "Se probaron estos canales de descarga:",
    ]
    for attempt in attempts:
        reason = attempt.get("reason") or "sin versión compatible"
        lines.append(f"  - {attempt.get('label', attempt.get('variant'))}: {reason}")
    lines += [
        "",
        "Qué puedes hacer:",
        "  1. Actualiza Python a 3.10 o superior (PyTorch ya no publica",
        "     wheels para versiones anteriores).",
        "  2. Si estás tras un proxy o firewall, revisa que puedas abrir",
        "     https://download.pytorch.org/whl/ y https://pypi.org/simple/.",
        "  3. Vuelve a intentarlo cuando recuperes la conexión: la lista de",
        "     versiones se consulta en vivo, no está fijada en el programa.",
    ]
    return PytorchPlanError("\n".join(lines), attempts)


def resolve_pytorch_plan(variant: str = "auto", use_cache: bool = True) -> dict:
    """Elige (variante, versiones) que DE VERDAD existen para este equipo.

    Devuelve ``{"variant", "versions": {"torch": "2.13.0", ...}, "source",
    "attempts"}``. Es el corazón del arreglo: en lugar de fijar
    ``torch==2.13.0`` sobre un índice elegido a ojo, se consultan los índices
    de PyTorch y se instala la pareja que tiene wheels para este CPython,
    esta plataforma y este driver.

    Preferencia (``auto``): la CUDA más nueva que el driver pueda cargar,
    empezando por la que el driver declara; se degrada a variantes CUDA más
    antiguas y, en último término, a CPU, en lugar de fallar.

    Lanza :class:`PytorchPlanError` (con la lista de intentos) si ningún
    candidato sirve. Si no hay red pero hay un plan cacheado válido, se usa;
    si no, se devuelve el plan offline con el set conocido bueno.
    """
    explicit = variant not in ("auto", "", None)
    requested = _resolve_variant(variant)

    # Suelo real de PyTorch: por debajo, ningún índice publica wheels y seguir
    # consultando solo alarga el error. Se informa con el mínimo exigido.
    minimum = _torch_min_python(_TORCH_MIN_PYTHON[0][0])
    if (sys.version_info.major, sys.version_info.minor) < minimum:
        raise PytorchPlanError(
            "PyTorch ya no publica wheels para Python "
            f"{_python_short()}. Se necesita Python {minimum[0]}.{minimum[1]} "
            "o superior para instalar los runtimes de IA.\n\n"
            "Actualiza tu Python (o usa el instalador de la aplicación, que "
            "incluye su propio intérprete) y vuelve a intentarlo.",
            [],
        )

    if not _platform_compatible_index():
        return _plan_from_offline_defaults(
            "macOS recibe PyTorch desde PyPI; se usa el set conocido bueno."
        )

    if use_cache and not explicit:
        with _probe_cache_lock:
            cached = _read_plan_cache()
        if cached is not None:
            logger.info(
                "plan de PyTorch desde caché: %s %s",
                cached.get("variant"), cached.get("versions"),
            )
            return cached

    if explicit:
        candidates = (
            [_CANDIDATES_BY_VARIANT[requested]]
            if requested in _CANDIDATES_BY_VARIANT
            else []
        )
    else:
        preferred = _gpu_preferred_variant()
        order = [c["variant"] for c in _VARIANT_CANDIDATES]
        start = order.index(preferred)
        # Desde la preferida hacia abajo y, si nada funciona, las CUDA que se
        # descartaron por driver (mejor intentarlo que rendirse) y CPU.
        below = [v for v in order[start:] if v != "cpu"]
        above = [v for v in order[:start] if v != "cpu"]
        candidates = [_CANDIDATES_BY_VARIANT[v] for v in (*below, *above)]
        if "cpu" in _CANDIDATES_BY_VARIANT:
            candidates.append(_CANDIDATES_BY_VARIANT["cpu"])

    attempts: list[dict] = []
    for candidate in candidates:
        index_url = candidate["url"]
        label = candidate["label"]
        variant_name = candidate["variant"]

        torch_version = _newest_usable_in_index(index_url, "torch")
        if not torch_version:
            # Índice sin wheels de torch para este intérprete: cu121/cu118
            # dejaron de publicar versiones hace tiempo y ESTE es el caso que
            # rompía la instalación (antes se elegía este índice a ojo).
            attempts.append({
                "variant": variant_name,
                "label": label,
                "reason": "no publica wheels de torch para Python "
                          f"{_python_short()}",
            })
            continue

        versions = {"torch": torch_version}
        detail = ""
        for package in ("torchvision", "torchaudio"):
            version = _newest_usable_in_index(index_url, package)
            if not version:
                # Algunos índices no sirven este paquete (torchaudio CUDA vive
                # en PyPI): se busca la versión que declare depender de esta
                # torch exacta, que es la relación real entre los paquetes.
                version = _pypi_newest_for_torch(package, torch_version)
                if version:
                    detail = f"{package} desde PyPI"
            if not version:
                break
            versions[package] = version

        if len(versions) != len(_PYTORCH_PACKAGES):
            missing = [p for p in _PYTORCH_PACKAGES if p not in versions]
            attempts.append({
                "variant": variant_name,
                "label": label,
                "reason": f"no hay {', '.join(missing)} compatible con torch "
                          f"{torch_version}",
            })
            continue

        plan = {
            "variant": variant_name,
            "versions": versions,
            "source": "index",
            "note": detail,
            "attempts": attempts,
        }
        if not explicit:
            _write_plan_cache(plan)
        logger.info(
            "PyTorch resuelto: %s %s (índice %s)",
            variant_name, versions, index_url,
        )
        return plan

    # Ningún candidato consultable. Antes de rendirse hay que distinguir dos
    # situaciones muy distintas: que la red funcione y de verdad no exista
    # nada instalable (error con la lista de intentos) o que los índices no
    # se hayan podido consultar (se prueba el set conocido bueno y pip dará
    # el error real si tampoco sirve).
    if attempts:
        try:
            report = run_connectivity_probe()
            network_ok = bool(report.get("ok"))
        except Exception:
            network_ok = False
        if network_ok:
            raise _pytorch_no_candidate_error(attempts)
        logger.warning(
            "no se pudieron consultar los índices de PyTorch; se usa el set "
            "conocido bueno (%s)", attempts,
        )
    return _plan_from_offline_defaults(
        "No se pudo consultar los índices de PyTorch; se usa el set conocido bueno."
    )


def pytorch_requirements(plan: dict) -> list[str]:
    """Requisitos PEP 508 exactos del plan (pin por paquete resuelto).

    Sin pin no habría garantía de que torch/torchvision/torchaudio salgan de
    la misma familia (una torchvision nueva puede arrastrar un torch viejo).
    """
    versions = plan.get("versions") or {}
    requirements = [
        f"{package}=={versions[package]}"
        for package in _PYTORCH_PACKAGES
        if versions.get(package)
    ]
    if len(requirements) == len(_PYTORCH_PACKAGES):
        return requirements
    # Plan offline sin versiones resueltas: set conocido bueno.
    return [_TORCH_PIN, _TORCHVISION_PIN, _TORCHAUDIO_PIN]


def describe_pytorch_plan(plan: dict) -> str:
    """Resumen humano del plan para la barra de progreso y los mensajes.

    El usuario debe ver QUÉ se está instalando (variante y versiones), no solo
    "Descargando PyTorch...": es la diferencia entre un error opaco y saber
    que se ha elegido otra CUDA porque la preferida no tenía wheels.
    """
    variant = plan.get("variant") or "cpu"
    label = (_CANDIDATES_BY_VARIANT.get(variant) or {}).get("label", variant)
    versions = plan.get("versions") or {}
    if versions:
        pieces = " + ".join(
            f"{pkg} {versions[pkg]}" for pkg in _PYTORCH_PACKAGES if versions.get(pkg)
        )
        return f"{label} · {pieces}"
    return label


def _build_pytorch_command(variant: str, plan: dict | None = None,
                           target: str | None = None) -> list[str]:
    """Construye el comando pip para instalar torch/torchvision/torchaudio.

    ``variant`` puede ser "auto": en ese caso se resuelve contra los índices
    reales (ver :func:`resolve_pytorch_plan`). ``plan`` permite reutilizar
    una resolución ya hecha (lo normal desde el instalador, para no consultar
    la red dos veces). ``target`` es el destino; por defecto el store real.
    """
    if plan is None:
        plan = resolve_pytorch_plan(variant)
    used_variant = plan.get("variant") or _resolve_variant(variant)
    index_url = _INDEX_URLS.get(used_variant, _INDEX_URLS["cpu"])

    # ``--index-url`` apunta SOLO al índice de CUDA/CPU de PyTorch, así que
    # hay que añadir PyPI como ``--extra-index-url``: ``onnx``, ``onnxscript``
    # y ``safetensors`` únicamente existen en PyPI y, sin él, pip falla con
    # "No matching distribution found for onnx".
    return [
        *_pip_command_prefix(), "install",
        "--isolated", "--no-cache-dir", "--disable-pip-version-check",
        "--ignore-installed", "--only-binary=:all:",
        "--target", target or RUNTIME_MODULES_DIR,
        "--no-warn-script-location",
        "--upgrade",
        "--index-url", index_url,
        "--extra-index-url", "https://pypi.org/simple",
        *pytorch_requirements(plan), "onnx", "onnxscript", "safetensors",
    ]


def _build_onnx_command(variant: str) -> list[str]:
    """Construye el comando pip para instalar onnxruntime.

    Se conserva como referencia del contrato (pip privado + flags aislantes)
    y para los tests; ``install_onnxruntime`` usa ahora la descarga directa
    del wheel (``_install_onnxruntime_from_wheel``), que es más robusta.
    """
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


def _cp_tag() -> str:
    """Tag de CPython exacto del intérprete en ejecución (``cp314``)."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _wheel_matches_cp(filename: str) -> bool:
    """True si el wheel es ABI-compatible con el intérprete en ejecución.

    Compara el tag de Python Y el tag ABI exactos (``cp314-cp314``). Así un
    wheel free-threaded (``cp314-cp314t``) NO coincide, porque sus últimos
    tres segmentos son ``python-abi-platform``.
    """
    stem = filename[:-4] if filename.endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 3:
        return False
    cp = _cp_tag()
    return parts[-3] == cp and parts[-2] == cp


def _wheel_matches_platform(filename: str) -> bool:
    """True si el wheel es para la plataforma/arquitectura actual."""
    stem = filename[:-4] if filename.endswith(".whl") else filename
    platform = stem.split("-")[-1]
    if sys.platform == "win32":
        return platform in ("win_amd64", "win32", "win_arm64")
    if sys.platform == "darwin":
        return platform.startswith("macosx")
    return platform.startswith("manylinux") or platform.startswith("linux")


def _onnx_wheel_info(variant: str) -> tuple[str, str, str]:
    """Resuelve ``(url, sha256, filename)`` del wheel oficial de ONNX Runtime.

    Consulta la API JSON de PyPI (lo mismo que usa pip, pero sin su resolución
    ni su pila SSL) y elige el wheel compatible con el intérprete y la
    plataforma actuales. Lanza ``RuntimeError`` con un mensaje accionable si
    no existe (p. ej. Python demasiado nuevo).
    """
    import requests

    variant = _resolve_variant(variant)
    pkg = "onnxruntime-gpu" if variant.startswith("cuda") else "onnxruntime"
    try:
        resp = requests.get(f"https://pypi.org/pypi/{pkg}/json", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"No se pudo consultar la versión disponible de {pkg}: {exc}"
        ) from exc
    for entry in data.get("urls", []):
        filename = entry.get("filename", "")
        if not filename.endswith(".whl"):
            continue
        if not _wheel_matches_cp(filename) or not _wheel_matches_platform(filename):
            continue
        sha256 = (entry.get("digests") or {}).get("sha256", "")
        return entry["url"], sha256, filename
    raise RuntimeError(
        f"No se encontró un wheel de {pkg} compatible con tu Python "
        f"({sys.version.split()[0]}) y plataforma ({sys.platform}). "
        "Prueba con otra versión de Python."
    )


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_resumable(
    url: str,
    dest: str,
    sha256_hex: str = "",
    progress_callback=None,
    start_pct: int = 0,
    end_pct: int = 100,
    message: str = "Descargando...",
    abort_event=None,
) -> str:
    """Descarga ``url`` a ``dest`` con reanudación (``Range``) y verificación.

    Escribe en ``dest + ".part"`` y, al terminar y verificar el SHA-256, lo
    mueve atómicamente a ``dest`` (``os.replace``). Si la descarga se
    interrumpe, la siguiente llamada reanuda desde lo ya descargado; un
    checksum incorrecto borra el ``.part`` para no arrastrar datos corruptos.
    Lanza ``RuntimeError("cancelled by user")`` si se setea ``abort_event``.
    """
    import requests

    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    try:
        with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
            if resp.status_code == 416:
                # El servidor dice que el rango pedido ya está completo; la
                # verificación de checksum de abajo lo confirma o lo rechaza.
                pass
            else:
                if have and resp.status_code != 206:
                    # El servidor ignoró Range: reiniciar desde cero.
                    have = 0
                    mode = "wb"
                total = have
                content_length = resp.headers.get("Content-Length", "")
                if content_length and content_length.isdigit():
                    total = have + int(content_length)
                with open(part, mode) as fh:
                    for chunk in resp.iter_content(1 << 20):
                        if abort_event is not None and abort_event.is_set():
                            raise RuntimeError("cancelled by user")
                        if not chunk:
                            continue
                        fh.write(chunk)
                        have += len(chunk)
                        if progress_callback and total:
                            frac = min(1.0, have / total)
                            pct = start_pct + (end_pct - start_pct) * frac
                            progress_callback(int(pct), message)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Fallo de red descargando el runtime: {exc}. "
            "Se guardó lo descargado; reintenta para continuar donde quedó."
        ) from exc

    if sha256_hex and _sha256_file(part).lower() != sha256_hex.lower():
        try:
            os.remove(part)
        except OSError:
            pass
        raise RuntimeError(
            "La descarga no pasó la verificación SHA-256 (archivo corrupto o "
            "incompleto). Vuelve a intentarlo."
        )

    os.replace(part, dest)
    return dest


def _install_onnxruntime_from_wheel(
    variant: str,
    report: dict,
    progress_callback,
    start_pct: int,
    end_pct: int,
    abort_event=None,
    timeout: float | None = None,
    target: str | None = None,
) -> None:
    """Descarga el wheel de ONNX Runtime (reanudable + SHA-256) y lo instala.

    La parte pesada (el wheel, 20–250 MB) se baja con ``_download_resumable``,
    de modo que un corte de red no reinicia la descarga y el contenido se
    verifica. pip instala el archivo LOCAL y solo resuelve las dependencias
    pequeñas (flatbuffers/numpy/packaging/protobuf), que fallan mucho menos.

    ``target`` permite instalar en el directorio de staging en lugar del store
    (el instalador hace el swap después de validar).
    """
    url, sha256_hex, filename = _onnx_wheel_info(variant)
    wheel_dir = tempfile.mkdtemp(prefix=".drag-onnx-")
    try:
        download_end = start_pct + int((end_pct - start_pct) * 0.7)
        wheel_path = _download_resumable(
            url,
            os.path.join(wheel_dir, filename),
            sha256_hex=sha256_hex,
            progress_callback=progress_callback,
            start_pct=start_pct,
            end_pct=download_end,
            message="Descargando ONNX Runtime...",
            abort_event=abort_event,
        )
        _run_pip_checked(
            report,
            [
                *_pip_command_prefix(), "install",
                "--isolated", "--no-cache-dir", "--disable-pip-version-check",
                "--ignore-installed", "--only-binary=:all:",
                "--target", target or RUNTIME_MODULES_DIR,
                "--no-warn-script-location",
                "--upgrade",
                wheel_path,
            ],
            progress_callback,
            start_pct=download_end,
            end_pct=end_pct,
            message="Instalando ONNX Runtime...",
            abort_event=abort_event,
            timeout=timeout,
        )
    finally:
        _rmtree_quiet(wheel_dir)


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
    cpu/cudaXXX. Los wheels CUDA de torch pesan varios GB instalados (más la
    descarga comprimida), frente a unas centenas de MB en CPU; ONNX Runtime
    es mucho más ligero. Se incluye margen para torchvision/torchaudio/onnx y
    futuros updates, de modo que el aviso dispare ANTES de que pip falle a
    mitad de una descarga de gigabytes.

    Además, la instalación usa un directorio temporal de staging en la MISMA
    unidad (se renombra al store, no se copia), así que el pico de disco es
    el nuevo contenido + el antiguo. Ese pico lo cubre el margen de la tabla;
    ``_check_disk_space`` exige descarga+instalado cuando ambas rutas están en
    el mismo dispositivo.
    """
    if kind == "pytorch":
        if variant == "cpu":
            return (1_200_000_000, 800_000_000)
        # Las CUDA recientes empaquetan el runtime (cuDNN, cuBLAS, ...) y
        # ocupan más que las antiguas. Valores medidos en una instalación real
        # de torch 2.14+cu130 (≈3.3 GB instalados, ≈2.5 GB descargados) más
        # margen para torchvision/torchaudio/onnx y actualizaciones.
        if variant == "cuda130":
            return (5_000_000_000, 3_000_000_000)
        if variant == "cuda126":
            return (4_500_000_000, 2_800_000_000)
        return (4_000_000_000, 2_500_000_000)
    if kind == "onnxruntime":
        if variant != "cpu":
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
        "Por favor, usa Python 3.10–3.14 para que la descarga funcione, "
        "o instala torch/onnxruntime manualmente en este entorno."
    )


# ---------------------------------------------------------------------------
# Instalación por staging + swap (nunca dejar el store a medias)
#
# El método antiguo instalaba con ``pip install --target <store> --upgrade
# --ignore-installed`` directamente sobre el store en uso. Consecuencias
# reales observadas en un store de producción:
#
#   * pip no retira los ``*.dist-info`` de la versión anterior, así que el
#     store acumulaba metadatos duplicados (numpy 2.4.4 Y 2.5.2, protobuf
#     7.35.1 Y 7.36.0, packaging 26.2 Y 26.3...) y ``importlib.metadata``
#     podía reportar la versión equivocada.
#   * una instalación abortada dejaba archivos sueltos mezclados con la
#     versión vieja: la causa del clásico "instaló pero no importa".
#   * no había forma de validar el resultado ANTES de tocar lo que ya
#     funcionaba.
#
# Ahora pip instala en un directorio temporal limpio en la MISMA unidad, se
# valida ahí (subproceso aislado) y solo entonces se sustituye el contenido
# del store. Si algo falla, se restaura lo anterior: el usuario nunca se
# queda peor que antes de intentarlo.
# ---------------------------------------------------------------------------

# Subcarpetas de trabajo del staging. Viven en bin/runtimes/ (fuera del
# store) para que un swap nunca se confunda con contenido instalado.
_STAGING_DIR_NAME = ".staging"
_ASIDE_DIR_NAME = ".previous"


def _install_base_dir() -> str:
    return os.path.dirname(RUNTIME_MODULES_DIR.rstrip("\\/")) or RUNTIME_MODULES_DIR


def _staging_dir() -> str:
    return os.path.join(_install_base_dir(), _STAGING_DIR_NAME)


def _aside_dir() -> str:
    return os.path.join(_install_base_dir(), _ASIDE_DIR_NAME)


def _store_entries(path: str) -> set[str]:
    """Nombres de primer nivel del store (set vacío si no existe)."""
    try:
        return set(os.listdir(path))
    except OSError:
        return set()


def _move_store_to_aside() -> None:
    """Aparta el store actual a ``bin/runtimes/.previous`` (un solo paso).

    Se hace UNA vez por ciclo de instalación, antes de la primera llamada, y
    sirve a la vez de copia de seguridad para restaurar y de estado antiguo
    del que reciclar los paquetes que la nueva instalación no toca (p. ej.
    instalar PyTorch conserva el onnxruntime ya presente en vez de borrarlo).

    Bloqueado por Windows: si el store está en uso, el ``rename`` puede
    fallar; en ese caso se deja el store donde está (el guard de versión
    cubrirá los wheels incompatibles) y se sigue con staging vacío.
    """
    if _store_entries(RUNTIME_MODULES_DIR):
        _rmtree_quiet(_aside_dir())
        try:
            os.makedirs(_install_base_dir(), exist_ok=True)
            os.rename(RUNTIME_MODULES_DIR, _aside_dir())
            return
        except OSError as exc:
            logger.warning("no se pudo apartar el store para el swap: %s", exc)
    try:
        os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    except Exception:
        pass


def _commit_staging(replace_prefixes: tuple[str, ...]) -> None:
    """Sustituye en el store los paquetes del prefijo dado por lo staged.

    Estrategia: se borran del store los paquetes que la nueva instalación
    reemplaza (el paquete, ``<paquete>.libs`` y sus ``*.dist-info``), se
    reciclan del estado anterior los que la nueva instalación no incluye, y
    se mueven los staged al store. Al final el store es autoconsistente y sin
    metadatos duplicados, porque todo lo que entra viene de una instalación
    limpia hecha de una vez.
    """
    staging = _staging_dir()
    if not os.path.isdir(staging):
        raise RuntimeError(
            "La instalación no generó archivos: el directorio temporal de "
            f"staging no existe ({staging})."
        )
    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    new_entries = _store_entries(staging)

    # 1. Fuera del store lo que se reemplaza (incluye dist-info huérfanos).
    for entry in list(_store_entries(RUNTIME_MODULES_DIR)):
        if _entry_matches_prefixes(entry, replace_prefixes):
            _rmtree_quiet(os.path.join(RUNTIME_MODULES_DIR, entry))

    # 2. Reciclar del estado apartado lo que la nueva instalación no trae.
    #    Se descartan los wheels compilados para OTRA versión de Python: son
    #    justo la basura que rompía la detección del store.
    aside = _aside_dir()
    for entry in list(_store_entries(aside)):
        if entry in new_entries or _entry_matches_prefixes(entry, replace_prefixes):
            continue
        if entry == "__pycache__":
            continue
        if entry.endswith(".dist-info") and _dist_info_is_foreign(entry):
            continue
        source = os.path.join(aside, entry)
        target = os.path.join(RUNTIME_MODULES_DIR, entry)
        if os.path.exists(target):
            continue
        try:
            os.rename(source, target)
        except OSError:
            try:
                shutil.move(source, target)
            except Exception:
                logger.warning("no se pudo reciclar %s del store anterior", entry)

    # 3. Mover lo staged dentro (rename, misma unidad: instantáneo).
    for entry in sorted(new_entries):
        source = os.path.join(staging, entry)
        target = os.path.join(RUNTIME_MODULES_DIR, entry)
        _rmtree_quiet(target)
        try:
            os.rename(source, target)
        except OSError:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
                _rmtree_quiet(source)
            else:
                shutil.copy2(source, target)
                _rmtree_quiet(source)
    _rmtree_quiet(staging)
    _schedule_aside_cleanup()


def _entry_matches_prefixes(entry: str, prefixes: tuple[str, ...]) -> bool:
    """True si una entrada del store pertenece a uno de los paquetes dados.

    Empareja el paquete (``torch``), sus datos compilados (``torch.libs``) y
    TODOS sus metadatos (``torch-2.13.0+cu130.dist-info``), que es justo la
    basura que ``pip --target --upgrade`` dejaba atrás.

    Las comparaciones normalizan ``-`` y ``_`` porque el nombre canónico de un
    paquete no coincide con el del directorio ni con el de sus metadatos:
    ``onnxruntime_gpu-1.30.0.dist-info`` declara ``Name: onnxruntime-gpu`` y
    ``onnx_ir-1.0.0.dist-info`` declara ``onnx-ir``. Comparando solo el prefijo
    del nombre de archivo, esos metadatos sobrevivían a la desinstalación y la
    app seguía reportando el runtime como instalado.
    """
    candidates = _dist_info_name_variants(entry) if entry.endswith(".dist-info") else ()
    for prefix in prefixes:
        if entry == prefix:
            return True
        if entry.startswith(prefix + "."):
            return True
        if candidates and prefix.replace("-", "_") in candidates:
            return True
    return False


def _dist_info_name(dist_info_dirname: str) -> str:
    """Nombre de paquete declarado por un ``*.dist-info`` (normalizado).

    Se prefiere el ``Name`` real de la metadata (fuente autoritativa); si no se
    puede leer, se cae al prefijo del nombre del directorio.
    """
    variants = _dist_info_name_variants(dist_info_dirname)
    if variants:
        return sorted(variants, key=len, reverse=True)[0].replace("_", "-")
    base = dist_info_dirname[: -len(".dist-info")]
    return base.split("-", 1)[0].replace("_", "-").lower()


def _dist_info_name_variants(dist_info_dirname: str) -> frozenset[str]:
    """Posibles nombres de paquete de un ``*.dist-info``, normalizados.

    Devuelve todas las formas con ``_`` (la convención que usan los prefijos
    internos) para poder comparar sin adivinar. Incluye el nombre del
    directorio truncado en el último guion antes de la versión, que es lo que
    permite reconocer metadatos huérfanos cuyo ``METADATA`` falta.
    """
    base = dist_info_dirname[: -len(".dist-info")]
    variants: set[str] = set()

    metadata_name = ""
    try:
        metadata_path = os.path.join(RUNTIME_MODULES_DIR, dist_info_dirname, "METADATA")
        with open(metadata_path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                if line.lower().startswith("name:"):
                    metadata_name = line.split(":", 1)[1].strip()
                    break
                if not line.strip():
                    break
    except OSError:
        metadata_name = ""

    if metadata_name:
        variants.add(metadata_name.replace("-", "_").lower())
    # Prefijo del directorio: "onnxruntime_gpu-1.30.0" -> "onnxruntime_gpu".
    head = base.split("-", 1)[0]
    if head:
        variants.add(head.replace("-", "_").lower())
        variants.add(head.replace("_", "-").replace("-", "_").lower())
    return frozenset(variants)


def _schedule_aside_cleanup() -> None:
    """Programa el borrado del store anterior para el próximo arranque.

    No se borra ahora a propósito: si este proceso ya importó torch desde la
    ruta vieja, Windows mantiene sus DLL mapeadas y borrarlas en caliente
    dejaría el runtime a medias. ``process_pending_cleanup()`` lo completa al
    arrancar, antes de importar nada.
    """
    aside = _aside_dir()
    if not os.path.isdir(aside):
        return
    try:
        from src.dependencies import _rmtree_force

        _rmtree_force(aside)
    except Exception:
        _rmtree_quiet(aside)


def cleanup_install_leftovers() -> int:
    """Borra restos de instalaciones anteriores (llamado al arrancar).

    Un cierre a mitad de swap puede dejar ``.staging`` (instalación sin
    validar) o ``.previous`` (store anterior). Ninguno debe sobrevivir: el
    staging a medias no debe poder confundirse con el store y el anterior ya
    no se necesita. Devuelve cuántos directorios limpió.

    Guarda de seguridad: ``.previous`` SOLO se borra si parece un store
    instalado de verdad (lo que deja un swap). Si un test o un script ha
    redirigido ``RUNTIME_MODULES_DIR`` y ha dejado ahí contenido ajeno a un
    swap, se respeta: el árbol ``bin/`` no está en git y borrar por error
    significa perder varios GB sin vuelta atrás (pasó durante el desarrollo).
    """
    cleaned = 0

    staging = _staging_dir()
    if os.path.isdir(staging):
        _rmtree_quiet(staging)
        if not os.path.isdir(staging):
            cleaned += 1

    aside = _aside_dir()
    if os.path.isdir(aside) and _looks_like_store(aside):
        try:
            from src.dependencies import _rmtree_force

            _rmtree_force(aside)
        except Exception:
            _rmtree_quiet(aside)
        if not os.path.isdir(aside):
            cleaned += 1

    return cleaned


def _looks_like_store(path: str) -> bool:
    """True si ``path`` parece un store de runtimes con paquetes instalados.

    Exige evidencia fuerte: un directorio de paquete conocido (``torch``,
    ``onnxruntime``, ...) o metadatos de wheel (``*.dist-info``). Un
    directorio con contenido arbitrario no pasa, y eso es deliberado: es la
    diferencia entre limpiar un swap de verdad y borrar datos que un test o un
    usuario hayan dejado ahí (el árbol ``bin/`` no está en git).
    """
    entries = _store_entries(path)
    if not entries:
        return False
    if any(
        os.path.isdir(os.path.join(path, name))
        for name in ("torch", "onnxruntime", "torchvision", "torchaudio")
    ):
        return True
    return any(entry.endswith(".dist-info") for entry in entries)


def _merge_missing_from_base(destination: str) -> None:
    """Copia al destino los paquetes de ``.previous`` que le falten.

    Necesario para validar en staging un paquete cuyas dependencias ya
    existían en el store (p. ej. PyTorch nuevo con numpy del store anterior):
    el subproceso de validación debe ver el mismo conjunto que verá la app
    tras el swap.
    """
    aside = _aside_dir()
    if not os.path.isdir(aside):
        return
    existing = _store_entries(destination)
    for entry in _store_entries(aside):
        if entry in existing or entry == "__pycache__":
            continue
        source = os.path.join(aside, entry)
        target = os.path.join(destination, entry)
        try:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
        except Exception:
            logger.warning("no se pudo preparar %s para la validación", entry)


def _validate_package_in_dir(module_name: str, target: str,
                             timeout: float = 180.0) -> tuple[bool, str]:
    """Verifica en un subproceso aislado que ``module_name`` importe de ``target``.

    Se usa antes del swap (validar staging) y después (confirmar el store).
    Nombre de módulo distinto por llamada a propósito: la sonda valida SIEMPRE
    el directorio que se le pasa y registra su origen, de modo que un éxito
    con el store viejo no pueda confundirse con un éxito con el staging.
    """
    python = get_embed_python_path()
    if python is None:
        if getattr(sys, "frozen", False):
            return False, (
                "No se encontró el Python privado (embed-python) junto al "
                "ejecutable: no se puede validar el runtime. Reinstala la app."
            )
        python = sys.executable
    try:
        proc = subprocess.run(
            [python, "-c", _IMPORT_PROBE_SOURCE, target, module_name],
            capture_output=True, text=True, timeout=timeout,
            env=_isolated_pip_env(),
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"La validación de '{module_name}' agotó {int(timeout)}s (el "
            "import se colgó; posible antivirus examinando las DLL recién "
            "extraídas)."
        )
    except OSError as exc:
        return False, f"No se pudo ejecutar la validación ({python}): {exc}"
    if proc.returncode == 0 and "DRAGGY-IMPORT-OK" in (proc.stdout or ""):
        return True, ""
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return False, output or f"(sin salida; exit code {proc.returncode})"


def _staged_install(
    report: dict,
    *,
    packages: tuple[str, ...],
    requirements: list[str],
    index_url: str,
    kind: str,
    variant: str,
    progress_callback,
    start_pct: int,
    end_pct: int,
    message: str,
    abort_event=None,
    timeout: float | None = None,
) -> None:
    """Instala en staging, valida y sustituye en el store los paquetes dados.

    ``packages`` son los nombres de paquete cuyas entradas del store se
    reemplazan al hacer el swap. ``requirements`` es lo que se pasa a pip.

    Antes del swap se valida en un subproceso aislado que CADA paquete se
    importe de verdad desde el staging: es la garantía de que el store nunca
    queda con una instalación que pip dio por buena pero no es usable (el
    caso clásico: "instaló pero no importa").
    """
    staging = _staging_dir()
    _rmtree_quiet(staging)
    os.makedirs(staging, exist_ok=True)
    _check_disk_space(kind, variant)

    command = [
        *_pip_command_prefix(), "install",
        "--isolated", "--no-cache-dir", "--disable-pip-version-check",
        "--ignore-installed", "--only-binary=:all:",
        "--target", staging,
        "--no-warn-script-location",
        "--upgrade",
        "--index-url", index_url,
        "--extra-index-url", "https://pypi.org/simple",
        *requirements,
    ]
    _run_pip_checked(
        report, command, progress_callback,
        start_pct=start_pct, end_pct=end_pct, message=message,
        abort_event=abort_event, timeout=timeout,
    )

    _merge_missing_from_base(staging)
    for module_name in packages:
        ok, detail = _validate_package_in_dir(module_name, staging)
        if not ok:
            _rmtree_quiet(staging)
            raise RuntimeError(
                f"La instalación de '{module_name}' se descargó pero no se "
                "puede importar; se ha conservado la versión anterior.\n\n"
                f"Detalles del import:\n{detail}\n\n"
                f"staging={staging}"
            )

    _commit_staging(packages)
    _invalidate_store_probe_cache()


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
    únicamente su paquete, evitando arrastrar varios GB de torch cuando el
    usuario solo quiere ONNX (y viceversa).

    La pareja (variante, versión) NO se fija a mano: la resuelve
    :func:`resolve_pytorch_plan` consultando los índices de PyTorch, porque un
    pin estático sobre un índice elegido por heurística es exactamente lo que
    hacía fallar la instalación con "Could not find a version that satisfies
    the requirement torch==X".

    ``abort_event`` cancela la descarga; ``timeout`` acota el tiempo total.
    """
    if not _is_python_version_supported():
        _raise_unsupported_python()
    report = _preflight_connectivity()
    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    # Purga previa: si el store tiene wheels de otra versión de Python
    # (cp314 vs cp311), quitarlos para que pip resuelva limpio.
    _purge_foreign_version_wheels()

    plan = resolve_pytorch_plan(variant)
    used_variant = plan.get("variant") or _resolve_variant(variant)
    # El preflight de disco va ANTES de apartar el store: si no hay espacio, la
    # instalación debe abortar sin haber tocado nada. (Antes se apartaba el
    # store primero y el abort lo dejaba movido de sitio: el runtime
    # "desaparecía" de su carpeta sin que se hubiera instalado nada.)
    _check_disk_space("pytorch", used_variant)
    _move_store_to_aside()

    # El plan debe fijar las tres piezas para que no se mezclen familias
    # (una torchvision nueva puede arrastrar un torch viejo).
    # ``safetensors`` va con PyTorch: es lo que hace falta para cargar los
    # modelos .safetensors (OmniSR, RIFE, FILM...) con ``safetensors.torch``.
    requirements = [*pytorch_requirements(plan), "onnx", "onnxscript",
                    "safetensors"]
    _staged_install(
        report,
        packages=("torch", "torchvision", "torchaudio", "onnx", "onnxscript",
                  "safetensors"),
        requirements=requirements,
        index_url=_INDEX_URLS.get(used_variant, _INDEX_URLS["cpu"]),
        kind="pytorch",
        variant=used_variant,
        progress_callback=progress_callback,
        start_pct=5,
        end_pct=99,
        message=f"Descargando PyTorch ({describe_pytorch_plan(plan)})...",
        abort_event=abort_event,
        timeout=timeout,
    )

    # Purga posterior: cubre wheels ajenos reciclados del store anterior.
    _purge_foreign_version_wheels()
    _invalidate_store_probe_cache("torch")
    if progress_callback:
        progress_callback(100, "PyTorch listo")
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    ok, detail = _post_install_import_check("torch")
    if not ok:
        hint = _vc_redist_hint(detail)
        # Fallback automático: si la variante CUDA falló por DLL y no era CPU,
        # reintentar con CPU puede salvar la instalación en equipos sin
        # toolchain CUDA. No se aplica si el error es de spec/namespace (pip
        # corrupto) para evitar bucle infinito.
        if used_variant != "cpu" and ("dll load failed" in detail.lower() or "vcruntime" in detail.lower()):
            # No reintento recursivo aquí (evitar download gigante de nuevo
            # sin consentimiento). Se informa al usuario para que elija CPU.
            hint += (
                "\nPrueba a instalar la variante CPU desde el panel "
                "(Dependencies → PyTorch → CPU) si no tienes GPU NVIDIA."
            )
        raise RuntimeError(
            "PyTorch installation finished but the local runtime could not "
            f"be imported. The installation may be incomplete or incompatible.{hint}\n\n"
            f"Detalles del import:\n{detail}\n\n"
            f"store={RUNTIME_MODULES_DIR}\n"
            f"python={sys.version.split()[0]} ({'frozen' if getattr(sys, 'frozen', False) else 'dev'}) "
            f"variant={used_variant}"
        )
    return used_variant


def _install_onnx_stack_into_staging(
    report: dict,
    staging: str,
    progress_callback,
    start_pct: int,
    end_pct: int,
    abort_event=None,
    timeout: float | None = None,
) -> None:
    """Instala ``onnx``/``onnxscript`` (y sus dependencias) en el staging.

    Es lo que usa ``torch.onnx.export`` para convertir checkpoints ``.pth`` a
    ONNX. Va aparte del wheel de ONNX Runtime porque vive en PyPI, no en el
    índice de PyTorch, y porque el instalador de PyTorch ya los instala: así
    instalar/desinstalar cualquiera de los dos runtimes es simétrico.

    Un fallo aquí NO aborta la instalación del runtime: ONNX Runtime sirve
    para ejecutar modelos .onnx aunque no se pueda convertir. Se registra y se
    sigue; el usuario verá el aviso correspondiente al intentar convertir.
    """
    try:
        _run_pip_checked(
            report,
            [
                *_pip_command_prefix(), "install",
                "--isolated", "--no-cache-dir", "--disable-pip-version-check",
                "--ignore-installed", "--only-binary=:all:",
                "--target", staging,
                "--no-warn-script-location",
                "--upgrade",
                "onnx", "onnxscript",
            ],
            progress_callback,
            start_pct=start_pct,
            end_pct=end_pct,
            message="Instalando el soporte de conversión ONNX...",
            abort_event=abort_event,
            timeout=timeout,
        )
    except RuntimeError as exc:
        if str(exc) == "cancelled by user":
            raise
        logger.warning("no se pudo instalar el stack onnx en el staging: %s", exc)


def install_onnxruntime(
    variant: str = "auto",
    progress_callback=None,
    parent=None,
    abort_event=None,
    timeout: float | None = None,
    staged: bool = True,
) -> str:
    """Instala ONNX Runtime (onnxruntime / onnxruntime-gpu) en
    ``bin/runtimes/ai_modules``.

    Instala además el stack ``onnx``/``onnxscript``: es lo que necesita
    ``torch.onnx.export`` para convertir un ``.pth`` a ONNX, y desinstalar
    ONNX Runtime se lo lleva (sin ninguno de los dos runtimes no sirve para
    nada). Si solo se instalara el runtime, el usuario que desinstala y
    reinstala se encontraría con que ya no puede convertir modelos, sin ningún
    aviso. Como el desinstalador los trata como parte del grupo, el instalador
    debe reponerlos: instalar y desinstalar tienen que ser simétricos.

    Usa el mismo esquema de staging que PyTorch: descarga y valida en un
    directorio limpio y solo entonces sustituye el store, conservando lo que
    ya estaba (torch, numpy, ...).

    ``abort_event`` cancela la descarga; ``timeout`` acota el tiempo total.
    """
    if not _is_python_version_supported():
        _raise_unsupported_python()
    report = _preflight_connectivity()
    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    _purge_foreign_version_wheels()
    used_variant = _resolve_variant(variant)

    # Paquete real que se instala (gpu/cpu) y, por tanto, el que se valida y
    # cuyas entradas del store se reemplazan.
    backend = "onnxruntime_gpu" if used_variant != "cpu" else "onnxruntime"
    replaces = (backend, "onnxruntime", "onnxruntime-gpu", "onnx", "onnxscript", "onnx_ir")
    onnx_stack = ("onnx", "onnxscript", "onnx_ir")

    if staged:
        staging = _staging_dir()
        _rmtree_quiet(staging)
        os.makedirs(staging, exist_ok=True)
        _check_disk_space("onnxruntime", used_variant)
        # Antes de apartar el store: un abort por disco no debe mover nada.
        _move_store_to_aside()
        # El runtime por su wheel oficial (descarga reanudable + SHA-256)...
        _install_onnxruntime_from_wheel(
            used_variant, report, progress_callback,
            start_pct=5, end_pct=90,
            abort_event=abort_event, timeout=timeout, target=staging,
        )
        # ...y el stack onnx desde PyPI, que es donde vive.
        _install_onnx_stack_into_staging(
            report, staging, progress_callback,
            start_pct=90, end_pct=97,
            abort_event=abort_event, timeout=timeout,
        )
        # Las dependencias que la instalación no trae se reciclan del store
        # anterior para que la validación vea el mismo conjunto que verá la app.
        _merge_missing_from_base(staging)
        for module_name in ("onnxruntime", *onnx_stack):
            ok, detail = _validate_package_in_dir(module_name, staging)
            if not ok:
                _rmtree_quiet(staging)
                raise RuntimeError(
                    f"'{module_name}' se descargó pero no se puede importar; "
                    "se ha conservado la versión anterior.\n\n"
                    f"Detalles del import:\n{detail}\n\n"
                    f"staging={staging}"
                )
        _commit_staging(replaces)
    else:
        _check_disk_space("onnxruntime", used_variant)
        _install_onnxruntime_from_wheel(
            used_variant, report, progress_callback,
            start_pct=5, end_pct=99,
            abort_event=abort_event, timeout=timeout,
        )

    _purge_foreign_version_wheels()
    _invalidate_store_probe_cache("onnxruntime")
    if progress_callback:
        progress_callback(100, "ONNX Runtime listo")
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    ok, detail = _post_install_import_check("onnxruntime")
    if not ok:
        hint = _vc_redist_hint(detail)
        raise RuntimeError(
            "ONNX Runtime installation finished but the local runtime could "
            f"not be imported. The installation may be incomplete or incompatible.{hint}\n\n"
            f"Detalles del import:\n{detail}\n\n"
            f"store={RUNTIME_MODULES_DIR}\n"
            f"python={sys.version.split()[0]} ({'frozen' if getattr(sys, 'frozen', False) else 'dev'}) "
            f"variant={used_variant}"
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

    # Una sola resolución para todo el ciclo: el índice, la variante y las
    # versiones se consultan una vez y se reutilizan en las dos fases.
    plan = resolve_pytorch_plan(variant)
    used_variant = plan.get("variant") or _resolve_variant(variant)

    # Umbral combinado: el pico de disco lo marcan los wheels CUDA de torch.
    # Antes de apartar el store, para que un abort por falta de espacio no
    # mueva la instalación existente.
    _check_disk_space("all", used_variant)
    _move_store_to_aside()

    # 1. PyTorch (la parte más grande)
    _staged_install(
        report,
        packages=("torch", "torchvision", "torchaudio", "onnx", "onnxscript",
                  "safetensors"),
        requirements=[*pytorch_requirements(plan), "onnx", "onnxscript",
                      "safetensors"],
        index_url=_INDEX_URLS.get(used_variant, _INDEX_URLS["cpu"]),
        kind="all",
        variant=used_variant,
        progress_callback=progress_callback,
        start_pct=5,
        end_pct=85,
        message=f"Descargando PyTorch ({describe_pytorch_plan(plan)})...",
        abort_event=abort_event,
        timeout=timeout,
    )

    # 2. ONNX Runtime (wheel oficial descargado de forma reanudable) sobre el
    #    store ya actualizado por la fase anterior.
    backend = "onnxruntime_gpu" if used_variant != "cpu" else "onnxruntime"
    staging = _staging_dir()
    _rmtree_quiet(staging)
    os.makedirs(staging, exist_ok=True)
    _install_onnxruntime_from_wheel(
        used_variant, report, progress_callback,
        start_pct=86, end_pct=99,
        abort_event=abort_event, timeout=timeout, target=staging,
    )
    _merge_missing_from_base(staging)
    ok, detail = _validate_package_in_dir("onnxruntime", staging)
    if not ok:
        _rmtree_quiet(staging)
        raise RuntimeError(
            "ONNX Runtime se descargó pero no se puede importar; se ha "
            "conservado la versión anterior.\n\n"
            f"Detalles del import:\n{detail}\n\nstaging={staging}"
        )
    _commit_staging((backend, "onnxruntime", "onnxruntime-gpu"))

    _purge_foreign_version_wheels()

    if progress_callback:
        progress_callback(100, "Runtimes de IA listos")

    # Después de instalar, reinjectar por si acaso y verificar ambos
    # paquetes. Pip puede terminar con código 0 aunque un wheel nativo no
    # sea importable en la versión/arquitectura actual.
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
    missing = []
    details: list[str] = []
    for name in ("torch", "onnxruntime"):
        ok, detail = _post_install_import_check(name)
        if not ok:
            missing.append(name)
            details.append(f"--- {name} ---\n{detail}\n{_vc_redist_hint(detail)}")
    if missing:
        raise RuntimeError(
            "Runtime installation finished but these local packages could "
            f"not be imported: {', '.join(missing)}\n\n"
            + "\n\n".join(details)
            + "\n\n"
            f"store={RUNTIME_MODULES_DIR}\n"
            f"python={sys.version.split()[0]} ({'frozen' if getattr(sys, 'frozen', False) else 'dev'}) "
            f"variant={used_variant}"
        )
    return used_variant


# ---------------------------------------------------------------------------
# Per-package status helpers
# ---------------------------------------------------------------------------


def _version_of(module_name: str, extra_names: tuple[str, ...] = ()) -> str:
    """Obtiene la versión exclusivamente desde dist-info del store local.

    ``extra_names`` cubre los paquetes cuyo nombre de DISTRIBUCIÓN no coincide
    con el de importación: ``import onnxruntime`` puede venir del wheel
    ``onnxruntime-gpu`` (y el driver CUDA decide cuál de los dos se instaló),
    así que sin esta lista la UI mostraba la versión como "—".
    """
    candidates = {module_name.replace("_", "-").lower()}
    candidates.update(name.replace("_", "-").lower() for name in extra_names)
    candidates.update(name.replace("_", "").lower() for name in extra_names)
    try:
        import importlib.metadata as _meta

        for dist in _meta.distributions(path=[RUNTIME_MODULES_DIR]):
            name = (dist.metadata.get("Name") or "").replace("_", "-").lower()
            if name in candidates or name.replace("-", "") in candidates:
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
    if not installed and _store_has_package_dir("torch") and _can_probe_subprocess():
        # El store contiene torch pero este proceso no consigue importarlo
        # (import fallido previo que dejó submódulos huérfanos, antivirus,
        # etc.). El veredicto lo da un subproceso limpio: si allí importa,
        # el store es válido y NO debe re-descargarse ~3 GB en cada arranque
        # (antes el runtime quedaba "not ready" eterno con reinstalaciones
        # en bucle a pesar de tener el paquete bien instalado).
        try:
            ok, _probe_detail = _subprocess_import_ok("torch")
        except Exception:
            ok = False
        if ok:
            logger.warning(
                "torch no importa in-process pero sí en subproceso: store "
                "válido, no se reinstala"
            )
            installed = True
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
    if not installed and _store_has_package_dir("onnxruntime") and _can_probe_subprocess():
        # Mismo guard que torch: el store manda sobre el proceso actual.
        try:
            ok, _probe_detail = _subprocess_import_ok("onnxruntime")
        except Exception:
            ok = False
        if ok:
            logger.warning(
                "onnxruntime no importa in-process pero sí en subproceso: "
                "store válido, no se reinstala"
            )
            installed = True
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
        "version": _version_of("onnxruntime", ("onnxruntime-gpu", "onnxruntime_gpu")) if installed else "—",
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


def _runtime_group_prefixes(dep_id: str) -> tuple[str, ...]:
    """Entradas del store que pertenecen a un runtime concreto.

    El store es COMPARTIDO por PyTorch y ONNX Runtime, y cada uno arrastra
    además sus propios satélites. Al desinstalar uno solo se deben ir sus
    paquetes y metadatos, y **nada más**: borrar el store entero se llevaba por
    delante el otro runtime y todas las dependencias compartidas (numpy,
    protobuf, ...) sin que el usuario lo hubiera pedido.

    ``onnx``/``onnxscript``/``onnx_ir`` no valen para nada sin PyTorch
    (``torch.onnx.export`` es su único consumidor) y sin ONNX Runtime no hay
    quien ejecute el resultado, así que se van con CUALQUIERA de los dos: es
    preferible quedarse corto de paquetes (la siguiente instalación los trae)
    que dejar cientos de MB huérfanos.

    ``safetensors`` se va SOLO con PyTorch: es la librería que lee los pesos
    ``.safetensors`` (OmniSR, RIFE, FILM) y sin torch no sirve para nada.
    """
    onnx_stack = ("onnx", "onnxscript", "onnx_ir")
    groups = {
        "torch": ("torch", "torchvision", "torchaudio", "torchgen",
                  "functorch", "safetensors", *onnx_stack),
        "onnxruntime": ("onnxruntime", "onnxruntime_gpu", *onnx_stack),
    }
    return groups.get(dep_id, ())


def _uninstall_runtime_group(dep_id: str) -> tuple[bool, list[str]]:
    """Borra del store los paquetes de un runtime. Devuelve ``(completo, restos)``.

    ``completo`` es False cuando Windows mantiene algún archivo bloqueado (el
    runtime ya está importado en este proceso): esos restos se programan para
    borrarse en el próximo arranque, antes de importar nada.

    Las dependencias compartidas (numpy, protobuf, packaging...) NO se tocan:
    pueden pertenecer tanto a PyTorch como a ONNX Runtime, y el otro runtime
    debe seguir funcionando. Lo que quede huérfano se limpia en la siguiente
    instalación, que reemplaza el paquete y sus metadatos a la vez.
    """
    from src.dependencies import _rmtree_force

    prefixes = _runtime_group_prefixes(dep_id)
    if not prefixes:
        raise RuntimeError(f"Runtime desconocido: {dep_id}")

    targets = [
        entry
        for entry in sorted(_store_entries(RUNTIME_MODULES_DIR))
        if _entry_matches_prefixes(entry, prefixes)
    ]
    if not targets:
        # Nada que borrar: el runtime no está en el store (o ya se limpió).
        return True, []

    leftovers: list[str] = []
    for entry in targets:
        path = os.path.join(RUNTIME_MODULES_DIR, entry)
        complete = _rmtree_force(path)
        if not complete or os.path.exists(path):
            leftovers.append(path)

    # Limpieza de restos evidentes de esa instalación (no compartidos).
    for extra in ("__pycache__",):
        path = os.path.join(RUNTIME_MODULES_DIR, extra)
        if os.path.isdir(path):
            _rmtree_quiet(path)

    _invalidate_store_probe_cache()
    return not leftovers, leftovers


def uninstall_runtime(dep_id: str, parent=None) -> bool:
    """Desinstala UN runtime (``torch`` u ``onnxruntime``) del store compartido.

    Devuelve True si el borrado quedó completo; False si algún archivo estaba
    bloqueado y se completará en el próximo arranque.
    """
    complete, leftovers = _uninstall_runtime_group(dep_id)
    if not complete:
        logger.info(
            "desinstalación de %s incompleta; restos programados: %s",
            dep_id, leftovers,
        )
    return complete


def uninstall_local_runtimes(parent=None) -> None:
    """Borra el store local de runtimes (``bin/runtimes/ai_modules``) completo.

    Borrado conjunto de PyTorch Y ONNX Runtime. Se mantiene para una limpieza
    total explícita; el botón "Uninstall" de cada fila usa
    :func:`uninstall_runtime` para respetar el otro runtime.

    En Windows, si algún DLL está cargado (runtime ya importado),
    ``dependencies._rmtree_force`` lo programa para el próximo arranque.
    """
    try:
        from src.dependencies import _rmtree_force
    except Exception:
        return
    if os.path.isdir(RUNTIME_MODULES_DIR):
        _rmtree_force(RUNTIME_MODULES_DIR)
    _invalidate_store_probe_cache()


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
    # torch/lib y onnxruntime/capi contienen las DLL nativas que
    # CPython carga via LoadLibrary. Añadirlas explícitamente antes
    # que `torch/__init__.py` lo haga evita condiciones de carrera
    # cuando el store se acaba de instalar en la misma sesión.
    for _dll_sub in (
        os.path.join(RUNTIME_MODULES_DIR, "torch", "lib"),
        os.path.join(RUNTIME_MODULES_DIR, "onnxruntime", "capi"),
        os.path.join(RUNTIME_MODULES_DIR, "onnxruntime", "capi", "build"),
    ):
        if os.path.isdir(_dll_sub):
            extra.append(_dll_sub)
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
