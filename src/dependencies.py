"""Registro unificado de dependencias externas del programa.

El panel de gestión muestra FFmpeg, los runtimes
de IA (PyTorch/ONNX Runtime) y las ranuras de modelos por categoría. El
sensor de estado es *honesto contra disco*: cada ``status_fn`` comprueba
la realidad (binario, store local ``bin/runtimes/ai_modules``, motores bajo
``bin/models/<Category>``) en lugar de adivinar.

La capa ``src.dependencies`` está pensada para que añadir una dependencia
nueva sea puramente declarativo: una entrada en ``REGISTRY`` + una función
``status`` que devuelva un dict con los campos expuestos aquí. La UI consume
este módulo a través de ``list_all()``, ``reinstall()`` y ``uninstall()``
exclusivamente.

Una fila es un dict, y sus campos estan escritos una sola vez, en
``ROW_FIELDS``: esa tabla es la que recorre ``list_all`` para garantizar los
suyos y la que usa el test del contrato para comprobar que una fila real trae
lo que la fila de la UI lee. Un ``status_fn`` no tiene por que devolver los
campos garantizados: si falta alguno, ``list_all`` lo rellena. Lo que no debe
hacer es devolver claves que no signifiquen nada para la UI, y por eso los
sensores de ``runtimes.manager`` no se copian enteros: son de otro esquema
(``installed``, ``system``, ``gpu``).

Lo que este modulo NO hace, porque es de otro dueno: medir el disco y
borrarlo aunque Windows no deje (``src.fsops``), y decidir que se le dice
al usuario en cada boton (``src.ui.dep_row``).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from src import fsops
from src.config import (
    ensure_runtime_dirs,
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
# Sensores de estado: una funcion por dependencia
# ---------------------------------------------------------------------------


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

def _cpu_only_reason() -> str:
    """Por qué la IA va en CPU cuando hay GPU pero no es NVIDIA.

    El runtime que instala esta app es el build CUDA de PyTorch y ONNX
    Runtime, y ese build solo acelera tarjetas NVIDIA: en un equipo AMD o
    Intel la inferencia funciona igual, pero en CPU. Sin esta frase la fila
    decía solo "solo CPU", que delante de una Radeon parece un fallo del
    equipo en vez de una limitación conocida.

    ``get_known_hardware()`` es la detección que ya hizo la ventana principal
    (``None`` = no lo sabemos): se lee de ahí y no se vuelve a sondear el
    sistema al pintar una fila. Si no sabemos qué GPU hay, no se afirma nada.
    """
    try:
        from src.hardware.detector import get_known_hardware

        hardware = get_known_hardware() or {}
    except Exception:
        return ""
    vendors = {
        str(gpu.get("vendor") or "").strip().lower()
        for gpu in (hardware.get("gpus") or [])
        if isinstance(gpu, dict)
    }
    if "nvidia" in vendors:
        # Con NVIDIA la causa sería otra (driver, build CPU): no se inventa.
        return ""
    if any(vendor.startswith(("amd", "intel")) for vendor in vendors):
        return (
            " · el runtime instalado es el build CUDA, que solo acelera "
            "tarjetas NVIDIA: en AMD/Intel la IA se ejecuta en CPU"
        )
    return ""


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
    local_bytes = fsops.safe_dir_size(store)
    gpu = bool(info.get("gpu"))
    system = bool(info.get("system"))

    if installed:
        status = STATUS_INSTALLED
        gpu_text = " · GPU (CUDA) disponible" if gpu else " · solo CPU"
        if not gpu:
            gpu_text += _cpu_only_reason()
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

    # Qué claves del dict del manager pasan a la fila, y cuáles no.
    #
    # Aquí SÍ hace falta una lista explícita, y no es la misma forma que
    # ``_ffmpeg_status``: la fuente de aquí (``runtimes.manager.*_status``)
    # es un SENSOR, no una fila. Sus claves son de otro esquema —``installed``,
    # ``system``, ``gpu``— y no campos que la UI sepa leer. Si se copiase el
    # dict entero, la fila arrastraría esas tres, y mañana el manager
    # añadiría ``cuda_smi`` y la fila también, sin que nadie lo pidiera.
    #
    # En FFmpeg la fuente (``get_install_info``) YA es una fila, así que
    # allí copiarla entera es correcto y además es lo que evita que un campo
    # nuevo se pierda por omisión. Aquí la lista es la que manda.
    fila = {
        # Campos de fila que esta capa decide o traduce del sensor.
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
        "deletable": installed and local_bytes > 0,
        "local": installed,
        # ``gpu`` del manager pasa renombrado a ``gpu_available`` porque es
        # un campo de fila; el original se queda en el sensor.
        "gpu_available": gpu,
    }
    return fila


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


def _uninstall_runtime(dep_id: str, parent=None) -> bool:
    """Desinstala un runtime concreto del store compartido.

    Envoltorio para que el REGISTRY no dependa del nombre exacto de la función
    del manager (que es quien registra el resultado y programa la limpieza de
    los archivos bloqueados para el próximo arranque).
    """
    return ai_runtimes_manager.uninstall_runtime(dep_id, parent=parent)


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
                    total += fsops.safe_dir_size(sub)
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

    # Aquí SÍ se construye el dict entero, y es a propósito: una ranura de
    # modelos no consulta ningún manager que le devuelva una fila —su sensor
    # cuenta carpetas en disco— así que no hay un dict de partida que
    # preservar. La forma de ``_ffmpeg_status`` y ``_runtime_status`` es
    # "empezar por lo que devuelve otro y sobreescribir lo mío"; esta no
    # tiene "otro". Lo que comparte con ellas es no depender de una lista de
    # campos ajenos: aquí la lista es propia y completa.
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


# ---------------------------------------------------------------------------
# Estado dinámico de FFmpeg
# ---------------------------------------------------------------------------

def _ffmpeg_status():
    info = ffmpeg_manager.get_install_info()
    status = info["status"]
    build_label = info.get("build_label") or "FFmpeg"
    sin_descarga = info.get("nothing_to_install")
    if status == STATUS_INSTALLED:
        detail = (
            f"Alojado en la propia carpeta del programa ({build_label}). "
            "Usa 'Update' si hay una build más reciente en el upstream."
        )
    elif status == STATUS_MISSING and sin_descarga:
        # Aquí no hay nada bajable, así que NO se manda a pulsar un botón que
        # no va a funcionar. Es la misma frase que ya usan el botón, el
        # banner y la confirmación de reinstalar: un solo enunciado para este
        # hecho, no cuatro redacciones distintas.
        detail = f"FFmpeg no está instalado y aquí no se puede descargar: {sin_descarga}."
    elif status == STATUS_MISSING:
        detail = (
            "FFmpeg no está instalado todavía. Pulsa 'Install' para "
            "descargarlo en la propia carpeta del programa."
        )
    else:
        detail = "FFmpeg está instalado pero no responde a -version."

    # Se parte del dict que devuelve ``get_install_info`` y se sobreescribe
    # SÓLO lo que esta capa cambia de verdad. Antes se reconstruía el dict
    # campo por campo, y por eso un campo nuevo añadido en el manager se
    # perdía en silencio por no estar en la lista: así es como
    # ``nothing_to_install`` llegaba a publicarse y no llegaba nunca a la fila.
    #
    # Aquí se copia ENTERO a propósito, y es la diferencia con
    # ``_runtime_status``: la fuente ya es una fila —sus claves son campos
    # que la UI sabe leer—, así que no hay nada interno que filtrar y una
    # lista de campos prohibidos sería otra lista que se puede quedar vieja.
    # El riesgo que sí tiene es al revés: que un campo de la fila se pierda,
    # y por eso se copia entero.
    fila = dict(info)
    fila.update({
        "status": status,
        "build_label": build_label,
        "is_updatable": True,
        "size_bytes": fsops.safe_dir_size(info["install_dir"]),
        "detail": detail,
        "deletable": status == STATUS_INSTALLED,
    })
    return fila


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
        "reinstall_fn": ai_runtimes_manager.install_pytorch,
        # Se desinstala SOLO PyTorch (y sus satélites onnx/onnxscript): el store
        # es compartido con ONNX Runtime, que debe seguir funcionando.
        "uninstall_fn": lambda parent=None: _uninstall_runtime("torch", parent),
        "is_updatable": True,
    },
    {
        "id": "onnxruntime",
        "kind": KIND_RUNTIME,
        "display_name": "ONNX Runtime (CPU/GPU)",
        "status_fn": _onnxruntime_status,
        "reinstall_fn": ai_runtimes_manager.install_onnxruntime,
        "uninstall_fn": lambda parent=None: _uninstall_runtime("onnxruntime", parent),
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
    },
]


# ---------------------------------------------------------------------------
# El contrato de una fila
# ---------------------------------------------------------------------------

# Dueños posibles de un campo de fila. Los dos primeros son los unicos que
# puede cumplir esta capa; el tercero es el resto del mundo.
# Los nombres llevan el prefijo DUENO_ porque ``REGISTRY`` sin mas ya es la
# lista de entradas del registro, y reutilizarlo aqui rompia ``list_all``.
DUENO_LIST_ALL = "list_all"    # esta capa lo garantiza
DUENO_REGISTRY = "registro"    # esta capa lo sobrescribe desde la entrada
DUENO_FUERA = "otra capa"      # lo trae otro, y se anota para no perderlo de vista

# Esta tabla es la UNICA lista de los campos de una fila. La recorre
# ``list_all`` para garantizar los suyos, y la usa el test del contrato para
# comprobar que una fila de verdad trae lo que la fila de la UI lee. Anadir un
# campo a la UI obliga a anadirlo aqui, y si no, el test lo dice.
#
# Cada entrada es (campo, defecto, dueño):
#
#   defecto - lo que esta capa pone si el ``status_fn`` no lo traia. Puede ser
#              una funcion de (entrada del REGISTRY, fila), porque la mayoria
#              dependen de la entrada y no de una constante.
#   dueño    - quien lo pone de verdad. ``DUENO_LIST_ALL`` se rellena con el
#              defecto, ``DUENO_REGISTRY`` se sobrescribe siempre, y
#              ``DUENO_FUERA`` significa que esta capa NO lo garantiza.
#
# Los ultimos los lee la fila de la UI y no los garantiza ``list_all``.
# No son adorno: son el hueco real entre lo que la fila necesita y lo que esta
# capa promete, y por eso viven aqui con su dueño escrito en vez de estar
# repartidos entre el docstring, el manager y el dialogo.
#
# Y no pueden meterse en el bucle de garantias tal cual, por una razon concreta:
# el dialogo decide montar el selector de build con ``"build_options" in info``,
# asi que rellenar esa clave en toda fila lo montaria tambien en torch y en las
# ranuras de modelos. Un campo que la UI consulta con ``in`` no puede pasar a
# estar siempre presente sin cambiar lo que se ve.
ROW_FIELDS = (
    # (campo, defecto, dueño)
    ("id", lambda e, i: e.get("id"), DUENO_REGISTRY),
    ("display_name", lambda e, i: e.get("display_name", e.get("id", "")), DUENO_LIST_ALL),
    ("kind", lambda e, i: e.get("kind", "unknown"), DUENO_LIST_ALL),
    ("status", STATUS_COMING_SOON, DUENO_LIST_ALL),
    ("version", "—", DUENO_LIST_ALL),
    ("install_dir", None, DUENO_LIST_ALL),
    ("source_url", None, DUENO_LIST_ALL),
    ("is_updatable", lambda e, i: bool(e.get("is_updatable", False)), DUENO_LIST_ALL),
    ("size_bytes", 0, DUENO_LIST_ALL),
    ("detail", "", DUENO_LIST_ALL),
    # Su defecto mira el status, asi que esta fila tiene que ir despues de
    # ``status`` en la tabla. No es un detalle: reordenarlo cambia el valor.
    ("deletable", lambda e, i: i.get("status") == STATUS_INSTALLED, DUENO_LIST_ALL),
    # --- de aqui para abajo, los lee la fila y NO los garantiza esta capa ---
    # La version que se LEE, distinta de ``version`` (el token tal cual). Solo
    # FFmpeg tiene las dos: en una nightly de master el token es un
    # `git describe` que no dice la linea, y la fila necesita las dos. Es de
    # ``ffmpeg.manager``, como ``build_label``, y la fila la lee con ``.get``
    # y cae a ``version`` en las filas que no la traen.
    ("version_text", None, DUENO_FUERA),
    ("build_label", None, DUENO_FUERA),
    ("build_id", None, DUENO_FUERA),
    ("preferred_build_id", None, DUENO_FUERA),
    ("latest_fingerprint", None, DUENO_FUERA),
    # Lo lee el banner de la ventana principal, no la fila, para decir
    # "checked 10 min ago". Vive aqui porque es un campo de fila igual
    # que los de arriba: lo publica el manager y solo llega a FFmpeg.
    ("last_check_iso", None, DUENO_FUERA),
    ("upstream_error", None, DUENO_FUERA),
    ("update_available", None, DUENO_FUERA),
    ("nothing_to_install", None, DUENO_FUERA),
    ("build_options", None, DUENO_FUERA),
)


def _defecto(entry, info, defecto):
    """Resuelve el defecto de un campo: constante, o funcion de la entrada."""
    return defecto(entry, info) if callable(defecto) else defecto


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
        # Los campos garantizados, en el orden que dicta ROW_FIELDS.
        for campo, defecto, dueno in ROW_FIELDS:
            if dueno == DUENO_REGISTRY:
                # El id canonico es el del REGISTRY (no permitimos que
                # status_fn lo pise con None: romperia reinstall/uninstall por
                # clave), asi que este se sobrescribe y no se rellena.
                info[campo] = _defecto(entry, info, defecto)
            elif dueno == DUENO_LIST_ALL:
                info.setdefault(campo, _defecto(entry, info, defecto))
        out.append(info)
    return out


def reinstall(dep_id: str, progress_callback: Optional[Callable[[int], None]] = None,
              parent=None, abort_event=None, timeout=None) -> bool:
    """Re-descarga / re-instala la dependencia indicada. Lanza excepción si falla.

    ``abort_event`` (threading.Event) y ``timeout`` (segundos) se propagan al
    instalador subyacente para que Instalar/Actualizar se pueda cancelar y
    nunca cuelgue indefinidamente. Todo ``reinstall_fn`` del REGISTRY debe
    aceptar estos dos parámetros opcionales.

    No hay parámetro de "qué versión": la build que se reinstala la decide
    cada instalador, y el de FFmpeg la lee de la preferencia (que es la
    elegida en su fila del panel).
    """
    ensure_runtime_dirs()
    for entry in REGISTRY:
        if entry.get("id") != dep_id:
            continue
        fn = entry.get("reinstall_fn")
        if not fn:
            raise RuntimeError(
                f"{dep_id} no tiene instalador automático todavía (coming soon)."
            )
        fn(progress_callback=progress_callback, parent=parent,
           abort_event=abort_event, timeout=timeout)
        return True
    raise KeyError(f"Dependencia desconocida: {dep_id}")


def uninstall(dep_id: str, parent=None) -> bool:
    """Desinstala/borra la dependencia indicada desde la app.

    Lanza excepción si la dependencia no admite borrado o el borrado falla
    (la UI traduce el error). Para runtimes y FFmpeg con archivos bloqueados
    por Windows, el borrado se programa para el próximo arranque
    (``src.fsops.process_pending_cleanup()``).
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
        # Si el instalador devuelve un bool, False significa "quedan archivos
        # bloqueados; el borrado se completará en el próximo arranque". La UI
        # lo usa para avisar en vez de decir que todo ha desaparecido ya.
        result = fn(parent=parent)
        return True if result is None else bool(result)
    raise KeyError(f"Dependencia desconocida: {dep_id}")


def remove_local_path(path: str) -> bool:
    """Borra un directorio/archivo local (fuerza + cleanup pendiente).

    Usado por el borrado de motores IA desde la pestaña AI Engines. Devuelve
    True si el borrado quedó completo; si algún archivo queda bloqueado, se
    programa para el próximo arranque.
    """
    return fsops.rmtree_force(path)


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
                fsops.rmtree_force(target)
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
