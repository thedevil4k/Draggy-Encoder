import sys
import os

# Asegurar que src/ esté en el path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# HuggingFace lee esto al importar el paquete. En el EXE windowed
# sys.stderr es None y tqdm falla con "'NoneType' object has no attribute
# 'write'" al descargar modelos. Debe quedar fijado ANTES de que hf_search
# importe huggingface_hub; hacerlo solo dentro del worker llega tarde.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# Logging centralizado + sys.excepthook (lo antes posible, antes de importar
# nada más). El ejecutable empaquetado es "windowed" (sin consola):
# sys.stdout/stderr son None y cualquier traceback se perdería en silencio.
# Por eso todo el diagnóstico se vuelca a bin/logs/app.log y un crash muestra
# un diálogo con la ruta del log para que el usuario pueda adjuntarla.
# ---------------------------------------------------------------------------
import logging

# get_logs_dir() es frozen-aware (junto al EXE en bundles portables, en el
# perfil del usuario en paquetes Linux instalados). Importar src.config aquí
# es seguro: solo usa os/sys/platform y NO dispara torch/PySide6.
try:
    from src.config import (
        configure_process_temp,
        ensure_runtime_dirs,
        get_bin_dir,
        get_logs_dir,
    )
    ensure_runtime_dirs()
    # La caché de HuggingFace también debe vivir junto a los datos de la
    # app, no en C:\Users\...\.cache\huggingface. Se fija antes de cualquier
    # import de huggingface_hub.
    _hf_home = os.path.join(get_bin_dir(), ".cache", "huggingface")
    os.makedirs(os.path.join(_hf_home, "hub"), exist_ok=True)
    os.environ["HF_HOME"] = _hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
    # El render, el host de IA y el guard del selftest crean temporales. En
    # un bundle portable deben vivir junto a la app, no en C:/%TEMP%.
    try:
        configure_process_temp()
    except Exception:
        pass
    _APP_LOG_PATH = os.path.join(get_logs_dir(), "app.log")
    _FAULTHANDLER_LOG_PATH = os.path.join(get_logs_dir(), "faulthandler.log")
except Exception:
    _APP_LOG_PATH = None
    _FAULTHANDLER_LOG_PATH = None

try:
    # BackupRotatingFileHandler: además de rotar app.log, archiva el fichero
    # rotado más antiguo a bin/logs/backup/ antes de borrarlo (y snapshotea
    # faulthandler.log), manteniendo los últimos 10 respaldos de cada tipo.
    from src.log_backup import BackupRotatingFileHandler, DEFAULT_KEEP

    _root_logger = logging.getLogger()
    _root_logger.setLevel(logging.INFO)
    if _APP_LOG_PATH:
        _log_handler = BackupRotatingFileHandler(
            _APP_LOG_PATH,
            maxBytes=2 * 1024 * 1024,  # 2 MB por archivo
            backupCount=3,
            encoding="utf-8",
            backup_prefix="app",
            keep=DEFAULT_KEEP,
            extra_snapshots=[(_FAULTHANDLER_LOG_PATH, "faulthandler")],
        )
        _log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        _root_logger.addHandler(_log_handler)
except Exception:
    # Log inaccesible (permisos, disco lleno): la app arranca igual y el
    # excepthook de abajo también es tolerante a fallos de E/S.
    pass


def _uncaught_exception(exc_type, exc_value, exc_tb):
    """sys.excepthook: registra el traceback completo y avisa al usuario.

    - Siempre escribe el traceback en ``bin/logs/app.log`` (y en stderr si
      existe consola).
    - En modo GUI muestra un diálogo con la ruta del log. En ``--selftest``
      (CI headless) solo registra, para no colgar el job esperando un clic.
    Nunca lanza: un fallo del propio hook no debe enmascarar el error real.
    """
    import traceback

    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        logging.getLogger("crash").error("Unhandled exception:\n%s", tb_text)
    except Exception:
        pass
    try:
        print(f"Unhandled exception:\n{tb_text}", file=sys.stderr)
    except Exception:
        pass

    if "--selftest" in sys.argv:
        return
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        log_path = _APP_LOG_PATH or "(log no disponible)"
        QMessageBox.critical(
            None,
            "DRAGGY ENCODER — Unexpected Error",
            f"An unexpected error occurred.\n\n"
            f"Full details were written to:\n{log_path}\n\n"
            f"Please include this file when reporting the problem.",
        )
    except Exception:
        # Sin GUI disponible (headless, libEGL ausente, ...): solo queda el log.
        pass

    # Con --crash-report, tras avisar al usuario abrimos un issue de GitHub
    # pre-rellenado con el traceback y el final de los logs. Solo en modo
    # GUI (--selftest ya retornó arriba) y tolerante a fallos de red/navegador.
    if "--crash-report" in sys.argv:
        try:
            from src.crash_report import open_crash_report
            open_crash_report(tb_text)
        except Exception:
            pass


sys.excepthook = _uncaught_exception

# faulthandler: vuelca tracebacks de TODOS los hilos en segfaults/fatal
# signals y, además, periódicamente (cada 5 min) para diagnosticar cuelgues
# que el excepthook no ve (p.ej. un ffmpeg/pip colgado en un worker). Va a un
# archivo propio (faulthandler.log) para no pelear con la rotación del
# RotatingFileHandler de app.log. La referencia al fichero se mantiene viva
# en módulo para que faulthandler no escriba a un handle cerrado.
try:
    import faulthandler

    if _FAULTHANDLER_LOG_PATH:
        _faulthandler_file = open(
            _FAULTHANDLER_LOG_PATH,
            "a",
            encoding="utf-8",
        )
        faulthandler.enable(file=_faulthandler_file)
        faulthandler.dump_traceback_later(300, repeat=True, file=_faulthandler_file)
    else:
        _faulthandler_file = None
        faulthandler.enable()
except Exception:
    _faulthandler_file = None

# Completar borrados pendientes (archivos DLL bloqueados por Windows en la
# sesión anterior) ANTES de que nada importe torch/onnxruntime.
#
# Entra por ``src.fsops`` y no por el registro de dependencias a proposito:
# ese modulo arrastra los dos managers que aqui hay que liberar de sus
# DLL. ``fsops`` solo importa os/json/shutil, asi que la garantia de
# "primero limpiar, luego cargar" es del sitio de la llamada y no del
# orden en que casualmente se importen las cosas.
try:
    from src.fsops import process_pending_cleanup
    process_pending_cleanup()
except Exception:
    pass

# Inyectar runtimes de IA locales (bin/runtimes/ai_modules) antes de que
# cualquier otro módulo importe torch/onnxruntime. Así el usuario no
# necesita instalarlos manualmente con pip.
try:
    from src.runtimes.manager import inject_runtime_paths, cleanup_install_leftovers
    # Restos de una instalación interrumpida (staging sin validar o store
    # anterior pendiente de borrar): se limpian ANTES de tocar sys.path, de
    # modo que un staging a medias nunca pueda confundirse con el store.
    cleanup_install_leftovers()
    inject_runtime_paths()
except Exception:
    pass

# Migrar token HF de preferences.json a personal/ si existe
try:
    from src.personal import migrate_from_preferences
    migrate_from_preferences()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Guard de runtimes pesados (la regresión del "instalado no arranca")
# ---------------------------------------------------------------------------
#
# Importar torch/onnxruntime DENTRO del proceso congelado (PyInstaller +
# PySide6) tumbaba DraggyEncoder.exe con una violación de acceso (0xc0000005
# en python313.dll): ocurría al VALIDAR el runtime recién descargado, con
# ``_can_import('torch')`` en la pila del crash. El guard monta un store de
# PRUEBA con paquetes falsos y recorre las mismas funciones que usa la app
# (``torch_status``, ``onnx_status``, ``_post_install_import_check``): si algún
# runtime pesado acaba en ``sys.modules`` de este proceso, el selftest falla y
# el CI no empaqueta el instalador. El store de prueba vive en un directorio
# temporal y ``RUNTIME_MODULES_DIR`` se restaura siempre: nunca entra en el
# instalador ni toca el store real.
_HEAVY_GUARD_INIT = {
    "torch": (
        '__version__ = "0.0.0-guard"\n'
        "class _GuardCuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "cuda = _GuardCuda()\n"
    ),
    "onnxruntime": (
        '__version__ = "0.0.0-guard"\n'
        "def get_available_providers():\n"
        '    return ["CUDAExecutionProvider", "CPUExecutionProvider"]\n'
    ),
}


def _heavy_guard_loaded(module_name: str) -> set:
    """Claves de ``sys.modules`` del runtime (``torch`` y ``torch.*``)."""
    prefix = module_name + "."
    return {
        key for key in sys.modules
        if key == module_name or key.startswith(prefix)
    }


def _heavy_guard_check(report, manager, module_name: str, label: str,
                       optional: bool = False) -> bool:
    """Comprueba UN runtime pesado: estado y post-install, sin importarlo aquí."""
    if optional and not manager._store_has_package_dir(module_name):
        report(f"  SKIP guard {module_name} ({label}): runtime no instalado")
        return True
    before = _heavy_guard_loaded(module_name)
    try:
        status = (
            manager.torch_status() if module_name == "torch"
            else manager.onnx_status()
        )
        post_ok, detail = manager._post_install_import_check(module_name)
    except Exception as exc:
        report(f"  FAIL guard {module_name} ({label}): {type(exc).__name__}: {exc}")
        return False
    added = _heavy_guard_loaded(module_name) - before
    if added:
        report(
            f"  FAIL guard {module_name} ({label}): se importó en ESTE proceso "
            f"(sys.modules += {sorted(added)[:4]})"
        )
        return False
    if before and getattr(sys, "frozen", False):
        report(
            f"  FAIL guard {module_name} ({label}): ya estaba cargado antes de "
            f"sondear ({sorted(before)[:4]})"
        )
        return False
    if not status.get("installed"):
        if optional:
            report(
                f"  WARN guard {module_name} ({label}): el store no valida el "
                "paquete (la app ofrecerá reinstalarlo)"
            )
            return True
        report(
            f"  FAIL guard {module_name} ({label}): la sonda no vio el store de "
            "prueba (¿el Python privado no puede importar?)"
        )
        return False
    # ``gpu`` es una CAPACIDAD, no un requisito: en un equipo sin CUDA
    # onnx_status() devuelve gpu=False porque el runtime CPU es válido.
    # Exigir True convertía una instalación CPU perfectamente utilizable en
    # FAIL. Lo que este guard debe demostrar es que la sonda respondió (la
    # clave existe) y que el post-install corrió fuera de este proceso.
    if "gpu" not in status:
        report(
            f"  FAIL guard {module_name} ({label}): la sonda no reportó "
            "capacidades (ni cuda ni cpu) desde el subproceso"
        )
        return False
    if not post_ok:
        report(f"  FAIL guard {module_name} ({label}): post-install: {detail}")
        return False
    capability = "GPU" if status.get("gpu") else "CPU"
    report(f"  OK   guard {module_name} ({label}): sin import in-process ({capability})")
    return True


def _selftest_heavy_runtime_guard(report, check_real_store: bool | None = None) -> bool:
    """¿La detección de runtimes pesados ocurre SIEMPRE fuera de este proceso?

    Ver ``_HEAVY_GUARD_INIT``. ``check_real_store`` (por defecto: solo en el
    EXE congelado) repite la comprobación contra el store real instalado, que
    es el escenario exacto del fallo reportado en el equipo del usuario.
    """
    import shutil
    import tempfile

    from src.config import get_temp_dir
    from src.runtimes import manager

    if check_real_store is None:
        check_real_store = bool(getattr(sys, "frozen", False))

    ok = True
    guard_root = get_temp_dir()
    os.makedirs(guard_root, exist_ok=True)
    store = tempfile.mkdtemp(prefix="draggy-guard-", dir=guard_root)
    original_store = manager.RUNTIME_MODULES_DIR
    original_path = list(sys.path)
    try:
        for module_name, source in _HEAVY_GUARD_INIT.items():
            package = os.path.join(store, module_name)
            os.makedirs(package, exist_ok=True)
            with open(os.path.join(package, "__init__.py"), "w", encoding="utf-8") as fh:
                fh.write(source)
        manager.RUNTIME_MODULES_DIR = store
        manager._invalidate_store_probe_cache()
        if not manager._can_probe_subprocess():
            report(
                "  FAIL guard (store de prueba): no hay intérprete privado para "
                "sondear (ver 'private runtime python')"
            )
            ok = False
        else:
            for module_name in _HEAVY_GUARD_INIT:
                if not _heavy_guard_check(report, manager, module_name, "store de prueba"):
                    ok = False
    finally:
        manager.RUNTIME_MODULES_DIR = original_store
        manager._invalidate_store_probe_cache()
        sys.path[:] = original_path
        shutil.rmtree(store, ignore_errors=True)

    if check_real_store:
        for module_name in _HEAVY_GUARD_INIT:
            if not _heavy_guard_check(report, manager, module_name, "store real",
                                      optional=True):
                ok = False
    return ok


def _selftest_ai_host(report) -> bool:
    """¿Arranca el host de IA y mueve frames? (el proceso que carga torch).

    La GUI ya no importa runtimes nativos: se lo pide a este proceso, que
    corre con el Python privado del instalador. Si el bundle no puede
    arrancarlo (falta el Python privado, falta la copia de fuentes
    ``ai_src`` que copia el spec, protocolo desfasado), la app no podrá hacer
    IA: es un fallo de empaquetado y debe verse en el build, no en el equipo
    del usuario. El motor ``mock`` comprueba el cableado completo (arranque,
    framing, registro de motores, ida y vuelta de blobs) sin necesitar torch.
    """
    from src.ai import host_client

    if not host_client.host_available():
        if os.environ.get("DRAGGY_AI_HOST", "").strip().lower() in ("off", "0", "false"):
            report("  SKIP host de IA (DRAGGY_AI_HOST=off)")
            return True
        report(
            "  FAIL host de IA: no disponible ("
            f"python={host_client.host_python() or 'no encontrado'}, "
            f"fuentes={host_client.host_source_root() or 'no encontradas'})"
        )
        return False

    try:
        import numpy as np

        host = host_client.get_host()
        info = host.ping()
        frame = np.zeros((16, 24, 3), dtype=np.uint8)
        frame[4:8, 4:8] = 255
        handle = host.engine_create("mock", "")
        if not host.engine_load(handle):
            raise RuntimeError("el motor de prueba no cargó")
        out = host.engine_process(handle, [frame])
        host.engine_unload(handle)
        if not out or tuple(out[0].shape) != frame.shape or not bool((out[0] == frame).all()):
            raise RuntimeError("los frames no volvieron intactos del host")
    except Exception as exc:
        report(f"  FAIL host de IA: {type(exc).__name__}: {exc}")
        return False

    report(
        "  OK   host de IA arranca "
        f"(python {info.get('python')}, pid {info.get('pid')}, store {info.get('store')})"
    )
    report("  OK   host de IA: frames ida y vuelta")
    return True


def _selftest_model_capabilities(report) -> bool:
    """Comprueba que la UI no ofrece modelos que este build no puede cargar.

    Los nodos Denoise/Colorize/FaceRestore deben quedar deshabilitados con el
    aviso explícito, y en Upscale ninguna entrada sin loader puede ser
    seleccionable ni descargable. Se ejecuta en cada build (fuente, bundle y
    AppImage) para que una regresión no llegue al usuario.
    """
    import traceback

    try:
        from PySide6.QtWidgets import QApplication

        from src.ai.arch_specs import NOT_AVAILABLE_LABEL
        from src.ui.workflow_widget import WorkflowWidget

        if QApplication.instance() is None:
            report("  SKIP workflow capabilities (sin QApplication)")
            return True

        widget = WorkflowWidget()
        problems = []
        try:
            installed_categories = []
            for name in ("Denoise", "Colorize", "FaceRestore"):
                step = next(
                    (s for s in widget.middle_steps if s.name == name), None
                )
                if step is None:
                    problems.append(f"{name}: nodo ausente")
                    continue
                # Un motor de usuario instalado es una configuración válida:
                # la app debe ofrecerlo y seleccionarlo. El supuesto antiguo
                # (todos estos nodos siempre bloqueados) solo es cierto en un
                # bundle limpio sin modelos instalados, no en esta instalación.
                installed_runnable = int(
                    getattr(step, "_runnable_installed", 0) or 0
                )
                if installed_runnable:
                    installed_categories.append(name)
                    if not step.combo.isEnabled():
                        problems.append(
                            f"{name}: motor instalado pero desplegable deshabilitado"
                        )
                    # Un nodo inactivo devuelve "" por diseño; la selección
                    # solo es obligatoria cuando el usuario lo activa.
                    if getattr(step, "active", False) and not step.get_selected_engine_id():
                        problems.append(
                            f"{name}: motor instalado pero sin selección"
                        )
                    continue
                text = step.combo.itemText(0) if step.combo.count() else ""
                if NOT_AVAILABLE_LABEL not in text.lower():
                    problems.append(f"{name}: no marcado como no disponible")
                if step.combo.isEnabled():
                    problems.append(f"{name}: desplegable habilitado")
                if step.get_selected_engine_id():
                    problems.append(f"{name}: motor seleccionado")
                if NOT_AVAILABLE_LABEL not in step._activation_block_reason().lower():
                    problems.append(f"{name}: sin mensaje de bloqueo")

            ups = next(
                (s for s in widget.middle_steps if s.name == "Upscale"), None
            )
            if ups is None:
                problems.append("Upscale: nodo ausente")
            else:
                runnable = 0
                for i in range(ups.combo.count()):
                    data = ups.combo.itemData(i)
                    if not isinstance(data, dict):
                        continue
                    if data.get("supported"):
                        runnable += 1
                    if data.get("supported") is False:
                        if "⬇" in ups.combo.itemText(i):
                            problems.append("Upscale: no ejecutable con descarga")
                        try:
                            enabled = ups.combo.model().item(i).isEnabled()
                        except Exception:
                            enabled = True
                        if enabled:
                            problems.append("Upscale: no ejecutable seleccionable")
                if runnable == 0:
                    problems.append("Upscale: sin opción ejecutable")
        finally:
            widget.deleteLater()

        if problems:
            report("  FAIL workflow capabilities: " + "; ".join(problems))
            return False
        if installed_categories:
            report(
                "  OK   workflow: motores instalados habilitados "
                f"({', '.join(installed_categories)}); "
                "categorías sin loader bloqueadas"
            )
        else:
            report("  OK   workflow: modelos sin loader marcados como no disponibles")
        return True
    except Exception:
        report(
            "  FAIL workflow capabilities:\n  "
            + traceback.format_exc().strip().replace("\n", "\n  ")
        )
        return False


def run_selftest() -> int:
    """Verificación rápida de salud sin abrir la GUI (flag ``--selftest``).

    La usa la CI del instalador y cualquiera que sospeche de una instalación
    rota. Escribe un log en ``bin/logs/selftest.log`` porque en el
    ejecutable empaquetado (consola deshabilitada) ``stdout`` puede no
    existir. Código de salida: 0 = todo OK, 1 = falló algo crítico.
    """
    import importlib
    import traceback
    import time

    # Asegura que el directorio de logs exista ANTES de abrir el archivo.
    # Esto es crítico en CI donde el selftest corre como paso separado y
    # bin/logs puede no existir aún (el Get-Content del step lo espera).
    from src.config import ensure_runtime_dirs
    ensure_runtime_dirs()

    from src.config import get_logs_dir

    started = time.time()
    # El log debe vivir junto al ejecutable (get_logs_dir es frozen-aware:
    # en un bundle va a <exe>\bin\logs, no al CWD, que en un doble-clic
    # puede ser cualquier directorio).
    log_path = os.path.join(get_logs_dir(), "selftest.log")

    def report(msg: str) -> None:
        try:
            print(msg)
        except Exception:
            pass
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass

    report("=== Draggy Encoder selftest ===")
    report(f"  frozen : {bool(getattr(sys, 'frozen', False))}")
    report(f"  python : {sys.version.split()[0]} ({sys.executable})")

    ok = True
    for mod in ("numpy", "PySide6", "psutil", "requests", "huggingface_hub",
                "src.config", "src.app", "src.ui.main_window",
                "src.ai.inference", "src.ai.hf_search"):
        try:
            importlib.import_module(mod)
            report(f"  OK   import {mod}")
        except Exception:
            ok = False
            report(f"  FAIL import {mod}:")
            report("  " + traceback.format_exc().strip().replace("\n", "\n  "))

    # El bundle es el anfitrión de los runtimes instalados a demanda
    # (torch/onnxruntime y su árbol de dependencias, ver
    # draggy_encoder.spec): esos paquetes importan stdlib que nuestro código
    # no usa y, si el spec se la deja fuera, torch no importa en el equipo
    # del usuario aunque pip haya instalado todo (caso real: "No module
    # named 'timeit'" desde torch._strobelight). Esta sonda falla alto en CI
    # si el spec deja caer de nuevo un módulo necesario.
    for mod in ("timeit", "struct", "zoneinfo", "cProfile", "optparse",
                "pickletools", "graphlib", "wave", "trace", "modulefinder",
                "pstats"):
        try:
            importlib.import_module(mod)
            report(f"  OK   stdlib {mod}")
        except Exception:
            ok = False
            report(f"  FAIL stdlib {mod}:")
            report("  " + traceback.format_exc().strip().replace("\n", "\n  "))

    # Validar Qt offscreen, MainWindow y DependenciesDialog con el mismo
    # bucle que capture_screenshots.py (run_ui_checks en src/qt_check.py):
    # es la comprobación más fuerte que el simple import y detecta la falta
    # de libEGL/libGL/X11, fallos de construcción de la UI y diálogos rotos
    # antes de empaquetar. Sin QApplication válida, los checks de ventanas
    # se reportan como SKIP.
    from src.qt_check import run_ui_checks

    for name, status, detail in run_ui_checks():
        if status == "ok":
            report(f"  OK   {name}: {detail}")
        elif status == "fail":
            ok = False
            report(f"  FAIL {name}:")
            report("  " + detail)
        else:
            report(f"  SKIP {name} (sin QApplication válida)")

    if not _selftest_model_capabilities(report):
        ok = False

    try:
        from src.config import get_bin_dir
        report(f"  OK   runtime dirs under {get_bin_dir()}")
    except Exception:
        ok = False
        report("  FAIL runtime dirs:\n  " + traceback.format_exc().strip())

    # La descarga de modelos de HuggingFace usa tqdm; en el EXE windowed
    # sys.stderr es None. El selftest comprueba que el arranque desactiva las
    # barras y que la caché HF queda dentro de bin/, no en C:.
    try:
        from src.ai.fetcher import _configure_hf_download_environment
        hf_cache = _configure_hf_download_environment()
        if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") != "1":
            raise RuntimeError("HF_HUB_DISABLE_PROGRESS_BARS no está activo")
        if not hf_cache:
            raise RuntimeError("No se pudo configurar la caché de HuggingFace")
        from src.config import get_bin_dir as _get_bin_dir
        bin_root = os.path.abspath(_get_bin_dir())
        if not os.path.abspath(hf_cache).startswith(bin_root + os.sep):
            raise RuntimeError(f"caché HF fuera de bin: {hf_cache}")
        report(f"  OK   HuggingFace: progreso desactivado, caché {hf_cache}")
    except Exception:
        ok = False
        report("  FAIL HuggingFace setup:\n  " + traceback.format_exc().strip())

    if getattr(sys, "frozen", False):
        try:
            from src.runtimes.manager import get_embed_python_path
            embed = get_embed_python_path()
            if embed is None:
                raise RuntimeError(
                    "No se encontró el Python privado para instalar runtimes "
                    "junto al ejecutable"
                )
            report(f"  OK   private runtime python: {embed}")
        except Exception:
            ok = False
            report("  FAIL embedded python:\n  " + traceback.format_exc().strip())

    # Regresión del arranque del instalador: validar un runtime pesado
    # importándolo DENTRO del proceso de la GUI (0xc0000005 en python313.dll).
    # El guard usa un store de prueba, así que también falla en CI, donde el
    # runner no tiene torch instalado y por eso el bug nunca se detectaba.
    if not _selftest_heavy_runtime_guard(report):
        ok = False

    # El host de IA es el proceso que carga torch/onnxruntime: si el bundle no
    # puede arrancarlo, la app instalada no puede hacer IA (y el crash de las
    # DLL nativas volvería al proceso de la GUI). Se comprueba aquí, con el
    # Python privado, en CADA build.
    if not _selftest_ai_host(report):
        ok = False

    try:
        from src.ffmpeg.manager import get_local_ffmpeg_path
        ffmpeg = get_local_ffmpeg_path()
        if ffmpeg and os.path.isfile(ffmpeg):
            report(f"  OK   ffmpeg: {ffmpeg}")
        else:
            report("  WARN ffmpeg no presente todavía (la app lo instala al arrancar)")
    except Exception:
        report("  WARN ffmpeg check:\n  " + traceback.format_exc().strip())

    report(f"  total: {time.time() - started:.2f}s")
    report("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    # --crash-report (sin --selftest): modo reporte. Abre un issue de GitHub
    # pre-rellenado con el final de app.log y faulthandler.log y termina, sin
    # lanzar la GUI. No requiere que haya habido un crash: sirve para reportar
    # cualquier problema con los logs actuales.
    if "--crash-report" in sys.argv and "--selftest" not in sys.argv:
        from src.crash_report import open_crash_report
        open_crash_report()
        sys.exit(0)
    if "--encode-job" in sys.argv:
        from src.cli_encode import run_encode_cli
        # sys.argv[0] es el ejecutable; el parser recibe solo los argumentos.
        sys.exit(run_encode_cli(sys.argv[1:]))
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    # Import diferido: importar PySide6/QtGUI solo cuando se va a abrir la
    # GUI de verdad. En entornos sin bibliotecas de ventana (CI, --selftest)
    # la app no debe reventar antes de que el selftest pueda reportar.
    from src.app import DraggyEncoderApp
    app = DraggyEncoderApp()
    sys.exit(app.run())
