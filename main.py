import sys
import os

# Asegurar que src/ esté en el path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    from src.config import ensure_runtime_dirs, get_logs_dir
    ensure_runtime_dirs()
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
try:
    from src.dependencies import process_pending_cleanup
    process_pending_cleanup()
except Exception:
    pass

# Inyectar runtimes de IA locales (bin/runtimes/ai_modules) antes de que
# cualquier otro módulo importe torch/onnxruntime. Así el usuario no
# necesita instalarlos manualmente con pip.
try:
    from src.runtimes.manager import inject_runtime_paths
    inject_runtime_paths()
except Exception:
    pass

# Migrar token HF de preferences.json a personal/ si existe
try:
    from src.personal import migrate_from_preferences
    migrate_from_preferences()
except Exception:
    pass

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

    try:
        from src.config import get_bin_dir
        report(f"  OK   runtime dirs under {get_bin_dir()}")
    except Exception:
        ok = False
        report("  FAIL runtime dirs:\n  " + traceback.format_exc().strip())

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
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    # Import diferido: importar PySide6/QtGUI solo cuando se va a abrir la
    # GUI de verdad. En entornos sin bibliotecas de ventana (CI, --selftest)
    # la app no debe reventar antes de que el selftest pueda reportar.
    from src.app import DraggyEncoderApp
    app = DraggyEncoderApp()
    sys.exit(app.run())
