import sys
import os

# Asegurar que src/ esté en el path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    # Import diferido: importar PySide6/QtGUI solo cuando se va a abrir la
    # GUI de verdad. En entornos sin bibliotecas de ventana (CI, --selftest)
    # la app no debe reventar antes de que el selftest pueda reportar.
    from src.app import DraggyEncoderApp
    app = DraggyEncoderApp()
    sys.exit(app.run())
