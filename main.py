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

from src.app import DraggyEncoderApp


if __name__ == "__main__":
    app = DraggyEncoderApp()
    sys.exit(app.run())
