import logging
import os
import sys

logger = logging.getLogger(__name__)

# Inyectar runtimes de IA locales lo antes posible para que imports
# posteriores de torch/onnxruntime encuentren los paquetes descargados
# en bin/runtimes/ai_modules.
try:
    from src.runtimes.manager import inject_runtime_paths, ensure_runtime_dlls_on_path
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
except Exception:
    pass

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from src.config import APP_NAME, get_asset_path
from src.ffmpeg.codec_tester import test_all_codecs
from src.ffmpeg.manager import ensure_ffmpeg
from src.hardware.detector import get_hardware_summary
from src.ui.loading_dialog import LoadingDialog
from src.ui.main_window import MainWindow
from src.ui.selectable import install_text_selection


def _set_windows_app_user_model_id():
    """Da a la ventana una identidad propia en la taskbar de Windows.

    Sin esto, Windows agrupa la ventana bajo el icono genérico del
    intérprete (``python.exe`` en la versión suelta) o del bootloader, y
    la taskbar muestra ESE icono aunque ``setWindowIcon`` esté bien
    puesto. Con un AppUserModelID explícito, la taskbar usa el icono de
    la ventana (el logo de la app).
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "com.draggyencoder.DraggyEncoder"
        )
    except Exception:
        pass


class StartupWorker(QThread):
    # Resultado estructurado (dict) para que el diálogo de arranque pueda
    # distinguir QUÉ fase falló en vez de traducir cualquier error a
    # "FFmpeg Error". Campos:
    #   phase            : "ok" | "ffmpeg" | "hardware" | "codecs" | "startup"
    #   ffmpeg_path      : str|None
    #   hardware_info    : dict (mejor esfuerzo; {} si falló la detección)
    #   compatible_codecs: dict (mejor esfuerzo; {} si falló el test)
    #   error            : str|None (solo para fases fatales: ffmpeg/startup)
    #   hardware_error   : str|None (warning NO fatal)
    #   codecs_error     : str|None (warning NO fatal)
    #   runtime_error    : str|None (warning NO fatal; falló el auto-install
    #                     del runtime IA en el primer arranque)
    startup_finished = Signal(object)
    progress = Signal(int, str)

    def run(self):
        ffmpeg_path = None
        hardware_info: dict = {}
        compatible_codecs: dict = {}
        hardware_error = None
        codecs_error = None
        runtime_error = None
        try:
            # 1. Hardware — best-effort. Un fallo aquí NO debe bloquear el
            #    arranque: el panel cae a CPU/lista vacía y avisamos después.
            self.progress.emit(10, "Detecting hardware...")
            try:
                hardware_info = get_hardware_summary()
            except Exception as exc:
                hardware_error = str(exc)
                logger.warning("Hardware detection error: %s", exc)

            # 2. FFmpeg (descarga si falta; ocupa el rango 30-55% del diálogo)
            #    Crítico: sin FFmpeg la app no puede comprimir.
            self.progress.emit(30, "Checking FFmpeg...")
            ffmpeg_path = ensure_ffmpeg(progress_callback=self._on_ffmpeg_progress)
            if not ffmpeg_path:
                self.startup_finished.emit({
                    "phase": "ffmpeg",
                    "ffmpeg_path": None,
                    "hardware_info": hardware_info,
                    "compatible_codecs": {},
                    "error": None,
                    "hardware_error": hardware_error,
                    "codecs_error": None,
                })
                return

            # 3. AI runtimes (ONNX Runtime + PyTorch) — auto-install en el
            #    primer arranque. NO bloqueante: un fallo aquí (sin red,
            #    antivirus, etc.) no impide abrir la app; el panel de
            #    dependencias mostrará el runtime como missing y el
            #    usuario podrá reintentarlo desde ahí.
            try:
                from src.runtimes.manager import (
                    install_onnxruntime,
                    install_pytorch,
                    onnx_status,
                    torch_status,
                )
                if not onnx_status().get("installed"):
                    self.progress.emit(55, "Installing ONNX Runtime...")
                    install_onnxruntime(
                        "auto",
                        progress_callback=lambda p, m, b=55, s=17: (
                            self._map_runtime_progress(p, m, b, s)
                        ),
                    )
                if not torch_status().get("installed"):
                    self.progress.emit(72, "Installing PyTorch...")
                    install_pytorch(
                        "auto",
                        progress_callback=lambda p, m, b=72, s=18: (
                            self._map_runtime_progress(p, m, b, s)
                        ),
                    )
            except Exception as exc:
                runtime_error = str(exc)
                logger.warning("AI runtime auto-install failed: %s", exc)

            # 4. Test codecs — best-effort. Un fallo aquí deja la lista de
            #    codecs vacía (el usuario puede Re-detect más tarde) sin
            #    impedir el arranque.
            self.progress.emit(90, "Testing codecs...")
            try:
                compatible_codecs = test_all_codecs(
                    ffmpeg_path,
                    progress_callback=lambda p: self.progress.emit(
                        90 + p // 10, "Testing codecs..."
                    ),
                )
            except Exception as exc:
                codecs_error = str(exc)
                logger.warning("Codec test error: %s", exc)

            self.progress.emit(100, "Ready")
            self.startup_finished.emit({
                "phase": "ok",
                "ffmpeg_path": ffmpeg_path,
                "hardware_info": hardware_info,
                "compatible_codecs": compatible_codecs,
                "error": None,
                "hardware_error": hardware_error,
                "codecs_error": codecs_error,
                "runtime_error": runtime_error,
            })
        except Exception as exc:
            logger.exception("Startup error")
            self.startup_finished.emit({
                "phase": "startup",
                "ffmpeg_path": ffmpeg_path,
                "hardware_info": hardware_info,
                "compatible_codecs": {},
                "error": str(exc),
                "hardware_error": hardware_error,
                "codecs_error": codecs_error,
                "runtime_error": runtime_error,
            })

    def _on_ffmpeg_progress(self, percent, message=""):
        """Map FFmpeg install/download progress (0-100) to dialog 30-60%.

        El callback del downloader (``download_ffmpeg``) emite mensajes
        fase-aware (``Downloading FFmpeg...``, ``Extracting zip...``,
        ``Installing binaries...``, ``FFmpeg ready``) que propagamos
        directamente al usuario para que sepa qué está pasando en lugar
        de quedarse mirando una barra quieta.
        """
        try:
            pct = max(0, min(100, int(percent or 0)))
        except Exception:
            pct = 0
        scaled = 30 + (pct * 30 // 100)
        self.progress.emit(scaled, message or "Checking FFmpeg...")

    def _map_runtime_progress(self, percent, message="", base=60, span=20):
        """Map AI runtime install progress (0-100) to a dialog range.

        ``install_onnxruntime``/``install_pytorch`` emiten porcentajes
        0-100 y fases amigables (``Descargando ONNX Runtime...``). Se
        escalan al rango ``[base, base+span]`` del diálogo de arranque
        (ONNX Runtime ocupa 55-72%; PyTorch 72-90%).
        """
        try:
            pct = max(0, min(100, int(percent or 0)))
        except Exception:
            pct = 0
        scaled = base + (pct * span // 100)
        self.progress.emit(scaled, message or "Installing AI runtime...")


class DraggyEncoderApp:
    def __init__(self):
        # Debe ir ANTES de crear la QApplication para que Windows asocie
        # el botón de la taskbar al icono correcto desde el primer frame.
        _set_windows_app_user_model_id()
        self.app = QApplication(sys.argv)
        self.app.setApplicationName(APP_NAME)
        self.app.setStyle("Fusion")
        # Una sola política para todo el programa: cualquier texto se puede
        # seleccionar y copiar. Va aquí, con la QApplication, para que también
        # la alcancen los cuadros de diálogo que se abren más adelante.
        install_text_selection(self.app)
        # El logo del programa es el mismo icono que el botón Import de
        # la cadena: un único asset en toda la app.
        logo_path = get_asset_path("Import.png")
        if logo_path and os.path.exists(logo_path):
            self.app.setWindowIcon(QIcon(logo_path))

    def run(self):
        self.loading = LoadingDialog("Starting DRAGGY ENCODER", "Initializing...")
        self.loading.show()

        self.worker = StartupWorker()
        self.worker.progress.connect(self._on_progress)
        self.worker.startup_finished.connect(self._on_finished)
        self.worker.start()

        return self.app.exec()

    def _on_progress(self, value, message):
        self.loading.set_progress(value)
        self.loading.set_message(message)

    def _on_finished(self, result):
        self.loading.close()
        result = result or {}
        phase = result.get("phase") or "startup"
        ffmpeg_path = result.get("ffmpeg_path")
        hardware_info = result.get("hardware_info") or {}
        compatible_codecs = result.get("compatible_codecs") or {}
        error = result.get("error")
        hardware_error = result.get("hardware_error")
        codecs_error = result.get("codecs_error")
        runtime_error = result.get("runtime_error")

        from PySide6.QtWidgets import QMessageBox

        # Fase crítica: sin FFmpeg la app no puede funcionar.
        if phase == "ffmpeg" or not ffmpeg_path:
            from src.config import get_ffmpeg_dir, get_logs_dir
            target_dir = get_ffmpeg_dir()
            log_path = os.path.join(get_logs_dir(), "ffmpeg_download.log")
            QMessageBox.critical(
                None,
                "FFmpeg Error",
                f"Could not find or download FFmpeg.\n\n"
                f"DRAGGY ENCODER uses its own bundled copy of FFmpeg inside\n"
                f"its installation folder. To fix this:\n\n"
                f"  • Try again with a working internet connection, OR\n"
                f"  • Download ffmpeg/ffprobe manually and drop them into:\n"
                f"    {target_dir}\n\n"
                f"Detailed log: {log_path}",
            )
            sys.exit(1)

        # Fallo inesperado no atribuible a una fase concreta.
        if phase == "startup":
            QMessageBox.critical(
                None,
                "Startup Error",
                f"DRAGGY ENCODER could not finish starting up.\n\n"
                f"{error or 'Unknown error.'}\n\n"
                "If this keeps happening, reinstall the application or "
                "report the error with the details above.",
            )
            sys.exit(1)

        # Avisos NO fatales: la app arranca igual, degradada.
        if hardware_error:
            QMessageBox.warning(
                None,
                "Hardware Detection Failed",
                f"Could not detect the hardware. The program will start in a "
                f"degraded mode (CPU, no codec list) — you can retry later "
                f"from the 'Re-detect' button.\n\nDetail: {hardware_error}",
            )
        if codecs_error:
            QMessageBox.warning(
                None,
                "Codec Test Failed",
                f"FFmpeg was found, but the codec compatibility test failed. "
                f"The codec list will be empty until you retry with 'Re-detect'.\n\n"
                f"Detail: {codecs_error}",
            )
        if runtime_error:
            # No fatal: la app arranca, pero los modelos IA no se podrán
            # usar hasta que el runtime se instale desde Dependencies.
            QMessageBox.warning(
                None,
                "AI Runtime Not Ready",
                "ONNX Runtime or PyTorch could not be installed automatically "
                "during startup, so AI models are disabled for now.\n\n"
                "Retry from Manage Dependencies → AI Runtimes → Install.\n\n"
                f"Detail: {runtime_error}",
            )

        self.window = MainWindow(ffmpeg_path, hardware_info, compatible_codecs)
        self.window.show()
