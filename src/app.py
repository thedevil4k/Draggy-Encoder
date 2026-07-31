import os
import sys

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
from PySide6.QtWidgets import QApplication

from src.config import APP_NAME
from src.ffmpeg.codec_tester import test_all_codecs
from src.ffmpeg.manager import ensure_ffmpeg
from src.hardware.detector import get_hardware_summary
from src.ui.loading_dialog import LoadingDialog
from src.ui.main_window import MainWindow


class StartupWorker(QThread):
    startup_finished = Signal(object, object, object)  # ffmpeg_path, hardware_info, compatible_codecs
    progress = Signal(int, str)

    def run(self):
        try:
            # 1. Hardware
            self.progress.emit(10, "Detecting hardware...")
            hardware_info = get_hardware_summary()

            # 2. FFmpeg (descarga si falta; ocupa el rango 30-60% del diálogo)
            self.progress.emit(30, "Checking FFmpeg...")
            ffmpeg_path = ensure_ffmpeg(progress_callback=self._on_ffmpeg_progress)
            if not ffmpeg_path:
                self.startup_finished.emit(None, hardware_info, {})
                return

            # 3. Test codecs
            self.progress.emit(60, "Testing codecs...")
            compatible_codecs = test_all_codecs(
                ffmpeg_path,
                progress_callback=lambda p: self.progress.emit(60 + p // 3, "Testing codecs..."),
            )

            self.progress.emit(100, "Ready")
            self.startup_finished.emit(ffmpeg_path, hardware_info, compatible_codecs)
        except Exception as exc:
            print(f"Startup error: {exc}", file=sys.stderr)
            self.startup_finished.emit(None, {}, {})

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


class DraggyEncoderApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName(APP_NAME)
        self.app.setStyle("Fusion")

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

    def _on_finished(self, ffmpeg_path, hardware_info, compatible_codecs):
        self.loading.close()
        if not ffmpeg_path:
            from PySide6.QtWidgets import QMessageBox
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

        self.window = MainWindow(ffmpeg_path, hardware_info, compatible_codecs)
        self.window.show()
