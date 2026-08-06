import os
import sys
import threading
import time

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from src.config import APP_NAME, get_asset_path
import src.preferences as preferences
from src.ffmpeg.runner import get_video_info, health_check_ffmpeg
from src.processing.pipeline import compress_video, compress_video_with_ai, generate_preview
from src.last_job import has_last_job as has_pending_last_job
from src.last_job import load_last_job as load_pending_last_job
from src.last_job import save_last_job as persist_last_job
from src.ui.dependencies_dialog import DependenciesDialog
from src.ui.hardware_dialog import HardwareDialog
from src.ui.preferences_dialog import PreferencesDialog
from src.ui.preview_dialog import PreviewDialog
from src.ui.settings_panel import SettingsPanel
from src.ui.update_banner import UpdateBanner
from src.ui.update_chain import _UpdateChainController, notify_log
from src.runtimes.manager import (
    get_python_support_info as _get_python_support_info,
    torch_status as _torch_status,
    onnx_status as _onnx_status,
)
from src.ui.preferences_dialog import (
    _format_bytes as _fmt_bytes,
    _format_duration as _fmt_dur,
)
from src.ui.workflow_widget import WorkflowWidget


# ---------------------------------------------------------------------------
# Workers especializados
# ---------------------------------------------------------------------------

class _FFmpegUpdateWorker(QThread):
    """Descarga la nueva build de FFmpeg en background.

    Conecta directo al ``MainWindow.progress`` y ``status_label``. En éxito
    emite ``finished_ok``; cualquier error propaga el mensaje por ``failed``.
    """

    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from src.ffmpeg.manager import force_redownload
            from src.ffmpeg.upstream import clear_cache
            try:
                clear_cache()
            except Exception:
                pass

            def _p(v):
                self.progress.emit(v)

            def _s(text):
                self.status.emit(text)

            force_redownload(progress_callback=_p, parent=self.parent())
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class _BannerCheckWorker(QThread):
    """Refresh del cache upstream (no descarga nada)."""

    done = Signal()

    def __init__(self, force=False, parent=None):
        super().__init__(parent)
        self.force = force

    def run(self):
        try:
            from src.ffmpeg.upstream import get_fingerprint
            get_fingerprint(force=self.force)
        except Exception:
            pass
        self.done.emit()


def resolve_fsr3_engine(upscale_info, interp_info):
    """Resolve the FSR 3.1 engine id / model path / mode from workflow nodes.

    FSR 3.1 decouples upscaling from frame interpolation. The user activates
    FSR in the Upscale node and/or the Interp node; the underlying engine
    runs both features when both nodes are active:

      - Upscale(FSR31) + Interp(FSR31) -> mode ``"both"``
      - Upscale(FSR31) alone            -> mode ``"upscale"``
      - Interp(FSR31) alone             -> mode ``"interpolate"``
      - anything else                   -> (upscale engine id, None, "upscale")

    Returns ``(engine_id, model_path, mode)``.
    """
    upscale_engine_id = None
    upscale_model_path = None
    upscale_scale_info = {}
    if isinstance(upscale_info, dict):
        upscale_engine_id = upscale_info.get("engine_id", "")
        upscale_model_path = upscale_info.get("model_path")
        upscale_scale_info = upscale_info
    elif upscale_info:
        upscale_engine_id = upscale_info

    interp_engine_id = None
    interp_scale_info = {}
    if isinstance(interp_info, dict):
        interp_engine_id = interp_info.get("engine_id", "")
        interp_scale_info = interp_info
    elif interp_info:
        interp_engine_id = interp_info

    fsr3_mode = "upscale"
    fsr3_engine_id = upscale_engine_id
    fsr3_model_path = upscale_model_path
    if upscale_engine_id and "fsr31" in upscale_engine_id:
        fsr3_mode = upscale_scale_info.get("mode", "upscale")
        if interp_engine_id and "fsr31" in interp_engine_id:
            fsr3_mode = "both"
    elif interp_engine_id and "fsr31" in interp_engine_id:
        fsr3_mode = "interpolate"
        fsr3_engine_id = interp_engine_id
        fsr3_model_path = interp_scale_info.get("model_path")
    return fsr3_engine_id, fsr3_model_path, fsr3_mode


class CompressionWorker(QThread):
    """Worker para ejecutar compresión o preview en segundo plano."""

    progress = Signal(int)
    status_message = Signal(str)

    def __init__(self, ffmpeg_path, input_path, output_path, settings, mode="compress",
                 ai_engines=None, abort_event=None):
        super().__init__()
        self.ffmpeg_path = ffmpeg_path
        self.input_path = input_path
        self.output_path = output_path
        self.settings = settings
        self.mode = mode
        # Optional mapping {"Upscale": "upscale-realesrgan-x4plus", ...}.
        # Source of truth is the WorkflowWidget at click time. Empty
        # means "no AI steps active right now". The pipeline logs the
        # selection so the user sees the chain wired end-to-end even
        # before real inference is hooked up.
        self.ai_engines = ai_engines
        # Cross-thread abort signal: setting this from the GUI thread
        # causes ``run_ffmpeg`` to terminate() the ffmpeg subprocess
        # and return False. We always create one locally so ``_abort``
        # and ``closeEvent`` can simply flip the flag without juggling
        # event ownership.
        #
        # Why not ``QThread.terminate()`` (the previous approach):
        # terminate() kills the Python wrapper thread but NOT the
        # ``subprocess.Popen`` child — ffmpeg kept running orphaned
        # after the program closed. The abort_event path reaches the
        # child because ``run_ffmpeg`` polls it on every stdout line.
        self.abort_event = abort_event if abort_event is not None else threading.Event()
        self.success = False
        self.error_message = ""
        self.preview_original = None
        self.preview_compressed = None

    def run(self):
        try:
            status_cb = self._emit_status
            # Determine if the AI chain should be used: any active workflow
            # node with an engine id (Upscale, Interp, Denoise, Colorize,
            # FaceRestore — in the re-ordered chain position).
            use_ai = False
            for _cat, _info in (self.ai_engines or {}).items():
                if isinstance(_info, dict):
                    _eid = _info.get("engine_id", "")
                else:
                    _eid = _info or ""
                if _eid:
                    use_ai = True
                    break

            if self.mode == "compress" and use_ai:
                # Build the AI CHAIN: each active workflow node (Upscale,
                # Interp, Denoise, Colorize, FaceRestore) resolves to a
                # loaded engine, in the re-ordered chain position. Engines
                # that cannot be loaded fail fast with a clear message.
                try:
                    engines, chain_labels = self._build_ai_chain(status_cb)
                except Exception as exc:
                    self.success = False
                    self.error_message = f"Failed to load AI model: {exc}"
                    return
                if not engines:
                    self.success = False
                    self.error_message = (
                        "No AI engines could be loaded for the selected chain."
                    )
                    return

                # Escala total de la cadena = producto de los factores de
                # escala de las engines que realmente upscalean. Se inyecta
                # en settings para que el encoder dimensione correctamente.
                from src.processing.pipeline import _compute_total_scale
                total_scale = _compute_total_scale(engines)
                self.settings["ai_scale"] = total_scale

                status_cb("AI chain: " + " -> ".join(chain_labels))
                status_cb("AI chain: frame-by-frame processing...")
                self.success = compress_video_with_ai(
                    self.ffmpeg_path,
                    self.input_path,
                    self.output_path,
                    self.settings,
                    inference_engines=engines,
                    progress_callback=self._emit_progress,
                    status_callback=status_cb,
                    abort_event=self.abort_event,
                    on_process=lambda p: setattr(self, "_proc", p),
                )
            elif self.mode == "compress":
                self.success = compress_video(
                    self.ffmpeg_path,
                    self.input_path,
                    self.output_path,
                    self.settings,
                    progress_callback=self._emit_progress,
                    status_callback=status_cb,
                    ai_engines=self.ai_engines,
                    abort_event=self.abort_event,
                    on_process=lambda p: setattr(self, "_proc", p),
                )
            elif self.mode == "preview":
                self.preview_original, self.preview_compressed = generate_preview(
                    self.ffmpeg_path,
                    self.input_path,
                    self.settings,
                    progress_callback=self._emit_progress,
                    status_callback=status_cb,
                    ai_engines=self.ai_engines,
                    abort_event=self.abort_event,
                    on_process=lambda p: setattr(self, "_proc", p),
                )
                self.success = bool(self.preview_original and self.preview_compressed)
                if not self.success and not self.error_message:
                    self.error_message = "Could not generate preview frames"
        except Exception as exc:
            self.success = False
            self.error_message = str(exc)
            print(f"CompressionWorker exception ({self.mode}): {exc}", file=sys.stderr)

    def _emit_progress(self, value):
        self.progress.emit(value)

    def _emit_status(self, message):
        self.status_message.emit(message)

    def _build_ai_chain(self, status_cb):
        """Build ONE loaded inference engine per active workflow node.

        ``self.ai_engines`` is an ordered mapping ``{"Upscale": {...},
        "Interpolate": {...}, "Denoise": {...}, "Colorize": {...},
        "FaceRestore": {...}}`` in the chain order chosen by the user
        (the WorkflowWidget reports it in ``middle_steps`` order).

        Returns ``(engines, labels)`` where engines is the ordered list
        of loaded ``AIInferenceEngine`` objects and labels is a parallel
        list of human-readable names for status lines.

        Raises RuntimeError if ANY selected engine cannot be loaded, so
        the user gets a precise, actionable message (which node failed
        instead of a silent partial chain).
        """
        from src.ai.inference import (
            create_inference_engine,
            get_model_path,
            convert_pth_to_onnx,
            PyTorchInferenceEngine,
            FSR31Engine,
        )
        from src.ai.architectures import is_torch_available

        engines = []        # loaded AIInferenceEngine, in chain order
        labels = []         # ["Upscale (Real-ESRGAN x4)", ...]
        errors = []         # human-readable per-node failures
        fsr31_built = False

        ai_map = self.ai_engines or {}

        # FSR 3.1 mode resolution across both workflow nodes (see
        # ``resolve_fsr3_engine``): Upscale(FSR31) + Interp(FSR31) -> "both",
        # Upscale(FSR31) alone -> "upscale", Interp(FSR31) alone ->
        # "interpolate".
        fsr3_engine_id, fsr3_model_path, fsr3_mode = resolve_fsr3_engine(
            ai_map.get("Upscale"),
            ai_map.get("Interpolate"),
        )

        # --- Build the chain in workflow order -------------------------
        for category, info in ai_map.items():
            if isinstance(info, dict):
                eid = info.get("engine_id", "") or ""
                model_path = info.get("model_path")
            else:
                eid = info or ""
                model_path = None
            if not eid:
                continue  # node active but "auto" — keep going

            node_label = f"{category} ({eid})"

            # --- FSR engines (pure algorithmic, no model downloads) ---
            if eid.startswith("upscale-fsr"):
                if "fsr31" in eid:
                    # FSR 3.1 decouples upscale (in Upscale node) from
                    # frame generation (in Interp node); build it ONCE, with
                    # the combined mode resolved across both nodes. The
                    # second FSR31 node in the chain adds nothing (it is
                    # the same engine, merged mode).
                    if fsr31_built:
                        continue
                    if fsr3_engine_id:
                        try:
                            eng = FSR31Engine(
                                fsr3_model_path or "",
                                fsr3_engine_id,
                                mode=fsr3_mode,
                            )
                            if eng.load():
                                engines.append(eng)
                                fsr31_built = True
                                labels.append(f"{category} (FSR 3.1, mode={fsr3_mode})")
                                status_cb(f"FSR 3.1 loaded (mode={fsr3_mode}, libplacebo)")
                            else:
                                errors.append(
                                    f"{node_label}: FSR 3.1 requires FFmpeg full "
                                    "with libplacebo support"
                                )
                        except Exception as exc:
                            errors.append(f"{node_label}: FSR 3.1 failed: {exc}")
                    continue
                try:
                    eng = create_inference_engine(eid, model_path or "")
                    if eng and getattr(eng, "is_loaded", False):
                        engines.append(eng)
                        labels.append(f"{category} (FSR {eid})")
                        status_cb(f"{category}: {eid} loaded")
                    else:
                        errors.append(f"{node_label}: engine did not load")
                except Exception as exc:
                    errors.append(f"{node_label}: {exc}")
                continue

            # --- Library model engines (any category) -------------------
            # Strategy: prefer ONNX Runtime when it has an ACTIVE GPU
            # provider (CUDA/TensorRT).  If ONNX only runs on CPU and
            # torch with CUDA is available, prefer PyTorch (GPU).  If no
            # .onnx exists and torch is available, convert .pth -> .onnx.
            # Used for EVERY model-based node so the chain works for any
            # catalog engine (Real-ESRGAN, NAFNet, DDColor, GFPGAN, ...).
            pth_path = model_path  # Prefer the exact path from UI
            if not pth_path or not os.path.exists(pth_path):
                pth_path = get_model_path(eid, "torch")

            try:
                inference_engine = None
                node_errors = []

                # 1) Existing .onnx next to .pth or from registry
                onnx_path = None
                if pth_path:
                    candidate = os.path.splitext(pth_path)[0] + ".onnx"
                    if os.path.exists(candidate):
                        onnx_path = candidate
                if not onnx_path:
                    onnx_path = get_model_path(eid, "onnxruntime")
                if onnx_path:
                    try:
                        engine_candidate = create_inference_engine("onnxruntime", onnx_path)
                    except Exception as exc:
                        node_errors.append(f"ONNX Runtime failed: {exc}")
                    else:
                        active_providers = getattr(engine_candidate, "get_active_providers", None)
                        active = (
                            active_providers()
                            if callable(active_providers)
                            else getattr(engine_candidate, "_providers", [])
                        )
                        if (
                            "CUDAExecutionProvider" not in active
                            and "TensorrtExecutionProvider" not in active
                            and pth_path
                            and is_torch_available()
                        ):
                            try:
                                torch_engine = PyTorchInferenceEngine(pth_path, eid)
                                if torch_engine.load():
                                    inference_engine = torch_engine
                                else:
                                    node_errors.append(
                                        "PyTorch load failed; keeping ONNX Runtime (CPU)"
                                    )
                                    inference_engine = engine_candidate
                            except Exception as exc:
                                node_errors.append(f"PyTorch (CUDA) failed: {exc}")
                                inference_engine = engine_candidate
                        else:
                            inference_engine = engine_candidate

                # 2) Convert .pth -> .onnx (needs torch)
                if inference_engine is None or not getattr(inference_engine, "is_loaded", False):
                    if pth_path and is_torch_available():
                        onnx_converted = convert_pth_to_onnx(pth_path)
                        if onnx_converted:
                            try:
                                inference_engine = create_inference_engine("onnxruntime", onnx_converted)
                            except Exception as exc:
                                node_errors.append(f"ONNX converted but load failed: {exc}")
                    elif pth_path:
                        node_errors.append(
                            "PyTorch is not installed (required to convert .pth to ONNX)"
                        )

                # 3) PyTorch direct
                if inference_engine is None or not getattr(inference_engine, "is_loaded", False):
                    if pth_path and is_torch_available():
                        try:
                            inference_engine = create_inference_engine("torch", pth_path)
                        except Exception as exc:
                            node_errors.append(f"PyTorch fallback failed: {exc}")

                if inference_engine is None or not getattr(inference_engine, "is_loaded", False):
                    detail = " | ".join(node_errors) if node_errors else "No model weights found."
                    hint = ""
                    if not node_errors or "No model weights" in detail:
                        hint = (
                            "This engine has no weight files installed. Open "
                            "Manage Dependencies → Runtimes & Binaries and click "
                            "'Add Engine' on the model row to download them."
                        )
                    raise RuntimeError(
                        f"{node_label} could not load.\nDetails: {detail}\n{hint}"
                    )

                engines.append(inference_engine)
                provider = getattr(inference_engine, "_providers", [])
                device = getattr(inference_engine, "_device", None)
                if "CUDAExecutionProvider" in provider or "TensorrtExecutionProvider" in provider:
                    status_cb(f"{category}: {eid} loaded (GPU, ONNX Runtime)")
                elif device is not None and "cuda" in str(device).lower():
                    status_cb(f"{category}: {eid} loaded (GPU, PyTorch)")
                else:
                    status_cb(f"{category}: {eid} loaded (CPU)")
                labels.append(f"{category} ({eid})")
            except RuntimeError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{node_label}: {exc}")
            continue

        if not engines:
            raise RuntimeError(
                "No AI engine could be loaded for your chain.\n"
                + ("\n".join(errors) if errors else "Check Manage Dependencies.")
            )
        if errors:
            logger.warning(
                "AI chain partially loaded; %d node(s) skipped: %s",
                len(errors), " | ".join(errors),
            )
        return engines, labels


class RedetectWorker(QThread):
    """Re-detecta hardware + codecs sin bloquear la UI."""

    progress = Signal(int, str)
    done = Signal(object, object)
    error = Signal(str)

    def __init__(self, ffmpeg_path):
        super().__init__()
        self.ffmpeg_path = ffmpeg_path

    def run(self):
        try:
            from src.hardware.detector import get_hardware_summary
            from src.ffmpeg.codec_tester import test_all_codecs

            self.progress.emit(15, "Re-detecting hardware...")
            hardware_info = get_hardware_summary()

            self.progress.emit(45, "Re-testing codecs...")
            compatible = test_all_codecs(
                self.ffmpeg_path,
                progress_callback=lambda p: self.progress.emit(
                    45 + p // 2, "Re-testing codecs..."
                ),
            )
            self.progress.emit(100, "Done")
            self.done.emit(hardware_info, compatible)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, ffmpeg_path, hardware_info, compatible_codecs, parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.hardware_info = hardware_info
        self.compatible_codecs = compatible_codecs
        self.input_path = None
        # Carpeta de output inicial: prefiere ``last_output_dir`` si la
        # preferencia ``auto_open_last_output`` está activa y la carpeta
        # sigue existiendo; cae al Desktop si falta o falla.
        self.output_dir = self._resolve_initial_output_dir()
        # Estados del ciclo de auto-update al cerrar:
        #   _close_pending       : el usuario pidio cerrar pero el worker aun
        #                         no termino. El banner-callback lo detecta y
        #                         ejecuta el camino de cierre nativo.
        #   _close_after_worker : el worker termino OK/fail, solo queda
        #                         aceptar el evento de cierre sin reentrar.
        #   _close_consumed      : la cadena de update ha terminado y se ha
        #                         consumido el intento de cierre del usuario.
        #                         El banner-success esta visible con los
        #                         mini-stats. Un segundo click en X cierra
        #                         directamente (sin re-prompt).
        self._close_pending = False
        self._close_after_worker = False
        self._close_consumed = False
        # Controller de cadena multi-dep para auto_update_all_on_close.
        # Solo se instancia cuando el usuario confirma el cierre y la
        # preferencia esta activa; vive en self._close_chain hasta que
        # termina la cadena. Se descarta en _on_chain_finished.
        self._close_chain = None
        # Cadena "Update + Compress último proyecto":
        # - _close_followup_job: el payload (load_last_job) que se va a
        #   ejecutar cuando termine el chain. None si el usuario eligió
        #   "Update only" o "Cancel close".
        # - _close_followup_complete: gate one-shot para closeEvent —
        #   cuando la cadena Y el CompressionWorker terminan OK permite
        #   cerrar sin volver a mostrar el proposal.
        # - _update_chain_summary: el dict del chain_finished que se
        #   combina en una sola tray notification tras el followup.
        self._close_followup_job = None
        self._close_followup_complete = False
        self._update_chain_summary = None
        # Best-effort: si la ventana es destruida mientras el chain corre
        # (kill externo, crash, cierre brusco), ``destroyed`` se dispara y
        # pedimos al worker del chain que pare. Esto evita que el QThread
        # quede como zombi heredado por Qt sin control.
        self.destroyed.connect(self._shutdown_chain)
        self._setup_tray_icon()
        self._setup_ui()

    def _setup_tray_icon(self):
        """Inicializa un QSystemTrayIcon estable para notificaciones nativas.

        Solo lo usamos para showMessage() al terminar una auto-update al
        cerrar el programa. Si el sistema no soporta tray (ej. Linux sin
        shell adecuado), el icono queda inactivo; el programa sigue
        funcionando porque _notify_completion tiene fallback a log.
        """
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = None
            return
        try:
            style = QApplication.style()
            icon = style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        except Exception:
            icon = QGuiApplication.windowIcon()
        self.tray_icon = QSystemTrayIcon(icon, self)
        self.tray_icon.setToolTip("DRAGGY ENCODER")
        # Menu contextual minimo: salir del programa.
        menu = QMenu(self)
        action_quit = menu.addAction("Quit DRAGGY ENCODER")
        action_quit.triggered.connect(self.close)
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()

    def _setup_ui(self):
        self.setWindowTitle(f"{APP_NAME} v1")
        self.setMinimumSize(1100, 800)
        self.setStyleSheet(self._stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Top buttons
        top_layout = QHBoxLayout()
        self.btn_select = QPushButton("Select Videos")
        self.btn_select.clicked.connect(self._select_video)
        self.btn_output = QPushButton(f"Output: ...{os.path.basename(self.output_dir)}")
        self.btn_output.clicked.connect(self._select_output)
        self.btn_hardware = QPushButton("Hardware Info")
        self.btn_hardware.clicked.connect(self._show_hardware)
        self.btn_redetect = QPushButton("Re-detect Hardware & Codecs")
        self.btn_redetect.clicked.connect(self._on_redetect_clicked)
        self.btn_dependencies = QPushButton(" AI Engines")
        self.btn_dependencies.setIcon(QIcon(get_asset_path("AI panel.png")))
        self.btn_dependencies.clicked.connect(self._show_dependencies)
        self.btn_preferences = QPushButton("Preferences")
        self.btn_preferences.clicked.connect(self._show_preferences)

        top_layout.addWidget(self.btn_select)
        top_layout.addWidget(self.btn_output)
        top_layout.addWidget(self.btn_hardware)
        top_layout.addWidget(self.btn_redetect)
        top_layout.addWidget(self.btn_dependencies)
        top_layout.addWidget(self.btn_preferences)
        layout.addLayout(top_layout)

        # Banner persistente de "Update available". Si esta oculto, QVBoxLayout
        # hace que la fila colapse automaticamente.
        self.update_banner = UpdateBanner()
        self.update_banner.update_requested.connect(self._on_banner_update_clicked)
        self.update_banner.panel_requested.connect(self._show_dependencies)
        self.update_banner.history_requested.connect(self._show_preferences_history)
        self.update_banner.dismissed.connect(self.update_banner.hide_banner)
        layout.addWidget(self.update_banner)

        # GPU badge
        recommended = self.hardware_info.get("recommended_device", "CPU")
        gpus = self.hardware_info.get("gpus", [])
        gpu_text = recommended
        if gpus:
            first_gpu = gpus[0]
            vendor = first_gpu.get("vendor", "")
            name_words = first_gpu.get("name", "").split()
            short_name = name_words[0] if name_words else ""
            gpu_text = f"{vendor} {short_name}".strip() or recommended
        self.gpu_label = QLabel(gpu_text)
        self.gpu_label.setStyleSheet(
            "background-color: #16213e; color: #00d2d3; padding: 4px 12px; border-radius: 6px; font-weight: bold;"
        )
        layout.addWidget(self.gpu_label)

        # AI runtime availability banner (below GPU badge).
        self.ai_runtime_banner = self._build_ai_runtime_banner()
        layout.addWidget(self.ai_runtime_banner)

        # Workflow
        self.workflow = WorkflowWidget()
        # When a node has no engines installed, ``AIWorkflowStep``
        # exposes a clickable "No engines — open AI Engines Manager"
        # hint. Bridge it to ``DependenciesDialog`` opened on the AI
        # Engines tab so the user lands in the right context.
        self.workflow.manage_engines_requested.connect(
            self._on_manage_engines_requested
        )
        self.workflow.model_selected.connect(
            self._on_model_selected_from_search
        )
        self.workflow.engine_changed.connect(
            self._on_engine_changed
        )
        layout.addWidget(self.workflow)

        # Settings
        self.settings_panel = SettingsPanel(
            hardware_info=self.hardware_info,
            compatible_codecs=self.compatible_codecs,
        )
        self.settings_panel.set_device(recommended)
        # Restaura codec/device desde prefs si la pref
        # ``remember_last_codec_device`` está activa. Idempotente:
        # no hace nada si los valores no son válidos o la pref está off.
        self._restore_last_codec_device()
        layout.addWidget(self.settings_panel, stretch=1)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        self.btn_compress = QPushButton("Compress")
        self.btn_compress.clicked.connect(self._compress)
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self._preview)
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._abort)

        bottom_layout.addWidget(self.btn_compress)
        bottom_layout.addWidget(self.btn_preview)
        bottom_layout.addWidget(self.btn_abort)
        layout.addLayout(bottom_layout)

        # Status and progress
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #a0a0b0;")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Estado interno del banner: cuadricula de timers que mantiene la
        # propuesta al dia durante la sesion.
        self._banner_check_worker = None
        self._banner_update_worker = None
        # Lectura instantanea del cache -> el banner aparece ya en el primer
        # paint si hay update pendiente.
        self._refresh_banner_from_cache()
        # Despues, 5 s mas tarde, disparamos un check en background que
        # respeta el cache de 24h.
        self._banner_initial_timer = QTimer(self)
        self._banner_initial_timer.setSingleShot(True)
        self._banner_initial_timer.timeout.connect(self._refresh_banner_async)
        self._banner_initial_timer.start(5000)
        # Y cada 30 minutos volvemos a chequear.
        self._banner_periodic_timer = QTimer(self)
        self._banner_periodic_timer.timeout.connect(self._refresh_banner_async)
        self._banner_periodic_timer.start(30 * 60 * 1000)

    @staticmethod
    def _build_ai_runtime_banner():
        """Build a small banner that warns when AI runtimes are missing/unsupported."""
        banner = QLabel("AI runtime status: checking...")
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel { background-color: #16213e; color: #a0a0b0; "
            "padding: 6px 12px; border-radius: 6px; font-size: 11px; }"
        )
        support = _get_python_support_info()
        torch_ok = _torch_status().get("installed", False)
        onnx_ok = _onnx_status().get("installed", False)
        if not support["supported"]:
            banner.setText(
                f"⚠ {support['message']} AI upscaling/interpolation/colorize are disabled."
            )
            banner.setStyleSheet(
                "QLabel { background-color: #5f3a00; color: #ffffff; "
                "padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: bold; }"
            )
        elif not torch_ok or not onnx_ok:
            banner.setText(
                "AI runtimes missing: PyTorch and/or ONNX Runtime are not installed. "
                "Open Manage Dependencies → Runtimes & Binaries to install them."
            )
            banner.setStyleSheet(
                "QLabel { background-color: #3a2c00; color: #f39c12; "
                "padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: bold; }"
            )
        else:
            banner.setText(
                "✓ AI runtimes ready. Select an AI model from the workflow nodes."
            )
            banner.setStyleSheet(
                "QLabel { background-color: #16213e; color: #00d2d3; "
                "padding: 6px 12px; border-radius: 6px; font-size: 11px; }"
            )
        return banner

    def _stylesheet(self):
        return """
        QMainWindow {
            background-color: #0f0f1a;
        }
        QWidget {
            background-color: #0f0f1a;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2d3, stop:1 #5f27cd);
            color: #0f0f1a;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00f0f1, stop:1 #7d3fe0);
        }
        QPushButton:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        QLineEdit, QComboBox, QSpinBox {
            background-color: #16213e;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 6px;
            color: #e0e0e0;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #16213e;
            color: #e0e0e0;
            selection-background-color: #00d2d3;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #2c3e50;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #00d2d3;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #00d2d3;
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QProgressBar {
            background-color: #16213e;
            border-radius: 6px;
            text-align: center;
            color: #ffffff;
        }
        QProgressBar::chunk {
            background-color: #00d2d3;
            border-radius: 6px;
        }
        """

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mkv *.avi *.wmv *.mov *.webm *.m4v);;All Files (*)",
        )
        if path:
            self.input_path = path
            self.status_label.setText(f"Selected: {os.path.basename(path)}")
            self._update_original_bitrate(path)

    def _update_original_bitrate(self, path):
        info = get_video_info(self.ffmpeg_path, path)
        try:
            bitrate = int(info.get("bit_rate", 0))
            if bitrate:
                self.settings_panel.set_original_bitrate(f"{bitrate // 1000} kbps")
            else:
                self.settings_panel.set_original_bitrate("unknown")
        except Exception:
            self.settings_panel.set_original_bitrate("unknown")

        try:
            width = info.get("width", "?")
            height = info.get("height", "?")
            duration = info.get("duration", "?")
            codec = info.get("codec_name", "?")
            fps_str = info.get("r_frame_rate", "")
            # Convert fractional fps like "30000/1001" to "29.97"
            fps_display = "?"
            fps_value = 30.0
            if fps_str and "/" in fps_str:
                try:
                    num, den = fps_str.split("/")
                    fps_value = int(num) / int(den)
                    fps_display = f"{fps_value:.2f}"
                except Exception:
                    fps_display = fps_str
            elif fps_str:
                fps_display = fps_str
                try:
                    fps_value = float(fps_str)
                except Exception:
                    fps_value = 30.0
            bitrate_text = f"{bitrate // 1000} kbps" if bitrate else "?"
            if duration not in (None, "?", ""):
                try:
                    duration = f"{float(duration):.1f}s"
                except Exception:
                    pass
            info_text = f"{width}x{height} @ {fps_display}fps\n{codec}\n{bitrate_text}\n{duration}"
            self.workflow.set_import_info(info_text)
            # Store input dimensions for upscale output resolution display.
            try:
                self._input_width = int(width)
                self._input_height = int(height)
            except (ValueError, TypeError):
                self._input_width = 0
                self._input_height = 0
            self._input_fps = fps_value
            self._input_bitrate_kbps = bitrate // 1000 if bitrate else 0
            self._update_upscale_output_info()
            self._update_ai_rendered_bitrate()
        except Exception:
            self.workflow.set_import_info("No info")

    def _update_upscale_output_info(self):
        """Update the Upscale node status with output resolution."""
        w = getattr(self, "_input_width", 0)
        h = getattr(self, "_input_height", 0)
        if not w or not h:
            return
        # Get the selected upscale model info from the workflow.
        for step in self.workflow.middle_steps:
            if getattr(step, "name", "") == "Upscale":
                info = step.get_selected_model_info()
                engine_id = info.get("engine_id", "")
                scale = info.get("scale", 1)
                model_path = info.get("model_path")
                if not engine_id:
                    self.workflow.set_upscale_output_info("")
                    return
                if scale > 1:
                    out_w, out_h = w * scale, h * scale
                    model_name = os.path.basename(model_path) if model_path else engine_id
                    self.workflow.set_upscale_output_info(
                        f"Output: {out_w}x{out_h} ({scale}x)"
                    )
                else:
                    self.workflow.set_upscale_output_info("")
                return

    def _update_ai_rendered_bitrate(self):
        """Show estimated output bitrate after AI processing.

        The idea: if the video goes through upscale (and eventually
        interpolation/denoise), the output has more pixels (and
        potentially more frames).  To keep the *same visual quality*
        the bitrate must scale proportionally.
        """
        w = getattr(self, "_input_width", 0)
        h = getattr(self, "_input_height", 0)
        orig_kbps = getattr(self, "_input_bitrate_kbps", 0)
        if not w or not h or not orig_kbps:
            self.settings_panel.set_ai_bitrate("")
            return

        # Determine effective output resolution from active AI steps.
        scale = 1
        for step in self.workflow.middle_steps:
            if getattr(step, "name", "") == "Upscale":
                info = step.get_selected_model_info()
                scale = info.get("scale", 1)
                break

        out_w = w * scale
        out_h = h * scale

        if out_w == w and out_h == h:
            self.settings_panel.set_ai_bitrate("")
            return

        pixel_ratio = (out_w * out_h) / (w * h)
        estimated = int(orig_kbps * pixel_ratio)
        self.settings_panel.set_ai_bitrate(
            f"AI Rendered bitrate: ~{estimated:,} kbps (estimated)"
        )

    def _select_output(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_dir
        )
        if dir_path:
            self.output_dir = dir_path
            self.btn_output.setText(f"Output: ...{os.path.basename(dir_path)}")
            # Persistimos ``last_output_dir`` para que la próxima sesión
            # pueda auto-abrirla si ``auto_open_last_output`` está activa.
            try:
                preferences.set("last_output_dir", dir_path)
            except Exception:
                pass

    def _on_engine_changed(self, node_name, engine_id):
        """Handle engine selection change in any AI workflow node."""
        if node_name == "Upscale":
            self._update_upscale_output_info()
            self._update_ai_rendered_bitrate()

    def _show_hardware(self):
        dialog = HardwareDialog(self.hardware_info, self)
        dialog.exec()

    def _show_dependencies(self):
        """Open the dependencies manager and land directly on AI Engines.

        The button that triggers this is labelled "AI Engines", so the
        user expects to land on that tab without an extra click.
        """
        try:
            dialog = DependenciesDialog(self, initial_tab="ai_engines")
            dialog.exec()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "AI Engines Manager error",
                f"Could not open the AI Engines Manager:\n\n{exc}\n\n"
                "See bin/logs for details."
            )

    def _on_manage_engines_requested(self, category):
        """Bridge for ``AIWorkflowStep.clicked_for_manage``.

        Opens ``DependenciesDialog`` directly on the AI Engines tab so
        the user lands in the right context without having to click
        around. ``category`` is captured for future use (e.g. a future
        iteration could pre-filter the catalog by category) but today
        we just default to the engines tab.
        """
        try:
            dialog = DependenciesDialog(self, initial_tab="ai_engines")
            dialog.exec()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "AI Engines Manager error",
                f"Could not open the AI Engines Manager:\n\n{exc}\n\n"
                "See bin/logs for details."
            )

    def _on_model_selected_from_search(self, model: dict):
        """Handle model selection from the HuggingFace search dialog.

        Opens the DependenciesDialog on the AI Engines tab and triggers
        installation of the selected engine.

        Los modelos del Hub (``is_user``) se registran aquí en
        ``user_models`` antes de abrir el diálogo: el flujo de instalación
        (``DependenciesDialog._find_engine``) resuelve el engine_id contra
        el catálogo + registro, y si el id no existiera mostraría
        "Unknown engine_id". El registro es inofensivo sin pesos en disco
        (el sensor de instalado sigue necesitando mark_installed).
        """
        engine_id = model.get("engine_id", "")
        if not engine_id:
            return
        if model.get("is_user"):
            self._register_user_model(model)
        dialog = DependenciesDialog(self, initial_tab="ai_engines")
        # Trigger installation after the dialog is shown.
        dialog._on_engine_install(engine_id)
        dialog.exec()

    def _register_user_model(self, model: dict):
        """Persiste el registro de un motor de usuario (buscador del Hub)."""
        try:
            from src.ai.user_models import register as _register
            weight_file = ""
            for f in (model.get("files") or []):
                low = str(f).lower()
                if low.endswith((".safetensors", ".pth", ".pt", ".onnx")):
                    weight_file = str(f)
                    break
            _register(
                model.get("engine_id", ""),
                display_name=model.get("display_name", ""),
                category=model.get("category") or "Other",
                hf_repo_id=model.get("hf_repo_id", ""),
                hf_url=model.get("hf_url", ""),
                weight_file=weight_file,
                license=model.get("license", ""),
                popularity_score=model.get("popularity_score", 0),
                quality_score=model.get("quality_score", 0),
            )
        except Exception as exc:
            logger.warning("Failed to register user model %s: %s",
                           model.get("engine_id"), exc)

    def _show_preferences(self):
        dialog = PreferencesDialog(self)
        dialog.exec()

    def _show_preferences_history(self):
        """Abre la pestaña History del dialogo de preferencias.

        Disparado por ``update_banner.history_requested`` (boton Show History
        del estado success). Reutiliza ``PreferencesDialog`` con el parametro
        ``initial_tab="history"`` para que el usuario aterrice directamente
        sobre la tabla de updates.
        """
        dialog = PreferencesDialog(self, initial_tab="history")
        dialog.exec()

    # -------------------- closeEvent + auto-update on close ---------------------------
    def closeEvent(self, event):
        """Cierre con auto-update opcional (multi-dependencia).

        Casos, en orden:
        0. ``CompressionWorker`` activo (NO followup chain): cancelar
           el worker + borrar el output parcial antes de cerrar, así
           FFmpeg no se queda huérfano cuando Qt mata el proceso.
        1. ``_close_followup_complete``: chain + followup-compress ya
           pasaron OK → self.close() se llamo desde _on_worker_finished;
           cerramos sin re-prompt.
        2. ``_close_consumed``: chain OK + banner-success visible;
           el usuario hizo 2º click X → cerramos sin re-prompt.
        3. ``_close_after_worker``: banner-Update legacy FFmpeg-only,
           el worker termino, cerramos.
        4. ``_close_followup_job + worker.isRunning()``: el
           CompressionWorker del followup esta corriendo → ignore().
        5. ``_close_chain.is_running()``: chain en curso → ignore().
        6. Preferencia activa y hay deps pendientes: dialog
           (3-botones si hay last_job, 2-botones si no); arrancamos la
           cadena si el usuario acepta con cualquiera de las dos
           variantes-update.
        7. Resto: cierre normal sin propuesta.
        """
        # 0) Cancel any in-flight regular compress/preview BEFORE
        #    the auto-update dialog runs. The followup path
        #    (Update + Compress + close) keeps its own event.ignore()
        #    semantics — that branch is operated by the chain, not
        #    the X button alone. Without this short-circuit, ffmpeg
        #    would be left running as a child of the dying Qt process.
        #
        # If a render is in flight, first confirm with the user. A stray
        # click on X during a 30+ minute encode silently aborting the job
        # is the worst UX; the Abort button already exists for that.
        if (
            self._close_followup_job is None
            and not self._close_followup_complete
            and getattr(self, "worker", None) is not None
            and self.worker.isRunning()
        ):
            ret = QMessageBox.question(
                self,
                "Render in progress",
                "A render is in progress. Abort it and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                event.ignore()
                return
            self._abort_running_worker(cleanup_partial_output=True)
            self._cleanup_after_abort()

        # Banner FFmpeg update en vuelo: NO destruir el QThread mientras
        # corre. Marcamos _close_pending para que _on_banner_update_finished
        # cierre limpio al terminar (o informe del fallo y cierre igualmente).
        # Este guard va ANTES de _collect_pending_close_updates para que el
        # dialog de update-chain no arranque una descarga concurrente del
        # mismo binario.
        if (
            getattr(self, "_banner_update_worker", None) is not None
            and self._banner_update_worker.isRunning()
        ):
            self._close_pending = True
            event.ignore()
            return

        if self._close_followup_complete:
            super().closeEvent(event)
            return

        if self._close_consumed:
            super().closeEvent(event)
            return

        if self._close_after_worker:
            super().closeEvent(event)
            return

        if (
            self._close_followup_job is not None
            and hasattr(self, "worker")
            and self.worker.isRunning()
        ):
            event.ignore()
            return

        if self._close_chain is not None and self._close_chain.is_running():
            event.ignore()
            return

        pending = self._collect_pending_close_updates()
        if not pending:
            super().closeEvent(event)
            return

        decision = self._show_close_update_dialog(pending)
        if decision == "skip_close":
            super().closeEvent(event)
            return
        if decision == "cancel_close":
            event.ignore()
            return
        # decision in {"update", "update_and_compress"}
        if decision == "update_and_compress":
            # Capturamos ahora (no en _on_chain_finished) para que un
            # save_last_job posterior (de otro Compress/Preview concurrente)
            # no nos pise el snapshot que el usuario va a ejecutar.
            self._close_followup_job = load_pending_last_job()

        event.ignore()
        self._start_close_update_chain(pending)

    def _auto_update_enabled(self):
        """Lee la preferencia 'auto_update_all_on_close' (con fallback legacy)."""
        try:
            import src.dependencies as dependencies
            return bool(dependencies.auto_update_all_pref_enabled())
        except Exception:
            return False

    def _collect_pending_close_updates(self):
        """Devuelve la lista de deps con update disponible para una cadena.

        Filtra por la preferencia: si no esta activa, devuelve lista vacia
        (que en ``closeEvent`` significa 'cierra normal sin prompt').
        """
        if not self._auto_update_enabled():
            return []
        try:
            import src.dependencies as dependencies
            return list(dependencies.pending_updatable_dependencies())
        except Exception:
            return []

    def _show_close_update_dialog(self, pending):
        """Muestra el dialog de confirmacion con la lista de deps a tocar.

        Devuelve una de las cuatro decisiones:

        - ``"update_and_compress"``: arranca la cadena Y al terminar
          ejecuta el último job (compress o preview) fotografiado en
          ``bin/.session/last_job.json``.
        - ``"update"``: arranca solo la cadena. Banner-success con
          mini-stats al final; app queda abierta (2as X para cerrar).
        - ``"skip_close"``: equivalente al "No" de hoy — cerrar la app
          sin actualizar nada. Solo se devuelve cuando NO hay un last_job
          guardado, porque sin file guardado el usuario no necesita ese
          modo.
        - ``"cancel_close"``: ignorar el closeEvent y quedarse en la app
          sin actualizar. Solo se ofrece cuando HAY un last_job guardado
          (la 3a opcion del dialog de 3 botones).

        Lazy import: ``src.last_job`` se consulta aqui (solo 1 vez por
        intento de cierre) para no cargar modulo si la pref esta off.
        """
        labels = [
            dep.get("display_name", dep.get("dep_id", "?"))
            for dep in pending
        ]
        listing = "\n".join(f"  • {name}" for name in labels)

        if not has_pending_last_job():
            # Sin last_job: conservamos la UX 2-botones de siempre para
            # que los usuarios sin cola activa no vean un dialog mas
            # confuso. Returns "update" / "skip_close".
            confirm = QMessageBox.question(
                self,
                "Update dependencies before close?",
                f"Hay {len(pending)} actualizacion(es) disponible(s):\n\n"
                f"{listing}\n\n"
                "Descargar y aplicar antes de cerrar? El detalle paso a "
                "paso queda en bin/logs/notifications.log.",
            )
            if confirm == QMessageBox.StandardButton.Yes:
                return "update"
            return "skip_close"

        # Con last_job: dialog de 3 botones. Cargamos el payload una sola
        # vez para mostrar el nombre del job al usuario y para que
        # closeEvent pueda capturarlo limpio.
        job = load_pending_last_job() or {}
        mode = job.get("mode", "compress")
        job_name = os.path.basename(job.get("input_path", "") or "?")

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Update + continue last job?")
        msg.setText(
            f"Hay {len(pending)} actualizacion(es) pendiente(s):\n"
            f"{listing}\n\n"
            f"Además, tu último {mode} sin terminar ('{job_name}') sigue "
            f"en cola.\n¿Qué hacemos antes de cerrar?"
        )
        btn_chain = msg.addButton(
            f"Update + {mode.title()} + close",
            QMessageBox.ButtonRole.AcceptRole,
        )
        btn_update = msg.addButton(
            "Update only", QMessageBox.ButtonRole.YesRole,
        )
        btn_cancel = msg.addButton(
            "Cancel close", QMessageBox.ButtonRole.RejectRole,
        )
        # Default + Escape a Cancel: usuarios con keyboard-only que dan
        # Enter o Escape esperando rechazar el proposal ahora SÍ cierran
        # el dialogo con "Cancel close" en lugar de arrancar una cadena
        # que no querian. RejectRole garantiza el resto del flujo.
        msg.setDefaultButton(btn_cancel)
        msg.setEscapeButton(btn_cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_chain:
            return "update_and_compress"
        if clicked == btn_update:
            return "update"
        return "cancel_close"

    def _start_close_update_chain(self, pending):
        """Arranca ``_UpdateChainController`` con la lista de deps.

        Cablea todas las signals al GUI; cada paso escribe a
        ``notifications.log`` con icono INFO/WARN. Al final, ``_on_chain_finished``
        acepta el cierre via ``self.close()`` re-entrante.
        """
        self._close_pending = True
        self.status_label.setText("Preparing updates...")
        self.progress.setValue(0)
        self.update_banner.hide_banner()
        # Bloquea interacciones: el chain reemplaza binarios, no queremos
        # que el usuario dispare una compresion mientras los binarios
        # estan a mitad de descarga.
        self.btn_compress.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_abort.setEnabled(False)
        # Reusar el _start_close_update legacy (FFmpeg-only) sigue
        # sirviendo para el banner-Update; el chain es ortogonal.

        # Log diagnostico: anotar skipped (is_updatable=True sin reinstall_fn).
        try:
            import src.dependencies as dependencies
            skipped = dependencies.report_skipped_dependencies()
            for s in skipped:
                notify_log(
                    "info", "auto-update chain skipped slot",
                    f"{s.get('dep_id', '?')} ({s.get('reason', '?')})",
                )
        except Exception:
            pass
        notify_log(
            "info", "auto-update chain started",
            f"{len(pending)} dep(s) queued: "
            + ", ".join(d.get("dep_id", "?") for d in pending),
        )

        self._close_chain = _UpdateChainController(deps=pending, parent=self)
        self._close_chain.chain_started.connect(self._on_chain_started)
        self._close_chain.step_started.connect(self._on_chain_step_started)
        self._close_chain.step_progress.connect(self.progress.setValue)
        self._close_chain.step_status.connect(self.status_label.setText)
        self._close_chain.step_finished.connect(self._on_chain_step_finished)
        self._close_chain.chain_finished.connect(self._on_chain_finished)
        self._close_chain.start()

    def _on_chain_started(self, total):
        self.status_label.setText(f"Auto-update chain: 0/{total}")

    def _on_chain_step_started(self, idx, total, dep_id, label):
        self.progress.setValue(0)
        self.status_label.setText(
            f"Updating {label} ({idx + 1}/{total})..."
        )

    def _on_chain_step_finished(self, idx, dep_id, ok, message):
        # El log por paso ya lo escribe el controller (notify_log INFO/WARN).
        # Aqui solo actualizamos el status para que el usuario vea el avance.
        if not ok:
            self.status_label.setText(
                f"Step {dep_id} failed ({message}); continuing..."
            )

    def _on_chain_finished(self, summary):
        ok = int(summary.get("ok", 0))
        failed = int(summary.get("failed", 0))
        total = int(summary.get("total", 0))
        results = summary.get("results", []) or []

        # Log final con icono segun el resultado.
        notify_log(
            "info" if failed == 0 else "warn",
            "auto-update chain finished",
            f"ok={ok} failed={failed} total={total}",
        )

        # Calcular mini-stats agregados para el banner-success: sumamos los
        # deltas de bytes y la duracion de los steps exitosos. Cada entry
        # ``results[i]`` ya incluye ``old_bytes`` / ``new_bytes`` /
        # ``duration_seconds`` que el controller capturo.
        successful = [r for r in results if r.get("ok")]
        total_old = sum((r.get("old_bytes") or 0) for r in successful)
        total_new = sum((r.get("new_bytes") or 0) for r in successful)
        total_duration = sum(
            float(r.get("duration_seconds") or 0.0) for r in successful
        )

        # Reset botones y banners antes del nuevo estado del banner.
        self.btn_compress.setEnabled(bool(self.input_path))
        self.btn_preview.setEnabled(bool(self.input_path))
        self.btn_abort.setEnabled(False)
        self.progress.setValue(100)

        # ---------------- Followup dispatch -----------------
        # Si el usuario eligió "Update + Compress + close" tenemos que
        # ejecutar el último job compress/preview fotografiado en
        # bin/.session/last_job.json DESPUÉS de la cadena y ANTES del
        # cierre. Lo disparamos aquí mismo para reusar CompressionWorker
        # sin duplicar la maquina de estado.
        if self._close_followup_job is not None:
            if failed == 0 and ok > 0:
                job = self._close_followup_job
                input_path = job.get("input_path", "")
                if input_path and os.path.exists(input_path):
                    # Stasheamos el summary para combinarlo en una sola
                    # notificación tray cuando el CompressionWorker termine.
                    self._update_chain_summary = summary
                    job_settings = dict(job.get("settings") or {})
                    job_output = job.get("output_path")
                    job_mode = job.get("mode", "compress")
                    self.status_label.setText(
                        f"Starting saved job: {os.path.basename(input_path)}..."
                    )
                    self.update_banner.hide_banner()
                    # Spawn CompressionWorker con el snapshot literal. NO
                    # releemos settings_panel aquí: el job es el de cuando
                    # el usuario hizo click, no los settings actuales.
                    # Incluimos ``ai_engines`` para que el followup respete
                    # la elección de IA que el usuario tenía al disparar
                    # el job original — sin esto, el followup arrancaría
                    # sin engines aunque el job original usase Real-ESRGAN.
                    job_ai_engines = job.get("ai_engines") or None
                    self._run_worker(
                        job_output,
                        job_settings,
                        job_mode,
                        ai_engines=job_ai_engines,
                        input_path=input_path,
                    )
                    return
                # input_path desapareció mid-update: skip silencioso del
                # followup + warning toast + caemos al path normal de
                # banner/tray abajo. El user verá el aviso.
                self._close_followup_job = None
                self._notify_completion(
                    "DRAGGY ENCODER updates",
                    f"Saved job input missing: {input_path}",
                    QSystemTrayIcon.MessageIcon.Warning,
                )
            else:
                # Chain no fue 100% OK: abandonamos el followup porque
                # no queremos comprimir con binarios a medio reemplazar.
                self._close_followup_job = None

        if successful:
            duration_detail = _fmt_dur(total_duration)
            dep_labels = [
                r.get("display_name") or r.get("dep_id", "?")
                for r in successful
            ]
            deps_str = ", ".join(dep_labels)
            banner_title = "Update complete"
            if failed:
                banner_title += f" ({ok}/{total})"
            banner_detail = (
                f"{deps_str} · {_fmt_bytes(total_old)} → {_fmt_bytes(total_new)} "
                f"in {duration_detail}"
            )
            self.update_banner.show_success(banner_title, banner_detail)
            icon = QSystemTrayIcon.MessageIcon.Information
            tray_title = "DRAGGY ENCODER updates"
            tray_body = (
                f"{ok}/{total} actualizadas. {_fmt_bytes(total_old)} → "
                f"{_fmt_bytes(total_new)} in {duration_detail}. "
                f"Detalle paso a paso en bin/logs/notifications.log."
            )
        else:
            # Cadena enteramente fallida: el banner no muestra exito,
            # solo el toast del system tray y un mensaje en status bar.
            icon = QSystemTrayIcon.MessageIcon.Warning
            tray_title = "DRAGGY ENCODER updates (failed)"
            tray_body = (
                f"{failed} actualizacion(es) fallaron. "
                f"Detalle en bin/logs/notifications.log."
            )

        # Status final siempre visible. La rama failed gana cuando los exitos
        # son cero (banner-success no se muestró pero si hubo intentos):
        if failed and not successful:
            self.status_label.setText(
                f"Auto-update failed: {failed} step(s) — ver notifications.log"
            )
        elif failed:
            self.status_label.setText(
                f"Updates finished: {ok} ok, {failed} failed"
            )
        elif successful:
            # Sin failed: dejamos el banner-detail en el status bar para
            # que el usuario vea el resumen aunque pase por alto el banner.
            self.status_label.setText(banner_detail)

        self._notify_completion(tray_title, tray_body, icon)

        # Cleanup + NO auto-close: dejamos el banner-success visible para
        # que el usuario revise los stats. El segundo click en X cierra
        # via la rama ``_close_consumed`` de ``closeEvent``.
        self._close_chain = None
        self._close_pending = False
        self._close_after_worker = False
        self._close_consumed = True
        # Status final siempre visible.
        if successful:
            self.status_label.setText(banner_detail)

    def _shutdown_chain(self, *_):
        """Best-effort cleanup del chain cuando la ventana es destruida.

        Este handler se dispara en el hilo GUI (el signal ``destroyed``
        de Qt es sincrono y bloqueante). Pedimos al controller que haga
        el soft-cancel; ver ``_UpdateChainController.request_stop`` para
        los detalles sobre por que hoy es soft y no hard.

        Idempotente + null-safe: si ``_close_chain`` ya fue limpiado por
        ``_on_chain_finished`` en otro tick del event loop, no hace nada.
        """
        if self._close_chain is None:
            return
        try:
            self._close_chain.request_stop(wait_ms=2000)
        except Exception:
            pass

    def _notify_completion(self, title, body, icon_kind):
        """Notificacion nativa con fallback a ``bin/logs/notifications.log``.

        QSystemTrayIcon requiere Action Center (Windows), NSUserNotification
        (macOS) o libnotify (Linux). Si el sistema no soporta o falla,
        appendeamos al log para que el usuario pueda revisarlo desde el
        boton ``Open logs`` del panel de dependencias.
        """
        if self.tray_icon is not None:
            try:
                self.tray_icon.showMessage(title, body, icon_kind, 6000)
                return
            except Exception:
                pass
        try:
            from src.config import get_logs_dir
            from datetime import datetime
            log_dir = get_logs_dir()
            os.makedirs(log_dir, exist_ok=True)
            with open(
                os.path.join(log_dir, "notifications.log"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"{title}: {body}\n"
                )
        except Exception:
            pass

    # -------------------- Banner --------------------------------------------------------
    def _refresh_banner_from_cache(self):
        """Lee el estado actual del cache upstream (instantaneo) sin red."""
        try:
            from src.ffmpeg.manager import get_install_info
            info = get_install_info()
        except Exception:
            return
        self._render_banner_state(info)

    def _refresh_banner_async(self):
        """Lanza un check upstream en background; cuando termina repinta el banner."""
        if self._banner_check_worker and self._banner_check_worker.isRunning():
            return
        self._banner_check_worker = _BannerCheckWorker(force=False, parent=self)
        self._banner_check_worker.done.connect(self._on_banner_check_done)
        self._banner_check_worker.finished.connect(self._on_banner_check_thread_finished)
        self._banner_check_worker.start()

    def _on_banner_check_done(self):
        self._refresh_banner_from_cache()

    def _on_banner_check_thread_finished(self):
        self._banner_check_worker = None

    def _render_banner_state(self, info):
        update_available = info.get("update_available")
        last_checked = info.get("last_check_iso")
        last_mod_iso = (info.get("latest_fingerprint") or {}).get("last_modified_iso")

        # Si el usuario tiene ffmpeg-full (libplacebo), NO mostrar el banner
        # de update del FFmpeg estándar — el usuario ya tiene la versión mejorada.
        if info.get("libplacebo_available"):
            self.update_banner.hide_banner()
            return

        if update_available is True:
            title = "FFmpeg update available"
            detail_bits = []
            if last_mod_iso:
                detail_bits.append(f"upstream build {last_mod_iso.split('T', 1)[0]}")
            if last_checked:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(last_checked)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
                    if hours < 1:
                        detail_bits.append(f"checked {int(hours * 60)} min ago")
                    elif hours < 24:
                        detail_bits.append(f"checked {hours:.1f} h ago")
                    else:
                        detail_bits.append(f"checked {hours / 24:.1f} d ago")
                except Exception:
                    pass
            if info.get("upstream_error"):
                self.update_banner.show_offline(
                    "FFmpeg status unavailable",
                    "Could not reach the upstream: " + str(info["upstream_error"]),
                )
            else:
                self.update_banner.show_update(
                    title, " · ".join(detail_bits) or "Click Update to install now."
                )
        elif update_available is None and info.get("upstream_error"):
            self.update_banner.show_offline(
                "FFmpeg status unavailable",
                "Could not reach the upstream: " + str(info["upstream_error"]),
            )
        else:
            self.update_banner.hide_banner()

    def _on_banner_update_clicked(self):
        if self._banner_update_worker and self._banner_update_worker.isRunning():
            QMessageBox.information(
                self, "Update in progress",
                "Ya hay una actualizacion de FFmpeg en curso. Espera a que termine.",
            )
            return
        try:
            from src.ffmpeg.manager import get_install_info
            info = get_install_info()
            size_mb = "?"
            cl = (info.get("latest_fingerprint") or {}).get("content_length")
            if cl:
                size_mb = f"{int(cl) / (1024 * 1024):.1f} MB"
        except Exception:
            size_mb = "?"

        confirm = QMessageBox.question(
            self,
            "Update FFmpeg now",
            f"Descargar la build mas reciente de FFmpeg ({size_mb}) ahora?\n"
            "El binario actual sera reemplazado. Cualquier compresion en curso "
            "se abortara.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.btn_compress.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_abort.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Updating FFmpeg...")
        self.update_banner.btn_update.setEnabled(False)

        self._banner_update_worker = _FFmpegUpdateWorker(parent=self)
        self._banner_update_worker.progress.connect(self.progress.setValue)
        self._banner_update_worker.status.connect(self.status_label.setText)
        self._banner_update_worker.finished_ok.connect(self._on_banner_update_finished)
        self._banner_update_worker.failed.connect(self._on_banner_update_failed)
        self._banner_update_worker.finished.connect(self._on_banner_update_thread_finished)
        self._banner_update_worker.start()

    def _on_banner_update_finished(self):
        self.btn_compress.setEnabled(bool(self.input_path))
        self.btn_preview.setEnabled(bool(self.input_path))
        self.btn_abort.setEnabled(False)
        self.progress.setValue(100)
        self.status_label.setText("FFmpeg updated.")
        self.update_banner.hide_banner()

        if self._close_pending:
            # El usuario pidio cerrar mientras el banner-Update estaba en
            # vuelo: saltamos el popup y vamos directo al cierre nativo.
            self._close_pending = False
            self._close_after_worker = True
            self._notify_completion(
                "DRAGGY ENCODER",
                "FFmpeg updated successfully.",
                QSystemTrayIcon.MessageIcon.Information,
            )
            self.close()
            return

        QMessageBox.information(
            self, "FFmpeg updated",
            "Se ha instalado la nueva build de FFmpeg. "
            "El binario se ha reemplazado en bin/ffmpeg/.",
        )
        self._refresh_banner_from_cache()

    def _on_banner_update_failed(self, msg):
        self.btn_compress.setEnabled(bool(self.input_path))
        self.btn_preview.setEnabled(bool(self.input_path))
        self.btn_abort.setEnabled(False)
        self.status_label.setText(f"FFmpeg update failed: {msg}")
        self.update_banner.btn_update.setEnabled(True)

        if self._close_pending:
            self._close_pending = False
            self._close_after_worker = True
            self._notify_completion(
                "DRAGGY ENCODER",
                f"FFmpeg update failed: {msg}",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            self.close()
            return

        QMessageBox.critical(
            self, "FFmpeg update failed",
            f"No se pudo actualizar FFmpeg:\n\n{msg}",
        )

    def _on_banner_update_thread_finished(self):
        self._banner_update_worker = None

    def _get_output_path(self):
        settings = self.settings_panel.get_settings()
        base, ext = os.path.splitext(os.path.basename(self.input_path))
        output_name = settings.get("output_name") or f"{base}_compressed"
        fmt = settings.get("format", "mp4")
        return os.path.join(self.output_dir, f"{output_name}.{fmt}")

    def _compress(self):
        # Guard de reentrada: si ya hay un CompressionWorker corriendo,
        # no volver a snapshottear ni lanzar un segundo pipeline. El
        # boton se deshabilita en _run_worker() al final de su
        # construcción, pero los modals interactivos (p.ej. salud de
        # ffmpeg) bombean eventos y permiten un segundo click antes de
        # llegar allí.
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return
        if not self.input_path:
            QMessageBox.warning(self, "No video", "Please select a video first.")
            return
        if not self._maybe_ffmpeg_health_check():
            # El usuario canceló por sospecha de binario corrupto. No
            # silenciosamente: respetamos la decisión sin tocar prefs
            # ni last_job.
            return
        output_path = self._get_output_path()
        settings = self.settings_panel.get_settings()
        settings["pipeline_order"] = self.workflow.get_pipeline()
        # Snapshot del device/codec en prefs (si la pref
        # ``remember_last_codec_device`` está activa). Se hace en commit,
        # no a cada keystroke, para reflejar lo que realmente ejecutó.
        self._remember_runtime_prefs(settings)
        # Pull AI engine choices from the workflow widget. The dict maps
        # category -> engine_id (or "" for auto). Empty dict means
        # "user did not toggle any AI step"; not the same as None
        # which means "we never asked".
        ai_engines = self.workflow.get_selected_engines()
        # Snapshot ANTES del worker: si el usuario cierra justo después
        # y el proposal "Update + Compress + close" se dispara, este es
        # el job a reanudar (no los settings del panel, que pueden haber
        # cambiado). Persistente en bin/.session/last_job.json.
        persist_last_job(
            mode="compress",
            input_path=self.input_path,
            output_path=output_path,
            settings=settings,
            ai_engines=ai_engines,
        )
        self._run_worker(output_path, settings, mode="compress", ai_engines=ai_engines)

    def _preview(self):
        # Mismo guard que en _compress: no permitir un segundo worker
        # mientras el primero sigue corriendo.
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return
        if not self.input_path:
            QMessageBox.warning(self, "No video", "Please select a video first.")
            return
        if not self._maybe_ffmpeg_health_check():
            return
        settings = self.settings_panel.get_settings()
        settings["pipeline_order"] = self.workflow.get_pipeline()
        self._remember_runtime_prefs(settings)
        # Idem: la preview también cuenta como "último job" para que el
        # proposal ofrezca re-ejecutarla tras updates.
        ai_engines = self.workflow.get_selected_engines()
        persist_last_job(
            mode="preview",
            input_path=self.input_path,
            output_path=None,
            settings=settings,
            ai_engines=ai_engines,
        )
        self._run_worker(None, settings, mode="preview", ai_engines=ai_engines)

    def _run_worker(self, output_path, settings, mode, ai_engines=None, input_path=None):
        self.btn_compress.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self._worker_started = time.time()

        # Normal path: the currently selected video. Followup path: the
        # job's original input (the selection may have changed while the
        # update chain was running).
        src = input_path if input_path is not None else self.input_path

        self.worker = CompressionWorker(
            self.ffmpeg_path, src, output_path, settings,
            mode=mode, ai_engines=ai_engines,
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.status_message.connect(self.status_label.setText)
        # Direct slot + QueuedConnection: el callback SIEMPRE corre en el
        # hilo de la GUI. Lambda en hilo secundario fallaria silenciosamente.
        self.worker.finished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self.worker.start()

    def _on_worker_progress(self, value):
        """Actualiza la barra + ETA estimado a partir del throughput medido."""
        self.progress.setValue(value)
        try:
            elapsed = time.time() - getattr(self, "_worker_started", time.time())
            if value > 1 and elapsed > 2:
                remaining = elapsed * (100 - value) / value
                mm, ss = divmod(int(remaining), 60)
                hh, mm = divmod(mm, 60)
                if hh:
                    self.progress.setFormat(f"%p%  ·  ETA ~{hh}h {mm:02d}m {ss:02d}s")
                else:
                    self.progress.setFormat(f"%p%  ·  ETA ~{mm:02d}:{ss:02d}")
            else:
                self.progress.setFormat("%p%")
        except Exception:
            pass

    def _on_worker_finished(self):
        self.btn_compress.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress.setFormat("%p%")

        worker = getattr(self, "worker", None)
        if not worker:
            return

        # Detect user-initiated abort: skip the standard error popup
        # because the user already knows they cancelled. The partial
        # output file was already deleted by ``_abort_running_worker``;
        # we just keep the UI in sync in case the worker's ``finished``
        # signal arrives after we already cleaned manually (e.g. the
        # closeEvent path runs both the abort AND then re-enters
        # ``_on_worker_finished`` via the QThread lifecycle).
        if (
            getattr(worker, "abort_event", None) is not None
            and worker.abort_event.is_set()
        ):
            self._cleanup_after_abort()
            return

        # ---------------- Followup branch -----------------
        # Si el CompressionWorker es el followup del "Update + Compress
        # + close", NO mostramos los QMessageBox normales (bloquean
        # self.close() y rompen el flujo); emitimos una sola notificación
        # tray combinada y cerramos. Falla del followup -> banner-error y
        # app queda abierta (close_consumed True; 2º X cierra).
        if self._close_followup_job is not None:
            job = self._close_followup_job
            self._close_followup_job = None
            followup_mode = job.get("mode", "compress")
            chain_summary = self._update_chain_summary or {}
            chain_ok = int(chain_summary.get("ok", 0))
            chain_total = int(chain_summary.get("total", 0))
            input_basename = os.path.basename(
                job.get("input_path", "") or "?"
            )

            if not worker.success:
                self.status_label.setText(
                    f"Updates OK; auto-{followup_mode} failed: "
                    f"{worker.error_message or 'unknown'}"
                )
                self.update_banner.show_update(
                    "Updates OK; auto-compress failed",
                    f"Saved job could not run: {worker.error_message}. "
                    "Re-run manually from Compress / Preview.",
                )
                self._close_consumed = True
                self._update_chain_summary = None
                return

            # Followup preview OK: dejamos al usuario comparar frames
            # antes de cerrar. Si el load o el dialog fallan, cerramos
            # igual — el followup tecnicamente fue OK.
            if followup_mode == "preview":
                try:
                    if worker.preview_original and worker.preview_compressed:
                        opm = QPixmap()
                        cpm = QPixmap()
                        opm.loadFromData(worker.preview_original, "JPG")
                        cpm.loadFromData(worker.preview_compressed, "JPG")
                        if not opm.isNull() and not cpm.isNull():
                            dlg = PreviewDialog(
                                opm, cpm,
                                "Original (left) vs Compressed (right). Use the slider to compare.",
                                self,
                            )
                            dlg.exec()
                except Exception:
                    pass

            self._close_followup_complete = True
            self._update_chain_summary = None
            self._notify_completion(
                "DRAGGY ENCODER updates + auto-compress",
                f"{chain_ok}/{chain_total} actualizadas. "
                f"Saved {followup_mode} complete for {input_basename}. "
                f"Detalle paso a paso en bin/logs/notifications.log.",
                QSystemTrayIcon.MessageIcon.Information,
            )
            # Log persistente: aterriza en bin/logs/notifications.log
            # junto a los chain-step siblings y se ve gratis en la
            # tab History (si en el futuro el panel de dependencias
            # la consume). Mantiene continuidad de auditoría aunque
            # el status_label sea invisible (dead-pixels UX post-close).
            notify_log(
                "info",
                "followup compress finished",
                f"{input_basename} ({followup_mode}) after chain ok={chain_ok}/{chain_total}",
            )
            # Re-entra a closeEvent; el gate ``_close_followup_complete``
            # deja pasar sin re-prompt. El status_label ya no se setea
            # aquí porque la ventana se destruye inmediatamente despues
            # de self.close() y el texto seria invisible (dead-pixels UX);
            # la notificación tray ya lleva la misma informacion.
            self.close()
            return

        # ---------------- Standard worker-finished -----------------
        self.progress.setValue(100 if worker.success else 0)
        mode = getattr(worker, "mode", "compress")

        if not worker.success:
            # Un fallo de encode deja el output parcial (truncado/incompleto)
            # en el disko si no lo borramos; antes esto solo pasaba con los
            # aborts explícitos. Asegúrate de no inducir falsa confianza.
            output_path = getattr(worker, "output_path", None)
            if output_path and os.path.exists(output_path):
                self._try_remove(output_path)
            self.status_label.setText(f"Error: {worker.error_message or 'Operation failed'}")
            QMessageBox.critical(self, "Error", f"Operation failed: {worker.error_message or 'Unknown error'}")
            return

        if mode == "compress":
            self.status_label.setText("Compression finished!")
            QMessageBox.information(self, "Done", "Video compressed successfully.")
        elif mode == "preview":
            if not worker.preview_original or not worker.preview_compressed:
                self.status_label.setText("Preview failed: could not generate frames.")
                QMessageBox.critical(
                    self, "Preview Error",
                    "Could not generate preview frames. Check FFmpeg and the input video.",
                )
                return
            self.status_label.setText("Preview ready.")
            # QPixmap construidos en el hilo principal (regla Qt).
            original_pixmap = QPixmap()
            compressed_pixmap = QPixmap()
            if worker.preview_original:
                original_pixmap.loadFromData(worker.preview_original, "JPG")
            if worker.preview_compressed:
                compressed_pixmap.loadFromData(worker.preview_compressed, "JPG")
            if original_pixmap.isNull() or compressed_pixmap.isNull():
                self.status_label.setText("Preview failed: could not load frames.")
                QMessageBox.critical(self, "Preview Error", "Could not load preview frames.")
                return
            dialog = PreviewDialog(
                original_pixmap,
                compressed_pixmap,
                "Original (left) vs Compressed (right). Use the slider to compare.",
                self,
            )
            dialog.exec()

    def _abort(self):
        """User clicked the Abort button.

        Sets the worker's ``abort_event`` so ``run_ffmpeg`` terminates
        the ffmpeg subprocess cleanly, joins the worker thread with a
        bounded wait, falls back to ``worker.terminate()`` only as
        last resort, then deletes the partial output file. The
        previous implementation called ``worker.terminate()`` directly
        which killed the QThread wrapper but NOT the
        ``subprocess.Popen`` child — that was the orphan-ffmpeg bug.
        """
        if not self._abort_running_worker(cleanup_partial_output=True):
            return
        self._cleanup_after_abort()

    def _abort_running_worker(self, cleanup_partial_output):
        """Cancel an active ``CompressionWorker`` and tidy up disk.

        Reused by ``_abort`` (button click) and ``closeEvent`` (window
        closing during a regular compress). Returns True if it actually
        aborted something; False if no worker was running.

        Multi-stage strategy because each path covers a different
        failure mode of the previous one:

        1. Cooperative: signal ``run_ffmpeg`` via ``abort_event`` so
           the ffmpeg subprocess terminates on the next stdout line
           (millisecond-level response).
        2. Bounded wait: ``worker.wait(5000)`` — short clips finish
           in microseconds; large encodes get up to five seconds.
        3. Force-terminate the QThread wrapper if cooperative didn't
           land.
        4. Delete the partial output ffmpeg wrote under the ``-y``
           flag (always succeeds unless the OS holds the handle — see
           ``_try_remove``).
        """
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning():
            return False
        # 1) Cooperative path: ask run_ffmpeg to terminate ffmpeg.
        try:
            if getattr(worker, "abort_event", None) is not None:
                worker.abort_event.set()
        except Exception:
            pass
        # 2) Bounded wait. run_ffmpeg polls every stdout line so this
        #    finishes within microseconds for short clips.
        if not worker.wait(5000):
            # 3) Last-resort: force-terminate the QThread wrapper.
            try:
                worker.terminate()
            except Exception:
                pass
        worker.wait(2000)
        # 3b) Direct child-process kill: when ``worker.terminate()``
        #     runs, the Python wrapper thread dies, but the ffmpeg
        #     ``subprocess.Popen`` child can survive (signals don't
        #     propagate to children). If ``run_ffmpeg`` hasn't polled
        #     ``abort_event`` yet — e.g. silent copy-mode encodes with no
        #     stdout line for several seconds — the child is reparented
        #     to init and our program exits orphaning it. Killing
        #     ``worker._proc`` directly is the safety net the
        #     cooperative path would have set up.
        proc = getattr(worker, "_proc", None)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        # 4) Delete partial output. On Windows a freshly terminated
        #    ffmpeg can hold the file handle briefly; _try_remove
        #    retries once on PermissionError.
        if cleanup_partial_output:
            outp = getattr(worker, "output_path", None)
            # Preview mode uses None for output_path (it goes through a
            # tempdir dropped automatically). Skip those.
            if outp and os.path.exists(outp):
                self._try_remove(outp)
        return True

    def _try_remove(self, path):
        """Best-effort ``os.remove`` with one transient-PermissionError retry.

        On Windows a freshly terminated ffmpeg can hold the file for a
        few hundred ms; one retry covers most cases without forcing a
        deep retry loop that would block closeEvent.
        """
        try:
            os.remove(path)
        except PermissionError:
            time.sleep(0.2)
            try:
                os.remove(path)
            except Exception as exc:
                print(
                    f"[Draggy] Could not delete partial {path}: {exc}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"[Draggy] Could not delete partial {path}: {exc}",
                file=sys.stderr,
            )

    def _cleanup_after_abort(self):
        """Reset UI state after the worker finished (successfully aborted)."""
        try:
            self.btn_compress.setEnabled(bool(self.input_path))
            self.btn_preview.setEnabled(bool(self.input_path))
            self.btn_abort.setEnabled(False)
        except Exception:
            pass
        try:
            self.progress.setValue(0)
        except Exception:
            pass
        # Surface the worker's error_message if any. ffmpeg can leave
        # a tail (e.g. "Conversion failed!") that explains why we
        # aborted — useful when the abort was triggered by a race
        # (e.g. the auto-update chain) rather than the user clicking
        # Abort directly. Empty ``message`` quietly hides the suffix.
        worker = getattr(self, "worker", None)
        tail = (worker.error_message or "") if worker is not None else ""
        suffix = f": {tail}" if tail else ""
        self.status_label.setText(
            f"Aborted — partial output cleaned up{suffix}."
        )

    def _on_redetect_clicked(self):
        if hasattr(self, "redetect_worker") and self.redetect_worker.isRunning():
            return
        self.btn_redetect.setEnabled(False)
        self.btn_compress.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Re-detecting hardware and codecs...")

        self.redetect_worker = RedetectWorker(self.ffmpeg_path)
        self.redetect_worker.progress.connect(
            lambda v, msg: (self.progress.setValue(v), self.status_label.setText(msg))
        )
        self.redetect_worker.done.connect(self._on_redetect_done)
        self.redetect_worker.error.connect(self._on_redetect_error)
        self.redetect_worker.start()

    def _on_redetect_done(self, hardware_info, compatible_codecs):
        self.btn_redetect.setEnabled(True)
        self.btn_compress.setEnabled(bool(self.input_path))
        self.btn_preview.setEnabled(bool(self.input_path))
        self.progress.setValue(100)
        self.hardware_info = hardware_info
        self.compatible_codecs = compatible_codecs

        recommended = hardware_info.get("recommended_device", "CPU")
        gpus = hardware_info.get("gpus", [])
        if gpus:
            first = gpus[0]
            gpu_text = f"{first.get('vendor', '')} {first.get('name', '')}".strip() or recommended
        else:
            gpu_text = recommended
        self.gpu_label.setText(gpu_text)

        self.settings_panel.refresh(hardware_info, compatible_codecs)
        recommended = hardware_info.get("recommended_device", "CPU")
        self.settings_panel.set_device(recommended)
        self.status_label.setText("Re-detection finished.")

    def _on_redetect_error(self, msg):
        self.btn_redetect.setEnabled(True)
        self.btn_compress.setEnabled(bool(self.input_path))
        self.btn_preview.setEnabled(bool(self.input_path))
        self.status_label.setText(f"Re-detection error: {msg}")
        QMessageBox.warning(self, "Re-detect Error", f"Could not re-detect: {msg}")

    # -------------------- Helpers de preferencias (Behavior extensions) --------------------

    def _resolve_initial_output_dir(self):
        """Carpeta de output inicial respetando ``auto_open_last_output``.

        Si la preferencia está activa y ``last_output_dir`` apunta a una
        carpeta que aún existe, úsala. Si no, cae al Desktop para no
        romper el arranque. Tolerante a fallos: cualquier excepción
        retorna el Desktop por defecto (fail-safe).
        """
        try:
            if preferences.get("auto_open_last_output", True):
                last_dir = preferences.get("last_output_dir", None)
                if (
                    isinstance(last_dir, str)
                    and last_dir
                    and os.path.isdir(last_dir)
                ):
                    return last_dir
        except Exception:
            pass
        return (
            os.path.expanduser("~\\Desktop")
            if os.name == "nt"
            else os.path.expanduser("~/Desktop")
        )

    def _restore_last_codec_device(self):
        """Aplica ``last_device`` + ``last_codec`` desde prefs si está activa
        la preferencia ``remember_last_codec_device``. Llamado desde
        ``_setup_ui`` justo después de configurar ``SettingsPanel``.
        ``set_device`` selecciona por nombre; ``set_codec`` (que acabamos
        de añadir) por ``codec_id`` (data del combo).
        """
        try:
            if not preferences.get("remember_last_codec_device", True):
                return
            last_device = preferences.get("last_device", None)
            last_codec = preferences.get("last_codec", None)
            if isinstance(last_device, str) and last_device:
                self.settings_panel.set_device(last_device)
            if isinstance(last_codec, str) and last_codec:
                self.settings_panel.set_codec(last_codec)
        except Exception:
            pass

    def _remember_runtime_prefs(self, settings):
        """Persiste ``last_device`` y ``last_codec`` en prefs al commit."""
        try:
            if not preferences.get("remember_last_codec_device", True):
                return
            device = ""
            codec = ""
            if isinstance(settings, dict):
                device = str(settings.get("device", "") or "")
                codec = str(settings.get("codec", "") or "")
            if isinstance(device, str) and device:
                preferences.set("last_device", device)
            if isinstance(codec, str) and codec:
                preferences.set("last_codec", codec)
        except Exception:
            pass
    def _maybe_ffmpeg_health_check(self):
        """Modal "FFmpeg unhealthy, ¿continuar?" si la pref está activa y
        el binario no responde a ``-version``. Devuelve ``True`` si se puede
        continuar (sin health-check, OK, o user forzó Yes).

        Fail-safe: cualquier excepción del runner se trata como "OK" para
        que un bug de ``health_check_ffmpeg`` NUNCA bloquee el flujo
        principal de compresión. La excepción llega a stderr.
        """
        try:
            if not preferences.get("ffmpeg_health_check_prompt", True):
                return True
        except Exception:
            return True
        if not self.ffmpeg_path or not os.path.exists(self.ffmpeg_path):
            # Sin binario, no hay nada que chequear; el _run_worker que
            # viene informará del fallo real más adelante.
            return True
        try:
            ok, _version, error = health_check_ffmpeg(
                self.ffmpeg_path, timeout=10
            )
        except Exception:
            return True
        if ok:
            return True
        msg = (
            f"FFmpeg no responde correctamente a -version:\n\n"
            f"  Binario: {self.ffmpeg_path}\n"
            f"  Detalle: {error or 'desconocido'}\n\n"
            "¿Continuar con la compresión de todos modos? Si el binario "
            "está corrupto, el job fallará después de gastar tiempo. "
            "Si acabas de copiar una build experimental y quieres "
            "probarla, pulsa Yes; en caso contrario cancela y revisa "
            "Manage Dependencies."
        )
        confirm = QMessageBox.question(
            self,
            "FFmpeg health check failed",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return confirm == QMessageBox.StandardButton.Yes
