import logging
import os
import shutil
import tempfile
import threading
import time

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from src.config import APP_NAME, get_asset_path
import src.preferences as preferences
from src.ffmpeg import builds
from src.ffmpeg.runner import get_audio_streams, get_video_info, health_check_ffmpeg
from src.ffmpeg.process_supervisor import ProcessSupervisor
from src.processing.pipeline import (
    compress_video,
    compress_video_with_ai,
    generate_preview,
    validate_ai_chain_contract,
    validate_render_preflight,
    validate_render_output,
)
from src.last_job import has_last_job as has_pending_last_job
from src.last_job import load_last_job as load_pending_last_job
from src.last_job import save_last_job as persist_last_job
from src.ui.dependencies_dialog import DependenciesDialog
from src.ui.dep_row import humanize_age
from src.task_persist import phase_writer
from src.ui.hardware_dialog import HardwareDialog
from src.ui.preferences_dialog import PreferencesDialog
from src.ui.preview_dialog import PreviewDialog
from src.ui.audio_preview import AudioPreviewEngine
from src.ui.settings_panel import SettingsPanel
from src.ui.tasks_panel import TasksPanel
from src.ui.tray_dashboard import TrayController, eta_text
from src.ui.metrics_strip import MetricsStripController
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
from src.ui.workflow_widget import WorkflowWidget, video_paths_from_mime
from src.utils.helpers import is_windows_elevated, detect_linux_container, parse_time_to_seconds

logger = logging.getLogger(__name__)


def _cleanup_preview_artifacts(artifacts):
    """Borra el directorio de artefactos del preview tras cerrar el diálogo.

    ``generate_preview`` persiste el clip comprimido en un ``mkdtemp`` para
    que el PreviewDialog pueda re-extraer frames sin recodificar. Este helper
    elimina ese directorio de forma best-effort (los artefactos viven en el
    directorio temporal del sistema; un fallo sólo deja basura temporal).
    """
    if not artifacts:
        return
    try:
        artifacts_dir = artifacts.get("dir")
        if artifacts_dir and os.path.isdir(artifacts_dir):
            shutil.rmtree(artifacts_dir, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Workers especializados
# ---------------------------------------------------------------------------

class _FFmpegUpdateWorker(QThread):
    """Descarga la nueva build de FFmpeg en background.

    Conecta directo al ``MainWindow.progress`` y ``status_label``. En éxito
    emite ``finished_ok``; cualquier error propaga el mensaje por ``failed``.
    Cancelable vía ``cancel()`` (setea un ``threading.Event`` que la descarga
    observa).
    """

    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.abort_event = threading.Event()

    def cancel(self):
        self.abort_event.set()

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

            force_redownload(
                progress_callback=_p, parent=self.parent(),
                abort_event=self.abort_event,
            )
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


class CompressionWorker(QThread):
    """Worker para ejecutar compresión o preview en segundo plano."""

    progress = Signal(int)
    status_message = Signal(str)
    phase_ev = Signal(dict)
    # Cifras de throughput (fps, multiplicador). Van aparte de ``phase_ev``
    # porque no son fases: no se guardan en el historial, no abren filas en el
    # panel y no se pintan alli. Solo las mira el desplegable de la bandeja.
    metrics_ev = Signal(dict)

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
        # means "no AI steps active right now". Compression and preview
        # resolve this snapshot into the same loaded inference chain.
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
        self.preview_artifacts = None
        self._procs = []
        self.supervisor = ProcessSupervisor()

    def _register_process(self, process, role="ffmpeg"):
        """Keep every child process visible to the abort/close path."""
        if process is None:
            return
        self._procs.append(process)
        self.supervisor.register(process, role)
        if role != "decoder":
            self._proc = process

    def run(self):
        try:
            status_cb = self._emit_status
            phase_cb = self._emit_phase
            metrics_cb = self._emit_metrics
            preflight_errors = validate_render_preflight(
                self.ffmpeg_path,
                self.input_path,
                self.output_path if self.mode == "compress" else None,
                self.settings,
            )
            if preflight_errors:
                self.error_message = "Preflight failed: " + " ".join(preflight_errors)
                self.status_message.emit(self.error_message)
                return
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

            if use_ai and self.mode in ("compress", "preview"):
                # Build the same AI CHAIN for compression and preview. The
                # preview must not silently fall back to an FFmpeg-only clip:
                # otherwise it shows a result different from the render the
                # user is about to launch.
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
                # en settings para que tanto render como preview dimensionen
                # correctamente el encoder.
                from src.processing.pipeline import _compute_total_scale
                total_scale = _compute_total_scale(engines)
                self.settings["ai_scale"] = total_scale
                self.settings["ai_native"] = any(
                    isinstance(v, dict) and bool(v.get("native"))
                    for v in (self.ai_engines or {}).values()
                )

                status_cb("AI chain: " + " -> ".join(chain_labels))
                if self.mode == "compress":
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
                        on_process=self._register_process,
                        on_processes=self._register_process,
                        phase_cb=phase_cb,
                        supervisor=self.supervisor,
                        metrics_cb=metrics_cb,
                    )
                else:
                    status_cb("AI preview: rendering the selected chain...")
                    self.preview_original, self.preview_compressed, self.preview_artifacts = generate_preview(
                        self.ffmpeg_path,
                        self.input_path,
                        self.settings,
                        progress_callback=self._emit_progress,
                        status_callback=status_cb,
                        ai_engines=self.ai_engines,
                        inference_engines=engines,
                        abort_event=self.abort_event,
                        on_process=self._register_process,
                        on_processes=self._register_process,
                        phase_cb=phase_cb,
                        supervisor=self.supervisor,
                        metrics_cb=metrics_cb,
                    )
                    self.success = bool(
                        self.preview_original and self.preview_compressed
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
                    on_process=self._register_process,
                    phase_cb=phase_cb,
                    supervisor=self.supervisor,
                    metrics_cb=metrics_cb,
                )
            elif self.mode == "preview":
                self.preview_original, self.preview_compressed, self.preview_artifacts = generate_preview(
                    self.ffmpeg_path,
                    self.input_path,
                    self.settings,
                    progress_callback=self._emit_progress,
                    status_callback=status_cb,
                    ai_engines=self.ai_engines,
                    abort_event=self.abort_event,
                    on_process=self._register_process,
                    phase_cb=phase_cb,
                    supervisor=self.supervisor,
                    metrics_cb=metrics_cb,
                )
                self.success = bool(
                    self.preview_original and self.preview_compressed
                )
                if not self.success and not self.error_message:
                    self.error_message = "Could not generate preview frames"
        except Exception as exc:
            self.success = False
            self.error_message = str(exc)
            logger.error("CompressionWorker exception (%s): %s", self.mode, exc)

    def _emit_progress(self, value):
        self.progress.emit(value)

    def _emit_phase(self, ev):
        try:
            self.phase_ev.emit(dict(ev or {}))
        except Exception:
            pass

    def _emit_metrics(self, ev):
        try:
            self.metrics_ev.emit(dict(ev or {}))
        except Exception:
            pass

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
            AIInferenceEngine,
            create_inference_engine,
            get_model_path,
            convert_pth_to_onnx_ex,
            PyTorchInferenceEngine,
        )
        from src.ai.architectures import is_torch_available

        engines = []        # loaded AIInferenceEngine, in chain order
        labels = []         # ["Upscale (Real-ESRGAN x4)", ...]
        errors = []         # human-readable per-node failures

        ai_map = self.ai_engines or {}

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

                # Backend de inferencia del motor. ``_get_backend`` es la
                # PREFERENCIA del usuario ("auto"|"torch"|"onnxruntime");
                # ``_resolve_backend`` la convierte en el backend CONCRETO:
                #   "auto"        -> el más rápido medido (ONNX vs PyTorch)
                #                    cuando existen benchmarks de ambos en
                #                    este equipo; si no, ONNX si hay .onnx
                #                    (GPU si es posible), si no PyTorch
                #   "torch"       -> forzar PyTorch: nunca usar .onnx ni
                #                    auto-convertir
                #   "onnxruntime" -> forzar ONNX: buscar/convertir .onnx;
                #                    nunca caer a PyTorch
                from src.ai.engines_state import get_backend as _get_backend
                from src.ai.engines_state import resolve_backend as _resolve_backend
                backend = _get_backend(eid)
                effective = _resolve_backend(eid)
                use_onnx = effective in ("auto", "onnxruntime")

                if use_onnx:
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
                            prefer_torch_over_cpu_onnx = (
                                effective == "auto"
                                and "CUDAExecutionProvider" not in active
                                and "TensorrtExecutionProvider" not in active
                                and pth_path
                                and is_torch_available()
                            )
                            if prefer_torch_over_cpu_onnx:
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
                            onnx_converted, conv_error = convert_pth_to_onnx_ex(pth_path)
                            if onnx_converted:
                                try:
                                    inference_engine = create_inference_engine("onnxruntime", onnx_converted)
                                except Exception as exc:
                                    node_errors.append(f"ONNX converted but load failed: {exc}")
                            elif conv_error:
                                node_errors.append(
                                    f"ONNX conversion skipped for {eid}: {conv_error}"
                                )
                        elif pth_path:
                            node_errors.append(
                                "PyTorch is not installed (required to convert .pth to ONNX)"
                            )

                # 3) PyTorch direct (backend 'torch' o 'auto' sin ONNX)
                if (
                    (inference_engine is None or not getattr(inference_engine, "is_loaded", False))
                    and effective in ("auto", "torch")
                ):
                    # Si el usuario eligió un archivo .onnx en el workflow
                    # pero forzó PyTorch, resolver el peso torch del engine.
                    torch_path = pth_path
                    if torch_path and torch_path.lower().endswith(".onnx"):
                        torch_path = get_model_path(eid, "torch")
                    if torch_path and is_torch_available():
                        try:
                            inference_engine = create_inference_engine("torch", torch_path)
                        except Exception as exc:
                            node_errors.append(f"PyTorch fallback failed: {exc}")
                    elif effective == "torch":
                        if not torch_path:
                            node_errors.append(
                                "Backend 'PyTorch' needs a .pth/.safetensors weight file; "
                                "only .onnx is present in the model folder."
                            )
                        elif not is_torch_available():
                            node_errors.append(
                                "Backend forced to PyTorch but PyTorch is not installed."
                            )

                # --- Fallback: try other weight files in the same directory ---
                # When the selected weight file fails (e.g. 12-channel model),
                # scan the engine directory for other .pth/.safetensors files
                # and try each one until one loads successfully.
                if inference_engine is None or not getattr(inference_engine, "is_loaded", False):
                    if pth_path and is_torch_available():
                        from src.ai.inference import list_weight_files, _pth_channel_count
                        engine_dir = os.path.dirname(pth_path)
                        for alt_path in list_weight_files(engine_dir, recursive=False):
                            if alt_path == pth_path:
                                continue
                            # Skip files with wrong channel count
                            _ch = _pth_channel_count(alt_path)
                            if _ch is not None and _ch != 3:
                                logger.debug(
                                    "Fallback: skipping %s (%d channels)",
                                    os.path.basename(alt_path), _ch,
                                )
                                continue
                            try:
                                inference_engine = create_inference_engine("torch", alt_path)
                                if inference_engine and getattr(inference_engine, "is_loaded", False):
                                    logger.info(
                                        "Fallback: loaded %s instead of %s",
                                        os.path.basename(alt_path), os.path.basename(pth_path),
                                    )
                                    break
                                inference_engine = None
                            except Exception as exc:
                                logger.debug(
                                    "Fallback: %s failed: %s", os.path.basename(alt_path), exc,
                                )
                                inference_engine = None

                if inference_engine is None or not getattr(inference_engine, "is_loaded", False):
                    detail = " | ".join(node_errors) if node_errors else "No model weights found."
                    hint = ""
                    if not node_errors or "No model weights" in detail:
                        hint = (
                            "This engine has no weight files installed. Open "
                            "AI Engines Manager and click 'Add Engine' on the "
                            "model row to download them."
                        )
                    if backend == "onnxruntime":
                        hint += (
                            "\nThis engine is set to 'ONNX Runtime' in the AI Engines "
                            "tab. The .pth is converted to .onnx automatically; if "
                            "that failed, switch to 'Auto' or 'PyTorch' in the "
                            "AI Engines Manager."
                        )
                    elif backend == "torch":
                        hint += (
                            "\nThis engine is set to 'PyTorch' in the AI Engines tab."
                        )
                    elif effective == "onnxruntime":
                        hint += (
                            "\nAuto chose ONNX Runtime (fastest measured on this "
                            "machine) but it could not be loaded. Switch to "
                            "'PyTorch' in the AI Engines Manager to force the "
                            "PyTorch backend."
                        )
                    raise RuntimeError(
                        f"{node_label} could not load.\nDetails: {detail}\n{hint}"
                    )

                # Escala objetivo: la del fichero elegido en el desplegable
                # del nodo Upscale. El desplegable sólo ofrece factores que
                # el fichero puede hacer DE VERDAD (se leen del modelo), así
                # que lo normal es que coincidan y aquí no haya nada que
                # forzar. Si algún camino antiguo trae un factor menor que
                # la nativa (p.ej. un motor del Hub con escala deducida del
                # catálogo), se comunica a la engine para que reporte esa y el
                # encoder re-encaje el tamaño. Con info.native NO se fuerza
                # nada: la engine reporta su escala real y el pipeline mide el
                # frame de la cadena para la resolución de salida.
                try:
                    _us = info.get("scale") if isinstance(info, dict) else None
                    _us_int = int(_us) if _us else 0
                except (TypeError, ValueError):
                    _us_int = 0
                _is_native = bool(
                    isinstance(info, dict) and info.get("native")
                )
                if (
                    _us_int > 1
                    and not _is_native
                    and isinstance(inference_engine, AIInferenceEngine)
                ):
                    inference_engine._target_scale = _us_int
                elif _is_native and isinstance(inference_engine, AIInferenceEngine):
                    # Garantizar que la engine reporte su escala NATIVA real
                    # (algunas engines ONNX solo la detectan fiablemente por
                    # shapes; si no coincide, el pipeline mide el frame real).
                    inference_engine._target_scale = None

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

        # Contracto estricto de la cadena: cada engine debe consumir y
        # devolver frames BGR uint8 (H, W, 3). Un nodo que devuelve RGB,
        # otro número de canales o cambia la resolución sin ser un
        # upscaler se detecta AHORA (con un frame de prueba barato), no
        # en el frame 5000 del render. Se degrada a warning: el render
        # sigue siendo posible si el usuario lo desea.
        try:
            validate_ai_chain_contract(engines)
            status_cb("AI chain contract validated (BGR/uint8/HWC)")
        except RuntimeError as exc:
            logger.warning("AI chain contract violation: %s", exc)
            status_cb(f"Warning: {exc}")
        except Exception as exc:
            logger.warning("AI chain contract probe failed: %s", exc)
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
        self.input_paths = []  # multi-selección (uno o más videos)
        # Dos slots de worker independientes: el render principal
        # (compress / item de cola) y el preview, que puede correr en
        # paralelo por ser un clip corto.
        self.worker = None
        self.preview_worker = None
        self._preview_task_id = None
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
        # La franja de metricas del borde inferior. Se queda en None
        # hasta que _setup_ui la construya: los helpers de muestreo se
        # llaman tambien antes (y despues) de que exista, y tienen que
        # aguantar ese None.
        self._metrics_strip = None
        self._setup_tray_icon()
        self._setup_ui()

    def _setup_tray_icon(self):
        """El icono de la bandeja: el desplegable y las dos salidas.

        Toda la logica vive en ``TrayController`` —el menu, el temporizador de
        muestreo y el canal de metricas—, que se puede probar sin montar esta
        ventana. Aqui solo se comprueba si hay bandeja, se construye el
        controlador con sus dos callbacks, y se le entrega el menu.

        Sin bandeja (``tray_icon is None``) el programa sigue igual, que es lo
        que pasaba antes de que existiera el desplegable.
        """
        self._tray = None
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
        self._tray = TrayController(
            self, on_open=self._on_tray_show, on_quit=self.close)
        self.tray_icon.setContextMenu(self._tray.menu())
        self.tray_icon.show()

    def _on_tray_show(self):
        """Traer la ventana al frente sin stealar el foco a ciegas.

        ``showNormal`` + ``raise_`` + ``activateWindow``: en Windows, si la
        ventana estaba minimizada hay que restaurar antes de traerla, o la
        llamada se queda en el aire sin error.
        """
        try:
            if self.isMinimized():
                self.showNormal()
            else:
                self.show()
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def _metrics_sampling_start(self):
        """Hay un render en marcha: la franja del borde inferior arranca.

        No abre NVML aqui. Eso lo hace el primer tic del controlador, que
        es justo cuando ya hay algo que medir; construir la ventana no se
        lleva por delante la inicializacion del driver.
        """
        strip = getattr(self, "_metrics_strip", None)
        if strip is not None:
            strip.start()

    def _metrics_sampling_stop(self):
        """Se acabó el render: la franja vuelve a "—" y se suelta NVML."""
        strip = getattr(self, "_metrics_strip", None)
        if strip is not None:
            strip.stop()

    def _tray_sampling_start(self):
        """Hay un render en marcha. Arranca la bandeja y la franja.

        Un solo interruptor para los dos: si uno de ellos se encendiera
        en un camino del render y el otro no, acabarian pidiendo NVML dos
        veces y mediendo cosas distintas en la misma pantalla.
        """
        self._metrics_sampling_start()
        if self._tray is not None:
            self._tray.start()
            self._tray.set_task(getattr(self, "_current_task_name", ""))
            if getattr(self, "_queue_active", False):
                self._tray.set_queue(
                    getattr(self, "_queue_done", 0),
                    getattr(self, "_queue_total", 0),
                )

    def _tray_sampling_stop(self):
        """Se acabó el render: se para el muestreo y se suelta NVML."""
        self._metrics_sampling_stop()
        if self._tray is not None:
            self._tray.stop()

    def _on_worker_metrics(self, ev):
        """Cifras del worker. Vive en el hilo de la GUI."""
        if self._tray is not None:
            self._tray.update_metrics(ev)

    def _setup_ui(self):
        self.setWindowTitle(f"{APP_NAME} v1")
        # Lowered from 1100×800 so the window can be used at 1024×640:
        # thanks to the new scroll areas (top bar, workflow chain,
        # settings panel) the layout no longer needs to force a huge
        # minimum to avoid the clip/overlap bug in your screenshot.
        self.setMinimumSize(1024, 640)
        # Icono de ventana explícito: la taskbar de Windows usa el icono de
        # la ventana (no el del exe). QApplication.setWindowIcon ya lo cubre
        # en la mayoría de casos, pero ser explícito evita el icono genérico
        # en WMs que no propagan el default.
        _logo = get_asset_path("Import.png")
        if _logo and os.path.exists(_logo):
            self.setWindowIcon(QIcon(_logo))
        self.setStyleSheet(self._stylesheet())
        # Drag & drop de videos en CUALQUIER punto de la ventana (no solo
        # sobre la casilla Import): el evento sube desde los widgets hijo
        # que no aceptan el drop y se resuelve aquí, igual que la casilla.
        self.setAcceptDrops(True)

        central = QWidget()
        self.setCentralWidget(central)
        # Layout raiz en vertical, aunque a dia de hoy solo tenga el
        # splitter dentro: es lo que deja la franja de metricas abajo del
        # todo y cruzando el panel de tareas. Con un solo widget, cambiar
        # la orientacion no mueve nada de lo que ya habia.
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Main column (everything the app had before the tasks panel).
        main_col = QWidget()
        layout = QVBoxLayout(main_col)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Top buttons
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)
        self.btn_runtimes = QPushButton("Dependencies")
        self.btn_runtimes.setToolTip(
            "FFmpeg, PyTorch, ONNX Runtime: download, update and uninstall "
            "the non-AI binaries and runtimes."
        )
        self.btn_runtimes.clicked.connect(self._show_runtimes)
        self.btn_output = QPushButton(f"Output: ...{os.path.basename(self.output_dir)}")
        self.btn_output.clicked.connect(self._select_output)
        self.btn_hardware = QPushButton("Hardware Info")
        self.btn_hardware.setToolTip(
            "Hardware detectado y códecs disponibles. La redetección de "
            "hardware y códecs está dentro de esa ventana."
        )
        self.btn_hardware.clicked.connect(self._show_hardware)
        self.btn_ai_engines = QPushButton(" AI Engines")
        self.btn_ai_engines.setIcon(QIcon(get_asset_path("AI panel.png")))
        self.btn_ai_engines.setToolTip(
            "AI model catalog: download, convert and remove upscale, "
            "interpolation, denoise, colorize and face-restore engines."
        )
        self.btn_ai_engines.clicked.connect(self._show_dependencies)
        self.btn_preferences = QPushButton("Preferences")
        self.btn_preferences.clicked.connect(self._show_preferences)
        for _btn in (self.btn_runtimes, self.btn_output, self.btn_hardware,
                     self.btn_ai_engines, self.btn_preferences):
            _btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            _btn.setMinimumWidth(0)

        top_layout.addWidget(self.btn_runtimes)
        top_layout.addWidget(self.btn_output)
        top_layout.addWidget(self.btn_hardware)
        top_layout.addWidget(self.btn_ai_engines)
        top_layout.addWidget(self.btn_preferences)
        layout.addLayout(top_layout)

        # Banner persistente de "Update available". Si esta oculto, QVBoxLayout
        # hace que la fila colapse automaticamente.
        self.update_banner = UpdateBanner()
        self.update_banner.update_requested.connect(self._on_banner_update_clicked)
        self.update_banner.panel_requested.connect(self._show_runtimes)
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

        # Aviso de elevación en Windows (drag & drop bloqueado por UIPI).
        self.elevation_banner = self._build_elevation_warning_banner()
        if self.elevation_banner is not None:
            layout.addWidget(self.elevation_banner)

        # Aviso de sandbox Flatpak (drag & drop del host limitado).
        self.sandbox_banner = self._build_sandbox_warning_banner()
        if self.sandbox_banner is not None:
            layout.addWidget(self.sandbox_banner)

        # Workflow — the chain is ~1600 px wide (Import 150 + 5×200 +
        # Compress 90 + arrows + reorder gaps). It must never be squeezed
        # or clipped when the window is narrow; instead a horizontal bar
        # appears (your screenshot bug). The steps keep their fixed size.
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
        self.workflow.active_changed.connect(
            self._on_ai_activity_changed
        )
        self.workflow.video_dropped.connect(self._on_video_dropped)
        # Click en la casilla Import = seleccionar videos (el botón
        # "Select Videos" de la barra superior ya no existe).
        self.workflow.import_clicked.connect(self._select_video)

        self.workflow_scroll = QScrollArea()
        self.workflow_scroll.setObjectName("workflowScroll")
        self.workflow_scroll.setWidgetResizable(False)
        self.workflow_scroll.setFrameShape(QFrame.NoFrame)
        self.workflow_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.workflow_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.workflow_scroll.setWidget(self.workflow)
        self.workflow_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # 150 px step height + 10+10 layout margins + 6 px bar
        self.workflow_scroll.setMinimumHeight(184)
        self.workflow_scroll.setMaximumHeight(188)
        self.workflow_scroll.setStyleSheet(
            "#workflowScroll { background: transparent; border: none; }"
            "QScrollBar:horizontal { height: 8px; background: #16213e; border-radius: 4px; }"
            "QScrollBar::handle:horizontal { background: #00d2d3; border-radius: 4px; }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }"
        )
        layout.addWidget(self.workflow_scroll)

        # Settings — wrapped in a vertical scroll so the panel never
        # overlaps / gets clipped when the window is short (your screenshot:
        # the bottom half was squashed with no vertical bar). The grid keeps
        # its ideal height; the scroll appears instead.
        self.settings_panel = SettingsPanel(
            hardware_info=self.hardware_info,
            compatible_codecs=self.compatible_codecs,
        )
        self.settings_panel.set_device(recommended)
        # Restaura codec/device desde prefs si la pref
        # ``remember_last_codec_device`` está activa. Idempotente:
        # no hace nada si los valores no son válidos o la pref está off.
        self._restore_last_codec_device()

        # Efectos de audio guardados + estado del desplegable. Se restaura al
        # construir la ventana para que el usuario recupere sus ajustes.
        try:
            self.settings_panel.restore_audio_fx()
        except Exception:
            pass

        # Motor de escucha de los efectos (genera un fragmento y lo reproduce).
        # Se conecta aquí porque necesita la ruta del vídeo y el ffmpeg de la
        # ventana, que el panel no conoce.
        self._audio_preview = AudioPreviewEngine(self)
        self._wire_audio_preview()

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setObjectName("settingsScroll")
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.settings_scroll.setWidget(self.settings_panel)
        self.settings_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.settings_scroll.setStyleSheet(
            "#settingsScroll { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 8px; background: #16213e; border-radius: 4px; }"
            "QScrollBar::handle:vertical { background: #2c3e50; border-radius: 4px; }"
            "QScrollBar:horizontal { height: 8px; background: #16213e; border-radius: 4px; }"
            "QScrollBar::handle:horizontal { background: #2c3e50; border-radius: 4px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,"
            " QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { height: 0px; width: 0px; }"
        )
        layout.addWidget(self.settings_scroll, stretch=1)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)
        self.btn_compress = QPushButton("Compress")
        self.btn_compress.clicked.connect(self._compress)
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self._preview)
        # Campo opcional: instante exacto a comparar en el preview
        # (HH:MM:SS o MM:SS o segundos). Vacío = frame aleatorio.
        self.preview_time_label = QLabel("Frame at")
        self.preview_time_label.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        self.preview_time_edit = QLineEdit()
        self.preview_time_edit.setPlaceholderText("auto")
        self.preview_time_edit.setToolTip(
            "Preview frame instant (HH:MM:SS, MM:SS or seconds).\n"
            "Empty = random frame. You can still scrub the time in the dialog."
        )
        self.preview_time_edit.setFixedWidth(90)
        self.preview_time_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_add_queue = QPushButton("Add to queue")
        self.btn_add_queue.setToolTip(
            "Queue this render (video + AI chain + settings).\n"
            "It will run later when you press Start queue."
        )
        self.btn_add_queue.clicked.connect(self._add_to_queue)
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._abort)
        for _btn in (self.btn_compress, self.btn_preview, self.btn_add_queue, self.btn_abort):
            _btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            _btn.setMinimumWidth(0)

        bottom_layout.addWidget(self.btn_compress)
        bottom_layout.addWidget(self.btn_preview)
        bottom_layout.addWidget(self.preview_time_label)
        bottom_layout.addWidget(self.preview_time_edit)
        bottom_layout.addWidget(self.btn_add_queue)
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

        # --- Tasks panel (right side, collapsible) ---
        self.tasks_panel = TasksPanel()
        self.tasks_panel.btn_clear.clicked.connect(self._on_clear_task_history)
        self.tasks_panel.btn_start_queue.clicked.connect(self._start_queue)
        self.tasks_panel.btn_clear_queue.clicked.connect(self._on_clear_queue)
        self.tasks_panel.cancel_requested.connect(self._on_cancel_queue_item)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.addWidget(main_col)
        self.splitter.addWidget(self.tasks_panel)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([980, 320])
        root_layout.addWidget(self.splitter)

        # Franja de metricas: GPU, CPU y VRAM abajo del todo y a lo ancho
        # de la ventana. Va en el layout raiz y no dentro de una columna
        # para que cruce tambien el panel de tareas, que es donde se mira
        # el progreso de una cola larga. El muestreo NO arranca aqui: lo
        # enciende y lo apaga el render (ver _metrics_sampling_start).
        self._metrics_strip = MetricsStripController(self)
        root_layout.addWidget(self._metrics_strip.widget())

        # Overlay a pantalla completa que se muestra mientras se arrastra un
        # video sobre la ventana: indica que se acepta el drop en cualquier
        # punto. Es transparente a eventos de ratón/drag para no interceptar
        # el hit-testing del drop (el evento sigue llegando al widget real
        # que hay debajo, p.ej. la casilla Import).
        self._drop_overlay = QLabel("Drop videos to import", central)
        self._drop_overlay.setObjectName("dropOverlay")
        self._drop_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_overlay.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._drop_overlay.setStyleSheet(
            "#dropOverlay {"
            " background-color: rgba(15, 15, 26, 215);"
            " color: #00d2d3;"
            " font-size: 26px;"
            " font-weight: bold;"
            " border: 3px dashed #00d2d3;"
            "}"
        )
        self._drop_overlay.hide()

        # Cargar el histórico persistente de tareas al arrancar.
        self._load_task_history()
        self._load_task_queue()

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
                "Open Dependencies to install them."
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

    @staticmethod
    def _build_elevation_warning_banner():
        """Advierte si la app corre elevada (admin) en Windows.

        Devuelve un ``QLabel`` de aviso, o ``None`` si el proceso no está
        elevado. Windows bloquea (UIPI) el drag & drop desde el Explorador
        hacia procesos elevados, así que el usuario debe relanzar la app
        desde una terminal normal para habilitar la importación por drop.
        """
        if not is_windows_elevated():
            return None
        banner = QLabel(
            "Running as administrator — Windows blocks drag & drop from "
            "Explorer into elevated apps (UIPI). Launch from a normal "
            "(non-admin) terminal to enable it."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel { background-color: #5f3a00; color: #ffffff; "
            "padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: bold; }"
        )
        return banner

    @staticmethod
    def _build_sandbox_warning_banner():
        """Advierte cuando la app corre en el sandbox de Flatpak.

        Devuelve un ``QLabel`` de aviso, o ``None`` si no hay limitación.
        En Flatpak el drag & drop de archivos desde el gestor del host no
        cruza la frontera del sandbox, así que la importación debe hacerse
        con el diálogo "Select Videos". AppImage no está aislado, por lo
        que no muestra aviso.
        """
        if detect_linux_container() != "flatpak":
            return None
        banner = QLabel(
            "Running in a Flatpak sandbox — dragging files from the host "
            "file manager into the app is not supported. Click the Import "
            "step in the workflow to select videos."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel { background-color: #5f3a00; color: #ffffff; "
            "padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: bold; }"
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
        """Selecciona UNO O VARIOS videos y los deja listos para la cola.

        La config (codec, resolución, cadena IA, ...) que el usuario ajuste
        después se aplica a TODOS los seleccionados en ``_add_to_queue``.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "",
            "Video Files (*.mp4 *.mkv *.avi *.wmv *.mov *.webm *.m4v);;All Files (*)",
        )
        if not paths:
            return
        self._accept_video_paths(paths)

    def _accept_video_paths(self, paths):
        """Fija la selección de videos a ``paths`` (archivos locales).

        Compartido por el diálogo de selección y el drag & drop sobre la
        casilla Import: ambos dejan los videos listos para la cola.
        """
        paths = [p for p in paths if p and os.path.exists(p)]
        if not paths:
            return
        self.input_paths = list(paths)
        # Backward compat: el primer video sigue siendo ``input_path`` para
        # los flujos single (preview / compress directo / last_job).
        self.input_path = paths[0]
        # Re-sincroniza botones: seleccionar vídeo siempre re-habilita
        # Compress/Preview si no hay worker ocupado (cura estados
        # atascados de rutas asíncronas que no llegaron a re-habilitar).
        self._update_action_buttons()
        if len(paths) == 1:
            self.status_label.setText(f"Selected: {os.path.basename(paths[0])}")
        else:
            self.status_label.setText(f"Selected: {len(paths)} videos")
        self._update_original_bitrate(paths[0])

    def _on_video_dropped(self, paths):
        """Drag & drop de videos sobre la casilla Import."""
        self._accept_video_paths(paths)

    # -- drag & drop sobre toda la ventana -----------------------------

    def _show_drop_overlay(self):
        overlay = getattr(self, "_drop_overlay", None)
        if overlay is None:
            return
        central = self.centralWidget()
        if central is not None:
            overlay.setGeometry(central.rect())
        overlay.raise_()
        overlay.show()

    def _hide_drop_overlay(self):
        overlay = getattr(self, "_drop_overlay", None)
        if overlay is not None:
            overlay.hide()

    def dragEnterEvent(self, event):
        if video_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            self._show_drop_overlay()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if video_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        # El arrastre salió del área de la ventana (o pasó a un hijo que
        # acepta el drop, p.ej. la casilla Import): retiramos el overlay.
        self._hide_drop_overlay()
        event.ignore()

    def dropEvent(self, event):
        paths = video_paths_from_mime(event.mimeData())
        self._hide_drop_overlay()
        if paths:
            self._accept_video_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _update_original_bitrate(self, path):
        # Probe every audio stream as well as the video metadata. The selected
        # ordinal is later propagated to FFmpeg as ``-map 0:a:N``.
        try:
            self.settings_panel.set_audio_tracks(
                get_audio_streams(self.ffmpeg_path, path)
            )
        except Exception:
            self.settings_panel.set_audio_tracks([])
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
            self.workflow.set_upscale_output_info("")
            return
        # An installed/default model is not an active pipeline step. Do not
        # advertise an output resolution until the user enables Upscale.
        for step in self.workflow.middle_steps:
            if getattr(step, "name", "") == "Upscale":
                if not getattr(step, "active", False):
                    self.workflow.set_upscale_output_info("")
                    return
                info = step.get_selected_model_info()
                engine_id = info.get("engine_id", "")
                scale = info.get("scale", 1)
                if not engine_id:
                    self.workflow.set_upscale_output_info("")
                    return
                if scale > 1:
                    out_w, out_h = w * scale, h * scale
                    self.workflow.set_upscale_output_info(
                        f"Output: {out_w}x{out_h} ({scale}x)"
                    )
                else:
                    self.workflow.set_upscale_output_info("")
                return

    def _update_ai_rendered_bitrate(self):
        """Show the estimate only when at least one AI node is active.

        Upscaling increases pixels and interpolation increases frames per
        second, so both factors affect the bitrate estimate. Denoise,
        colorize and face restoration keep the source dimensions but still
        count as an active AI pipeline and therefore show the unchanged
        baseline instead of a misleading always-visible label.
        """
        active_steps = [
            step for step in getattr(self.workflow, "middle_steps", [])
            if getattr(step, "active", False)
        ]
        w = getattr(self, "_input_width", 0)
        h = getattr(self, "_input_height", 0)
        orig_kbps = getattr(self, "_input_bitrate_kbps", 0)
        if not active_steps or not w or not h or not orig_kbps:
            self.settings_panel.set_ai_bitrate("")
            return

        scale = 1
        interpolation_factor = 1
        for step in active_steps:
            if getattr(step, "name", "") == "Upscale":
                info = step.get_selected_model_info()
                try:
                    scale *= max(1, int(info.get("scale", 1) or 1))
                except (TypeError, ValueError):
                    pass
            elif getattr(step, "name", "") == "Interp":
                # The current interpolation pipeline doubles the frame rate.
                interpolation_factor = 2

        pixel_ratio = float(scale * scale)
        estimated = max(1, int(orig_kbps * pixel_ratio * interpolation_factor))
        suffix = []
        if scale > 1:
            suffix.append(f"{scale}x pixels")
        if interpolation_factor > 1:
            suffix.append("2x FPS")
        reason = f"; {', '.join(suffix)}" if suffix else ""
        self.settings_panel.set_ai_bitrate(
            f"AI Rendered bitrate: ~{estimated:,} kbps (estimated{reason})"
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
        """Refresh AI-derived hints after any node/model change."""
        self._update_upscale_output_info()
        self._update_ai_rendered_bitrate()

    def _on_ai_activity_changed(self, node_name, active):
        """Refresh AI-derived hints when a node is toggled on or off."""
        self._update_upscale_output_info()
        self._update_ai_rendered_bitrate()

    def _show_hardware(self):
        """Abre la ventana de hardware, con los códecs y la redetección dentro.

        El botón de redetectar vive ahora en esa ventana (antes estaba en la
        barra superior): al pulsarlo se lanza el mismo worker de siempre y, al
        terminar, la ventana se refresca en el sitio sin cerrarse.
        """
        dialog = HardwareDialog(
            self.hardware_info,
            compatible_codecs=getattr(self, "compatible_codecs", None),
            parent=self,
            redetect_cb=self._on_redetect_clicked,
        )
        self._hardware_dialog = dialog
        try:
            dialog.exec()
        finally:
            self._hardware_dialog = None

    def _show_dependencies(self):
        """Open the AI Engines manager (model catalog only).

        The button that triggers this is labelled "AI Engines"; the
        non-AI binaries and runtimes (FFmpeg, PyTorch, ONNX Runtime)
        live in the separate "Dependencies" dialog (``_show_runtimes``).
        """
        self._open_dependencies(initial_tab="ai_engines")

    def _show_runtimes(self):
        """Open the Dependencies dialog (binaries & runtimes only).

        FFmpeg, PyTorch and ONNX Runtime install/update/uninstall.
        Separate from AI Engines so the model catalog never mixes
        with non-AI dependencies.
        """
        self._open_dependencies(initial_tab="binaries")

    def _on_manage_engines_requested(self, category):
        """Bridge for ``AIWorkflowStep.clicked_for_manage``.

        Opens the AI Engines dialog so the user lands in the right
        context without having to click around. ``category`` is
        captured for future use (e.g. a future iteration could
        pre-filter the catalog by category) but today we just default
        to the engines dialog.
        """
        self._open_dependencies(initial_tab="ai_engines")

    def _open_dependencies(self, initial_tab=None, install_engine_id=None):
        """Abre el diálogo de dependencias en el modo indicado.

        ``initial_tab``: "ai_engines" abre el gestor de modelos (solo
        catálogo); "binaries" (o None) abre Dependencies con los
        binarios y runtimes (FFmpeg, PyTorch, ONNX Runtime).
        """
        try:
            dialog = DependenciesDialog(self, initial_tab=initial_tab)
            if install_engine_id:
                # Disparar la instalación del motor seleccionado en el buscador.
                dialog._on_engine_install(install_engine_id)
            # Registrado mientras está abierto para que una redetección de
            # hardware pueda repintarlo; se borra al cerrar, igual que
            # ``_hardware_dialog``. Sin esta referencia el panel seguiría
            # enseñando un veredicto viejo hasta que el usuario lo cerrara.
            self._dependencies_dialog = dialog
            try:
                dialog.exec()
            finally:
                self._dependencies_dialog = None
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Dependencies error",
                f"Could not open the dependencies manager:\n\n{exc}\n\n"
                "See bin/logs for details."
            )

    def _on_model_selected_from_search(self, model: dict):
        """Handle model selection from the HuggingFace search dialog.

        Opens the AI Engines dialog and triggers installation of the
        selected engine.

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
        self._open_dependencies(initial_tab="ai_engines", install_engine_id=engine_id)

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
        # Lo que el escritor de fases tenga pendiente se escribe antes de
        # nada: aunque el cierre se acabe ignorando, el progreso ya esta en el
        # archivo antes de que se vuelva a mirar.
        #
        # ``take_unwritten`` devuelve True si habia algo sin avisar, y se lo
        # guarda para no repetir la noticia. Si lo hay, se avisa: el archivo
        # se queda en su ultimo estado correcto y el usuario sabe que hay un
        # render que no quedo guardado. Cerrar igualmente: quedarse aqui para
        # siempre seria el "no responde" que esto vino a quitar.
        try:
            if phase_writer.take_unwritten():
                logger.warning(
                    "Cerrando con progreso sin escribir: el archivo de tareas "
                    "se queda en su ultimo estado correcto.")
        except Exception:
            pass
        # La escucha de efectos abre un ffmpeg y un reproductor: se detienen
        # siempre, pase lo que pase con la cadena de auto-actualización, para
        # no dejar audio sonando ni un proceso huérfano tras cerrar.
        try:
            preview = getattr(self, "_audio_preview", None)
            if preview is not None:
                preview.cleanup()
        except Exception:
            pass
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
            and (
                (getattr(self, "worker", None) is not None and self.worker.isRunning())
                or (getattr(self, "preview_worker", None) is not None and self.preview_worker.isRunning())
            )
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

        # Abortar también el preview si está corriendo (clip corto, sin
        # confirmación extra — el usuario ya confirmó el cierre).
        if (
            getattr(self, "preview_worker", None) is not None
            and self.preview_worker.isRunning()
        ):
            self._abort_running_worker(
                cleanup_partial_output=True, worker=self.preview_worker
            )
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
            self._cerrar_ya(event)
            return

        if self._close_consumed:
            self._cerrar_ya(event)
            return

        if self._close_after_worker:
            self._cerrar_ya(event)
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
            self._cerrar_ya(event)
            return

        decision = self._show_close_update_dialog(pending)
        if decision == "skip_close":
            self._cerrar_ya(event)
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

    def _cerrar_ya(self, event):
        """El cierre que sí va en serio: para la franja y cierra.

        Va en un helper y no al principio de ``closeEvent`` porque este
        metodo tambien acaba en ``event.ignore()``: si un render está
        corriendo, o hay una cadena de auto-update en vuelo, o el usuario
        contesta "Cancelar" al dialogo, la ventana se queda abierta. Parar
        el muestreo ahi dejaria una franja congelada en el ultimo numero
        que se leyo, que es justo lo que hace parecer roto el panel.

        ``stop()`` es idempotente, asi que los cinco caminos por los que
        esto se puede llamar mas de una vez —incluido el reintento cuando
        el chain vuelve a pedir cerrar— no rompen nada.
        """
        self._metrics_sampling_stop()
        super().closeEvent(event)

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
        self._close_chain.step_status.connect(self._on_status_message)
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

        if update_available is True:
            # En una plataforma sin mirrors no hay nada que ofrecer, aunque el
            # upstream diga que hay versión nueva: anunciarla con el botón
            # Update vivo sería prometer una descarga que va a fallar. La
            # misma política que vacía el selector dice aquí que no hay nada.
            #
            # La plataforma se nombra explícitamente en vez de dejar que la
            # política la busque: el selector la recibe en sus datos, y este
            # banner va del equipo en el que corre, así que los tres leen de
            # un solo sitio y no pueden acabar hablando de sistemas distintos.
            plataforma = builds.current_platform()
            sin_descarga = builds.nothing_to_offer_reason(plataforma)
            if sin_descarga:
                # El motivo viene en minúscula para poder ir en medio de una
                # frase; aquí empieza párrafo, así que sólo se sube la primera
                # letra (``.capitalize()`` hundiría "FFmpeg" y "Darwin").
                self.update_banner.show_offline(
                    "FFmpeg not available for this system",
                    sin_descarga[:1].upper() + sin_descarga[1:] + ".",
                )
                return
            title = f"FFmpeg update available: {builds.chosen_build_label()}"
            detail_bits = []
            if last_mod_iso:
                detail_bits.append(f"upstream build {last_mod_iso.split('T', 1)[0]}")
            # La edad la formatea dep_row, que es su unico dueño: la fila del
            # panel y este banner dicen lo mismo para el mismo instante. Aqui
            # solo se decide si la frase se añade; sin fecha, no se añade.
            age = humanize_age(last_checked)
            if age:
                detail_bits.append(f"checked {age}")
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
        # El banner ya avisa de esto, pero el botón se puede pulsar igual
        # (queda vivo de un estado anterior). Preguntar "¿Descargar X?" aquí
        # sería prometer la misma descarga imposible que el banner ya niega.
        sin_descarga = builds.nothing_to_offer_reason(builds.current_platform())
        if sin_descarga:
            QMessageBox.information(
                self, "FFmpeg not available for this system",
                f"No se puede descargar FFmpeg en este sistema: {sin_descarga}.",
            )
            return
        try:
            from src.ffmpeg.manager import get_install_info
            info = get_install_info()
            size_mb = "?"
            cl = (info.get("latest_fingerprint") or {}).get("content_length")
            if cl:
                size_mb = f"{int(cl) / (1024 * 1024):.1f} MB"
            build_label = builds.chosen_build_label()
        except Exception:
            size_mb = "?"
            build_label = "FFmpeg"

        confirm = QMessageBox.question(
            self,
            "Update FFmpeg now",
            f"Descargar {build_label} ({size_mb}) ahora?\n"
            "Se comprueba que arranca antes de tocar nada: si la descarga, la "
            "extraccion o la comprobacion fallan, el FFmpeg que tienes ahora "
            "sigue instalado. Cualquier compresion en curso se abortara.",
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
        self._banner_update_worker.status.connect(self._on_status_message)
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

    def _get_output_path(self, input_path=None):
        settings = self.settings_panel.get_settings()
        src = input_path or self.input_path
        base, ext = os.path.splitext(os.path.basename(src))
        output_name = settings.get("output_name") or f"{base}_compressed"
        # Con varios videos en cola, un output_name fijo chocaría entre
        # ellos; lo prefijamos con el nombre base de cada archivo.
        if (
            output_name
            and output_name != f"{base}_compressed"
            and len(getattr(self, "input_paths", None) or []) > 1
        ):
            output_name = f"{output_name}_{base}"
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

    def _update_action_buttons(self):
        """Estado centralizado de los botones de acción.

        - Compress: requiere vídeo y que el worker principal esté libre.
        - Preview: requiere vídeo y que no haya OTRO preview en vuelo;
          puede correr EN PARALELO con el render principal (clip corto)
          para comparar el frame original vs el resultante sin esperar
          a que termine la cola.
        - Abort: activo si hay worker principal o preview corriendo.
        - Add to queue: siempre clicable; su guard interno informa al
          usuario si hay un render en curso.
        """
        main_running = (
            getattr(self, "worker", None) is not None and self.worker.isRunning()
        )
        preview_running = (
            getattr(self, "preview_worker", None) is not None
            and self.preview_worker.isRunning()
        )
        has_input = bool(self.input_path)
        self.btn_compress.setEnabled(has_input and not main_running)
        self.btn_preview.setEnabled(has_input and not preview_running)
        self.btn_abort.setEnabled(main_running or preview_running)

    def _preview(self):
        # El preview va en su PROPIO worker: puede lanzarse incluso con
        # un render/cola en curso (ver _run_preview_worker). Sólo se
        # bloquea si ya hay otro preview corriendo.
        if (
            getattr(self, "preview_worker", None) is not None
            and self.preview_worker.isRunning()
        ):
            self._on_status_message("A preview is already running.")
            return
        if not self.input_path:
            QMessageBox.warning(self, "No video", "Please select a video first.")
            return
        if not self._maybe_ffmpeg_health_check():
            return
        settings = self.settings_panel.get_settings()
        settings["pipeline_order"] = self.workflow.get_pipeline()
        self._remember_runtime_prefs(settings)
        # Instante exacto opcional del campo "Frame at". Vacío = aleatorio.
        # Se valida aquí; un valor inválido no bloquea el preview (cae a
        # aleatorio) pero sí avisa para que el usuario corrija el formato.
        preview_time_text = self.preview_time_edit.text().strip()
        if preview_time_text:
            try:
                settings["preview_time"] = parse_time_to_seconds(preview_time_text)
            except Exception:
                self._on_status_message(
                    "Preview frame time ignored (use HH:MM:SS, MM:SS or seconds)."
                )
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
        self._run_preview_worker(settings, ai_engines=ai_engines)

    def _run_preview_worker(self, settings, ai_engines=None):
        """Lanza el preview en ``self.preview_worker``.

        A diferencia de ``_run_worker`` (slot único del render principal),
        el preview tiene handlers propios para no pisar el progreso, el
        task_id ni la cola del render en curso.
        """
        main_running = (
            getattr(self, "worker", None) is not None and self.worker.isRunning()
        )
        if main_running and ai_engines:
            # El preview puede ejecutar la cadena IA mientras el render
            # principal comprime en GPU: competencia puntual por VRAM.
            # El clip es corto; se permite y se deja constancia en el log.
            logger.info(
                "[Draggy] Preview launched while a render is running; "
                "AI engines may contend for GPU/VRAM."
            )
        import uuid
        self._preview_task_id = str(uuid.uuid4())
        task_name = os.path.basename(self.input_path) + " (preview)"
        self.tasks_panel.get_or_create(self._preview_task_id, task_name)
        self.tasks_panel._cards[self._preview_task_id].set_running()
        try:
            import src.task_history as task_history
            task_history.start_task(
                self._preview_task_id, task_name, input_path=self.input_path,
            )
        except Exception:
            pass

        if not main_running:
            # Sin render en curso, el preview usa la barra principal
            # (comportamiento clásico). Con render, sólo status + tarjeta.
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self._worker_started = time.time()

        self.preview_worker = CompressionWorker(
            self.ffmpeg_path, self.input_path, None, settings,
            mode="preview", ai_engines=ai_engines,
        )
        self.preview_worker.task_id = self._preview_task_id
        self.preview_worker.progress.connect(self._on_preview_progress)
        self.preview_worker.status_message.connect(self._on_preview_status)
        self.preview_worker.phase_ev.connect(self._on_preview_phase_event)
        self.preview_worker.finished.connect(
            self._on_preview_finished, Qt.ConnectionType.QueuedConnection
        )
        self.preview_worker.start()
        self._update_action_buttons()

    def _on_preview_progress(self, value):
        """Progreso del preview: barra principal sólo si no hay render;
        si lo hay, status para no pisar el progreso del render."""
        main_running = (
            getattr(self, "worker", None) is not None and self.worker.isRunning()
        )
        if main_running:
            self.status_label.setText(f"Preview: {value}% (render in progress)")
        else:
            self._on_worker_progress(value)

    def _on_preview_status(self, message):
        self.status_label.setText(f"Preview: {message}")
        self.tasks_panel.append_log(f"Preview: {message}")

    def _on_preview_phase_event(self, ev):
        """Fases del preview a su tarjeta + historial (sin tocar la cola)."""
        task_id = getattr(self, "_preview_task_id", None)
        if not task_id:
            return
        self.tasks_panel.update_phase_event(task_id, ev)
        try:
            phase_writer.note_phase_event(task_id, ev)
        except Exception:
            pass

    def _mark_preview_task_finished(self, status):
        task_id = getattr(self, "_preview_task_id", None)
        if task_id:
            self.tasks_panel.mark_finished(task_id, status)
            try:
                # El orden —fases antes que cierre— y el aviso viven en el
                # escritor; el panel no lee el veredicto.
                phase_writer.finish_task(task_id, status)
            except Exception:
                pass
        self._preview_task_id = None

    def _on_preview_finished(self):
        self._update_action_buttons()
        worker = getattr(self, "preview_worker", None)
        if worker is None:
            return
        aborted = (
            getattr(worker, "abort_event", None) is not None
            and worker.abort_event.is_set()
        )
        self._mark_preview_task_finished(
            "aborted" if aborted else ("success" if worker.success else "failed")
        )
        # La barra principal sólo la toca el preview si no hay render.
        main_running = (
            getattr(self, "worker", None) is not None and self.worker.isRunning()
        )
        artifacts = getattr(worker, "preview_artifacts", None)
        if not main_running:
            self.progress.setValue(100 if worker.success else 0)
            self.progress.setFormat("%p%")
        if aborted:
            self.status_label.setText("Preview aborted.")
            _cleanup_preview_artifacts(artifacts)
            return
        if (
            not worker.success
            or not worker.preview_original
            or not worker.preview_compressed
        ):
            error_detail = worker.error_message or "Could not generate preview frames. Check FFmpeg and the input video."
            self.status_label.setText("Preview failed: could not generate frames.")
            QMessageBox.critical(
                self, "Preview Error",
                error_detail,
            )
            _cleanup_preview_artifacts(artifacts)
            return
        self.status_label.setText("Preview ready.")
        # Pixmaps construidos en el hilo principal (regla Qt).
        original_pixmap = QPixmap()
        compressed_pixmap = QPixmap()
        original_pixmap.loadFromData(worker.preview_original, "JPG")
        compressed_pixmap.loadFromData(worker.preview_compressed, "JPG")
        if original_pixmap.isNull() or compressed_pixmap.isNull():
            self.status_label.setText("Preview failed: could not load frames.")
            QMessageBox.critical(self, "Preview Error", "Could not load preview frames.")
            _cleanup_preview_artifacts(artifacts)
            return
        dialog = PreviewDialog(
            original_pixmap,
            compressed_pixmap,
            "Original (left) vs Compressed (right). Use the slider to compare.",
            self,
            artifacts=artifacts,
        )
        dialog.exec()
        _cleanup_preview_artifacts(artifacts)

    def _run_worker(self, output_path, settings, mode, ai_engines=None, input_path=None,
                    queue_id=None):
        # Estado centralizado de botones: el preview queda disponible
        # (en su propio worker) aunque el render principal esté corriendo.
        self._update_action_buttons()
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self._worker_started = time.time()

        # Normal path: the currently selected video. Followup path: the
        # job's original input (the selection may have changed while the
        # update chain was running).
        src = input_path if input_path is not None else self.input_path

# Register the task in the side panel + persisted history.
        import uuid
        if queue_id:
            # Queue: el id del item se reusa como task_id del panel para
            # que fases y progreso queden asociados a la fila correcta.
            self._current_task_id = queue_id
        else:
            self._current_task_id = str(uuid.uuid4())
        task_name = os.path.basename(src)
        if mode == "preview":
            task_name += " (preview)"
        # Como atributo y no como variable local: es lo que enseña el nombre
        # del desplegable de la bandeja, que no tiene otra forma de saber
        # sobre qué render va la barra.
        self._current_task_name = task_name
        self.tasks_panel.get_or_create(self._current_task_id, task_name)
        self.tasks_panel._cards[self._current_task_id].set_running()
        try:
            import src.task_history as task_history
            task_history.start_task(
                self._current_task_id, task_name, input_path=src,
            )
        except Exception:
            pass

        # Compress runs against a same-directory staging path. The final
        # filename is published with os.replace only after the worker reports
        # success, so cancellation, encoder failure or a power loss cannot
        # leave a truncated video under the user's chosen name.
        worker_output = output_path
        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
            os.makedirs(output_dir, exist_ok=True)
            ext = os.path.splitext(output_path)[1] or ".tmp"
            fd, worker_output = tempfile.mkstemp(
                prefix=".draggy-staging-", suffix=ext, dir=output_dir
            )
            os.close(fd)
            self._try_remove(worker_output)

        self.worker = CompressionWorker(
            self.ffmpeg_path, src, worker_output, settings,
            mode=mode, ai_engines=ai_engines,
        )
        self.worker.final_output_path = output_path
        self.worker.task_id = self._current_task_id
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.status_message.connect(self._on_status_message)
        self.worker.phase_ev.connect(self._on_phase_event)
        # Direct slot + QueuedConnection: el callback SIEMPRE corre en el
        # hilo de la GUI. Lambda en hilo secundario fallaria silenciosamente.
        self.worker.finished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self.worker.metrics_ev.connect(self._on_worker_metrics)
        self._tray_sampling_start()
        self.worker.start()
        # Worker ya asignado: re-sincronizar botones (habilita Abort).
        self._update_action_buttons()

    def _on_worker_progress(self, value):
        """Actualiza la barra + ETA estimado a partir del throughput medido.

        El ETA lo calcula ``tray_dashboard.eta_text`` y no esta codigo: la
        barra y el desplegable de la bandeja enseñan el mismo numero, y si el
        formato viviese en los dos sitios acabarían discrepando sobre el
        mismo render.
        """
        self.progress.setValue(value)
        eta = None
        try:
            elapsed = time.time() - getattr(self, "_worker_started", time.time())
            eta = eta_text(elapsed, value)
            self.progress.setFormat(f"%p%  ·  {eta}" if eta else "%p%")
        except Exception:
            pass
        if self._tray is not None:
            self._tray.update_progress(value, eta)

    def _on_status_message(self, message):
        """Estado del worker: label + log del panel lateral."""
        self.status_label.setText(message)
        self.tasks_panel.append_log(message)

    def _on_phase_event(self, ev):
        """Traduce un evento de fase del worker al panel + historial."""
        task_id = getattr(self, "_current_task_id", None)
        if not task_id:
            return
        self.tasks_panel.update_phase_event(task_id, ev)
        try:
            # El historial y la cola los escribe un hilo aparte. Aqui esto es
            # solo una anotacion en memoria: antes se escribia en este mismo
            # hilo, y cada escritura releia y reescribia el archivo entero con
            # fsync, una vez por fase y por fotograma, con la GUI esperando al
            # disco justo cuando el disco esta mas lleno. Lo que se ve —barra,
            # estado, tarjetas— lo sigue pintando esta misma senal, igual y en el
            # mismo orden, porque eso no se ha tocado.
            #
            # Aqui no queda ninguna condicion: que un evento cierre la fase, y
            # que un cierre se escriba entero y sin fundirse, lo decide el
            # modulo. La ventana solo dice si hay una cola en marcha.
            phase_writer.note_render_event(
                task_id, ev, getattr(self, "_queue_active", False))
        except Exception:
            pass

    def _mark_task_finished(self, status):
        task_id = getattr(self, "_current_task_id", None)
        # El muestreo de la bandeja para aqui y no despues de escribir el
        # cierre: un temporizador vivo sobre un widget que ya no pinta es un
        # temporador que sobrevive a la tarea.
        self._tray_sampling_stop()
        if task_id:
            self.tasks_panel.mark_finished(task_id, status)
            try:
                # Las fases pendientes salen antes que el cierre: si el archivo
                # dice "success" tiene que decir tambien hasta donde llego. El
                # orden y el aviso viven en el escritor; aqui no se mira el
                # veredicto, porque el panel no es quien decide si algo se
                # guardo.
                phase_writer.finish_task(task_id, status)
            except Exception:
                pass
        self._current_task_id = None

    def _load_task_history(self):
        """Carga el histórico persistente de tareas en el panel."""
        try:
            import src.task_history as task_history
            interrupted = task_history.recover_interrupted_tasks()
            self.tasks_panel.rebuild(task_history.list_tasks(limit=50))
            if interrupted:
                self._on_status_message(
                    f"{interrupted} task(s) from a previous session were marked "
                    "as interrupted (the app closed before they finished)."
                )
        except Exception:
            pass

    def _on_clear_task_history(self):
        try:
            import src.task_history as task_history
            task_history.clear()
        except Exception:
            pass
        self.tasks_panel.clear_all()

    # --- Cola de renders --------------------------------------------------

    def _load_task_queue(self):
        """Carga la cola persistente en la pestaña Queue (arranque)."""
        try:
            import src.task_queue as task_queue
            recovered = task_queue.recover_interrupted()
            from src.processing.pipeline import cleanup_stale_staging
            stale = cleanup_stale_staging()
            if stale:
                self._on_status_message(
                    f"Cleaned up {stale} abandoned temporary file(s) from a "
                    "previous session."
                )
            if recovered:
                self._on_status_message(
                    f"{recovered} queued render(s) recovered after an interrupted "
                    "session and marked for retry."
                )
            self.tasks_panel.rebuild_queue(task_queue.list_items())
        except Exception:
            self.tasks_panel.rebuild_queue([])

    def _add_to_queue(self):
        """Snapshot del/los video(s) + cadena IA + ajustes -> items de cola.

        Con varios videos seleccionados, TODOS entran a la cola con la
        MISMA config actual del usuario (codec, resolución, cadena IA...).
        """
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self._on_status_message("Finish the current render before queuing.")
            return
        paths = list(getattr(self, "input_paths", None) or [])
        if not paths and self.input_path:
            paths = [self.input_path]
        if not paths:
            QMessageBox.warning(self, "No video", "Please select a video first.")
            return
        if not self._maybe_ffmpeg_health_check():
            return

        settings = self.settings_panel.get_settings()
        settings["pipeline_order"] = self.workflow.get_pipeline()
        self._remember_runtime_prefs(settings)
        ai_engines = self.workflow.get_selected_engines()

        queued = 0
        for path in paths:
            if not path or not os.path.exists(path):
                self._on_status_message(f"Skipped missing file: {path}")
                continue
            output_path = self._get_output_path(path)
            try:
                import src.task_queue as task_queue
                queue_id = task_queue.enqueue(
                    name=os.path.basename(path),
                    input_path=path,
                    output_path=output_path,
                    settings=settings,
                    ai_engines=ai_engines,
                    pipeline_order=settings.get("pipeline_order", []),
                )
            except Exception:
                queue_id = ""
            if queue_id:
                queued += 1
            else:
                self._on_status_message(f"Could not add to the queue: {path}")
        if queued:
            self.tasks_panel.rebuild_queue(task_queue.list_items())
            if queued == 1:
                self._on_status_message(
                    f"Queued: {os.path.basename(paths[0])} -> "
                    f"{os.path.basename(self._get_output_path(paths[0]))}"
                )
            else:
                self._on_status_message(f"Queued {queued} videos with the current settings.")

    def _on_cancel_queue_item(self, queue_id: str):
        """``✕`` en una fila de la cola: cancela ESE job, y solo ese.

        Un job en espera sale de la cola y el total de la pasada baja con
        él, para que la barra no prometa renders que el usuario ya no
        quiere. El que está EN MARCHA se para de verdad (el mismo camino
        que el botón Abort): quitarlo de la fila sin detenerlo dejaba un
        ffmpeg escribiendo un archivo que la app ya no contaba, y su cierre
        ya no tenía item al que escribirle el final.
        """
        try:
            import src.task_queue as task_queue
            item = task_queue.get_item(queue_id) or {}
            in_flight = task_queue.is_active(queue_id)
        except Exception:
            item, in_flight = {}, False
        name = str(item.get("name") or queue_id)

        if not in_flight:
            # En espera (o ya terminado): sale de la lista sin preguntar.
            result = self.tasks_panel.remove_queue_item(queue_id)
            if result == "cancelled":
                self._on_status_message(f"Queue item cancelled: {name}")
                self.tasks_panel.append_log(f"Queue item cancelled: {name}")
                # El total de la pasada baja con el: ese render ya no lo
                # va a ejecutar nadie y no puede seguir contando.
                self._retarget_queue_total()
            elif result == "finished":
                self.tasks_panel.append_log(f"Queue item dismissed: {name}")
            return

        # Render en marcha: pararlo tira el trabajo hecho, así que se pregunta.
        ret = QMessageBox.question(
            self,
            "Cancel render",
            f"Stop the render in progress?\n\n{name}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        result = self.tasks_panel.remove_queue_item(queue_id)
        if result != "aborted":
            return
        # El item queda marcado aborted por el panel; ahora toca el proceso.
        if self._abort_running_worker(cleanup_partial_output=True):
            # ``_on_worker_finished`` ve ``abort_event`` y cierra la cola
            # él mismo con "aborted": aquí NO se cuenta nada para no
            # duplicar el avance ni arrancar dos renders a la vez.
            self._cleanup_after_abort()
            self._on_status_message(f"Render cancelled: {name}")
            return
        # No había worker detrás (fila obsoleta): la cola no puede quedarse
        # esperando a un render que ya no existe, así que se cierra la cuenta.
        self.tasks_panel.append_log(f"Render cancelled: {name} (nothing was running)")
        if getattr(self, "_queue_active", False):
            self._advance_queue_after_worker("aborted", "Cancelled by user")

    def _retarget_queue_total(self):
        """Recalcula el total de la pasada tras cancelar un job en espera.

        Sin esto la barra queda en "1/4 done" para siempre: el job cancelado
        ya no lo va a ejecutar nadie, así que deja de contar.
        """
        if not getattr(self, "_queue_active", False):
            return
        try:
            import src.task_queue as task_queue
            pending = sum(
                1 for it in task_queue.list_items()
                if it.get("status") == "queued"
            )
        except Exception:
            return
        done = getattr(self, "_queue_done", 0)
        worker = getattr(self, "worker", None)
        running = 1 if worker is not None and worker.isRunning() else 0
        self._queue_total = max(done + pending + running, done)
        self.tasks_panel.set_queue_progress(done, self._queue_total)
        if self._tray is not None:
            self._tray.set_queue(done, self._queue_total)

    def _on_clear_queue(self):
        ret = QMessageBox.question(
            self,
            "Clear queue",
            "Remove all queued jobs?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        # Vaciar el archivo con un render en marcha dejaba a ese worker sin
        # item al que escribirle el final (cola descuadrada y un ffmpeg
        # escribiendo un archivo que ya nadie cuenta). Se para primero por
        # su mismo camino que el ``✕`` de la fila y luego se limpia todo.
        try:
            import src.task_queue as task_queue
            running_id = next(
                (
                    it["queue_id"] for it in task_queue.list_items()
                    if it.get("queue_id") and task_queue.is_active(it["queue_id"])
                ),
                None,
            )
        except Exception:
            running_id = None
        if running_id:
            self._on_cancel_queue_item(running_id)
        try:
            import src.task_queue as task_queue
            task_queue.clear()
        except Exception:
            pass
        # La cola vacía no puede dejar la barra prometiendo renders que ya
        # no existen. Si había un render en marcha (que se acaba de parar),
        # ese es el único trabajo que queda por cerrar.
        worker = getattr(self, "worker", None)
        running = worker is not None and worker.isRunning()
        self._queue_done = getattr(self, "_queue_done", 0)
        self._queue_total = self._queue_done + (1 if running else 0)
        self.tasks_panel.clear_queue_rows()
        self._on_status_message("Queue cleared.")

    def _start_queue(self):
        """Arranca la cola: ejecuta los items "queued" en secuencia."""
        if getattr(self, "_queue_active", False):
            return
        try:
            import src.task_queue as task_queue
            if task_queue.first_pending() is None:
                self._on_status_message("Queue is already empty.")
                return
            # Total de jobs que va a ejecutar ESTA pasada (solo los "queued";
            # los ya terminados/failed no se vuelven a tocar).
            self._queue_total = sum(
                1 for it in task_queue.list_items()
                if it.get("status") == "queued"
            )
        except Exception:
            self._on_status_message("Could not read the queue.")
            return
        self._queue_done = 0
        self._queue_active = True
        self.tasks_panel.set_queue_running(True, total=self._queue_total)
        self._on_status_message(
            f"Queue started: {self._queue_total} render(s) pending."
        )
        self._start_next_queue_item()

    def _start_next_queue_item(self):
        """Ejecuta el siguiente item "queued"; al agotarse resume la UI."""
        if not getattr(self, "_queue_active", False):
            return
        try:
            import src.task_queue as task_queue
            item = task_queue.first_pending()
        except Exception:
            item = None
        if item is None:
            self._queue_active = False
            total = getattr(self, "_queue_total", 0)
            done = getattr(self, "_queue_done", 0)
            self.tasks_panel.set_queue_progress(done, total)
            self.tasks_panel.set_queue_running(False)
            self._on_status_message(
                f"Queue finished: {done}/{total} render(s) processed."
            )
            self.tasks_panel.append_log("Queue finished: all jobs processed.")
            return

        queue_id = item.get("queue_id", "")
        task_queue.set_status(queue_id, "running")
        self.tasks_panel.set_queue_status(queue_id, "running")

        done = getattr(self, "_queue_done", 0)
        total = getattr(self, "_queue_total", 0)
        self._on_status_message(
            f"Queue [{done + 1}/{total}]: {item.get('name') or queue_id} — running..."
        )

        input_path = item.get("input_path", "")
        if not input_path or not os.path.exists(input_path):
            task_queue.set_status(
                queue_id, "failed", error="Input file is missing: " + input_path
            )
            self.tasks_panel.set_queue_status(queue_id, "failed")
            self.tasks_panel.append_log(
                f"Queue item failed (missing input): {item.get('name')}"
            )
            self._queue_done = done + 1
            self.tasks_panel.set_queue_progress(
                self._queue_done, getattr(self, "_queue_total", 0)
            )
            self._start_next_queue_item()
            return

        settings = dict(item.get("settings") or {})
        settings["pipeline_order"] = list(item.get("pipeline_order") or [])
        ai_engines = dict(item.get("ai_engines") or {})
        output_path = item.get("output_path") or ""
        self._run_worker(
            output_path,
            settings,
            mode="compress",
            ai_engines=ai_engines,
            input_path=input_path,
            queue_id=queue_id,
        )

    def _advance_queue_after_worker(self, status: str, error: str = ""):
        """Marca el item que acaba de terminar y arranca el siguiente."""
        queue_id = getattr(self, "_current_task_id", None)
        if not queue_id:
            worker = getattr(self, "worker", None)
            queue_id = getattr(worker, "task_id", None)
        if queue_id:
            try:
                import src.task_queue as task_queue
                task_queue.set_status(queue_id, status, error=error or "")
                item = task_queue.get_item(queue_id)
                name = (item or {}).get("name", queue_id)
            except Exception:
                name = queue_id
            self.tasks_panel.set_queue_status(queue_id, status)
            self.tasks_panel.append_log(
                f"Queue item {status}: {name}"
                + (f" ({error})" if error else "")
            )
            self._queue_done = getattr(self, "_queue_done", 0) + 1
            self.tasks_panel.set_queue_progress(
                self._queue_done, getattr(self, "_queue_total", 0)
            )
        self._start_next_queue_item()

    def _on_worker_finished(self):
        self._update_action_buttons()
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
            self._mark_task_finished("aborted")
            self._cleanup_after_abort()
            if getattr(self, "_queue_active", False):
                self._advance_queue_after_worker("aborted", "Aborted by user")
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
                            artifacts = getattr(worker, "preview_artifacts", None)
                            dlg = PreviewDialog(
                                opm, cpm,
                                "Original (left) vs Compressed (right). Use the slider to compare.",
                                self,
                                artifacts=artifacts,
                            )
                            dlg.exec()
                            _cleanup_preview_artifacts(artifacts)
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
        # Publishing is a distinct state: only enter it after all render
        # phases passed and before the atomic rename.
        if worker.success and getattr(self, "_queue_active", False):
            try:
                import src.task_queue as task_queue
                task_queue.set_status(worker.task_id, "publishing")
            except Exception:
                pass
        # Publish the staged file only after every pipeline stage succeeded.
        if worker.success and getattr(worker, "final_output_path", None):
            staged = getattr(worker, "output_path", None)
            final = worker.final_output_path
            valid, detail = validate_render_output(
                self.ffmpeg_path, staged,
            ) if staged else (False, "Render produced no staging path.")
            if not valid:
                worker.success = False
                worker.error_message = f"Render finished with invalid output: {detail}"
            else:
                try:
                    os.replace(staged, final)
                except Exception as exc:
                    worker.success = False
                    worker.error_message = f"Could not publish output atomically: {exc}"
                    self._try_remove(staged)

        self.progress.setValue(100 if worker.success else 0)
        mode = getattr(worker, "mode", "compress")
        self._mark_task_finished("success" if worker.success else "failed")

        if getattr(self, "_queue_active", False):
            # Modo cola: sin diálogos modales por item; marcamos y
            # seguimos con el siguiente render programado.
            self._advance_queue_after_worker(
                "done" if worker.success else "failed",
                "" if worker.success else (worker.error_message or "Operation failed"),
            )
            return

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
                error_detail = worker.error_message or "Could not generate preview frames. Check FFmpeg and the input video."
                self.status_label.setText("Preview failed: could not generate frames.")
                QMessageBox.critical(
                    self, "Preview Error",
                    error_detail,
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
            artifacts = getattr(worker, "preview_artifacts", None)
            dialog = PreviewDialog(
                original_pixmap,
                compressed_pixmap,
                "Original (left) vs Compressed (right). Use the slider to compare.",
                self,
                artifacts=artifacts,
            )
            dialog.exec()
            _cleanup_preview_artifacts(artifacts)

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

    def _abort_running_worker(self, cleanup_partial_output, worker=None):
        """Cancel an active ``CompressionWorker`` and tidy up disk.

        Reused by ``_abort`` (button click) and ``closeEvent`` (window
        closing during a regular compress). Returns True if it actually
        aborted something; False if no worker was running.

        ``worker`` puede pasarse explícitamente (p.ej. el preview); por
        defecto aborta el worker principal y, si no hay, el preview en
        vuelo.

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
        if worker is None:
            worker = getattr(self, "worker", None)
            if worker is None or not worker.isRunning():
                worker = getattr(self, "preview_worker", None)
        if worker is None or not worker.isRunning():
            return False
        # 1) Cooperative path: ask run_ffmpeg to terminate ffmpeg.
        try:
            if getattr(worker, "abort_event", None) is not None:
                worker.abort_event.set()
        except Exception:
            pass
        # 2) Stop the complete process tree, not only the last Popen.
        #    This covers decoder, encoder, audio and mux children even when
        #    one of them is blocked on a pipe and cannot observe the event.
        try:
            worker.supervisor.stop_all(grace_seconds=2.0, kill_seconds=2.0)
        except Exception:
            pass
        # 3) Wait for Python cleanup. QThread.terminate() remains an absolute
        #    last resort only; normally the supervisor makes the worker leave
        #    every blocking read and lets its finally blocks run.
        if not worker.wait(8000):
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
        # Kill every child known to the AI pipeline (decoder + encoder),
        # not just the last encoder reference. This matters when the Python
        # QThread has to be force-terminated while the decoder is blocked
        # in a raw pipe read.
        processes = list(getattr(worker, "_procs", []) or [])
        proc = getattr(worker, "_proc", None)
        if proc is not None and proc not in processes:
            processes.append(proc)
        for child in processes:
            try:
                child.kill()
            except Exception:
                pass
            try:
                child.wait(timeout=2)
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
                logger.warning("[Draggy] Could not delete partial %s: %s", path, exc)
        except Exception as exc:
            logger.warning("[Draggy] Could not delete partial %s: %s", path, exc)

    def _cleanup_after_abort(self):
        """Reset UI state after the worker finished (successfully aborted)."""
        try:
            self._update_action_buttons()
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
        self._set_redetect_busy(True)
        self.btn_compress.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Re-detecting hardware and codecs...")

        self.redetect_worker = RedetectWorker(self.ffmpeg_path)
        self.redetect_worker.progress.connect(
            lambda v, msg: (self.progress.setValue(v), self._on_status_message(msg))
        )
        self.redetect_worker.done.connect(self._on_redetect_done)
        self.redetect_worker.error.connect(self._on_redetect_error)
        self.redetect_worker.start()

    def _set_redetect_busy(self, busy, message=""):
        """Marca el estado 'redetectando' en la ventana de hardware abierta.

        El botón vive ahora dentro de ``HardwareDialog``; si no hay ninguna
        ventana abierta no hay nada que marcar.
        """
        dialog = getattr(self, "_hardware_dialog", None)
        if dialog is not None:
            try:
                dialog.set_busy(busy, message)
            except Exception:
                logger.debug("could not update the hardware dialog", exc_info=True)

    def _refresh_hardware_dialog(self, hardware_info, compatible_codecs):
        """Refresca la ventana de hardware si está abierta."""
        dialog = getattr(self, "_hardware_dialog", None)
        if dialog is None:
            return
        try:
            dialog.refresh(hardware_info, compatible_codecs)
        except Exception:
            logger.debug("could not refresh the hardware dialog", exc_info=True)

    def _refresh_dependencies_dialog(self):
        """Repinta el panel de dependencias si está abierto.

        El selector de builds manda al usuario a 'Re-detect Hardware &
        Codecs' cuando no pudo leer el hardware, así que cuando esa
        redetección termina y el panel sigue en pantalla tiene que ver el
        veredicto nuevo sin cerrarlo. Si no hay panel abierto, no hay nada
        que hacer y aquí no pasa nada.
        """
        dialog = getattr(self, "_dependencies_dialog", None)
        if dialog is None:
            return
        try:
            dialog.repopulate()
        except Exception:
            logger.debug(
                "could not refresh the dependencies dialog", exc_info=True
            )

    def _on_redetect_done(self, hardware_info, compatible_codecs):
        self._set_redetect_busy(False)
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
        self._refresh_hardware_dialog(hardware_info, compatible_codecs)
        self._refresh_dependencies_dialog()
        self.status_label.setText("Re-detection finished.")

    def _on_redetect_error(self, msg):
        self._set_redetect_busy(False)
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

    def _wire_audio_preview(self):
        """Conecta el botón 'Escuchar efecto' del panel con el motor de audio.

        El panel no conoce la ruta del vídeo ni el ffmpeg (los tiene la
        ventana), así que la conexión vive aquí y el panel solo emite señales.
        """
        panel = self.settings_panel.audio_fx_panel
        panel.preview_requested.connect(self._on_audio_preview_requested)
        panel.preview_stop_requested.connect(self._on_audio_preview_stopped)
        self._audio_preview.status.connect(panel.set_status)
        self._audio_preview.failed.connect(self._on_audio_preview_failed)
        self._audio_preview.ready.connect(self._on_audio_preview_ready)

    def _on_audio_preview_requested(self):
        """Genera y reproduce el fragmento con los efectos actuales."""
        video_path = getattr(self, "input_path", None)
        if not video_path or not os.path.exists(str(video_path)):
            preview_panel = self.settings_panel.audio_fx_panel
            preview_panel.set_playback_state(False)
            preview_panel.set_status(
                "Importa un vídeo para poder escuchar los efectos."
            )
            return
        settings = self.settings_panel.get_settings()
        ffmpeg_path = getattr(self, "ffmpeg_path", None)
        if not ffmpeg_path:
            try:
                from src.ffmpeg.manager import get_local_ffmpeg_path

                ffmpeg_path = get_local_ffmpeg_path()
            except Exception:
                ffmpeg_path = None
        self.settings_panel.audio_fx_panel.set_playback_state(True)
        self._audio_preview.play(
            ffmpeg_path,
            str(video_path),
            settings.get("audio_fx"),
            start_time=settings.get("start_time"),
            audio_stream_index=settings.get("audio_stream_index") or 0,
        )

    def _on_audio_preview_stopped(self):
        self._audio_preview.stop()

    def _on_audio_preview_ready(self, _path):
        self.settings_panel.audio_fx_panel.set_playback_state(True)

    def _on_audio_preview_failed(self, message):
        panel = self.settings_panel.audio_fx_panel
        panel.set_playback_state(False)
        panel.set_status(message)

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
