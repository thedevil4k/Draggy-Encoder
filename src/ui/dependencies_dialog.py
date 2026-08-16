"""Panel de gestión de dependencias externas del programa.

Muestra una lista de dependencias (FFmpeg real, más placeholders de runtimes y
ranuras de modelos) con su estado, versión instalada y acciones disponibles
(instalar / reinstalar, abrir carpeta, abrir logs).
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
import traceback
from datetime import datetime, timezone

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import src.dependencies as dependencies
from src.ai.bus import bus as _state_bus
from src.runtimes.manager import get_python_support_info as _get_python_support_info
from src.ai.engines_state import (
    get_state as engines_state_get,
    mark_failed as engines_mark_failed,
    mark_installed as engines_mark_installed,
    mark_onnx as engines_mark_onnx,
    mark_uninstalled as engines_mark_uninstalled,
    patch as engines_state_patch,
    set_backend as engines_set_backend,
    clear_benchmark as engines_clear_benchmark,
)
from src.ai.fetcher import (
    GitCloneWorker,
    HFDownloadWorker,
    OnnxConvertWorker,
    is_git_available,
    is_hf_hub_available,
)
from src.ai.registry import CATALOG, get_engine_dir
from src.ai.engines_state import dir_has_weights
from src.config import get_logs_dir


# Fases del worker de conversión a mensajes amigables para la fila del
# modelo (el usuario final no debe ver inglés técnico del worker).
_CONVERT_STATUS_MAP = {
    "Locating model weights...": "Buscando los pesos del modelo...",
    "Converting to ONNX with torch.onnx.export...": "Convirtiendo a ONNX (torch.onnx.export)...",
    "export complete": "Conversión completada.",
}


def _friendly_convert_status(raw: str) -> str:
    """Traduce las fases crudas del OnnxConvertWorker a español."""
    return _CONVERT_STATUS_MAP.get(str(raw or "").strip(), str(raw or ""))


from src.ffmpeg.upstream import clear_cache
from src.ui.engines_tab import EnginesTab


_STATUS_COLORS = {
    dependencies.STATUS_INSTALLED: ("#00d2d3", "#0f0f1a"),      # turquesa (sobre fondo claro)
    dependencies.STATUS_MISSING: ("#e74c3c", "#ffffff"),         # rojo
    dependencies.STATUS_ERROR: ("#f39c12", "#0f0f1a"),           # ámbar
    dependencies.STATUS_COMING_SOON: ("#5f27cd", "#ffffff"),     # lila (coming soon)
}


def _fmt_bytes(n):
    """Formatea bytes a una unidad legible (B/KB/MB/GB/TB)."""
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _format_latest_line(info):
    """Devuelve el texto "Latest: … · last check: …" para la columna Latest."""
    fp = info.get("latest_fingerprint") or {}
    last_mod = fp.get("last_modified_iso")
    checked_at = fp.get("checked_at")
    if not last_mod:
        if info.get("upstream_error"):
            return "Latest: — (offline) · last check: never"
        return "Latest: — · last check: never"
    # Mostrar solo la fecha (YYYY-MM-DD) para no saturar la línea
    short = last_mod.split("T", 1)[0]
    # Edad del check en horas
    age_text = "—"
    if checked_at:
        try:
            dt = datetime.fromisoformat(checked_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
            if hours < 1:
                age_text = f"{int(hours * 60)} min ago"
            elif hours < 24:
                age_text = f"{hours:.1f} h ago"
            else:
                age_text = f"{hours / 24:.1f} d ago"
        except Exception:
            pass
    return f"Latest upstream build: {short}  ·  last check: {age_text}"


class _ReinstallWorker(QThread):
    """Hilo separado para no bloquear la UI durante la descarga.

    Cancelable: ``cancel()`` setea un ``threading.Event`` que el instalador
    (pip / descarga de FFmpeg) observa y propaga como ``RuntimeError
    ("cancelled by user")``; el worker distingue ese caso y emite
    ``cancelled`` en vez de ``failed``.
    """

    progress = Signal(int)
    finished_ok = Signal(str)         # dep_id
    failed = Signal(str, str)         # dep_id, mensaje
    cancelled = Signal(str)           # dep_id

    def __init__(self, dep_id, parent=None):
        super().__init__(parent)
        self.dep_id = dep_id
        self.abort_event = threading.Event()

    def cancel(self):
        self.abort_event.set()

    def run(self):
        try:
            dependencies.reinstall(
                self.dep_id,
                progress_callback=lambda v, m: self.progress.emit(v),
                parent=self.parent(),
                abort_event=self.abort_event,
            )
            self.finished_ok.emit(self.dep_id)
        except Exception as exc:
            if self.abort_event.is_set():
                self.cancelled.emit(self.dep_id)
            else:
                self.failed.emit(self.dep_id, str(exc))


class _UninstallWorker(QThread):
    """Hilo separado para borrar una dependencia sin congelar la UI."""

    finished_ok = Signal(str)         # dep_id
    failed = Signal(str, str)         # dep_id, mensaje

    def __init__(self, dep_id, parent=None):
        super().__init__(parent)
        self.dep_id = dep_id

    def run(self):
        try:
            dependencies.uninstall(self.dep_id, parent=self.parent())
            self.finished_ok.emit(self.dep_id)
        except Exception as exc:
            self.failed.emit(self.dep_id, str(exc))


class _EngineRemoveWorker(QThread):
    """Borra los archivos de un motor IA (modelos / FFmpeg full) en segundo plano."""

    finished_ok = Signal(str)         # engine_id
    failed = Signal(str, str)         # engine_id, mensaje

    def __init__(self, engine_id, remove_fn, parent=None):
        super().__init__(parent)
        self.engine_id = engine_id
        self.remove_fn = remove_fn

    def run(self):
        try:
            self.remove_fn()
            self.finished_ok.emit(self.engine_id)
        except Exception as exc:
            self.failed.emit(self.engine_id, str(exc))


class _BenchmarkWorker(QThread):
    """Mide ms/frame de UN motor (ambos backends) en background.

    Silencioso a propósito: alimenta la caché de benchmarks que usa el
    selector de backend de la fila (ONNX vs PyTorch) y el ETA del render.
    Si un backend no se puede medir (sin .onnx, torch no instalado, ...)
    se salta; no perturbamos el flujo de instalación/conversión.
    """

    def __init__(self, engine_id, target_dir, parent=None):
        super().__init__(parent)
        self.engine_id = engine_id
        self.target_dir = target_dir

    def run(self):
        try:
            from src.ai.benchmark import benchmark_backend
            for backend in ("onnx", "torch"):
                try:
                    benchmark_backend(
                        self.engine_id, backend,
                        target_dir=self.target_dir,
                        use_cache=True,
                    )
                except Exception:
                    pass
        except Exception:
            pass


class _UpstreamCheckWorker(QThread):
    """Hilo independiente para refrescar la huella del upstream."""

    done = Signal()

    def __init__(self, force=False, parent=None):
        super().__init__(parent)
        # Cuando ``force`` es True se ignora el cache de 24h (Refresh manual).
        # Cuando es False, sólo se hace HEAD si el cache tiene más de 24h.
        self.force = force

    def run(self):
        # Llamada bloqueante pero acotada por timeout de requests.
        try:
            from src.ffmpeg.upstream import get_fingerprint
            get_fingerprint(force=self.force)
        except Exception:
            # La falla queda registrada en el cache (last_error); no
            # necesitamos propagarla. El panel leerá el cache y mostrará
            # el último estado conocido.
            pass
        self.done.emit()


class _DepRow(QFrame):
    """Una fila con el estado + versión + acciones de una dependencia."""

    redownload_requested = Signal(str)
    uninstall_requested = Signal(str)

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.setObjectName("DepRow")
        self.setStyleSheet(self._stylesheet())
        self._build(info)

    def _stylesheet(self):
        return """
        #DepRow {
            background-color: #16213e;
            border: 1px solid #2c3e50;
            border-radius: 8px;
        }
        #DepRow QPushButton {
            background-color: #1f2c4c;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 6px 12px;
        }
        #DepRow QPushButton:hover {
            background-color: #00d2d3;
            color: #0f0f1a;
        }
        #DepRow QPushButton:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #DepRow QPushButton#uninstallBtn {
            background-color: #3a1620;
            color: #ff6b6b;
            border: 1px solid #6e2233;
        }
        #DepRow QPushButton#uninstallBtn:hover {
            background-color: #e74c3c;
            color: #ffffff;
        }
        #DepRow QPushButton#uninstallBtn:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #StatusBadge {
            border-radius: 9px;
            padding: 2px 10px;
            font-weight: bold;
            font-size: 11px;
        }
        """

    def _build(self, info):
        outer = QHBoxLayout(self)
        outer.setSpacing(14)
        outer.setContentsMargins(14, 12, 14, 12)

        # Columna izquierda: nombre + versión + ruta
        left = QVBoxLayout()
        left.setSpacing(4)
        title = QLabel(info.get("display_name", "—"))
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #ffffff;")
        left.addWidget(title)

        kind_label = info.get("kind", "").upper()
        version_text = info.get("version") or "—"
        meta = QLabel(f"{kind_label}  ·  version: {version_text}")
        meta.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        left.addWidget(meta)

        path_label = QLabel(
            info.get("install_dir") or "(no install dir)"
        )
        path_label.setStyleSheet(
            "color: #707080; font-size: 10px; font-family: Consolas, monospace;"
        )
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(path_label)

        if info.get("source_url"):
            url_label = QLabel(info["source_url"])
            url_label.setStyleSheet(
                "color: #00d2d3; font-size: 10px;"
            )
            url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            left.addWidget(url_label)

        # Línea "Latest <fecha>" cuando hay datos de upstream (sólo BINARY)
        if info.get("kind") == dependencies.KIND_BINARY:
            latest_text = _format_latest_line(info)
            if latest_text:
                latest_label = QLabel(latest_text)
                latest_label.setStyleSheet("color: #00d2d3; font-size: 11px;")
                latest_label.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                left.addWidget(latest_label)

        if info.get("detail"):
            detail = QLabel(info["detail"])
            detail.setStyleSheet("color: #c0c0d0; font-size: 11px;")
            detail.setWordWrap(True)
            left.addWidget(detail)

        # AI runtime compatibility warning (Python 3.14, etc.)
        support = self._ai_runtime_support_info(info)
        if support and not support["supported"]:
            warn = QLabel(support["message"])
            warn.setStyleSheet("color: #f39c12; font-size: 11px; font-weight: bold;")
            warn.setWordWrap(True)
            left.addWidget(warn)

        outer.addLayout(left, stretch=1)

        # Columna derecha: badge principal + (opcional) "Update available"
        right = QVBoxLayout()
        right.setSpacing(6)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        badge_row.setAlignment(Qt.AlignmentFlag.AlignRight)

        badge = QLabel(self._badge_text(info.get("status")))
        status = info.get("status")
        fg_bg = _STATUS_COLORS.get(status, ("#a0a0b0", "#0f0f1a"))
        badge.setStyleSheet(
            f"#StatusBadge {{ background-color: {fg_bg[0]}; color: {fg_bg[1]}; }}"
        )
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setObjectName("StatusBadge")
        badge.setMaximumWidth(140)
        badge_row.addWidget(badge)

        # Insignia secundaria "Update available" cuando aplica
        if info.get("update_available") is True:
            ua = QLabel("UPDATE AVAILABLE")
            ua.setObjectName("UpdateBadge")
            ua.setStyleSheet(
                "#UpdateBadge { background-color: #f39c12; color: #0f0f1a; "
                "border-radius: 9px; padding: 2px 10px; font-weight: bold; font-size: 11px; }"
            )
            ua.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge_row.addWidget(ua)
        elif info.get("upstream_error"):
            ua = QLabel("OFFLINE")
            ua.setObjectName("UpdateBadge")
            ua.setStyleSheet(
                "#UpdateBadge { background-color: #2c3e50; color: #a0a0b0; "
                "border-radius: 9px; padding: 2px 10px; font-weight: bold; font-size: 11px; }"
            )
            ua.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ua.setToolTip("Could not reach the upstream URL.\n" + str(info["upstream_error"]))
            badge_row.addWidget(ua)

        right.addLayout(badge_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        primary_label = self._primary_label(info)
        support = self._ai_runtime_support_info(info)
        self.btn_primary = QPushButton(primary_label)
        self.btn_primary.setEnabled(self._primary_enabled(info, support=support))
        if info.get("kind") == dependencies.KIND_MODEL:
            self.btn_primary.setToolTip(
                "Add or remove individual engines from the 'AI Engines' tab."
            )
        if info.get("update_available") is True:
            self.btn_primary.setStyleSheet(
                "QPushButton { background-color: #f39c12; color: #0f0f1a; "
                "border: none; border-radius: 6px; padding: 6px 12px; font-weight: bold; }"
                "QPushButton:hover { background-color: #f5b041; }"
                "QPushButton:disabled { background-color: #2c3e50; color: #5f6f7f; }"
            )
        self.btn_primary.clicked.connect(
            lambda: self.redownload_requested.emit(info.get("id") or info.get("display_name"))
        )
        btn_row.addWidget(self.btn_primary)

        self.btn_open = QPushButton("Open folder")
        self.btn_open.setEnabled(bool(info.get("install_dir")))
        self.btn_open.clicked.connect(lambda: self._open_path(info.get("install_dir")))
        btn_row.addWidget(self.btn_open)

        # Desinstalar / borrar lo instalado (binarios, runtimes, modelos).
        can_uninstall = bool(info.get("deletable")) and info.get(
            "status"
        ) == dependencies.STATUS_INSTALLED
        self.btn_uninstall = QPushButton("Uninstall")
        self.btn_uninstall.setObjectName("uninstallBtn")
        self.btn_uninstall.setEnabled(can_uninstall)
        if not can_uninstall:
            self.btn_uninstall.setToolTip(
                "Nothing installed to remove, or it cannot be removed from the app."
            )
        self.btn_uninstall.clicked.connect(
            lambda: self.uninstall_requested.emit(
                info.get("id") or info.get("display_name")
            )
        )
        btn_row.addWidget(self.btn_uninstall)

        right.addLayout(btn_row)
        right.addStretch()
        outer.addLayout(right)

    def _badge_text(self, status):
        return {
            dependencies.STATUS_INSTALLED: "INSTALLED",
            dependencies.STATUS_MISSING: "MISSING",
            dependencies.STATUS_ERROR: "ERROR",
            dependencies.STATUS_COMING_SOON: "COMING SOON",
        }.get(status, status or "—")

    def _primary_label(self, info):
        # Ranuras que no tienen instalador automático desde este panel:
        # los modelos se gestionan en la pestaña AI Engines.
        if not info.get("is_updatable"):
            return "No disponible"
        # Si upstream sugiere build más reciente y está instalado,
        # el botón principal pasa a ser "Update" en lugar de "Reinstall".
        if (
            info.get("status") == dependencies.STATUS_INSTALLED
            and info.get("update_available") is True
        ):
            return "Actualizar"
        if info.get("status") == dependencies.STATUS_INSTALLED:
            return "Reinstalar"
        if info.get("status") == dependencies.STATUS_MISSING:
            return "Instalar"
        if info.get("status") == dependencies.STATUS_ERROR:
            return "Reintentar"
        return "No disponible"

    def _primary_enabled(self, info, support=None):
        if not info.get("is_updatable"):
            return False
        # AI runtimes cannot be installed on Python versions without wheels.
        if support is None:
            support = self._ai_runtime_support_info(info)
        if support and not support["supported"]:
            return False
        return True

    def _ai_runtime_support_info(self, info):
        """Return support info for AI runtime rows (torch/onnxruntime)."""
        if info.get("kind") != dependencies.KIND_RUNTIME:
            return None
        dep_id = info.get("id") or ""
        if dep_id in ("torch", "onnxruntime"):
            return _get_python_support_info()
        return None

    def _open_path(self, path):
        if not path:
            return
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
        try:
            if platform.system() == "Windows":
                os.startfile(path)  # noqa: S606
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            QMessageBox.warning(self, "Open folder", f"No se pudo abrir la carpeta:\n{exc}")


class DependenciesDialog(QDialog):
    """Diálogo modal con la lista de dependencias externas."""

    def __init__(self, parent=None, initial_tab=None):
        super().__init__(parent)
        # Modo pane único: "ai_engines" abre SOLO el catálogo de modelos;
        # cualquier otro valor (o None) abre SOLO binarios y runtimes.
        # Los dos paneles ya no conviven en un mismo diálogo con pestañas.
        self._engines_mode = initial_tab == "ai_engines"
        self.setWindowTitle(
            "AI Engines — DRAGGY ENCODER" if self._engines_mode
            else "Dependencies — DRAGGY ENCODER"
        )
        self.setMinimumSize(820, 560)
        self.resize(960, 640)
        self.setStyleSheet(
            "QDialog { background-color: #0f0f1a; color: #e0e0e0; }"
            " QLabel { color: #e0e0e0; }"
            " QScrollArea { background-color: #0f0f1a; border: none; }"
        )
        self._row_widgets = []
        self._active_worker = None
        self._uninstall_worker = None
        self._upstream_worker = None
        self._check_force_next = False
        # Workers de la pestaña AI Engines (lazy init en cada handler:
        # el destructor del QThread libera solo).
        self._git_worker = None
        self._onnx_worker = None
        # Worker de borrado de motores IA.
        self._remove_worker = None
        # Pane de modelos; solo existe en modo AI Engines.
        self.engines_tab = None
        # Escuchamos el bus de estado para repintar la pestaña AI Engines
        # aunque las mutaciones ocurran en otra ventana (workflow widget,
        # followup al cerrar, etc.). UniqueConnection evita handlers
        # duplicados si el dialog se reconstruye en tests.
        #
        # NOTA: una lambda enlazada al bus NO se autodesconecta cuando el
        # diálogo muere (no es un método bound de QObject). Patrón igual
        # que WorkflowWidget: método bound + disconnect en destroyed.
        try:
            _state_bus.state_changed.connect(
                self._on_state_changed_for_engines,
                Qt.ConnectionType.UniqueConnection,
            )
            self.destroyed.connect(self._disconnect_state_bus_for_engines)
        except Exception:
            pass
        # Timer único para disparo automático; se (re)programa desde
        # _schedule_check(delay_ms, force).
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.timeout.connect(self._run_upstream_check)
        self._build()

    def _on_state_changed_for_engines(self, _engine_id=None):
        """Slot del bus de estado (método bound → autodesconexión fiable)."""
        self._refresh_engines_tab()

    def _disconnect_state_bus_for_engines(self):
        """Al destruirse el diálogo, corta la suscripción al bus.

        Sin esto el bus (process-lifetime) conserva una referencia al
        slot del diálogo ya destruido y un ``emit`` posterior intentaría
        repintar un widget muerto (RuntimeError de PySide6)."""
        try:
            _state_bus.state_changed.disconnect(
                self._on_state_changed_for_engines
            )
        except Exception:
            pass

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 14, 16, 14)

        if self._engines_mode:
            title = QLabel("AI Engines")
            subtitle_text = (
                "Catálogo de motores de IA (HuggingFace): descarga, convierte "
                "a ONNX, cambia de backend y desinstala modelos de upscale, "
                "interpolación, denoise, colorize y face restore. Los binarios "
                "y runtimes (FFmpeg, PyTorch, ONNX Runtime) se gestionan en el "
                "diálogo 'Dependencies'."
            )
        else:
            title = QLabel("Dependencies")
            subtitle_text = (
                "Binarios y runtimes (FFmpeg, PyTorch, ONNX Runtime) se "
                "almacenan dentro de la carpeta del programa (bin/). Los "
                "modelos de IA se gestionan en el diálogo 'AI Engines'. El "
                "panel consulta el upstream una vez cada 24 h y muestra una "
                "insignia 'UPDATE AVAILABLE' cuando hay una build más "
                "reciente; pulsa 'Update' para traerla."
            )
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00d2d3;")
        layout.addWidget(title)

        subtitle = QLabel(subtitle_text)
        subtitle.setStyleSheet("color: #a0a0b0;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        if self._engines_mode:
            # Pane único: solo el catálogo de modelos, sin mezcla con
            # binarios/runtimes.
            self.engines_tab = self._create_engines_tab()
            layout.addWidget(self.engines_tab, stretch=1)
        else:
            # Pane único: solo binarios y runtimes.
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.container = QWidget()
            self.container_layout = QVBoxLayout(self.container)
            self.container_layout.setSpacing(10)
            self.container_layout.setContentsMargins(0, 0, 0, 0)
            self.container_layout.addStretch()
            self.scroll.setWidget(self.container)
            layout.addWidget(self.scroll, stretch=1)

        bottom = QHBoxLayout()
        if not self._engines_mode:
            self.btn_refresh = QPushButton("Refresh")
            self.btn_refresh.setToolTip(
                "Re-checks the FFmpeg upstream cache and redraws the binaries list."
            )
            self.btn_refresh.clicked.connect(self._refresh)
        self.btn_logs = QPushButton("Open logs")
        self.btn_logs.clicked.connect(self._open_logs)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        if not self._engines_mode:
            bottom.addWidget(self.btn_refresh)
        bottom.addWidget(self.btn_logs)
        bottom.addStretch()
        bottom.addWidget(self.btn_close)
        layout.addLayout(bottom)

        if self._engines_mode:
            return

        self._populate()

        # Auto-check inicial en background: respeta cache de 24 h, espera
        # 1.5 s para no estorbar la animación de apertura del diálogo.
        self._schedule_upstream_check(delay_ms=1500, force=False)

    def _create_engines_tab(self):
        """Create the AI Engines tab, with a graceful fallback on error."""

        class _EnginesTabErrorWidget(QWidget):
            """Placeholder shown when the real AI Engines tab fails to build."""

            def __init__(self, exc, parent=None):
                super().__init__(parent)
                layout = QVBoxLayout(self)
                layout.setContentsMargins(16, 16, 16, 16)
                error_lbl = QLabel(
                    f"Could not build the AI Engines panel:\n\n{exc}\n\n"
                    "Check bin/logs for details."
                )
                error_lbl.setStyleSheet("color: #e74c3c; font-size: 12px;")
                error_lbl.setWordWrap(True)
                layout.addWidget(error_lbl)
                tb = QLabel(traceback.format_exc())
                tb.setStyleSheet(
                    "color: #707080; font-family: Consolas, monospace; font-size: 10px;"
                )
                tb.setWordWrap(True)
                layout.addWidget(tb)
                layout.addStretch()

            def repopulate(self):
                """No-op so ``_refresh_engines_tab`` does not crash."""

        try:
            tab = EnginesTab(self)
            tab.install_requested.connect(self._on_engine_install)
            tab.convert_requested.connect(self._on_engine_convert)
            tab.backend_changed.connect(self._on_engine_backend_changed)
            tab.cancel_convert_requested.connect(self._on_engine_cancel_convert)
            tab.open_folder.connect(self._on_engine_open_folder)
            tab.remove_requested.connect(self._on_engine_remove)
            return tab
        except Exception as exc:
            logger.exception("Failed to build AI Engines tab")
            return _EnginesTabErrorWidget(exc, self)

    def _populate(self):
        # Limpiar filas existentes
        for w in self._row_widgets:
            w.setParent(None)
            w.deleteLater()
        self._row_widgets.clear()

        # El panel Dependencies muestra solo binarios y runtimes.
        # Las ranuras de modelos (KIND_MODEL) se gestionan por motor desde
        # el diálogo 'AI Engines'; no se listan aquí para no duplicar
        # categorías.
        for info in dependencies.list_all():
            if info.get("kind") == dependencies.KIND_MODEL:
                continue
            row = _DepRow(info, parent=self.container)
            row.redownload_requested.connect(self._on_redownload_requested)
            row.uninstall_requested.connect(self._on_uninstall_requested)
            self.container_layout.insertWidget(self.container_layout.count() - 1, row)
            self._row_widgets.append(row)

    def _schedule_upstream_check(self, delay_ms=1500, force=False):
        """(Re)programa un único check upstream.

        - ``force=False``: respeta cache de 24 h. Para auto-check al abrir y
          tras una reinstalación exitosa (queremos comparar contra el
          upstream inmediatamente para validar la nueva build).
        - ``force=True``: bypass del cache, HEAD fresco. Para el botón
          'Refresh' manual.
        Cancela cualquier timer pendiente antes de programar el nuevo.

        IMPORTANTE: este método debe invocarse sólo desde el hilo de la GUI.
        El flag ``_check_force_next`` lo lee después el worker para construir
        el ``_UpstreamCheckWorker(force=...)``, así que una llamada desde un
        hilo no-GUI introduciría una carrera.
        """
        self._check_force_next = force
        try:
            self._check_timer.stop()
        except Exception:
            pass
        self._check_timer.start(delay_ms)

    def _run_upstream_check(self):
        if self._upstream_worker and self._upstream_worker.isRunning():
            return
        self._upstream_worker = _UpstreamCheckWorker(
            force=self._check_force_next, parent=self
        )
        self._upstream_worker.done.connect(self._on_check_finished)
        # finished = QThread terminado correctamente (cleanup)
        self._upstream_worker.finished.connect(self._on_check_thread_finished)
        self._upstream_worker.start()

    def _on_check_finished(self):
        # ``done`` se emite desde el run() del worker; aquí repoblamos.
        self._populate()

    def _on_check_thread_finished(self):
        # Libera referencia una vez el QThread queda idle.
        self._upstream_worker = None

    def _refresh(self):
        # Refresh manual: repobla inmediatamente + fuerza un HEAD fresco.
        self._populate()
        self._schedule_upstream_check(delay_ms=0, force=True)

    def _open_logs(self):
        log_dir = get_logs_dir()
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        try:
            if platform.system() == "Windows":
                os.startfile(log_dir)  # noqa: S606
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", log_dir])
            else:
                subprocess.Popen(["xdg-open", log_dir])
        except Exception as exc:
            QMessageBox.warning(self, "Open logs", f"No se pudo abrir la carpeta de logs:\n{exc}")

    @staticmethod
    def _dep_display_name(dep_id):
        try:
            for d in dependencies.list_all():
                if d.get("id") == dep_id:
                    return d.get("display_name") or dep_id
        except Exception:
            pass
        return dep_id

    def _on_redownload_requested(self, dep_id):
        if not dep_id:
            return
        if self._active_worker and self._active_worker.isRunning():
            QMessageBox.information(self, "Ocupado", "Otra descarga está en curso. Espera a que termine.")
            return
        display = self._dep_display_name(dep_id)
        # Confirmación rápida
        confirm = QMessageBox.question(
            self,
            "Reinstalar dependencia",
            f"¿Volver a descargar e instalar '{display}'?\n"
            "La instalación actual será reemplazada.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        progress = QProgressDialog(
            f"Descargando {display}...\n"
            "Esto puede tardar unos minutos según tu conexión.",
            "Cancel", 0, 100, self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        self._active_worker = _ReinstallWorker(dep_id, parent=self)
        self._active_worker.progress.connect(progress.setValue)
        self._active_worker.finished_ok.connect(
            lambda did: self._on_redownload_finished(did, progress)
        )
        self._active_worker.failed.connect(
            lambda did, msg: self._on_redownload_failed(did, msg, progress)
        )
        self._active_worker.cancelled.connect(
            lambda did: self._on_redownload_cancelled(did, progress)
        )
        progress.canceled.connect(self._active_worker.cancel)
        self._active_worker.start()

    def _on_redownload_finished(self, dep_id, progress):
        progress.close()
        self._active_worker = None
        # Invalidar el cache upstream para que el siguiente check traiga una
        # huella fresca, no la anterior a esta reinstalación.
        try:
            clear_cache()
        except Exception:
            pass
        display = self._dep_display_name(dep_id)
        QMessageBox.information(
            self, "Listo", f"'{display}' se instaló correctamente."
        )
        self._populate()
        # Tras reinstalar queremos comparar la nueva build contra el upstream
        # (forzar — el cache fue limpiado, el siguiente check lo repuebla).
        self._schedule_upstream_check(delay_ms=1500, force=True)

    def _on_redownload_cancelled(self, dep_id, progress):
        progress.close()
        self._active_worker = None
        display = self._dep_display_name(dep_id)
        QMessageBox.information(
            self, "Cancelado",
            f"La instalación de '{display}' se canceló.",
        )
        self._populate()

    def _on_redownload_failed(self, dep_id, msg, progress):
        progress.close()
        self._active_worker = None
        # El fallo ya viene clasificado (firewall/SSL/proxy/espacio en disco)
        # por la sonda de conectividad o por el clasificador de pip: su primer
        # párrafo nombra la causa real. La pista inferior se adapta a esa
        # causa en lugar de culpar siempre a la conexión. Los detalles
        # técnicos completos quedan en "Mostrar detalles".
        lines = str(msg).strip().splitlines()
        first_block = "\n".join(lines[:3]).strip() if lines else str(msg)
        low = first_block.lower()
        if "espacio" in low or "space" in low:
            hint = "Libera espacio en disco y vuelve a intentarlo."
        elif "ssl" in low or "certificado" in low:
            hint = "Revisa los certificados SSL y el proxy de tu red."
        elif "proxy" in low:
            hint = "Revisa la configuración del proxy (posible autenticación)."
        elif "firewall" in low or "red" in low or "conexión" in low or "dns" in low:
            hint = "Revisa el firewall, la conexión y el DNS, y vuelve a intentarlo."
        else:
            hint = "Revisa el motivo indicado arriba y vuelve a intentarlo."
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("La instalación falló")
        box.setText(
            f"No se pudo instalar '{dep_id}'.\n\n{first_block}\n\n"
            f"{hint}\n\nLos detalles completos están en 'Mostrar detalles'."
        )
        box.setDetailedText(str(msg))
        box.exec()

    # -----------------------------------------------------------------
    # Uninstall (borrar lo instalado: binarios / runtimes / modelos)
    # -----------------------------------------------------------------

    def _render_in_progress(self) -> bool:
        """True si MainWindow tiene un CompressionWorker activo."""
        try:
            mw = self.parent()
            w = getattr(mw, "worker", None)
            if w is not None and w.isRunning():
                return True
        except Exception:
            pass
        return False

    def _on_uninstall_requested(self, dep_id):
        if not dep_id:
            return
        if self._uninstall_worker and self._uninstall_worker.isRunning():
            QMessageBox.information(
                self, "Busy", "Otro borrado está en curso. Espera a que termine."
            )
            return
        if self._render_in_progress():
            QMessageBox.warning(
                self, "Render in progress",
                "No se puede desinstalar mientras hay un render/compresión en curso.\n"
                "Espera a que termine antes de continuar.",
            )
            return

        info = next(
            (d for d in dependencies.list_all() if d.get("id") == dep_id), None
        )
        path = (info or {}).get("install_dir") or dep_id
        size = (info or {}).get("size_bytes", 0) or 0
        size_txt = f" (~{_fmt_bytes(size)})" if size else ""

        confirm = QMessageBox.question(
            self,
            "Uninstall dependency",
            f"¿Seguro que quieres borrar '{dep_id}'?\n\n"
            f"Se eliminará de forma permanente desde:\n  {path}{size_txt}\n\n"
            "Esta acción no se puede deshacer.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._uninstall_worker = _UninstallWorker(dep_id, parent=self)
        self._uninstall_worker.finished_ok.connect(self._on_uninstall_finished)
        self._uninstall_worker.failed.connect(self._on_uninstall_failed)
        self._uninstall_worker.start()

    def _on_uninstall_finished(self, dep_id):
        self._uninstall_worker = None
        QMessageBox.information(
            self, "Done", f"'{dep_id}' se ha desinstalado correctamente."
        )
        self._populate()
        self._refresh_engines_tab()

    def _on_uninstall_failed(self, dep_id, msg):
        self._uninstall_worker = None
        QMessageBox.critical(
            self, "Uninstall failed",
            f"No se pudo desinstalar '{dep_id}':\n\n{msg}\n\n"
            "Si algún archivo está en uso, el borrado se completará en el "
            "próximo arranque de la aplicación.",
        )
        self._populate()

    # ====================================================================
    # AI Engines tab handlers
    # ====================================================================
    #
    # Las cuatro señales emitidas por ``EnginesTab`` se enrutan aquí. Cada
    # handler es responsable de:
    #   - resolver el engine_id a su metadata (hf_url, install_dir, etc.)
    #   - actualizar el estado persistente (engines.json) ANTES y DESPUÉS
    #   - lanzar el QThread apropiado o mutar el widget appropriate
    #
    # Decisión: cada acción (install / convert / toggle) REPUEBLA sólo el
    # panel AI Engines (/ self.engines_tab._rebuild() si existiera). El
    # panel de binarios no se toca porque su contenido no cambió.
    # --------------------------------------------------------------------

    @staticmethod
    def _find_engine(engine_id):
        # Resuelve contra el catálogo estático Y el registro de motores de
        # usuario (instalados desde el buscador del Hub).
        from src.ai.registry import get_entry
        return get_entry(engine_id)

    def _refresh_engines_tab(self):
        """Repaints the engines panel after a state-changing action.

        No-op en modo Dependencies (sin catálogo): ``engines_tab`` es
        None y los handlers de motores no existen.

        Preferred path: ``EnginesTab.repopulate()`` re-emits every row
        from disk while preserving the user's scroll position in the
        same QScrollArea. Fallback: rebuild the panel if repopulate
        raises (for example if the layout has been corrupted).

        Guard ``_engines_rebuilding``: ``get_state()`` puede disparar un
        self-heal (``patch`` -> bus ``state_changed``) DURANTE el propio
        repopulate, lo que reentra en este método y reconstruye la lista
        dos veces (el lag de apertura). El flag corta esa reentrada.
        """
        if getattr(self, "_engines_rebuilding", False):
            return
        if self.engines_tab is None:
            return
        self._engines_rebuilding = True
        try:
            try:
                self.engines_tab.repopulate()
                return
            except Exception:
                pass

            old_tab = self.engines_tab
            self.engines_tab = EnginesTab(self)
            self.engines_tab.install_requested.connect(self._on_engine_install)
            self.engines_tab.convert_requested.connect(self._on_engine_convert)
            self.engines_tab.backend_changed.connect(self._on_engine_backend_changed)
            self.engines_tab.cancel_convert_requested.connect(self._on_engine_cancel_convert)
            self.engines_tab.open_folder.connect(self._on_engine_open_folder)
            self.engines_tab.remove_requested.connect(self._on_engine_remove)
            self.layout().replaceWidget(old_tab, self.engines_tab)
            old_tab.deleteLater()
        finally:
            self._engines_rebuilding = False

    def _on_engine_install(self, engine_id):
        if not engine_id:
            return
        if self._git_worker and self._git_worker.isRunning():
            QMessageBox.information(
                self, "Busy",
                "Another engine installation is in progress."
            )
            return

        engine = self._find_engine(engine_id)
        if not engine:
            QMessageBox.warning(self, "Engine not found",
                                f"Unknown engine_id: {engine_id}")
            return
        if not engine.get("is_implemented"):
            QMessageBox.information(
                self, "Coming soon",
                f"{engine.get('display_name', engine_id)} - backend integration pending. "
                "We will surface the install button when the runtime hook is wired."
            )
            return

        target_dir = get_engine_dir(engine_id)
        # Un directorio con pesos reales ya es una instalación: confirmar.
        # Un directorio con solo código fuente (clone antiguo sin pesos) es
        # basura a reparar: se procede directamente con la descarga.
        if os.path.isdir(target_dir) and os.listdir(target_dir) and dir_has_weights(target_dir):
            confirm = QMessageBox.question(
                self, "Engine already installed",
                f"The directory for {engine.get('display_name', engine_id)} already "
                f"exists at:\n{target_dir}\n\nRe-clone may fail if it is not empty. "
                "Continue?",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        # Choose download strategy: prefer huggingface_hub (no git needed),
        # fall back to git clone if huggingface_hub is not installed.
        use_hf_hub = is_hf_hub_available()
        repo_id = engine.get("hf_repo_id", "")

        if use_hf_hub and repo_id:
            # Direct download via huggingface_hub — no git required.
            progress = QProgressDialog(
                f"Downloading {engine.get('display_name', engine_id)} from HuggingFace...",
                None, 0, 100, self,
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setAutoClose(False)

            self._git_worker = HFDownloadWorker(repo_id, target_dir, parent=self)
            self._git_worker.progress.connect(progress.setValue)
            self._git_worker.finished_ok.connect(
                lambda td: self._on_engine_install_ok(engine_id, td, progress)
            )
            self._git_worker.failed.connect(
                lambda msg: self._on_engine_install_failed(engine_id, msg, progress)
            )
            self._git_worker.start()
        elif is_git_available():
            # Git fallback.
            url = engine.get("hf_url", "")
            progress = QProgressDialog(
                f"Cloning {engine.get('display_name', engine_id)} from HuggingFace...",
                None, 0, 100, self,
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setAutoClose(False)

            self._git_worker = GitCloneWorker(url, target_dir, parent=self)
            self._git_worker.progress.connect(progress.setValue)
            self._git_worker.finished_ok.connect(
                lambda td: self._on_engine_install_ok(engine_id, td, progress)
            )
            self._git_worker.failed.connect(
                lambda msg: self._on_engine_install_failed(engine_id, msg, progress)
            )
            self._git_worker.start()
        else:
            # Neither huggingface_hub nor git available.
            QMessageBox.warning(
                self, "No download method available",
                "Neither Git nor huggingface_hub is available.\n\n"
                "Install one of:\n"
                "  pip install huggingface_hub\n"
                "  git (from git-scm.com)\n\n"
                "Or manually download the model files into:\n"
                f"  {target_dir}",
            )

    def _on_engine_install_ok(self, engine_id, target_dir, progress):
        progress.close()
        self._git_worker = None

        # Validar que la descarga produjo pesos de modelo reales. Un clone
        # de un repo de código fuente (solo .py/README) crea el directorio
        # pero NO es una instalación válida: el render fallaría después con
        # "No model weights found" sin explicación.
        engine = self._find_engine(engine_id) or {}
        sentinel = (engine.get("sentinel", "") or "").strip()
        repo_id = (engine.get("hf_repo_id", "") or "").strip()
        algorithmic = not repo_id and not sentinel  # motores sin repo HF: nada que descargar
        has_weights = dir_has_weights(target_dir)
        has_sentinel = bool(sentinel) and os.path.isfile(os.path.join(target_dir, sentinel))
        if not algorithmic and not has_weights and not has_sentinel:
            self._on_engine_install_failed(
                engine_id,
                "Download finished but no model weight files were found in:\n"
                f"  {target_dir}\n\n"
                "The source is a code-only repository (source files, no "
                "pretrained weights).\n\n"
                "Options:\n"
                "  - Try another engine from the catalog.\n"
                "  - Or download the .pth/.safetensors/.onnx files manually "
                "into that folder and re-open AI Engines.",
                progress,
            )
            return

        try:
            engines_mark_installed(engine_id, install_path=target_dir)
        except Exception:
            pass

        # Motores de usuario (Hub): registrar arch/scale reales detectados
        # de los pesos recién descargados + el archivo de pesos usado como
        # sentinel. El id del motor de usuario se creó en la selección
        # (main_window), así que aquí solo enriquecemos el registro.
        try:
            from src.ai.user_models import is_user_engine as _is_user_engine
            if _is_user_engine(engine_id):
                self._refresh_user_model_record(engine_id, target_dir)
        except Exception:
            logger.exception("Failed to enrich user model record for %s", engine_id)

        # Benchmark en background: mide ONNX y PyTorch para que el selector
        # de backend de la fila compare ms/frame y el ETA del render tenga
        # datos. Guard: si ya hay un benchmark en marcha no lanzar otro.
        if has_weights:
            self._maybe_benchmark_backends(engine_id, target_dir)

        self._refresh_engines_tab()
        QMessageBox.information(
            self, "Engine added",
            f"'{engine_id}' installed successfully into:\n{target_dir}\n\n"
            "Select it from the workflow node to use it.",
        )

    def _refresh_user_model_record(self, engine_id, target_dir):
        """Detecta arch/scale reales de un motor de usuario ya descargado.

        En la selección del Hub se registró el modelo con display_name,
        categoría y repo; aquí se enriquece el registro con ``weight_file``,
        ``arch`` y ``scale`` deducidos del archivo de pesos en disco.
        """
        try:
            import os
            from src.ai.user_models import get_model, register
            from src.ai.architectures import detect_arch_and_scale
        except Exception as exc:
            logger.warning("user-model refresh helpers unavailable: %s", exc)
            return

        rec = get_model(engine_id) or {}
        weight_file = (rec.get("weight_file") or "").strip()
        weight_path = os.path.join(target_dir, weight_file) if weight_file else ""
        if not os.path.isfile(weight_path):
            # Sin sentinel válido: localizar el primer peso de torch del repo
            # (prefiere .safetensors/.pth/.pt sobre .onnx, que no se detecta).
            candidates = []
            for root, dirs, files in os.walk(target_dir):
                for fn in files:
                    low = fn.lower()
                    if low.endswith((".safetensors", ".pth", ".pt")):
                        candidates.append((0, os.path.join(root, fn)))
                    elif low.endswith(".onnx"):
                        candidates.append((1, os.path.join(root, fn)))
            candidates.sort(key=lambda c: (c[0], c[1].lower()))
            if candidates:
                weight_path = candidates[0][1]
                weight_file = os.path.relpath(weight_path, target_dir)

        arch, scale = (None, None)
        if weight_path and os.path.isfile(weight_path):
            arch, scale = detect_arch_and_scale(weight_path)

        fields = {}
        if weight_file:
            fields["weight_file"] = weight_file.replace("\\", "/")
        if arch:
            fields["arch"] = arch
        if scale:
            fields["scale"] = int(scale)
        if fields:
            register(engine_id, **fields)

    def _on_engine_install_failed(self, engine_id, msg, progress):
        progress.close()
        try:
            engines_mark_failed(engine_id, msg)
        except Exception:
            pass
        self._git_worker = None
        self._refresh_engines_tab()
        hint = "\n\nHints:"
        if not is_git_available() and not is_hf_hub_available():
            hint += "\n  - Neither Git nor huggingface_hub is available."
            hint += "\n  - Install: pip install huggingface_hub"
            hint += "\n  - Or install Git from git-scm.com."
        elif not is_git_available():
            hint += "\n  - Git is not on PATH. Install it from git-scm.com."
        hint += "\n  - Or drop the model files manually into:"
        hint += f"\n    {get_engine_dir(engine_id)}"
        hint += "\n    and re-open the AI Engines tab so it detects them."

        # Check if this is an auth-related failure (403/401/Access Denied).
        is_auth_error = any(
            kw in msg for kw in ("401", "403", "Access Denied", "token")
        )
        if is_auth_error:
            from PySide6.QtWidgets import QMessageBox as _MB
            box = _MB(self)
            box.setIcon(_MB.Icon.Critical)
            box.setWindowTitle("Add Engine failed")
            box.setText(f"Could not install '{engine_id}':\n\n{msg}{hint}")
            btn_get_token = box.addButton("Open Token Page", _MB.ButtonRole.ActionRole)
            box.addButton(_MB.StandardButton.Ok)
            box.exec()
            if box.clickedButton() == btn_get_token:
                import webbrowser
                webbrowser.open("https://huggingface.co/settings/tokens")
        else:
            QMessageBox.critical(
                self, "Add Engine failed",
                f"Could not install '{engine_id}':\n\n{msg}{hint}",
            )

    def _on_engine_convert(self, engine_id):
        if not engine_id:
            return
        if self._onnx_worker and self._onnx_worker.isRunning():
            QMessageBox.information(
                self, "Busy",
                "Another conversion is already running."
            )
            return

        engine = self._find_engine(engine_id)
        if not engine:
            return
        state = engines_state_get(engine_id)
        if not state.get("installed"):
            QMessageBox.information(
                self, "Install first",
                f"Install '{engine.get('display_name', engine_id)}' before converting "
                "its weights to ONNX."
            )
            return

        target_dir = state.get("install_path") or get_engine_dir(engine_id)

        # Conversión en segundo plano: el progreso se muestra en la propia
        # fila del modelo (barra + estado + botón Cancelar). Nada bloquea la
        # UI; el usuario puede seguir usando el diálogo mientras convierte.
        self._onnx_worker = OnnxConvertWorker(
            engine_id,
            target_dir,
            engine.get("category", ""),
            parent=self,
        )
        self._onnx_worker_engine_id = engine_id
        def _set_converting(**kwargs):
            if self.engines_tab is not None:
                self.engines_tab.set_engine_converting(engine_id, True, **kwargs)

        self._onnx_worker.progress.connect(
            lambda p: _set_converting(progress=p)
        )
        self._onnx_worker.status.connect(
            lambda s: _set_converting(status=_friendly_convert_status(s))
        )
        self._onnx_worker.finished_ok.connect(
            lambda out: self._on_engine_convert_ok(engine_id, out)
        )
        self._onnx_worker.failed.connect(
            lambda msg: self._on_engine_convert_failed(engine_id, msg)
        )
        _set_converting(status="Preparando la conversión...")
        self._onnx_worker.start()

    def _on_engine_convert_ok(self, engine_id, out_path):
        """Conversión terminada: marca el estado ONNX (el repintado muestra el
        pill 'INSTALLED · ONNX'). Sin diálogo modal."""
        try:
            engines_mark_onnx(engine_id, onnx_path=out_path, converted=True)
        except Exception:
            pass
        self._onnx_worker = None
        self._onnx_worker_engine_id = None
        # Medir el backend ONNX recién convertido en background para que el
        # selector muestre ms/frame comparado (ONNX vs PyTorch).
        self._maybe_benchmark_backends(engine_id)
        self._refresh_engines_tab()

    def _on_engine_convert_failed(self, engine_id, msg):
        """Conversión fallida/cancelada: el error queda visible en la fila.

        No se escribe estado ni se muestra diálogo modal: un fallo de
        reconversión no debe desmarcar un .onnx válido previo, y el mensaje
        rojo en la fila es la retroalimentación inmediata.
        """
        self._onnx_worker = None
        self._onnx_worker_engine_id = None
        if self.engines_tab is None:
            return
        self.engines_tab.set_engine_converting(engine_id, False)
        if "cancell" in (msg or "").lower():
            self.engines_tab.show_engine_convert_cancelled(engine_id)
        else:
            self.engines_tab.show_engine_convert_error(engine_id, msg)

    def _maybe_benchmark_backends(self, engine_id, target_dir=None):
        """Lanza (una sola vez) la medición de ambos backends en background.

        Guard de carrera: si un benchmark previo sigue en marcha no se
        lanza otro — la caché quedaría con dos writers pisándose.
        """
        try:
            if target_dir is None:
                state = engines_state_get(engine_id)
                target_dir = state.get("install_path") or get_engine_dir(engine_id)
            if not target_dir or not os.path.isdir(target_dir):
                return
        except Exception:
            return
        bench_running = (
            getattr(self, "_bench_worker", None) is not None
            and self._bench_worker.isRunning()
        )
        if bench_running:
            return
        self._bench_worker = _BenchmarkWorker(engine_id, target_dir, parent=self)
        self._bench_worker.finished.connect(
            lambda: setattr(self, "_bench_worker", None)
        )
        self._bench_worker.start()

    def _on_engine_cancel_convert(self, engine_id):
        """El usuario pulsó Cancelar en la fila: el worker coopera para
        abortar entre fases (un export de torch en marcha no se interrumpe
        a mitad, pero el resultado se descarta como cancelado)."""
        worker = self._onnx_worker
        if not worker or not worker.isRunning():
            return
        worker.cancel()
        if self.engines_tab is not None:
            self.engines_tab.set_engine_converting(
                engine_id, True, status="Cancelando..."
            )
            self.engines_tab.set_cancel_convert_enabled(engine_id, False)

    def _on_engine_backend_changed(self, engine_id, backend):
        """Persiste el motor de inferencia elegido (auto/torch/onnxruntime).

        Si el usuario fuerza ONNX Runtime y el modelo aún no tiene .onnx,
        se lanza la conversión automáticamente para que la elección sea
        real: el render podrá usar ONNX en cuanto termine.
        """
        if not engine_id:
            return
        backend = str(backend or "auto").strip().lower()
        if backend not in ("torch", "onnxruntime"):
            backend = "auto"
        try:
            engines_set_backend(engine_id, backend)
        except Exception:
            pass
        self._refresh_engines_tab()
        if backend != "onnxruntime":
            return
        try:
            state = engines_state_get(engine_id)
            if state.get("onnx_converted"):
                return
            model_dir = state.get("install_path") or get_engine_dir(engine_id)
            from src.ai.inference import find_onnx_model
            if find_onnx_model(model_dir):
                return
        except Exception:
            return
        self._on_engine_convert(engine_id)

    def _on_engine_open_folder(self, engine_id):
        engine = self._find_engine(engine_id)
        if not engine:
            return
        path = get_engine_dir(engine_id)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
        try:
            if platform.system() == "Windows":
                os.startfile(path)  # noqa: S606
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            QMessageBox.warning(self, "Open folder",
                                f"Could not open the engine folder:\n{exc}")

    # -----------------------------------------------------------------
    # Remove engine (borrar modelos / FFmpeg full desde la app)
    # -----------------------------------------------------------------

    def _on_engine_remove(self, engine_id):
        if not engine_id:
            return
        if self._remove_worker and self._remove_worker.isRunning():
            QMessageBox.information(
                self, "Busy", "Otro borrado está en curso. Espera a que termine."
            )
            return
        if self._render_in_progress():
            QMessageBox.warning(
                self, "Render in progress",
                "No se puede borrar un modelo mientras hay un render/compresión en curso.\n"
                "Espera a que termine antes de continuar.",
            )
            return

        engine = self._find_engine(engine_id)
        name = (engine or {}).get("display_name", engine_id)
        state = engines_state_get(engine_id)
        if not state.get("installed"):
            return

        target_dir = state.get("install_path") or get_engine_dir(engine_id)
        what = "model weights\n  " + target_dir
        affected = [engine_id]
        extra = ""

        size = dependencies._safe_dir_size(target_dir)
        size_txt = f" (~{_fmt_bytes(size)})" if size else ""

        confirm = QMessageBox.question(
            self,
            "Remove engine",
            f"¿Borrar '{name}'?\n\n"
            f"Se eliminará de forma permanente:{size_txt}\n  {what}{extra}\n\n"
            "Esta acción no se puede deshacer.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        def _remove():
            dependencies.remove_local_path(target_dir)
            for eid in affected:
                try:
                    engines_mark_uninstalled(eid)
                except Exception:
                    pass
                try:
                    from src.ai.user_models import is_user_engine as _is_user_engine
                    from src.ai.user_models import unregister as _unregister
                    if _is_user_engine(eid):
                        _unregister(eid)
                except Exception:
                    pass

        self._remove_worker = _EngineRemoveWorker(engine_id, _remove, parent=self)
        self._remove_worker.finished_ok.connect(self._on_engine_remove_ok)
        self._remove_worker.failed.connect(self._on_engine_remove_failed)
        self._remove_worker.start()

    def _on_engine_remove_ok(self, engine_id):
        self._remove_worker = None
        try:
            engines_clear_benchmark(engine_id)
        except Exception:
            pass
        QMessageBox.information(
            self, "Engine removed",
            f"'{engine_id}' se ha eliminado correctamente.",
        )
        self._refresh_engines_tab()
        self._populate()

    def _on_engine_remove_failed(self, engine_id, msg):
        self._remove_worker = None
        QMessageBox.critical(
            self, "Remove failed",
            f"No se pudo eliminar '{engine_id}':\n\n{msg}\n\n"
            "Si algún archivo está en uso, el borrado se completará en el "
            "próximo arranque de la aplicación.",
        )
        self._refresh_engines_tab()
        self._populate()

    # ====================================================================
    # Dialog lifecycle / worker cancellation
    # ====================================================================
    #
    # QThread is parented to ``self``, but ``subprocess.Popen`` keeps
    # the underlying OS process alive past QThread.terminate() because
    # the Python loop doesn't act on Qt's terminate signal. Adding
    # explicit cancel hooks here stops the active git / optimum-cli
    # subprocess before the dialog tears down so the OS process is
    # never orphaned.
    # --------------------------------------------------------------------

    def accept(self):
        self._kill_workers()
        super().accept()

    def reject(self):
        self._kill_workers()
        super().reject()

    def _kill_workers(self):
        for w_attr in ("_git_worker", "_onnx_worker",
                       "_uninstall_worker", "_remove_worker", "_bench_worker",
                       "_upstream_worker", "_active_worker"):
            w = getattr(self, w_attr, None)
            if w is None:
                continue
            try:
                if w.isRunning():
                    try:
                        w.cancel()
                    except Exception:
                        pass
                    try:
                        w.wait(2000)
                    except Exception:
                        pass
            except Exception:
                pass
            setattr(self, w_attr, None)
