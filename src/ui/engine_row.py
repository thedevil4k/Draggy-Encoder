"""Visual cell for one AI engine row inside the AI Engines tab.

Two custom widgets:

    QualityBar        - 8px-tall horizontal bar drawn with QPainter.
                        Just two rounded rects: track + fill. Reliable
                        across Windows/Linux without depending on
                        ``QProgressBar``-quirky styling.
    EngineRow(QFrame) - the row itself, with all metadata + actions:
                        Add Engine / Convert to ONNX / Open folder.

The row is *state-driven*: it receives ``engine`` (catalog entry as
returned by ``src.ai.registry.iter_catalog``) and renders accordingly.
Group-on-style subcategory badges are simple colored pills drawn
analogously to ``_STATUS_COLORS`` in dependencies_dialog.py.

Note:
    There is no global "Enabled" toggle. The workflow itself selects
    which installed engine to use for each run.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from src.ai.fetcher import is_git_available as _is_git_available
from src.ai.fetcher import is_hf_hub_available as _is_hf_hub_available
from src.ai.architectures import (
    ONNX_CONVERTIBLE_ARCHS as _ONNX_CONVERTIBLE_ARCHS,
)


# ---------------------------------------------------------------------------
# Quality bars
# ---------------------------------------------------------------------------


class QualityBar(QWidget):
    """A compact 0..100 progress bar drawn manually for cross-OS stability."""

    def __init__(self, score: int, color_hex: str = "#00d2d3",
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self.score = max(0, min(100, int(score or 0)))
        self.color = QColor(color_hex)
        self.setToolTip(tooltip or f"{self.score}/100")
        self.setFixedHeight(8)
        self.setMinimumWidth(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_score(self, score: int) -> None:
        self.score = max(0, min(100, int(score or 0)))
        self.update()

    def paintEvent(self, event):  # noqa: D401 - Qt callback name
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            track = rect.adjusted(0, 2, 0, -2)
            painter.setBrush(QColor("#1f2c4c"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(track, 3, 3)
            if self.score > 0:
                fill_w = max(1, int(track.width() * self.score / 100.0))
                fill = track.adjusted(0, 0, fill_w - track.width(), 0)
                painter.setBrush(self.color)
                painter.drawRoundedRect(fill, 3, 3)
        finally:
            painter.end()


# ---------------------------------------------------------------------------
# Category badge (pill coloring)
# ---------------------------------------------------------------------------

_CATEGORY_COLORS = {
    "Upscale":      ("#00d2d3", "#0f0f1a"),
    "Interpolate":  ("#5f27cd", "#ffffff"),
    "Colorize":     ("#e74c3c", "#ffffff"),
    "Denoise":      ("#f39c12", "#0f0f1a"),
    "FaceRestore":  ("#16a085", "#ffffff"),
}


def _category_colors(cat: str) -> tuple[str, str]:
    return _CATEGORY_COLORS.get(cat, ("#2c3e50", "#e0e0e0"))


# ---------------------------------------------------------------------------
# Engine row
# ---------------------------------------------------------------------------


class EngineRow(QFrame):
    """A single engine card displayed inside the AI Engines tab.

    Signals:
        install_requested(str) - engine_id to install (HF clone / git clone).
        fsr31_download_requested(str) - engine_id; FSR 3.1 entry clicked. The
            parent dialog handles this by downloading the FFmpeg "full" build
            (with libplacebo) into ``bin/ffmpeg-full/`` instead of cloning a
            model repo. Marked installed on success.
        convert_requested(str) - engine_id to convert to ONNX.
        backend_changed(str, str) - (engine_id, "auto"|"torch"|"onnxruntime")
            fired when the user switches the inference backend for this engine.
        cancel_convert_requested(str) - engine_id; the user clicked Cancel
            while the ONNX conversion was running.
        open_folder(str) - engine_id; user wants to see the local weights dir.

    Note:
        There is no global "Enabled" toggle. The workflow itself decides
        which installed engine is active for each run, so the row only
        shows installed/converted state and offers actions per engine.
    """

    install_requested = Signal(str)
    fsr31_download_requested = Signal(str)
    convert_requested = Signal(str)
    backend_changed = Signal(str, str)
    cancel_convert_requested = Signal(str)
    open_folder = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, engine: dict, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("EngineRow")
        # Paint opaco: evita el compositing translúcido del QSS en cada
        # repaint y reduce el lag de scroll en listas largas.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setStyleSheet(self._stylesheet())
        self._build()

    # -- styling ----------------------------------------------------------

    def _stylesheet(self) -> str:
        return """
        #EngineRow {
            background-color: #16213e;
            border: 1px solid #2c3e50;
            border-radius: 8px;
        }
        #EngineRow QPushButton {
            background-color: #1f2c4c;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: bold;
        }
        #EngineRow QPushButton:hover {
            background-color: #00d2d3;
            color: #0f0f1a;
        }
        #EngineRow QPushButton:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #EngineRow QPushButton#removeBtn {
            background-color: #3a1620;
            color: #ff6b6b;
            border: 1px solid #6e2233;
        }
        #EngineRow QPushButton#removeBtn:hover {
            background-color: #e74c3c;
            color: #ffffff;
        }
        #EngineRow QPushButton#removeBtn:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #EngineRow QPushButton#primaryBtn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2d3, stop:1 #5f27cd);
            color: #0f0f1a;
            border: none;
        }
        #EngineRow QPushButton#primaryBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00f0f1, stop:1 #7d3fe0);
        }
        #EngineRow QCheckBox {
            color: #e0e0e0;
            spacing: 6px;
        }
        #EngineRow QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        #EngineRow QLabel {
            color: #e0e0e0;
        }
        #EngineRow QLabel#title {
            font-size: 13px;
            font-weight: bold;
            color: #ffffff;
        }
        #EngineRow QLabel#meta {
            color: #a0a0b0;
            font-size: 10px;
            font-family: Consolas, monospace;
        }
        #EngineRow QLabel#url {
            color: #00d2d3;
            font-size: 10px;
        }
        """

    # -- build ------------------------------------------------------------

    def _build(self) -> None:
        # Estructura de la tarjeta: fila principal (left/middle/right) + una
        # franja inferior oculta para el progreso de conversión a ONNX.
        outer = QVBoxLayout(self)
        outer.setSpacing(4)
        outer.setContentsMargins(14, 12, 14, 12)
        top = QHBoxLayout()
        top.setSpacing(14)

        eng = self.engine
        eid = eng.get("engine_id", "")
        name = eng.get("display_name", eid)
        cat = eng.get("category", "Other")
        link = eng.get("hf_url", "")
        size = eng.get("size_estimate", "")
        shape = eng.get("sample_shape", "")
        lic = eng.get("license", "")
        pop = int(eng.get("popularity_score", 0))
        qual = int(eng.get("quality_score", 0))
        implemented = bool(eng.get("is_implemented", False))
        lfs_warn = bool(eng.get("lfs_warning", False))

        # ---- Left column: title + meta + URL ----
        left = QVBoxLayout()
        left.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        lbl_title = QLabel(name)
        lbl_title.setObjectName("title")
        title_row.addWidget(lbl_title)

        cat_bg, cat_fg = _category_colors(cat)
        badge = QLabel(cat.upper())
        badge.setStyleSheet(
            f"background-color: {cat_bg}; color: {cat_fg}; "
            f"border-radius: 8px; padding: 1px 8px; font-size: 9px; font-weight: bold;"
        )
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(badge)

        # Motores de usuario instalados desde el buscador del Hub.
        if eng.get("is_user"):
            hub = QLabel("HUB")
            hub.setStyleSheet(
                "background-color: #8e44ad; color: #ffffff; "
                "border-radius: 8px; padding: 1px 8px; font-size: 9px; font-weight: bold;"
            )
            hub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hub.setToolTip("Community model installed from the HuggingFace Hub.")
            title_row.addWidget(hub)

        if not implemented:
            cs = QLabel("COMING SOON")
            cs.setStyleSheet(
                "background-color: #5f27cd; color: #ffffff; "
                "border-radius: 8px; padding: 1px 8px; font-size: 9px; font-weight: bold;"
            )
            cs.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_row.addWidget(cs)

        title_row.addStretch()
        left.addLayout(title_row)

        meta = QLabel(f"{eid}  ·  size: {size}  ·  license: {lic}")
        meta.setObjectName("meta")
        left.addWidget(meta)

        if shape:
            shape_lbl = QLabel(f"shape: {shape}")
            shape_lbl.setStyleSheet("color: #a0a0b0; font-size: 10px; font-family: Consolas, monospace;")
            left.addWidget(shape_lbl)

        url_lbl = QLabel(link)
        url_lbl.setObjectName("url")
        url_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(url_lbl)

        if lfs_warn:
            tip = QLabel("Weights in this repo use Git LFS — first clone downloads ~hundreds of MB.")
            tip.setStyleSheet("color: #f39c12; font-size: 10px;")
            tip.setWordWrap(True)
            left.addWidget(tip)

        top.addLayout(left, stretch=1)

        # ---- Middle column: quality bars + state pill ----
        middle = QVBoxLayout()
        middle.setSpacing(6)

        pop_row = QHBoxLayout()
        pop_row.setSpacing(6)
        pop_row.addWidget(QLabel("Popularity:"))
        pop_row.addWidget(QualityBar(pop, "#f39c12",
                                     tooltip=f"Popularity score: {pop}/100"))
        middle.addLayout(pop_row)

        qual_row = QHBoxLayout()
        qual_row.setSpacing(6)
        qual_row.addWidget(QLabel("Quality:"))
        qual_row.addWidget(QualityBar(qual, "#00d2d3",
                                     tooltip=f"Quality score: {qual}/100"))
        middle.addLayout(qual_row)

        # State badge — también muestra el motor de inferencia en uso
        # (PYTORCH por defecto, ONNX cuando está convertido; la preferencia
        # del usuario manda vía ``inference_backend``).
        state = eng.get("state", {}) or {}
        installed = bool(state.get("installed"))
        converted = bool(state.get("onnx_converted"))
        backend = str(state.get("inference_backend") or "auto").strip().lower()
        if backend not in ("torch", "onnxruntime"):
            backend = "auto"
        # El pill muestra el backend CONCRETO que usará el render: con
        # "auto" y benchmarks de ambos, el más rápido medido (misma regla
        # que ``resolve_backend`` en el render); sin benchmarks, la
        # heurística clásica (ONNX si hay .onnx, si no PyTorch).
        display_backend = backend
        if backend == "auto":
            try:
                from src.ai.engines_state import resolve_backend as _resolve_backend
                resolved = _resolve_backend(eng.get("engine_id", ""))
                if resolved in ("torch", "onnxruntime"):
                    display_backend = resolved
            except Exception:
                pass
        is_fsr31 = "fsr31" in eng.get("engine_id", "")
        is_fsr = is_fsr31

        engine_label = self._engine_label(installed, converted, display_backend, is_fsr31)
        state_lbl = QLabel(engine_label)
        state_qss = self._state_qss(installed, converted, display_backend, is_fsr31)
        state_lbl.setStyleSheet(state_qss)
        state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_lbl.setMaximumWidth(180)
        middle.addWidget(state_lbl, alignment=Qt.AlignmentFlag.AlignLeft)

        # Benchmarks medidos en este equipo por backend (ms/frame).
        bench = eng.get("benchmarks") or {}
        onnx_bench = bench.get("onnx") or None
        torch_bench = bench.get("torch") or None

        # Selector de motor de inferencia (Auto / PyTorch / ONNX Runtime).
        # Los items muestran los ms/frame medidos cuando existen.
        self.combo_backend = QComboBox()
        self.combo_backend.addItem("Auto", "auto")
        self.combo_backend.addItem(
            self._backend_item_label("PyTorch", torch_bench), "torch"
        )
        self.combo_backend.addItem(
            self._backend_item_label("ONNX Runtime", onnx_bench), "onnxruntime"
        )
        backend_idx = self.combo_backend.findData(backend)
        self.combo_backend.setCurrentIndex(backend_idx if backend_idx >= 0 else 0)
        is_backend_switchable = bool(
            implemented and installed and not is_fsr
        )
        self.combo_backend.setEnabled(is_backend_switchable)
        if not is_backend_switchable:
            self.combo_backend.setToolTip(
                "Backend selector requires an installed model engine "
                "(FSR engines are algorithmic and have no model backend)."
            )
        else:
            self.combo_backend.setToolTip(
                "Inference backend for this engine:\n"
                "  Auto        - the fastest measured backend (ONNX vs PyTorch)\n"
                "                when both are benchmarked; otherwise ONNX if\n"
                "                a .onnx exists, PyTorch if not.\n"
                "  PyTorch     - always use PyTorch (never converts).\n"
                "  ONNX Runtime - always use ONNX; converts the weights if needed.\n\n"
                "ms/frame medidos en este equipo a 512x288 cuando hay datos."
            )
        self.combo_backend.currentIndexChanged.connect(self._on_backend_changed)
        backend_row = QHBoxLayout()
        backend_row.setSpacing(6)
        backend_lbl = QLabel("Backend:")
        backend_lbl.setStyleSheet("color: #a0a0b0; font-size: 10px;")
        backend_row.addWidget(backend_lbl)
        backend_row.addWidget(self.combo_backend)
        backend_row.addStretch()
        middle.addLayout(backend_row)

        # Sugerencia automática: cuál de los dos backends es más rápido en
        # este equipo (solo con ambos medidos) + botón 'Usar' para aplicarlo.
        self._build_backend_suggestion(middle, onnx_bench, torch_bench)

        top.addLayout(middle)

        # ---- Right column: buttons + toggle ----
        right = QVBoxLayout()
        right.setSpacing(6)

        # Primary: Add Engine / Reinstall / Open Folder
        self.btn_install = QPushButton(self._install_label(installed, is_fsr31))
        self.btn_install.setObjectName("primaryBtn")
        if not implemented:
            self.btn_install.setEnabled(False)
            self.btn_install.setText("Coming soon")
        elif is_fsr31:
            if installed:
                self.btn_install.setEnabled(False)
                self.btn_install.setText("Ready")
                self.btn_install.setToolTip("FSR 3.1 is ready — FFmpeg full with libplacebo is available.")
            else:
                self.btn_install.setEnabled(True)
                self.btn_install.setText("Download FFmpeg")
                self.btn_install.setToolTip("FSR 3.1 requires FFmpeg full with libplacebo. Click to download.")
        self.btn_install.setToolTip(self._install_tooltip(eng))
        self.btn_install.clicked.connect(self._on_install_clicked)
        right.addWidget(self.btn_install)

        # Convert to ONNX button
        arch = (eng.get("arch") or "").strip().lower()
        arch_known = bool(arch)
        arch_convertible = arch in _ONNX_CONVERTIBLE_ARCHS
        self.btn_convert = QPushButton("Convert to ONNX")
        if not implemented:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("Convert — TBD")
        elif not installed:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("Convert to ONNX")
            self.btn_convert.setToolTip(
                "Install the engine first, then convert to ONNX. "
                "Converting .pth weights requires PyTorch; .onnx models "
                "only need ONNX Runtime."
            )
        elif is_fsr:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("N/A (built-in)")
            self.btn_convert.setToolTip("FSR is algorithmic — no ONNX conversion needed.")
        elif converted:
            self.btn_convert.setEnabled(True)
            self.btn_convert.setText("Re-convert ONNX")
            self.btn_convert.setToolTip(
                "Weights already converted to ONNX. Click to refresh. "
                "Re-converting requires PyTorch (torch.onnx.export)."
            )
        elif arch_known and not arch_convertible:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("No ONNX")
            self.btn_convert.setToolTip(
                f"Architecture '{arch}' cannot be exported to ONNX: only "
                "Real-ESRGAN (RRDB/SRVGG), Real-CUGAN and OmniSR weights "
                "are supported. The render will use PyTorch instead."
            )
        else:
            # Convertible + installed + not yet converted: the main path.
            self.btn_convert.setEnabled(True)
            self.btn_convert.setToolTip(
                "Convert .pth weights to .onnx with torch.onnx.export. "
                "Requires PyTorch — models that already ship as .onnx only "
                "need ONNX Runtime."
            )
        self.btn_convert.clicked.connect(self._on_convert_clicked)
        right.addWidget(self.btn_convert)

        # Open folder (always present if install_dir exists)
        self.btn_open = QPushButton("Open folder")
        inst_dir = eng.get("install_dir") or ""
        self.btn_open.setEnabled(bool(inst_dir))
        self.btn_open.clicked.connect(self._on_open_clicked)
        right.addWidget(self.btn_open)

        # Remove installed weights / binary (borrar desde la app).
        can_remove = installed
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setObjectName("removeBtn")
        self.btn_remove.setEnabled(can_remove)
        if is_fsr31:
            self.btn_remove.setToolTip(
                "Removes the FFmpeg full (libplacebo) build and uninstalls "
                "this engine (affects all FSR 3.1 engines)."
            )
        else:
            self.btn_remove.setToolTip(
                "Deletes the downloaded model weights from the program folder."
            )
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        right.addWidget(self.btn_remove)

        # "View similar on OpenModelDB" — solo si el catalog entry declara
        # ``similar_models_url``. Abre la URL en el navegador del usuario
        # para descubrir variantes / forks / nuevos escalados del mismo
        # modelo sin tener que navegar manualmente.
        similar_url = eng.get("similar_models_url", "") or ""
        self.btn_similar = QPushButton("View similar models")
        self.btn_similar.setObjectName("similarModelsBtn")
        self.btn_similar.setEnabled(bool(similar_url))
        if similar_url:
            self.btn_similar.setToolTip(
                "Open OpenModelDB to see similar models, forks, and alternatives.\n"
                f"{similar_url}"
            )
        else:
            self.btn_similar.setToolTip(
                "No OpenModelDB link configured for this engine."
            )
        self.btn_similar.clicked.connect(self._on_similar_clicked)
        right.addWidget(self.btn_similar)

        right.addStretch()
        top.addLayout(right)

        outer.addLayout(top)
        # Franja de progreso de conversión a ONNX (oculta por defecto).
        self._build_convert_area(outer)
        outer.addStretch(1)

    # -- Backend suggestion (measured ms/frame) ----------------------------

    @staticmethod
    def _backend_item_label(name: str, bench) -> str:
        """Texto del item del selector: nombre + ms/frame medido si existe."""
        label = name
        if bench and bench.get("ms_per_frame"):
            return f"{label} — {bench['ms_per_frame']} ms/f"
        return label

    @staticmethod
    def _recommended_backend(onnx_bench, torch_bench) -> Optional[str]:
        """Backend más rápido medido ("onnx"|"torch"), o None sin datos.

        Con empate se prefiere ONNX (el backend por defecto cuando existe
        .onnx)."""
        onnx_ms = onnx_bench.get("ms_per_frame") if onnx_bench else None
        torch_ms = torch_bench.get("ms_per_frame") if torch_bench else None
        if onnx_ms is None and torch_ms is None:
            return None
        if onnx_ms is None:
            return "torch"
        if torch_ms is None:
            return "onnx"
        if torch_ms < onnx_ms:
            return "torch"
        return "onnx"

    def _build_backend_suggestion(self, parent_layout, onnx_bench, torch_bench) -> None:
        """Línea '⚡ Más rápido: X (a ms/f vs b ms/f)' + botón 'Usar'."""
        self.lbl_backend_suggest = QLabel("")
        self.lbl_backend_suggest.setStyleSheet("color: #00d2d3; font-size: 10px;")
        self.lbl_backend_suggest.setVisible(False)
        self.btn_use_recommended = QPushButton("Usar")
        self.btn_use_recommended.setEnabled(False)
        self.btn_use_recommended.setVisible(False)
        self.btn_use_recommended.clicked.connect(self._on_use_recommended)
        rec_row = QHBoxLayout()
        rec_row.setSpacing(6)
        rec_row.addWidget(self.lbl_backend_suggest, stretch=1)
        rec_row.addWidget(self.btn_use_recommended)
        parent_layout.addLayout(rec_row)

        rec = self._recommended_backend(onnx_bench, torch_bench)
        if not rec or not (onnx_bench and torch_bench):
            # Solo se sugiere cuando hay ambos backends medidos y comparables.
            return
        fast = onnx_bench if rec == "onnx" else torch_bench
        slow = torch_bench if rec == "onnx" else onnx_bench
        fast_name = "ONNX Runtime" if rec == "onnx" else "PyTorch"
        slow_name = "PyTorch" if rec == "onnx" else "ONNX Runtime"
        self.lbl_backend_suggest.setText(
            f"⚡ Más rápido: {fast_name} ({fast['ms_per_frame']} ms/f vs "
            f"{slow_name} {slow['ms_per_frame']} ms/f)"
        )
        self.lbl_backend_suggest.setToolTip(
            "Medido en este equipo a 512x288. Pulsa 'Usar' para cambiar "
            "el backend al más rápido."
        )
        self.lbl_backend_suggest.setVisible(True)
        self.btn_use_recommended.setEnabled(True)
        self.btn_use_recommended.setVisible(True)
        self._recommended = rec

    def _on_use_recommended(self) -> None:
        rec = getattr(self, "_recommended", None)
        if not rec:
            return
        self.backend_changed.emit(self.engine.get("engine_id", ""), rec)

    # -- ONNX conversion progress strip ------------------------------------

    def _build_convert_area(self, parent_layout) -> None:
        """Franja inferior de la tarjeta con el progreso de conversión a ONNX.

        Oculto por defecto; ``set_converting(True, ...)`` lo muestra mientras
        el worker convierte en segundo plano y ``show_convert_error`` deja el
        mensaje de fallo visible en rojo.
        """
        area = QVBoxLayout()
        area.setSpacing(4)

        self.conv_bar = QProgressBar()
        self.conv_bar.setRange(0, 100)
        self.conv_bar.setValue(0)
        self.conv_bar.setTextVisible(True)
        self.conv_bar.setFixedHeight(14)
        self.conv_bar.setVisible(False)
        area.addWidget(self.conv_bar)

        self.conv_status = QLabel("")
        self.conv_status.setStyleSheet("color: #a0a0b0; font-size: 10px;")
        self.conv_status.setWordWrap(True)
        self.conv_status.setVisible(False)

        self.btn_cancel_convert = QPushButton("Cancel")
        self.btn_cancel_convert.setObjectName("removeBtn")
        self.btn_cancel_convert.setEnabled(False)
        self.btn_cancel_convert.setVisible(False)
        self.btn_cancel_convert.clicked.connect(self._on_cancel_convert_clicked)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addWidget(self.conv_status, stretch=1)
        status_row.addWidget(self.btn_cancel_convert)
        area.addLayout(status_row)

        parent_layout.addLayout(area)
        self._conv_area = area

    def set_converting(self, converting: bool, progress: Optional[int] = None,
                       status: str = "") -> None:
        """Muestra/oculta la franja de conversión y bloquea las acciones de la
        fila mientras el worker de ONNX está en marcha."""
        if not hasattr(self, "conv_bar"):
            return  # fila a medio construir o sin área (tests)
        if converting:
            if not getattr(self, "_converting_active", False):
                # Capturar los estados habilitados actuales para restaurarlos
                # al terminar (la fila NO se repuebla en fallo/cancel).
                self._convert_was_enabled = self.btn_convert.isEnabled()
                self._remove_was_enabled = self.btn_remove.isEnabled()
                self._backend_was_enabled = self.combo_backend.isEnabled()
            self._converting_active = True
            if progress is None or progress < 0:
                self.conv_bar.setRange(0, 0)  # modo ocupado / indeterminado
            else:
                self.conv_bar.setRange(0, 100)
                self.conv_bar.setValue(int(progress))
            self.conv_bar.setVisible(True)
            self.conv_status.setVisible(True)
            self.btn_cancel_convert.setVisible(True)
            self.btn_cancel_convert.setEnabled(True)
            self.btn_cancel_convert.setText("Cancel")
            if status:
                self.conv_status.setText(status)
                self.conv_status.setStyleSheet("color: #a0a0b0; font-size: 10px;")
            self.btn_convert.setEnabled(False)
            self.btn_remove.setEnabled(False)
            self.combo_backend.setEnabled(False)
        else:
            self._converting_active = False
            self.conv_bar.setVisible(False)
            self.conv_status.setVisible(False)
            self.btn_cancel_convert.setVisible(False)
            # Restaurar los estados anteriores solo si estaban capturados.
            if hasattr(self, "_convert_was_enabled"):
                self.btn_convert.setEnabled(self._convert_was_enabled)
            if hasattr(self, "_remove_was_enabled"):
                self.btn_remove.setEnabled(self._remove_was_enabled)
            if hasattr(self, "_backend_was_enabled"):
                self.combo_backend.setEnabled(self._backend_was_enabled)

    def set_cancel_enabled(self, enabled: bool) -> None:
        """Habilita/deshabilita el botón Cancelar ("Cancelling..." mientras
        el worker coopera para abortar)."""
        if not hasattr(self, "btn_cancel_convert"):
            return
        self.btn_cancel_convert.setEnabled(enabled)
        self.btn_cancel_convert.setText("Cancelling..." if not enabled else "Cancel")

    def show_convert_error(self, msg: str) -> None:
        """Deja el error de conversión visible en rojo en la fila."""
        if not hasattr(self, "conv_status"):
            return
        self.conv_bar.setVisible(False)
        self.btn_cancel_convert.setVisible(False)
        self.conv_status.setText(msg)
        self.conv_status.setStyleSheet(
            "color: #e74c3c; font-size: 10px; font-weight: bold;"
        )
        self.conv_status.setVisible(True)

    def show_convert_cancelled(self) -> None:
        """Aviso neutro de cancelación, auto-oculto a los 3 segundos."""
        if not hasattr(self, "conv_status"):
            return
        self.conv_bar.setVisible(False)
        self.btn_cancel_convert.setVisible(False)
        self.conv_status.setText("Conversion cancelled.")
        self.conv_status.setStyleSheet("color: #a0a0b0; font-size: 10px;")
        self.conv_status.setVisible(True)
        # Con contexto (``self``): si la fila se destruye antes de que el
        # timer dispare, Qt descarta el callback — un singleShot sin
        # contexto invocaría el método sobre un QWidget ya destruido y
        # reventaría con access violation en el hilo de la GUI.
        QTimer.singleShot(3000, self, self._hide_convert_status)

    def _hide_convert_status(self) -> None:
        if hasattr(self, "conv_status"):
            self.conv_status.setVisible(False)

    def _on_cancel_convert_clicked(self) -> None:
        self.cancel_convert_requested.emit(self.engine.get("engine_id", ""))

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _engine_label(installed: bool, converted: bool, backend: str,
                      is_fsr31: bool = False) -> str:
        """Etiqueta del motor en uso: PYTORCH por defecto, ONNX si hay .onnx
        (o si el usuario forzó el backend correspondiente)."""
        if not installed:
            return "NOT INSTALLED"
        if is_fsr31:
            return "FSR 3.1 (GPU)"
        if backend == "torch" or (backend == "auto" and not converted):
            return "INSTALLED · PYTORCH"
        return "INSTALLED · ONNX"

    @staticmethod
    def _state_qss(installed: bool, converted: bool, backend: str,
                   is_fsr31: bool = False) -> str:
        """Color del pill de estado según el motor activo."""
        if not installed:
            return (
                "QLabel { background-color: #2c3e50; color: #a0a0b0; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        if is_fsr31:
            # FSR 3.1 (GPU via libplacebo + interpolation) - teal-green
            return (
                "QLabel { background-color: #16a085; color: #ffffff; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        if backend == "torch" or (backend == "auto" and not converted):
            # PYTORCH (por defecto) - naranja
            return (
                "QLabel { background-color: #e67e22; color: #ffffff; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        # ONNX - teal
        return (
            "QLabel { background-color: #00d2d3; color: #0f0f1a; "
            "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
        )

    def _install_label(self, installed: bool, is_fsr31: bool = False) -> str:
        if is_fsr31:
            return "Download FFmpeg" if not installed else "Ready"
        return "Re-clone" if installed else "Add Engine"

    def _install_tooltip(self, eng: dict) -> str:
        bits = []
        if not eng.get("is_implemented"):
            bits.append("Backend integration pending — button is informational only.")
            return "\n".join(bits)
        if not _is_git_available() and not _is_hf_hub_available():
            bits.append("Neither Git nor huggingface_hub available. Install: pip install huggingface_hub")
        elif not _is_git_available():
            bits.append("Git is NOT on PATH (huggingface_hub will be used for download).")
        bits.append(f"Target: {eng.get('install_dir', '')}")
        bits.append("Download from: " + str(eng.get("hf_url", "")))
        return "\n".join(bits)

    def _on_install_clicked(self) -> None:
        eid = self.engine.get("engine_id", "")
        # FSR 3.1 entries route to a different signal so the parent dialog
        # downloads the FFmpeg "full" binary instead of trying to git-clone
        # an HF repo (which has no model weights for FSR).
        if "fsr31" in eid and eid.startswith("upscale-fsr"):
            self.fsr31_download_requested.emit(eid)
            return
        self.install_requested.emit(eid)

    def _on_convert_clicked(self) -> None:
        self.convert_requested.emit(self.engine.get("engine_id", ""))

    def _on_backend_changed(self, index: int) -> None:
        eid = self.engine.get("engine_id", "")
        if not eid:
            return
        backend = self.combo_backend.itemData(index)
        self.backend_changed.emit(eid, backend or "auto")

    def _on_remove_clicked(self) -> None:
        self.remove_requested.emit(self.engine.get("engine_id", ""))

    def _on_open_clicked(self) -> None:
        path = self.engine.get("install_dir") or ""
        if not path:
            return
        try:
            os.makedirs(path, exist_ok=True)
            if platform.system() == "Windows":
                os.startfile(path)  # noqa: S606
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            QMessageBox.warning(self, "Open folder",
                                f"Could not open folder:\n{exc}")

    def _on_similar_clicked(self) -> None:
        """Open the OpenModelDB page for this engine in the user's browser."""
        url = self.engine.get("similar_models_url", "") or ""
        if not url:
            return
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as exc:
            QMessageBox.warning(
                self, "View similar models",
                f"Could not open the browser:\n{exc}\n\nURL: {url}",
            )
