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

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from src.ai.fetcher import is_git_available as _is_git_available
from src.ai.fetcher import is_hf_hub_available as _is_hf_hub_available


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
        fsr2_download_requested(str) - engine_id; FSR 2.0 entry clicked. The
            parent dialog handles this by downloading the FFmpeg "full" build
            (with libplacebo) into ``bin/ffmpeg-full/`` instead of cloning a
            model repo. Marked installed on success.
        convert_requested(str) - engine_id to convert to ONNX.
        open_folder(str) - engine_id; user wants to see the local weights dir.

    Note:
        There is no global "Enabled" toggle. The workflow itself decides
        which installed engine is active for each run, so the row only
        shows installed/converted state and offers actions per engine.
    """

    install_requested = Signal(str)
    fsr2_download_requested = Signal(str)
    convert_requested = Signal(str)
    open_folder = Signal(str)

    def __init__(self, engine: dict, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("EngineRow")
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
        outer = QHBoxLayout(self)
        outer.setSpacing(14)
        outer.setContentsMargins(14, 12, 14, 12)

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

        outer.addLayout(left, stretch=1)

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

        # State badge
        state = eng.get("state", {}) or {}
        installed = bool(state.get("installed"))
        converted = bool(state.get("onnx_converted"))
        is_fsr = eng.get("engine_id", "").startswith("upscale-fsr")
        is_fsr2 = eng.get("engine_id", "").endswith("-v2")
        is_fsr31 = "fsr31" in eng.get("engine_id", "")

        state_text = self._state_label(installed, converted, is_fsr, is_fsr2)
        state_lbl = QLabel(state_text)
        # Inline the stylesheet so a stale .pyc or import problem cannot
        # trigger "'EngineRow' object has no attribute '_state_stylesheet'".
        if not installed:
            state_qss = (
                "QLabel { background-color: #2c3e50; color: #a0a0b0; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        elif is_fsr31:
            # FSR 3.0 (GPU via libplacebo + interpolation) - teal-blue color
            state_qss = (
                "QLabel { background-color: #16a085; color: #ffffff; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        elif is_fsr2:
            # FSR 2.0 (GPU via libplacebo) - blue color
            state_qss = (
                "QLabel { background-color: #2980b9; color: #ffffff; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        elif is_fsr:
            # FSR 1.0 (CPU) - purple color
            state_qss = (
                "QLabel { background-color: #8e44ad; color: #ffffff; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        elif converted:
            state_qss = (
                "QLabel { background-color: #00d2d3; color: #0f0f1a; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        else:
            state_qss = (
                "QLabel { background-color: #16a085; color: #ffffff; "
                "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: bold; }"
            )
        state_lbl.setStyleSheet(state_qss)
        state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_lbl.setMaximumWidth(180)
        middle.addWidget(state_lbl, alignment=Qt.AlignmentFlag.AlignLeft)

        outer.addLayout(middle)

        # ---- Right column: buttons + toggle ----
        right = QVBoxLayout()
        right.setSpacing(6)

        # Primary: Add Engine / Reinstall / Open Folder
        self.btn_install = QPushButton(self._install_label(installed, is_fsr, is_fsr2))
        self.btn_install.setObjectName("primaryBtn")
        if not implemented:
            self.btn_install.setEnabled(False)
            self.btn_install.setText("Coming soon")
        elif is_fsr and not is_fsr2:
            self.btn_install.setEnabled(False)
            self.btn_install.setText("Built-in")
            self.btn_install.setToolTip("FSR 1.0 is built into the application — no download needed.")
        elif is_fsr2:
            if installed:
                self.btn_install.setEnabled(False)
                self.btn_install.setText("Ready")
                self.btn_install.setToolTip("FSR 2.0 is ready — FFmpeg full with libplacebo is available.")
            else:
                self.btn_install.setEnabled(True)
                self.btn_install.setText("Download FFmpeg")
                self.btn_install.setToolTip("FSR 2.0 requires FFmpeg full with libplacebo. Click to download.")
        self.btn_install.setToolTip(self._install_tooltip(eng))
        self.btn_install.clicked.connect(self._on_install_clicked)
        right.addWidget(self.btn_install)

        # Convert to ONNX button
        self.btn_convert = QPushButton("Convert to ONNX")
        if not implemented:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("Convert — TBD")
        elif not installed:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("Convert to ONNX")
            self.btn_convert.setToolTip("Install the engine first, then convert to ONNX.")
        elif is_fsr:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("N/A (built-in)")
            self.btn_convert.setToolTip("FSR is algorithmic — no ONNX conversion needed.")
        self.btn_convert.clicked.connect(self._on_convert_clicked)
        right.addWidget(self.btn_convert)

        # Open folder (always present if install_dir exists)
        self.btn_open = QPushButton("Open folder")
        inst_dir = eng.get("install_dir") or ""
        self.btn_open.setEnabled(bool(inst_dir))
        self.btn_open.clicked.connect(self._on_open_clicked)
        right.addWidget(self.btn_open)

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
        outer.addLayout(right)

        # Spacer at the end so cards don't compact too much.
        outer.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding))

    # -- helpers ----------------------------------------------------------

    def _state_label(self, installed: bool, converted: bool, is_fsr: bool = False, is_fsr2: bool = False) -> str:
        if not installed:
            return "NOT INSTALLED"
        if is_fsr2:
            return "FSR 2.0 (GPU)"
        if is_fsr and "fsr31" in (self.engine.get("engine_id", "")):
            return "FSR 3.1 (GPU)"
        if is_fsr:
            return "FSR 1.0 (CPU)"
        if converted:
            return "INSTALLED · ONNX"
        return "INSTALLED"

    def _install_label(self, installed: bool, is_fsr: bool = False, is_fsr2: bool = False) -> str:
        if is_fsr and "fsr31" in (self.engine.get("engine_id", "")):
            return "Download FFmpeg" if not installed else "Ready"
        if is_fsr and not is_fsr2:
            return "Built-in"
        if is_fsr2:
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
        # FSR 2.0 and FSR 3.0 entries route to a different signal so the parent dialog
        # downloads the FFmpeg "full" binary instead of trying to git-clone
        # an HF repo (which has no model weights for FSR).
        if (eid.endswith("-v2") or "fsr31" in eid) and eid.startswith("upscale-fsr"):
            self.fsr2_download_requested.emit(eid)
            return
        self.install_requested.emit(eid)

    def _on_convert_clicked(self) -> None:
        self.convert_requested.emit(self.engine.get("engine_id", ""))

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
