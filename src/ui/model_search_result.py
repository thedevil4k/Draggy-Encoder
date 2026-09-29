"""Widget de tarjeta para un resultado de búsqueda de modelos.

Muestra la información de un modelo de HuggingFace (nombre, tamaño, downloads,
archivos) y botones contextuales según el estado:
  - Install (si no está instalado)
  - Convert to ONNX (si está instalado pero no convertido)
  - Open folder (si está instalado)
  - Installed badge (si ya está instalado y convertido)
"""

from __future__ import annotations

import os
import platform
import subprocess

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from src.ai.architectures import (
    ONNX_CONVERTIBLE_ARCHS as _ONNX_CONVERTIBLE_ARCHS,
)


class ModelSearchResult(QFrame):
    """Tarjeta de un modelo en los resultados de búsqueda.

    Signals:
        install_requested(str) - engine_id a instalar.
        convert_requested(str) - engine_id a convertir a ONNX.
        enable_toggled(str, bool) - engine_id + nuevo estado enabled.
        open_folder(str) - engine_id; abrir carpeta local.
    """

    install_requested = Signal(str)
    convert_requested = Signal(str)
    enable_toggled = Signal(str, bool)
    open_folder = Signal(str)

    def __init__(self, model: dict, parent=None):
        super().__init__(parent)
        self.model = model
        self.setObjectName("ModelSearchResult")
        self.setStyleSheet(self._stylesheet())
        self._build()

    def _stylesheet(self) -> str:
        return """
        #ModelSearchResult {
            background-color: #16213e;
            border: 1px solid #2c3e50;
            border-radius: 8px;
        }
        #ModelSearchResult QPushButton {
            background-color: #1f2c4c;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: bold;
        }
        #ModelSearchResult QPushButton:hover {
            background-color: #00d2d3;
            color: #0f0f1a;
        }
        #ModelSearchResult QPushButton:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #ModelSearchResult QPushButton#primaryBtn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2d3, stop:1 #5f27cd);
            color: #0f0f1a;
            border: none;
        }
        #ModelSearchResult QPushButton#primaryBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00f0f1, stop:1 #7d3fe0);
        }
        #ModelSearchResult QLabel {
            color: #e0e0e0;
        }
        #ModelSearchResult QLabel#title {
            font-size: 13px;
            font-weight: bold;
            color: #ffffff;
        }
        #ModelSearchResult QLabel#meta {
            color: #a0a0b0;
            font-size: 10px;
            font-family: Consolas, monospace;
        }
        #ModelSearchResult QLabel#url {
            color: #00d2d3;
            font-size: 10px;
        }
        #ModelSearchResult QLabel#stateBadge {
            border-radius: 8px;
            padding: 2px 10px;
            font-weight: bold;
            font-size: 11px;
        }
        """

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setSpacing(14)
        outer.setContentsMargins(14, 12, 14, 12)

        eng = self.model
        eid = eng.get("engine_id", "")
        name = eng.get("display_name", eid)
        repo_id = eng.get("hf_repo_id", "")
        size_str = eng.get("size_estimate", "")
        downloads = eng.get("downloads", 0)
        license_str = eng.get("license", "")
        quality = int(eng.get("quality_score", 0))
        popularity = int(eng.get("popularity_score", 0))
        installed = bool(eng.get("installed", False))
        enabled = bool(eng.get("enabled", False))
        onnx_converted = bool(eng.get("onnx_converted", False))
        compat = eng.get("compatibility") or {}
        needs_conversion = compat.get("needs_conversion", False)
        compat_reason = compat.get("reason", "")

        # ---- Left column: title + meta + URL ----
        left = QVBoxLayout()
        left.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        lbl_title = QLabel(name)
        lbl_title.setObjectName("title")
        title_row.addWidget(lbl_title)

        # Badge de categoría
        cat = eng.get("category", "Other")
        cat_colors = {
            "Upscale": ("#00d2d3", "#0f0f1a"),
            "Interpolate": ("#5f27cd", "#ffffff"),
            "Colorize": ("#e74c3c", "#ffffff"),
            "Denoise": ("#f39c12", "#0f0f1a"),
            "FaceRestore": ("#16a085", "#ffffff"),
        }
        cat_bg, cat_fg = cat_colors.get(cat, ("#2c3e50", "#e0e0e0"))
        badge = QLabel(cat.upper())
        badge.setStyleSheet(
            f"background-color: {cat_bg}; color: {cat_fg}; "
            f"border-radius: 8px; padding: 1px 8px; font-size: 9px; font-weight: bold;"
        )
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(badge)

        if not eng.get("is_implemented", False):
            cs = QLabel("COMING SOON")
            cs.setStyleSheet(
                "background-color: #5f27cd; color: #ffffff; "
                "border-radius: 8px; padding: 1px 8px; font-size: 9px; font-weight: bold;"
            )
            cs.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_row.addWidget(cs)

        title_row.addStretch()
        left.addLayout(title_row)

        # Meta: repo_id, tamaño, downloads, licencia
        meta_parts = [repo_id]
        if size_str:
            meta_parts.append(f"size: {size_str}")
        if downloads:
            meta_parts.append(f"downloads: {self._format_downloads(downloads)}")
        if license_str:
            meta_parts.append(f"license: {license_str}")
        meta = QLabel("  ·  ".join(meta_parts))
        meta.setObjectName("meta")
        left.addWidget(meta)

        # Compatibilidad
        if compat:
            compat_text = "Compatible" if compat.get("compatible") else "Not compatible"
            compat_style = (
                "color: #00d2d3; font-size: 10px;"
                if compat.get("compatible")
                else "color: #e74c3c; font-size: 10px;"
            )
            compat_lbl = QLabel(f"Engine: {compat_text} — {compat_reason}")
            compat_lbl.setStyleSheet(compat_style)
            left.addWidget(compat_lbl)

        # URL
        if eng.get("hf_url"):
            url_lbl = QLabel(eng["hf_url"])
            url_lbl.setObjectName("url")
            left.addWidget(url_lbl)

        # LFS warning
        if eng.get("lfs_warning"):
            tip = QLabel("Weights use Git LFS — first clone downloads heavier files.")
            tip.setStyleSheet("color: #f39c12; font-size: 10px;")
            tip.setWordWrap(True)
            left.addWidget(tip)

        outer.addLayout(left, stretch=1)

        # ---- Middle column: quality bars + state badge ----
        middle = QVBoxLayout()
        middle.setSpacing(6)

        # Quality bar
        qual_row = QHBoxLayout()
        qual_row.setSpacing(6)
        qual_row.addWidget(QLabel("Quality:"))
        qual_bar = _SimpleBar(quality, "#00d2d3")
        qual_row.addWidget(qual_bar)
        middle.addLayout(qual_row)

        # Popularity bar
        pop_row = QHBoxLayout()
        pop_row.setSpacing(6)
        pop_row.addWidget(QLabel("Popularity:"))
        pop_bar = _SimpleBar(popularity, "#f39c12")
        pop_row.addWidget(pop_bar)
        middle.addLayout(pop_row)

        # State badge
        state_text = self._state_label(installed, enabled, onnx_converted)
        state_lbl = QLabel(state_text)
        state_lbl.setObjectName("stateBadge")
        state_lbl.setStyleSheet(self._state_stylesheet(installed, enabled, onnx_converted))
        state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_lbl.setMaximumWidth(180)
        middle.addWidget(state_lbl, alignment=Qt.AlignmentFlag.AlignLeft)

        outer.addLayout(middle)

        # ---- Right column: buttons + toggle ----
        right = QVBoxLayout()
        right.setSpacing(6)

        # Primary button: Install / Re-clone / Open folder
        self.btn_install = QPushButton(self._install_label(installed))
        self.btn_install.setObjectName("primaryBtn")
        if not eng.get("is_implemented", False):
            self.btn_install.setEnabled(False)
            self.btn_install.setText("Coming soon")
        self.btn_install.clicked.connect(self._on_install_clicked)
        right.addWidget(self.btn_install)

        # Convert to ONNX button
        arch = (eng.get("arch") or "").strip().lower()
        arch_known = bool(arch)
        arch_convertible = arch in _ONNX_CONVERTIBLE_ARCHS
        self.btn_convert = QPushButton("Convert to ONNX")
        if not eng.get("is_implemented", False):
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("Convert — TBD")
        elif not installed:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setToolTip(
                "Install the engine first, then convert to ONNX. "
                "Converting .pth weights requires PyTorch; .onnx models "
                "only need ONNX Runtime."
            )
        elif onnx_converted:
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
        elif needs_conversion:
            self.btn_convert.setEnabled(True)
            self.btn_convert.setToolTip(
                "Convert .pth weights to .onnx with torch.onnx.export. "
                "Requires PyTorch — models that already ship as .onnx only "
                "need ONNX Runtime."
            )
        else:
            # Already has ONNX file
            self.btn_convert.setEnabled(False)
            self.btn_convert.setText("ONNX ready")
        self.btn_convert.clicked.connect(self._on_convert_clicked)
        right.addWidget(self.btn_convert)

        # Enabled toggle
        self.chk_enabled = QCheckBox("Enabled")
        self.chk_enabled.setChecked(enabled)
        if not eng.get("is_implemented", False):
            self.chk_enabled.setEnabled(False)
            self.chk_enabled.setText("Enabled (later)")
        self.chk_enabled.toggled.connect(self._on_toggle_changed)
        right.addWidget(self.chk_enabled)

        # Open folder
        self.btn_open = QPushButton("Open folder")
        inst_dir = eng.get("install_path") or ""
        self.btn_open.setEnabled(bool(inst_dir))
        self.btn_open.clicked.connect(self._on_open_clicked)
        right.addWidget(self.btn_open)

        # "View similar models" — abre OpenModelDB en el navegador si el
        # catalog entry declara ``similar_models_url``. Disabled si no
        # hay URL configurada.
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

        outer.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding))

    # -- helpers --

    def _format_downloads(self, n: int) -> str:
        if n >= 1000000:
            return f"{n / 1000000:.1f}M"
        if n >= 1000:
            return f"{n / 1000:.1f}K"
        return str(n)

    def _state_label(self, installed: bool, enabled: bool, converted: bool) -> str:
        if not installed:
            return "NOT INSTALLED"
        if enabled:
            return "ENABLED" + (" · ONNX" if converted else "")
        return "INSTALLED · DISABLED"

    def _state_stylesheet(self, installed: bool, enabled: bool, converted: bool) -> str:
        if not installed:
            return "background-color: #e74c3c; color: #ffffff; border-radius: 8px; padding: 2px 10px; font-weight: bold; font-size: 11px;"
        if enabled:
            return "background-color: #00d2d3; color: #0f0f1a; border-radius: 8px; padding: 2px 10px; font-weight: bold; font-size: 11px;"
        return "background-color: #2c3e50; color: #a0a0b0; border-radius: 8px; padding: 2px 10px; font-weight: bold; font-size: 11px;"

    def _install_label(self, installed: bool) -> str:
        return "Re-clone" if installed else "Install"

    def _on_install_clicked(self) -> None:
        self.install_requested.emit(self.model.get("engine_id", ""))

    def _on_convert_clicked(self) -> None:
        self.convert_requested.emit(self.model.get("engine_id", ""))

    def _on_toggle_changed(self, checked: bool) -> None:
        eng = self.model
        if checked and not eng.get("is_implemented"):
            self.chk_enabled.blockSignals(True)
            self.chk_enabled.setChecked(False)
            self.chk_enabled.blockSignals(False)
            return
        self.enable_toggled.emit(eng.get("engine_id", ""), bool(checked))

    def _on_open_clicked(self) -> None:
        path = self.model.get("install_path") or ""
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
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Open folder", f"Could not open folder:\n{exc}")

    def _on_similar_clicked(self) -> None:
        """Open the OpenModelDB page for this engine in the user's browser."""
        url = self.model.get("similar_models_url", "") or ""
        if not url:
            return
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "View similar models",
                f"Could not open the browser:\n{exc}\n\nURL: {url}",
            )


class _SimpleBar(QWidget):
    """Barra de calificación simple (0-100) dibujada manualmente."""

    def __init__(self, score: int, color_hex: str = "#00d2d3", parent=None):
        super().__init__(parent)
        self.score = max(0, min(100, int(score or 0)))
        from PySide6.QtGui import QColor, QPainter
        self._color = QColor(color_hex)
        self.setFixedHeight(8)
        self.setMinimumWidth(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        from PySide6.QtCore import Qt
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
                painter.setBrush(self._color)
                painter.drawRoundedRect(fill, 3, 3)
        finally:
            painter.end()
