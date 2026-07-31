"""The 'AI Engines' tab embedded inside ``DependenciesDialog``.

This widget is responsible only for rendering the engine cards and
forwarding user gestures (install / convert / open) up to the parent
dialog. The parent owns the heavy workers because the QThread lifecycle
and progress / status messaging is best kept in one place (the dialog)
so behaviour is consistent across binary and AI flows.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ai.registry import iter_catalog
from src.ui.engine_row import EngineRow

logger = logging.getLogger(__name__)


class _CategorySection(QWidget):
    """A small section header + a vertical stack of engine rows."""

    install_requested = Signal(str)
    fsr2_download_requested = Signal(str)
    convert_requested = Signal(str)
    open_folder = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, category: str, engines: list[dict], parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(6)

        header = QLabel(category.upper())
        header.setStyleSheet(
            "color: #00d2d3; font-size: 12px; font-weight: bold; "
            "letter-spacing: 1px; padding: 0 0 4px 4px;"
        )
        layout.addWidget(header)

        for engine in engines:
            try:
                row = EngineRow(engine, parent=self)
                row.install_requested.connect(self.install_requested.emit)
                row.fsr2_download_requested.connect(self.fsr2_download_requested.emit)
                row.convert_requested.connect(self.convert_requested.emit)
                row.open_folder.connect(self.open_folder.emit)
                row.remove_requested.connect(self.remove_requested.emit)
                layout.addWidget(row)
            except Exception as exc:
                # A single broken engine row must not break the whole panel.
                logger.exception("Failed to render engine row for %s", engine.get("engine_id"))
                error_lbl = QLabel(
                    f"Could not render {engine.get('engine_id', 'unknown')}: {exc}"
                )
                error_lbl.setStyleSheet("color: #e74c3c; font-size: 11px;")
                error_lbl.setWordWrap(True)
                layout.addWidget(error_lbl)


class EnginesTab(QWidget):
    """A scrollable, vertically-stacked grid of ``EngineRow`` cards.

    Signals maxed out — they are simply forwarded from
    ``_CategorySection`` so the host dialog can connect exactly once.
    """

    install_requested = Signal(str)            # engine_id
    fsr2_download_requested = Signal(str)      # engine_id (FSR 2.0)
    convert_requested = Signal(str)            # engine_id
    open_folder = Signal(str)                  # engine_id
    remove_requested = Signal(str)             # engine_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._section_widgets = []
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._container = container
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(8)
        container_layout.setContentsMargins(8, 8, 8, 8)

        intro = QLabel(
            "AI engines are optional building blocks. 'Add Engine' clones the "
            "weights from HuggingFace into bin/models/<category>/. Pick an "
            "installed engine directly from the workflow nodes. 'Convert to "
            "ONNX' prepares the weights for the onnxruntime backend."
        )
        intro.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        intro.setWordWrap(True)
        container_layout.addWidget(intro)

        # Group engines by category preserving catalog order.
        groups: dict[str, list[dict]] = {}
        order: list[str] = []

        try:
            for engine in iter_catalog():
                cat = engine.get("category") or "Other"
                if cat not in groups:
                    groups[cat] = []
                    order.append(cat)
                groups[cat].append(engine)
        except Exception as exc:
            logger.exception("Failed to iterate AI engine catalog")
            error_lbl = QLabel(f"Could not load engine catalog: {exc}")
            error_lbl.setStyleSheet("color: #e74c3c; font-size: 12px;")
            error_lbl.setWordWrap(True)
            container_layout.addWidget(error_lbl)

        for cat in order:
            try:
                section = _CategorySection(cat, groups[cat], parent=container)
                section.install_requested.connect(self.install_requested.emit)
                section.fsr2_download_requested.connect(self.fsr2_download_requested.emit)
                section.convert_requested.connect(self.convert_requested.emit)
                section.open_folder.connect(self.open_folder.emit)
                section.remove_requested.connect(self.remove_requested.emit)
                container_layout.addWidget(section)
                self._section_widgets.append(section)
            except Exception as exc:
                logger.exception("Failed to build category section %s", cat)
                error_lbl = QLabel(f"Could not build {cat} section: {exc}")
                error_lbl.setStyleSheet("color: #e74c3c; font-size: 11px;")
                error_lbl.setWordWrap(True)
                container_layout.addWidget(error_lbl)

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def repopulate(self) -> None:
        """Re-emit every row from the catalog without rebuilding the QScrollArea.

        Used by ``DependenciesDialog._refresh_engines_tab`` after a
        state-changing action (install / convert). Keeps the user's
        scroll position intact instead of resetting to top and preserves
        the existing QTabWidget slot.
        """
        container = self._container
        layout = container.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._section_widgets.clear()

        intro = QLabel(
            "AI engines are optional building blocks. 'Add Engine' clones the "
            "weights from HuggingFace into bin/models/<category>/. Pick an "
            "installed engine directly from the workflow nodes. 'Convert to "
            "ONNX' prepares the weights for the onnxruntime backend."
        )
        intro.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        groups: dict = {}
        order: list = []

        try:
            for engine in iter_catalog():
                cat = engine.get("category") or "Other"
                if cat not in groups:
                    groups[cat] = []
                    order.append(cat)
                groups[cat].append(engine)
        except Exception as exc:
            logger.exception("Failed to iterate AI engine catalog during repopulate")
            error_lbl = QLabel(f"Could not load engine catalog: {exc}")
            error_lbl.setStyleSheet("color: #e74c3c; font-size: 12px;")
            error_lbl.setWordWrap(True)
            layout.addWidget(error_lbl)

        for cat in order:
            try:
                section = _CategorySection(cat, groups[cat], parent=container)
                section.install_requested.connect(self.install_requested.emit)
                section.fsr2_download_requested.connect(self.fsr2_download_requested.emit)
                section.convert_requested.connect(self.convert_requested.emit)
                section.open_folder.connect(self.open_folder.emit)
                section.remove_requested.connect(self.remove_requested.emit)
                layout.addWidget(section)
                self._section_widgets.append(section)
            except Exception as exc:
                logger.exception("Failed to build category section %s during repopulate", cat)
                error_lbl = QLabel(f"Could not build {cat} section: {exc}")
                error_lbl.setStyleSheet("color: #e74c3c; font-size: 11px;")
                error_lbl.setWordWrap(True)
                layout.addWidget(error_lbl)

        layout.addStretch()
