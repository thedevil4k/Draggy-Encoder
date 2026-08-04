"""The 'AI Engines' tab embedded inside ``DependenciesDialog``.

This widget is responsible only for rendering the engine cards and
forwarding user gestures (install / convert / open) up to the parent
dialog. The parent owns the heavy workers because the QThread lifecycle
and progress / status messaging is best kept in one place (the dialog)
so behaviour is consistent across binary and AI flows.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
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
        self._chunks: list[tuple[str, list[dict]]] = []
        self._loading_label: QLabel | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._build_next_chunk)
        self._build()

    @staticmethod
    def _make_intro() -> QLabel:
        intro = QLabel(
            "AI engines are optional building blocks. 'Add Engine' clones the "
            "weights from HuggingFace into bin/models/<category>/. Pick an "
            "installed engine directly from the workflow nodes. 'Convert to "
            "ONNX' prepares the weights for the onnxruntime backend."
        )
        intro.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        intro.setWordWrap(True)
        return intro

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._container = container
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(8)
        container_layout.setContentsMargins(8, 8, 8, 8)
        self._container_layout = container_layout

        container_layout.addWidget(self._make_intro())

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Las tarjetas se construyen por chunks con QTimer para que el
        # diálogo aparezca al instante y el hilo de la GUI no se congele
        # construyendo ~30 EngineRow de golpe (~10 ms cada una).
        self._start_progressive_build()

    def _group_catalog(self) -> tuple[dict, list]:
        """Agrupa el catálogo por categoría preservando el orden del catálogo."""
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
            # El label de error se añade en _start_progressive_build.
            self._catalog_error = str(exc)
            return {}, []
        self._catalog_error = None
        return groups, order

    def _start_progressive_build(self) -> None:
        groups, order = self._group_catalog()
        if self._catalog_error:
            error_lbl = QLabel(f"Could not load engine catalog: {self._catalog_error}")
            error_lbl.setStyleSheet("color: #e74c3c; font-size: 12px;")
            error_lbl.setWordWrap(True)
            self._container_layout.addWidget(error_lbl)
            self._container_layout.addStretch()
            return
        self._chunks = [(cat, groups[cat]) for cat in order]
        if not self._chunks:
            self._container_layout.addStretch()
            return
        self._loading_label = QLabel("Loading engines…")
        self._loading_label.setStyleSheet("color: #a0a0b0; font-size: 11px; padding: 4px 0;")
        self._container_layout.addWidget(self._loading_label)
        self._timer.start()

    def _build_next_chunk(self) -> None:
        if self._chunks:
            cat, engines = self._chunks.pop(0)
            try:
                section = _CategorySection(cat, engines, parent=self._container)
                section.install_requested.connect(self.install_requested.emit)
                section.fsr2_download_requested.connect(self.fsr2_download_requested.emit)
                section.convert_requested.connect(self.convert_requested.emit)
                section.open_folder.connect(self.open_folder.emit)
                section.remove_requested.connect(self.remove_requested.emit)
                self._container_layout.addWidget(section)
                self._section_widgets.append(section)
            except Exception as exc:
                logger.exception("Failed to build category section %s", cat)
                error_lbl = QLabel(f"Could not build {cat} section: {exc}")
                error_lbl.setStyleSheet("color: #e74c3c; font-size: 11px;")
                error_lbl.setWordWrap(True)
                self._container_layout.addWidget(error_lbl)

        if self._chunks:
            self._timer.start()
        else:
            self._timer.stop()
            if self._loading_label is not None:
                self._loading_label.deleteLater()
                self._loading_label = None
            self._container_layout.addStretch()

    def is_building(self) -> bool:
        return bool(self._chunks) or (self._timer is not None and self._timer.isActive())

    def repopulate(self) -> None:
        """Re-emit every row from the catalog without rebuilding the QScrollArea.

        Used by ``DependenciesDialog._refresh_engines_tab`` after a
        state-changing action (install / convert). Restarts the chunked,
        progressive build so the UI does not freeze while rows are
        reconstructed.
        """
        try:
            self._timer.stop()
        except Exception:
            pass
        self._chunks.clear()
        container = self._container
        layout = container.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._section_widgets.clear()

        layout.addWidget(self._make_intro())
        self._start_progressive_build()
