"""Diálogo de búsqueda de modelos de HuggingFace para la barra de herramientas del Workflow.

Permite filtrar por motor de inferencia, categoría, tamaño, ordenar por popularidad
o calidad, y mostrar los resultados como tarjetas con información del modelo.
Seleccionar un modelo cierra el diálogo y devuelve el dict del modelo seleccionado.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import logging

from src.ai.hf_search import get_inference_engines, search_catalog_models
from src.ui.model_search_result import ModelSearchResult

logger = logging.getLogger(__name__)


class _SearchWorker(QThread):
    """Worker que ejecuta la búsqueda en background para no bloquear la UI."""

    done = Signal(list)
    error = Signal(str)

    def __init__(self, engine_filter: str | None = None, category_filter: str | None = None, query: str = "", parent=None):
        super().__init__(parent)
        self._engine = engine_filter
        self._category = category_filter
        self._query = query

    def run(self) -> None:
        try:
            logger.info("SearchWorker: starting search category=%s query=%s engine=%s", self._category, self._query, self._engine)
            results = search_catalog_models(
                category=self._category or "Upscale",
                query=self._query or "",
                engine_id=self._engine or None,
            )
            logger.info("SearchWorker: got %d results", len(results or []))
            self.done.emit(results or [])
        except Exception as exc:
            logger.exception("SearchWorker: error")
            self.error.emit(str(exc))


class ModelSearchDialog(QDialog):
    """Diálogo modal de búsqueda de modelos HuggingFace."""

    model_selected = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Search AI Models — HuggingFace")
        self.setModal(True)
        self.setMinimumSize(900, 600)
        self.resize(960, 640)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._selected: dict | None = None
        self._worker: _SearchWorker | None = None
        self._results: list[dict] = []
        self._build()

    def get_selected_model(self) -> dict | None:
        return self._selected

    # -- UI construction --

    def _build(self) -> None:
        self.setStyleSheet(self._global_css())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        # Title
        title = QLabel("Search HuggingFace Models")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Search and install community upscaling, denoising, and interpolation models."
        )
        subtitle.setStyleSheet("color: #a0a0b0; font-size: 11px; margin-bottom: 6px;")
        outer.addWidget(subtitle)

        # Token warning banner (shown if no HF token is set).
        has_token = False
        try:
            from src.personal import has_token as _has_token
            has_token = _has_token()
        except Exception:
            pass
        if not has_token:
            banner = QLabel(
                "\u26a0\ufe0f  Some models require a HuggingFace token for downloads.  "
                '<a href="https://huggingface.co/settings/tokens" style="color:#00d2d3;">'
                "Get free token</a> \u2014 add it in Preferences > Behavior"
            )
            banner.setStyleSheet(
                "background-color: #2c1810; color: #f0c040; border: 1px solid #5f27cd; "
                "border-radius: 6px; padding: 6px 10px; font-size: 11px;"
            )
            banner.setOpenExternalLinks(True)
            banner.setTextFormat(Qt.TextFormat.RichText)
            outer.addWidget(banner)

        # Filters row
        filters = QHBoxLayout()
        filters.setSpacing(8)

        filters.addWidget(self._make_label("Engine:"))
        self.combo_engine = QComboBox()
        self.combo_engine.setMinimumWidth(160)
        filters.addWidget(self.combo_engine)

        filters.addWidget(self._make_label("Category:"))
        self.combo_category = QComboBox()
        self.combo_category.setMinimumWidth(120)
        filters.addWidget(self.combo_category)

        filters.addWidget(self._make_label("Size:"))
        self.combo_size = QComboBox()
        self.combo_size.addItems(["All", "Small (<2GB)", "Medium (2–5GB)", "Large (>5GB)"])
        self.combo_size.setMinimumWidth(140)
        filters.addWidget(self.combo_size)

        filters.addWidget(self._make_label("Sort:"))
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Popularity ↓", "Quality ↓", "Downloads ↓", "Name A→Z"])
        self.combo_sort.setMinimumWidth(140)
        filters.addWidget(self.combo_sort)

        filters.addStretch()
        outer.addLayout(filters)

        # Search bar
        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models on HuggingFace...")
        self.search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self.search_input)

        self.btn_search = QPushButton("Search")
        self.btn_search.setObjectName("searchBtn")
        self.btn_search.clicked.connect(self._on_search)
        self.btn_search.setFixedWidth(100)
        search_row.addWidget(self.btn_search)

        outer.addLayout(search_row)

        # Status label
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        outer.addWidget(self.lbl_status)

        # Scroll area for results
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        self.scroll.setWidget(self.results_container)
        outer.addWidget(self.scroll, stretch=1)

        # Populate filter combos
        self._populate_engine_combo()
        self._populate_category_combo()

        # Trigger initial search
        self._on_search()

    def _populate_engine_combo(self) -> None:
        self.combo_engine.addItem("All engines", None)
        try:
            engines = get_inference_engines()
            for e in engines:
                self.combo_engine.addItem(e.get("name", e.get("engine_id", "")), e.get("engine_id"))
        except Exception:
            self.combo_engine.addItem("ONNX Runtime (GPU)", "onnxruntime")
            self.combo_engine.addItem("PyTorch (CUDA)", "pytorch_cuda")
            self.combo_engine.addItem("PyTorch (CPU)", "pytorch_cpu")

    def _populate_category_combo(self) -> None:
        self.combo_category.addItem("All categories", None)
        for cat in ["Upscale", "Denoise", "Interpolate", "Colorize", "FaceRestore"]:
            self.combo_category.addItem(cat, cat)

    def _on_search(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._show_loading(True)
        self.btn_search.setEnabled(False)

        engine_id = self.combo_engine.currentData()
        category = self.combo_category.currentData()
        query = self.search_input.text().strip()

        self._worker = _SearchWorker(engine_filter=engine_id, category_filter=category, query=query)
        self._worker.done.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_results(self, results: list[dict]) -> None:
        logger.info("ModelSearchDialog: received %d raw results", len(results))
        self._results = self._apply_filters(results)
        logger.info("ModelSearchDialog: %d results after filters", len(self._results))
        self._populate_results(self._results)
        self._show_loading(False)
        self.btn_search.setEnabled(True)
        count = len(self._results)
        self.lbl_status.setText(f"{count} model{'s' if count != 1 else ''} found")

    def _on_error(self, msg: str) -> None:
        self._show_loading(False)
        self.btn_search.setEnabled(True)
        self.lbl_status.setText(f"Search failed: {msg}")
        QMessageBox.warning(self, "Search Error", f"Failed to search models:\n\n{msg}")

    def _apply_filters(self, results: list[dict]) -> list[dict]:
        filtered = list(results)
        # Size filter
        size_text = self.combo_size.currentText()
        if "Small" in size_text:
            filtered = [r for r in filtered if self._parse_size_gb(r.get("size_estimate", "")) < 2]
        elif "Medium" in size_text:
            filtered = [r for r in filtered if 2 <= self._parse_size_gb(r.get("size_estimate", "")) < 5]
        elif "Large" in size_text:
            filtered = [r for r in filtered if self._parse_size_gb(r.get("size_estimate", "")) >= 5]

        # Sort
        sort_text = self.combo_sort.currentText()
        if "Popularity" in sort_text:
            filtered.sort(key=lambda x: int(x.get("popularity_score", 0)), reverse=True)
        elif "Quality" in sort_text:
            filtered.sort(key=lambda x: int(x.get("quality_score", 0)), reverse=True)
        elif "Downloads" in sort_text:
            filtered.sort(key=lambda x: int(x.get("downloads", 0)), reverse=True)
        elif "Name" in sort_text:
            filtered.sort(key=lambda x: x.get("display_name", x.get("engine_id", "")).lower())

        # Text search
        q = self.search_input.text().strip().lower()
        if q:
            filtered = [
                r for r in filtered
                if q in (r.get("display_name", "") + " " + r.get("engine_id", "") + " " + r.get("hf_repo_id", "")).lower()
            ]

        return filtered

    def _parse_size_gb(self, size_str: str) -> float:
        if not size_str:
            return 999
        s = size_str.lower().strip()
        try:
            if "gb" in s:
                return float(s.replace("gb", "").strip())
            if "mb" in s:
                return float(s.replace("mb", "").strip()) / 1024
        except ValueError:
            pass
        return 999

    def _populate_results(self, models: list[dict]) -> None:
        # Clear old results
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not models:
            lbl = QLabel("No models found matching your filters.")
            lbl.setStyleSheet("color: #5f6f7f; font-size: 13px; padding: 24px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(lbl)
            self.results_layout.addStretch()
            return

        for m in models:
            card = ModelSearchResult(m)
            card.install_requested.connect(self._on_install)
            self.results_layout.addWidget(card)

        self.results_layout.addStretch()

    def _on_install(self, engine_id: str) -> None:
        for m in self._results:
            if m.get("engine_id") == engine_id:
                self._selected = m
                self.model_selected.emit(m)
                self.accept()
                return

    def _show_loading(self, loading: bool) -> None:
        if loading:
            self.lbl_status.setText("Searching HuggingFace...")
            # Remove old results, show loading placeholder
            while self.results_layout.count():
                item = self.results_layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            lbl = QLabel("Searching HuggingFace...")
            lbl.setStyleSheet("color: #a0a0b0; font-size: 14px; padding: 48px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(lbl)
            self.results_layout.addStretch()

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a0a0b0; font-size: 11px; font-weight: bold;")
        return lbl

    def _global_css(self) -> str:
        return """
        QDialog {
            background-color: #0f0f1a;
            color: #e0e0e0;
        }
        QLineEdit {
            background-color: #16213e;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
        }
        QLineEdit:focus {
            border: 1px solid #00d2d3;
        }
        QComboBox {
            background-color: #16213e;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 11px;
        }
        QComboBox::drop-down {
            border: none;
            padding-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #16213e;
            color: #e0e0e0;
            selection-background-color: #5f27cd;
        }
        QScrollArea {
            background-color: #0f0f1a;
            border: none;
        }
        QPushButton#searchBtn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2d3, stop:1 #5f27cd);
            color: #0f0f1a;
            border: none;
            border-radius: 6px;
            padding: 8px;
            font-weight: bold;
            font-size: 12px;
        }
        QPushButton#searchBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00f0f1, stop:1 #7d3fe0);
        }
        QPushButton#searchBtn:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        """
