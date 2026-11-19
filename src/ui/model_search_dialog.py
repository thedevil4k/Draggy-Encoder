"""Diálogo de búsqueda de modelos con dos pestañas: Catálogo y Hub.

Catálogo: filtra y busca los modelos del catálogo estático (curated) en
HuggingFace — mismo comportamiento que la versión original del diálogo.

Hub: búsqueda libre del Hugging Face Hub vía ``search_hub_models``
(búsqueda de texto, filtro por motor/categoría y paginación). Los
resultados del Hub se devuelven como ``is_user`` y el caller los registra
en ``src.ai.user_models`` al seleccionarlos.

Ambas pestañas muestran tarjetas ``ModelSearchResult``; seleccionar un
modelo cierra el diálogo y emite ``model_selected(dict)``.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess

from PySide6.QtCore import Qt, QThread, Signal
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.ai.hardware_fit import GpuProfile, current_profile, recommend_models
from src.ai.hf_search import get_inference_engines, search_catalog_models, search_hub_models
from src.ui.model_search_result import ModelSearchResult

logger = logging.getLogger(__name__)


class _CatalogWorker(QThread):
    """Búsqueda de catálogo en background (no bloquea la UI)."""

    done = Signal(list)
    error = Signal(str)

    def __init__(self, engine_filter=None, category_filter=None, query="", parent=None):
        super().__init__(parent)
        self._engine = engine_filter
        self._category = category_filter
        self._query = query
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            results = search_catalog_models(
                category=self._category or "Upscale",
                query=self._query or "",
                engine_id=self._engine or None,
            )
            if self._cancel:
                return
            self.done.emit(results or [])
        except Exception as exc:
            logger.exception("CatalogWorker: error")
            self.error.emit(str(exc))


class _HubWorker(QThread):
    """Búsqueda libre del Hub en background, con página para paginar."""

    done = Signal(list)
    error = Signal(str)

    def __init__(self, engine_filter=None, category_filter=None, query="",
                 page=0, limit=30, parent=None):
        super().__init__(parent)
        self._engine = engine_filter
        self._category = category_filter
        self._query = query
        self._page = page
        self._limit = limit
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            results = search_hub_models(
                category=self._category,  # None = cualquier categoría
                query=self._query or "",
                engine_id=self._engine or None,
                page=self._page,
                limit=self._limit,
            )
            if self._cancel:
                return
            self.done.emit(results or [])
        except Exception as exc:
            logger.exception("HubWorker: error")
            self.error.emit(str(exc))


class _SearchPageBase(QWidget):
    """Base común de las pestañas: scroll de resultados + tarjetas."""

    model_selected = Signal(dict)

    def __init__(self, parent=None, hardware_profile: GpuProfile | None = None):
        super().__init__(parent)
        self._results: list[dict] = []
        self._hardware_profile = hardware_profile or current_profile()
        self._build_results_area()
        self.hw_banner = QLabel()
        self.hw_banner.setWordWrap(True)
        self.hw_banner.setStyleSheet(
            "background-color: #12263a; color: #cfe9ff; "
            "border: 1px solid #1f4b6e; border-radius: 6px; "
            "padding: 6px 10px; font-size: 10px;"
        )
        self._update_hw_banner(None)

    # -- results area ----------------------------------------------------

    def _build_results_area(self) -> None:
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

    def _current_category(self) -> str | None:
        combo = getattr(self, "combo_category", None)
        if combo is None:
            return None
        try:
            return combo.currentData()
        except Exception:
            return None

    def _update_hw_banner(self, recommendation: dict | None) -> None:
        """Muestra GPU + recomendación para la categoría activa."""
        banner = getattr(self, "hw_banner", None)
        if banner is None:
            return
        profile = self._hardware_profile
        if not profile.has_gpu:
            banner.setText(
                "GPU: none detected · running on CPU. "
                "Recommendations prioritize the lightest models."
            )
            return
        text = f"GPU: {profile.label()}"
        if recommendation is None:
            banner.setText(text)
            return
        rec = recommendation or {}
        if rec.get("engine_id"):
            cat = self._current_category() or "this function"
            text += f"  ·  Recommended for {cat}: {rec.get('display_name') or rec['engine_id']}"
            reason = rec.get("reason") or ""
            estimate = rec.get("vram_estimate_mb")
            if reason:
                text += f" — {reason}"
            if estimate:
                suffix = (
                    "GB measured"
                    if rec.get("vram_source") == "measured"
                    else "GB VRAM est."
                )
                text += f" · ~{float(estimate) / 1024.0:.1f} {suffix}"
        else:
            text += "  ·  No curated model fits the detected VRAM; pick a smaller one."
        banner.setText(text)

    def _set_results(self, models: list[dict]) -> None:
        category = self._current_category()
        recommendation = recommend_models(
            models, category=category, profile=self._hardware_profile
        )
        self._results = list(recommendation.get("models") or models)
        recommended_id = recommendation.get("engine_id") or ""
        for model in self._results:
            model["recommended"] = bool(recommended_id) and (
                model.get("engine_id") == recommended_id
            )
        self._update_hw_banner(recommendation)
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

        for m in self._results:
            card = ModelSearchResult(m, recommended=bool(m.get("recommended")))
            card.install_requested.connect(self._on_install)
            card.convert_requested.connect(self._on_install)
            card.enable_toggled.connect(self._on_enable_toggled)
            card.open_folder.connect(self._on_open_folder)
            self.results_layout.addWidget(card)
        self.results_layout.addStretch()

    def _show_loading(self, message: str = "Searching...") -> None:
        self.lbl_status.setText(message)
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        lbl = QLabel(message)
        lbl.setStyleSheet("color: #a0a0b0; font-size: 14px; padding: 48px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(lbl)
        self.results_layout.addStretch()

    @staticmethod
    def _select_category(combo, category):
        """Pre-selecciona una categoría en el combo si existe (sin efecto si no)."""
        if not category:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == category:
                combo.setCurrentIndex(i)
                return

    # -- card signal handlers -------------------------------------------

    def _on_install(self, engine_id: str) -> None:
        for m in self._results:
            if m.get("engine_id") == engine_id:
                self.model_selected.emit(m)
                return

    def _on_enable_toggled(self, engine_id: str, checked: bool) -> None:
        try:
            from src.ai.engines_state import patch
            patch(engine_id, enabled=bool(checked))
        except Exception as exc:
            logger.warning("Failed to persist enable toggle for %s: %s", engine_id, exc)

    def _on_open_folder(self, engine_id: str) -> None:
        try:
            from src.ai.registry import get_engine_dir
            path = get_engine_dir(engine_id)
        except Exception:
            path = None
        if not path or not os.path.isdir(path):
            return
        try:
            if platform.system() == "Windows":
                os.startfile(path)  # noqa: S606
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            logger.warning("Failed to open folder for %s: %s", engine_id, exc)

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a0a0b0; font-size: 11px; font-weight: bold;")
        return lbl


class _CatalogSearchPage(_SearchPageBase):
    """Pestaña Catálogo: modelos curated con filtros engine/categoría/tamaño/orden."""

    def __init__(self, parent=None, initial_category=None, hardware_profile=None):
        super().__init__(parent, hardware_profile=hardware_profile)
        self._initial_category = initial_category
        self._worker: _CatalogWorker | None = None
        self._build_page()
        # Trigger initial search (comportamiento original del diálogo).
        self._on_search()

    def cancel_worker(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)

    def _build_page(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

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
        layout.addLayout(filters)
        layout.addWidget(self.hw_banner)

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
        layout.addLayout(search_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        layout.addWidget(self.scroll, stretch=1)

        self._populate_engine_combo()
        self._populate_category_combo()
        self._select_category(self.combo_category, self._initial_category)

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
        self._show_loading("Searching HuggingFace...")
        self.btn_search.setEnabled(False)

        self._worker = _CatalogWorker(
            engine_filter=self.combo_engine.currentData(),
            category_filter=self.combo_category.currentData(),
            query=self.search_input.text().strip(),
            parent=self,
        )
        self._worker.done.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_results(self, results: list[dict]) -> None:
        self._results = self._apply_filters(results)
        self._set_results(self._results)
        self.btn_search.setEnabled(True)
        count = len(self._results)
        self.lbl_status.setText(f"{count} model{'s' if count != 1 else ''} found")

    def _on_error(self, msg: str) -> None:
        self.btn_search.setEnabled(True)
        self.lbl_status.setText(f"Search failed: {msg}")
        QMessageBox.warning(self, "Search Error", f"Failed to search models:\n\n{msg}")

    def _apply_filters(self, results: list[dict]) -> list[dict]:
        filtered = list(results)
        size_text = self.combo_size.currentText()
        if "Small" in size_text:
            filtered = [r for r in filtered if self._parse_size_gb(r.get("size_estimate", "")) < 2]
        elif "Medium" in size_text:
            filtered = [r for r in filtered if 2 <= self._parse_size_gb(r.get("size_estimate", "")) < 5]
        elif "Large" in size_text:
            filtered = [r for r in filtered if self._parse_size_gb(r.get("size_estimate", "")) >= 5]

        sort_text = self.combo_sort.currentText()
        if "Popularity" in sort_text:
            filtered.sort(key=lambda x: int(x.get("popularity_score", 0)), reverse=True)
        elif "Quality" in sort_text:
            filtered.sort(key=lambda x: int(x.get("quality_score", 0)), reverse=True)
        elif "Downloads" in sort_text:
            filtered.sort(key=lambda x: int(x.get("downloads", 0)), reverse=True)
        elif "Name" in sort_text:
            filtered.sort(key=lambda x: x.get("display_name", x.get("engine_id", "")).lower())

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


class _HubSearchPage(_SearchPageBase):
    """Pestaña Hub: búsqueda libre del Hugging Face Hub con paginación."""

    PAGE_SIZE = 30

    def __init__(self, parent=None, initial_category=None, hardware_profile=None):
        super().__init__(parent, hardware_profile=hardware_profile)
        self._initial_category = initial_category
        self._worker: _HubWorker | None = None
        self._page = 0
        self._build_page()
        # Vista inicial: modelos más descargados de la categoría por defecto.
        self._on_search()

    def cancel_worker(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)

    def _build_page(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        filters = QHBoxLayout()
        filters.setSpacing(8)

        filters.addWidget(self._make_label("Engine:"))
        self.combo_engine = QComboBox()
        self.combo_engine.setMinimumWidth(160)
        filters.addWidget(self.combo_engine)

        filters.addWidget(self._make_label("Category:"))
        self.combo_category = QComboBox()
        self.combo_category.setMinimumWidth(140)
        filters.addWidget(self.combo_category)

        filters.addWidget(self._make_label("Per page:"))
        self.combo_page_size = QComboBox()
        for n in (30, 60, 100):
            self.combo_page_size.addItem(str(n), n)
        self.combo_page_size.setCurrentIndex(0)
        self.combo_page_size.setMinimumWidth(90)
        filters.addWidget(self.combo_page_size)

        filters.addStretch()
        layout.addLayout(filters)
        layout.addWidget(self.hw_banner)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search any model on the Hub (e.g. 4x-UltraSharp, Real-ESRGAN...)"
        )
        self.search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self.search_input)
        self.btn_search = QPushButton("Search")
        self.btn_search.setObjectName("searchBtn")
        self.btn_search.clicked.connect(self._on_search)
        self.btn_search.setFixedWidth(100)
        search_row.addWidget(self.btn_search)
        layout.addLayout(search_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        layout.addWidget(self.scroll, stretch=1)

        # Pagination row
        nav = QHBoxLayout()
        nav.setSpacing(8)
        self.btn_prev = QPushButton("← Prev")
        self.btn_prev.clicked.connect(self._on_prev)
        nav.addWidget(self.btn_prev)
        self.lbl_page = QLabel("Page 1")
        self.lbl_page.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        nav.addWidget(self.lbl_page)
        self.btn_next = QPushButton("Next →")
        self.btn_next.clicked.connect(self._on_next)
        nav.addWidget(self.btn_next)
        nav.addStretch()
        layout.addLayout(nav)

        self._populate_engine_combo()
        self._populate_category_combo()
        self._select_category(self.combo_category, self._initial_category)

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
        try:
            from src.ai.registry import categories
            for cat in categories():
                self.combo_category.addItem(cat, cat)
        except Exception:
            for cat in ["Upscale", "Denoise", "Interpolate", "Colorize", "FaceRestore"]:
                self.combo_category.addItem(cat, cat)

    def _on_search(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._page = 0
        self._run_search()

    def _on_prev(self) -> None:
        if self._page <= 0 or (self._worker and self._worker.isRunning()):
            return
        self._page -= 1
        self._run_search()

    def _on_next(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._page += 1
        self._run_search()

    def _run_search(self) -> None:
        self._show_loading("Searching HuggingFace Hub...")
        self.btn_search.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

        self._worker = _HubWorker(
            engine_filter=self.combo_engine.currentData(),
            category_filter=self.combo_category.currentData(),
            query=self.search_input.text().strip(),
            page=self._page,
            limit=int(self.combo_page_size.currentData() or self.PAGE_SIZE),
            parent=self,
        )
        self._worker.done.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_results(self, results: list[dict]) -> None:
        self._set_results(results)
        self.btn_search.setEnabled(True)
        self.btn_prev.setEnabled(self._page > 0)
        self.btn_next.setEnabled(True)
        self.lbl_page.setText(f"Page {self._page + 1}")
        count = len(results)
        self.lbl_status.setText(
            f"{count} model{'s' if count != 1 else ''} found on the Hub "
            f"(page {self._page + 1})"
        )

    def _on_error(self, msg: str) -> None:
        self.btn_search.setEnabled(True)
        self.btn_prev.setEnabled(self._page > 0)
        self.btn_next.setEnabled(True)
        self.lbl_status.setText(f"Search failed: {msg}")
        QMessageBox.warning(self, "Hub Search Error", f"Failed to search the Hub:\n\n{msg}")


class ModelSearchDialog(QDialog):
    """Diálogo modal de búsqueda de modelos: pestañas Catálogo y Hub.

    ``initial_category`` (opcional) pre-selecciona la categoría en ambas
    pestañas — lo usa el workflow para abrir la lupa de un nodo concreto
    (Upscale, Interpolate, Denoise, Colorize, FaceRestore) ya filtrado.
    """

    model_selected = Signal(dict)

    def __init__(self, parent: QWidget | None = None, initial_category: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Search AI Models")
        self.setModal(True)
        self.setMinimumSize(900, 600)
        self.resize(980, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._selected: dict | None = None
        self._initial_category = initial_category
        self._hardware_profile = current_profile()
        self._build()

    def get_selected_model(self) -> dict | None:
        return self._selected

    # -- UI construction --

    def _build(self) -> None:
        self.setStyleSheet(self._global_css())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        title = QLabel("Search AI Models")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Browse the curated catalog or search the whole HuggingFace Hub for "
            "upscaling, denoising, and interpolation models."
        )
        subtitle.setStyleSheet("color: #a0a0b0; font-size: 11px; margin-bottom: 6px;")
        subtitle.setWordWrap(True)
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
                "\u26a0\ufe0f  No Hugging Face API token is configured. Public "
                "models still work; gated/private models need one.  "
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

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab { background: #16213e; color: #a0a0b0; padding: 8px 18px;"
            " border: 1px solid #2c3e50; border-bottom: none;"
            " border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; }"
            "QTabBar::tab:selected { background: #5f27cd; color: #ffffff; }"
            "QTabBar::tab:hover:!selected { background: #1f2c4c; }"
            "QTabWidget::pane { border: 1px solid #2c3e50; border-radius: 6px;"
            " background-color: #0f0f1a; }"
        )
        self.tabs.addTab(self._make_catalog_page(), "Catálogo")
        self.tabs.addTab(self._make_hub_page(), "HuggingFace Hub")
        outer.addWidget(self.tabs, stretch=1)

    def _make_catalog_page(self) -> QWidget:
        page = _CatalogSearchPage(
            self,
            initial_category=self._initial_category,
            hardware_profile=self._hardware_profile,
        )
        page.model_selected.connect(self._on_page_model_selected)
        self._catalog_page = page
        return page

    def _make_hub_page(self) -> QWidget:
        page = _HubSearchPage(
            self,
            initial_category=self._initial_category,
            hardware_profile=self._hardware_profile,
        )
        page.model_selected.connect(self._on_page_model_selected)
        self._hub_page = page
        return page

    def _on_page_model_selected(self, model: dict) -> None:
        self._selected = model
        self.model_selected.emit(model)
        self.accept()

    def done(self, result: int) -> None:
        """Cualquier cierre del diálogo debe cancelar y esperar ambos
        workers (QThread vivo destruido por Qt crashea)."""
        for page in (getattr(self, "_catalog_page", None), getattr(self, "_hub_page", None)):
            if page is not None:
                page.cancel_worker()
        super().done(result)

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
        QPushButton {
            background-color: #1f2c4c;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: bold;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #00d2d3;
            color: #0f0f1a;
        }
        QPushButton:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
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
