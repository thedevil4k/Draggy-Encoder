import logging
import os

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.config import get_asset_path
from src.ui.model_search_dialog import ModelSearchDialog

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Mapping of workflow node name -> engine category.
# -----------------------------------------------------------------------
NODE_CATEGORY = {
    "Upscale": ("Upscale", "Upscale engines (Real-ESRGAN, CUGAN, ...)"),
    "Interp":  ("Interpolate", "Frame interpolation (RIFE, FILM, ...)"),
    "Denoise": ("Denoise", "Denoising (NAFNet, ...)"),
    "Colorize": ("Colorize",   "Colorization (DDColor, DeOldify, ...)"),
    "FaceRestore": ("FaceRestore", "Face restoration (GFPGAN, ...)"),
}


# -----------------------------------------------------------------------
# Scale parsing from model filenames.
# -----------------------------------------------------------------------


def _parse_model_scale_from_name(fname: str) -> int:
    """Extract an upscale factor (2, 3, 4, 8...) from a weight filename.

    Handles the naming conventions found in the Real-ESRGAN / Real-CUGAN /
    OmniSR ecosystem, both orders ("x4plus", "x4v3", "4x-UltraSharp") and
    with or without separators. Returns 1 when no scale can be parsed.
    """
    import re

    # ``(\d+)x`` or ``x(\d+)`` where the following character is a word
    # boundary or separator, so "x4plus"/"x4v3" and "4x-UltraSharp" match
    # but "anime6B" (no x) and "somethingdx" (x only) do not.
    patterns = [
        # x2, x4, x8... possibly followed by plus/vN or ending the name
        re.compile(r"[xX](\d+)(?=(?:plus|v\d+)?(?:$|[\s_.-]))", re.IGNORECASE),
        # 2x, 4x ... prefixed to the name
        re.compile(r"(?:^|[\s_.-])(\d+)[xX](?=(?:plus|v\d+)?(?:$|[\s_.-]))", re.IGNORECASE),
    ]
    best = 1
    for _rx in patterns:
        for _m in _rx.finditer(fname):
            try:
                _v = int(_m.group(1))
            except (TypeError, ValueError):
                continue
            # Ignore absurd numbers that look like model versions/resolutions.
            if 1 < _v <= 64:
                best = max(best, _v)
    return best


# -----------------------------------------------------------------------
# Base step
# -----------------------------------------------------------------------


class WorkflowStep(QWidget):
    """A workflow node, with click-to-toggle behaviour."""

    def __init__(self, name, icon_path, active=False, fixed_width=90, parent=None):
        super().__init__(parent)
        self.name = name
        self.active = active
        self.setFixedSize(fixed_width, 150)
        self._setup_ui(icon_path)

    def _setup_ui(self, icon_path):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(72, 72)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setScaledContents(True)
        self._load_icon(icon_path)

        self.name_label = QLabel(self.name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("color: #00d2d3; font-size: 10px; font-weight: bold;")

        layout.addWidget(self.icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet(
            "WorkflowStep { border: 2px solid #2c3e50; border-radius: 8px; background-color: #16213e; }"
        )
        self.update_state()

    def _load_icon(self, icon_path):
        if icon_path and icon_path != "":
            pixmap = QPixmap(icon_path)
            if not pixmap.isNull():
                self.icon_label.setPixmap(pixmap)

    def update_state(self):
        if self.active:
            self.setStyleSheet(
                "WorkflowStep { border: 2px solid #00d2d3; border-radius: 8px; background-color: #16213e; }"
            )
            self.name_label.setStyleSheet("color: #00d2d3; font-size: 10px; font-weight: bold;")
        else:
            self.setStyleSheet(
                "WorkflowStep { border: 2px solid #2c3e50; border-radius: 8px; background-color: #16213e; }"
            )
            self.name_label.setStyleSheet("color: #5f6f7f; font-size: 10px; font-weight: bold;")

    def mousePressEvent(self, event):
        if self.name in ("Upscale", "Interp", "Colorize"):
            self.active = not self.active
            self.update_state()
        super().mousePressEvent(event)


# -----------------------------------------------------------------------
# Import step (preserved exactly)
# -----------------------------------------------------------------------

# Extensiones de video aceptadas en Import (mismas que el diálogo de
# selección en MainWindow._select_video).
_VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".wmv", ".mov", ".webm", ".m4v")


def video_paths_from_mime(mime) -> list[str]:
    """Extrae rutas locales de video de un QMimeData de drag & drop.

    Filtra por extensión y descarta URLs remotas o no locales, devolviendo
    solo ficheros que existen en disco (en orden de arrastre).
    """
    import os

    if mime is None or not mime.hasUrls():
        return []
    paths = []
    for url in mime.urls():
        local = url.toLocalFile()
        if not local:
            continue
        if not local.lower().endswith(_VIDEO_EXTS):
            continue
        if not os.path.isfile(local):
            continue
        if local not in paths:
            paths.append(local)
    return paths


class ImportStep(WorkflowStep):
    """Paso de Import con espacio extra para mostrar info del video.

    También acepta drag & drop de videos desde el explorador del SO:
    suelta ficheros sobre la casilla para seleccionarlos (equivale a
    pulsar el botón importar). Emite ``video_dropped(list[str])``.
    """

    video_dropped = Signal(list)

    def __init__(self, icon_path, parent=None):
        super().__init__("Import", icon_path, active=True, fixed_width=150, parent=parent)
        self.setAcceptDrops(True)

    def _setup_ui(self, icon_path):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(72, 72)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setScaledContents(True)
        self._load_icon(icon_path)

        self.name_label = QLabel("Import")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("color: #00d2d3; font-size: 10px; font-weight: bold;")

        self.info_label = QLabel("No video")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #a0a0b0; font-size: 8px;")

        layout.addWidget(self.icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet(
            "ImportStep { border: 2px solid #00d2d3; border-radius: 8px; background-color: #16213e; }"
        )

    def update_state(self):
        self.setStyleSheet(
            "ImportStep { border: 2px solid #00d2d3; border-radius: 8px; background-color: #16213e; }"
        )
        self.name_label.setStyleSheet("color: #00d2d3; font-size: 10px; font-weight: bold;")

    def set_info(self, text):
        self.info_label.setText(text)

    # -- drag & drop ----------------------------------------------------

    def dragEnterEvent(self, event):
        if video_paths_from_mime(event.mimeData()):
            self.setStyleSheet(
                "ImportStep { border: 2px solid #ffd700; border-radius: 8px; background-color: #16213e; }"
            )
            self.info_label.setStyleSheet("color: #ffd700; font-size: 8px;")
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        # Acepta el arrastre mientras el cursor se mueve sobre la casilla;
        # sin esto algunos gestores de ventana/plataformas no entregan el
        # dropEvent (el QWidget base ignora el movimiento por defecto).
        if video_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.update_state()
        self.info_label.setStyleSheet("color: #a0a0b0; font-size: 8px;")
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        paths = video_paths_from_mime(event.mimeData())
        self.update_state()
        self.info_label.setStyleSheet("color: #a0a0b0; font-size: 8px;")
        if paths:
            event.acceptProposedAction()
            self.video_dropped.emit(paths)
        else:
            event.ignore()


# -----------------------------------------------------------------------
# StatusLabel — clickable QLabel variant for the empty-state hint
# -----------------------------------------------------------------------


class StatusLabel(QLabel):
    """Clickable QLabel inside ``AIWorkflowStep`` "no engines" row.

    ``clicked`` only fires when ``self.property("link")`` has been
    set to True by the parent frame. We call ``event.accept()`` on user
    clicks here so they do NOT propagate to the parent
    ``AIWorkflowStep`` (which would also re-toggle ``active``).
    """

    clicked = Signal()

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("statusLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setProperty("link", False)

    def mousePressEvent(self, event):
        if (
            self.property("link") is True
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


# -----------------------------------------------------------------------
# AIWorkflowStep (NEW) — category-aware dropdown above each AI node.
# -----------------------------------------------------------------------


class AIWorkflowStep(QFrame):
    """A workflow node that surfaces installed+enabled engines per category.

    Behaviour:
      - Click the body of the frame (outside the dropdown / status label)
        to toggle ``self.active``. Click on the ``QComboBox`` is handled
        by Qt and never toggles.
      - Dropdown shows "[Auto-pick best]" + every
        ``list_enabled(self.category)`` engine by display name.
      - Status label below the dropdown summarises installed/enabled
        counts; if zero installed yet, it becomes a clickable label that
        emits ``manage_requested`` so the user lands in AI Engines Manager.
        ``Compressed to view_v1.json`` resolves to enabled engines.
      - All UI strings are clamped so the fixed-width frame stays stable.
        The frame is wider than the legacy ``WorkflowStep`` (200 vs 90)
        because it carries the picker; reorder buttons in
        ``WorkflowWidget`` automatically pick up the new width.

    Signals:
        clicked_for_manage(str)  - emitted when the empty-state label is
                                    pressed; arg is the category name.
        engine_changed(str, str)  - ``(node_name, engine_id_or_empty)``.
        active_changed(str, bool) - ``(node_name, active)``.
    """

    clicked_for_manage = Signal(str)
    engine_changed = Signal(str, str)
    active_changed = Signal(str, bool)
    search_requested = Signal(str)  # arg = categoría del nodo

    def __init__(self, name, icon_path, parent=None,
                 fixed_width=200, fixed_height=150):
        super().__init__(parent)
        self.name = name
        if name in NODE_CATEGORY:
            self.category, self._hint = NODE_CATEGORY[name]
        else:
            # Loud at dev time so a future developer adding a workflow
            # node doesn't silently end up with "No engines — click to
            # open AI Engines Manager" forever because the catalog
            # category name and the workflow node name disagree.
            logger.error(
                "AIWorkflowStep(%r) not in NODE_CATEGORY; falling back to "
                "(node_name, hint). Add an entry in workflow_widget.py "
                "NODE_CATEGORY so list_enabled(category) matches.",
                name,
            )
            self.category, self._hint = (name, f"{name} engines")
        self.active = False
        self._all_model_files = []
        self.setObjectName("AIWorkflowStep")
        self.setFixedSize(fixed_width, fixed_height)
        self._build(icon_path)
        self.setStyleSheet(self._stylesheet())
        # La lupa (búsqueda en Hub) está disponible en TODOS los pasos
        # AI: misma affordance que Upscale para que la UI sea uniforme.
        self.btn_search.setVisible(True)
        self.refresh_from_registry()  # first paint

    # -- styling ----------------------------------------------------------

    def _stylesheet(self):
        return """
        #AIWorkflowStep {
            background-color: #16213e;
            border: 2px solid #2c3e50;
            border-radius: 8px;
        }
        #AIWorkflowStep[active="true"] {
            border: 2px solid #00d2d3;
        }
        #AIWorkflowStep QComboBox {
            background-color: #1f2c4c;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 4px;
            padding: 3px;
            font-size: 10px;
        }
        #AIWorkflowStep QComboBox:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #AIWorkflowStep QComboBox QAbstractItemView {
            background-color: #16213e;
            color: #e0e0e0;
            selection-background-color: #00d2d3;
        }
        #AIWorkflowStep QLabel {
            color: #e0e0e0;
        }
        #AIWorkflowStep QLabel#statusLabel {
            color: #a0a0b0;
            font-size: 9px;
        }
        #AIWorkflowStep QLabel#statusLabel[link="true"] {
            color: #00d2d3;
            font-size: 9px;
            font-style: italic;
        }
        #AIWorkflowStep QLabel#nameLabel {
            color: #00d2d3;
            font-size: 11px;
            font-weight: bold;
        }
        #AIWorkflowStep[active="true"] QLabel#nameLabel {
            color: #00d2d3;
        }
        #AIWorkflowStep QLabel#nameLabel[inactive="true"] {
            color: #5f6f7f;
        }
        #AIWorkflowStep QPushButton#searchBtn {
            background-color: #5f27cd;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            padding: 0px;
        }
        #AIWorkflowStep QPushButton#searchBtn:hover {
            background-color: #7d3fe0;
        }
        """

    # -- build ------------------------------------------------------------

    def _build(self, icon_path):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        header = QHBoxLayout()
        header.setSpacing(5)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(36, 36)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setScaledContents(True)
        self._load_icon(icon_path)
        header.addWidget(self.icon_label)

        name_col = QVBoxLayout()
        name_col.setSpacing(0)
        self.name_label = QLabel(self.name)
        self.name_label.setObjectName("nameLabel")
        name_col.addWidget(self.name_label)

        self.sub_label = QLabel(self._hint)
        self.sub_label.setStyleSheet("color: #5f6f7f; font-size: 8px;")
        self.sub_label.setWordWrap(True)
        name_col.addWidget(self.sub_label)

        header.addLayout(name_col, stretch=1)
        layout.addLayout(header)

        # Engine dropdown + search button.
        combo_row = QHBoxLayout()
        combo_row.setSpacing(4)
        self.combo = QComboBox()
        self.combo.setEnabled(False)
        self.combo.currentIndexChanged.connect(self._on_engine_changed)
        self.combo.setMinimumHeight(24)
        combo_row.addWidget(self.combo, stretch=1)

        self.btn_search = QPushButton("🔍")
        self.btn_search.setFixedSize(24, 24)
        self.btn_search.setToolTip("Search HuggingFace models...")
        self.btn_search.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_search.setStyleSheet(
            "QPushButton { background-color: #5f27cd; color: #ffffff; border: none; "
            "border-radius: 4px; font-size: 12px; padding: 0px; }"
            "QPushButton:hover { background-color: #7d3fe0; }"
        )
        self.btn_search.clicked.connect(
            lambda: self.search_requested.emit(self.category)
        )
        combo_row.addWidget(self.btn_search)
        layout.addLayout(combo_row)

        # Scale dropdown (only visible for Upscale).
        self.combo_scale = QComboBox()
        self.combo_scale.setEnabled(False)
        self.combo_scale.currentIndexChanged.connect(self._on_scale_changed)
        self.combo_scale.setMinimumHeight(22)
        self.combo_scale.setVisible(False)
        layout.addWidget(self.combo_scale)

        # Internal data: list of model file dicts for the current engine.
        self._engine_models = []

        # Status label.
        self.status_label = StatusLabel("Loading…")
        self.status_label.clicked.connect(
            lambda: self.clicked_for_manage.emit(self.category)
        )
        layout.addWidget(self.status_label)

    def _load_icon(self, icon_path):
        if icon_path and icon_path != "":
            pixmap = QPixmap(icon_path)
            if not pixmap.isNull():
                self.icon_label.setPixmap(pixmap)

    # -- event handlers --------------------------------------------------

    def _on_engine_changed(self, _idx):
        """Engine dropdown changed — update scale dropdown."""
        engine_id = self.combo.currentData() or ""
        self._populate_scale_dropdown(engine_id)
        self._emit_engine_changed()

    def _on_scale_changed(self, _idx):
        """Scale dropdown changed — emit engine_changed."""
        self._emit_engine_changed()

    def _emit_engine_changed(self):
        info = self.get_selected_model_info()
        self.engine_changed.emit(self.name, info.get("engine_id", ""))

    def _populate_scale_dropdown(self, engine_id):
        """Populate the scale dropdown with available scale factors for this engine."""
        self.combo_scale.blockSignals(True)
        self.combo_scale.clear()

        # FSR 3.1 scale is encoded in the engine_id (fsr31-2x, fsr31-3x, fsr31-4x)
        if engine_id and engine_id.startswith("upscale-fsr"):
            import re
            from src.ai.registry import get_entry
            m = re.search(r'fsr31-(\d+)x', engine_id)
            scale = int(m.group(1)) if m else 2

            # FSR 3.1 engines have a "mode" field in the catalog entry
            # (upscale, interpolate, both). Pass it through so the pipeline
            # knows whether to interpolate frames.
            mode = "upscale"  # default
            if "fsr31" in engine_id:
                entry = get_entry(engine_id) or {}
                mode = entry.get("mode", "upscale")

            self.combo_scale.addItem(f"{scale}x", {
                "engine_id": engine_id,
                "model_path": None,
                "scale": scale,
                "mode": mode,
            })
            self.combo_scale.setCurrentIndex(0)
            self.combo_scale.setEnabled(False)
            self.combo_scale.setVisible(True)
            self.combo_scale.blockSignals(False)
            return

        # Find all model files for this engine_id.
        models = [m for m in self._all_model_files if m["engine_id"] == engine_id]
        # Deduplicate scales, keep first occurrence for each scale.
        seen_scales = {}
        for m in models:
            s = m.get("scale", 1)
            if s not in seen_scales:
                seen_scales[s] = m

        # Scales que declara el catálogo (p.ej. AnimeVideo v3 soporta
        # 2x/3x/4x como en video2x). Si el nombre del archivo no las
        # revela todas, completamos el dropdown con ellas: la escala se
        # comunica luego a la engine vía _target_scale y el encoder
        # re-encaja el frame al factor elegido.
        try:
            from src.ai.registry import get_supported_scales
            catalog_scales = get_supported_scales(engine_id) or []
        except Exception:
            catalog_scales = []

        def _data_for_scale(s):
            # Prefiere el modelo real con esa escala; si no existe
            # (archivo sin sufijo de escala), usa cualquier modelo de
            # la engine — la engine upscala a su nativa y el encoder
            # dimensiona al factor elegido.
            m = seen_scales.get(s)
            if m is not None:
                return dict(m)
            return {
                "engine_id": engine_id,
                "model_path": None,
                "scale": s,
            }

        scales = sorted({s for s in seen_scales if s > 1} | set(catalog_scales))

        if len(scales) > 1:
            # "Model native": la resolución de salida es EXACTAMENTE la que
            # produce el modelo (su escala nativa), sin re-encaje del
            # encoder a un factor elegido. Lleva ``scale`` = nativa del
            # catálogo (para que la UI muestre la resolución real) y el
            # flag ``native`` que el pipeline usa para medir el frame real.
            native_scale = max(scales)
            native_data = _data_for_scale(native_scale)
            native_data["native"] = True

            self.combo_scale.clear()
            self.combo_scale.addItem("Model native", native_data)
            for s in scales:
                self.combo_scale.addItem(f"{s}x", _data_for_scale(s))
            self.combo_scale.setCurrentIndex(0)
            self.combo_scale.setEnabled(True)
            self.combo_scale.setVisible(True)
        elif len(scales) == 1:
            s = scales[0]
            self.combo_scale.addItem(f"{s}x", _data_for_scale(s))
            self.combo_scale.setCurrentIndex(0)
            self.combo_scale.setEnabled(False)
            self.combo_scale.setVisible(True)
        else:
            self.combo_scale.setVisible(False)

        self.combo_scale.blockSignals(False)

    # NOTE: _on_status_clicked removed; ``StatusLabel.clicked`` is
    # connected directly in ``_build``.

    # -- mouse press toggles active ----------------------------------------

    def mousePressEvent(self, event):
        # ``QComboBox`` swallows its own clicks. ``StatusLabel`` (when
        # activated by parent) calls ``event.accept()`` so its clicks
        # never propagate here. Anything that lands on this method is
        # therefore on the body of the frame → toggle active.
        self._toggle_active()

    def _toggle_active(self):
        self.active = not self.active
        self._refresh_active_styles()
        self.active_changed.emit(self.name, bool(self.active))
        self.combo.setEnabled(self.active and bool(self._installed))

    def _refresh_active_styles(self):
        self.setProperty("active", "true" if self.active else "false")
        # Re-polish so dynamic properties restyle immediately.
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    # -- data refresh ------------------------------------------------------

    def refresh_from_registry(self):
        """Re-query the catalog+state. Called on construct AND whenever
        the bus broadcasts a state change."""
        try:
            from src.ai.registry import list_installed as _list_installed
            self._installed = _list_installed(self.category)
        except Exception as exc:
            self._installed = []
            self.status_label.setText(f"Registry error: {exc}")
            self.status_label.setProperty("link", False)
            return

        prev_engine = self.combo.currentData()

        self.combo.blockSignals(True)
        self.combo.clear()

        # Scan model files for all installed engines.
        self._all_model_files = self._scan_model_files()

        if self._installed:
            # Show unique engine entries (not per-file).
            seen = set()
            for entry in self._installed:
                eid = entry["engine_id"]
                if eid in seen:
                    continue
                seen.add(eid)
                label = entry.get("display_name", eid)
                self.combo.addItem(label, eid)
        else:
            self.combo.addItem("No engines installed", None)

        # Restore previous selection.
        idx_to_set = 0
        if prev_engine:
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == prev_engine:
                    idx_to_set = i
                    break
        self.combo.setCurrentIndex(idx_to_set)
        self.combo.blockSignals(False)

        # Populate scale dropdown for the current engine.
        self._populate_scale_dropdown(self.combo.currentData() or "")

        self.combo.setEnabled(self.active and bool(self._installed))
        self._render_status()

    def _scan_model_files(self):
        """Scan installed engine directories for .pth/.onnx model files.

        Returns a list of dicts with keys: label, data, engine_id, model_path, scale.
        For upscale engines, extracts scale factor from filename (x2, x4, x8).
        """
        import re
        from src.ai.registry import get_engine_dir

        results = []
        _WEIGHT_EXTS = {".pth", ".pt", ".onnx", ".safetensors", ".bin"}

        for entry in self._installed:
            eid = entry["engine_id"]
            engine_dir = get_engine_dir(eid)
            if not os.path.isdir(engine_dir):
                continue

            display_name = entry.get("display_name", eid)

            # Find all model weight files (skip .git, .cache, .huggingface).
            model_paths = []
            for root, dirs, files in os.walk(engine_dir):
                # Prune hidden/cache dirs.
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__",)]
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in _WEIGHT_EXTS:
                        model_paths.append(os.path.join(root, f))

            if not model_paths:
                # No model files found — still show the engine entry.
                results.append({
                    "label": display_name,
                    "data": {"engine_id": eid, "model_path": None, "scale": 1},
                    "engine_id": eid,
                })
                continue

            for mp in sorted(model_paths):
                fname = os.path.basename(mp)
                # Extract scale from filename: x2, x3, x4, x8, 4x, 2x, etc.
                # Cubre nombres reales del ecosistema: "x4plus", "x4v3",
                # "realesr-animevideov3-x4", "4x-UltraSharp", "Real_CUGAN_x2".
                scale = _parse_model_scale_from_name(fname)
                if scale < 2:
                    # Ninguna pista en el nombre (p.ej. "model.safetensors"):
                    # cae a la escala declarada en el catálogo si es una
                    # engine upscale conocida y tiene escala > 1.
                    try:
                        from src.ai.registry import get_entry, get_scale_factor as _cat_scale
                        _entry = get_entry(eid) or {}
                        if _entry.get("category") == "Upscale":
                            _cs = int(_cat_scale(eid) or 1)
                            scale = _cs if _cs > 1 else 1
                    except Exception:
                        pass

                # Build a human label.
                short_name = os.path.splitext(fname)[0]
                # Clean up common prefixes for readability.
                for prefix in ["RealESRGAN_", "Real_CUGAN_", "SwinIR_",
                               "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_",
                               "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_"]:
                    if short_name.startswith(prefix):
                        short_name = short_name[len(prefix):]
                label = f"{display_name} — {short_name} ({scale}x)"

                results.append({
                    "label": label,
                    "data": {"engine_id": eid, "model_path": mp, "scale": scale},
                    "engine_id": eid,
                    "model_path": mp,
                    "scale": scale,
                })

        return results

    def _render_status(self):
        # If output info override is set (e.g. upscale output res), use it.
        if getattr(self, "_output_info", None):
            self.status_label.setText(self._output_info)
            self.status_label.setProperty("link", False)
            self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
            return
        n_total = len(self._installed)
        if n_total == 0:
            self.status_label.setText("No engines — click to open AI Engines Manager")
            self.status_label.setProperty("link", True)
            self.status_label.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            current = self.combo.currentText() if self.combo.count() else ""
            current_id = self.combo.currentData()
            if not current_id or not current:
                human = f"{n_total} installed"
            else:
                human = f"{n_total} installed · active: {current}"
            self.status_label.setText(human)
            self.status_label.setProperty("link", False)
            self.status_label.setCursor(Qt.CursorShape.ArrowCursor)

    # -- Mutation convenience methods --------------------------------------

    def set_output_info(self, text):
        """Override the status label text with extra info (e.g. output res)."""
        self._output_info = text
        self._render_status()

    def set_active(self, value: bool):
        """External drive (e.g. pre-flight restore from session)."""
        if bool(value) != self.active:
            self.active = bool(value)
            self._refresh_active_styles()
            self.active_changed.emit(self.name, self.active)
        self.combo.setEnabled(self.active and bool(self._installed))

    def get_selected_engine_id(self) -> str:
        """Return the dropdown's engine_id (empty string if none)."""
        if not self.active:
            return ""
        data = self.combo.currentData()
        if data is None:
            return ""
        return str(data)

    def get_selected_model_info(self):
        """Return dict with engine_id, model_path, scale, mode for current selection."""
        engine_id = self.combo.currentData() or ""
        if isinstance(engine_id, dict):
            engine_id = engine_id.get("engine_id", "")
        if not self.combo_scale.isHidden() and self.combo_scale.count() > 0:
            scale_data = self.combo_scale.currentData()
            if scale_data and isinstance(scale_data, dict):
                return {
                    "engine_id": str(engine_id),
                    "model_path": scale_data.get("model_path"),
                    "scale": scale_data.get("scale", 1),
                    "mode": scale_data.get("mode", "upscale"),
                    "native": bool(scale_data.get("native", False)),
                }
        for m in getattr(self, "_all_model_files", []):
            if m.get("engine_id") == engine_id:
                return {
                    "engine_id": engine_id,
                    "model_path": m.get("model_path"),
                    "scale": m.get("scale", 1),
                }
        return {"engine_id": engine_id, "model_path": None, "scale": 1}


# -----------------------------------------------------------------------
# WorkflowWidget (preserved + new signals)
# -----------------------------------------------------------------------


class WorkflowWidget(QWidget):
    """Hosts the workflow chain: Import → middle AI steps → Compress.

    Connects to ``src.ai.bus.bus`` once and re-paints the AI steps
    whenever any engine's installed/enabled flag changes anywhere
    in the app (AI Engines Manager, banner-worker, followup chain).
    """

    # Emitted when the user clicks an "open AI Engines Manager" link
    # on an empty workflow node. Arg: category name.
    manage_engines_requested = Signal(str)
    # Emitted when user selects a model from the HuggingFace search dialog.
    # Arg: full model dict.
    model_selected = Signal(dict)
    # Emitted when any AI step's engine dropdown changes. Arg: (node_name, engine_id).
    engine_changed = Signal(str, str)
    # Emitted when the user drops video files onto the Import step.
    # Arg: list of local video paths.
    video_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_steps()
        self._render_layout()
        self._connect_state_bus()

    def _setup_steps(self):
        self.step_import = ImportStep(get_asset_path("Import.png"))
        self.step_import.video_dropped.connect(self.video_dropped.emit)
        self.step_compress = WorkflowStep(
            "Compress", get_asset_path("Export.png"), active=True
        )
# AIWorkflowStep subclasses WorkflowStep's interface so the
        # rest of WorkflowWidget's logic (reorder, mouse toggle) keeps
        # working unchanged.
        self.middle_steps = [
            AIWorkflowStep("Upscale",  get_asset_path("AI Upscaling.png")),
            AIWorkflowStep("Interp",   get_asset_path("AI Interpolation.png")),
            AIWorkflowStep("Denoise",  get_asset_path("AI Denoise.png")),
            AIWorkflowStep("Colorize", get_asset_path("AI Colorize.png")),
            AIWorkflowStep("FaceRestore", get_asset_path("AI Face Restore.png")),
        ]
        for step in self.middle_steps:
            step.clicked_for_manage.connect(self._on_step_manage_requested)
            step.search_requested.connect(self._on_search_requested)
            step.engine_changed.connect(self.engine_changed.emit)

    def _connect_state_bus(self):
        """Wire workflow nodes to the AI state bus.

        Connections use a bound method reference
        (``self._on_state_bus_event``) so Qt can correctly deduplicate
        via ``UniqueConnection`` and ``self.destroyed`` triggers
        ``_disconnect_state_bus`` to unbind on widget destruction. The
        bus is a process-lifetime QObject so the bus itself survives;
        we only unbind our subscription.
        """
        try:
            from src.ai.bus import bus
        except Exception:
            return
        self._state_bus_handler = self._on_state_bus_event
        try:
            bus.state_changed.connect(
                self._state_bus_handler,
                Qt.ConnectionType.UniqueConnection,
            )
        except Exception:
            pass
        try:
            self.destroyed.connect(
                self._disconnect_state_bus,
                Qt.ConnectionType.UniqueConnection,
            )
        except Exception:
            pass

    def _disconnect_state_bus(self, *_):
        """Unbind our handler when the widget is destroyed.

        Idempotent + null-safe. ``UniqueConnection`` on the
        ``destroyed.connect`` call prevents duplicates if this widget
        is rebuilt in test harnesses.
        """
        handler = getattr(self, "_state_bus_handler", None)
        if handler is None:
            return
        try:
            from src.ai.bus import bus
            bus.state_changed.disconnect(handler)
        except Exception:
            pass

    def _on_state_bus_event(self, _engine_id):
        # We don't care WHICH engine changed — refresh ours wholesale.
        try:
            self.refresh_steps()
        except Exception:
            pass

    def _on_step_manage_requested(self, category):
        self.manage_engines_requested.emit(category)

    def _on_search_requested(self, category):
        """Open the HuggingFace model search dialog for the given node.

        ``category`` es la categoría del nodo (Upscale, Interpolate,
        Denoise, Colorize, FaceRestore): el diálogo pre-selecciona esa
        categoría en ambas pestañas para que el usuario encuentre
        modelos relevantes sin tener que filtrar a mano.
        """
        dlg = ModelSearchDialog(initial_category=category, parent=self)
        dlg.model_selected.connect(self.model_selected.emit)
        dlg.exec()

    def _render_layout(self):
        if self.layout() is not None:
            layout = self.layout()
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
        else:
            layout = QHBoxLayout(self)
            layout.setSpacing(8)
            layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(self.step_import)
        layout.addWidget(self._make_arrow())

        for i, step in enumerate(self.middle_steps):
            layout.addWidget(step)
            layout.addWidget(self._make_reorder_controls(i))
            if i < len(self.middle_steps):
                layout.addWidget(self._make_arrow())

        layout.addWidget(self.step_compress)
        layout.addStretch()
        self.setLayout(layout)

    def _make_arrow(self):
        arrow = QLabel("→")
        arrow.setStyleSheet("color: #00d2d3; font-size: 20px;")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return arrow

    def _make_reorder_controls(self, index):
        """Crea botones ▶ ◀ para mover un paso intermedio."""
        widget = QWidget()
        widget.setFixedWidth(40)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        btn_left = QPushButton("◀")
        btn_left.setFixedSize(18, 18)
        btn_left.setStyleSheet(
            "QPushButton { background-color: #16213e; color: #00d2d3; border: 1px solid #2c3e50; "
            "border-radius: 4px; font-size: 8px; padding: 0px; }"
            "QPushButton:hover { background-color: #00d2d3; color: #0f0f1a; }"
        )
        btn_left.clicked.connect(lambda _checked, idx=index: self._move_step(idx, -1))

        btn_right = QPushButton("▶")
        btn_right.setFixedSize(18, 18)
        btn_right.setStyleSheet(
            "QPushButton { background-color: #16213e; color: #00d2d3; border: 1px solid #2c3e50; "
            "border-radius: 4px; font-size: 8px; padding: 0px; }"
            "QPushButton:hover { background-color: #00d2d3; color: #0f0f1a; }"
        )
        btn_right.clicked.connect(lambda _checked, idx=index: self._move_step(idx, 1))

        layout.addWidget(btn_left)
        layout.addWidget(btn_right)
        return widget

    def _move_step(self, index, direction):
        new_index = index + direction
        if new_index < 0 or new_index >= len(self.middle_steps):
            return
        self.middle_steps[index], self.middle_steps[new_index] = (
            self.middle_steps[new_index], self.middle_steps[index],
        )
        self._render_layout()

    # -- public hooks -----------------------------------------------------

    def set_import_info(self, info_text):
        self.step_import.set_info(info_text)

    def set_upscale_output_info(self, text):
        """Set output resolution text on the Upscale node's status label."""
        for step in self.middle_steps:
            if isinstance(step, AIWorkflowStep) and step.name == "Upscale":
                step.set_output_info(text)
                break

    def refresh_steps(self):
        """Re-query the registry+state for every AI step.

        Public so MainWindow / DependenciesDialog can call this when
        a state change is detected via other channels (e.g. closing
        the dependencies dialog after a successful install).
        """
        for step in self.middle_steps:
            if isinstance(step, AIWorkflowStep):
                step.refresh_from_registry()

    def get_pipeline(self):
        """Legacy: list of step names that are currently active (in order)."""
        pipeline = ["Import"]
        for step in self.middle_steps:
            if step.active:
                pipeline.append(step.name)
        pipeline.append("Compress")
        return pipeline

    def get_active_steps(self):
        return {step.name: step.active for step in self.middle_steps}

    def get_selected_engines(self) -> dict:
        """Return ``{"Upscale": {"engine_id": ..., "model_path": ..., "scale": ...}, ...}``
        for every currently active AI node. Includes the specific model file
        and scale factor chosen in the dropdown so the compression worker
        uses exactly what the user selected."""
        out: dict = {}
        for step in self.middle_steps:
            if isinstance(step, AIWorkflowStep) and step.active:
                info = step.get_selected_model_info()
                eid = info.get("engine_id", "")
                if eid or step.active:
                    out[step.category] = info
        return out
