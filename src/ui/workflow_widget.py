import logging
import os

from PySide6.QtCore import QMimeData, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDrag, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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


# --- Reordenado por arrastre -------------------------------------------------
#: MIME propio para arrastrar un nodo intermedio dentro del flujo.
_STEP_MIME = "application/x-draggy-workflow-step"
#: Fichero de sesión donde se recuerda el orden de los nodos.
_ORDER_FILE = "workflow_order.json"


# -----------------------------------------------------------------------
# Escala nativa de un modelo: se lee el fichero, no su nombre
# -----------------------------------------------------------------------


def _detect_file_scale(model_path: str):
    """Escala NATIVA de un fichero de pesos: ``(escala, probada)``.

    La escala se lee por dentro -- grafo del ONNX o formas de los tensores
    del state_dict -- porque el nombre y el catálogo mienten:
    ``OmniSR_X2_DIV2K.onnx`` es un x8, los "x2plus" de AI-Forever son x4 y
    el catálogo declaraba 2x/3x/4x para un único .onnx que sólo hace x4.

    Ofrecer un factor que el modelo no tiene significa pagar el upscale
    entero para luego tirar el resultado hacia abajo (perf-audit.md §3.5),
    así que cuando la escala no se puede PROBAR se devuelve la pista del
    nombre marcada como no probada y la UI no la ofrece.
    """
    from src.ai.arch_specs import parse_scale_from_name
    from src.ai import host_client

    # El trabajo pesado (abrir el grafo ONNX o el state_dict, que necesita
    # torch) se pide al host de IA: se ejecuta en SU proceso. Importar
    # ``src.ai.architectures`` aquí cargaba torch dentro de la GUI y Windows
    # mataba la app (0xc0000005 en python313.dll) al listar los modelos.
    if model_path and host_client.host_available():
        try:
            data = host_client.get_host().call_op("model_scale", path=model_path)
        except host_client.HostError as exc:
            logger.warning(
                "no se pudo medir la escala de %s en el host de IA (%s); "
                "se usa la pista del nombre", model_path, exc,
            )
        else:
            proven = int(data.get("scale") or 1)
            hint = int(data.get("hint") or 1)
            if proven > 1:
                if hint > 1 and hint != proven:
                    logger.info(
                        "%s really upscales x%d (its name says x%d)",
                        os.path.basename(model_path), proven, hint,
                    )
                return proven, True
            return (hint if hint > 1 else 1), False

    # Sin host (o desactivado): solo la pista del nombre, SIN probar. Es la
    # misma degradación honesta que cuando el runtime no está instalado.
    return parse_scale_from_name(os.path.basename(model_path or "")), False


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

    Un click simple (sin arrastre) emite ``clicked()``: MainWindow lo
    usa para abrir el diálogo de selección de videos, de modo que el
    botón "Select Videos" de la barra superior sobra.
    """

    video_dropped = Signal(list)
    clicked = Signal()

    def __init__(self, icon_path, parent=None):
        super().__init__("Import", icon_path, active=True, fixed_width=150, parent=parent)
        self.setAcceptDrops(True)
        self._press_pos = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

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

    # -- click (equivale al botón "Select Videos" eliminado) ------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            # Si el puntero se movió más allá de la distancia de arrastre
            # fue un drag & drop (o un desplazamiento), no un click.
            if (event.position().toPoint() - self._press_pos).manhattanLength() < QApplication.startDragDistance():
                self.clicked.emit()
        self._press_pos = None
        super().mouseReleaseEvent(event)

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
# DragHandle — arrastra un nodo para reordenar el flujo
# -----------------------------------------------------------------------


class DragHandle(QLabel):
    """Punto de agarre (⠿) para arrastrar un nodo intermedio.

    ``WorkflowWidget`` acepta el drop y recoloca el paso en la posición
    horizontal donde se suelta. Las flechas ◀ ▶ siguen existiendo como
    alternativa accesible (y son lo que se usa en tests/teclado).
    """

    def __init__(self, name: str, parent=None):
        super().__init__("⠿", parent)
        self._name = name
        self._press_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to reorder this step")
        self.setStyleSheet("color: #5f6f7f; font-size: 14px; font-weight: bold;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            # No dejar que el click suba al nodo (activarlo/desactivarlo)
            # cuando el usuario sólo quiere agarrar el asa.
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is None:
            super().mouseMoveEvent(event)
            return
        moved = (event.position().toPoint() - self._press_pos).manhattanLength()
        if moved < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_STEP_MIME, self._name.encode("utf-8"))
        drag.setMimeData(mime)
        source = self.parentWidget()
        if source is not None:
            pixmap = source.grab()
            if not pixmap.isNull():
                drag.setPixmap(pixmap.scaled(pixmap.size() / 2))
        self._press_pos = None
        drag.exec(Qt.DropAction.MoveAction)


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
    install_requested = Signal(str)  # arg = engine_id a descargar

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
        self._installed = []
        self._available = set()
        self._last_engine_id = ""
        self._pending_engine_id = ""
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
        self.drag_handle = DragHandle(self.name)
        header.addWidget(self.drag_handle, alignment=Qt.AlignmentFlag.AlignTop)
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

    def _current_engine_id(self) -> str:
        """engine_id de una entrada INSTALADA seleccionada ("" si no hay)."""
        data = self.combo.currentData()
        if isinstance(data, dict):
            if data.get("search") or data.get("installed") is False:
                return ""
            return str(data.get("engine_id") or "")
        return str(data or "")

    def _restore_engine_selection(self, engine_id):
        """Vuelve a ``engine_id`` sin re-emitir ``engine_changed``."""
        if not engine_id:
            return
        for i in range(self.combo.count()):
            data = self.combo.itemData(i)
            if isinstance(data, dict) and data.get("engine_id") == engine_id:
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(i)
                self.combo.blockSignals(False)
                self._last_engine_id = engine_id
                return

    def _on_engine_changed(self, _idx):
        """Cambio en el desplegable: instalado, descarga o buscador HF."""
        data = self.combo.currentData()
        if isinstance(data, dict) and data.get("search"):
            self._restore_engine_selection(getattr(self, "_last_engine_id", ""))
            self.search_requested.emit(self.category)
            return
        # OJO: el id crudo, no ``_current_engine_id()`` (que devuelve ""
        # justo para las entradas no instaladas).
        raw_id = (
            str(data.get("engine_id") or "") if isinstance(data, dict) else ""
        )
        if isinstance(data, dict) and data.get("installed") is False and raw_id:
            # Modelo del catálogo aún no descargado: se pide la instalación
            # y el desplegable vuelve a lo que había elegido antes. Al
            # terminar, ``refresh_from_registry`` selecciona el recién
            # instalado (``_pending_engine_id``).
            self._pending_engine_id = raw_id
            self._restore_engine_selection(getattr(self, "_last_engine_id", ""))
            self.install_requested.emit(raw_id)
            return
        engine_id = self._current_engine_id()
        self._last_engine_id = engine_id
        self._populate_scale_dropdown(engine_id)
        self._emit_engine_changed()

    def _on_scale_changed(self, _idx):
        """Scale dropdown changed — emit engine_changed."""
        self._emit_engine_changed()

    def _emit_engine_changed(self):
        info = self.get_selected_model_info()
        self.engine_changed.emit(self.name, info.get("engine_id", ""))

    def _populate_scale_dropdown(self, engine_id):
        """Rellena el desplegable con los factores que esta engine SÍ puede hacer.

        Cada opción sale de un fichero instalado cuya escala se ha podido
        PROBAR leyendo el modelo. Un modelo no admite varias escalas: cada
        fichero hornea la suya, así que la opción significa "este fichero
        produce x4" y nada más. Con un solo factor real no hay elección (se
        muestra fijo); si no se puede probar ninguno, no se inventa ninguno
        y el pipeline mide el frame real al renderizar.
        """
        self.combo_scale.blockSignals(True)
        self.combo_scale.clear()

        # Un factor por fichero; los duplicados (p.ej. un .onnx y su .pth,
        # que son el mismo upscale) colapsan en una sola opción.
        seen_scales = {}
        for m in self._all_model_files:
            if m.get("engine_id") != engine_id or not m.get("scale_proven"):
                continue
            try:
                s = int(m.get("scale", 1))
            except (TypeError, ValueError):
                continue
            if s > 1 and s not in seen_scales:
                seen_scales[s] = m

        scales = sorted(seen_scales)

        def _data_for_scale(s):
            m = seen_scales[s]
            # ``native``: esta opción ES la salida del modelo, sin re-encaje
            # del encoder a otro factor. El pipeline mide un frame de la
            # cadena para dimensionar el encoder con la medida real.
            return {
                "engine_id": engine_id,
                "model_path": m.get("model_path"),
                "scale": s,
                "native": True,
            }

        if scales:
            for s in scales:
                self.combo_scale.addItem(f"{s}x", _data_for_scale(s))
            self.combo_scale.setCurrentIndex(0)
            self.combo_scale.setEnabled(len(scales) > 1)
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
        self.combo.setEnabled(self.combo.count() > 1)

    def _refresh_active_styles(self):
        self.setProperty("active", "true" if self.active else "false")
        # Re-polish so dynamic properties restyle immediately.
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    # -- data refresh ------------------------------------------------------

    def refresh_from_registry(self):
        """Re-query the catalog+state. Called on construct AND whenever
        the bus broadcasts a state change.

        El desplegable muestra TODO el catálogo de la categoría, no sólo lo
        instalado: un nodo "vacío" era la razón de que Real-ESR AnimeVideo
        v3 (y los modelos de face restore/denoise/colorize) no se pudieran
        ni ver ni elegir. Los no instalados se marcan con ⬇ y su descarga se
        pide al pulsarlos.
        """
        try:
            from src.ai.registry import CATALOG, list_installed as _list_installed
            self._installed = _list_installed(self.category)
            catalog = [
                entry for entry in CATALOG
                if entry.get("category") == self.category
            ]
        except Exception as exc:
            self._installed = []
            catalog = []
            self.status_label.setText(f"Registry error: {exc}")
            self.status_label.setProperty("link", False)
            return

        # Pesos ya presentes en disco aunque engines.json no los marque como
        # instalados (p.ej. GFPGAN/CodeFormer/DeOldify descargados desde el
        # buscador HF): se tratan como instalados para poder elegirlos sin
        # volver a descargar cientos de MB.
        known = {e.get("engine_id") for e in self._installed}
        for entry in catalog:
            eid = entry.get("engine_id", "")
            if eid and eid not in known and self._engine_has_weights(eid):
                self._installed.append(entry)
                known.add(eid)

        prev_engine = self._current_engine_id() or getattr(
            self, "_last_engine_id", ""
        )

        self.combo.blockSignals(True)
        self.combo.clear()

        # Scan model files for all installed engines.
        self._all_model_files = self._scan_model_files()

        # 1) Instalados primero: son los únicos que pueden renderizar ya.
        installed_by_id = {}
        for entry in self._installed:
            eid = entry.get("engine_id", "")
            if eid:
                installed_by_id.setdefault(eid, entry)
        for eid, entry in installed_by_id.items():
            self.combo.addItem(
                entry.get("display_name", eid),
                {"engine_id": eid, "installed": True},
            )

        # 2) Resto del catálogo: visible y descargable desde el propio nodo.
        available = set()
        for entry in catalog:
            eid = entry.get("engine_id", "")
            if not eid or eid in installed_by_id:
                continue
            size = entry.get("onnx_size_estimate") or entry.get("size_estimate") or ""
            label = f"⬇ {entry.get('display_name', eid)}"
            if size:
                label += f" ({size})"
            self.combo.addItem(label, {"engine_id": eid, "installed": False})
            available.add(eid)
        self._available = available

        # 3) Buscador de Hugging Face siempre a mano en el propio nodo.
        if self.combo.count():
            self.combo.insertSeparator(self.combo.count())
        self.combo.addItem("🔍 Search Hugging Face…", {"search": True})

        # Si venimos de pedir la descarga de un motor que ya está en disco,
        # seleccionarlo automáticamente.
        pending = getattr(self, "_pending_engine_id", "")
        if pending and pending in installed_by_id:
            prev_engine = pending
            self._pending_engine_id = ""

        # Restore previous selection (only installed entries count).
        idx_to_set = 0
        for i in range(self.combo.count()):
            data = self.combo.itemData(i)
            if isinstance(data, dict) and data.get("engine_id") == prev_engine:
                idx_to_set = i
                break
        self.combo.setCurrentIndex(idx_to_set)
        selected = self.combo.itemData(idx_to_set)
        self._last_engine_id = (
            str(selected.get("engine_id", ""))
            if isinstance(selected, dict) and selected.get("installed")
            else ""
        )
        self.combo.blockSignals(False)

        # Populate scale dropdown for the current engine.
        self._populate_scale_dropdown(self._current_engine_id())

        # Siempre operativo: elegir modelo no exige activar el nodo antes.
        self.combo.setEnabled(self.combo.count() > 1)
        self._render_status()

    def _engine_has_weights(self, engine_id: str) -> bool:
        """True si el directorio del motor ya contiene pesos utilizables."""
        try:
            from src.ai.registry import get_engine_dir
            root_dir = get_engine_dir(engine_id)
        except Exception:
            return False
        if not root_dir or not os.path.isdir(root_dir):
            return False
        weight_exts = {".pth", ".pt", ".onnx", ".safetensors", ".bin"}
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d != "__pycache__"
            ]
            for fname in files:
                if os.path.splitext(fname)[1].lower() in weight_exts:
                    return True
        return False

    def _scan_model_files(self):
        """Scan installed engine directories for .pth/.onnx model files.

        Returns a list of dicts with keys: label, data, engine_id,
        model_path, scale, scale_proven. La escala se lee del FICHERO
        (grafo ONNX o tensores del state_dict), no del nombre ni del
        catálogo: ``scale_proven`` dice si se pudo demostrar.
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
                    "scale": 1,
                    "scale_proven": False,
                })
                continue

            for mp in sorted(model_paths):
                fname = os.path.basename(mp)
                # Skip non-weight files (README.md, etc.)
                if not fname.lower().endswith(tuple(_WEIGHT_EXTS)):
                    continue

                # Check channel count for .pth/.pt/.safetensors files.
                # Models with != 3 input channels are incompatible with the
                # BGR pipeline and will always fail to load. Skip them from
                # the dropdown so the user can't select them.
                _skip = False
                if fname.lower().endswith((".pth", ".pt", ".safetensors")):
                    try:
                        # Los canales se leen EN EL HOST de IA: importar
                        # ``src.ai.inference`` aquí cargaba torch dentro de la
                        # GUI (crash 0xc0000005) al listar los modelos.
                        from src.ai.weight_tools import weight_input_channels

                        _ch = weight_input_channels(mp)
                        if _ch is not None and _ch != 3:
                            logger.debug(
                                "Skipping incompatible model %s (%d channels)",
                                fname, _ch,
                            )
                            _skip = True
                    except Exception:
                        pass
                if _skip:
                    continue

                # Escala real del fichero, leída por dentro (ver
                # _detect_file_scale). Nada de fiarse del nombre: el
                # dropdown de escala sólo ofrece factores probados. Sólo
                # Upscale la necesita; en el resto de categorías el factor
                # no se usa y leer cada peso retrasaría el arranque.
                if self.category == "Upscale":
                    scale, scale_proven = _detect_file_scale(mp)
                else:
                    scale, scale_proven = 1, False

                # Build a human label.
                short_name = os.path.splitext(fname)[0]
                # Clean up common prefixes for readability.
                for prefix in ["RealESRGAN_", "Real_CUGAN_", "SwinIR_",
                               "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_",
                               "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_"]:
                    if short_name.startswith(prefix):
                        short_name = short_name[len(prefix):]
                # El factor se muestra como lo que es: probado o dudoso.
                if scale > 1:
                    factor = f"({scale}x)" if scale_proven else f"(~{scale}x?)"
                else:
                    factor = ""
                label = f"{display_name} — {short_name} {factor}".rstrip()

                results.append({
                    "label": label,
                    "data": {"engine_id": eid, "model_path": mp, "scale": scale},
                    "engine_id": eid,
                    "model_path": mp,
                    "scale": scale,
                    "scale_proven": scale_proven,
                })

        return results

    def _render_status(self):
        # If output info override is set (e.g. upscale output res), use it.
        if getattr(self, "_output_info", None):
            self.status_label.setText(self._output_info)
            self.status_label.setProperty("link", False)
            self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
            return
        n_installed = len({
            e.get("engine_id") for e in getattr(self, "_installed", [])
            if e.get("engine_id")
        })
        n_available = len(getattr(self, "_available", set()))
        if n_installed == 0:
            if n_available:
                self.status_label.setText(
                    f"No engines installed · {n_available} available — pick one to download"
                )
            else:
                self.status_label.setText(
                    "No engines — click to open AI Engines Manager"
                )
            self.status_label.setProperty("link", True)
            self.status_label.setCursor(Qt.CursorShape.PointingHandCursor)
            return
        current_id = self._current_engine_id()
        current = self.combo.currentText() if current_id else ""
        if len(current) > 26:
            current = current[:25] + "…"
        human = f"{n_installed} installed · {n_available} available"
        if current:
            human += f" · active: {current}"
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
        self.combo.setEnabled(self.combo.count() > 1)

    def get_selected_engine_id(self) -> str:
        """Return the dropdown's engine_id (empty string if none)."""
        if not self.active:
            return ""
        return self._current_engine_id()

    def get_selected_model_info(self):
        """Return dict with engine_id, model_path, scale, mode for current selection."""
        engine_id = self._current_engine_id()
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
        # Sin desplegable (una sola escala real, o ninguna demostrable): se
        # usa el primer fichero de la engine, prefiriendo los que tienen la
        # escala probada, que es lo que el encoder puede esperar.
        candidates = [
            m for m in getattr(self, "_all_model_files", [])
            if m.get("engine_id") == engine_id
        ]
        proven = [m for m in candidates if m.get("scale_proven")]
        for m in (proven or candidates):
            return {
                "engine_id": engine_id,
                "model_path": m.get("model_path"),
                "scale": m.get("scale", 1),
                # Un fichero de pesos tiene una única salida; si no sabemos
                # cuál es, el pipeline la midiendo (native=True) es lo
                # único honesto: mejor un frame de prueba que un factor
                # inventado por el nombre.
                "native": True,
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
    # Emitted when an AI node is enabled/disabled. MainWindow uses this to
    # refresh derived hints such as the estimated AI bitrate immediately.
    active_changed = Signal(str, bool)
    # Emitted when the user drops video files onto the Import step.
    # Arg: list of local video paths.
    video_dropped = Signal(list)
    # Emitted when the user clicks the Import step (replaces the removed
    # "Select Videos" button; MainWindow opens the file dialog).
    import_clicked = Signal()
    # Emitted when a not-yet-installed model is chosen in a node dropdown.
    # Arg: engine_id from the catalog; MainWindow opens the install dialog.
    install_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_steps()
        self._restore_order()
        self._render_layout()
        self._connect_state_bus()

    def _setup_steps(self):
        self.step_import = ImportStep(get_asset_path("Import.png"))
        self.step_import.video_dropped.connect(self.video_dropped.emit)
        self.step_import.clicked.connect(self.import_clicked.emit)
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
            AIWorkflowStep("FaceRestore", get_asset_path("AI FaceRestore.png")),
        ]
        for step in self.middle_steps:
            step.clicked_for_manage.connect(self._on_step_manage_requested)
            step.search_requested.connect(self._on_search_requested)
            step.install_requested.connect(self.install_requested.emit)
            step.engine_changed.connect(self.engine_changed.emit)
            step.active_changed.connect(self.active_changed.emit)

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
        self._save_order()

    # -- order persistence + drag & drop ----------------------------------

    def _order_file(self) -> str:
        from src.config import get_session_dir
        return os.path.join(get_session_dir(), _ORDER_FILE)

    def _restore_order(self):
        """Restaura el orden guardado de la sesión anterior."""
        import json

        try:
            with open(self._order_file(), "r", encoding="utf-8") as fh:
                saved = json.load(fh).get("order") or []
        except (OSError, ValueError, AttributeError):
            return
        if not isinstance(saved, list) or not saved:
            return
        by_name = {step.name: step for step in self.middle_steps}
        restored = []
        for name in saved:
            step = by_name.pop(name, None)
            if step is not None:
                restored.append(step)
        # Nodos nuevos (añadidos en una versión posterior) van al final.
        restored.extend(by_name.values())
        self.middle_steps = restored

    def _save_order(self):
        """Persiste el orden actual para el próximo arranque."""
        import json

        path = self._order_file()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"order": [s.name for s in self.middle_steps]}, fh)
            os.replace(tmp, path)
        except OSError as exc:
            logger.debug("No se pudo guardar el orden del flujo: %s", exc)

    def _reorder_from_drop(self, name: str, x: float):
        """Reubica el paso ``name`` en la posición horizontal ``x``."""
        src = next(
            (i for i, s in enumerate(self.middle_steps) if s.name == name), None
        )
        if src is None:
            return
        target = 0
        for step in self.middle_steps:
            center = step.mapTo(self, step.rect().center()).x()
            if x > center:
                target += 1
        step = self.middle_steps.pop(src)
        if target > src:
            target -= 1
        target = max(0, min(target, len(self.middle_steps)))
        if target == src:
            return
        self.middle_steps.insert(target, step)
        self._render_layout()
        self._save_order()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_STEP_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_STEP_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(_STEP_MIME):
            event.ignore()
            return
        name = bytes(event.mimeData().data(_STEP_MIME)).decode("utf-8", "replace")
        x = event.position().x()
        event.acceptProposedAction()
        # El reordenado se aplica cuando el QDrag del origen ha terminado:
        # reconstruir el layout en mitad del drop reparenta el widget que
        # está siendo arrastrado.
        QTimer.singleShot(0, lambda: self._reorder_from_drop(name, x))

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
