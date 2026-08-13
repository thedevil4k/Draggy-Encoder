"""Diálogo modal de Preferencias con pestañas **Behavior**, **History**,
**Diagnostics** y **About**.

- **Behavior**: contiene el toggle ``auto_update_all_on_close`` (con
  alias legacy ``auto_update_on_close``) que controla si la cadena de
  auto-update se dispara al cerrar la app.

- **History**: muestra los últimos ``HISTORY_LIMIT`` updates persistidos
  en ``bin/.session/update_history.json``. Cada fila expone timestamp,
  dependencia, version antes/después, duración y delta de bytes.

- **Diagnostics**: muestra la ruta de ``bin/logs``, botones para abrir la
  carpeta de logs y la de respaldos, y el nº/tamaño de los respaldos de
  crash report con opción de borrarlos.

- **About**: versión, arquitectura, SO/Python y enlaces al repositorio y al
  bugtracker para reportar problemas con contexto.

Construido como ``QTabWidget`` para admitir futuras secciones (paths
extra, modelos por defecto) sin tocar el layout principal.

Los valores de Behavior se persisten via ``src.preferences`` (JSON en
``bin/.session/preferences.json``). El contenido de History es de sólo
lectura: para limpiarlo hoy hay que borrar el JSON a mano o esperar al
botón ``Reset history`` que ya esta cableado al módulo.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import src.preferences as preferences
import src.update_history as update_history
from src.config import (
    APP_NAME,
    APP_VERSION,
    ISSUES_URL,
    REPO_URL,
    ensure_runtime_dirs,
    get_logs_dir,
    get_session_dir,
)
from src.log_backup import backup_dir, backup_stats, clear_backups
from src.personal import get_token as _get_hf_token, set_token as _set_hf_token


_AUTO_UPDATE_KEY_NEW = "auto_update_all_on_close"
_AUTO_UPDATE_KEY_LEGACY = "auto_update_on_close"
# Claves persistidas en bin/.session/preferences.json
_AUTO_OPEN_LAST_OUTPUT_KEY = "auto_open_last_output"
_HEALTH_CHECK_PROMPT_KEY = "ffmpeg_health_check_prompt"
_REMEMBER_CODEC_KEY = "remember_last_codec_device"
_EXEC_PROVIDER_KEY = "execution_provider"
_INSECURE_CHECKPOINT_KEY = "allow_insecure_checkpoint_load"


# ---------------------------------------------------------------------------
# Helpers de formato (compartidos por History tab y por main_window)
# ---------------------------------------------------------------------------

def _format_bytes(n):
    """Devuelve un tamaño legible en binario (KiB / MiB / GiB)."""
    try:
        n = float(n)
    except Exception:
        return "—"
    if n < 0:
        return "—"
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PiB"


def _format_duration(seconds):
    """Devuelve una duración legible (``8 s``, ``1m 30s``)."""
    try:
        seconds = float(seconds)
    except Exception:
        return "—"
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    if seconds < 60:
        return f"{int(seconds)} s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        rest = int(seconds % 60)
        return f"{minutes}m {rest}s"
    hours = int(seconds // 3600)
    rest_minutes = int((seconds % 3600) // 60)
    return f"{hours}h {rest_minutes}m"


def _format_version_delta(old, new):
    """Compacta old/new en una sola columna de la tabla History."""
    old_str = old if old else "—"
    new_str = new if new else "—"
    return f"{old_str} → {new_str}"


def _format_bytes_delta(old, new):
    """Cadena con el delta de bytes tras la actualización."""
    if old is None and new is None:
        return "—"
    if old is None:
        return f"first install · {_format_bytes(new)}"
    if new is None:
        return f"{_format_bytes(old)} → ?"
    return f"{_format_bytes(old)} → {_format_bytes(new)}"


def _format_timestamp(iso_ts):
    """Formatea timestamp ISO a 'YYYY-MM-DD HH:MM:SS'."""
    if not iso_ts:
        return "—"
    try:
        # Acepta 'YYYY-MM-DDTHH:MM:SS+00:00' y variantes sin tz.
        ts = iso_ts.replace("T", " ")
        if "+" in ts:
            ts = ts.split("+", 1)[0]
        elif ts.endswith("Z"):
            ts = ts[:-1]
        # Truncar microsegundos (variantas tipo 'YYYY-MM-DDTHH:MM:SS.fff').
        return ts.split(".", 1)[0]
    except Exception:
        return iso_ts


# ---------------------------------------------------------------------------
# Dialog principal
# ---------------------------------------------------------------------------

class PreferencesDialog(QDialog):
    """Modal con dos pestañas: Behavior y History.

    El parametro ``initial_tab`` admite ``"behavior"`` (default) o
    ``"history"`` para que el banner-success pueda saltar directo al
    historial sin obligar al usuario a hacer clic extra.
    """

    def __init__(self, parent=None, initial_tab="behavior"):
        super().__init__(parent)
        self.setWindowTitle("Preferences — DRAGGY ENCODER")
        self.setMinimumSize(640, 420)
        self.resize(720, 460)
        self.setStyleSheet(
            "QDialog { background-color: #0f0f1a; color: #e0e0e0; }"
            " QLabel { color: #e0e0e0; }"
            " QTabWidget::pane {"
            "   border: 1px solid #2c3e50; border-radius: 6px;"
            " }"
            " QTabBar::tab {"
            "   background-color: #16213e; color: #e0e0e0;"
            "   padding: 8px 16px; border: 1px solid #2c3e50;"
            "   border-bottom: none; border-top-left-radius: 6px;"
            "   border-top-right-radius: 6px;"
            " }"
            " QTabBar::tab:selected {"
            "   background-color: #0f0f1a; color: #00d2d3;"
            " }"
            " QTableWidget {"
            "   background-color: #16213e; color: #e0e0e0;"
            "   gridline-color: #2c3e50; border-radius: 6px;"
            " }"
            " QHeaderView::section {"
            "   background-color: #1f2c4c; color: #00d2d3;"
            "   padding: 6px; border: none; font-weight: bold;"
            " }"
            " QTableWidget::item { padding: 4px 8px; }"
            " QTableWidget::item:selected {"
            "   background-color: #00d2d3; color: #0f0f1a;"
            " }"
        )
        self._initial_tab = initial_tab if initial_tab in ("behavior", "history") else "behavior"
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(12, 12, 12, 12)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_behavior_tab(), "Behavior")
        self.tabs.addTab(self._build_history_tab(), "History")
        self.tabs.addTab(self._build_diagnostics_tab(), "Diagnostics")
        self.tabs.addTab(self._build_about_tab(), "About")
        if self._initial_tab == "history":
            self.tabs.setCurrentIndex(1)
        else:
            self.tabs.setCurrentIndex(0)
        outer.addWidget(self.tabs, stretch=1)

        # ------------------ Botones ------------------
        bottom = QHBoxLayout()
        bottom.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._on_accept)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_ok)
        outer.addLayout(bottom)

    # -------------------- Behavior tab --------------------

    def _build_behavior_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 16, 18, 16)

        title = QLabel("Behavior")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00d2d3;")
        layout.addWidget(title)

        # --------- Section: Auto-updates ---------
        auto_group = self._build_section(
            "Auto-updates",
            "Cuando esté activo, si al cerrar el programa hay actualizaciones "
            "pendientes en el registro de dependencias, DRAGGY ENCODER pedirá "
            "confirmación. Sólo entran en la cadena las dependencias marcadas "
            "como 'updatable' con instalador automático (hoy FFmpeg; cuando "
            "se integren runtimes/modelos se sumarán automáticamente). Por "
            "cada paso se anota un icono INFO/WARN en bin/logs/notifications.log "
            "y al terminar sale una notificación nativa con el resumen.",
        )
        auto_group_layout = auto_group.layout()
        self.chk_auto_update = QCheckBox(
            "Automatically download updates when closing the app"
        )
        auto_group_layout.addWidget(self.chk_auto_update)
        layout.addWidget(auto_group)

        # --------- Section: Output folder ---------
        output_group = self._build_section(
            "Output folder",
            "Al iniciar la app, recuerda la última carpeta donde guardaste "
            "un vídeo comprimido y la deja seleccionada en el botón 'Output' "
            "para que no tengas que volver a elegirla cada sesión.",
        )
        output_group_layout = output_group.layout()
        self.chk_auto_open_output = QCheckBox(
            "Auto-open the last output folder on startup"
        )
        output_group_layout.addWidget(self.chk_auto_open_output)
        layout.addWidget(output_group)

        # --------- Section: Compression preferences ---------
        comp_group = self._build_section(
            "Compression preferences",
            "Comportamientos del motor de compresión antes de ejecutar un "
            "trabajo. La confirmación de salud aplica solo cuando el binario "
            "FFmpeg existe pero no responde a 'ffmpeg -version' (binario "
            "corrupto, truncado, mal extraído o sin permisos); en ese caso "
            "el modal pregunta antes de gastar tiempo en una codificación "
            "que va a fallar.",
        )
        comp_group_layout = comp_group.layout()
        self.chk_health_check = QCheckBox(
            "Ask before compressing if FFmpeg binary doesn't respond to -version"
        )
        comp_group_layout.addWidget(self.chk_health_check)
        self.chk_remember_codec = QCheckBox(
            "Remember last codec/device used"
        )
        comp_group_layout.addWidget(self.chk_remember_codec)
        layout.addWidget(comp_group)

        # --------- Section: AI Execution Provider ---------
        provider_group = self._build_section(
            "AI Execution Provider",
            "Choose the hardware backend for AI models. 'Auto' (default) tries "
            "the GPU (CUDA) first and automatically falls back to CPU if the "
            "drivers or cuDNN are missing. 'CPU' forces CPU inference. 'CUDA' "
            "forces GPU and fails if it cannot be used.",
        )
        provider_group_layout = provider_group.layout()
        self.cmb_exec_provider = QComboBox()
        self.cmb_exec_provider.addItems(["Auto", "CPU", "CUDA"])
        self.cmb_exec_provider.setMinimumWidth(180)
        provider_group_layout.addWidget(self.cmb_exec_provider)
        layout.addWidget(provider_group)

        # --------- Section: Checkpoint security ---------
        security_group = self._build_section(
            "Checkpoint security",
            "Los checkpoints de terceros (.pth/.pt) se cargan por defecto en "
            "modo seguro (weights_only=True), que no ejecuta código embebido "
            "en el archivo. Algunos modelos antiguos guardan objetos extra "
            "(metadata, optimizadores) que solo se pueden leer con pickle "
            "completo, lo cual permite ejecutar código arbitrario. Activa esta "
            "opción SOLO si confías plenamente en el origen del modelo y "
            "aceptas ese riesgo; el programa registrará cada carga insegura "
            "en el log.",
        )
        security_group_layout = security_group.layout()
        self.chk_insecure_checkpoint = QCheckBox(
            "Allow insecure checkpoint loading (weights_only=False)"
        )
        security_group_layout.addWidget(self.chk_insecure_checkpoint)
        layout.addWidget(security_group)

        # --------- Section: HuggingFace API Token ---------
        hf_group = self._build_section(
            "HuggingFace API Token (required for some models)",
            "Some HuggingFace models require a free token for LFS file downloads. "
            "Without it, repos like Real-CUGAN will fail with 403 Forbidden.\n\n"
            "Steps:\n"
            "  1. Create a free account at https://huggingface.co\n"
            "  2. Go to https://huggingface.co/settings/tokens\n"
            "  3. Create a token with 'Read' access\n"
            "  4. Paste it below",
        )
        hf_group_layout = hf_group.layout()
        token_row = QHBoxLayout()
        self.txt_hf_token = QLineEdit()
        self.txt_hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_hf_token.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        self.txt_hf_token.setMinimumWidth(300)
        token_row.addWidget(self.txt_hf_token, stretch=1)
        self.btn_toggle_token = QPushButton("Mostrar/Ocultar")
        self.btn_toggle_token.setFixedWidth(120)
        self.btn_toggle_token.setCheckable(True)
        self.btn_toggle_token.setToolTip(
            "Muestra u oculta el token de HuggingFace mientras lo escribes/pegas."
        )
        self.btn_toggle_token.toggled.connect(self._on_toggle_token)
        token_row.addWidget(self.btn_toggle_token)
        hf_group_layout.addLayout(token_row)
        layout.addWidget(hf_group)

        layout.addStretch()
        return page

    def _on_toggle_token(self, checked):
        # La etiqueta es fija ("Mostrar/Ocultar"): describe la acción, el
        # estado real lo da el echo mode del campo + el pressed del botón.
        if checked:
            self.txt_hf_token.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.txt_hf_token.setEchoMode(QLineEdit.EchoMode.Password)

    def _build_section(self, title_text, body_text):
        """Construye un ``QGroupBox`` con título y descripción de sección.

        Centralizado para que los tres groupboxes del Behavior tab tengan
        exactamente el mismo look (título cian + label gris pequeño + body
        wordWrap + spacing consistente) y que cualquiera que añada una
        sección nueva lo consiga con una sola línea.
        """
        group = QGroupBox(title_text)
        group.setStyleSheet(
            "QGroupBox {"
            "  color: #00d2d3; font-weight: bold;"
            "  border: 1px solid #2c3e50; border-radius: 6px;"
            "  margin-top: 14px; padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin; left: 10px; padding: 0 6px;"
            "  background-color: #0f0f1a;"
            "}"
            "QGroupBox QCheckBox {"
            "  color: #e0e0e0; font-size: 12px; padding: 4px 0;"
            "}"
        )
        v = QVBoxLayout(group)
        v.setSpacing(6)
        v.setContentsMargins(12, 8, 12, 10)
        body = QLabel(body_text)
        body.setStyleSheet("color: #a0a0b0; font-size: 10px;")
        body.setWordWrap(True)
        v.addWidget(body)
        return group

    # -------------------- History tab --------------------

    def _build_history_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)
        layout.setContentsMargins(18, 16, 18, 16)

        header = QHBoxLayout()
        title = QLabel("Update History")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00d2d3;")
        header.addWidget(title)

        header.addStretch()
        self.btn_reset_history = QPushButton("Clear history")
        self.btn_reset_history.setStyleSheet(
            "background-color: #2c3e50; color: #e0e0e0;"
            "padding: 6px 12px; border-radius: 6px;"
        )
        self.btn_reset_history.clicked.connect(self._on_clear_history)
        header.addWidget(self.btn_reset_history)

        layout.addLayout(header)

        sub = QLabel(
            f"Últimos {update_history.HISTORY_LIMIT} updates "
            "(más recientes primero). Persistido en "
            "bin/.session/update_history.json."
        )
        sub.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            ["When", "Dependency", "Old → New", "Duration", "Bytes"]
        )
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.verticalHeader().setVisible(False)
        header_view = self.history_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.history_table, stretch=1)

        self._populate_history()
        return page

    def _populate_history(self):
        entries = update_history.get_recent()
        self.history_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            self._set_history_row(row, entry)
        if not entries:
            self.history_table.setRowCount(1)
            placeholder = QTableWidgetItem(
                "Aún no hay updates registrados. Activa 'Auto-update on close' "
                "y la próxima vez que cierres con deps pendientes aparecerán aquí."
            )
            placeholder.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.history_table.setItem(0, 0, placeholder)
            self.history_table.setSpan(0, 0, 1, 5)

    def _set_history_row(self, row, entry):
        ts = _format_timestamp(entry.get("timestamp"))
        dep = entry.get("display_name") or entry.get("dep_id") or "?"
        old_v = entry.get("old_version")
        new_v = entry.get("new_version")
        version_delta = _format_version_delta(old_v, new_v)
        duration = _format_duration(entry.get("duration_seconds"))
        bytes_delta = _format_bytes_delta(
            entry.get("old_bytes"), entry.get("new_bytes")
        )
        success = bool(entry.get("success"))

        cells = [ts, dep, version_delta, duration, bytes_delta]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if not success:
                # Marca fila fallida en grisaceo para que el usuario la
                # distinga visualmente.
                item.setForeground(Qt.GlobalColor.gray)
            self.history_table.setItem(row, col, item)
        # Tintar fondo de filas fallidas para reforzar el feedback visual.
        if not success:
            for col in range(5):
                cell = self.history_table.item(row, col)
                if cell is not None:
                    cell.setBackground(Qt.GlobalColor.darkRed)

    def _on_clear_history(self):
        try:
            from PySide6.QtWidgets import QMessageBox
            confirm = QMessageBox.question(
                self,
                "Clear update history",
                "¿Borrar todos los registros de update history? "
                "Esta acción no se puede deshacer.",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            update_history.clear()
            self._populate_history()
        except Exception:
            pass

    # -------------------- Diagnostics tab --------------------

    def _build_diagnostics_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 16, 18, 16)

        title = QLabel("Diagnostics")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00d2d3;")
        layout.addWidget(title)

        body = QLabel(
            "Los registros de la aplicación (errores, actualizaciones, "
            "selftest y volcados de faulthandler) se guardan en esta carpeta. "
            "Adjunta app.log al reportar un problema."
        )
        body.setWordWrap(True)
        body.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        layout.addWidget(body)

        self._logs_dir = get_logs_dir()
        self.lbl_logs_dir = QLabel(self._logs_dir)
        self.lbl_logs_dir.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.lbl_logs_dir.setWordWrap(True)
        self.lbl_logs_dir.setStyleSheet(
            "color: #e0e0e0; font-size: 12px; background-color: #16213e;"
            "padding: 8px; border-radius: 6px;"
        )
        layout.addWidget(self.lbl_logs_dir)

        row = QHBoxLayout()
        self.btn_open_logs = QPushButton("Open logs folder")
        self.btn_open_logs.setStyleSheet(
            "background-color: #2c3e50; color: #e0e0e0;"
            "padding: 8px 14px; border-radius: 6px;"
        )
        self.btn_open_logs.clicked.connect(self._open_logs_folder)
        row.addWidget(self.btn_open_logs)
        self.btn_open_backups = QPushButton("Open backups folder")
        self.btn_open_backups.setStyleSheet(
            "background-color: #2c3e50; color: #e0e0e0;"
            "padding: 8px 14px; border-radius: 6px;"
        )
        self.btn_open_backups.clicked.connect(self._open_backups_folder)
        row.addWidget(self.btn_open_backups)
        row.addStretch()
        layout.addLayout(row)

        backups_row = QHBoxLayout()
        self.lbl_backups = QLabel("")
        self.lbl_backups.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        backups_row.addWidget(self.lbl_backups, stretch=1)
        self.btn_clear_backups = QPushButton("Clear backups")
        self.btn_clear_backups.setStyleSheet(
            "background-color: #2c3e50; color: #e0e0e0;"
            "padding: 8px 14px; border-radius: 6px;"
        )
        self.btn_clear_backups.clicked.connect(self._on_clear_backups)
        backups_row.addWidget(self.btn_clear_backups)
        layout.addLayout(backups_row)

        self._refresh_backup_info()
        layout.addStretch()
        return page

    def _refresh_backup_info(self):
        """Actualiza la etiqueta con nº de respaldos y su tamaño total."""
        try:
            num, total = backup_stats()
            if num == 0:
                self.lbl_backups.setText("Crash report backups: none")
            else:
                self.lbl_backups.setText(
                    f"Crash report backups: {num} files · {_format_bytes(total)}"
                )
        except Exception:
            self.lbl_backups.setText("Crash report backups: unavailable")

    def _on_clear_backups(self):
        """Pide confirmación y borra todos los respaldos de crash report."""
        try:
            from PySide6.QtWidgets import QMessageBox

            num, total = backup_stats()
            if num == 0:
                return
            confirm = QMessageBox.question(
                self,
                "Clear crash report backups",
                f"¿Borrar {num} respaldo(s) de crash report "
                f"({_format_bytes(total)})? Esta acción no se puede deshacer.",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            clear_backups()
            self._refresh_backup_info()
        except Exception:
            pass

    def _open_logs_folder(self):
        """Abre la carpeta de logs en el gestor de archivos del SO."""
        try:
            logs_dir = getattr(self, "_logs_dir", None) or get_logs_dir()
            os.makedirs(logs_dir, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(logs_dir))
        except Exception:
            pass

    def _open_backups_folder(self):
        """Abre la carpeta de respaldos (backup/) en el gestor de archivos."""
        try:
            # backup_dir() crea la carpeta si falta antes de abrirla.
            QDesktopServices.openUrl(QUrl.fromLocalFile(backup_dir()))
        except Exception:
            pass

    # -------------------- About tab --------------------

    def _build_about_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 16, 18, 16)

        title = QLabel(APP_NAME)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00d2d3;")
        layout.addWidget(title)

        subtitle = QLabel(f"Version {APP_VERSION}")
        subtitle.setStyleSheet("color: #a0a0b0; font-size: 12px;")
        layout.addWidget(subtitle)

        bits = struct.calcsize("P") * 8
        info = QLabel(
            f"Architecture: {platform.machine()} ({bits}-bit)\n"
            f"OS: {platform.platform()}\n"
            f"Python: {sys.version.split()[0]}\n"
            f"Packaged (PyInstaller): "
            f"{'yes' if getattr(sys, 'frozen', False) else 'no'}"
        )
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setStyleSheet(
            "color: #e0e0e0; font-size: 12px; background-color: #16213e;"
            "padding: 8px; border-radius: 6px;"
        )
        layout.addWidget(info)

        links = QLabel(
            f'<a href="{REPO_URL}">Project repository</a><br>'
            f'<a href="{ISSUES_URL}">Report a problem / issues</a>'
        )
        links.setOpenExternalLinks(True)
        links.setStyleSheet("font-size: 12px;")
        layout.addWidget(links)

        layout.addStretch()
        return page

    # -------------------- Persistencia (Behavior) --------------------

    def _load_values(self):
        try:
            # Auto-update: preferimos la clave nueva; si no existe, heredamos
            # el ajuste legacy 'auto_update_on_close' para no perder la
            # configuración de usuarios que ya tenían la versión FFmpeg-only
            # activa.
            value = preferences.get(_AUTO_UPDATE_KEY_NEW, None)
            if value is None:
                value = preferences.get(_AUTO_UPDATE_KEY_LEGACY, False)
            self.chk_auto_update.setChecked(bool(value))

            # Output folder + Compression preferences: defaults conservadores.
            self.chk_auto_open_output.setChecked(
                bool(preferences.get(_AUTO_OPEN_LAST_OUTPUT_KEY, True))
            )
            self.chk_health_check.setChecked(
                bool(preferences.get(_HEALTH_CHECK_PROMPT_KEY, True))
            )
            self.chk_remember_codec.setChecked(
                bool(preferences.get(_REMEMBER_CODEC_KEY, True))
            )
            mode = str(preferences.get(_EXEC_PROVIDER_KEY, "Auto"))
            index = self.cmb_exec_provider.findText(mode)
            if index >= 0:
                self.cmb_exec_provider.setCurrentIndex(index)
            self.chk_insecure_checkpoint.setChecked(
                bool(preferences.get(_INSECURE_CHECKPOINT_KEY, False))
            )
            self.txt_hf_token.setText(_get_hf_token())
        except Exception:
            self.chk_auto_update.setChecked(False)
            self.chk_auto_open_output.setChecked(True)
            self.chk_health_check.setChecked(True)
            self.chk_remember_codec.setChecked(True)
            self.chk_insecure_checkpoint.setChecked(False)
            self.txt_hf_token.setText("")

    def _on_accept(self):
        # Bloque unico: todas las preferencias nuevas + limpieza optional
        # de la legacy + dialog aceptar. Las excepciones se acotan a
        # errores de E/S del JSON (OSError, JSONDecodeError) para que un
        # bug del codigo (typo, KeyError inesperado) NO se trague
        # silenciosamente.
        try:
            preferences.set(
                _AUTO_UPDATE_KEY_NEW,
                bool(self.chk_auto_update.isChecked()),
            )
            preferences.set(
                _AUTO_OPEN_LAST_OUTPUT_KEY,
                bool(self.chk_auto_open_output.isChecked()),
            )
            preferences.set(
                _HEALTH_CHECK_PROMPT_KEY,
                bool(self.chk_health_check.isChecked()),
            )
            preferences.set(
                _REMEMBER_CODEC_KEY,
                bool(self.chk_remember_codec.isChecked()),
            )
            preferences.set(
                _EXEC_PROVIDER_KEY,
                str(self.cmb_exec_provider.currentText()),
            )
            preferences.set(
                _INSECURE_CHECKPOINT_KEY,
                bool(self.chk_insecure_checkpoint.isChecked()),
            )
            _set_hf_token(str(self.txt_hf_token.text() or ""))
            # Limpiamos la clave legacy si estaba presente para que no
            # acaben conviendo las dos: queremos una única fuente de
            # verdad a partir de ahora.
            if preferences.has(_AUTO_UPDATE_KEY_LEGACY):
                path = os.path.join(
                    get_session_dir(), "preferences.json"
                )
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    data.pop(_AUTO_UPDATE_KEY_LEGACY, None)
                    ensure_runtime_dirs()
                    # Atómico (tmp + replace): preferences.json se escribe
                    # en cada aceptar; un corte de energía mid-write no
                    # debería volver todas las preferencias a default.
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    os.replace(tmp, path)
        except (OSError, json.JSONDecodeError):
            # I/O o corrupcion del JSON: no podemos persistir el cambio,
            # pero tampoco queremos romper el cierre del dialogo.
            pass
        self.accept()
