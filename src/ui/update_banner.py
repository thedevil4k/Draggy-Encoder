"""Banner persistente que avisa al usuario cuando hay una build más reciente
de FFmpeg en el upstream.

Se muestra en ``MainWindow`` en cuanto ``is_update_available(manifest, fp)``
devuelve ``True``. Ofrece tres acciones:
  - **Update now**: dispara una descarga en background con progreso atado a
    la barra principal.
  - **Open panel**: abre el diálogo de gestión de dependencias (``Manage
    Dependencies``) para más opciones.
  - **Dismiss**: oculta el banner durante la sesión actual; reaparece al
    próximo arranque si el upstream sigue ofertando una build más reciente.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)


class UpdateBanner(QFrame):
    """Banner horizontal anclado debajo de los botones del header."""

    update_requested = Signal()
    panel_requested = Signal()
    history_requested = Signal()
    dismissed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("UpdateBanner")
        self.setStyleSheet(self._stylesheet())
        # Comienza oculto. Cuando está oculta no ocupa espacio en el layout
        # del padre porque gestionamos su tamaño con setVisible().
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setVisible(False)
        self._build()

    def _stylesheet(self):
        return """
        #UpdateBanner {
            background-color: #16213e;
            border: 1px solid #f39c12;
            border-radius: 8px;
        }
        #UpdateBanner[state="success"] {
            background-color: #16332e;
            border: 1px solid #2ecc71;
        }
        #UpdateBanner QPushButton {
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: bold;
        }
        #UpdateBanner #btn_update {
            background-color: #f39c12;
            color: #0f0f1a;
        }
        #UpdateBanner #btn_update:hover {
            background-color: #f5b041;
        }
        #UpdateBanner #btn_update:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #UpdateBanner #btn_panel,
        #UpdateBanner #btn_history {
            background-color: #1f2c4c;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
        }
        #UpdateBanner #btn_panel:hover,
        #UpdateBanner #btn_history:hover {
            background-color: #00d2d3;
            color: #0f0f1a;
        }
        #UpdateBanner #btn_dismiss {
            background-color: transparent;
            color: #a0a0b0;
        }
        #UpdateBanner #btn_dismiss:hover {
            color: #e0e0e0;
        }
        """

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 10, 14, 10)

        self.icon_label = QLabel()
        icon_font = QFont()
        icon_font.setPointSize(14)
        icon_font.setBold(True)
        self.icon_label.setFont(icon_font)
        self.icon_label.setStyleSheet("color: #f39c12;")
        self.icon_label.setFixedWidth(24)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        text_col = QHBoxLayout()
        text_col.setSpacing(6)

        self.title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #ffffff;")
        text_col.addWidget(self.title_label)

        self.detail_label = QLabel()
        self.detail_label.setStyleSheet("color: #c0c0d0; font-size: 11px;")
        text_col.addWidget(self.detail_label, stretch=1)

        layout.addLayout(text_col, stretch=1)

        self.btn_update = QPushButton("Update now")
        self.btn_update.setObjectName("btn_update")
        self.btn_update.clicked.connect(self.update_requested.emit)
        layout.addWidget(self.btn_update)

        self.btn_panel = QPushButton("Open panel")
        self.btn_panel.setObjectName("btn_panel")
        self.btn_panel.clicked.connect(self.panel_requested.emit)
        layout.addWidget(self.btn_panel)

        self.btn_history = QPushButton("Show History")
        self.btn_history.setObjectName("btn_history")
        self.btn_history.clicked.connect(self.history_requested.emit)
        self.btn_history.setVisible(False)
        layout.addWidget(self.btn_history)

        self.btn_dismiss = QPushButton("Dismiss")
        self.btn_dismiss.setObjectName("btn_dismiss")
        self.btn_dismiss.clicked.connect(self.dismissed.emit)
        layout.addWidget(self.btn_dismiss)

    # -------------------- API pública --------------------

    def show_update(self, title, detail=""):
        """Estado 'update disponible': botón Update habilitado, icono flecha."""
        self.icon_label.setText("\u2191")  # flecha arriba
        self.icon_label.setStyleSheet("color: #f39c12;")
        self.title_label.setText(title)
        self.detail_label.setText(detail)
        self.btn_update.setEnabled(True)
        self._refresh_visibility(True)

    def show_offline(self, title, detail=""):
        """Estado 'no se pudo contactar upstream': botón Update deshabilitado."""
        self.icon_label.setText("\u26A0")  # signo de advertencia
        self.icon_label.setStyleSheet("color: #e74c3c;")
        self.title_label.setText(title)
        self.detail_label.setText(detail)
        self.btn_update.setEnabled(False)
        self._refresh_visibility(True)

    def show_success(self, title, detail=""):
        """Estado 'update completo': borde verde + check + Show History + Dismiss.

        Los botones ``btn_update`` y ``btn_panel`` se ocultan: la actualización
        ya ocurrió, no queremos confundir al usuario con un Update now.
        ``btn_history`` queda visible y conectado a ``history_requested``.
        """
        # QSS selector de atributo: aplicamos borde verde solo en estado success.
        self.setProperty("state", "success")
        self._repolish_style()
        self.icon_label.setText("\u2713")  # check
        self.icon_label.setStyleSheet("color: #2ecc71;")
        self.title_label.setText(title)
        self.title_label.setStyleSheet("color: #2ecc71;")
        self.detail_label.setText(detail)
        # Reposiciona los botones para el estado de exito: solo Show History
        # + Dismiss son relevantes. Update y Open panel se ocultan.
        self.btn_update.setVisible(False)
        self.btn_panel.setVisible(False)
        self.btn_history.setVisible(True)
        self.btn_history.setText("Show History")
        self.btn_dismiss.setVisible(True)
        self._refresh_visibility(True)

    def _repolish_style(self):
        """Re-aplica el stylesheet tras cambiar la property ``state``.

        Qt cachea el QSS cuando se asigna ``setStyleSheet`` una sola vez;
        cambiar ``setProperty`` no invalida esa cache, por lo que tenemos
        que forzar la repolish para que el selector ``#UpdateBanner[state="x"]``
        surta efecto visualmente.
        """
        try:
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
        except Exception:
            pass

    def hide_banner(self):
        # Reset state to '' so a future show_update / show_offline renders
        # with the default border + Update-enabled buttons, not with the
        # green success border. The QSS selector #UpdateBanner[state="x"]
        # is cached by Qt when setStyleSheet runs in __init__, so we have
        # to clear the property AND repolish for the change to take effect.
        if self.property("state"):
            self.setProperty("state", "")
            self._repolish_style()
        self._refresh_visibility(False)

    def _refresh_visibility(self, should_show):
        # Llamada idempotente. Usada por las señales para no togglear el
        # estado del layout si ya estaba en el estado deseado.
        if self.isVisible() != should_show:
            self.setVisible(should_show)
