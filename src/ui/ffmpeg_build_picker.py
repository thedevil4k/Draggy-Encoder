"""Selector de versión de FFmpeg (la fila de FFmpeg del panel de dependencias).

Widget tonto a propósito: recibe el carrier de
``src.ffmpeg.build_fit.evaluate_build_list`` —la lista de builds ya
evaluadas (id, etiqueta, status, reason, min_driver…) **y la plataforma a la
que pertenecen**— y la pinta. Aquí no hay lógica de catálogo ni de hardware,
sólo presentación:

* ``STATUS_STYLE`` es el ÚNICO sitio donde un veredicto se traduce a marca y
  color. No hay fallbacks dispersos por el código.
* ``note_text()`` es el único sitio donde se compone el texto bajo el
  desplegable, y nombra la plataforma que **viene en el carrier**, nunca una
  consultada aparte: el texto tiene que describir la lista que tiene delante.
* ``incompatibility_warning()`` es el único sitio donde se decide qué avisar
  antes de sustituir el FFmpeg que ya funciona.

La elección se persiste aquí mismo (``src.ffmpeg.builds`` es su dueño): este
widget *es* el control de esa preferencia, así que nadie más tiene que
recordar guardarla. Quien quiera saber qué se instalará pregunta a
``builds.get_preferred_build_id()``, no al widget.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from src.ffmpeg import build_fit, builds

logger = logging.getLogger(__name__)

# Semáforo: verde sólo cuando hay evidencia (GPU reconocida + driver
# suficiente), ámbar cuando no se puede comprobar, rojo cuando el driver
# instalado no llega al mínimo de esa build, gris cuando da igual.
STATUS_STYLE = {
    build_fit.COMPATIBLE: ("✔", "#2ecc71"),
    build_fit.INCOMPATIBLE: ("✖", "#e74c3c"),
    build_fit.UNVERIFIED: ("?", "#f39c12"),
    build_fit.NEUTRAL: ("•", "#c0c0d0"),
}
_FALLBACK_STYLE = ("•", "#c0c0d0")

# Sin mirrors para este sistema, el catálogo no tiene nada publicable y el
# desplegable llega vacío. Antes eso se traducía en un desplegable mudo y sin
# nota: el usuario veía que faltaba el selector sin saber por qué. Este texto
# es el que explica el vacío, y por eso vive aquí, que es donde vive el resto
# del texto del selector.
_NO_BUILDS_NOTE = (
    "No hay ninguna versión de FFmpeg publicable para {plataforma}: ninguna "
    "build del catálogo tiene mirror para este sistema, así que no hay nada "
    "que descargar. El selector queda desactivado en vez de ofrecer una "
    "opción que fallaría al instalarse."
)

# El otro vacío, que no es el mismo: aqui las builds SI existen para la
# plataforma, lo que falló fue la lectura del hardware, así que no se puede
# decir cuál le sirve al equipo. Decirle al usuario que no hay builds cuando solo
# no se pudo comprobar sería mentira, y además cerraría la puerta a reintentar
# algo que tiene sentido reintentar.
_EVALUATION_FAILED_NOTE = (
    "No se ha podido comprobar qué versiones de FFmpeg sirven en este equipo, "
    "así que el selector queda vacío. No es que falten builds para {plataforma}: "
    "es que no se ha podido leer el hardware, y sin ese dato no se puede decir "
    "cuál te serviría. Pulsa 'Re-detect Hardware & Codecs' e inténtalo de nuevo; "
    "mientras tanto, el FFmpeg que ya tengas instalado sigue funcionando."
)


def _style(status):
    return STATUS_STYLE.get(status, _FALLBACK_STYLE)


class FFmpegBuildPicker(QWidget):
    """Desplegable de builds + la nota que explica el veredicto de la elegida."""

    def __init__(self, build_list, installed_build_id=None,
                 preferred_build_id=None, parent=None):
        super().__init__(parent)
        # El carrier (``build_fit.evaluate_build_list``) trae la plataforma
        # JUNTO a la lista, y el texto se compone con esa. Antes el selector
        # llamaba a ``builds.current_platform()`` por su cuenta: si la lista
        # era de otra plataforma, el texto podía nombrar la de la máquina y
        # los datos y el texto acababan hablando de sistemas distintos. Con
        # la plataforma en el carrier no hay dónde mirar otro sitio.
        #
        # ``evaluation_failed`` distingue los dos vacíos: lista vacía es "no
        # hay builds para esta plataforma" (un hecho del catálogo) y esto es
        # "no se ha podido comprobar" (no se sabe), y el texto cambia.
        self._platform = build_list["platform"] or "este sistema"
        self._evaluation_failed = build_list["evaluation_failed"]
        options = list(build_list["entries"])
        self._options = {opt["id"]: opt for opt in options}
        self.selected_build_id = None
        self.selected_build_status = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("Versión de FFmpeg a instalar")
        title.setStyleSheet("color: #00d2d3; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        self.combo = QComboBox()
        self.combo.setToolTip(
            "Elige qué versión se descarga al instalar o reinstalar FFmpeg.\n"
            "El color indica si esa versión sirve para tu GPU: verde con "
            "evidencia, ámbar si no se puede comprobar y rojo si no encaja."
        )
        for option in options:
            self.combo.addItem(self._option_label(option, installed_build_id),
                               option["id"])
            index = self.combo.count() - 1
            self.combo.setItemData(
                index,
                QColor(_style(option.get("status"))[1]),
                Qt.ItemDataRole.ForegroundRole,
            )
        if not self.combo.count():
            # Nada que elegir aquí: un desplegable activo y vacío parece un
            # fallo de la interfaz, y además invite a tocarlo.
            self.combo.setEnabled(False)
        layout.addWidget(self.combo)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        self.setLayout(layout)

        # Preferencia del usuario; si no es válida, la build instalada.
        index = self.combo.findData(preferred_build_id or builds.DEFAULT_BUILD_ID)
        if index < 0:
            index = next(
                (i for i, opt in enumerate(options)
                 if opt["id"] == installed_build_id), 0
            )
        # La nota se pinta antes de fijar el índice para que el veredicto de la
        # build elegida aparezca desde el primer frame; y la señal se conecta
        # después de ``setCurrentIndex`` para no persistir nada al pintar.
        self.selected_build_id = self.combo.itemData(index)
        self._refresh_note()
        self.combo.setCurrentIndex(index)
        self.combo.currentIndexChanged.connect(self._on_changed)

    # -- texto ------------------------------------------------------------

    def _option_label(self, option, installed_build_id):
        mark = _style(option.get("status"))[0]
        suffix = "  (instalada)" if option["id"] == installed_build_id else ""
        return f"{mark}  {option.get('label') or option['id']}{suffix}"

    def note_text(self):
        """Texto bajo el desplegable para la build seleccionada.

        Sin build seleccionada hay dos huecos distintos que explicar, y el
        usuario tiene que poder leer en los dos por qué no puede elegir
        versión: que no haya nada para su sistema, o que no hayamos podido
        comprobarlo. Lo segundo además dice que reintentar tiene sentido.
        """
        if self._evaluation_failed:
            return _EVALUATION_FAILED_NOTE.format(plataforma=self._platform)
        if not self._options:
            return _NO_BUILDS_NOTE.format(plataforma=self._platform)
        option = self._options.get(self.selected_build_id) or {}
        parts = [option.get("reason") or ""]
        if option.get("min_driver"):
            parts.append(f"driver NVIDIA mínimo: {option['min_driver']}")
        # El aviso del catálogo: si la build se reconstruye a diario su mínimo
        # sube con ella, y eso es justo lo que hace que cambiar de versión no
        # arregle un driver viejo. Sin esto el selector promete de más.
        if option.get("min_driver_note"):
            parts.append(option["min_driver_note"])
        return "  ·  ".join(p for p in parts if p)

    def incompatibility_warning(self):
        """Aviso previo a reinstalar, o ``None`` si la elegida no es un problema.

        Instalar una build que el driver no soporta deja al equipo sin
        codificar por GPU, y esto es lo último que se ve antes de sustituir el
        FFmpeg que ya funciona.
        """
        if self.selected_build_status != build_fit.INCOMPATIBLE:
            return None
        reason = (self._options.get(self.selected_build_id) or {}).get("reason", "")
        first = reason.split(". ")[0].strip()
        return f"{first}." if first else reason

    # -- estado -----------------------------------------------------------

    def _refresh_note(self):
        option = self._options.get(self.selected_build_id) or {}
        self.selected_build_status = option.get("status")
        text = self.note_text()
        self._note.setText(text)
        self._note.setStyleSheet(
            f"color: {_style(self.selected_build_status)[1]}; font-size: 11px;"
        )
        self.combo.setToolTip(text)

    def _on_changed(self, _index):
        """Guarda la versión elegida y repinta su nota."""
        self.selected_build_id = self.combo.currentData()
        self._refresh_note()
        if builds.get_preferred_build_id() == self.selected_build_id:
            return
        try:
            # Se persiste porque es la versión que se descargará si FFmpeg falta
            # en el próximo arranque y la que reinstala la actualización
            # automática. Si no ha cambiado no se toca el preferences.json.
            builds.set_preferred_build_id(self.selected_build_id)
        except Exception:
            logger.exception("Could not persist the chosen FFmpeg build")
