"""La fila de una dependencia: lo que se ve y lo que se puede hacer con ella.

Vive fuera de ``dependencies_dialog`` por la razon que ya movio a
``ffmpeg_row`` y a ``ffmpeg_build_picker``: es la parte que hay que abrir
para tocarla, y no deberia obligar a leer el panel entero para entender por
que el boton principal dice una cosa u otra.

Dentro hay dos cosas de naturaleza distinta, y esa es la separacion que
importa:

* **Lo que se pinta**: ``DepRow`` monta etiquetas, insignias y botones a
  partir de una fila de ``src.dependencies``. No decide nada.
* **Lo que se decide**: ``badge_text``, ``primary_label``,
  ``primary_enabled`` y ``ai_runtime_support_info`` son funciones puras de
  la fila. Son el unico sitio donde se responde "que pone el boton" y "esta
  vivo el boton", y por eso son funciones y no metodos: no usan el widget
  para nada, se pueden probar sin ``QApplication`` y se pueden cambiar sin
  tocar la maqueta.

Que el boton principal este muerto tiene tres motivos distintos, y los tres
estan aqui: que la dependencia no se actualiza desde este panel (los modelos
se gestionan en la pestaña AI Engines), que en esta plataforma no hay
build de la que bajar (``nothing_to_install``, que llega en los datos y que
el manager ya calculo), y que el runtime de IA no tiene ruedas para este
Python. El mismo motivo llega a la etiqueta, al boton, a la URL y al
tooltip, y por eso se pregunta una sola vez.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

import src.dependencies as dependencies
from src.ffmpeg.builds import timestamp_age_hours
from src.runtimes.manager import get_python_support_info as _get_python_support_info
from src.ui.dialog_utils import open_folder
from src.ui.ffmpeg_row import mount_build_picker


# ---------------------------------------------------------------------------
# Que dice la fila, y que se puede hacer con ella
# ---------------------------------------------------------------------------


def badge_text(status):
    return {
        dependencies.STATUS_INSTALLED: "INSTALLED",
        dependencies.STATUS_MISSING: "MISSING",
        dependencies.STATUS_ERROR: "ERROR",
        dependencies.STATUS_COMING_SOON: "COMING SOON",
    }.get(status, status or "—")


def primary_label(info):
    # Ranuras que no tienen instalador automático desde este panel:
    # los modelos se gestionan en la pestaña AI Engines.
    if not info.get("is_updatable"):
        return "No disponible"
    # Una build de la que esta plataforma no tiene mirror no se puede
    # instalar: ofrecer el botón sería prometer una descarga que va a
    # fallar. El motivo llega en los datos (``nothing_to_install``), que
    # es donde ya lo preguntaron la fila del selector, el banner y la
    # confirmación de reinstalar. Sólo aplica a FFmpeg: torch y ONNX
    # Runtime se instalan con su propio instalador.
    if info.get("nothing_to_install"):
        return "No se puede instalar"
    # Si upstream sugiere build más reciente y está instalado,
    # el botón principal pasa a ser "Update" en lugar de "Reinstall".
    if (
        info.get("status") == dependencies.STATUS_INSTALLED
        and info.get("update_available") is True
    ):
        return "Actualizar"
    if info.get("status") == dependencies.STATUS_INSTALLED:
        return "Reinstalar"
    if info.get("status") == dependencies.STATUS_MISSING:
        return "Instalar"
    if info.get("status") == dependencies.STATUS_ERROR:
        return "Reintentar"
    return "No disponible"


def primary_enabled(info, support=None):
    if not info.get("is_updatable"):
        return False
    # Mismo motivo que en la etiqueta: sin mirror para esta plataforma no
    # hay nada que instalar, y el botón muerto es la respuesta honesta.
    if info.get("nothing_to_install"):
        return False
    # AI runtimes cannot be installed on Python versions without wheels.
    if support is None:
        support = ai_runtime_support_info(info)
    if support and not support["supported"]:
        return False
    return True


def ai_runtime_support_info(info):
    """Return support info for AI runtime rows (torch/onnxruntime)."""
    if info.get("kind") != dependencies.KIND_RUNTIME:
        return None
    dep_id = info.get("id") or ""
    if dep_id in ("torch", "onnxruntime"):
        return _get_python_support_info()
    return None


def humanize_age(iso_text):
    """Dice "hace cuanto" en un solo formato, o ``None`` si no hay fecha.

    Es el unico sitio donde se decide como se escribe una edad, y lo consultan
    las dos superficies que la enseñan: la columna ``Latest`` de la fila y el
    banner de la ventana principal. Antes de que existiera, cada una hacia su
    propia cuenta y el usuario leia "3.1 h ago" en una y "3 h ago" en la otra
    para el mismo instante.

    Los numeros van enteros y se truncan. "3 h ago" es lo que dice la gente, y
    "3.1 h ago" daria precision de tres minutos sobre una marca de tiempo que
    solo tiene minutos. Truncar ademas hace que el criterio sea coherente
    consigo mismo: antes se mezclaba ``int()``, que trunca, con ``:.1f``, que
    redondea, y esa mezcla empujaba las 47 horas a "2.0 d ago" cuando faltaba
    media hora para el segundo dia. Truncando, 47 horas son un dia y no media
    hora de mentira.

    Los cortes no se tocan —menos de una hora en minutos, menos de un dia en
    horas, y a partir de ahi en dias— porque los sitios que compartian el
    formato ya coincidian en ellos, y lo unico que se duplicaba era como se
    escribia el numero.

    Una edad negativa se recorta a cero en vez de imprimirse: un reloj atrasado
    daria "checked -30 min ago", que no es una edad, es un fallo del reloj.

    Devuelve ``None`` cuando no hay fecha o cuando no se entiende. La palabra
    que pone cada superficie en ese caso es suya y no se unifica aqui, porque
    no es un formato: una celda tiene que pintar algo y un banner puede
    callarse. Lo que no puede pasar es que las dos digan una edad distinta, y
    para eso no hace falta que el ``None`` se convierta en texto aquí.

    Leer la marca no lo decide esta función: lo decide
    ``builds.timestamp_age_hours``, que es donde vive el unico parseo del
    arbol. Aqui solo se traduce una edad a palabras.
    """
    hours = timestamp_age_hours(iso_text)
    if hours == float("inf"):
        # El centinela de "no hay marca" es inf, no None, porque el otro
        # consumidor —el cache de 24 h— lo compara y con None tendria que
        # comprobarlo a mano. El None de aqui es otra cosa: es que esta
        # superficie no tiene nada que decir.
        return None
    if hours < 0:
        # Un reloj atrasado, o una cache escrita por una maquina que iba
        # rapida, daria una edad negativa. La edad se recorta aqui y no antes
        # porque "cuanto" y "como se dice cuanto" son dos preguntas.
        hours = 0.0
    if hours < 1:
        return f"{int(hours * 60)} min ago"
    if hours < 24:
        return f"{int(hours)} h ago"
    return f"{int(hours / 24)} d ago"


def format_latest_line(info):
    """Devuelve el texto "Latest: … · last check: …" para la columna Latest."""
    fp = info.get("latest_fingerprint") or {}
    last_mod = fp.get("last_modified_iso")
    checked_at = fp.get("checked_at")
    if not last_mod:
        if info.get("upstream_error"):
            return "Latest: — (offline) · last check: never"
        return "Latest: — · last check: never"
    # Mostrar solo la fecha (YYYY-MM-DD) para no saturar la línea
    short = last_mod.split("T", 1)[0]
    # La edad la dice humanize_age, que es su unico dueño. Aqui solo se decide
    # que palabra poner cuando no hay fecha: una celda no puede quedar vacia.
    age_text = humanize_age(checked_at) or "—"
    return f"Latest upstream build: {short}  ·  last check: {age_text}"



# ---------------------------------------------------------------------------
# La fila
# ---------------------------------------------------------------------------


_STATUS_COLORS = {
    dependencies.STATUS_INSTALLED: ("#00d2d3", "#0f0f1a"),      # turquesa (sobre fondo claro)
    dependencies.STATUS_MISSING: ("#e74c3c", "#ffffff"),         # rojo
    dependencies.STATUS_ERROR: ("#f39c12", "#0f0f1a"),           # ámbar
    dependencies.STATUS_COMING_SOON: ("#5f27cd", "#ffffff"),     # lila (coming soon)
}


class DepRow(QFrame):
    """Una fila con el estado + versión + acciones de una dependencia."""

    redownload_requested = Signal(str)
    uninstall_requested = Signal(str)

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.dep_id = info.get("id")
        # Selector de versión (None si la dependencia no es versionable,
        # p. ej. PyTorch).
        self.build_picker = None
        self.setObjectName("DepRow")
        self.setStyleSheet(self._stylesheet())
        self._build(info)

    def _stylesheet(self):
        return """
        #DepRow {
            background-color: #16213e;
            border: 1px solid #2c3e50;
            border-radius: 8px;
        }
        #DepRow QPushButton {
            background-color: #1f2c4c;
            color: #e0e0e0;
            border: 1px solid #2c3e50;
            border-radius: 6px;
            padding: 6px 12px;
        }
        #DepRow QPushButton:hover {
            background-color: #00d2d3;
            color: #0f0f1a;
        }
        #DepRow QPushButton:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #DepRow QPushButton#uninstallBtn {
            background-color: #3a1620;
            color: #ff6b6b;
            border: 1px solid #6e2233;
        }
        #DepRow QPushButton#uninstallBtn:hover {
            background-color: #e74c3c;
            color: #ffffff;
        }
        #DepRow QPushButton#uninstallBtn:disabled {
            background-color: #2c3e50;
            color: #5f6f7f;
        }
        #StatusBadge {
            border-radius: 9px;
            padding: 2px 10px;
            font-weight: bold;
            font-size: 11px;
        }
        """

    def _build(self, info):
        outer = QHBoxLayout(self)
        outer.setSpacing(14)
        outer.setContentsMargins(14, 12, 14, 12)

        # Columna izquierda: nombre + versión + ruta
        left = QVBoxLayout()
        left.setSpacing(4)
        title = QLabel(info.get("display_name", "—"))
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #ffffff;")
        left.addWidget(title)

        kind_label = info.get("kind", "").upper()
        # La versión que se lee de un vistazo ("9.x-dev (master)", "9.0.2") y,
        # cuando no es lo mismo que ella, el token de compilación que imprime
        # el binario. `build_label` es la build ELEGIDA, no la instalada, así
        # que su campo no puede seguir llamándose `build`: era justo esa
        # ambigüedad lo que hacía imposible saber qué versión había instalada.
        version_text = info.get("version_text") or info.get("version") or "—"
        meta_text = f"{kind_label}  ·  version: {version_text}"
        if info.get("build_label"):
            if info.get("version") and info["version"] != version_text:
                meta_text += f"  ·  build: {info['version']}"
            meta_text += f"  ·  selected: {info['build_label']}"
        meta = QLabel(meta_text)
        meta.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        left.addWidget(meta)

        path_label = QLabel(
            info.get("install_dir") or "(no install dir)"
        )
        path_label.setStyleSheet(
            "color: #707080; font-size: 10px; font-family: Consolas, monospace;"
        )
        left.addWidget(path_label)

        # La URL de origen se guarda en el manifest como hecho (de dónde salió
        # ESTE binario), pero no se pinta cuando aquí no se puede instalar
        # nada: junto a un botón muerto, una URL que no se puede pulsar es un
        # callejón sin salida. En una plataforma con mirrors no cambia nada.
        if info.get("source_url") and not info.get("nothing_to_install"):
            url_label = QLabel(info["source_url"])
            url_label.setStyleSheet(
                "color: #00d2d3; font-size: 10px;"
            )
            left.addWidget(url_label)

        # Línea "Latest <fecha>" cuando hay datos de upstream (sólo BINARY)
        if info.get("kind") == dependencies.KIND_BINARY:
            latest_text = format_latest_line(info)
            if latest_text:
                latest_label = QLabel(latest_text)
                latest_label.setStyleSheet("color: #00d2d3; font-size: 11px;")
                left.addWidget(latest_label)

        if info.get("detail"):
            detail = QLabel(info["detail"])
            detail.setStyleSheet("color: #c0c0d0; font-size: 11px;")
            detail.setWordWrap(True)
            left.addWidget(detail)

        # Selector de versión: sólo la fila de FFmpeg trae ``build_options``.
        if "build_options" in info:
            self.build_picker = mount_build_picker(self, left, info)

        # AI runtime compatibility warning (Python 3.14, etc.)
        support = ai_runtime_support_info(info)
        if support and not support["supported"]:
            warn = QLabel(support["message"])
            warn.setStyleSheet("color: #f39c12; font-size: 11px; font-weight: bold;")
            warn.setWordWrap(True)
            left.addWidget(warn)

        outer.addLayout(left, stretch=1)

        # Columna derecha: badge principal + (opcional) "Update available"
        right = QVBoxLayout()
        right.setSpacing(6)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        badge_row.setAlignment(Qt.AlignmentFlag.AlignRight)

        badge = QLabel(badge_text(info.get("status")))
        status = info.get("status")
        fg_bg = _STATUS_COLORS.get(status, ("#a0a0b0", "#0f0f1a"))
        badge.setStyleSheet(
            f"#StatusBadge {{ background-color: {fg_bg[0]}; color: {fg_bg[1]}; }}"
        )
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setObjectName("StatusBadge")
        badge.setMaximumWidth(140)
        badge_row.addWidget(badge)

        # Insignia secundaria "Update available" cuando aplica
        if info.get("update_available") is True:
            ua = QLabel("UPDATE AVAILABLE")
            ua.setObjectName("UpdateBadge")
            ua.setStyleSheet(
                "#UpdateBadge { background-color: #f39c12; color: #0f0f1a; "
                "border-radius: 9px; padding: 2px 10px; font-weight: bold; font-size: 11px; }"
            )
            ua.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge_row.addWidget(ua)
        elif info.get("upstream_error"):
            ua = QLabel("OFFLINE")
            ua.setObjectName("UpdateBadge")
            ua.setStyleSheet(
                "#UpdateBadge { background-color: #2c3e50; color: #a0a0b0; "
                "border-radius: 9px; padding: 2px 10px; font-weight: bold; font-size: 11px; }"
            )
            ua.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ua.setToolTip("Could not reach the upstream URL.\n" + str(info["upstream_error"]))
            badge_row.addWidget(ua)

        right.addLayout(badge_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        # La variable local no se llama como la función: si se llamara
        # ``primary_label``, Python la tomaria por local en todo el metodo y
        # la llamada de la linea siguiente fallaria por no estar aun asignada.
        texto_principal = primary_label(info)
        support = ai_runtime_support_info(info)
        self.btn_primary = QPushButton(texto_principal)
        self.btn_primary.setEnabled(primary_enabled(info, support=support))
        if info.get("kind") == dependencies.KIND_MODEL:
            self.btn_primary.setToolTip(
                "Add or remove individual engines from the 'AI Engines' tab."
            )
        if info.get("nothing_to_install"):
            # El botón muerto necesita una razón a la vista, y es la MISMA
            # frase que ya usan el banner, el selector y la confirmación de
            # reinstalar: un solo enunciado para este hecho, no cuatro.
            self.btn_primary.setToolTip(
                f"No se puede instalar aquí: {info['nothing_to_install']}."
            )
        if info.get("update_available") is True:
            self.btn_primary.setStyleSheet(
                "QPushButton { background-color: #f39c12; color: #0f0f1a; "
                "border: none; border-radius: 6px; padding: 6px 12px; font-weight: bold; }"
                "QPushButton:hover { background-color: #f5b041; }"
                "QPushButton:disabled { background-color: #2c3e50; color: #5f6f7f; }"
            )
        self.btn_primary.clicked.connect(
            lambda: self.redownload_requested.emit(info.get("id") or info.get("display_name"))
        )
        btn_row.addWidget(self.btn_primary)

        self.btn_open = QPushButton("Open folder")
        self.btn_open.setEnabled(bool(info.get("install_dir")))
        self.btn_open.clicked.connect(lambda: self._open_path(info.get("install_dir")))
        btn_row.addWidget(self.btn_open)

        # Desinstalar / borrar lo instalado (binarios, runtimes, modelos).
        can_uninstall = bool(info.get("deletable")) and info.get(
            "status"
        ) == dependencies.STATUS_INSTALLED
        self.btn_uninstall = QPushButton("Uninstall")
        self.btn_uninstall.setObjectName("uninstallBtn")
        self.btn_uninstall.setEnabled(can_uninstall)
        if not can_uninstall:
            self.btn_uninstall.setToolTip(
                "Nothing installed to remove, or it cannot be removed from the app."
            )
        self.btn_uninstall.clicked.connect(
            lambda: self.uninstall_requested.emit(
                info.get("id") or info.get("display_name")
            )
        )
        btn_row.addWidget(self.btn_uninstall)

        right.addLayout(btn_row)
        right.addStretch()
        outer.addLayout(right)

    def _open_path(self, path):
        # El boton se puede pulsar sin ``install_dir``; sin carpeta no hay
        # nada que abrir, y el aviso de "no se pudo abrir" seria ruido.
        if not path:
            return
        open_folder(path, self)
