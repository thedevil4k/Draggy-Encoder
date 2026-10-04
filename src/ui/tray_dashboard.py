"""El desplegable de la bandeja: cómo va la tarea, sin abrir la ventana.

Un ``QMenu`` con acciones de texto se queda congelado en el instante en que
se abre, que es justo el instante en que uno lo mira. Por eso esto no son
``addAction`` sino un ``QWidgetAction`` con un widget dentro: el menú lo
pinta una vez y el widget se sigue actualizando mientras está abierto. La
diferencia se nota justo cuando más falta hace, a mitad de un render largo.

El widget **no** mide nada y **no** decide nada. Recibe un dict y lo pinta;
los números los trae ``src.hardware.metrics`` y el texto lo componen las
funciones puras de aquí, que se pueden probar sin ``QApplication``. Es la
misma separación que ``dep_row`` / ``ffmpeg_row``: lo que se dice, aparte de
lo que se pinta.

Dos cosas que este módulo no hace a propósito:

* **No enseña ceros falsos.** Si no hay GPU, no hay línea de GPU. Si la CPU
  todavía no tiene un intervalo que promediar, sale ``—``. Una línea que
  dice "0 %" y una que dice "no lo sé" son afirmaciones distintas, y aquí
  sólo se hace la segunda.
* **No se come los clics.** Va con ``WA_TransparentForMouseEvents``: son
  números, no botones, y no tienen por qué cerrar el menú.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from src.hardware.metrics import format_fps, format_percent, format_vram

# Ancho fijo: el desplegable de un menú del sistema se dimensiona por su
# contenido, y si el nombre del vídeo cambia de largo el menú da un tirón
# cada vez que el usuario abre otra tarea. Ancho constante, texto el que toque.
ANCHO = 320


def eta_text(elapsed_seconds, pct):
    """El tiempo que queda, como se escribe en la barra de la ventana.

    Vive aquí, y no duplicado en el desplegable, porque los dos sitios
    muestran lo mismo y no pueden discrepar: si mañana cambia el formato en
    uno y no en el otro, la app dice dos cosas distintas sobre el mismo
    render. ``None`` cuando todavía no hay bastante datos para decirlo
    (``pct`` a 0, o nada más de dos segundos medidos), porque un ETA
    calculado con dos muestras es un numero que miente con mucha seguridad.
    """
    try:
        pct = int(pct)
        elapsed = float(elapsed_seconds)
    except (TypeError, ValueError):
        return None
    if pct <= 1 or elapsed <= 2:
        return None
    remaining = elapsed * (100 - pct) / pct
    if remaining < 0:
        return None
    mm, ss = divmod(int(remaining), 60)
    hh, mm = divmod(mm, 60)
    if hh:
        return "ETA ~%dh %02dm %02ds" % (hh, mm, ss)
    return "ETA ~%02d:%02d" % (mm, ss)


def speed_text(fps=None, speed=None):
    """La velocidad de ffmpeg, en las dos unidades que existen.

    ffmpeg manda ``fps`` y ``speed`` (multiplicador de tiempo real) y los dos
    son verdad, pero en un render corto ``fps`` sale 0 porque ffmpeg no ha
    tenido tiempo de calcular un promedio. En ese caso se enseña el
    multiplicador, que sí está. Con los dos, los dos: "48.2 fps · 36.1x"
    dice más que cualquiera de los dos solo.
    """
    partes = []
    try:
        if fps is not None and float(fps) > 0:
            partes.append(format_fps(fps))
    except (TypeError, ValueError):
        pass
    try:
        if speed is not None and float(speed) > 0:
            partes.append("%.1fx" % float(speed))
    except (TypeError, ValueError):
        pass
    return "  ·  ".join(partes) if partes else None


def rend_text(fps=None):
    """La velocidad del render de IA: solo su propio fps.

    Deliberadamente **otra** función y no ``speed_text`` con un ``speed`` de
    ``False``: el render de IA va a tantos frames por segundo y punto, no tiene
    multiplicador contra el tiempo del archivo. Mantenerlas separadas es lo
    que impide que las dos velocidades vuelvan a la misma casilla, que es como
    se llegó a comparar un x2 contra un x3 creyendo que la GPU tardaba el doble
    en uno que en el otro.
    """
    try:
        if fps is not None and float(fps) > 0:
            return format_fps(fps)
    except (TypeError, ValueError):
        pass
    return None


def metric_lines(metrics):
    """Las líneas de rendimiento, ya escritas. Oculta lo que no se sabe.

    Recibe el dict de ``LiveMetrics.sample()`` y devuelve sólo las líneas que
    tienen un dato detrás. Una lista vacía significa "no hay nada que
    decir", que el widget pinta como ``—``.
    """
    if not isinstance(metrics, dict):
        return []
    lineas = []
    if metrics.get("gpu_percent") is not None:
        texto = "GPU  %s" % format_percent(metrics.get("gpu_percent"))
        nombre = str(metrics.get("gpu_name") or "").strip()
        if nombre:
            texto += "  ·  %s" % nombre
        lineas.append(texto)
    if metrics.get("cpu_percent") is not None:
        lineas.append("CPU  %s" % format_percent(metrics.get("cpu_percent")))
    vram = format_vram(metrics.get("vram_used_mb"), metrics.get("vram_total_mb"))
    if vram != "—":
        lineas.append("VRAM  %s" % vram)
    return lineas


def queue_text(done, total):
    """``(3, 7) -> "3 de 7 en cola"``. ``None`` si no hay cola en marcha."""
    try:
        done = int(done)
        total = int(total)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    if done <= 0:
        return "1 de %d en cola" % total
    return "%d de %d en cola" % (done + 1, total)


class TrayDashboard(QWidget):
    """El desplegable en sí: sólo pinta el estado que le pasa la ventana."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(ANCHO)
        # Los números no son botones: que el ratón los atraviese sin pulsar
        # nada y sin cerrar el menú.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(6)

        self._title = QLabel("En reposo")
        self._title.setStyleSheet(
            "color: #e0e0e0; font-size: 12px; font-weight: bold;")
        self._title.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Preferred)
        self._layout.addWidget(self._title)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(
            "QProgressBar { background-color: #1f2c4c; border: none;"
            " border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #00d2c3;"
            " border-radius: 3px; }")
        self._layout.addWidget(self._bar)

        self._sub = QLabel("")
        self._sub.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        self._sub.setSizePolicy(QSizePolicy.Policy.Ignored,
                                QSizePolicy.Policy.Preferred)
        self._layout.addWidget(self._sub)

        self._spacer = QWidget()
        self._spacer.setFixedHeight(4)
        self._layout.addWidget(self._spacer)

        self._metrics = QLabel("")
        self._metrics.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        self._layout.addWidget(self._metrics)

    # -- estado -----------------------------------------------------------

    def set_idle(self):
        """Sin render en marcha: una línea y nada más.

        Los números de CPU y GPU en reposo son ruido, y además
        ralentizarían un temporizador que no tiene nada que mirar.
        """
        self._title.setText("En reposo")
        self._bar.setValue(0)
        self._sub.setText("")
        self._metrics.setText("")

    def set_state(self, state):
        """Pinta un estado. ``state`` es el dict que compone la ventana.

        Las claves que no importan se ignoran: quien llama puede pasar el
        mismo dict que lleva a otras partes sin recortarlo antes.
        """
        if not isinstance(state, dict) or not state.get("active"):
            self.set_idle()
            return

        name = str(state.get("task_name") or "").strip()
        self._title.setText(name or "Render en curso")

        try:
            pct = int(state.get("pct"))
        except (TypeError, ValueError):
            pct = 0
        pct = max(0, min(100, pct))
        self._bar.setValue(pct)

        trozos = ["%d %%" % pct]
        eta = state.get("eta")
        if eta:
            trozos.append(str(eta))
        cola = queue_text(state.get("queue_done"), state.get("queue_total"))
        if cola:
            trozos.append(cola)
        self._sub.setText("  ·  ".join(trozos))

        lineas = []
        rend = rend_text(state.get("ai_fps"))
        if rend:
            lineas.append("REND  %s" % rend)
        enc = speed_text(state.get("enc_fps"), state.get("enc_speed"))
        if enc:
            lineas.append("ENC   %s" % enc)
        lineas.extend(metric_lines(state.get("metrics")))
        self._metrics.setText("\n".join(lineas))

    # -- lectura para los tests ------------------------------------------

    def texts(self):
        """Las cuatro etiquetas, para poder afirmar sobre ellas sin mirarlas."""
        return {
            "title": self._title.text(),
            "sub": self._sub.text(),
            "metrics": self._metrics.text(),
            "pct": self._bar.value(),
        }


class TrayController(QObject):
    """La bandeja entera: menú, muestreo y canal de métricas.

    No sabe nada de la ventana. Le dicen "empieza un render", "va por el
    40 %", "el worker mandó estas cifras" y "el render terminó", y compone
    el estado del desplegable. Quien la usa decide cuándo; aquí sólo se
    ocupa de que lo que se ve sea verdad.
    """

    # Claves que el canal de métricas puede traer. Se copian tal cual al
    # estado; lo que no venga, se queda como estaba. Los dos prefijos van
    # porque las cifras las escriben dos fuentes distintas —el bucle de render
    # de IA y ffmpeg— y antes compartían nombre: la que hablaba último pisaba
    # a la otra y el número que se veía en el desplegable no sabía de quién
    # era. Con ``ai_`` y ``enc_`` conviven y se enseñan por separado.
    CAMPOS_METRICAS = (
        "ai_fps", "ai_frame", "ai_total_frames",
        "enc_fps", "enc_speed", "enc_frame", "enc_out_time",
    )

    def __init__(self, parent=None, on_open=None, on_quit=None):
        super().__init__(parent)
        self.dashboard = TrayDashboard()
        self.dashboard.set_idle()

        self._menu = QMenu()
        # El widget va SIN padre a propósito: ``setDefaultWidget`` lo da como
        # propiedad del action, y si además fuese hijo del menú lo tendrían
        # los dos y al destruirse se liberaría dos veces.
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(self.dashboard)
        action.setText("")
        action.setToolTip("")
        self._menu.addAction(action)
        self._menu.addSeparator()
        accion_abrir = self._menu.addAction("Open DRAGGY ENCODER")
        accion_abrir.triggered.connect(on_open or (lambda: None))
        accion_salir = self._menu.addAction("Quit DRAGGY ENCODER")
        accion_salir.triggered.connect(on_quit or (lambda: None))

        self._metrics = None
        self._timer = None
        self._state = {}

    # -- el menu ----------------------------------------------------------

    def menu(self):
        """El ``QMenu`` que hay que pasar a ``QSystemTrayIcon.setContextMenu``."""
        return self._menu

    def acciones(self):
        """Los textos del menú, para poder afirmar sobre ellos."""
        return [accion.text() for accion in self._menu.actions()]

    # -- el render --------------------------------------------------------

    def start(self):
        """Arranca el muestreo. Solo cuando hay un render en marcha.

        ``LiveMetrics`` se crea aquí y no antes: ``nvmlInit()`` abre un handle
        contra el driver, y no tiene sentido tenerlo abierto en un equipo que
        no está renderizando.
        """
        if self._metrics is None:
            try:
                from src.hardware.metrics import LiveMetrics

                self._metrics = LiveMetrics()
            except Exception:
                self._metrics = None
        if self._metrics is None:
            # Sin muestreo no hay desplegable con cifras, pero el resto
            # —progreso, tarea, cola— sigue siendo cierto y se enseña.
            self._state["active"] = True
            self._refresh()
            return
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(1000)
            self._timer.timeout.connect(self._sample)
        if not self._timer.isActive():
            self._timer.start()
        self._state["active"] = True
        self._refresh()

    def stop(self):
        """Para el muestreo y suelta NVML. Idempotente.

        Se llama al terminar el render **y** al cerrar la ventana. Sin esto,
        el handle de NVML sobrevive al proceso y el temporizador sigue
        despertando a una aplicación que ya no existe.
        """
        if self._timer is not None:
            self._timer.stop()
        self._state = {}
        self.dashboard.set_idle()
        if self._metrics is not None:
            try:
                self._metrics.shutdown()
            except Exception:
                pass
            self._metrics = None

    def running(self):
        """¿Está el temporizador vivo? Para los tests y para el cierre."""
        return self._timer is not None and self._timer.isActive()

    # -- lo que llega -----------------------------------------------------

    # Los tres setters repintan. Suena obvio, pero dejarlos solo escribiendo
    # en el estado hacia que el nombre del vídeo y la cola solo se vieran
    # cuando llegara el siguiente tic del temporizador: el desplegable
    # enseñaba el render anterior hasta un segundo después de empezar el
    # nuevo.
    def set_task(self, name):
        self._state["task_name"] = str(name or "")
        self._refresh()

    def set_queue(self, done, total):
        self._state["queue_done"] = done
        self._state["queue_total"] = total
        self._refresh()

    def update_progress(self, pct, eta=None):
        self._state["pct"] = pct
        self._state["eta"] = eta
        self._refresh()

    def update_metrics(self, ev):
        """El canal del worker. Un dict plano que se copia tal cual.

        Viene de dos Productores que ya no se pisan: el bucle de render de IA
        manda ``ai_fps``/``ai_frame``/``ai_total_frames`` y ffmpeg manda
        ``enc_fps``/``enc_speed``/``enc_frame``/``enc_out_time``. Cada uno rellena
        su lado del estado y el desplegable enseña los dos, cada uno en su
        línea.
        """
        if not isinstance(ev, dict):
            return
        for clave in self.CAMPOS_METRICAS:
            if clave in ev:
                self._state[clave] = ev[clave]
        self._refresh()

    def _sample(self):
        if self._metrics is None:
            return
        try:
            self._state["metrics"] = self._metrics.sample()
        except Exception:
            self._state["metrics"] = {}
        self._refresh()

    def _refresh(self):
        state = dict(self._state)
        state.setdefault("task_name", "")
        state.setdefault("pct", 0)
        state.setdefault("eta", None)
        state.setdefault("ai_fps", None)
        state.setdefault("enc_fps", None)
        state.setdefault("enc_speed", None)
        state.setdefault("metrics", {})
        try:
            self.dashboard.set_state(state)
        except Exception:
            pass

    # -- lectura para los tests -------------------------------------------

    def texts(self):
        return self.dashboard.texts()
