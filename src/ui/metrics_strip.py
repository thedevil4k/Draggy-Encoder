"""La franja de métricas de la ventana principal: GPU, CPU y VRAM en horizontal.

El desplegable de la bandeja del sistema (``src/ui/tray_dashboard.py``) ya
sabe medir esas tres cosas, pero vive en un sitio al que sólo se llega con dos
clicks derechos, y además se duerme en cuanto no hay un render en marcha. Esta
franja es el mismo dato en el sitio donde ya se está mirando la ventana: ancha,
siempre abajo del todo, y responde sin abrir nada a la pregunta que aparece a
mitad de un render largo: ¿está la GPU al 100 % o se está quedando parada?

Reparte el trabajo igual que el resto del proyecto:

* **No mide.** Los números los trae ``src.hardware.metrics.LiveMetrics``, los
  textos los compone ``gauge_items`` y el widget sólo pinta lo que le llega.
* **No inventa.** Un dato que no existe no se pinta como cero. Si la CPU aún no
  tiene un intervalo que promediar, el medidor dice "—", no "0 %": son dos
  afirmaciones distintas y aquí sólo se hace la segunda.
* **No interrumpe.** El muestreo va en un ``QTimer`` de un segundo, que es lo
  que tarda ``psutil`` en devolver un intervalo de CPU que signifique algo.

Comparte con el desplegable una decisión que parece caprichosa y no lo es: **en
reposo la franja se apaga**. Solo mide mientras hay un render en marcha, que es
cuando las cifras están diciendo algo —"la GPU va al 12 %, esto no está
codificando" es un dato, y solo se puede ver durante el render—. Además, un
``LiveMetrics`` vivo con la ventana cerrada es una aplicación que no se cierra,
y un handle de NVML abierto en un equipo que no está codificando es un
problema esperando. Así que ``start``/``stop`` los manda el render, no la
apertura de la ventana, y ``stop()`` devuelve la franja a "—".
"""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QWidget,
)

from src.hardware.metrics import format_percent, format_vram

# Un color por medidor. El de la GPU es el mismo cian del desplegable de la
# bandeja y de los botones, para que la franja parezca parte de la app y no un
# añadido de otra.
COLORES = {
    "gpu": "#00d2c3",
    "cpu": "#4dabf7",
    "vram": "#b197fc",
}

# Alto fijo: la ventana va de 1024 px a lo que dé, y una franja que encoge deja
# de leerse de un vistazo.
ALTO = 46

# Orden de los medidores, de izquierda a derecha.
MEDIDORES = (("gpu", "GPU"), ("cpu", "CPU"), ("vram", "VRAM"))


def _pct(value):
    """Un porcentaje acotado a 0..100, o ``None`` si no se sabe.

    Lo que no es un número finito se trata como "no lo sé" a propósito.
    ``min(100.0, nan)`` da ``100.0`` en Python, así que un filtro que sólo acota
    pinta una barra **llena** al lado de un "—" que dice que no lo sabe. Eso no
    es una cifra redondeada, es una mentira.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(100.0, number))


def _vram_pct(used, total):
    """La VRAM como porcentaje de la total, para que su barra signifique algo."""
    pct = _pct(used)
    try:
        total_f = float(total)
    except (TypeError, ValueError):
        return None
    if pct is None or not math.isfinite(total_f) or total_f <= 0:
        return None
    return _pct(float(used) / total_f * 100.0)


def gpu_name(metrics):
    """El nombre de la tarjeta que está trabajando, o cadena vacía.

    Vive fuera de los medidores y por un motivo concreto: "91 %" no dice si el
    render va sobre la integrada o sobre la dedicada, y ese es justo el dato
    que hace falta entender un 91 % que no cuadra.
    """
    if not isinstance(metrics, dict):
        return ""
    return str(metrics.get("gpu_name") or "").strip()


def gauge_items(metrics):
    """Los tres medidores, siempre los tres y siempre en el mismo orden.

    Cada item lleva ``key``, ``label``, ``value``, ``pct`` y ``known``. Los tres
    huecos existen aunque no haya dato detrás: un panel al que le aparece y
    desaparece una casilla da un tirón en cada muestreo, y lo que se está
    mirando aquí es un número que cambia solo.

    ``known`` es justo lo que separa "0 %" de "no lo sé". Con la CPU, el primer
    muestreo devuelve ``None`` porque no hay todavía intervalo que promediar; con
    la VRAM no hay nada que devolver si no hay NVIDIA. En los dos casos el
    medidor sale con la barra vacía y "—", que es la verdad.
    """
    if not isinstance(metrics, dict):
        metrics = {}
    usados = metrics.get("vram_used_mb")
    total = metrics.get("vram_total_mb")
    vram = format_vram(usados, total)
    gpu_pct = metrics.get("gpu_percent")
    cpu_pct = metrics.get("cpu_percent")
    return [
        {
            "key": "gpu",
            "label": "GPU",
            "value": format_percent(gpu_pct),
            "pct": _pct(gpu_pct),
            "known": gpu_pct is not None,
        },
        {
            "key": "cpu",
            "label": "CPU",
            "value": format_percent(cpu_pct),
            "pct": _pct(cpu_pct),
            "known": cpu_pct is not None,
        },
        {
            "key": "vram",
            "label": "VRAM",
            "value": None if vram == "—" else vram,
            "pct": _vram_pct(usados, total),
            "known": vram != "—",
        },
    ]


class _Gauge(QWidget):
    """Un medidor en una fila: etiqueta, barra y valor.

    La barra es lo único elástico. Las cifras van con ancho fijo porque son
    ellas lo que se lee de un vistazo, y un medidor que se ensancha con el
    texto es un medidor al que hay que volver a buscar con la vista cada vez que
    cambia un dígito.
    """

    ANCHO_ETIQUETA = 38
    ANCHO_VALOR = 84
    # "2.0 / 12.0 GB" mide unos 143 px contra los 84 de "91 %", y una VRAM
    # recortada a "11.5 / 2..." no dice nada. El ancho va por medidor en vez de
    # ensanchar los otros dos: 72 px de los 1024 de la ventana, y sin que la
    # barra de la GPU deje de respirar.
    ANCHO_VALOR_VRAM = 156
    ANCHO_BARRA_MIN = 60

    def __init__(self, key, label, parent=None):
        super().__init__(parent)
        self.key = key
        color = COLORES.get(key, "#00d2c3")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._label = QLabel(label)
        self._label.setFixedWidth(self.ANCHO_ETIQUETA)
        self._label.setStyleSheet(
            "color: %s; font-size: 11px; font-weight: bold;" % color
        )
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setMinimumWidth(self.ANCHO_BARRA_MIN)
        self._bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._bar.setStyleSheet(
            "QProgressBar { background-color: #1f2c4c; border: none;"
            " border-radius: 3px; }"
            "QProgressBar::chunk { background-color: %s; border-radius: 3px; }" % color
        )
        layout.addWidget(self._bar)

        self._value = QLabel("—")
        self._value.setFixedWidth(
            self.ANCHO_VALOR_VRAM if key == "vram" else self.ANCHO_VALOR
        )
        self._value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._value.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        layout.addWidget(self._value)

    def apply(self, item):
        """Pinta un item de ``gauge_items``. Nunca lanza."""
        if not isinstance(item, dict):
            return
        pct = item.get("pct")
        try:
            self._bar.setValue(int(round(pct)) if pct is not None else 0)
        except (TypeError, ValueError):
            self._bar.setValue(0)
        value = item.get("value")
        self._value.setText(str(value) if value else "—")
        self.setToolTip("%s: %s" % (item.get("label") or self._label.text(),
                                    self._value.text()))

    def texts(self):
        """Lo que se ve, para poder afirmar sobre la franja sin mirarla."""
        return {
            "label": self._label.text(),
            "value": self._value.text(),
            "pct": self._bar.value(),
        }


class _ElidedLabel(QLabel):
    """Una etiqueta que recorta por la derecha en vez de cortar el texto.

    ``QLabel`` sola, alchengada por su padre, deja el texto a medias o lo
    desborda: el nombre de una tarjeta no cabe en 200 px y ensancharlo por
    mucho que se estire la ventana tampoco vale, porque se come el hueco de los
    medidores. Guardando el texto entero y re-elidándolo en cada
    ``resizeEvent`` se obtiene lo que se quiere: el nombre visible entero si
    cabe, "NVIDIA G…" si no, y el texto completo disponible para el tooltip.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text):
        """Fija el texto entero y pinta la versión que cabe."""
        self._full = str(text or "")
        self._relabel()

    def full_text(self):
        """El texto sin recortar: lo que hay, no lo que se ve."""
        return self._full

    def _relabel(self):
        ancho = self.width() - 4
        if not self._full or ancho <= 0:
            # Sin ancho todavía (recién construida, sin layout) no se puede
            # decidir cuánto cabe: se enseña entero y el primer resizeEvent
            # recorta lo que sobre.
            super().setText(self._full)
            return
        super().setText(self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, ancho))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relabel()


class MetricsStrip(QWidget):
    """La franja horizontal: GPU, CPU y VRAM, y a la derecha la GPU que es.

    No mide ni decide: recibe un dict de ``LiveMetrics.sample()`` y lo pinta
    con lo que compone ``gauge_items``. Los tres huecos están siempre, haya
    dato o no, para que la franja no baile de un muestreo a otro.
    """

    # La tarjeta que está trabajando: ancho fijo, porque el nombre de una GPU
    # puede ser largo y el resto de la franja no debe depender de cuál sea.
    ANCHO_NOMBRE = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("metricsStrip")
        self.setFixedHeight(ALTO)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "#metricsStrip { background-color: #0d1424;"
            " border-top: 1px solid #1f2c4c; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(24)
        self._gauges = {}
        for key, label in MEDIDORES:
            gauge = _Gauge(key, label)
            self._gauges[key] = gauge
            layout.addWidget(gauge, 1)

        self._name = _ElidedLabel()
        self._name.setFixedWidth(self.ANCHO_NOMBRE)
        self._name.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._name.setStyleSheet("color: #6a6a7a; font-size: 11px;")
        layout.addWidget(self._name)

    def set_sample(self, metrics):
        """Repinta los tres medidores y el nombre con una lectura. Nunca lanza."""
        for item in gauge_items(metrics):
            gauge = self._gauges.get(item.get("key"))
            if gauge is not None:
                gauge.apply(item)
        nombre = gpu_name(metrics)
        self._name.set_full_text(nombre)
        self._name.setToolTip(nombre)

    def set_idle(self):
        """Sin render en marcha: los tres a "—" y la barra vacía.

        Pasa por ``set_sample({})`` a propósito: el camino que pinta "no lo sé"
        es el mismo que se usa cuando falta un dato, y no una versión aparte
        que se pueda quedar sin sincronizar.
        """
        self.set_sample({})

    def texts(self):
        """Lo que se ve, medidor por medidor, para los tests."""
        out = {key: gauge.texts() for key, gauge in self._gauges.items()}
        out["nombre"] = self._name.text()
        out["nombre_completo"] = self._name.full_text()
        return out


class MetricsStripController(QObject):
    """Muestrea una vez por segundo y mantiene la franja al día.

    Vive mientras el render, no mientras la ventana: el mismo interruptor que
    arranca y para el desplegable de la bandeja. Lo que comparte con él es no
    abrir NVML antes de tiempo: ``LiveMetrics`` se crea en el **primer** tic, no
    al construir la ventana, para que abrir la app no se lleve por delante la
    inicialización del driver.
    """

    def __init__(self, parent=None, interval_ms=1000):
        super().__init__(parent)
        self.strip = MetricsStrip()
        self._interval_ms = int(interval_ms)
        self._timer = None
        self._metrics = None

    def widget(self):
        """El widget a poner en la ventana."""
        return self.strip

    def start(self):
        """Arranca el muestreo. Idempotente."""
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(self._interval_ms)
            self._timer.timeout.connect(self.sample_now)
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        """Para el muestreo, suelta NVML y deja la franja en reposo. Idempotente.

        Sin esto, un ``QTimer`` vivo es una aplicación que no se cierra y un
        handle de NVML sobrevive al proceso.
        """
        if self._timer is not None:
            self._timer.stop()
        if self._metrics is not None:
            try:
                self._metrics.shutdown()
            except Exception:
                pass
            self._metrics = None
        self.strip.set_idle()

    def running(self):
        """¿Vivo el temporizador? Para los tests y para el cierre."""
        return self._timer is not None and self._timer.isActive()

    def sample_now(self):
        """Un muestreo, ya sea por el temporizador o a mano. Nunca lanza.

        Un fallo al medir no puede tumbar la franja ni, mucho menos, la app:
        en el peor de los casos la ventana se queda con la última lectura que
        sí era verdad.
        """
        if self._metrics is None:
            try:
                from src.hardware.metrics import LiveMetrics as _LiveMetrics

                self._metrics = _LiveMetrics()
            except Exception:
                self._metrics = None
        if self._metrics is None:
            return
        try:
            sample = self._metrics.sample()
        except Exception:
            return
        try:
            self.strip.set_sample(sample)
        except Exception:
            pass