"""Lo que la máquina está haciendo AHORA, no lo que tiene puesto.

``src.hardware.detector`` contesta a una pregunta distinta: *qué GPU y qué
driver tiene este equipo*. Eso se mira una vez, al abrir el panel. Aquí la
pregunta es otra —*cuánta CPU está gastando la app en este segundo, y qué
está haciendo la GPU*— y se hace mal si se responde una sola vez: un
``psutil.cpu_percent()`` da el uso **medio desde la última llamada**, así que
preguntar de tarde en tarde da números que no significan nada.

Por eso esto es un muestreador con estado: remembers la llamada anterior y
devuelve el intervalo. Es lo que necesita el desplegable de la bandeja del
sistema, que se refresca cada segundo mientras hay un render en marcha.

Dos reglas que este módulo cumple sí o sí, porque las otras dos están en
pantalla y una pantalla que miente es peor que una que no mide:

* **Nunca lanza.** Le da igual que NVML no esté, que la GPU se haya
  reiniciado a mitad de un render o que ``psutil`` devuelva algo raro: un
  error al medir el uso de la GPU no puede tumbar un render. Devuelve lo que
  haya podido leer y fuera.
* **Un dato que no existe no se enseña como cero.** Si no hay NVIDIA, la
  clave ``gpu_percent`` no está, y quien pinta no escribe "0 %": escribe
  otra cosa, porque "0 %" y "no lo sé" son afirmaciones distintas y sólo una
  es verdad.

``psutil`` es dependencia de la app de toda la vida. ``pynvml`` no: se
instala con ``nvidia-ml-py`` (el paquete de NVIDIA; el ``pynvml`` que hay en
PyPI está congelado y obsoleto, pero **se importa igual** con el nombre
``pynvml``). Si no está, esto sigue funcionando y sólo pierde la GPU, que
es justo lo que tiene que pasar en un equipo sin NVIDIA.
"""

from __future__ import annotations

try:
    import pynvml
except Exception:  # pragma: no cover - depende de la maquina
    # Sin la libreria de NVML no hay lectura de GPU. No es un error: es un
    # equipo sin NVIDIA, o una instalacion que no la trajo.
    pynvml = None

import psutil


def _mb(bytes_):
    """Bytes a MiB, o ``None`` si el valor no sirve."""
    try:
        return int(bytes_) // (1024 * 1024)
    except (TypeError, ValueError):
        return None


class LiveMetrics:
    """Muestrea CPU y GPU cada vez que se le pregunta.

    No tiene hilo propio: quien la usa es un ``QTimer`` de un segundo, y un
    hilo que se despierta solo para mirar un número que ya se sabe va a
    terminar despedido. Lo que sí hace es recordar la última lectura, que es
    lo que da sentido a la anterior.
    """

    def __init__(self, enable_gpu: bool = True):
        self.gpu_available = False
        self.gpu_name = ""
        self._handles = []
        # Muestras de CPU servidas. La de "__init__" no cuenta: es la que
        # abre el intervalo, y un intervalo sin nada que promediar no es un
        # cero, es una falta de datos.
        self._cpu_samples = 0
        if enable_gpu:
            self._init_gpu()
        # La primera llamada a ``cpu_percent(interval=None)`` devuelve SIEMPRE
        # 0.0, porque no hay intervalo que promediar. Si no se primase aquí,
        # el primer muestreo del desplegable diría "0 %" con la aplicación
        # usando la CPU en absoluto.
        self._prime_cpu()

    # -- arranque ---------------------------------------------------------

    def _init_gpu(self):
        """Abre NVML y guarda un handle por GPU. Nunca lanza."""
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                self._handles.append((handle, str(name)))
            self.gpu_available = bool(self._handles)
            if self._handles:
                self.gpu_name = self._handles[0][1]
        except Exception:
            # Sin driver, sin NVML en este equipo, o una que no entiende: se
            # apaga la GPU y el resto sigue.
            self._handles = []
            self.gpu_available = False
            self.gpu_name = ""

    def _prime_cpu(self):
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    # -- muestreo ---------------------------------------------------------

    def sample(self):
        """Una lectura. Devuelve un dict; nunca lanza.

        ``cpu_percent`` es ``None`` en la **primera** lectura después de
        arrancar, y también si ``psutil`` falla. No es un detalle: la primera
        lectura promedia el intervalo desde el arranque, que puede ser de
        medio segundo y daría un número que no representa nada. Preferimos no
        tener número a tener uno falso.
        """
        out = {
            "cpu_percent": None,
            "gpu_percent": None,
            "gpu_name": self.gpu_name,
            "vram_used_mb": None,
            "vram_total_mb": None,
        }
        try:
            value = psutil.cpu_percent(interval=None)
            self._cpu_samples += 1
            if self._cpu_samples > 1:
                out["cpu_percent"] = float(value)
        except Exception:
            pass
        if self.gpu_available:
            self._read_gpu(out)
        return out

    def _read_gpu(self, out):
        """Coge la GPU más cargada. Si NVML falla, se apaga y no se reintenta.

        Un fallo de NVML a mitad de sesión casi siempre significa que la GPU
        se ha reiniciado o que el driver se ha actualizado. Reintentar en
        bucle sólo añade ruido; se marca como no disponible y el desplegable
        deja de enseñar la línea en vez de enseñar un cero.
        """
        best = None
        for handle, name in self._handles:
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util = int(rates.gpu)
                if best is None or util > best[0]:
                    best = (util, name, handle)
            except Exception:
                self._disable_gpu()
                return
        if best is None:
            return
        util, name, handle = best
        out["gpu_percent"] = util
        out["gpu_name"] = name
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out["vram_used_mb"] = _mb(memory.used)
            out["vram_total_mb"] = _mb(memory.total)
        except Exception:
            # La utilisation y la memoria son cosas distintas: que una falle
            # no quita la otra.
            pass

    def _disable_gpu(self):
        self._handles = []
        self.gpu_available = False
        self.gpu_name = ""

    # -- cierre -----------------------------------------------------------

    def shutdown(self):
        """Suelta NVML. Sin esto, el handle sobrevive al cierre de la app."""
        if pynvml is not None and self._handles:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._handles = []


def format_percent(value):
    """``61.4 -> "61 %"``, ``None -> "—"``. Para que el cero falso no exista."""
    if value is None:
        return "—"
    try:
        return "%d %%" % int(round(float(value)))
    except (TypeError, ValueError):
        return "—"


def format_fps(value):
    """``12.34 -> "12.3 fps"``, ``None -> "—"``."""
    if value is None:
        return "—"
    try:
        return "%.1f fps" % float(value)
    except (TypeError, ValueError):
        return "—"


def format_vram(used_mb, total_mb):
    """``(3255, 6144) -> "3.2 / 6.0 GB"``, si falta algo -> ``"—"``.

    La VRAM es el número que en una tarjeta de 6 GB decide si un upscale cabe
    o no, así que vale la pena enseñar el usado y el total juntos: un "3.2 GB"
    suelto no dice si eso es mucho o poco.
    """
    if used_mb is None or total_mb is None:
        return "—"
    try:
        return "%.1f / %.1f GB" % (used_mb / 1024.0, total_mb / 1024.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"
