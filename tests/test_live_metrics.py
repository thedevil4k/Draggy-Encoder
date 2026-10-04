"""Lo que la máquina está haciendo ahora: que se mida, y que no mienta.

Dos cosas se prueban aquí y las dos importan más de lo que parece.

**La primera es que no se confunde "no lo sé" con "cero".** Es tentador
devolver ``0`` cuando NVML no está, y el desplegable pondría "GPU 0 %" en
un equipo sin NVIDIA: una afirmación falsa, y de las que la gente se cree.
Aquí la clave **no está**, y quien la pinta decide qué hacer con la ausencia.

**La segunda es que la primera muestra de CPU no se enseña.** Un
``psutil.cpu_percent()`` da el uso *medio desde la última llamada*; la
primera llamada no promedia nada y devuelve 0.0 siempre. Un desplegable que
dijera "CPU 0 %" en su primer segundo estaría mintiendo con mucha seguridad,
así que ese primer ``sample()`` devuelve ``None`` y el widget enseña «—».

Las funciones de formateo se prueban aparte del muestreo porque son puras:
es lo que permite comprobar todos los casos raros (sin GPU, sin VRAM, sin
velocidad) sin una GPU delante.
"""

import pytest

from src.hardware import metrics as metrics_mod
from src.hardware.metrics import (
    LiveMetrics,
    format_fps,
    format_percent,
    format_vram,
)


# ---------------------------------------------------------------------------
# El cero que no existe
# ---------------------------------------------------------------------------

def test_sin_nvml_no_hay_linea_de_gpu_y_no_hay_que_inventarse_un_cero(monkeypatch):
    """Un equipo sin NVIDIA no enseña "GPU 0 %": no enseña línea.

    Este es el caso que más se va a dar en la vida real, porque la app se
    instala en máquinas sin NVIDIA. Lo que no puede pasar es que el
    desplegable afirme un 0 % que nadie ha medido.
    """
    monkeypatch.setattr(metrics_mod, "pynvml", None)
    live = LiveMetrics()
    try:
        assert live.gpu_available is False
        assert live.gpu_name == ""
        muestra = live.sample()
        assert "gpu_percent" in muestra
        assert muestra["gpu_percent"] is None
        assert muestra["vram_used_mb"] is None
    finally:
        live.shutdown()


def test_nvml_que_falla_al_arrancar_deja_la_app_sin_gpu_pero_viva(monkeypatch):
    """Si ``nvmlInit()`` revienta, esto se apaga solo y no tumba nada.

    El caso real es un driver que no está, o un NVML en otro sitio del
    disco. Lo que no puede ser es que un fallo al abrir la librería de
    métricas tumbe un render que ya estaba en marcha.
    """
    class _Roto:
        @staticmethod
        def nvmlInit():
            raise OSError("NVML shared library not found")

    monkeypatch.setattr(metrics_mod, "pynvml", _Roto)
    live = LiveMetrics()
    try:
        assert live.gpu_available is False
        # Y aun asi mide CPU, que no depende de NVIDIA.
        assert live.sample() is not None
    finally:
        live.shutdown()


def test_una_gpu_que_se_reinicia_apaga_la_linea_y_no_sigue_dando_ceros(monkeypatch):
    """Si NVML falla a mitad de sesión, se deja de preguntar.

    Reintentar en bucle sólo añade ruido, y un reintento que vuelve a fallar
    cada segundo es un temporizador despertando para nada. Se marca como no
    disponible y el desplegable deja de enseñar la línea.
    """
    class _NVMLQueMuere:
        def __init__(self):
            self.llamadas = 0

        def nvmlInit(self):
            return None

        def nvmlDeviceGetCount(self):
            return 1

        def nvmlDeviceGetHandleByIndex(self, index):
            return object()

        def nvmlDeviceGetName(self, handle):
            return b"RTX 3060"

        def nvmlDeviceGetUtilizationRates(self, handle):
            self.llamadas += 1
            if self.llamadas > 1:
                raise RuntimeError("GPU fell off the bus")
            return type("R", (), {"gpu": 42})()

        def nvmlDeviceGetMemoryInfo(self, handle):
            return type("M", (), {"used": 1024 ** 3, "total": 6 * 1024 ** 3})()

        def nvmlShutdown(self):
            return None

    fake = _NVMLQueMuere()
    monkeypatch.setattr(metrics_mod, "pynvml", fake)
    live = LiveMetrics()
    try:
        assert live.gpu_available is True
        assert live.sample()["gpu_percent"] == 42
        # El segundo muestreo revienta por dentro y se apaga sin propagar.
        live.sample()
        assert live.gpu_available is False
        assert live.sample()["gpu_percent"] is None
    finally:
        live.shutdown()


# ---------------------------------------------------------------------------
# La primera muestra
# ---------------------------------------------------------------------------

def test_la_primera_muestra_de_cpu_no_dice_cero_dice_que_no_sabe():
    """Sin intervalo que promediar no hay dato, y ``None`` lo dice."""
    live = LiveMetrics(enable_gpu=False)
    try:
        primera = live.sample()
        assert primera["cpu_percent"] is None
        # Y las siguientes si son numero: solo falta la primera.
        assert isinstance(live.sample()["cpu_percent"], float)
    finally:
        live.shutdown()


def test_sin_gpu_no_se_pregunta_nunca_por_la_gpu():
    """``enable_gpu=False`` ni siquiera intenta abrir NVML.

    Existe para los tests y para un futuro modo "solo CPU", pero sobre
    todo para que se pueda comprobar el resto sin una GPU en la maquina.
    """
    live = LiveMetrics(enable_gpu=False)
    try:
        assert live.gpu_available is False
        assert live._handles == []
    finally:
        live.shutdown()


def test_shutdown_es_idempotente_y_no_lanza():
    """Se puede llamar dos veces, y sin NVML inicializado tampoco."""
    live = LiveMetrics(enable_gpu=False)
    live.shutdown()
    live.shutdown()


# ---------------------------------------------------------------------------
# Formateo: lo que se ve en pantalla
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valor,esperado", [
    (0, "0 %"),
    (61.4, "61 %"),
    (61.6, "62 %"),
    (100, "100 %"),
    (None, "—"),
])
def test_un_porcentaje_se_escribe_como_lo_lee_una_persona(valor, esperado):
    assert format_percent(valor) == esperado


def test_un_porcentaje_que_no_es_numero_no_revienta_la_pantalla():
    """Un dato corrupto se enseña como "no sé", no como un fallo."""
    for basura in ("mucho", object(), [], float("nan"), float("inf"),
                   float("-inf")):
        assert format_percent(basura) == "—"


def test_la_vram_va_con_el_total_porque_sola_no_dice_nada():
    """``3.2 GB`` suelto no dice si eso es mucho o poco."""
    assert format_vram(3255, 6144) == "3.2 / 6.0 GB"
    assert format_vram(0, 0) == "0.0 / 0.0 GB"


def test_la_vram_incompleta_no_inventa_la_parte_que_le_falta():
    """Si solo hay un numero, no se rellena el otro con un cero."""
    assert format_vram(3255, None) == "—"
    assert format_vram(None, 6144) == "—"
    assert format_vram(None, None) == "—"


def test_los_fps_redondean_a_un_decimal_que_es_lo_que_se_lee():
    assert format_fps(12.34) == "12.3 fps"
    assert format_fps(147.0) == "147.0 fps"
    assert format_fps(None) == "—"
    assert format_fps("nada") == "—"
