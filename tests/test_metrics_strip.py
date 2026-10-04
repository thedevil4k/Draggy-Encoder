"""La franja de métricas del borde inferior: qué enseña y, sobre todo, cuándo.

Es la misma exigencia que el desplegable de la bandeja, trasladada al sitio
donde ya se está mirando: **no puede mentir**. Si la CPU todavía no tiene un
intervalo que promediar, no puede decir que el equipo está parado; si no hay
NVIDIA, no puede inventar un "0 %" de GPU. Aquí se comprueba que los datos que
no existen no se convierten en ceros, y que los que existen se escriben con una
sola forma.

Y se comprueba la decisión que hace que esta franja sea distinta del
desplegable: **solo mide mientras hay un render en marcha**. No es un capricho
estético, son dos razones a la vez. Una: en reposo las cifras no significan
nada —"GPU al 4 %" con la app cerrada es ruido—. Dos, y esta es la que de
verdad importa: un ``LiveMetrics`` vivo con la ventana cerrada es una aplicación
que no se cierra, y un handle de NVML abierto en un equipo que no codifica es un
problema esperando. Por eso el temporizador nace con el render y muere con él,
y por eso al parar la franja vuelve a "—" en vez de quedarse congelada en el
último número que se leyó.

El cableado en la ventana se comprueba mirando su código y no probando la
ventana entera, que ni se puede montar en un test sin display: que los cinco
caminos de ``closeEvent`` pasen por el helper que para la franja, y que el
widget se añada al layout raíz. Un texto aquí no probaría nada.
"""

import inspect

import pytest

from src.hardware import metrics as metrics_mod
from src.ui.metrics_strip import (
    ALTO,
    MEDIDORES,
    MetricsStrip,
    MetricsStripController,
    _ElidedLabel,
    _Gauge,
    gpu_name,
    gauge_items,
)

_ANCHO_VALOR = _Gauge.ANCHO_VALOR
_ANCHO_VALOR_VRAM = _Gauge.ANCHO_VALOR_VRAM


def _muestra(**extra):
    base = {
        "cpu_percent": 11.0,
        "gpu_percent": 91.0,
        "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
        "vram_used_mb": 2048,
        "vram_total_mb": 12288,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Las funciones puras
# ---------------------------------------------------------------------------


def test_los_tres_medidores_siempre_en_el_mismo_orden():
    """La franja no puede cambiar de forma entre muestreos.

    Es lo que permite leerla de un vistazo: si un tic "?GPU/CPU" "?CPU/GPU",
    el ojo tiene que volver a buscar la etiqueta cada vez que se mueve un
    dígito.
    """
    assert MEDIDORES == (("gpu", "GPU"), ("cpu", "CPU"), ("vram", "VRAM"))
    assert [item["key"] for item in gauge_items(_muestra())] == [
        "gpu", "cpu", "vram"]


def test_una_lectura_entera_pinta_los_tres_medidores():
    items = {item["key"]: item for item in gauge_items(_muestra())}
    assert items["gpu"]["value"] == "91 %"
    assert items["gpu"]["pct"] == 91.0
    assert items["cpu"]["value"] == "11 %"
    assert items["vram"]["value"] == "2.0 / 12.0 GB"
    assert round(items["vram"]["pct"], 1) == 16.7
    assert all(item["known"] for item in items.values())


@pytest.mark.parametrize("vacio", [{}, None, "basura", 42, []])
def test_un_dato_que_no_existe_no_se_pinta_como_cero(vacio):
    """"—" y barra vacía, que es distinto de "0 %".

    Con la CPU, el primer muestreo devuelve ``None`` porque no hay todavía
    intervalo que promediar. Pintar un 0 ahí es una afirmación —"el equipo está
    parado"— que el muestreo no puede sostener, y en mitad de un render es la
    afirmación que más caro sale: hace parecer que el motor se ha colgado.
    """
    for item in gauge_items(vacio):
        assert item["value"] in (None, "—"), item
        assert item["pct"] is None, item
        assert item["known"] is False


def test_sin_total_de_vram_no_hay_barra_que_señale_a_nada():
    """El porcentaje de la VRAM necesita el total; sin él, nada.

    Se sabe cuánto ocupa y no cuánto cabe: eso sirve para un texto, no para una
    barra. Un 50 % inventado al lado de una barra llena sería justo la clase de
    cifra que hace que dejen de fiarse del panel.
    """
    item = {i["key"]: i for i in gauge_items(
        {"vram_used_mb": 2048, "vram_total_mb": None})}["vram"]
    assert item["value"] in (None, "—")
    assert item["pct"] is None
    assert item["known"] is False


def test_un_total_de_vram_cero_no_divide_por_cero():
    """Hay hardware que reporta 0 en el total de VRAM, y eso no es un 0 %."""
    item = {i["key"]: i for i in gauge_items(
        {"vram_used_mb": 512, "vram_total_mb": 0})}["vram"]
    assert item["pct"] is None


@pytest.mark.parametrize("valor,esperado", [
    (-40.0, 0.0),
    (0.0, 0.0),
    (100.0, 100.0),
    (4300.0, 100.0),
    (None, None),
    ("basura", None),
    (float("nan"), None),
    (float("inf"), None),
    (float("-inf"), None),
])
def test_un_porcentaje_se_acota_y_un_nan_no_sale_como_barra_llena(valor, esperado):
    """La barra vive entre 0 y 100, y un ``NaN`` no se dibuja al 100.

    ``min(100.0, nan)`` da ``100.0`` en Python, así que un filtro que sólo acota
    pinta una barra **llena** al lado de un "—" que dice que no lo sabe. Es la
    peor versión posible del fallo: no es que falte el dato, es que se enseña un
    dato falso y muy visible. Los infinitos son el mismo caso: ni el 0 ni el
    100 son verdad, así que no hay nada que dibujar.
    """
    items = {i["key"]: i for i in gauge_items(
        {"gpu_percent": valor, "cpu_percent": valor})}
    assert items["gpu"]["pct"] == esperado
    assert items["cpu"]["pct"] == esperado


def test_el_nombre_de_la_gpu_aparece_una_sola_vez():
    """El nombre va fuera de los medidores, y sin inventarse ninguno.

    "91 %" no dice si el render va sobre la integrada o sobre la dedicada, y
    ese es justo el dato que hace falta para entender un 91 % que no cuadra.
    """
    assert gpu_name(_muestra()) == "NVIDIA GeForce RTX 3060 Laptop GPU"
    for item in gauge_items(_muestra()):
        assert "detail" not in item, (
            "el detalle volvio al medidor; el nombre va una sola vez, a la "
            "derecha, para que el ancho de las cifras no dependa de la tarjeta")
    assert gpu_name({"gpu_name": "   "}) == ""
    assert gpu_name(None) == ""


# ---------------------------------------------------------------------------
# El widget
# ---------------------------------------------------------------------------


def test_la_franja_pinta_lo_que_le_llega(qapp):
    strip = MetricsStrip()
    try:
        strip.resize(1024, ALTO)
        strip.show()
        qapp.processEvents()
        strip.set_sample(_muestra())
        textos = strip.texts()
        assert textos["gpu"]["value"] == "91 %"
        assert textos["gpu"]["pct"] == 91
        assert textos["cpu"]["value"] == "11 %"
        assert textos["vram"]["value"] == "2.0 / 12.0 GB"
        assert textos["nombre_completo"] == "NVIDIA GeForce RTX 3060 Laptop GPU"
        assert textos["nombre"].startswith("NVIDIA GeForce")
    finally:
        strip.deleteLater()


def test_en_reposo_la_franja_no_dice_ninguna_cifra(qapp):
    """``set_idle`` es lo que ve la ventana entre render y render.

    Y va por el mismo camino que "no lo sé" a propósito: si el reposo tuviera
    su propia versión del pintado, bastaría con tocar una para que los dos
    caminos dejaran de decir lo mismo.
    """
    strip = MetricsStrip()
    try:
        strip.set_sample(_muestra())
        strip.set_idle()
        textos = strip.texts()
        for clave in ("gpu", "cpu", "vram"):
            assert textos[clave]["value"] == "—", clave
            assert textos[clave]["pct"] == 0, clave
        assert textos["nombre"] == ""
        assert textos["nombre_completo"] == ""
    finally:
        strip.deleteLater()


def test_la_franja_no_tiene_altura_variable(qapp):
    """46 px fijos, para que sea una franja y no una línea más del layout."""
    strip = MetricsStrip()
    try:
        assert ALTO == 46
        assert strip.minimumHeight() == ALTO
        assert strip.maximumHeight() == ALTO
    finally:
        strip.deleteLater()


def test_el_nombre_se_recorta_en_lugar_de_desbordar(qapp):
    """A 200 px se ve "NVIDIA G…" y el texto entero sigue disponible.

    El nombre de una tarjeta no cabe en 200 px y ensancharlo por mucho que se
    estire la ventana tampoco vale: se come el hueco de los medidores. Lo que se
    quiere es el texto entero en el tooltip y una elipsis en pantalla.
    """
    label = _ElidedLabel()
    try:
        label.setStyleSheet("font-size: 11px;")
        label.resize(120, 20)
        label.show()
        qapp.processEvents()
        label.set_full_text("NVIDIA GeForce RTX 3060 Laptop GPU")
        assert label.full_text() == "NVIDIA GeForce RTX 3060 Laptop GPU"
        assert label.text() != label.full_text()
        assert label.text().endswith("…")

        label.resize(400, 20)
        qapp.processEvents()
        assert label.text() == "NVIDIA GeForce RTX 3060 Laptop GPU"

        label.set_full_text("")
        assert label.text() == ""
        assert label.full_text() == ""
    finally:
        label.deleteLater()


def test_la_franja_cabe_en_la_ventana_minima(qapp):
    """Los 1024 px de ancho mínimo de la ventana tienen que bastar.

    La franja va al layout raíz y no dentro de una columna: si su ancho mínimo
    no cupiera junto al splitter, la ventana no podría ni llegar a su mínimo y
    el tirón aparecería con cada resize.
    """
    strip = MetricsStrip()
    try:
        minimo = strip.layout().minimumSize().width()
        assert minimo <= 1024, (
            f"la franja necesita {minimo} px y la ventana baja a 1024: el "
            "layout raíz no podría montar el splitter y la franja a la vez")
    finally:
        strip.deleteLater()


def test_la_vram_tiene_mas_hueco_que_un_porcentaje(qapp):
    """"2.0 / 12.0 GB" necesita mucho más que "91 %", y por eso lo tiene.

    Medido en el equipo de desarrollo, "11.5 / 23.6 GB" ocupa unos 154 px con
    la fuente de la app, contra los 84 px de un porcentaje. Con un único ancho
    para los tres, la VRAM salía cortada por el medio —justo el dato que
    decide si un upscale cabe en la tarjeta—. Se ensancha solo la VRAM: 72 px de
    los 1024 de la ventana, y las barras siguen teniendo sitio.
    """
    strip = MetricsStrip()
    try:
        ancho_vram = strip._gauges["vram"]._value.width()
        ancho_pct = strip._gauges["gpu"]._value.width()
        assert ancho_vram > ancho_pct, (
            "la VRAM comparte ancho con un porcentaje: se va a recortar")
        assert ancho_pct == _ANCHO_VALOR
        assert ancho_vram == _ANCHO_VALOR_VRAM
    finally:
        strip.deleteLater()


def test_una_basura_no_deja_la_franja_a_medias(qapp):
    """Una lectura corrupta deja las tres filas en "—" y ninguna a medias.

    Se repintan siempre las tres, porque una lectura llega siempre entera: si
    un dato se pierde, se pierden todos, y dejar la fila que sí llegó pintada
    sería enseñar una cifra vieja como si fuera de ahora mismo.
    """
    strip = MetricsStrip()
    try:
        strip.set_sample(_muestra())
        strip.set_sample({"gpu_percent": "mucho"})
        textos = strip.texts()
        for clave in ("gpu", "cpu", "vram"):
            assert textos[clave]["value"] == "—", clave
            assert textos[clave]["pct"] == 0, clave
        # Y ni un tipo raro ni una fila a medias.
        strip.set_sample("esto no es un dict")
        assert strip.texts()["gpu"]["value"] == "—"
        strip.set_sample({"gpu_percent": float("nan")})
        assert strip.texts()["gpu"]["pct"] == 0, (
            "un NaN en la barra se pintaría al 100%: min(100.0, nan) da 100.0")
    finally:
        strip.deleteLater()


# ---------------------------------------------------------------------------
# El controlador
# ---------------------------------------------------------------------------


def test_el_temporizador_solo_vive_con_un_render_en_marcha(qapp, monkeypatch):
    """Este es el test que justifica el diseño de la franja.

    Con el temporizador siempre vivo, la app se quedaba despierta con la
    ventana cerrada y además midiendo una CPU que no le interesaba a nadie. La
    franja nace y muere con el render, igual que el desplegable.
    """
    monkeypatch.setattr(metrics_mod, "pynvml", None)
    control = MetricsStripController()
    try:
        assert control.running() is False
        assert control.widget().texts()["gpu"]["value"] == "—"

        control.start()
        assert control.running() is True

        control.stop()
        assert control.running() is False
        assert control.widget().texts()["gpu"]["value"] == "—"
    finally:
        control.deleteLater()


def test_parar_dos_veces_no_rompe_nada(qapp, monkeypatch):
    """Se para al terminar el render y otra vez al cerrar la ventana."""
    monkeypatch.setattr(metrics_mod, "pynvml", None)
    control = MetricsStripController()
    try:
        control.start()
        control.sample_now()
        control.stop()
        control.stop()
        assert control.running() is False
    finally:
        control.deleteLater()


def test_abrir_nvml_llega_al_primer_tic_y_no_al_construir_la_ventana(qapp, monkeypatch):
    """Construir la franja no puede llevarse el driver por delante.

    ``LiveMetrics`` abre un handle contra el driver en el constructor. Si eso
    pasara al montar la ventana, abrir la app costaría una inicialización de
    NVML en un equipo que igual no va a codificar nunca.
    """
    control = MetricsStripController()
    try:
        assert control._metrics is None, "LiveMetrics se construyó demasiado pronto"
        control.start()
        assert control._metrics is None, (
            "start() no debe abrir NVML: lo hace el primer tic")
        control.sample_now()
        assert control._metrics is not None
    finally:
        control.stop()
        control.deleteLater()


def test_sin_muestreo_la_franja_sigue_diciendo_la_verdad(qapp, monkeypatch):
    """Si no se puede medir, no se inventa un cero: se deja en "—"."""
    class _Revienta:
        def __init__(self, *_a, **_k):
            raise OSError("no hay NVML")

    monkeypatch.setattr(metrics_mod, "LiveMetrics", _Revienta)
    control = MetricsStripController()
    try:
        control.start()
        control.sample_now()
        textos = control.widget().texts()
        assert textos["gpu"]["value"] == "—"
        assert textos["cpu"]["value"] == "—"
        assert "0 %" not in textos["gpu"]["value"]
    finally:
        control.stop()
        control.deleteLater()


def test_un_muestreo_que_revienta_no_tumba_la_franja(qapp, monkeypatch):
    """NVML que muere a mitad de sesión no puede tumbar la ventana."""
    class _Muerto:
        def __init__(self, *_a, **_k):
            self.gpu_name = "NVIDIA GeForce RTX 3060"

        def sample(self):
            raise RuntimeError("la GPU se ha reiniciado")

        def shutdown(self):
            pass

    monkeypatch.setattr(metrics_mod, "LiveMetrics", _Muerto)
    control = MetricsStripController()
    try:
        control.start()
        control.sample_now()
        assert control.running() is True
        assert control.widget().texts()["gpu"]["value"] == "—"
        control.stop()
        assert control.running() is False
    finally:
        control.deleteLater()


# ---------------------------------------------------------------------------
# El cableado en la ventana
# ---------------------------------------------------------------------------


def test_los_cinco_caminos_de_cierre_pasan_por_el_helper():
    """Todos, no solo el último.

    ``closeEvent`` tiene cinco salidas que cierran de verdad y cuatro que
    hacen ``event.ignore()``. Si el muestreo se parara al entrar en el método,
    un "Cancelar" del diálogo dejaría la ventana abierta con la franja
    congelada en el último número leído —que es exactamente lo que hace parecer
    roto el panel—. Por eso el Parar va en ``_cerrar_ya``, justo antes de
    cerrar, y no antes de preguntar nada.
    """
    from src.ui import main_window

    fuente = inspect.getsource(main_window.MainWindow.closeEvent)
    assert "super().closeEvent(event)" not in fuente, (
        "closeEvent sigue cerrando por su cuenta en algún camino: ese camino "
        "no para la franja ni suelta NVML")
    assert fuente.count("self._cerrar_ya(event)") == 5

    helper = inspect.getsource(main_window.MainWindow._cerrar_ya)
    assert "self._metrics_sampling_stop()" in helper
    assert helper.index("self._metrics_sampling_stop()") < helper.index(
        "super().closeEvent(event)")


def test_la_franja_se_crea_y_se_para_desde_el_render():
    """Ni se crea al abrir la ventana, ni se apaga en un camino suelto.

    Se comprueba por AST y no por texto porque lo que importa es que la ventana
    llame a los helpers, no que el helper se llame ``_metrics_sampling_start``.
    Lo que sí tiene que ser cierto es que hay un solo interruptor para la
    bandeja y para la franja: si uno se encendiera en un camino del render y el
    otro no, acabarían pidiendo NVML dos veces y midiendo cosas distintas en la
    misma pantalla.
    """
    import ast
    from pathlib import Path

    raiz = Path(__file__).resolve().parent.parent
    arbol = ast.parse((raiz / "src" / "ui" / "main_window.py").read_text(
        encoding="utf-8"))
    ventana = next(n for n in ast.walk(arbol)
                   if isinstance(n, ast.ClassDef) and n.name == "MainWindow")
    metodos = {n.name: n for n in ventana.body if isinstance(n, ast.FunctionDef)}

    for nombre in ("_metrics_sampling_start", "_metrics_sampling_stop",
                   "_cerrar_ya"):
        assert nombre in metodos, f"MainWindow.{nombre} no existe: la franja no para"

    llamadas = {
        nombre: {
            n.func.attr for n in ast.walk(metodos[nombre])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        for nombre in ("_tray_sampling_start", "_tray_sampling_stop")
    }
    assert "_metrics_sampling_start" in llamadas["_tray_sampling_start"]
    assert "_metrics_sampling_stop" in llamadas["_tray_sampling_stop"]

    # El arranque y el apagado vienen del render, no de la apertura de la
    # ventana: si _setup_ui encendiera el muestreo, la franja mediría en reposo.
    assert "_metrics_sampling_start" not in {
        n.func.attr for n in ast.walk(metodos["_setup_ui"])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }


def test_la_franja_va_al_layout_raiz_debajo_del_splitter():
    """Abajo del todo, y cruzando también el panel de tareas.

    Va en el layout raíz y no dentro de una columna porque el panel de tareas
    es donde se mira el progreso de una cola larga, y es ahí donde hace falta
    ver si la GPU está trabajando o parada.
    """
    fuente = inspect.getsource(__import__(
        "src.ui.main_window", fromlist=["MainWindow"]).MainWindow._setup_ui)
    assert "root_layout = QVBoxLayout(central)" in fuente, (
        "root_layout es horizontal: la franja se pondria AL LADO del splitter, "
        "ocupando media ventana, en vez de debajo")
    assert "MetricsStripController(" in fuente
    lineas = [linea.strip() for linea in fuente.splitlines()
              if "addWidget" in linea]
    assert "root_layout.addWidget(self._metrics_strip.widget())" in lineas, (
        "la franja no se añade al layout raíz: no queda abajo del todo")
    assert lineas.index(
        "root_layout.addWidget(self._metrics_strip.widget())") > lineas.index(
        "root_layout.addWidget(self.splitter)"), (
        "la franja ha quedado por encima del splitter")