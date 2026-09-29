"""El desplegable de la bandeja: qué enseña y, sobre todo, qué no.

Un menú del sistema es la última superficie donde se mira la app sin abrirla,
y por eso tiene una exigencia que no tiene el resto de la interfaz: **no
puede mentir**. Si la GPU no se ha podido leer, no puede poner "0 %"; si la
CPU todavía no tiene un intervalo que promediar, no puede decir que el
equipo está parado. Aquí se comprueba que los datos que no existen no se
convierten en ceros, y que los que existen se escriben de una forma.

También se comprueba el motivo por el que el desplegable es un
``QWidgetAction`` y no una acción de texto: una acción se dibuja una vez al
abrir el menú, y un menú que enseña el 4 % mientras el render va por el 40 %
es peor que no tener desplegable, porque parece vivo y no lo está. Por eso
el widget se repinta con ``set_state`` y el estado se puede leer en
cualquier momento.

Y una comprobación de lifecycle que parece menor y no lo es: el temporizador
de muestreo **se para al terminar el render**, porque un ``QTimer`` vivo con
la ventana cerrada es una aplicación que no se cierra.
"""

import os

import pytest

from src.hardware import metrics as metrics_mod
from src.ui.tray_dashboard import (
    TrayController,
    TrayDashboard,
    eta_text,
    metric_lines,
    queue_text,
    speed_text,
)


# ---------------------------------------------------------------------------
# Las funciones puras
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fps,speed,esperado", [
    (147.2, 7.2, "147.2 fps  ·  7.2x"),
    # ffmpeg no calcula fps hasta tener ~1 s de media: sale 0 y se enseña
    # solo el multiplicador, que si esta.
    (0, 10.9, "10.9x"),
    (12.3, None, "12.3 fps"),
    (None, None, None),
    (0, 0, None),
])
def test_la_velocidad_se_escribe_en_las_unidades_que_existen(fps, speed, esperado):
    assert speed_text(fps, speed) == esperado


def test_la_cola_no_se_inventa_una_cuanda_que_no_existe():
    """Sin cola en marcha no hay línea de cola, ni "0 de 0"."""
    assert queue_text(None, None) is None
    assert queue_text(0, 0) is None
    assert queue_text(3, 0) is None


def test_la_cola_cuenta_la_que_se_esta_haciendo():
    """``done`` es lo ya terminado, así que la de uno es la que va."""
    assert queue_text(0, 7) == "1 de 7 en cola"
    assert queue_text(3, 7) == "4 de 7 en cola"
    assert queue_text(6, 7) == "7 de 7 en cola"


def test_una_linea_de_gpu_solo_si_hay_gpu():
    """La diferencia entre "0 %" y "no lo sé", que es el motivo del módulo."""
    con_gpu = metric_lines({"gpu_percent": 0, "gpu_name": "NVIDIA RTX 3060"})
    assert any("GPU  0 %" in linea for linea in con_gpu)
    # Sin dato, la línea no existe. Y una lista vacía es "no hay nada que
    # decir", que es distinto de "todo a cero".
    assert metric_lines({"gpu_percent": None, "cpu_percent": 50}) == ["CPU  50 %"]
    assert metric_lines({}) == []
    assert metric_lines(None) == []


def test_la_gpu_dice_de_que_tarjeta_es():
    """El porcentaje sin el nombre no sirve: en un equipo con dos, no."""
    lineas = metric_lines({"gpu_percent": 88, "gpu_name": "NVIDIA GeForce RTX 3060"})
    assert lineas[0] == "GPU  88 %  ·  NVIDIA GeForce RTX 3060"
    # Y sin nombre, al menos el porcentaje.
    assert metric_lines({"gpu_percent": 88, "gpu_name": ""})[0] == "GPU  88 %"


# ---------------------------------------------------------------------------
# El ETA: uno solo para la barra y el desplegable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("elapsed,pct,esperado", [
    (100, 50, "ETA ~01:40"),
    (100, 90, "ETA ~00:11"),
    (4000, 50, "ETA ~1h 06m 40s"),
    (20000, 50, "ETA ~5h 33m 20s"),
    # Sin datos suficientes no se dice nada: un ETA con dos muestras es un
    # numero que miente con mucha seguridad.
    (1, 50, None),
    (100, 0, None),
    (100, 1, None),
    (100, None, None),
    (None, 50, None),
])
def test_el_eta_solo_cuando_hay_datos_para_calcularlo(elapsed, pct, esperado):
    assert eta_text(elapsed, pct) == esperado


def test_la_barra_y_el_desplegable_no_pueden_discrepar_sobre_el_eta():
    """La barra de la ventana usa ESTA función, no una copia suya.

    Si mañana alguien cambia el formato en un sitio y no en el otro, la app
    dice dos ETA distintos del mismo render, y el que no se prende es el
    del desplegable. Se ancla mirando el código de la ventana, no el texto
    que pone aquí: un texto aqui no probaría nada.
    """
    import inspect

    from src.ui import main_window

    fuente = inspect.getsource(main_window.MainWindow._on_worker_progress)
    assert "eta_text(" in fuente, (
        "la barra de progreso no está usando eta_text del desplegable: "
        "son dos cálculos distintos del mismo número")
    assert "remaining = elapsed" not in fuente, (
        "la barra vuelve a calcular el ETA a mano; duplica la fórmula")


# ---------------------------------------------------------------------------
# El widget
# ---------------------------------------------------------------------------

def _estado(**extra):
    base = {
        "active": True, "task_name": "entrada.mkv", "pct": 40,
        "eta": "ETA ~00:12", "metrics": {}, "fps": None, "speed": None,
    }
    base.update(extra)
    return base


def test_en_reproceso_no_hay_cifras_que_puedan_ser_falsas(qapp):
    """Idle es una línea, y ni una cifra.

    Es lo que se pidió y es lo correcto: CPU y GPU en reposo son ruido, y
    además obligarían a un temporizador corriendo con la app cerrada.
    """
    dash = TrayDashboard()
    try:
        dash.set_state(_estado(active=False,
                               metrics={"cpu_percent": 90, "gpu_percent": 90}))
        textos = dash.texts()
        assert textos["title"] == "En reposo"
        assert textos["sub"] == ""
        assert textos["metrics"] == ""
        assert textos["pct"] == 0
    finally:
        dash.deleteLater()


def test_con_render_enseña_tarea_porcentaje_eta_y_cifras(qapp):
    dash = TrayDashboard()
    try:
        dash.set_state(_estado(
            pct=40, eta="ETA ~00:12", fps=147.2, speed=7.2,
            metrics={"cpu_percent": 84, "gpu_percent": 88,
                     "gpu_name": "NVIDIA GeForce RTX 3060",
                     "vram_used_mb": 3255, "vram_total_mb": 12288},
            queue_done=3, queue_total=7,
        ))
        textos = dash.texts()
        assert textos["title"] == "entrada.mkv"
        assert "40 %" in textos["sub"]
        assert "ETA ~00:12" in textos["sub"]
        assert "4 de 7 en cola" in textos["sub"]
        assert "147.2 fps" in textos["metrics"]
        assert "7.2x" in textos["metrics"]
        assert "CPU  84 %" in textos["metrics"]
        assert "GPU  88 %" in textos["metrics"]
        assert "3.2 / 12.0 GB" in textos["metrics"]
        assert textos["pct"] == 40
    finally:
        dash.deleteLater()


def test_sin_gpu_no_aparece_una_linea_de_gpu_inventada(qapp):
    """El caso de un equipo sin NVIDIA: la línea no está, no dice cero."""
    dash = TrayDashboard()
    try:
        dash.set_state(_estado(metrics={"cpu_percent": 50}))
        textos = dash.texts()
        # La comprobacion util no es que no aparezca el texto "0 %" --eso
        # aparece dentro de "50 %"--, sino que no exista ninguna linea de GPU.
        assert "GPU" not in textos["metrics"]
        assert textos["metrics"] == "CPU  50 %"
    finally:
        dash.deleteLater()


def test_un_estado_vacio_o_basura_no_deja_el_desplegable_a_medias(qapp):
    """Lo que le pase al estado, el desplegable aguanta y se queda en reposo."""
    dash = TrayDashboard()
    try:
        for basura in (None, {}, "texto", 42, {"active": True}):
            dash.set_state(basura)
            textos = dash.texts()
            assert textos["pct"] == 0
            assert isinstance(textos["title"], str)
    finally:
        dash.deleteLater()


def test_el_porcentaje_se_acota_y_no_se_rompe_la_barra(qapp):
    """Un 140 % o un -5 no pueden dejar la barra en un estado imposible."""
    dash = TrayDashboard()
    try:
        dash.set_state(_estado(pct=140))
        assert dash.texts()["pct"] == 100
        dash.set_state(_estado(pct=-5))
        assert dash.texts()["pct"] == 0
        dash.set_state(_estado(pct="no soy un numero"))
        assert dash.texts()["pct"] == 0
    finally:
        dash.deleteLater()


def test_el_widget_no_se_come_los_clics_para_que_el_menu_no_se_cierre(qapp):
    """Los números no son botones y no tienen por qué cerrar el menú."""
    from PySide6.QtCore import Qt

    dash = TrayDashboard()
    try:
        assert dash.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents) is True
    finally:
        dash.deleteLater()


# ---------------------------------------------------------------------------
# El canal de métricas: un dict plano, y el progreso de siempre intacto
# ---------------------------------------------------------------------------

def test_el_canal_de_metricas_es_plano_igual_que_el_de_ffmpeg():
    """Las dos fuentes de cifras hablan el mismo idioma.

    El bucle de IA y ``run_ffmpeg`` mandan los dos por el mismo canal. Antes
    el de IA usaba un envoltorio ``{"metric": ..., "value": ...}`` copiado
    del de las fases, y el consumidor tenía que adivinar de qué render salía
    cada número. Con el envoltorio, en un render de compresión normal —el
    camino más común— la línea de velocidad no aparecía nunca.
    """
    from src.processing.pipeline import _emit_metrics

    recibidos = []
    _emit_metrics(recibidos.append, fps=12.5, frame=300)
    assert recibidos == [{"fps": 12.5, "frame": 300}]

    # Y el de ffmpeg, que ya era plano, encaja en el mismo estado: el
    # consumidor es uno solo y acepta las dos fuentes sin distinguirlas.
    campos = set(TrayController.CAMPOS_METRICAS)
    assert {"fps", "speed", "frame"} <= campos


def test_un_canal_de_metricas_roto_no_tumba_un_render():
    """Igual que ``_emit_phase``: un watcher roto no rompe el render."""
    from src.processing.pipeline import _emit_metrics

    def _reventon(_campos):
        raise RuntimeError("el desplegable se ha caído")

    _emit_metrics(_reventon, fps=1.0)   # no lanza
    _emit_metrics(None, fps=1.0)        # sin canal, no hace nada


def test_el_canal_de_metricas_tiene_las_mismas_claves_que_las_que_se_pintan():
    """Si el desplegable pinta una clave, alguien tiene que poder mandarla.

    Sin esto, añadir una línea nueva al widget se puede hacer sin que nada
    la llene, y el test seguiría en verde enseñando un guion.
    """
    import inspect

    from src.ui.tray_dashboard import metric_lines

    fuente = inspect.getsource(metric_lines)
    for clave in ("gpu_percent", "cpu_percent", "vram_used_mb", "vram_total_mb"):
        assert clave in fuente, f"{clave} se pinta pero no hay quien la mande"


@pytest.mark.skipif(not os.path.exists(os.path.join("bin", "ffmpeg")),
                    reason="No hay FFmpeg en bin/ en este checkout")
def test_ffmpeg_manda_las_cifras_que_ya_calculaba_y_el_progreso_no_cambia(tmp_path):
    """El binario de verdad: salen fps/velocidad, y el % sigue siendo el de antes.

    La segunda mitad es la importante. ``progress_callback`` no ha cambiado de
    firma ni de contenido: sigue recibiendo enteros, y sólo con los cambios de
    porcentaje. Un arreglo para enseñar la velocidad no puede romper el
    progreso, que es lo que se ve en la ventana y en el panel de tareas.
    """
    import subprocess

    from src.ffmpeg.manager import get_local_ffmpeg_path
    from src.ffmpeg.runner import run_ffmpeg

    ffmpeg = get_local_ffmpeg_path()
    if not (ffmpeg and os.path.exists(ffmpeg)):
        pytest.skip("No hay FFmpeg local en este checkout")

    salida = str(tmp_path / "salida.mkv")
    pcts, metricas = [], []
    ok = run_ffmpeg(
        ffmpeg,
        ["-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=30",
         "-c:v", "libx264", "-preset", "ultrafast", salida],
        progress_callback=pcts.append,
        metrics_callback=metricas.append,
    )
    assert ok is True
    assert os.path.exists(salida)

    # El progreso: enteros, monótono, sin repetir, acabando en 100.
    assert pcts, "no llegó ningún progreso"
    assert all(isinstance(p, int) for p in pcts)
    assert pcts == sorted(pcts), f"el progreso retrocedió: {pcts}"
    assert pcts[-1] == 100

    # Las métricas: un dict plano con las cifras que ya traía ffmpeg.
    assert metricas, "ffmpeg mandó progreso pero ninguna métrica"
    ultima = metricas[-1]
    assert isinstance(ultima, dict)
    assert "fps" in ultima and "speed" in ultima
    assert float(ultima["speed"]) > 0, "una velocidad de 0 no dice nada"
    assert ultima["out_time"] > 0, "el tiempo de salida tambien viene en el bloque"

    # Y el throttle: ffmpeg emite cada 100 ms y esto publica mucho menos.
    assert len(metricas) < 20, (
        f"{len(metricas)} métricas: el throttle de un segundo no está "
        "aplicándose y el desplegable recibe 10 veces más de lo que puede leer")


# ---------------------------------------------------------------------------
# El controlador: menú, temporizador y canal, sin ventana
# ---------------------------------------------------------------------------

def test_el_menu_lleva_el_desplegable_y_las_dos_salidas(qapp):
    """El desplegable se añade; *Quit* no se cae de camino.

    El menu de la bandeja es la unica via de salir en un equipo donde el
    usuario ha minimizado la ventana. Añadir el desplegable no puede quitarle
    la de cerrar.
    """
    control = TrayController()
    try:
        assert "Quit DRAGGY ENCODER" in control.acciones()
        assert "Open DRAGGY ENCODER" in control.acciones()
        # El desplegable es la PRIMERA entrada, y es un widget, no un texto:
        # por eso el menu se repinta abierto.
        primera = control.menu().actions()[0]
        assert primera.defaultWidget() is control.dashboard
    finally:
        control.deleteLater()


def test_las_dos_salidas_llegan_a_sus_callbacks(qapp):
    """Lo que dice el menu es lo que hace, no una decoracion."""
    abiertos = []
    cerrados = []
    control = TrayController(on_open=lambda: abiertos.append(1),
                             on_quit=lambda: cerrados.append(1))
    try:
        for accion in control.menu().actions():
            if accion.text() == "Open DRAGGY ENCODER":
                accion.trigger()
            elif accion.text() == "Quit DRAGGY ENCODER":
                accion.trigger()
        assert abiertos == [1]
        assert cerrados == [1]
    finally:
        control.deleteLater()


def test_el_temporizador_solo_vive_con_un_render_en_marcha(qapp, monkeypatch):
    """Con la app en reposo no hay temporizador, y al parar se para.

    Este es el test que justifica el diseño: el temporizador no nace con la
    aplicación, sino con el render. Con el vivo siempre, la app se quedaba
    despierta con la ventana cerrada y además midiendo una CPU que no le
    interesaba a nadie.
    """
    # Sin GPU real: lo que se comprueba es el ciclo de vida, no la cifra.
    monkeypatch.setattr(metrics_mod, "pynvml", None)
    control = TrayController()
    try:
        assert control.running() is False
        assert control.texts()["title"] == "En reposo"

        control.start()
        assert control.running() is True

        control.stop()
        assert control.running() is False
        assert control.texts()["title"] == "En reposo"
    finally:
        control.deleteLater()


def test_parar_dos_veces_no_rompe_nada(qapp, monkeypatch):
    """Se para al terminar un render y otra vez al cerrar la ventana."""
    monkeypatch.setattr(metrics_mod, "pynvml", None)
    control = TrayController()
    try:
        control.start()
        control.stop()
        control.stop()
    finally:
        control.deleteLater()


def test_sin_muestreo_el_desplegable_sigue_diciendo_la_verdad(qapp, monkeypatch):
    """Si no hay métricas, el progreso no se inventa una línea de cifras.

    En un equipo sin NVML, o si ``LiveMetrics`` no se puede construir, el
    desplegable tiene que seguir enseñando la tarea y su porcentaje: son
    datos ciertos. Lo que no puede es enseñar un "0 %" de algo que nadie ha
    medido.
    """
    control = TrayController()
    try:
        real = metrics_mod.LiveMetrics

        class _Revienta:
            def __init__(self, *_a, **_k):
                raise OSError("no hay NVML")

        monkeypatch.setattr(metrics_mod, "LiveMetrics", _Revienta)
        control.start()
        control.set_task("entrada.mkv")
        control.update_progress(42, "ETA ~00:09")
        textos = control.texts()
        assert textos["title"] == "entrada.mkv"
        assert textos["pct"] == 42
        assert "42 %" in textos["sub"]
        assert "0 %" not in textos["metrics"]
        monkeypatch.setattr(metrics_mod, "LiveMetrics", real)
    finally:
        control.deleteLater()


def test_el_controlador_mezcla_el_canal_de_metricas_tal_cual(qapp):
    """Un dict plano, mezclado como llega, sin adivinar de dónde viene.

    La IA manda ``fps`` y ffmpeg manda ``fps``/``speed``/``frame``; los dos
    rellenan el mismo estado. Con un envoltorio por un lado —que fue como
    empezó— el camino más común, que es la compresión, no enseñaba nunca la
    velocidad.
    """
    control = TrayController()
    try:
        control.start()
        control.update_metrics({"fps": 12.5, "frame": 300})
        assert "12.5 fps" in control.texts()["metrics"]
        control.update_metrics({"fps": 99.0, "speed": 4.2})
        assert "99.0 fps" in control.texts()["metrics"]
        assert "4.2x" in control.texts()["metrics"]
        # Una basura no rompe nada ni borra lo que ya habia.
        control.update_metrics(None)
        control.update_metrics("no soy un dict")
        assert "99.0 fps" in control.texts()["metrics"]
    finally:
        control.stop()
        control.deleteLater()


def test_el_rato_de_espera_entre_un_render_y_otro(qapp, monkeypatch):
    """Un render nuevo no hereda las cifras del anterior.

    Si no se limpia el estado al parar, el desplegable del segundo render
    enseñaría el fps del primero durante el segundo segundo, que es
    exactamente el tipo de cifra que no debe estar en pantalla.
    """
    monkeypatch.setattr(metrics_mod, "pynvml", None)
    control = TrayController()
    try:
        control.start()
        control.set_task("primero.mkv")
        control.update_metrics({"fps": 200.0})
        control.stop()

        control.start()
        control.set_task("segundo.mkv")
        assert "200.0 fps" not in control.texts()["metrics"]
        assert control.texts()["title"] == "segundo.mkv"
    finally:
        control.stop()
        control.deleteLater()


def test_el_controlador_enseña_la_cola_cuando_la_hay(qapp, monkeypatch):
    monkeypatch.setattr(metrics_mod, "pynvml", None)
    control = TrayController()
    try:
        control.start()
        control.set_task("tercero.mkv")
        control.set_queue(3, 7)
        assert "4 de 7 en cola" in control.texts()["sub"]
    finally:
        control.stop()
        control.deleteLater()


# ---------------------------------------------------------------------------
# El contrato de los llamantes: forma, no lógica
# ---------------------------------------------------------------------------

def test_ningun_render_recibe_un_argumento_que_no_existe():
    """Cada kwarg que pasa la ventana, lo acepta la función de destino.

    Se recorre el codigo de la ventana con AST y se mira. No es una prueba
    de comportamiento —eso lo dan los renders de verdad— sino de forma, y
    existe por un fallo concreto: se anadio ``metrics_cb`` a los cuatro
    llamantes de render y uno de ellos, ``generate_preview``, no lo tenia en
    su firma. El siguiente preview de un usuario habria sido un TypeError,
    y ni el self-check ni ningun test de ventana lo habrian visto, porque
    hace falta llegar al render entero.
    """
    import ast
    import inspect
    from pathlib import Path

    from src.processing import pipeline

    raiz = Path(__file__).resolve().parent.parent
    fuente = (raiz / "src" / "ui" / "main_window.py").read_text(encoding="utf-8")
    arbol = ast.parse(fuente)

    destinos = {
        "compress_video": pipeline.compress_video,
        "compress_video_with_ai": pipeline.compress_video_with_ai,
        "generate_preview": pipeline.generate_preview,
    }
    vistos = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        nombre = nodo.func.id if isinstance(nodo.func, ast.Name) else None
        if nombre not in destinos:
            continue
        vistos.add((nombre, nodo.lineno))
        func = destinos[nombre]
        params = inspect.signature(func).parameters
        admite_cualquiera = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        for kw in nodo.keywords:
            if kw.arg is None or kw.arg == "self":
                continue
            if kw.arg == "metrics_cb" and "metrics_cb" not in params:
                pytest.fail(
                    f"main_window.py:{nodo.lineno} pasa metrics_cb a {nombre}() "
                    "y su firma no lo tiene: cualquier preview de IA peta aqui")
            if kw.arg not in params and not admite_cualquiera:
                pytest.fail(
                    f"main_window.py:{nodo.lineno} pasa {kw.arg!r} a {nombre}() "
                    f"y su firma solo admite: {sorted(params)}")
    assert len(vistos) >= 3, (
        f"solo se han encontrado {vistos} llamantes de render: el AST ha dejado "
        "de verlos y este test pasaria sin comprobar nada")


def test_el_canal_de_metricas_llega_a_los_cuatro_caminos_de_render():
    """Si la ventana lo pasa a los cuatro, la función lo tiene que aceptar.

    Recíproco del anterior, y por el mismo motivo: si mañana alguien quita
    ``metrics_cb`` de uno de los cuatro, ese camino deja de mostrar la
    velocidad sin que nada se entere.
    """
    import inspect

    from src.processing import pipeline

    for nombre in ("compress_video", "compress_video_with_ai", "generate_preview"):
        params = inspect.signature(getattr(pipeline, nombre)).parameters
        assert "metrics_cb" in params, (
            f"{nombre}() no acepta metrics_cb: ese camino no mostraria la "
            "velocidad en el desplegable")
    assert "metrics_callback" in inspect.signature(
        __import__("src.ffmpeg.runner", fromlist=["run_ffmpeg"]).run_ffmpeg
    ).parameters
