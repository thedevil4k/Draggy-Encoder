"""Que el texto de toda la aplicación se pueda copiar con el ratón.

El cambio es de los que no se ven: antes sólo eran seleccionables las pocas
etiquetas que alguien se había acordado de marcar con un
``setTextInteractionFlags`` repartido por cuatro ficheros, y en el panel de
dependencias eso eran 2 de 23. Copiar una ruta de instalación, un log o un
número de versión es justamente para lo que existe ``src.ui.selectable``.

La política tiene una sola excepción, y aquí se prueba como lo que es: una
cosas de la app que **no** se pueden marcar porque reaccionan al clic ellas
mismas. ``WorkflowStep`` (Upscale / Interp / Colorize) y sus cinco hermanos
conmutan en ``mousePressEvent``; si su etiqueta se volviese seleccionable,
pulsar en el nombre del paso dejaría de conmutarlo, y eso sería un arreglo
que rompe una función. Aquí no se comprueba el flag: se comprueba que al
pulsar el paso, el paso conmuta.

Y hay un segundo motivo para la forma que tiene la comprobación. Mirar el MRO
entero no vale: ``QAbstractScrollArea`` vuelve a declarar ``mousePressEvent``
en su cabecera, así que cualquier etiqueta dentro de un scroll —que en esta
app es casi todo— se declaraba bloqueada. De ahí que el recorrido pare en
cuanto aparece una clase de Qt.
"""

import pytest


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

_SELECCIONABLE = None


def _flags_de_seleccion():
    global _SELECCIONABLE
    if _SELECCIONABLE is None:
        from PySide6.QtCore import Qt

        _SELECCIONABLE = Qt.TextInteractionFlag.TextSelectableByMouse
    return _SELECCIONABLE


def _es_seleccionable(etiqueta):
    from PySide6.QtWidgets import QLabel

    assert isinstance(etiqueta, QLabel)
    return bool(etiqueta.textInteractionFlags() & _flags_de_seleccion())


def _todas_seleccionables(raiz):
    """Las etiquetas de ``raiz`` que NO son seleccionables, con su texto.

    Devuelve la lista de las que se quedaron fuera, no un booleano, para
    que un fallo diga qué texto se quedó sin poder copiarse.
    """
    from PySide6.QtWidgets import QLabel

    return [etiqueta.text() for etiqueta in raiz.findChildren(QLabel)
            if not _es_seleccionable(etiqueta)]


def _instalar(qapp):
    from src.ui.selectable import install_text_selection

    return install_text_selection(qapp)


# ---------------------------------------------------------------------------
# La política
# ---------------------------------------------------------------------------

def test_una_etiqueta_suelta_si_deja_seleccionar_texto(qapp):
    """El caso normal: una etiqueta en un contenedor que no gestiona el clic.

    Esto es lo que se estaba haciendo a mano en cuatro ficheros, y lo que
    ahora ocurre para todas las etiquetas sin pedir permiso.
    """
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    _instalar(qapp)
    raiz = QWidget()
    etiqueta = QLabel("C:/Program Files/draggy/bin/ffmpeg")
    QVBoxLayout(raiz).addWidget(etiqueta)

    try:
        assert _es_seleccionable(etiqueta) is True
    finally:
        raiz.deleteLater()


def test_una_etiqueta_nacida_despues_ya_nace_seleccionable(qapp):
    """Lo instalado es app-wide, no un repaso de lo que ya habia montado.

    Si sólo se marcara al instalar, las etiquetas que aparecieran después —
y en esta interfaz se crean paneles, filas y resultados de búsqueda a
medida que el usuario navega— se quedarian sin poder copiarse.
    """
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    _instalar(qapp)
    raiz = QWidget()
    QVBoxLayout(raiz)
    try:
        tardia = QLabel("aparecida despues de instalar el filtro")
        raiz.layout().addWidget(tardia)
        assert _es_seleccionable(tardia) is True
    finally:
        raiz.deleteLater()


def test_un_scroll_no_declara_bloqueadas_las_etiquetas_de_abajo(qapp):
    """``QAbstractScrollArea`` redeclara ``mousePressEvent``, y no cuenta.

    Este es el fallo que se cometió al medir: mirando el MRO entero, todo lo
    que estuviera dentro de un scroll se declaraba no seleccionable, que en
    esta app es casi todo. Se para en cuanto aparece una clase de Qt.
    """
    from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

    _instalar(qapp)
    area = QScrollArea()
    contenido = QWidget()
    etiqueta = QLabel("dentro de un scroll")
    QVBoxLayout(contenido).addWidget(etiqueta)
    area.setWidget(contenido)

    try:
        assert _es_seleccionable(etiqueta) is True
    finally:
        area.deleteLater()


def test_una_etiqueta_con_enlaces_conserva_el_uso_de_los_enlaces(qapp):
    """El flag se suma a los que hubiera, no los sustituye.

    Una etiqueta con ``LinksAccessibleByMouse`` tiene que seguir abriendo sus
    enlaces. Si el filtro reemplazara el valor en vez de sumarlo, el texto
    seguiría siendo seleccionable pero los enlaces dejarían de funcionar, y
    eso es peor que no poder copiar nada.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    _instalar(qapp)
    raiz = QWidget()
    etiqueta = QLabel('<a href="https://example.invalid">enlace</a>')
    etiqueta.setTextInteractionFlags(
        Qt.TextInteractionFlag.LinksAccessibleByMouse)
    QVBoxLayout(raiz).addWidget(etiqueta)

    try:
        flags = etiqueta.textInteractionFlags()
        assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
        assert flags & Qt.TextInteractionFlag.LinksAccessibleByMouse
    finally:
        raiz.deleteLater()


def test_una_etiqueta_ya_seleccionable_no_se_toca_dos_veces(qapp):
    """Quien lo marque a mano manda; el filtro no le discute el valor.

    Installarlo dos veces, o encontrar una etiqueta que ya venía marcada de
    la interfaz, tiene que dejar las cosas como estaban.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    _instalar(qapp)
    raiz = QWidget()
    etiqueta = QLabel("marcada a mano")
    marcado = (Qt.TextInteractionFlag.TextSelectableByMouse
               | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    etiqueta.setTextInteractionFlags(marcado)
    QVBoxLayout(raiz).addWidget(etiqueta)

    try:
        assert etiqueta.textInteractionFlags() == marcado
    finally:
        raiz.deleteLater()


def test_instalarlo_dos_veces_devuelve_el_mismo_filtro(qapp):
    """Idempotente a propósito: lo llaman el arranque y el self-check.

    Sin esto, abrir el self-check y luego la app dejaría dos filtros, y cada
    etiqueta se marcaría dos veces. Se afirma sobre la identidad del filtro
    que se guarda, no sobre un detalle interno del Qt.
    """
    primero = _instalar(qapp)
    segundo = _instalar(qapp)
    assert primero is not None
    assert primero is segundo


# ---------------------------------------------------------------------------
# La excepción: lo que gestiona su propio clic
# ---------------------------------------------------------------------------

def test_los_pasos_de_la_cadena_siguen_conmutando_al_pulsar(qapp):
    """El motivo de la excepción, comprobado por lo que de verdad importa.

    No se mira el flag del ``name_label``: se pulsa el paso y se mira si
    conmuta. Si algún día este test se pone rojo, el arreglo de "que todo sea
    seleccionable" ha apagado los pasos Upscale / Interp / Colorize, y eso es
    una función perdida, no un texto que cuesta más de copiar.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from src.ui.workflow_widget import WorkflowStep

    _instalar(qapp)
    paso = WorkflowStep("Upscale", "")
    try:
        paso.show()
        assert paso.active is False
        # Se pulsa la ETIQUETA, que es justo donde el usuario apunta.
        QTest.mouseClick(paso.name_label, Qt.MouseButton.LeftButton)
        assert paso.active is True
        QTest.mouseClick(paso.name_label, Qt.MouseButton.LeftButton)
        assert paso.active is False
    finally:
        paso.close()
        paso.deleteLater()


def test_una_etiqueta_dentro_de_una_tarjeta_con_clic_propio_no_se_marca(qapp):
    """La regla, escrita como regla, sobre una tarjeta de mentira.

    Se declara una tarjeta mínima con su propio ``mousePressEvent`` -- la
    misma forma que ``WorkflowStep``-- y se comprueba que la etiqueta de
    dentro no se marca. Un widget de la app y otro del test tienen que caer
    en la misma rama, no en una por un motivo y otra por suerte.
    """
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    _instalar(qapp)

    class _Tarjeta(QWidget):
        """Como ``WorkflowStep``: gestiona su propio clic."""

        def __init__(self):
            super().__init__()
            self.presiones = 0
            self.etiqueta = QLabel("pulsame")
            QVBoxLayout(self).addWidget(self.etiqueta)

        def mousePressEvent(self, event):
            self.presiones += 1
            super().mousePressEvent(event)

    tarjeta = _Tarjeta()
    try:
        assert _es_seleccionable(tarjeta.etiqueta) is False
    finally:
        tarjeta.deleteLater()


def test_una_clase_de_qt_que_redeclare_el_clic_no_cuenta_como_tarjeta(qapp):
    """El otro lado de la moneda, que es donde se rompió al medirlo.

    ``QAbstractScrollArea`` tiene ``mousePressEvent`` en su ``__dict__``. Si
    el recorrido no parara en las clases de Qt, cualquier etiqueta dentro de
    un scroll se declararía intocable y la mitad de la app perdería el texto
    seleccionable.
    """
    from PySide6.QtWidgets import QAbstractScrollArea, QLabel, QScrollArea

    from src.ui.selectable import _gestiona_el_clic

    # El hecho que hacía daño: Qt redeclara el clic en su propia cabecera.
    assert "mousePressEvent" in QAbstractScrollArea.__dict__

    class _MiScroll(QScrollArea):
        """Subclase PROPIA que hereda de una clase que ya redeclara el clic.

        Es el caso que rompió la medición, y por eso el recorrido tiene que
        seguir subiendo por una clase nuestra sin pararse en la primera
        clase de Qt que encuentra.
        """

    _instalar(qapp)
    scroll = _MiScroll()
    etiqueta = QLabel("dentro de un scroll propio")
    scroll.setWidget(etiqueta)
    try:
        # Nuestra clase no lo reescribe; el de más arriba, sí. Y aun así no
        # cuenta: para en cuanto llega a Qt.
        assert "mousePressEvent" not in _MiScroll.__dict__
        assert _gestiona_el_clic(scroll) is False
        assert _es_seleccionable(etiqueta) is True
    finally:
        scroll.deleteLater()


# ---------------------------------------------------------------------------
# La app entera
# ---------------------------------------------------------------------------

@pytest.fixture()
def dependencias_aisladas(monkeypatch):
    """Filas fijas y preferencias en memoria, para no depender del disco."""
    import src.preferences as preferences

    store = {}
    monkeypatch.setattr(preferences, "get", lambda key, default=None: store.get(key, default))
    monkeypatch.setattr(preferences, "set", lambda key, value: store.__setitem__(key, value) or True)

    from src.hardware import detector

    detector.remember_hardware(None)
    yield store
    detector.remember_hardware(None)


def test_el_panel_de_dependencias_deja_copiar_todas_sus_etiquetas(
        qapp, dependencias_aisladas, monkeypatch):
    """La medición, como test: en el diálogo de verdad, todas.

    Antes eran 2 de 23. Este es el número que tenía que subir, y se
    fija aqui para que no vuelva a bajar en silencio. La lista se imprime en
    el fallo, que es lo único que sirve cuando se rompe.
    """
    from src.ui import dependencies_dialog as dd

    filas = [
        {"id": "ffmpeg", "display_name": "FFmpeg (encoder suite)",
         "kind": "binary", "status": "installed", "version": "8.1.3",
         "version_text": "8.1.3", "install_dir": "/tmp/ffmpeg",
         "source_url": "https://example.invalid/x.zip", "detail": "",
         "is_updatable": True, "deletable": True, "update_available": None,
         "build_id": "n8.1", "build_label": "FFmpeg 8.1",
         "preferred_build_id": "latest"},
        {"id": "torch", "display_name": "PyTorch runtime", "kind": "runtime",
         "status": "installed", "version": "2.14.0+cu130",
         "install_dir": "/tmp/rt", "source_url": None, "detail": "",
         "is_updatable": True, "deletable": True, "update_available": None},
        {"id": "onnxruntime", "display_name": "ONNX Runtime", "kind": "runtime",
         "status": "missing", "version": "—", "install_dir": "",
         "source_url": None, "detail": "", "is_updatable": True,
         "deletable": False, "update_available": None},
    ]
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all", lambda: filas)

    _instalar(qapp)
    padre = None
    dialog = dd.DependenciesDialog(None)
    try:
        dialog._check_timer.stop()
        dialog._populate()
        dialog.show()

        from PySide6.QtWidgets import QLabel

        etiquetas = dialog.findChildren(QLabel)
        assert len(etiquetas) > 10, (
            f"solo hay {len(etiquetas)} etiquetas: el dialogo se ha degradado "
            "y este test pasaria sin comprobar nada")
        no_seleccionables = _todas_seleccionables(dialog)
        assert no_seleccionables == [], (
            f"{len(no_seleccionables)} de {len(etiquetas)} etiquetas no se "
            f"pueden copiar: {no_seleccionables}")
    finally:
        dialog.close()
        del dialog, padre
