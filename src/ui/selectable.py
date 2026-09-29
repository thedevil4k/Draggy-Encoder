"""Que cualquier texto del programa se pueda seleccionar y copiar con el ratón.

Por defecto Qt no deja copiar nada de una etiqueta: si arrastras el cursor
sobre un texto que quieres llevar al portapapeles, no se lleva nada. Por eso
estaban los ocho ``setTextInteractionFlags`` sueltos que había repartidos por
la interfaz: copiar una ruta de instalación, un log o un número de versión es
exactamente el uso para el que existe este módulo.

Aquí la regla es una sola y se aplica en la ``QApplication``, así que también
coge las etiquetas que se creen mañana, y no sólo las de hoy. Installarla es
una línea, y quien quiera que una etiqueta concreta no lo sea no tiene que
desactivarla aquí: le pone su propio flag después.

La excepción no es opcional. Una etiqueta seleccionable se queda con el clic,
y hay contenedores que reaccionan al clic ellos mismos: los pasos Upscale /
Interp / Colorize de la cadena conmutan en ``mousePressEvent`` y su
``name_label`` es una etiqueta dentro de la tarjeta. Si se hicieran
seleccionables, pulsar en el nombre de un paso dejaría de conmutar ese paso.
Por eso sólo se marca una etiqueta si **ningún** ancestro gestiona el clic por
su cuenta.

Y "gestiona el clic" se busca **en el MRO de la parte de la jerarquía que es
nuestra**, parando en cuanto aparece una clase de Qt. Mirar el MRO entero no
vale: ``QAbstractScrollArea`` vuelve a declarar ``mousePressEvent`` en su
cabecera, de modo que con la comprobación entera el primer scroll de la app
declaría bloqueadas todas las etiquetas que había debajo. Lo que decide es
una clase escrita en este repositorio.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QLabel

# Se pone además de lo que tuviera, no en vez de: una etiqueta con
# ``LinksAccessibleByMouse`` sigue teniendo que abrir sus enlaces.
_SELECCIONABLE = Qt.TextInteractionFlag.TextSelectableByMouse

_DE_QT = ("PySide6", "shiboken6", "PyQt5", "PyQt6")


def _gestiona_el_clic(widget):
    """¿``widget`` es nuestro y reescribe por su cuenta el clic izquierdo?"""
    for clase in type(widget).__mro__:
        if clase.__module__.split(".")[0] in _DE_QT:
            return False
        if "mousePressEvent" in clase.__dict__:
            return True
    return False


def _marcar(widget):
    """Pone la etiqueta seleccionable, si es etiqueta y puede serlo."""
    if not isinstance(widget, QLabel):
        return
    flags = widget.textInteractionFlags()
    if flags & _SELECCIONABLE:
        return
    ancestro = widget.parent()
    while ancestro is not None:
        if _gestiona_el_clic(ancestro):
            return
        ancestro = ancestro.parent()
    widget.setTextInteractionFlags(flags | _SELECCIONABLE)


class _FiltroDeTextos(QObject):
    """Marca cada etiqueta que aparece, y las que ya había al instalar."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.ChildAdded:
            _marcar(event.child())
        return False


def install_text_selection(app=None):
    """Deja seleccionable el texto de toda la aplicación.

    Idempotente: llamarlo dos veces no duplica filtros ni cambia el
    resultado, así que puede llamarlo el arranque normal y también el
    self-check sin que haya que coordinar los dos.
    """
    app = app or QApplication.instance()
    if app is None:
        return None
    if getattr(app, "_draggy_text_selection", None) is not None:
        return app._draggy_text_selection
    filtro = _FiltroDeTextos(app)
    app.installEventFilter(filtro)
    # ChildAdded sólo ve lo que nazca a partir de ahora: lo que ya estuviera
    # montado (un diálogo abierto antes de instalar el filtro) se marca ahora.
    for widget in app.allWidgets():
        _marcar(widget)
    app._draggy_text_selection = filtro
    return filtro
