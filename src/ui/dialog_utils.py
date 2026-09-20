"""Utilidades comunes para diálogos (scroll y ajuste a la pantalla).

Salen de dos fallos reales:

  * Un diálogo cuyo contenido crece más que la pantalla se sale por abajo y sus
    botones quedan inalcanzables (le pasó a Preferences: 1236 px de contenido).
  * Al meter contenido en un ``QScrollArea``, su *viewport* se pinta con el
    color base de la paleta (blanco) y rompe el tema oscuro de la app.

Ambas cosas se resuelven igual en cualquier diálogo, así que viven aquí.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFrame, QScrollArea, QWidget


def wrap_scrollable(page: QWidget, margins=(0, 0, 0, 0)) -> QScrollArea:
    """Devuelve un ``QScrollArea`` con ``page`` dentro, sin romper el tema.

    Las barras aparecen sólo si hacen falta: lo que no quepa se alcanza
    desplazándose, nunca se recorta.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setContentsMargins(*margins)
    area.setWidget(page)
    # El viewport de un QScrollArea se pinta con el color base de la paleta:
    # sin esto, el fondo oscuro del diálogo se vuelve blanco dentro del scroll.
    area.setStyleSheet("QScrollArea { background: transparent; }")
    area.viewport().setAutoFillBackground(False)
    area.viewport().setStyleSheet("background: transparent;")
    page.setAutoFillBackground(False)
    return area


def fit_to_screen(dialog, preferred_width: int = 940, preferred_height: int = 840,
                  min_width: int = 560, min_height: int = 360) -> None:
    """Deja ``dialog`` dentro del área útil de la pantalla (y lo centra).

    El ancho se pide al menos ``preferred_width`` (para que quepan los textos
    de una línea, como los de los ``QCheckBox``) pero nunca más de lo que hay.
    El alto se limita al área disponible descontando la barra de tareas. Nunca
    lanza: recolocar es una comodidad, no un requisito.
    """
    try:
        screen = dialog.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(max(preferred_width, min_width), max(min_width, available.width() - 40))
        height = min(preferred_height, max(min_height, available.height() - 60))
        dialog.resize(width, height)
        dialog.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(0, (available.height() - height) // 2),
        )
    except Exception:
        pass
