"""Validación de salud de Qt: comprobar que se puede crear una QApplication
en modo offscreen y que un widget se renderiza de verdad.

La comparten ``main.py`` (selftest) y ``packaging/linux/capture_screenshots.py``
para que ambos detecten la misma clase de fallos (libEGL/libGL/X11 ausentes,
platform plugin roto, etc.) con exactamente la misma lógica.
"""

from __future__ import annotations

import os
from typing import Tuple


def check_qapplication_offscreen() -> Tuple[bool, str]:
    """Crea una QApplication real en el platform ``offscreen`` y verifica que
    un widget mínimo se renderiza.

    Devuelve ``(ok, detalle)``:

    - ``(True, "offscreen (widget render OK)")`` si todo funciona.
    - ``(False, razon)`` si Qt no puede arrancar o el platform no es
      ``offscreen``. Nunca lanza: el fallo se devuelve como valor para que
      cada llamador decida cómo reportarlo.

    Nota: si Qt no encuentra el platform plugin, ``qFatal`` aborta el proceso
    sin dar oportunidad a Python (no lanza excepción). En ese caso no hay
    forma de capturarlo desde aquí; la comprobación cubre los fallos que sí
    son capturables (plataforma distinta, widget no visible, excepciones de
    inicialización).
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication, QWidget

        qt_app = QApplication.instance() or QApplication(["draggy-qt-check"])
        platform = qt_app.platformName()
        if platform != "offscreen":
            return (
                False,
                f"QApplication arrancó en '{platform}', se esperaba 'offscreen'",
            )
        widget = QWidget()
        widget.resize(64, 64)
        widget.show()
        qt_app.processEvents()
        if not widget.isVisible():
            return False, "QWidget no quedó visible tras processEvents()"
        widget.close()
        qt_app.processEvents()
        return True, "offscreen (widget render OK)"
    except Exception as exc:  # noqa: BLE001 - reportamos, no propagamos
        return False, f"{type(exc).__name__}: {exc}"


def check_main_window_offscreen() -> Tuple[bool, str]:
    """Construye la MainWindow completa de la app en modo offscreen.

    Es la comprobación más fuerte posible sin abrir la GUI de verdad: valida
    que toda la construcción de la UI (widgets, tabs, estilos, workers
    diferidos, etc.) funciona en un entorno sin display. Un fallo aquí es un
    fallo de empaquetado que conviene detectar antes de generar el bundle.

    Devuelve ``(ok, detalle)`` igual que :func:`check_qapplication_offscreen`;
    nunca lanza. Requiere que exista una QApplication (llamar antes a
    :func:`check_qapplication_offscreen` o crear una instancia propia).
    """
    try:
        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance()
        if qt_app is None:
            return False, "no hay QApplication (llama antes a check_qapplication_offscreen)"

        from src.ui.main_window import MainWindow

        window = MainWindow(
            "",
            {"recommended_device": "CPU", "gpus": []},
            {"libx264": True, "libx265": True},
        )
        window.resize(1280, 900)
        window.show()
        qt_app.processEvents()
        if not window.isVisible():
            window.close()
            return False, "MainWindow no quedó visible tras processEvents()"
        window.close()
        qt_app.processEvents()
        window.deleteLater()
        qt_app.processEvents()
        return True, "MainWindow construida y visible (1280x900)"
    except Exception as exc:  # noqa: BLE001 - reportamos, no propagamos
        return False, f"{type(exc).__name__}: {exc}"


def check_dependencies_dialog_offscreen() -> Tuple[bool, str]:
    """Construye el diálogo de engines/runtimes (DependenciesDialog) en modo
    offscreen.

    Cubre la segunda pantalla principal de la app: las pestañas de engines,
    runtimes y binaries se construyen con sus filas y workers diferidos. Un
    fallo aquí es un fallo de empaquetado que conviene detectar antes de
    generar el bundle, igual que con :func:`check_main_window_offscreen`.

    Devuelve ``(ok, detalle)``; nunca lanza. Requiere una QApplication
    existente (llamar antes a :func:`check_qapplication_offscreen`).
    """
    try:
        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance()
        if qt_app is None:
            return False, "no hay QApplication (llama antes a check_qapplication_offscreen)"

        from src.ui.dependencies_dialog import DependenciesDialog

        dialog = DependenciesDialog(initial_tab="ai_engines")
        dialog.resize(1200, 820)
        dialog.show()
        qt_app.processEvents()
        if not dialog.isVisible():
            dialog.close()
            return False, "DependenciesDialog no quedó visible tras processEvents()"
        dialog.close()
        qt_app.processEvents()
        dialog.deleteLater()
        qt_app.processEvents()
        return True, "DependenciesDialog construido y visible (1200x820)"
    except Exception as exc:  # noqa: BLE001 - reportamos, no propagamos
        return False, f"{type(exc).__name__}: {exc}"


# Hook de validación usado por el smoke job de CI: con esta variable se
# simula una MainWindow rota para comprobar que el pre-flight (run_ui_checks)
# falla con exit 1 y mensaje claro ANTES de generar ningún PNG.
if os.environ.get("DRAGGY_TEST_BROKEN_MAIN_WINDOW") == "1":
    check_main_window_offscreen = lambda: (False, "simulated failure (smoke test)")


def run_ui_checks() -> list[Tuple[str, str, str]]:
    """Ejecuta las tres validaciones de UI en orden de dependencia.

    Es el bucle único que comparten el selftest (``main.py``) y
    ``capture_screenshots.py``: evita duplicar el orden de los checks y el
    manejo de resultados en cada llamador.

    Devuelve una lista de ``(nombre, estado, detalle)`` en este orden:

    - ``("QApplication offscreen", ...)``
    - ``("MainWindow offscreen", ...)``
    - ``("DependenciesDialog offscreen", ...)``

    ``estado`` es ``"ok"``, ``"fail"`` o ``"skip"``. Si la QApplication
    falla, los checks de ventanas se marcan ``"skip"`` y NO se ejecutan:
    construir la MainWindow o el diálogo sin app base solo produciría un
    segundo fallo sin información nueva.

    ``check_qapplication_offscreen`` crea la QApplication, que los checks
    de ventanas reutilizan vía ``QApplication.instance()``: el orden es
    parte del contrato.
    """
    qapp_ok, qapp_detail = check_qapplication_offscreen()
    results = [("QApplication offscreen", "ok" if qapp_ok else "fail", qapp_detail)]

    if not qapp_ok:
        results.append(("MainWindow offscreen", "skip", "sin QApplication válida"))
        results.append(
            ("DependenciesDialog offscreen", "skip", "sin QApplication válida")
        )
        return results

    for name, check in (
        ("MainWindow offscreen", check_main_window_offscreen),
        ("DependenciesDialog offscreen", check_dependencies_dialog_offscreen),
    ):
        ok, detail = check()
        results.append((name, "ok" if ok else "fail", detail))
    return results
