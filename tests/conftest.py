"""Test suite bootstrap.

Sets the Qt platform to ``offscreen`` *before* PySide6 is imported so the
headless UI tests can construct QWidgets without a real display server.

The Qt application object is exposed via the ``qapp`` fixture (function scoped)
so each test gets a clean event loop and widgets can be safely garbage
collected between tests.

``isolated_runtime`` is the opt-in isolation for tests that exercise the AI
runtime installers. It is deliberately NOT autouse: most of the suite needs the
real runtime store of the checkout (``local_store_runtime`` in test_draggy.py
imports torch from it), so redirecting the store globally would break those
tests instead of protecting anything.
"""
import os
import shutil
import sys

# Must run before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Los datos de referencia de los tests (p. ej. ``tests/data/nvidia_matrix.py``,
# la matriz de NVENC transcrita) son módulos sueltos, no código de la app: por eso
# viven fuera de ``src/``. Se añaden al path aquí para que el contraste pueda
# importarlos por nombre sin depender del modo de import de pytest.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

import pytest

# Igual que hace ``main.py`` al arrancar: el store de runtimes del checkout se
# inyecta en ``sys.path`` para que los tests puedan importar el torch/onnxruntime
# reales (los que ejercitan conversión .pth → .onnx y carga de checkpoints).
# Sin esto, esos tests dependían de que otro test anterior hubiera llamado a
# ``inject_runtime_paths()``: pasaban o se saltaban según el orden de ejecución.
try:
    from src.runtimes.manager import inject_runtime_paths

    inject_runtime_paths()
except Exception:  # el store puede no existir (checkout recién clonado)
    pass

# Funciones originales que ``isolated_runtime`` dobla. Los tests que necesitan
# el comportamiento real (resolución contra índices) las usan desde aquí en
# lugar de depender de cómo monkeypatch guarda el valor previo.
_ORIGINALS: dict = {}


@pytest.fixture()
def qapp():
    """A QApplication configured for the offscreen platform.

    If the offscreen platform is unavailable (no Qt platform plugin at all),
    the test is skipped rather than crashing the whole suite.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        # ``-platform offscreen`` as an argv token is the most reliable way to
        # force the offscreen plugin across environments (a bare env-var set
        # can be ignored by some PySide6/Platform builds and segfaults at
        # construction time in headless sandboxes).
        app = QApplication(["draggy-test", "-platform", "offscreen"])
    return app


@pytest.fixture(autouse=True, scope="session")
def isolated_logs_dir(tmp_path_factory):
    """Redirige la carpeta de logs (``bin/logs``) a un temporal del test.

    Dos motivos, los dos medidos en un checkout real:

    * Los tests que lanzan un render escribian el ``render_*.json`` de verdad
      en ``bin/logs``: **3109 ficheros / 21 MB**, casi todos de ejecuciones de
      pytest, en la carpeta que el usuario abre desde Preferences -> Logs.
    * Los logs de render se podan por antiguedad al escribir uno nuevo. Sin
      aislar, un test que renderiza PODARIA los logs reales del desarrollador.

    Es autouse (y de sesion) a proposito: cualquier test puede renderizar sin
    saberlo, asi que el aislamiento tiene que ser la tonica y no una decision
    que cada test tenga que recordar. Los tests que necesitan una carpeta
    propia siguen parcheando ``get_logs_dir`` como hasta ahora (el monkeypatch
    de cada test gana mientras dura el test).
    """
    logs = str(tmp_path_factory.mktemp("logs"))
    mp = pytest.MonkeyPatch()
    # ``get_logs_dir`` se parchea en cada módulo que lo captura al importarse
    # (``from src.config import get_logs_dir``): los que lo importan dentro de
    # una función ya quedan cubiertos por ``src.config``, pero estos lo miran a
    # nivel de módulo y son los que ESCRIBEN de verdad en la carpeta
    # (app.log, informes de crash, backup/, avisos de actualización).
    import src.config as _config
    import src.crash_report as _crash_report
    import src.log_backup as _log_backup
    import src.processing.pipeline as _pipeline
    import src.ui.dependencies_dialog as _dependencies_dialog
    import src.ui.preferences_dialog as _preferences_dialog

    for module in (_config, _crash_report, _log_backup, _pipeline,
                   _dependencies_dialog, _preferences_dialog):
        mp.setattr(module, "get_logs_dir", lambda: logs, raising=False)
    # Estos tres calculan la carpeta con os.path.join(get_bin_dir(), "logs").
    import src.ffmpeg.codec_tester as _codec_tester
    import src.ffmpeg.loudnorm as _loudnorm
    import src.ffmpeg.runner as _runner

    for module in (_codec_tester, _loudnorm, _runner):
        mp.setattr(module, "_log_dir", lambda: logs, raising=False)
    try:
        yield logs
    finally:
        mp.undo()


@pytest.fixture()
def isolated_runtime(tmp_path_factory, monkeypatch):
    """Store de runtimes temporal y sin red, para tests de instalación.

    Hace falta por tres motivos, todos aprendidos de fallos reales:

    * ``install_*`` instalan en staging y luego dan un swap del store
      (renombrar el viejo a ``.previous`` y mover el nuevo a su sitio). Sin
      aislar, un test podía renombrar y BORRAR el store real del checkout
      (~3 GB) que usa el desarrollador.
    * ``resolve_pytorch_plan`` consulta los índices de PyTorch y PyPI: sin
      anclaje, un test de instalación haría I/O de red no determinista.
    * El auto-install de arranque llama a los instaladores cuando no detecta
      runtime; con el store de test vacío ese camino descargaría GB de verdad.
      Se corta en el propio instalador, que es el punto por el que pasa toda
      descarga.

    Devuelve la ruta del store temporal para los tests que la necesiten.
    """
    import sys

    # Import por submódulo (no por paquete): así ``manager`` es SIEMPRE el
    # mismo objeto que el que parchea monkeypatch, sin depender de cómo se
    # resolvió ``src.runtimes.manager`` en cada test.
    import src.runtimes.manager  # noqa: F401  (registra el submódulo)
    manager = sys.modules["src.runtimes.manager"]
    _ORIGINALS.setdefault("resolve_pytorch_plan", manager.resolve_pytorch_plan)
    # Los tests de instalación ejercitan el instalador de verdad (con pip
    # doblado): necesitan la función original, porque este fixture la bloquea
    # para que ningún test dispare una descarga real por accidente.
    _ORIGINALS.setdefault("install_pytorch", manager.install_pytorch)
    _ORIGINALS.setdefault("install_onnxruntime", manager.install_onnxruntime)
    _ORIGINALS.setdefault("install_runtimes", manager.install_runtimes)

    fake_bin = tmp_path_factory.mktemp("bin")
    store = fake_bin / "runtimes" / "ai_modules"
    store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(manager, "RUNTIME_MODULES_DIR", str(store))
    monkeypatch.setattr(manager, "get_bin_dir", lambda: str(fake_bin))

    def _no_download(*args, **kwargs):
        raise RuntimeError(
            "descarga deshabilitada en tests (isolated_runtime): este test ha "
            "llegado a un instalador real de runtimes de IA. Si el test "
            "ejercita el instalador a propósito, llama a "
            "conftest._ORIGINALS['install_pytorch']."
        )

    monkeypatch.setattr(manager, "install_pytorch", _no_download)
    monkeypatch.setattr(manager, "install_onnxruntime", _no_download)
    monkeypatch.setattr(manager, "install_runtimes", _no_download)

    # Resolver sin red: el set conocido bueno sobre el índice preferido.
    monkeypatch.setattr(
        manager,
        "resolve_pytorch_plan",
        lambda variant="auto", use_cache=True: {
            "variant": manager._resolve_variant(variant),
            "versions": {},
            "source": "offline",
            "note": "tests",
            "attempts": [],
        },
    )
    # La metadata de PyPI (versión de Python por release) tampoco se consulta:
    # los tests que la necesiten la doblan explícitamente.
    monkeypatch.setattr(
        manager, "_pypi_release_requires_python", lambda package, version: None
    )
    monkeypatch.setattr(manager, "_pypi_versions_newest_first", lambda package: [])

    yield str(store)

    # El store temporal se va con tmp_path; el checkout no se toca nunca.
    for leftover in (manager._staging_dir(), manager._aside_dir()):
        shutil.rmtree(leftover, ignore_errors=True)


def pytest_runtest_setup(item):
    """Attach the offscreen platform to the Qt tests even if imported lazily."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
