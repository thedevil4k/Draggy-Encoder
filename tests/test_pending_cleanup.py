"""Limpieza de archivos bloqueados por Windows (``pending_cleanup``).

Contexto (fallo real): al desinstalar un runtime con sus DLL cargadas, Windows
impide borrar sus archivos. La app lo programa para el arranque siguiente. El
problema era que, si el usuario REINSTALABA el mismo runtime antes de
reiniciar, la ruta registrada apuntaba ya a la instalación NUEVA: al arrancar,
la limpieza pendiente borraba lo recién instalado y el runtime desaparecía sin
que nadie lo hubiera pedido.
"""
from __future__ import annotations

import json
import os

import pytest

import src.fsops as fsops
import src.runtimes.manager as rm


@pytest.fixture()
def spool(tmp_path, monkeypatch):
    """Aísla el fichero de limpiezas pendientes en un directorio temporal."""
    session = tmp_path / ".session"
    session.mkdir(parents=True, exist_ok=True)
    path = str(session / "pending_cleanup.json")
    monkeypatch.setattr(fsops, "_pending_cleanup_path", lambda: path)
    monkeypatch.setattr(fsops, "ensure_runtime_dirs", lambda: None)
    return path


def _write_pending(path: str, entries: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"paths": entries}, fh)


def _read_pending(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["paths"]


def test_pending_cleanup_deletes_registered_directory(spool, tmp_path):
    """Caso normal: el directorio registrado (y sin tocar) se borra."""
    target = tmp_path / "onnxruntime"
    target.mkdir()
    (target / "capi").mkdir()
    (target / "capi" / "x.dll").write_text("x")

    fsops.register_pending_cleanup(str(target))
    assert os.path.isdir(target)

    fsops.process_pending_cleanup()

    assert not os.path.exists(target)
    assert not os.path.exists(spool), "sin restos, el fichero debe desaparecer"


def test_pending_cleanup_respects_reinstalled_runtime(spool, tmp_path):
    """EL FALLO REAL: si el directorio se re-creó (reinstalación), no se borra.

    La marca de creación registrada ya no coincide con la del directorio que
    hay ahora, así que es una instalación nueva y se respeta.
    """
    target = tmp_path / "onnxruntime"
    target.mkdir()
    fsops.register_pending_cleanup(str(target))

    # Reinstalación: el usuario borró el directorio y volvió a instalar.
    import shutil
    import time

    shutil.rmtree(target)
    time.sleep(0.01)
    target.mkdir()
    (target / "__init__.py").write_text("__version__ = '1.30.0'\n")

    fsops.process_pending_cleanup()

    assert os.path.isdir(target), "la instalación nueva no debe borrarse"
    assert os.path.isfile(os.path.join(target, "__init__.py"))
    # La entrada se descarta: no hay nada que borrar, y dejarla viva solo
    # haría que cada arranque reevaluase una ruta que ya está resuelta.
    assert not os.path.exists(spool)


def test_pending_cleanup_handles_legacy_plain_path_format(spool, tmp_path):
    """Compatibilidad: el formato antiguo era una lista de rutas sueltas."""
    target = tmp_path / "torch"
    target.mkdir()
    (target / "lib").mkdir()
    _write_pending(spool, [str(target)])

    fsops.process_pending_cleanup()

    assert not os.path.exists(target)
    assert not os.path.exists(spool)


def test_pending_cleanup_ignores_missing_paths(spool, tmp_path):
    """Una ruta que ya no existe se descarta sin dejar el fichero colgado."""
    _write_pending(spool, [str(tmp_path / "ya-no-esta")])

    fsops.process_pending_cleanup()

    assert not os.path.exists(spool)


def test_pending_cleanup_removes_empty_leftover_directory(spool, tmp_path):
    """Si solo queda la carcasa vacía (los DLL bloqueados se fueron con el
    proceso), se retira en vez de dejarla para siempre."""
    target = tmp_path / "onnxruntime"
    (target / "capi").mkdir(parents=True)
    fsops.register_pending_cleanup(str(target))

    # El árbol ya está vacío salvo una carpeta sin archivos.
    (target / "capi").rmdir()
    fsops.process_pending_cleanup()

    assert not os.path.exists(target)


def test_register_replaces_previous_entry_for_same_path(spool, tmp_path):
    """Registrar dos veces la misma ruta no duplica entradas ni pierde la marca."""
    target = tmp_path / "torch"
    target.mkdir()
    fsops.register_pending_cleanup(str(target))
    fsops.register_pending_cleanup(str(target))

    entries = _read_pending(spool)
    paths = [fsops._pending_entry_path(e) for e in entries]
    assert paths == [str(target)]
    assert fsops._pending_entry_signature(entries[0]) != 0


def test_uninstall_defers_when_files_are_locked(monkeypatch, tmp_path):
    """Si ``fsops.rmtree_force`` no puede completar el borrado, el
    desinstalador lo reporta (False) para que la UI avise en vez de
    mentir."""
    store = str(tmp_path / "runtimes" / "ai_modules")
    os.makedirs(os.path.join(store, "onnxruntime"), exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(fsops, "rmtree_force", lambda path: False)

    complete = rm.uninstall_runtime("onnxruntime")

    assert complete is False
