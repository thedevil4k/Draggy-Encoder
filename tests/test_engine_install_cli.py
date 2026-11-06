"""Focused tests for the headless Add Engine entrypoint.

The real download is exercised against the installed EXE (network + model
weights), so these tests keep the fast path deterministic: argument parsing
and the non-blocking message-box shim used when the packaged app has no
console.
"""

from __future__ import annotations


def test_parse_engine_id_accepts_separate_and_equals_forms():
    from src.cli_install_engine import _parse_engine_id

    assert _parse_engine_id(["--install-engine", "upscale-demo"]) == "upscale-demo"
    assert _parse_engine_id(["--install-engine=upscale-demo"]) == "upscale-demo"
    assert _parse_engine_id(["--install-engine"]) == ""
    assert _parse_engine_id(["--selftest"]) == ""


def test_auto_message_box_never_blocks_and_confirms_questions():
    from PySide6.QtWidgets import QMessageBox

    from src.cli_install_engine import _auto_message_box_class

    shim = _auto_message_box_class(QMessageBox)
    assert shim.question(None, "title", "text") == QMessageBox.StandardButton.Yes
    assert shim.information(None, "title", "text") is None
    assert shim.critical(None, "title", "text") is None
    box = shim(None)
    assert box.exec() == 0
    assert box.clickedButton() is None
