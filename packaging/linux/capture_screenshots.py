"""Capture representative UI screenshots for AppStream metadata.

This runs only in the Linux GitHub Actions job with Qt's offscreen platform;
the generated PNGs are uploaded as release assets referenced by metainfo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The script lives in packaging/linux/, but imports the app from src/.
# Put the repository root on sys.path so this works no matter the cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PySide6.QtWidgets import QApplication

from src.qt_check import run_ui_checks
from src.ui.dependencies_dialog import DependenciesDialog
from src.ui.main_window import MainWindow


def _save(widget, path: Path) -> None:
    widget.grab().save(str(path), "PNG")


def main() -> int:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Las mismas validaciones que el selftest (run_ui_checks en
    # src/qt_check.py), ANTES de generar nada: si Qt no puede crear la
    # QApplication en offscreen, o la MainWindow o el DependenciesDialog
    # no se construyen, los screenshots serían basura (o el script
    # reventaría a mitad). Falla con mensaje claro y exit 1.
    for name, status, detail in run_ui_checks():
        if status != "ok":
            print(f"ERROR: {name} check failed: {detail}", file=sys.stderr)
            return 1

    app = QApplication.instance() or QApplication(["draggy-encoder-screenshot"])
    window = MainWindow(
        "",
        {"recommended_device": "CPU", "gpus": []},
        {"libx264": True, "libx265": True},
    )
    window.resize(1280, 900)
    window.show()
    app.processEvents()
    _save(window, output_dir / "DraggyEncoder-screenshot-workflow.png")

    dialog = DependenciesDialog(window, initial_tab="ai_engines")
    dialog.resize(1200, 820)
    dialog.show()
    app.processEvents()
    _save(dialog, output_dir / "DraggyEncoder-screenshot-engines.png")

    dialog.close()
    window.close()
    app.processEvents()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
