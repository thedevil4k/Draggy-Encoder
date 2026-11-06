"""Headless model installation for the installed DraggyEncoder executable.

This is the same path as the GUI's AI Engines -> "Add Engine" button: it
constructs the real ``DependenciesDialog``, calls ``_on_engine_install`` and
waits for its ``HFDownloadWorker``.  It exists so a model download can be
driven and verified from the packaged app without a human clicking, while
still exercising the production download code instead of a separate test
client.

The command writes a detailed log to ``bin/logs/engine-install.log`` because
the packaged executable is windowed (``sys.stdout``/``sys.stderr`` are None).
After the download it also loads the new weights in the AI host and processes
one synthetic frame, so "installed" means the engine is operational rather
than merely present on disk.
"""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


def _parse_engine_id(argv: list[str]) -> str:
    """Return the ID after ``--install-engine`` (or ``--install-engine=ID``)."""
    for index, arg in enumerate(argv):
        if arg == "--install-engine":
            if index + 1 < len(argv):
                return str(argv[index + 1]).strip()
            return ""
        if arg.startswith("--install-engine="):
            return arg.split("=", 1)[1].strip()
    return ""


def _auto_message_box_class(real_qmessagebox):
    """A non-blocking QMessageBox replacement for the headless flow."""

    class _AutoMessageBox:
        StandardButton = real_qmessagebox.StandardButton
        Icon = real_qmessagebox.Icon
        ButtonRole = real_qmessagebox.ButtonRole

        def __init__(self, *args, **kwargs):
            self._clicked = None

        @staticmethod
        def question(*args, **kwargs):
            return real_qmessagebox.StandardButton.Yes

        @staticmethod
        def information(*args, **kwargs):
            return None

        @staticmethod
        def warning(*args, **kwargs):
            return None

        @staticmethod
        def critical(*args, **kwargs):
            return None

        def setIcon(self, *args, **kwargs):
            return None

        def setWindowTitle(self, *args, **kwargs):
            return None

        def setText(self, *args, **kwargs):
            return None

        def addButton(self, *args, **kwargs):
            return object()

        def exec(self):
            return 0

        def clickedButton(self):
            return self._clicked

    return _AutoMessageBox


def _write_log(text: str) -> None:
    """Append to the install log without assuming stdout exists."""
    try:
        from src.config import get_logs_dir

        path = os.path.join(get_logs_dir(), "engine-install.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
    except Exception:
        pass


def run_install_engine_cli(argv: list[str]) -> int:
    """Download one catalog engine through the real Add Engine flow.

    Returns 0 only when the complete repo has been downloaded, the engine is
    persisted as installed, and the AI host can load it and process a frame.
    """
    from src.config import configure_process_temp, ensure_runtime_dirs, get_logs_dir

    ensure_runtime_dirs()
    try:
        configure_process_temp()
    except Exception:
        pass

    engine_id = _parse_engine_id(argv)
    started = time.time()
    _write_log(f"=== engine install {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    _write_log(f"engine_id={engine_id!r}")

    if not engine_id:
        _write_log("ERROR: missing engine id (use --install-engine <id>)")
        return 2

    # The frozen EXE has no console.  Offscreen Qt lets the real dialog and
    # its QThread worker run without opening a window; the offscreen plugin is
    # part of the PySide6 bundle (verified by the Windows build).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = None
    dialog = None
    worker = None
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox as _RealQMessageBox
        import PySide6.QtWidgets as _qtwidgets

        app = QApplication.instance() or QApplication([])
        app.setQuitOnLastWindowClosed(False)

        auto_box = _auto_message_box_class(_RealQMessageBox)
        # Patch both the module imported by dependencies_dialog and the Qt
        # namespace used by its auth-error branch.  Every box would otherwise
        # block forever in a windowed process with no console.
        from src.ui import dependencies_dialog as dd

        dd.QMessageBox = auto_box
        _qtwidgets.QMessageBox = auto_box

        dialog = dd.DependenciesDialog(initial_tab="ai_engines")
        dialog.hide()
        try:
            dialog._check_timer.stop()
        except Exception:
            pass

        _write_log("starting DependenciesDialog._on_engine_install()")
        dialog._on_engine_install(engine_id)
        worker = dialog._git_worker
        if worker is None:
            state = {}
            try:
                from src.ai.engines_state import get_state

                state = get_state(engine_id)
            except Exception:
                pass
            _write_log(
                "ERROR: Add Engine did not start a download "
                f"(installed={state.get('installed')!r}, "
                f"last_error={state.get('last_error')!r})"
            )
            return 2

        worker.status.connect(
            lambda text: _write_log(f"STATUS {text}")
        )
        worker.progress.connect(
            lambda value: _write_log(f"PROGRESS {value}")
        )

        # Keep the GUI event loop pumping while QThread signals are delivered.
        deadline = time.monotonic() + 2 * 60 * 60
        while True:
            app.processEvents()
            if dialog._git_worker is None:
                break
            if not worker.isRunning():
                # The thread just finished; let the queued finished_ok/failed
                # slot run before deciding it is stuck.
                for _ in range(100):
                    app.processEvents()
                    if dialog._git_worker is None:
                        break
                    time.sleep(0.01)
                if dialog._git_worker is None:
                    break
            if time.monotonic() > deadline:
                _write_log("ERROR: download timed out after 2 hours")
                try:
                    worker.cancel()
                except Exception:
                    pass
                return 2
            time.sleep(0.05)

        app.processEvents()

        from src.ai.engines_state import dir_has_weights, get_state
        from src.ai.registry import get_engine_dir, get_engine_meta

        state = get_state(engine_id) or {}
        target_dir = state.get("install_path") or get_engine_dir(engine_id)
        has_weights = dir_has_weights(target_dir)
        _write_log(
            f"state installed={state.get('installed')!r} "
            f"last_error={state.get('last_error')!r} target={target_dir}"
        )
        if not state.get("installed") or not has_weights:
            _write_log(
                "ERROR: Add Engine did not leave an installed engine with weights"
            )
            return 2

        # Load and execute the actual downloaded weights in the same AI host
        # used by the renderer.  This catches a corrupt/incompatible download
        # that would otherwise look installed but fail at the first frame.
        try:
            import numpy as np

            from src.ai.inference import create_inference_engine, pick_weight_file

            weight_path = pick_weight_file(target_dir, engine_id)
            if not weight_path or not os.path.isfile(weight_path):
                raise RuntimeError(f"no usable weight file found in {target_dir}")
            _write_log(f"VERIFY weight={weight_path} bytes={os.path.getsize(weight_path)}")
            engine = create_inference_engine(engine_id, weight_path)
            try:
                if not getattr(engine, "is_loaded", False):
                    raise RuntimeError("AI host did not load the downloaded engine")
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[16:48, 16:48] = 128
                output = engine.process_frame(frame)
                if output is None:
                    raise RuntimeError("AI host returned no frame")
                _write_log(
                    "VERIFY frame_ok input=64x64 output="
                    f"{getattr(output, 'shape', None)} "
                    f"dtype={getattr(output, 'dtype', None)}"
                )
            finally:
                try:
                    engine.unload()
                except Exception:
                    pass
        except Exception as exc:
            _write_log(
                "ERROR: installed engine failed operational verification: "
                f"{type(exc).__name__}: {exc}"
            )
            return 2

        meta = get_engine_meta(engine_id) or {}
        files = []
        for root, _dirs, names in os.walk(target_dir):
            for name in names:
                path = os.path.join(root, name)
                try:
                    files.append((os.path.relpath(path, target_dir), os.path.getsize(path)))
                except OSError:
                    pass
        _write_log(
            f"OK engine={engine_id!r} display_name={meta.get('display_name')!r} "
            f"installed={state.get('installed')!r} files={len(files)} "
            f"bytes={sum(size for _name, size in files)}"
        )
        for name, size in sorted(files):
            _write_log(f"FILE {name} {size}")
        _write_log(f"DONE elapsed={time.time() - started:.1f}s")
        return 0
    except Exception as exc:
        _write_log(f"ERROR: unexpected {type(exc).__name__}: {exc}")
        return 2
    finally:
        try:
            if dialog is not None:
                dialog._kill_workers()
        except Exception:
            pass
        try:
            if dialog is not None:
                dialog.close()
        except Exception:
            pass
        try:
            from src.ai import host_client

            host_client.stop_host()
        except Exception:
            pass
        try:
            if app is not None:
                app.processEvents()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(run_install_engine_cli(sys.argv[1:]))
