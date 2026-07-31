"""QThread wrapper around ``download_ffmpeg_full``.

Kept separate from ``downloader.py`` so that the pure download function
remains usable from headless contexts (tests, CLI) without dragging
PySide6 as a dependency.

The worker emits progress signals compatible with the 2-arg contract
``(int percent, str message)`` already used elsewhere in the app via
``_emit`` polyfills.
"""

from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import QThread, Signal

from src.ffmpeg.downloader import download_ffmpeg_full

logger = logging.getLogger(__name__)


class FFmpegFullDownloadWorker(QThread):
    """Background worker that downloads the FFmpeg "full" build (with libplacebo).

    Signals
    -------
    progress(int, str)   - emitted from inside download_ffmpeg_full via the
                           polyfilled callback. ``str`` may be empty.
    finished_ok(str)     - emitted on success with the absolute path to the
                           freshly-installed ``ffmpeg.exe`` / ``ffmpeg`` binary.
    failed(str)          - emitted on failure with the exception message.

    Only one of ``finished_ok`` / ``failed`` is emitted per ``start()``.
    """

    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation. The download loop will exit between chunks."""
        self._cancelled = True

    def _on_progress(self, percent: int, message: str = "") -> None:
        try:
            self.progress.emit(int(percent), str(message or ""))
        except Exception:
            pass

    def run(self) -> None:
        try:
            path = download_ffmpeg_full(progress_callback=self._on_progress)
            if self._cancelled:
                self.failed.emit("cancelled by user")
                return
            self.finished_ok.emit(path)
        except Exception as exc:
            logger.error("FFmpeg full download failed: %s", exc, exc_info=True)
            # Attach traceback to the message so the UI dialog shows something
            # actionable. Traceback itself is verbose; keep last 3 lines.
            tb = traceback.format_exc().strip().splitlines()
            tail = "\n".join(tb[-3:]) if len(tb) >= 3 else "\n".join(tb)
            self.failed.emit("{}\n\n{}".format(exc, tail))


__all__ = ["FFmpegFullDownloadWorker"]
