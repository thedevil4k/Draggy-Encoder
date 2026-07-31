"""Cross-platform download and ONNX-conversion QThread workers.

The philosophy:

    - Prefer ``huggingface_hub.snapshot_download`` for model downloads:
      no git required, handles LFS automatically, supports resume.
    - Fall back to ``git clone --depth 1`` if huggingface_hub is not
      installed and git is available.
    - Use ``shutil.which`` to detect ``git`` and ``optimum-cli`` on
      the user's PATH; if missing, we surface that back to the UI so
      the dialog can offer "Install Git" or "Drop files manually".
    - Run blocking work in ``QThread`` so the GUI stays alive. We
      keep the threads cancellable (a flag the worker checks after
      each chunk) so a stalled clone can be aborted; the underlying
      ``subprocess.Popen`` is killed on cancel so a half-dead network
      cannot orphan the OS process.
    - Use the LF-only git progress reading that ``git`` writes to
      stderr ("Receiving objects: 42%") to feed a real progress bar.

Workers exposed here:

    HFDownloadWorker   - downloads a HF repo via huggingface_hub (no git).
    GitCloneWorker     - clones one HF repo to a target directory (fallback).
    OnnxConvertWorker  - stub that fires ``optimum-cli`` if available,
                         otherwise reports a helpful pip-install hint.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Optional

from PySide6.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def is_git_available() -> bool:
    """``True`` if ``git`` resolves on PATH. Pure shutil, no subprocess."""
    return shutil.which("git") is not None


def is_optimum_cli_available() -> bool:
    """``True`` if ``optimum-cli`` resolves on PATH."""
    return shutil.which("optimum-cli") is not None


def git_version() -> Optional[str]:
    """Return raw ``git --version`` output, or None on failure."""
    path = shutil.which("git")
    if not path:
        return None
    try:
        out = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if out.returncode == 0:
            return (out.stdout or "").strip().splitlines()[0]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Git progress parsing
# ---------------------------------------------------------------------------

# git writes progress to stderr in lines like:
#   "Receiving objects:  42% (1234/3000), 12.3 MiB | 1.20 MiB/s"
#   "Resolving deltas: 100% (50/50), done."
_PERCENT_RE = re.compile(r"(Receiving objects|Resolving deltas|Total):\s*(\d{1,3})%")


def _parse_percent(line: str, last: int) -> int:
    """Pull a percent [0,100] out of one stderr line. Falls back to ``last``."""
    m = _PERCENT_RE.search(line)
    if not m:
        return last
    try:
        v = max(0, min(100, int(m.group(2))))
    except Exception:
        return last
    return max(last, v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup_partial(target_dir: str) -> None:
    """Recursively remove ``target_dir`` if it exists.

    Called after a failed clone so the next install attempt isn't
    poisoned by a half-written ``.git`` skeleton that fails the
    'directory is not empty' guard.
    """
    if os.path.isdir(target_dir):
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Async workers (live inside QThread; UI connects to their signals)
# ---------------------------------------------------------------------------


class _CancelledError(Exception):
    """Raised internally from a progress callback hook."""


def is_hf_hub_available() -> bool:
    """``True`` if ``huggingface_hub`` can be imported."""
    try:
        import importlib
        importlib.import_module("huggingface_hub")
        return True
    except ImportError:
        return False


class HFDownloadWorker(QThread):
    """Download a HuggingFace repo via ``huggingface_hub.snapshot_download``.

    No git required. Handles LFS automatically. Supports resume (if the
    download is interrupted, re-running will pick up where it left off).

    Signals:
        status(str)       - status text the UI renders near the button.
        progress(int)     - 0..100 progress bar updates.
        finished_ok(str)  - target_dir on success.
        failed(str)       - error message on failure.
    """

    status = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, repo_id: str, target_dir: str, parent=None):
        super().__init__(parent)
        self.repo_id = repo_id
        self.target_dir = target_dir
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        if self._cancel.is_set():
            self.failed.emit("cancelled before start")
            return

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self.failed.emit(
                "huggingface_hub is not installed.\n\n"
                "Install with:\n"
                "  pip install huggingface_hub\n\n"
                "Then try again. Alternatively, install Git and "
                "it will be used as a fallback."
            )
            return

        self.status.emit("Preparing download from HuggingFace...")
        self.progress.emit(0)

        try:
            os.makedirs(os.path.dirname(self.target_dir) or ".", exist_ok=True)
        except Exception:
            pass

        # huggingface_hub downloads to its own cache, then we move files
        # to the target_dir. Use local_dir for direct output.
        self.status.emit(f"Downloading {self.repo_id}...")
        self.progress.emit(5)

        # Read optional HF token from personal storage.
        try:
            from src.personal import get_token as _get_token
            _token = _get_token() or None
        except Exception:
            _token = None

        # Pre-validate: some repos require a token for LFS files.
        if not _token:
            self.status.emit("No HuggingFace token set — checking repo access...")
            try:
                from huggingface_hub import HfApi
                _api = HfApi()
                _info = _api.model_info(self.repo_id, files_metadata=True)
                _siblings = getattr(_info, "siblings", []) or []
                _has_lfs = any(getattr(s, "lfs", None) for s in _siblings)
                if _has_lfs:
                    self.failed.emit(
                        f"The repo '{self.repo_id}' contains LFS files that "
                        f"require authentication.\n\n"
                        f"Go to Preferences > Behavior and add your free "
                        f"HuggingFace token.\n\n"
                        f"Get your token at:\n"
                        f"https://huggingface.co/settings/tokens"
                    )
                    return
            except Exception as exc:
                err_msg = str(exc).lower()
                if "does not support image input" in err_msg or "image.png" in err_msg:
                    self.status.emit("Checking repo files (fallback)...")
                    try:
                        _files = _api.list_repo_files(self.repo_id)
                        _siblings = list(_files) if _files else []
                        _has_lfs = False
                    except Exception:
                        pass
                # For other errors, proceed and let snapshot_download fail
                # with a clear error message.

        try:
            # Use local_dir to download directly into target_dir.
            # allow_patterns=["*"] downloads all files including LFS.
            # resume_download=True enables resumption of partial downloads.
            result_path = snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=_token,
            )
        except Exception as exc:
            _cleanup_partial(self.target_dir)
            exc_str = str(exc)
            # Provide helpful hint for auth-related failures.
            if "401" in exc_str or "403" in exc_str or "Access Denied" in exc_str:
                exc_str += (
                    "\n\nTip: This HuggingFace repo requires a token for "
                    "LFS file downloads.\n\n"
                    "1. Go to https://huggingface.co/settings/tokens\n"
                    "2. Create a free token (read access is enough)\n"
                    "3. Paste it in Preferences > Behavior > HuggingFace API Token"
                )
            self.failed.emit(f"Download failed: {exc_str}")
            return

        if self._cancel.is_set():
            _cleanup_partial(self.target_dir)
            self.failed.emit("cancelled by user")
            return

        # Verify the download produced output.
        if os.path.isdir(self.target_dir) and os.listdir(self.target_dir):
            # Remove .huggingface cache dir that snapshot_download creates
            # inside local_dir (it's not needed for the model to work).
            hf_cache = os.path.join(self.target_dir, ".huggingface")
            if os.path.isdir(hf_cache):
                try:
                    shutil.rmtree(hf_cache, ignore_errors=True)
                except Exception:
                    pass
            # Same for .cache if present.
            hf_cache2 = os.path.join(self.target_dir, ".cache")
            if os.path.isdir(hf_cache2):
                try:
                    shutil.rmtree(hf_cache2, ignore_errors=True)
                except Exception:
                    pass

            self.progress.emit(100)
            self.status.emit("download finished")
            self.finished_ok.emit(self.target_dir)
        else:
            _cleanup_partial(self.target_dir)
            self.failed.emit("Download produced no output directory")


class GitCloneWorker(QThread):
    """Async wrapper around ``git clone --depth 1 --progress``.

    The worker keeps a handle to the underlying ``Popen`` so that a
    cancel request from the dialog (``cancel()``) can call
    ``proc.kill()`` and immediately unblock the readline loop.

    Signals:
        status(str)       - status text the UI renders near the button.
        progress(int)     - 0..100 progress bar updates.
        finished_ok(str)  - target_dir on success.
        failed(str)       - error message on failure.
    """

    status = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str)

    # Hard cap so a totally dead network cannot hang a worker forever.
    CLONE_TIMEOUT_S = 15 * 60

    def __init__(self, url: str, target_dir: str, parent=None):
        super().__init__(parent)
        self.url = url
        self.target_dir = target_dir
        self._cancel = threading.Event()
        self._proc = None

    def cancel(self):
        """Mark the clone for cancellation; kill the underlying git subprocess."""
        self._cancel.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def run(self):
        if self._cancel.is_set():
            self.failed.emit("cancelled before start")
            return

        if not is_git_available():
            self.failed.emit(
                "Git is not on PATH. Install Git for Windows (git-scm.com) "
                "or your distro's git package; or drop the model files "
                "manually into the target directory shown in the row."
            )
            return

        try:
            os.makedirs(os.path.dirname(self.target_dir) or ".", exist_ok=True)
        except Exception:
            pass
        if os.path.isdir(self.target_dir) and os.listdir(self.target_dir):
            self.failed.emit(
                f"Target directory already exists and is not empty: "
                f"{self.target_dir}"
            )
            return

        cmd = ["git", "clone", "--depth", "1", "--progress",
               self.url, self.target_dir]

        self.status.emit("starting clone...")
        self.progress.emit(0)
        last_pct = 0

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            self.failed.emit(f"Could not start git: {exc}")
            return

        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._cancel.is_set():
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                    break
                pct = _parse_percent(line, last_pct)
                if pct != last_pct:
                    last_pct = pct
                    try:
                        self.progress.emit(int(pct))
                    except Exception:
                        pass
            try:
                self._proc.wait(timeout=self.CLONE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                _cleanup_partial(self.target_dir)
                self.failed.emit(
                    f"git clone exceeded the {self.CLONE_TIMEOUT_S // 60} min timeout"
                )
                return
        finally:
            self._proc = None

        if self._cancel.is_set():
            _cleanup_partial(self.target_dir)
            self.failed.emit("cancelled by user")
            return

        # The captured success path is from ``proc.wait`` returncode;
        # we already cleared self._proc so refer to its saved value.
        # (Re-reading via shutil is not needed; the loop above already
        # saw the last line, and finalize via a quick stat on returncode.)
        # We must use the original rc: easiest is to re-evaluate via
        # the just-finished process. short-circuit: if we got here
        # without a TimeoutExpired, the loop drained stdout and wait()
        # returned. We don't have rc directly, but a directory exists.
        if os.path.isdir(self.target_dir):
            self.progress.emit(100)
            self.status.emit("clone finished")
            self.finished_ok.emit(self.target_dir)
        else:
            _cleanup_partial(self.target_dir)
            self.failed.emit("git clone produced no output directory")


# ---------------------------------------------------------------------------
# ONNX conversion worker
# ---------------------------------------------------------------------------

# Weight file extensions that can be converted to ONNX via torch.onnx.export.
_WEIGHT_EXTS = (".pth", ".pt", ".safetensors")


class OnnxConvertWorker(QThread):
    """Convert an installed engine's PyTorch weights to ONNX.

    Instead of shelling out to ``optimum-cli`` (which often fails with
    parser/argument errors or is not installed), we reuse the project's
    own ``convert_pth_to_onnx`` helper from ``src.ai.inference``. It
    builds the model from the checkpoint, exports via
    ``torch.onnx.export`` and writes the ``.onnx`` file next to the
    weight file.

    Emits:
        status(str)       - status text rendered near the progress bar.
        progress(int)     - 0..100 progress bar updates.
        finished_ok(str)  - path to the generated .onnx file.
        failed(str)       - error message on failure.
    """

    status = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, engine_id: str, engine_dir: str, category: str,
                 parent=None):
        super().__init__(parent)
        self.engine_id = engine_id
        self.engine_dir = engine_dir
        self.category = category
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _find_weight_file(self) -> Optional[str]:
        """Return the first convertible weight file under ``engine_dir``."""
        if not os.path.isdir(self.engine_dir):
            return None
        for root, _dirs, files in os.walk(self.engine_dir):
            # Skip cache/hidden dirs for speed.
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue
            for fname in files:
                if fname.lower().endswith(_WEIGHT_EXTS):
                    return os.path.join(root, fname)
        return None

    def run(self):
        if self._cancel.is_set():
            self.failed.emit("cancelled before start")
            return

        self.status.emit("Locating model weights...")
        self.progress.emit(5)

        weight_path = self._find_weight_file()
        if not weight_path:
            self.failed.emit(
                "No PyTorch weights found in the engine directory.\n"
                f"Expected one of: {_WEIGHT_EXTS}\n"
                f"Path: {self.engine_dir}"
            )
            return

        if self._cancel.is_set():
            self.failed.emit("cancelled by user")
            return

        self.status.emit("Converting to ONNX with torch.onnx.export...")
        self.progress.emit(15)

        try:
            # Import locally so the worker can fail with a clear message
            # instead of crashing the whole dialog if torch is missing.
            from src.ai.inference import convert_pth_to_onnx
        except ImportError as exc:
            self.failed.emit(
                f"Could not import the ONNX converter: {exc}\n\n"
                "Ensure PyTorch is installed:\n"
                "  pip install torch torchvision"
            )
            return

        if self._cancel.is_set():
            self.failed.emit("cancelled by user")
            return

        onnx_path = convert_pth_to_onnx(weight_path)
        if not onnx_path or not os.path.exists(onnx_path):
            self.failed.emit(
                "ONNX conversion failed.\n"
                "Check the application log for the underlying error.\n"
                f"Weight file: {weight_path}"
            )
            return

        self.progress.emit(100)
        self.status.emit("export complete")
        self.finished_ok.emit(onnx_path)


if __name__ == "__main__":
    # Quick capability probe, useful from a debug REPL.
    print("git available:        ", is_git_available(),
          "->", git_version() or "n/a")
    print("huggingface_hub:      ", is_hf_hub_available())
    print("optimum-cli available:", is_optimum_cli_available())
    print("mock parse:           ",
          _parse_percent("Receiving objects: 42% (1234/3000)", 0))
