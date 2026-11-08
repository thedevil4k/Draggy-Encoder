"""Cross-platform download and ONNX-conversion QThread workers.

The philosophy:

    - ``huggingface_hub`` is the ONLY download path for models: it ships
      bundled inside the packaged app (requirements-build.txt), handles
      LFS automatically and supports resume. The old ``git clone``
      fallback was removed because the Windows installer does not bundle
      Git, so that path was a dead end on the packaged app.
    - Each file is downloaded one by one (``hf_hub_download``) so
      cancellation is honored between files.
    - Run blocking work in ``QThread`` so the GUI stays alive. Workers
      are cancellable (a flag checked between files) and each download
      has its own retry loop with exponential backoff.

Workers exposed here:

    HFDownloadWorker        - downloads a HF repo via huggingface_hub.
    OnnxFileDownloadWorker  - downloads a SINGLE .onnx file (catalog ONNX
                              variant) without PyTorch or conversion.
    OnnxConvertWorker       - converts PyTorch weights to ONNX with the
                              project's own ``convert_pth_to_onnx`` helper.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

_log = logging.getLogger(__name__)

# Se fija también a nivel de módulo para cualquier entrypoint que importe
# este worker sin pasar por main.py. El helper de abajo corrige además el
# caso en que huggingface_hub ya se haya importado (hf_search lo hace para
# el navegador de modelos) y sus constantes ya estén cacheadas.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def is_optimum_cli_available() -> bool:
    """``True`` if ``optimum-cli`` resolves on PATH."""
    return shutil.which("optimum-cli") is not None


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


def _configure_hf_download_environment() -> Optional[str]:
    """Disable HF progress bars and keep its cache off the system drive.

    ``HF_HUB_DISABLE_PROGRESS_BARS`` is read when huggingface_hub is first
    imported. The model browser may import it before this worker runs, so
    setting the environment variable here is not enough; we also flip the
    already-imported constant. ``HF_HOME``/``HF_HUB_CACHE`` are pointed at
    the app's own ``bin/.cache`` so a model download never fills C:.
    """
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    cache_dir = None
    try:
        from src.config import get_bin_dir
        hf_home = os.path.join(get_bin_dir(), ".cache", "huggingface")
        cache_dir = os.path.join(hf_home, "hub")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = cache_dir
    except Exception as exc:
        _log.warning("No se pudo configurar la caché HF en el directorio de datos: %s", exc)
    try:
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_DISABLE_PROGRESS_BARS = True
    except Exception:
        pass
    return cache_dir


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


# Respuestas que Hugging Face usa cuando el repo es gated/privado o cuando
# el token falta/es inválido. Incluye el texto de ``RepositoryNotFoundError``
# para repos privados: HF responde 404 a propósito para no revelar su
# existencia, pero añade "make sure you are authenticated" al mensaje.
_HF_AUTH_ERROR_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "access denied",
    "access to model",
    "restricted",
    "gated",
    "authentication",
    "authenticated",
    "invalid username or password",
    "token",
)


def is_hf_auth_error(message: object) -> bool:
    """True si el error de Hugging Face parece de autenticación/gating."""
    text = str(message or "").lower()
    return any(marker in text for marker in _HF_AUTH_ERROR_MARKERS)


def hf_auth_error_hint(message: object, token_configured: bool) -> str:
    """Devuelve un mensaje accionable para fallos de autenticación de HF.

    La diferencia importante es explícita: si no hay token, el usuario debe
    saber que falta la API key, no ver sólo un 401/403/404 genérico. Si sí
    hay token, el mensaje indica que el token configurado fue rechazado.
    """
    if not is_hf_auth_error(message):
        return ""
    if token_configured:
        return (
            "\n\nHugging Face rejected the configured API token. "
            "Check that it is valid, has Read access, and that your account "
            "has accepted the model's license. Update it in "
            "Preferences > Behavior > HuggingFace API Token."
        )
    return (
        "\n\nHugging Face API token (API key) is NOT configured. "
        "This model requires authentication. Add a free Read token in "
        "Preferences > Behavior > HuggingFace API Token "
        "(https://huggingface.co/settings/tokens), then try again."
    )


class HFDownloadWorker(QThread):
    """Download a HuggingFace repo via ``huggingface_hub``.

    No git required. Handles LFS automatically. Files are downloaded one
    by one (``hf_hub_download``) so cancellation is honored between
    files instead of only at the end of one big ``snapshot_download``.

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

        # In PyInstaller GUI builds (console=False), sys.stderr is None.
        # HuggingFace's tqdm progress bars crash trying to write to it.
        # Disable them — the worker already manages its own progress via
        # Qt signals. Also keep the HF cache beside the app's data.
        cache_dir = _configure_hf_download_environment()

        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            self.failed.emit(
                "huggingface_hub is not available, so models cannot be "
                "downloaded.\n\n"
                "In the packaged app huggingface_hub is bundled: reinstall "
                "the application to restore it.\n"
                "In a source checkout, install it with:\n"
                "  pip install huggingface_hub\n\n"
                "Then try again. Alternatively, drop the model files manually "
                "into the engine folder and re-open the AI Engines tab."
            )
            return

        self.status.emit("Preparing download from HuggingFace...")
        self.progress.emit(0)

        try:
            os.makedirs(self.target_dir, exist_ok=True)
        except Exception as exc:
            _log.warning("makedirs failed for %s: %s", self.target_dir, exc)

        self.status.emit(f"Downloading {self.repo_id}...")
        self.progress.emit(5)

        # Read optional HF token from personal storage.
        try:
            from src.personal import get_token as _get_token
            _token = _get_token() or None
        except Exception:
            _token = None

        token_configured = bool(_token)
        if not token_configured:
            self.status.emit(
                "No Hugging Face API token configured — public models only."
            )

        try:
            repo_files = HfApi(token=_token).list_repo_files(self.repo_id)
        except Exception as exc:
            _cleanup_partial(self.target_dir)
            exc_str = str(exc)
            exc_str += hf_auth_error_hint(exc_str, token_configured)
            self.failed.emit(f"Could not read repo file list: {exc_str}")
            return

        if not repo_files:
            _cleanup_partial(self.target_dir)
            self.failed.emit("Repo contains no files to download.")
            return

        total = len(repo_files)
        MAX_RETRIES = 3
        RETRY_DELAY_BASE = 5  # seconds

        try:
            for idx, filename in enumerate(repo_files):
                if self._cancel.is_set():
                    _cleanup_partial(self.target_dir)
                    self.failed.emit("cancelled by user")
                    return
                if total > 1:
                    self.status.emit(
                        f"Downloading {filename} ({idx + 1}/{total})..."
                    )
                else:
                    self.status.emit(f"Downloading {self.repo_id}...")
                self.progress.emit(5 + int(90 * idx / total))

                # Retry loop per file with exponential backoff.
                last_error = None
                for attempt in range(MAX_RETRIES):
                    if self._cancel.is_set():
                        _cleanup_partial(self.target_dir)
                        self.failed.emit("cancelled by user")
                        return
                    try:
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=filename,
                            local_dir=self.target_dir,
                            cache_dir=cache_dir,
                            token=_token,
                        )
                        last_error = None
                        break  # success
                    except Exception as exc:
                        last_error = exc
                        _log.warning(
                            "hf_hub_download attempt %d/%d failed for %s: %s",
                            attempt + 1, MAX_RETRIES, filename, exc,
                        )
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_DELAY_BASE * (2 ** attempt)
                            self.status.emit(
                                f"Retry {attempt + 2}/{MAX_RETRIES} for "
                                f"{filename} in {delay}s..."
                            )
                            time.sleep(delay)

                if last_error is not None:
                    _cleanup_partial(self.target_dir)
                    exc_str = str(last_error)
                    exc_str += hf_auth_error_hint(exc_str, token_configured)
                    if "NoneType" in exc_str and "write" in exc_str:
                        exc_str += (
                            "\n\nTip: This may be a permissions or disk space issue. "
                            "Try running as administrator or free up disk space."
                        )
                    self.failed.emit(
                        f"Download failed after {MAX_RETRIES} attempts: {exc_str}"
                    )
                    return
        except Exception as exc:
            _cleanup_partial(self.target_dir)
            self.failed.emit(f"Download failed: {exc}")
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


# ---------------------------------------------------------------------------
# Single-file ONNX download worker
# ---------------------------------------------------------------------------

# Nombre por defecto para un archivo ONNX descargado directamente a la raíz
# del directorio del motor. Los catálogos pueden declarar un subpath
# (p.ej. "onnx/model.onnx"); al descargarlo se conserva la estructura del
# repo de HF para que find_onnx_model() lo localice en cualquiera de los dos
# sitios (raíz o subdir onnx/).


class OnnxFileDownloadWorker(QThread):
    """Descarga UN archivo .onnx concreto de un repo HuggingFace.

    A diferencia de ``HFDownloadWorker`` (que baja el repo completo), este
    worker descarga un unico archivo — la variante ONNX del catalogo — sin
    necesitar PyTorch ni conversion. Reutiliza ``hf_hub_download`` (LFS +
    resume) con la misma politica de reintentos y cancelacion que el worker
    de repo completo.

    Signals:
        status(str)       - status text near the progress bar.
        progress(int)     - 0..100 progress updates.
        finished_ok(str)  - path to the downloaded .onnx file.
        failed(str)       - error message on failure.
    """

    status = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, repo_id: str, filename: str, target_dir: str,
                 parent=None):
        super().__init__(parent)
        self.repo_id = repo_id
        self.filename = filename
        self.target_dir = target_dir
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        if self._cancel.is_set():
            self.failed.emit("cancelled before start")
            return

        cache_dir = _configure_hf_download_environment()

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            self.failed.emit(
                "huggingface_hub is not available, so models cannot be "
                "downloaded.\n\n"
                "In the packaged app huggingface_hub is bundled: reinstall "
                "the application to restore it.\n"
                "In a source checkout, install it with:\n"
                "  pip install huggingface_hub"
            )
            return

        self.status.emit("Preparing ONNX download from HuggingFace...")
        self.progress.emit(0)

        try:
            os.makedirs(self.target_dir, exist_ok=True)
        except Exception as exc:
            _log.warning("makedirs failed for %s: %s", self.target_dir, exc)

        self.status.emit(
            f"Downloading {self.repo_id} / {self.filename}..."
        )
        self.progress.emit(5)

        try:
            from src.personal import get_token as _get_token
            _token = _get_token() or None
        except Exception:
            _token = None

        token_configured = bool(_token)
        if not token_configured:
            self.status.emit(
                "No Hugging Face API token configured — public models only."
            )

        MAX_RETRIES = 3
        RETRY_DELAY_BASE = 5  # seconds

        last_error = None
        final_path = None
        for attempt in range(MAX_RETRIES):
            if self._cancel.is_set():
                _cleanup_partial(self.target_dir)
                self.failed.emit("cancelled by user")
                return
            try:
                final_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_dir=self.target_dir,
                    cache_dir=cache_dir,
                    token=_token,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                _log.warning(
                    "hf_hub_download attempt %d/%d failed for %s: %s",
                    attempt + 1, MAX_RETRIES, self.filename, exc,
                )
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    self.status.emit(
                        f"Retry {attempt + 2}/{MAX_RETRIES} in {delay}s..."
                    )
                    time.sleep(delay)

        if self._cancel.is_set():
            _cleanup_partial(self.target_dir)
            self.failed.emit("cancelled by user")
            return

        if last_error is not None or not final_path or not os.path.exists(final_path):
            _cleanup_partial(self.target_dir)
            exc_str = str(last_error) if last_error else "no output file produced"
            exc_str += hf_auth_error_hint(exc_str, token_configured)
            self.failed.emit(
                f"ONNX download failed after {MAX_RETRIES} attempts: {exc_str}"
            )
            return

        # Eliminar cachés internos de HF dentro de target_dir (no hacen falta).
        for cache_name in (".huggingface", ".cache"):
            cache_path = os.path.join(self.target_dir, cache_name)
            if os.path.isdir(cache_path):
                try:
                    shutil.rmtree(cache_path, ignore_errors=True)
                except Exception:
                    pass

        self.progress.emit(100)
        self.status.emit("download finished")
        self.finished_ok.emit(os.path.abspath(final_path))
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
        """Return the convertible weight file that BEST matches engine_id.

        El directorio de un engine puede contener MUCHOS pesos (el repo de
        Real-ESRGAN trae 9 .pth: 6 RRDBNet + 3 SRVGG). Coger el primero
        del walk era un azar: el conversor podía convertir un RRDBNet de
        67 MB en vez del SRVGG de 2,4 MB del AnimeVideo v3, o fallar en
        un peso envuelto con formato antiguo. Se reutiliza el mismo
        scoring que ``get_model_path(..., 'torch')``.
        """
        try:
            from src.ai.inference import pick_weight_file
            picked = pick_weight_file(self.engine_dir, self.engine_id)
            if picked and os.path.isfile(picked):
                return picked
        except Exception:
            pass
        # Fallback: primer peso convertible del walk (comportamiento previo).
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

        self.status.emit("Converting to ONNX in the AI host (torch.onnx.export)...")
        self.progress.emit(15)

        try:
            # La conversión la hace el HOST de IA: importar torch en este
            # proceso (la GUI) lo mata con 0xc0000005 en python313.dll.
            from src.ai.weight_tools import convert_weight_to_onnx
        except ImportError as exc:
            self.failed.emit(
                f"Could not import the ONNX converter: {exc}\n\n"
                "The AI host ships with the application; reinstall Draggy Encoder."
            )
            return

        if self._cancel.is_set():
            self.failed.emit("cancelled by user")
            return

        def _on_progress(pct, msg=""):
            try:
                self.progress.emit(int(pct))
                if msg:
                    self.status.emit(msg)
            except Exception:
                pass

        onnx_path, error_msg = convert_weight_to_onnx(weight_path, progress_callback=_on_progress)

        # Cancelación pedida DURANTE la exportación: torch.onnx.export no se
        # puede abortar a mitad (haría falta matar el thread), así que la
        # conversión termina y se descarta como cancelada. El .onnx generado
        # queda en disco (es válido) pero no se marca como convertido.
        if self._cancel.is_set():
            self.failed.emit("cancelled by user")
            return

        if not onnx_path or not os.path.exists(onnx_path):
            self.failed.emit(
                "ONNX conversion failed.\n\n"
                f"{error_msg or 'Unknown error (check the application log).'}\n\n"
                f"Weight file: {weight_path}"
            )
            return

        self.progress.emit(100)
        self.status.emit("export complete")
        self.finished_ok.emit(onnx_path)


if __name__ == "__main__":
    # Quick capability probe, useful from a debug REPL.
    print("huggingface_hub:      ", is_hf_hub_available())
    print("optimum-cli available:", is_optimum_cli_available())
