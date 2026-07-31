"""Gestión de runtimes de IA descargables.

Los runtimes (PyTorch, ONNX Runtime y futuros motores) se instalan en
``bin/runtimes/ai_modules`` usando ``pip install --target``. Esto mantiene
la aplicación autocontenida y evita tocar el entorno Python del usuario.

La prioridad de importación es:
  1. bin/runtimes/ai_modules (runtime local descargado por la app)
  2. Entorno Python actual (sistema, venv, etc.)

Para conseguirlo, ``inject_runtime_paths()`` debe llamarse lo antes posible
en el arranque de la aplicación (main.py) y también antes de importar
``torch`` u ``onnxruntime`` desde src.ai.inference.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

from src.config import ensure_runtime_dirs, get_bin_dir


RUNTIME_MODULES_DIR = os.path.join(get_bin_dir(), "runtimes", "ai_modules")


# ---------------------------------------------------------------------------
# Path injection
# ---------------------------------------------------------------------------


def get_runtime_modules_dir() -> str:
    return RUNTIME_MODULES_DIR


def inject_runtime_paths() -> None:
    """Inserta el directorio de módulos locales al principio de sys.path."""
    try:
        os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    except Exception:
        pass
    if os.path.isdir(RUNTIME_MODULES_DIR) and RUNTIME_MODULES_DIR not in sys.path:
        sys.path.insert(0, RUNTIME_MODULES_DIR)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _module_is_local(module_name: str) -> bool:
    """True si el módulo se puede importar y su __file__ está dentro de
    ``bin/runtimes/ai_modules``.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return False
        return spec.origin.startswith(os.path.abspath(RUNTIME_MODULES_DIR))
    except Exception:
        return False


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _has_nvidia_gpu() -> bool:
    """Heurística rápida: ¿existe nvidia-smi en PATH?"""
    return shutil.which("nvidia-smi") is not None


def _python_short() -> str:
    """Versión corta de Python, e.g. '3.11'."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _is_python_version_supported() -> bool:
    """True si es una versión de Python con wheels oficiales de PyTorch/ONNX.

    PyTorch y ONNX Runtime suelen tardar en publicar wheels para las
    versiones más nuevas de Python. A día de hoy (2026) el soporte
    estable es 3.9–3.14.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    if major != 3:
        return False
    return 9 <= minor <= 14


def get_python_support_info() -> dict:
    """Devuelve información de compatibilidad para mostrar en la UI."""
    supported = _is_python_version_supported()
    short = _python_short()
    if supported:
        return {
            "supported": True,
            "short": short,
            "message": f"Python {short} is supported. Click Install to download AI runtimes.",
        }
    return {
        "supported": False,
        "short": short,
        "message": (
            f"Python {short} is NOT supported by official PyTorch/ONNX wheels yet. "
            "Please use Python 3.11–3.14, or build the PyInstaller bundle with one of those versions. "
            "AI upscaling/interpolation/colorize will be disabled."
        ),
    }


# ---------------------------------------------------------------------------
# Package specs
# ---------------------------------------------------------------------------


def _default_gpu_variant() -> str:
    """Variant CUDA por defecto según la versión de Python en uso.

    PyTorch publica wheels CUDA por versión de Python: en 3.14 los wheels
    oficiales son cu126/cu130; en 3.9–3.13 seguimos en cu121 (comportamiento
    histórico, no romper a usuarios existentes).
    """
    minor = sys.version_info.minor
    if minor >= 14:
        return "cuda130"
    return "cuda121"


def _resolve_variant(variant: str) -> str:
    """Resuelve 'auto' a cpu/cuda121/cuda118/cuda126/cuda130."""
    if variant == "auto":
        if _has_nvidia_gpu():
            return _default_gpu_variant()
        return "cpu"
    if variant in ("cpu", "cuda118", "cuda11.8", "cuda121", "cuda126", "cuda130"):
        if variant == "cuda11.8":
            return "cuda118"
        return variant
    return "cpu"


def _build_pytorch_command(variant: str) -> list[str]:
    """Construye el comando pip para instalar torch/torchvision/torchaudio."""
    variant = _resolve_variant(variant)
    if variant == "cpu":
        index_url = "https://download.pytorch.org/whl/cpu"
    elif variant == "cuda118":
        index_url = "https://download.pytorch.org/whl/cu118"
    elif variant == "cuda126":
        index_url = "https://download.pytorch.org/whl/cu126"
    elif variant == "cuda130":
        index_url = "https://download.pytorch.org/whl/cu130"
    else:  # cuda121
        index_url = "https://download.pytorch.org/whl/cu121"

    return [
        sys.executable, "-m", "pip", "install",
        "--target", RUNTIME_MODULES_DIR,
        "--no-warn-script-location",
        "--upgrade",
        "--index-url", index_url,
        "torch", "torchvision", "torchaudio",
    ]


def _build_onnx_command(variant: str) -> list[str]:
    """Construye el comando pip para instalar onnxruntime."""
    variant = _resolve_variant(variant)
    pkg = "onnxruntime-gpu" if variant.startswith("cuda") else "onnxruntime"
    return [
        sys.executable, "-m", "pip", "install",
        "--target", RUNTIME_MODULES_DIR,
        "--no-warn-script-location",
        "--upgrade",
        pkg,
    ]


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def _run_pip(cmd: list[str], progress_callback, start_pct: int, end_pct: int,
             message: str):
    """Ejecuta un comando pip y emite progreso interpolado."""
    if progress_callback:
        progress_callback(start_pct, f"{message}...")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    # Verificar que pip esté disponible
    try:
        test_proc = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, env=env,
        )
        if test_proc.returncode != 0:
            raise RuntimeError("pip is not available or not working")
    except Exception as exc:
        raise RuntimeError(
            "pip no se encuentra disponible. "
            "Asegúrate de que Python se instaló con pip."
        ) from exc

    last_progress = start_pct
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        ) as proc:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                # Avanzar un poco la barra para que no parezca congelada
                if progress_callback and last_progress < end_pct:
                    last_progress += 1
                    progress_callback(last_progress, f"[{message}] {line[:120]}")
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"pip exited with code {ret}")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se pudo ejecutar pip. Verifica que Python esté en el PATH."
        ) from exc

    if progress_callback:
        progress_callback(end_pct, f"{message} done")


def _raise_unsupported_python() -> None:
    short = _python_short()
    raise RuntimeError(
        f"Python {short} no tiene wheels oficiales de PyTorch/ONNX todavía. "
        "Por favor, usa Python 3.9–3.14 para que la descarga funcione, "
        "o instala torch/onnxruntime manualmente en este entorno."
    )


def install_runtimes(
    variant: str = "auto",
    progress_callback=None,
    parent=None,
) -> str:
    """Instala PyTorch y ONNX Runtime en ``bin/runtimes/ai_modules``.

    Args:
        variant: "cpu", "cuda118", "cuda121", "cuda126", "cuda130" o "auto".
        progress_callback: callable(percent, message) -> None.
        parent: QWidget padre (no usado actualmente, reservado).

    Returns:
        El variant real usado ("cpu", "cuda121", "cuda130", ...).
    """
    if not _is_python_version_supported():
        _raise_unsupported_python()

    os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)

    used_variant = _resolve_variant(variant)

    # 1. PyTorch (la parte más grande)
    _run_pip(
        _build_pytorch_command(used_variant),
        progress_callback,
        start_pct=5,
        end_pct=85,
        message="Installing PyTorch",
    )

    # 2. ONNX Runtime
    _run_pip(
        _build_onnx_command(used_variant),
        progress_callback,
        start_pct=86,
        end_pct=99,
        message="Installing ONNX Runtime",
    )

    if progress_callback:
        progress_callback(100, "AI runtimes installed")

    # Después de instalar, reinjectar por si acaso
    inject_runtime_paths()
    return used_variant


# ---------------------------------------------------------------------------
# Per-package status helpers
# ---------------------------------------------------------------------------


def _version_of(module_name: str) -> str:
    try:
        import importlib.metadata as _meta
        return _meta.version(module_name) or "—"
    except Exception:
        return "—"


def torch_status() -> dict:
    installed = _can_import("torch")
    local = _module_is_local("torch") if installed else False
    gpu = False
    if installed:
        try:
            import torch as _torch
            gpu = _torch.cuda.is_available()
        except Exception:
            pass
    return {
        "id": "torch",
        "display_name": "PyTorch runtime",
        "installed": installed,
        "local": local,
        "version": _version_of("torch"),
        "gpu": gpu,
        "install_dir": RUNTIME_MODULES_DIR,
    }


def onnx_status() -> dict:
    installed = _can_import("onnxruntime")
    local = _module_is_local("onnxruntime") if installed else False
    gpu = False
    if installed:
        try:
            import onnxruntime as _ort
            gpu = "CUDAExecutionProvider" in _ort.get_available_providers()
        except Exception:
            pass
    return {
        "id": "onnxruntime",
        "display_name": "ONNX Runtime (CPU/GPU)",
        "installed": installed,
        "local": local,
        "version": _version_of("onnxruntime"),
        "gpu": gpu,
        "install_dir": RUNTIME_MODULES_DIR,
    }


# ---------------------------------------------------------------------------
# Entry points from dependencies.py
# ---------------------------------------------------------------------------


def install_all(progress_callback=None, parent=None, variant: str = "auto") -> None:
    """Instala PyTorch + ONNX en un solo comando."""
    install_runtimes(variant=variant, progress_callback=progress_callback, parent=parent)


def uninstall_local_runtimes(parent=None) -> None:
    """Borra el store local de runtimes (``bin/runtimes/ai_modules``) completo.

    PyTorch y ONNX Runtime se instalan juntos ahí, así que el borrado es
    conjunto. En Windows, si algún DLL está cargado (runtime ya importado),
    ``dependencies._rmtree_force`` lo programa para el próximo arranque.
    """
    try:
        from src.dependencies import _rmtree_force
    except Exception:
        return
    if os.path.isdir(RUNTIME_MODULES_DIR):
        _rmtree_force(RUNTIME_MODULES_DIR)


def get_status() -> dict:
    """Estado consolidado de los runtimes de IA para dependencies.py."""
    torch = torch_status()
    onnx = onnx_status()

    if torch["installed"] and onnx["installed"]:
        status = "installed"
        detail = (
            f"PyTorch {torch['version']} ({'CUDA' if torch['gpu'] else 'CPU'}), "
            f"ONNX Runtime {onnx['version']} ({'CUDA' if onnx['gpu'] else 'CPU'})"
        )
    else:
        status = "missing"
        missing = []
        if not torch["installed"]:
            missing.append("PyTorch")
        if not onnx["installed"]:
            missing.append("ONNX Runtime")
        detail = f"Missing: {', '.join(missing)}. Click Install to download."

    return {
        "id": "ai_runtimes",
        "display_name": "AI Runtimes (PyTorch + ONNX)",
        "kind": "runtime",
        "status": status,
        "version": torch.get("version") or "—",
        "install_dir": RUNTIME_MODULES_DIR,
        "source_url": "https://pytorch.org/get-started/locally/",
        "is_updatable": True,
        "size_bytes": _safe_dir_size(RUNTIME_MODULES_DIR),
        "detail": detail,
        "torch": torch,
        "onnx": onnx,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
    except Exception:
        pass
    return total


# ---------------------------------------------------------------------------
# Windows DLL path helper
# ---------------------------------------------------------------------------


def ensure_runtime_dlls_on_path() -> None:
    """Añade bin/runtimes/ai_modules y sus subcarpetas nvidia/*/bin a PATH.

    PyTorch/ONNX GPU instalan DLLs de NVIDIA bajo ai_modules/nvidia/. En
    Windows es necesario que estén en PATH para que torch pueda cargarlos.
    """
    if sys.platform != "win32":
        return
    try:
        os.makedirs(RUNTIME_MODULES_DIR, exist_ok=True)
    except Exception:
        pass

    extra = [RUNTIME_MODULES_DIR]
    nvidia_dir = os.path.join(RUNTIME_MODULES_DIR, "nvidia")
    if os.path.isdir(nvidia_dir):
        for sub in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, sub, "bin")
            if os.path.isdir(bin_dir):
                extra.append(bin_dir)
                for entry in os.listdir(bin_dir):
                    d = os.path.join(bin_dir, entry)
                    if os.path.isdir(d):
                        extra.append(d)

    current_path = os.environ.get("PATH", "")
    new_parts = [p for p in extra if p not in current_path]
    if new_parts:
        os.environ["PATH"] = ";".join(new_parts) + ";" + current_path
