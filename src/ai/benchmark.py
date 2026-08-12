"""Benchmark de motores de inferencia para el selector de backend y el ETA.

``benchmark_backend`` mide ms/frame de UN backend concreto ("onnx" o
"torch") construyendo la engine directamente (PyTorch NO auto-convierte a
ONNX: la medida refleja el backend real) con un frame sintético a
resolución canónica. Cada backend se cachea por separado en ``engines.json``
(``benchmarks[engine_id][backend]``, ver ``engines_state``) para que el
selector de backend de la UI compare ONNX vs PyTorch y el ETA del render
lea la medición del backend activo sin re-medir.

``benchmark_engine`` se conserva como envoltorio legacy (mide el backend
"natural" del directorio) para no romper llamadores externos.

Medición conservadora:
  - warmup de 1 frame (JIT / lazy allocations de CUDA fuera de la medida).
  - ``torch.cuda.synchronize()`` tras cada frame medido si CUDA está activo
    (sin él, la ejecución async de CUDA inflaría el throughput).
  - mediana de las muestras para ignorar outliers del primer paso.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from src.ai.engines_state import (
    get_backend_benchmarks,
    get_benchmark,
    set_backend_benchmark,
)  # noqa: E402

DEFAULT_RESOLUTION = (512, 288)  # (w, h) — canonical, parecido a 720p recortado
DEFAULT_FRAMES = 5
DEFAULT_WARMUP = 1


# ---------------------------------------------------------------------------
# Resolución de pesos
# ---------------------------------------------------------------------------

def find_weight_file(target_dir: str) -> Optional[str]:
    """Primer archivo de pesos (prefiere .onnx, luego torch) en ``target_dir``."""
    if not target_dir or not os.path.isdir(target_dir):
        return None
    found = {"onnx": None, "torch": None}
    for root, _, files in os.walk(target_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith(".onnx") and found["onnx"] is None:
                found["onnx"] = os.path.join(root, fn)
            elif low.endswith((".safetensors", ".pth", ".pt")) and found["torch"] is None:
                found["torch"] = os.path.join(root, fn)
            if found["onnx"] and found["torch"]:
                break
    return found["onnx"] or found["torch"]


def _find_onnx_file(target_dir: str) -> Optional[str]:
    try:
        from src.ai.inference import find_onnx_model
        return find_onnx_model(target_dir)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------

def _sync_cuda() -> None:
    """Bloquea hasta que la GPU termine todo lo pendiente (si CUDA activo)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def current_environment_signature(backend: str) -> dict:
    """Describe the runtime/GPU used by a backend benchmark.

    The signature is deliberately JSON-friendly and excludes volatile values
    such as process ids. A changed Python/runtime/provider/GPU invalidates
    old timings instead of silently steering Auto with stale data.
    """
    backend = str(backend or "").strip().lower()
    signature = {
        "backend": backend,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        if backend == "torch":
            import torch
            signature.update({
                "runtime": getattr(torch, "__version__", ""),
                "cuda": bool(torch.cuda.is_available()),
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None
                ),
            })
        elif backend == "onnx":
            import onnxruntime as ort
            signature.update({
                "runtime": getattr(ort, "__version__", ""),
                "providers": sorted(ort.get_available_providers()),
            })
    except Exception as exc:
        signature["runtime_error"] = type(exc).__name__
    return signature


def model_fingerprint(model_path: Optional[str]) -> Optional[dict]:
    """Cheap identity for a local weight file used by a benchmark."""
    if not model_path:
        return None
    try:
        st = os.stat(model_path)
        return {"path": os.path.abspath(model_path), "size": st.st_size,
                "mtime_ns": st.st_mtime_ns}
    except OSError:
        return None


def _engine_device(engine) -> str:
    device = getattr(engine, "_device", None)
    if device is not None:
        return str(device)
    providers = getattr(engine, "_providers", None) or []
    if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
        return "cuda"
    if "CPUExecutionProvider" in providers:
        return "cpu"
    # Sin _device ni _providers: engines algorítmicos (FSR 3.1) o fallback
    # puro CPU/numpy.
    return "cpu"


def _measure_engine(
    engine,
    engine_id: str,
    resolution: tuple[int, int],
    frames: int,
    warmup: int,
) -> Optional[dict]:
    """Mide ms/frame de una engine YA cargada (sin persistir)."""
    if not getattr(engine, "is_loaded", False):
        logger.warning("benchmark: engine %s did not load", engine_id)
        return None

    w, h = resolution
    try:
        import numpy as np
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
    except Exception:
        dummy = None

    process = getattr(engine, "process_frame", None)
    if not callable(process):
        logger.warning("benchmark: engine %s has no process_frame", engine_id)
        return None

    # warmup (fuera de la medida)
    try:
        for _ in range(max(1, warmup)):
            process(dummy)
    except Exception as exc:
        logger.warning("benchmark: warmup failed for %s: %s", engine_id, exc)
        return None
    _sync_cuda()

    samples: list[float] = []
    for _ in range(max(1, frames)):
        t0 = time.perf_counter()
        try:
            process(dummy)
        except Exception as exc:
            logger.warning("benchmark: frame failed for %s: %s", engine_id, exc)
            return None
        _sync_cuda()
        samples.append((time.perf_counter() - t0) * 1000.0)

    ms = _median(samples)
    if ms <= 0:
        return None
    return {
        "engine_id": engine_id,
        "ms_per_frame": round(ms, 3),
        "fps": round(1000.0 / ms, 2),
        "resolution": [w, h],
        "device": _engine_device(engine),
        "measured_at_iso": datetime.now(timezone.utc).isoformat(),
    }


def _build_backend_engine(engine_id: str, backend: str, model_path: str):
    """Construye la engine del backend pedido SIN auto-convertir.

    ``backend == "torch"`` -> PyTorchInferenceEngine directo (un .pth no
    se convierte a ONNX durante la medida, para que refleje el backend
    real). ``backend == "onnx"`` -> ONNXRUNTIMEInferenceEngine directo.
    """
    from src.ai.inference import ONNXRUNTIMEInferenceEngine, PyTorchInferenceEngine
    if backend == "torch":
        return PyTorchInferenceEngine(model_path, engine_id)
    return ONNXRUNTIMEInferenceEngine(model_path, engine_id)


def benchmark_backend(
    engine_id: str,
    backend: str,
    target_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    frames: int = DEFAULT_FRAMES,
    warmup: int = DEFAULT_WARMUP,
    use_cache: bool = True,
) -> Optional[dict]:
    """Mide ms/frame de UN backend ("onnx" o "torch") y lo cachea.

    Args:
        engine_id: id de catálogo o ``user-*``.
        backend: "onnx" u "torch".
        target_dir: directorio de instalación del modelo; se resuelven los
            pesos ahí dentro si ``model_path`` no se pasa.
        model_path: ruta directa a los pesos (prioridad sobre target_dir).
        resolution: (w, h) del frame sintético de prueba.
        frames: muestras medidas (mediana).
        warmup: frames de calentamiento fuera de la medida.
        use_cache: si True y ya hay benchmark de ese backend, lo devuelve
            sin volver a medir.

    Returns:
        dict con ms_per_frame/fps/device/... o None si no se pudo medir
        (backend sin pesos, libs sin cargar, ...). Se persiste en
        ``benchmarks[engine_id][backend]``.
    """
    backend = str(backend or "").strip().lower()
    if backend not in ("onnx", "torch"):
        return None

    if model_path is None and target_dir:
        if backend == "onnx":
            model_path = _find_onnx_file(target_dir)
        else:
            try:
                from src.ai.inference import pick_weight_file
                model_path = pick_weight_file(target_dir, engine_id)
            except Exception:
                model_path = None
    if not model_path or not os.path.isfile(model_path):
        logger.info("benchmark: no %s weights for %s", backend, engine_id)
        return None

    fingerprint = model_fingerprint(model_path)
    if use_cache:
        cached = get_backend_benchmarks(engine_id).get(backend)
        if cached and (
            not cached.get("model_fingerprint")
            or cached.get("model_fingerprint") == fingerprint
        ):
            return cached

    try:
        engine = _build_backend_engine(engine_id, backend, model_path)
        if not engine.load():
            return None
    except Exception as exc:
        logger.warning("benchmark: cannot load %s engine %s: %s",
                       backend, engine_id, exc)
        return None

    result = _measure_engine(engine, engine_id, resolution, frames, warmup)
    if result:
        result["environment_signature"] = current_environment_signature(backend)
        result["model_fingerprint"] = fingerprint
    try:
        engine.unload()
    except Exception:
        pass
    if not result:
        return None
    set_backend_benchmark(engine_id, backend, result)
    logger.info("benchmark %s/%s: %.2f ms/frame (%.1f fps) on %s",
                engine_id, backend, result["ms_per_frame"],
                result["fps"], result["device"])
    return result


def benchmark_engine(
    engine_id: str,
    target_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    frames: int = DEFAULT_FRAMES,
    warmup: int = DEFAULT_WARMUP,
    use_cache: bool = True,
) -> Optional[dict]:
    """[Legacy] Mide el backend "natural" del directorio y cachea el resultado.

    Compat con llamadores antiguos: resuelve onnx si hay un .onnx en
    ``target_dir`` (o si ``model_path`` acaba en .onnx), si no torch, y
    delega en :func:`benchmark_backend`.

    Returns:
        dict con ms_per_frame/fps/device/... o None si no se pudo medir.
    """
    if model_path:
        backend = "onnx" if model_path.lower().endswith(".onnx") else "torch"
    elif target_dir:
        backend = "onnx" if _find_onnx_file(target_dir) else "torch"
    else:
        backend = "torch"
    return benchmark_backend(
        engine_id, backend,
        target_dir=target_dir, model_path=model_path,
        resolution=resolution, frames=frames, warmup=warmup,
        use_cache=use_cache,
    )


def estimate_seconds(engine_id: str, frame_count: int) -> Optional[float]:
    """Segundos estimados de inferencia pura para ``frame_count`` frames,
    usando el benchmark cacheado (None si no hay benchmark)."""
    bench = get_benchmark(engine_id)
    if not bench or frame_count is None or frame_count <= 0:
        return None
    return round(bench.get("ms_per_frame", 0) * float(frame_count) / 1000.0, 2)


def estimate_status_line(engine_id: str, frame_count: int) -> Optional[str]:
    """Línea de status para el render: '~MM:SS (benchmark X ms/frame)' o None."""
    bench = get_benchmark(engine_id)
    if not bench:
        return None
    secs = estimate_seconds(engine_id, frame_count)
    if secs is None:
        return None
    mm, ss = divmod(int(secs), 60)
    hh, mm = divmod(mm, 60)
    if hh:
        eta = f"{hh}h {mm:02d}m {ss:02d}s"
    else:
        eta = f"{mm}:{ss:02d}"
    return (
        f"Estimated AI time: ~{eta} for {frame_count} frames "
        f"({bench.get('ms_per_frame', 0)} ms/frame, {bench.get('device', '?')})"
    )
