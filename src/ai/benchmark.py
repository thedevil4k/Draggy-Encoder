"""Benchmark de motores de inferencia para el estimador de tiempo (ETA).

``benchmark_engine`` carga el motor con la misma estrategia que el render
(ONNX si hay .onnx disponible, PyTorch como vía principal para ids de
catálogo, FSR algorítmico sin pesos) y mide ms/frame con un frame sintético
a resolución canónica. El resultado se cachea en ``engines.json`` (clave
``benchmarks``, ver ``engines_state.set_benchmark``) para que el ETA del
render lo lea sin re-medir cada vez.

Medición conservadora:
  - warmup de 1 frame (JIT / lazy allocations de CUDA fuera de la medida).
  - ``torch.cuda.synchronize()`` tras cada frame medido si CUDA está activo
    (sin él, la ejecución async de CUDA inflaría el throughput).
  - mediana de las muestras para ignorar outliers del primer paso.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from src.ai.engines_state import get_benchmark, set_benchmark  # noqa: E402

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


def _engine_device(engine) -> str:
    device = getattr(engine, "_device", None)
    if device is not None:
        return str(device)
    providers = getattr(engine, "_providers", None) or []
    if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
        return "cuda"
    if "CPUExecutionProvider" in providers:
        return "cpu"
    # Sin _device ni _providers: engines algorítmicos (FSR 1.0) o fallback
    # puro CPU/numpy.
    return "cpu"


def benchmark_engine(
    engine_id: str,
    target_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    frames: int = DEFAULT_FRAMES,
    warmup: int = DEFAULT_WARMUP,
    use_cache: bool = True,
) -> Optional[dict]:
    """Mide ms/frame del motor y cachea el resultado en engines.json.

    Args:
        engine_id: id de catálogo o ``user-*`` (o ``torch``/``onnxruntime``).
        target_dir: directorio de instalación del modelo (se resuelven los
            pesos ahí dentro si ``model_path`` no se pasa).
        model_path: ruta directa a los pesos (prioridad sobre target_dir).
        resolution: (w, h) del frame sintético de prueba.
        frames: muestras medidas (mediana).
        warmup: frames de calentamiento fuera de la medida.
        use_cache: si True y ya hay benchmark cacheado, lo devuelve sin
            volver a medir.

    Returns:
        dict con ms_per_frame/fps/device/... o None si no se pudo medir.
        El resultado se persiste con ``engines_state.set_benchmark``.
    """
    if use_cache:
        cached = get_benchmark(engine_id)
        if cached:
            return cached

    if model_path is None and target_dir:
        model_path = find_weight_file(target_dir)

    from src.ai.inference import create_inference_engine

    try:
        engine = create_inference_engine(engine_id, model_path or "")
    except Exception as exc:
        logger.warning("benchmark: cannot create engine %s: %s", engine_id, exc)
        return None
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
    result = {
        "engine_id": engine_id,
        "ms_per_frame": round(ms, 3),
        "fps": round(1000.0 / ms, 2),
        "resolution": [w, h],
        "device": _engine_device(engine),
        "measured_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    set_benchmark(engine_id, result)
    logger.info("benchmark %s: %.2f ms/frame (%.1f fps) on %s",
                engine_id, ms, result["fps"], result["device"])
    return result


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
