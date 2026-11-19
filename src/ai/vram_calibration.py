"""Mediciones reales de VRAM por motor, para calibrar la estimación.

La estimación de ``hardware_fit`` es una fórmula (pesos + activaciones) y no
puede saber cuánto reserva realmente CUDA/cuDNN para un modelo concreto. Este
módulo guarda el pico medido con ``nvidia-smi`` durante renders reales y lo
expone para que el catálogo use el dato medido cuando exista.

El fichero vive en la sesión de la app (``bin/.session/vram_calibration.json``)
y se escribe de forma atómica. La medida se guarda como **delta sobre la VRAM
usada antes de cargar el motor**: incluye pesos, contexto CUDA y buffers de
inferencia, que es lo que necesita el usuario para decidir si cabe.

No se inventan valores: si no hay medida para un motor, ``lookup`` devuelve
``None`` y ``hardware_fit`` sigue usando su estimación heurística.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from src.config import get_session_dir

_FILENAME = "vram_calibration.json"
_VERSION = 1
_LOCK = threading.Lock()


def _path() -> str:
    return os.path.join(get_session_dir(), _FILENAME)


def _empty() -> dict:
    return {"version": _VERSION, "measurements": {}}


def _load(path: Optional[str] = None) -> dict:
    path = path or _path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return _empty()
    except Exception:
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    data["version"] = _VERSION
    if not isinstance(data.get("measurements"), dict):
        data["measurements"] = {}
    return data


def _save(data: dict, path: Optional[str] = None) -> None:
    path = path or _path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def record_measurement(
    engine_id: str,
    peak_delta_mb: float,
    *,
    estimate_mb: Optional[float] = None,
    weights_mb: Optional[float] = None,
    resolution: str = "",
    backend: str = "",
    gpu_name: str = "",
    frames: int = 0,
    path: Optional[str] = None,
) -> Optional[dict]:
    """Guarda (o conserva el peor) pico medido para ``engine_id``.

    Se conserva el pico mayor: para decidir si un modelo cabe, subestimar es
    peor que sobreestimar. Devuelve la entrada guardada, o ``None`` si la
    medida no es válida.
    """
    try:
        delta = float(peak_delta_mb)
    except (TypeError, ValueError):
        return None
    if delta <= 0 or not engine_id:
        return None

    now = datetime.now(timezone.utc).isoformat()
    with _LOCK:
        data = _load(path)
        measurements = data.setdefault("measurements", {})
        previous = measurements.get(engine_id)
        previous_delta = 0.0
        if isinstance(previous, dict):
            try:
                previous_delta = float(previous.get("peak_delta_mb") or 0.0)
            except (TypeError, ValueError):
                previous_delta = 0.0
        if previous_delta > delta:
            # Mantener el peor caso, pero registrar cuándo se volvió a ver.
            previous["last_measured_at_iso"] = now
            _save(data, path)
            return previous
        entry = {
            "engine_id": engine_id,
            "peak_delta_mb": round(delta, 1),
            "estimate_mb": round(float(estimate_mb), 1) if estimate_mb is not None else None,
            "weights_mb": round(float(weights_mb), 2) if weights_mb is not None else None,
            "resolution": str(resolution or ""),
            "backend": str(backend or ""),
            "gpu_name": str(gpu_name or ""),
            "frames": int(frames or 0),
            "measured_at_iso": now,
        }
        measurements[engine_id] = entry
        _save(data, path)
    return entry


def lookup(engine_id: str, path: Optional[str] = None) -> Optional[dict]:
    """Devuelve la medición de un motor, o ``None`` si no hay."""
    if not engine_id:
        return None
    entry = _load(path).get("measurements", {}).get(engine_id)
    return dict(entry) if isinstance(entry, dict) else None


def all_measurements(path: Optional[str] = None) -> dict:
    """Copia de todas las mediciones (para diagnóstico/UI)."""
    return dict(_load(path).get("measurements", {}))
