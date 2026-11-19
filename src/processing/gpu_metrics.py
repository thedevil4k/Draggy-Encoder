"""Muestreo de utilización de GPU durante un render.

No promete nada: mide. Cada segundo ejecuta ``nvidia-smi`` y guarda qué
porcentaje de la GPU, del encoder y del decoder se está usando, más la VRAM.
El resumen (media, p95, máximo) se escribe en ``render_*.json`` para poder
afirmar —o desmentir— que el render aprovecha la GPU.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading


_QUERY = (
    "utilization.gpu,utilization.memory,utilization.encoder,"
    "utilization.decoder,memory.used"
)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


class GpuSampler(threading.Thread):
    """Hilo daemon que muestrea ``nvidia-smi`` hasta que se le pide parar."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = max(0.2, float(interval))
        self._stop_event = threading.Event()
        self.samples: list[dict] = []
        self.available = False

    def run(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self.available = True
        while not self._stop_event.is_set():
            sample = self._sample_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop_event.wait(self.interval)

    def _sample_once(self) -> dict | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            line = (result.stdout or "").strip().splitlines()
            if not line:
                return None
            parts = [p.strip() for p in line[0].split(",")]
            if len(parts) < 5:
                return None
            return {
                "gpu": float(parts[0]),
                "memory": float(parts[1]),
                "encoder": float(parts[2]),
                "decoder": float(parts[3]),
                "vram_mb": float(parts[4]),
            }
        except Exception:
            return None

    def stop(self) -> dict:
        self._stop_event.set()
        try:
            self.join(timeout=3.0)
        except Exception:
            pass
        return self.summary()

    def summary(self) -> dict:
        if not self.samples:
            return {
                "available": False,
                "samples": 0,
                "note": "nvidia-smi returned no samples",
            }

        def stats(key: str) -> dict:
            values = [float(s.get(key) or 0.0) for s in self.samples]
            return {
                "avg": round(statistics.fmean(values), 1),
                "p95": round(_p95(values), 1),
                "max": round(max(values), 1),
            }

        return {
            "available": True,
            "samples": len(self.samples),
            "gpu_util": stats("gpu"),
            "encoder_util": stats("encoder"),
            "decoder_util": stats("decoder"),
            "vram_used_mb": stats("vram_mb"),
        }
