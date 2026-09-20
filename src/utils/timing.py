"""Acumulador de milisegundos por etapa del render (telemetría P0).

Nace de una medición real: un render de upscale a 5 fps gastaba 71 ms en la
inferencia, 68 ms limpiando el tensor en CPU y 34 ms encogiendo la imagen con
PIL, sin que el log dijera nada de eso. Sin este desglose cualquier
optimización es a ciegas y las ETAs mienten.

Diseño:
  - Coste despreciable: un ``add()`` son dos operaciones sobre floats.
  - Memoria acotada: si una etapa supera ``cap`` muestras se reemplazan en
    round-robin (la mediana de una muestra uniforme es igual de útil y un
    render de 134 000 frames no puede guardar 8 listas de 134 000 floats).
  - Nunca lanza: degradarse a "sin telemetría" es siempre mejor que romper
    un render.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

DEFAULT_CAP = 20000


class StageTimer:
    """Suma y muestrea milisegundos por etapa y resume en mediana/media/p95."""

    def __init__(self, cap: int = DEFAULT_CAP):
        self._cap = max(1, int(cap))
        self._samples: dict[str, list[float]] = {}
        self._seen: dict[str, int] = {}
        self._frames = 0

    # ------------------------------------------------------------------ API
    def add(self, stage: str, ms: float) -> None:
        """Registra una muestra de ``stage`` (milisegundos). Nunca lanza."""
        try:
            value = float(ms)
        except (TypeError, ValueError):
            return
        if value < 0 or value != value:  # negativo o NaN
            return
        try:
            bucket = self._samples.get(stage)
            if bucket is None:
                bucket = self._samples[stage] = []
                self._seen[stage] = 0
            seen = self._seen[stage]
            if len(bucket) < self._cap:
                bucket.append(value)
            else:
                bucket[seen % self._cap] = value
            self._seen[stage] = seen + 1
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str):
        """``with timer.stage("decode"): ...`` — mide aunque el bloque falle."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1000.0)

    def mark_frame(self) -> None:
        """Cuenta un frame procesado (para fps medios)."""
        self._frames += 1

    @property
    def frames(self) -> int:
        return self._frames

    def samples(self, stage: str) -> list[float]:
        return list(self._samples.get(stage, ()))

    # -------------------------------------------------------------- resumen
    def summary(self) -> dict:
        """Resumen listo para el JSON del render.

        ``ms_per_frame`` es la media por frame de CADA etapa: sumando todas
        las etapas de un frame se obtiene el presupuesto completo del frame.
        """
        out: dict = {"frames": self._frames, "stages": {}}
        total_of_medians = 0.0
        for stage, values in self._samples.items():
            if not values:
                continue
            ordered = sorted(values)
            mean = sum(ordered) / len(ordered)
            median = _percentile(ordered, 50.0)
            out["stages"][stage] = {
                "n": self._seen.get(stage, len(ordered)),
                "mean_ms": round(mean, 3),
                "median_ms": round(median, 3),
                "p95_ms": round(_percentile(ordered, 95.0), 3),
                "max_ms": round(ordered[-1], 3),
            }
            total_of_medians += median
        out["sum_of_medians_ms"] = round(total_of_medians, 3)
        if self._frames and total_of_medians > 0:
            out["fps_implied_by_stages"] = round(1000.0 / total_of_medians, 2)
        return out


def _percentile(ordered: list[float], pct: float) -> float:
    """Percentil con interpolación lineal sobre una lista YA ordenada.

    Interpolar (en vez de tomar el rango más cercano) importa para la mediana:
    con 4 muestras ``[10, 20, 30, 40]`` el rango más cercano devolvería 30 en
    lugar de 25, y esa cifra es la que el usuario lee como "lo que tarda una
    etapa".
    """
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction
