"""Mapa de volumen de una pista de audio (para el editor visual de efectos).

Produce, para un archivo, una serie temporal de nivel (LUFS momentáneo y
corto) más el pico, que es lo que necesita una interfaz tipo vúmetro/waveform
para dibujar el volumen a lo largo del tiempo y dejar que el usuario elija un
tramo.

Se usa ``ebur128`` con metadatos a texto, que es la vía barata: medido sobre
contenido real rinde ~65x tiempo real (una película de 92 min en ~85 s) y
saca un punto cada 100 ms. Como es análisis puro, se puede lanzar en segundo
plano y cachear en disco; el mismo archivo no se analiza dos veces.

Formato de la salida de FFmpeg que se parsea (stdout)::

    frame:12   pts:57600   pts_time:1.2
    lavfi.r128.M=-23.691
    lavfi.r128.S=-24.102
    lavfi.r128.true_peak=-3.100
    ...

OJO: la metadata va a **stdout** (``file=-``), no a stderr, y los tiempos son
relativos al punto de inicio. Por eso el recorte se hace con ``-ss`` DESPUÉS
de ``-i`` (seek de salida): con el seek de entrada los ``pts_time`` se
reinician y se pierde la posición absoluta en la pista.
PENDIENTE CONOCIDO (no verificado): con recorte (``-ss``/``-to``) los
``pts_time`` que devuelve la cadena de metadatos no coinciden con la posición
real en la pista: midiendo 120 s desde el segundo 2700 de un MKV 5.1, el
primer punto sale en t=2700.4 (correcto) pero el último en t=5520 en lugar de
~2820. La primera muestra es fiable, la deriva no. Antes de construir la
interfaz sobre esto hay que determinar si la causa es el seek de salida, el
resampleo de 5.1 a estéreo del visualizador, o un PTS con base distinta, y
anclar la serie a ``duration``/número de muestras en vez de a ``pts_time``.
Sin recorte (archivo completo) la serie avanza 100 ms por punto, que es el
comportamiento esperado.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time

from src.config import get_bin_dir
from src.ffmpeg.process_supervisor import (
    ProcessSupervisor,
    process_group_popen_kwargs,
)
from src.ffmpeg.runner import (
    KEEP_ALL_AUDIO,
    _run_kwargs,
    parse_audio_stream_index,
)

logger = logging.getLogger(__name__)


# Silencio digital: ebur128 reporta -120.7 LUFS cuando no hay señal. Se acota
# para que el gráfico no tenga un salto absurdo de escala.
SILENCE_LUFS = -70.0
DEFAULT_MAX_POINTS = 4000

_CACHE_VERSION = 1


def _cache_path(cache_dir: str | None = None) -> str:
    base = cache_dir or os.path.join(get_bin_dir(), ".session")
    return os.path.join(base, "audio_map.json")


def _file_signature(path: str, stream_index: int) -> dict:
    """Identidad del análisis: si cambia el archivo, la caché se invalida."""
    try:
        stat = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "stream": int(stream_index),
        }
    except OSError:
        return {"path": os.path.abspath(path), "size": 0, "mtime_ns": 0,
                "stream": int(stream_index)}


def load_cached_map(path: str, stream_index: int = 0,
                    cache_dir: str | None = None) -> dict | None:
    """Devuelve el mapa cacheado para ese archivo, o None si no sirve."""
    try:
        with open(_cache_path(cache_dir), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
        return None
    signature = _file_signature(path, stream_index)
    for entry in payload.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("signature") == signature and isinstance(entry.get("map"), dict):
            return entry["map"]
    return None


def save_cached_map(path: str, stream_index: int, volume_map: dict,
                    cache_dir: str | None = None, keep: int = 12) -> None:
    """Guarda el mapa (best effort: la caché nunca debe romper nada)."""
    cache_file = _cache_path(cache_dir)
    try:
        payload = {"version": _CACHE_VERSION, "entries": []}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if isinstance(existing, dict) and existing.get("version") == _CACHE_VERSION:
                    payload["entries"] = [
                        e for e in (existing.get("entries") or [])
                        if isinstance(e, dict)
                    ]
            except (OSError, ValueError):
                pass
        signature = _file_signature(path, stream_index)
        payload["entries"] = [
            e for e in payload["entries"] if e.get("signature") != signature
        ]
        payload["entries"].insert(0, {"signature": signature, "map": volume_map})
        payload["entries"] = payload["entries"][:keep]
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as exc:  # pragma: no cover - E/S del usuario
        logger.info("no se pudo cachear el mapa de volumen: %s", exc)


def _parse_metadata_stream(stdout: str, offset: float = 0.0) -> list[dict]:
    """Convierte la salida de ``ametadata=mode=print`` en muestras.

    Cada muestra es ``{"t": segundos absolutos, "m": LUFS momentáneo,
    "s": LUFS corto, "tp": pico}``.
    """
    samples: list[dict] = []
    current: dict | None = None
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # OJO: re.search, no re.match: la línea del tiempo empieza por
        # "frame:N", así que un match anclado al inicio nunca encuentra
        # "pts_time" y el mapa sale vacío.
        time_match = re.search(r"pts_time:([\d.]+)", line)
        if time_match:
            current = {"t": offset + float(time_match.group(1)), "m": None,
                       "s": None, "tp": None}
            samples.append(current)
            continue
        if current is None:
            continue
        value_match = re.search(r"lavfi\.r128\.(M|S|true_peak)=(-?[\d.]+)", line)
        if value_match:
            key = {"M": "m", "S": "s", "true_peak": "tp"}[value_match.group(1)]
            current[key] = float(value_match.group(2))
    return [s for s in samples if s.get("m") is not None]


def _clamp_silence(value: float | None) -> float | None:
    if value is None:
        return None
    # -120.7 LUFS es el "cero digital" que reporta ebur128 para silencio.
    if value <= -119.0:
        return SILENCE_LUFS
    return max(SILENCE_LUFS, value)


def _downsample(samples: list[dict], max_points: int) -> list[list[float]]:
    """Reduce la serie a ``[t, m, s, tp]`` con como mucho ``max_points``.

    Para tramos largos se usa el MÁXIMO de cada bloque (no la media): en una
    pista interesa ver los picos, y promediar los escondería.
    """
    if not samples:
        return []
    step = max(1, len(samples) // max(1, max_points))
    series: list[list[float]] = []
    for start in range(0, len(samples), step):
        chunk = samples[start:start + step]
        best = max(chunk, key=lambda s: s.get("m") if s.get("m") is not None else -999)
        series.append([
            round(float(best["t"]), 3),
            round(float(_clamp_silence(best.get("m"))), 2),
            round(float(_clamp_silence(best.get("s"))), 2) if best.get("s") is not None else None,
            round(float(best.get("tp")), 2) if best.get("tp") is not None else None,
        ])
    return series


def analyze_volume_map(ffmpeg_path, input_path, *, start_time=None, end_time=None,
                       audio_stream_index=0, max_points: int = DEFAULT_MAX_POINTS,
                       abort_event=None, supervisor: ProcessSupervisor | None = None,
                       timeout_seconds: float = 3600.0,
                       progress_callback=None) -> dict | None:
    """Analiza el nivel del audio y devuelve el mapa de volumen.

    Devuelve ``{"series": [[t, m, s, tp], ...], "duration", "samples",
    "engine": "ebur128", "stream"}`` o None si no se pudo analizar.

    ``progress_callback(percent)`` recibe el avance cuando FFmpeg reporta
    tiempo procesado (útil para una barra en el diálogo de efectos).
    """
    if not ffmpeg_path or not input_path or not os.path.exists(input_path):
        return None

    index = parse_audio_stream_index(audio_stream_index)
    map_index = 0 if index == KEEP_ALL_AUDIO else index

    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostats"]
    cmd += ["-i", input_path]
    # Seek de SALIDA (después de -i) para conservar la posición absoluta.
    if start_time:
        cmd += ["-ss", str(start_time)]
    if end_time:
        cmd += ["-to", str(end_time)]
    cmd += ["-map", f"0:a:{map_index}?", "-vn"]
    cmd += ["-af", "ebur128=metadata=1:peak=true,ametadata=mode=print:file=-"]
    cmd += ["-f", "null", "-"]

    kwargs = _run_kwargs()
    kwargs.pop("capture_output", None)
    kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL})
    kwargs.update(process_group_popen_kwargs())

    proc = None
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        if supervisor is not None:
            supervisor.register(proc, "audio-map")
        chunks: list[str] = []

        def _drain():
            try:
                for line in proc.stdout:
                    chunks.append(line)
            except (ValueError, OSError):
                pass

        reader = threading.Thread(target=_drain, daemon=True, name="audio-map-reader")
        reader.start()
        deadline = time.monotonic() + timeout_seconds
        while proc.poll() is None:
            if abort_event is not None and abort_event.is_set():
                _stop(proc, supervisor)
                reader.join(timeout=2)
                return None
            if time.monotonic() >= deadline:
                _stop(proc, supervisor)
                reader.join(timeout=2)
                logger.warning("análisis del mapa de volumen agotó el tiempo")
                return None
            if progress_callback is not None and chunks:
                try:
                    progress_callback(len(chunks))
                except Exception:
                    pass
            time.sleep(0.2)
        reader.join(timeout=3)
        returncode = proc.returncode
        stdout = "".join(chunks)
    except Exception as exc:  # pragma: no cover - fallo de sistema
        logger.warning("mapa de volumen: %s", exc)
        if proc is not None:
            _stop(proc, supervisor)
        return None

    if returncode != 0:
        logger.info("mapa de volumen: ffmpeg terminó con rc=%s", returncode)
        return None

    samples = _parse_metadata_stream(stdout, offset=float(start_time or 0.0))
    if not samples:
        logger.info("mapa de volumen: sin muestras (¿pista sin audio?)")
        return None

    # La duración es el INTERVALO cubierto, no la última marca de tiempo: los
    # ``pts_time`` son absolutos respecto al archivo (se hizo seek de salida),
    # así que restar el inicio daba una duración absurda (2820 s para un tramo
    # de 120 s).
    try:
        duration = float(samples[-1]["t"]) - float(samples[0]["t"])
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "series": _downsample(samples, max_points),
        "duration": round(duration, 3),
        "samples": len(samples),
        "engine": "ebur128",
        "stream": map_index,
    }


def _stop(proc, supervisor) -> None:
    """Detiene el proceso de análisis con gracia."""
    if supervisor is not None:
        try:
            supervisor.stop_all(grace_seconds=1, kill_seconds=1)
            return
        except Exception:
            pass
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def get_volume_map(ffmpeg_path, input_path, *, audio_stream_index=0,
                   use_cache: bool = True, cache_dir: str | None = None,
                   **kwargs) -> dict | None:
    """``analyze_volume_map`` con caché en disco."""
    if use_cache:
        cached = load_cached_map(input_path, audio_stream_index, cache_dir)
        if cached is not None:
            return cached
    result = analyze_volume_map(
        ffmpeg_path, input_path, audio_stream_index=audio_stream_index, **kwargs
    )
    if result is not None and use_cache:
        save_cached_map(input_path, audio_stream_index, result, cache_dir)
    return result
