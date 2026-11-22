"""Desfase entre dos doblajes del mismo episodio.

Comparar las formas de onda de dos idiomas no sirve: el diálogo es distinto.
Lo que sí coincide es la **envolvente de energía** (música, efectos, pausas,
cortes de escena), así que se correlaciona eso y el pico da el retardo.

La medición se hace a 8 kHz en mono: para una envolvente de 20 ms sobra, y así
comparar diez minutos de audio cuesta muy poco.

Este módulo no importa Qt ni toca el pipeline: recibe rutas, ejecuta ffmpeg y
devuelve números.
"""

from __future__ import annotations

import subprocess

import numpy as np
from src.ffmpeg.process_supervisor import process_group_popen_kwargs

# Resolución de la envolvente. 20 ms bastan para colocar un doblaje y no
# confunden un corte de escena con un desfase.
FRAME_SECONDS = 0.02
SAMPLE_RATE = 8000
# Cuánto audio se compara como máximo. Una intro distinta se nota en el primer
# minuto; analizar el episodio entero solo cuesta tiempo.
ANALYSIS_SECONDS = 600.0
# Diferencia máxima que se busca. Una intro/recap de más de 30 s ya no es un
# desfase, es otro montaje.
MAX_OFFSET_SECONDS = 30.0
# Por debajo de esto la correlación no es una prueba: mejor no tocar nada.
MIN_CONFIDENCE = 0.25

# Para escuchar el resultado hace falta audio de verdad, no la envolvente de
# 8 kHz de la detección. Mono a 48 kHz sobra para juzgar un doblaje y un trozo
# de dos minutos ocupa unos 12 MB.
PREVIEW_SAMPLE_RATE = 48000
# Trozo que se decodifica para escuchar en el editor de sincronía. Cubre una
# escena con varias réplicas sin cargar el episodio entero en memoria.
PREVIEW_SECONDS = 120.0


def read_audio(ffmpeg_path: str, path: str, *, seconds=ANALYSIS_SECONDS,
               stream_index: int = 0) -> np.ndarray:
    """Decodifica la primera ``seconds`` de una pista de audio a mono float32."""
    command = [
        ffmpeg_path, "-hide_banner", "-nostdin", "-vn", "-sn", "-dn",
        "-i", path,
        "-map", f"0:a:{stream_index}",
        "-t", f"{float(seconds):.3f}",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-",
    ]
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
    }
    kwargs.update(process_group_popen_kwargs())
    completed = subprocess.run(command, **kwargs)
    if completed.returncode != 0:
        raise RuntimeError("FFmpeg could not read the audio of this file.")
    samples = np.frombuffer(completed.stdout, dtype=np.int16)
    if samples.size == 0:
        raise RuntimeError("This file has no audio to compare.")
    return samples.astype(np.float32) / 32768.0


def energy_envelope(samples: np.ndarray, *, sample_rate=SAMPLE_RATE,
                    frame_seconds=FRAME_SECONDS) -> np.ndarray:
    """Energía por tramas en escala logarítmica (una muestra por trama)."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)
    frame = max(1, int(round(sample_rate * frame_seconds)))
    usable = (samples.size // frame) * frame
    if usable <= 0:
        return np.zeros(0, dtype=np.float64)
    frames = samples[:usable].reshape(-1, frame).astype(np.float64)
    energy = np.mean(frames * frames, axis=1)
    # log1p comprime los picos para que un golpe de música no domine toda la
    # correlación; el suelo evita -inf en silencios digitales.
    return np.log1p(energy * 1e4)


def estimate_offset(base: np.ndarray, dub: np.ndarray, *,
                    frame_seconds=FRAME_SECONDS,
                    max_offset=MAX_OFFSET_SECONDS) -> tuple[float, float]:
    """Retardo ``(segundos, confianza)`` que hay que aplicar al doblaje.

    Positivo = el doblaje va por delante y hay que retrasarlo; negativo = va por
    detrás y hay que adelantarlo. La confianza es la correlación normalizada en
    el pico (0 a 1); sin picos claros se queda cerca de cero.
    """
    if base.size < 4 or dub.size < 4:
        raise ValueError("Not enough audio to compare.")
    # Se correlaciona la DERIVADA de la envolvente, no la envolvente: quitar la
    # parte lenta (nivel medio, música de fondo sostenida) deja los cambios de
    # escena y de energía, que son los que se alinean entre dos idiomas. Con la
    # envolvente cruda, dos tramos con el mismo nivel se parecen entre sí y el
    # pico puede caer en el sitio equivocado.
    a = np.diff(energy_envelope(base, frame_seconds=frame_seconds))
    b = np.diff(energy_envelope(dub, frame_seconds=frame_seconds))
    if a.size < 4 or b.size < 4:
        raise ValueError("Not enough audio to compare.")
    a = a - a.mean()
    b = b - b.mean()
    norm = float(np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))
    if norm <= 0:
        return 0.0, 0.0

    # El tamaño es al menos len(a) + len(b) - 1 para que la correlación
    # circular sea la lineal. En esa disposición el desfase 0 cae en el índice
    # 0 y los negativos en la cola (size - k), no al revés: de ahí el mapa de
    # lags en vez de restar len(b) - 1 como haría numpy.correlate.
    size = 1 << (a.size + b.size - 1).bit_length()
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    correlation = np.fft.irfft(spectrum, size)
    indices = np.arange(size)
    lags = np.where(indices <= size // 2, indices, indices - size)
    limit = int(round(max_offset / frame_seconds))
    window = np.abs(lags) <= limit
    if not np.any(window):
        return 0.0, 0.0
    correlation = correlation[window] / norm
    lags = lags[window]
    best = int(np.argmax(correlation))
    peak = float(correlation[best])
    # Un pico alto no basta: si el resto de desfases puntúa casi igual, ese
    # "pico" es ruido. La confianza mide cuánto destaca sobre el mejor valor
    # fuera de medio segundo alrededor, en escala 0-1.
    guard = max(1, int(round(0.5 / frame_seconds)))
    mask = np.ones(correlation.size, dtype=bool)
    mask[max(0, best - guard):best + guard + 1] = False
    sidelobe = float(np.max(correlation[mask])) if np.any(mask) else 0.0
    ratio = peak / max(abs(sidelobe), 1e-9)
    confidence = 0.0 if ratio <= 0 else 1.0 - 1.0 / ratio
    return float(lags[best] * frame_seconds), float(confidence)


def detect_delay(ffmpeg_path: str, base_path: str, dub_path: str, *,
                 seconds=ANALYSIS_SECONDS, max_offset=MAX_OFFSET_SECONDS,
                 dub_stream_index: int = 0, abort_event=None) -> dict:
    """Compara el audio de los dos ficheros y devuelve el desfase medido.

    ``{"offset": float | None, "confidence": float, "reason": str}``. ``offset``
    es ``None`` cuando no hay una medición en la que apoyarse (sin audio, poca
    confianza); el llamante decide entonces no tocar el desfase.
    ``dub_stream_index`` es la pista del fichero añadido que se va a usar, no
    siempre la primera.
    """
    try:
        base = read_audio(ffmpeg_path, base_path, seconds=seconds)
        if abort_event is not None and abort_event.is_set():
            return {"offset": None, "confidence": 0.0, "reason": "Cancelled by user."}
        dub = read_audio(
            ffmpeg_path, dub_path, seconds=seconds,
            stream_index=dub_stream_index,
        )
    except Exception as exc:
        return {"offset": None, "confidence": 0.0, "reason": str(exc)}
    if abort_event is not None and abort_event.is_set():
        return {"offset": None, "confidence": 0.0, "reason": "Cancelled by user."}
    try:
        offset, confidence = estimate_offset(
            base, dub, max_offset=max_offset
        )
    except ValueError as exc:
        return {"offset": None, "confidence": 0.0, "reason": str(exc)}
    if confidence < MIN_CONFIDENCE:
        return {
            "offset": None,
            "confidence": confidence,
            "reason": (
                f"Could not match the audio (confidence {confidence:.2f}). "
                "Set the delay by hand."
            ),
        }
    return {
        "offset": round(offset, 2),
        "confidence": confidence,
        "reason": "",
    }


# ---------------------------------------------------------------------------
# Forma de onda y escucha (editor de sincronía)
# ---------------------------------------------------------------------------


def peak_frames(samples: np.ndarray, frame: int) -> np.ndarray:
    """Pico absoluto de cada trama de ``frame`` muestras, en escala 0-1.

    Se calcula sobre int16 sin normalizar antes: dibujar la onda con valores
    ya escalados evita depender del volumen del fichero en cada repintado.
    """
    frame = max(1, int(frame))
    frames = samples.size // frame
    if frames <= 0:
        return np.zeros(0, dtype=np.float32)
    # int32 antes de abs(): abs(-32768) desborda en int16 y devuelve -32768.
    block = samples[:frames * frame].reshape(frames, frame).astype(np.int32)
    peaks = np.abs(block).max(axis=1).astype(np.float32) / 32768.0
    return np.clip(peaks, 0.0, 1.0)


def read_waveform(ffmpeg_path: str, path: str, *, stream_index: int = 0,
                  sample_rate: int = SAMPLE_RATE,
                  frame_seconds: float = FRAME_SECONDS,
                  abort_event=None) -> np.ndarray:
    """Un pico por trama de una pista, sin cargar el audio entero.

    Se lee la salida de FFmpeg por bloques y se reduce cada bloque a su pico
    por trama. Un episodio de dos horas acaba en unos 360 000 valores (1,4 MB)
    en vez de en cientos de megas de muestras, así que se puede dibujar la
    pista completa y hacer zoom sin volver a decodificar.
    """
    frame = max(1, int(round(sample_rate * frame_seconds)))
    command = [
        ffmpeg_path, "-hide_banner", "-nostdin", "-vn", "-sn", "-dn",
        "-i", path, "-map", f"0:a:{int(stream_index)}",
        "-ac", "1", "-ar", str(int(sample_rate)), "-f", "s16le", "-",
    ]
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
    }
    kwargs.update(process_group_popen_kwargs())
    proc = subprocess.Popen(command, **kwargs)
    pending = b""
    peaks = []
    try:
        while True:
            if abort_event is not None and abort_event.is_set():
                _terminate(proc)
                raise RuntimeError("Cancelled by user.")
            block = proc.stdout.read(1 << 19)
            if not block:
                break
            pending += block
            usable = (len(pending) // (2 * frame)) * (2 * frame)
            if not usable:
                continue
            chunk = np.frombuffer(pending[:usable], dtype=np.int16)
            peaks.append(peak_frames(chunk, frame))
            pending = pending[usable:]
    finally:
        if proc.poll() is None:
            _terminate(proc)
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError("FFmpeg could not read the audio of this file.")
    if not peaks:
        raise RuntimeError("This file has no audio to compare.")
    return np.concatenate(peaks)


def read_pcm(ffmpeg_path: str, path: str, *, start: float = 0.0,
             seconds: float | None = None, stream_index: int = 0,
             sample_rate: int = PREVIEW_SAMPLE_RATE) -> np.ndarray:
    """Muestras int16 mono de un trozo de una pista, listas para escuchar."""
    command = [ffmpeg_path, "-hide_banner", "-nostdin", "-vn", "-sn", "-dn"]
    if start > 0:
        command += ["-ss", f"{float(start):.3f}"]
    command += ["-i", path, "-map", f"0:a:{int(stream_index)}"]
    if seconds is not None:
        command += ["-t", f"{float(seconds):.3f}"]
    command += [
        "-ac", "1", "-ar", str(int(sample_rate)), "-f", "s16le", "-",
    ]
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
    }
    kwargs.update(process_group_popen_kwargs())
    completed = subprocess.run(command, **kwargs)
    if completed.returncode != 0:
        raise RuntimeError("FFmpeg could not read the audio of this file.")
    if not completed.stdout:
        raise RuntimeError("This file has no audio to compare.")
    return np.frombuffer(completed.stdout, dtype=np.int16)


def align_clip(raw: np.ndarray, raw_start: float, start: float, count: int,
               delay: float, *,
               sample_rate: int = PREVIEW_SAMPLE_RATE) -> np.ndarray:
    """Trozo del doblaje tal como sonará al aplicarle ``delay``.

    ``raw`` empieza en ``raw_start`` (segundos del fichero), ``start`` es el
    inicio de la ventana en la línea de tiempo del vídeo y ``count`` sus
    muestras. Un desfase positivo retrasa el doblaje, así que en la ventana se
    oye el audio que en el fichero estaba ``delay`` segundos ANTES; lo que no
    existe (antes del principio o después del final) se rellena con silencio,
    igual que hará ``adelay`` / ``atrim`` en el mux.
    """
    out = np.zeros(int(count), dtype=np.int16)
    if raw.size == 0:
        return out
    source_start = float(start) - float(delay)
    begin = int(round((source_start - float(raw_start)) * sample_rate))
    source_from = max(0, begin)
    destination_from = max(0, -begin)
    take = min(int(count) - destination_from, int(raw.size) - source_from)
    if take > 0:
        out[destination_from:destination_from + take] = \
            raw[source_from:source_from + take]
    return out


def _terminate(proc) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
