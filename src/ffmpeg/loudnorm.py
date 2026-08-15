"""
Two-pass loudnorm measurement for EBU R128 broadcast precision.

Pass 1 (measure):
    ffmpeg -hide_banner -nostats -i INPUT \
           -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json \
           -f null -
The loudnorm filter prints a JSON block to stderr with measured stats of the
source audio.

Pass 2 uses ``build_args(...)`` with these measured values to apply
linear-mode loudnorm, which gives significantly more accurate normalization
than the single-pass with default values.

NOTE: This module only imports ``_run_kwargs`` from ``src.ffmpeg.runner``.
It must NOT import anything back from pipeline.py or anything that imports
pipeline.py, to avoid a circular dependency.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time

from src.ffmpeg.process_supervisor import (
    ProcessSupervisor,
    process_group_popen_kwargs,
)
from datetime import datetime

from src.config import get_bin_dir
from src.ffmpeg.runner import (
    KEEP_ALL_AUDIO,
    _parse_time_to_seconds,
    _run_kwargs,
    parse_audio_stream_index,
)


def _log_dir():
    """Directorio de logs (mismo patrón que codec_tester)."""
    log_dir = os.path.join(get_bin_dir(), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    return log_dir


def _append_log(status, detail):
    """Anexa una línea a bin/logs/loudnorm.log con timestamp."""
    try:
        log_file = os.path.join(_log_dir(), "loudnorm.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {status} {detail}\n")
    except Exception:
        pass


def measure_loudness(ffmpeg_path, input_path, timeout_seconds=600,
                       start_time=None, end_time=None, status_callback=None,
                       abort_event=None, supervisor: ProcessSupervisor | None = None,
                       audio_stream_index=0):
    """Run pass-1 loudnorm measurement and return parsed JSON dict, or None.

    Returns a dict with keys: input_i, input_tp, input_lra, input_thresh,
    target_offset (all strings with the measured LUFS / dB values).
    Returns None if:
        - The ffmpeg process errors out
        - No audio stream is present
        - The JSON block cannot be located in stderr
        - The process times out

    Optional ``start_time`` / ``end_time`` (seconds) let you measure a clip
    instead of the whole file (used by preview).
    """
    if not ffmpeg_path or not input_path:
        _append_log("SKIP", f"missing ffmpeg_path or input_path ({input_path})")
        return None

    if status_callback:
        status_callback("Analyzing audio levels...")

    cmd = [ffmpeg_path, "-hide_banner", "-nostats"]
    # Normalizar a segundos con el mismo parser que build_args para que el
    # recorte medido coincida exactamente con el segmento a codificar
    # (soporta "HH:MM:SS" y números).
    start_sec = _parse_time_to_seconds(start_time)
    end_sec = _parse_time_to_seconds(end_time)
    if start_sec is not None:
        cmd.extend(["-ss", str(start_sec)])
    cmd.extend(["-i", input_path])
    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        cmd.extend(["-t", str(end_sec - start_sec)])
    elif end_sec is not None:
        cmd.extend(["-to", str(end_sec)])
    audio_stream_index = parse_audio_stream_index(audio_stream_index)
    # Para "keep all" medimos la PRIMERA pista como referencia del boost;
    # una única medición no puede mezclar pistas con niveles distintos.
    map_idx = 0 if audio_stream_index == KEEP_ALL_AUDIO else audio_stream_index
    cmd.extend([
        "-map", f"0:a:{map_idx}?",
        "-vn",              # sin vídeo: no decodificar la pista más pesada
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-",
    ])

    proc = None
    try:
        kwargs = _run_kwargs()
        kwargs.pop("capture_output", None)
        # The JSON is emitted on stderr, so drain it concurrently. Waiting
        # for poll() before reading a PIPE can deadlock when FFmpeg's banner
        # and filter diagnostics exceed the Windows pipe buffer.
        kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE})
        kwargs.update(process_group_popen_kwargs())
        proc = subprocess.Popen(cmd, **kwargs)
        if supervisor is not None:
            supervisor.register(proc, "loudnorm")
        stderr_chunks = []

        def _drain_stderr():
            try:
                for line in iter(proc.stderr.readline, ""):
                    if line:
                        stderr_chunks.append(line)
            except (ValueError, OSError):
                pass

        stderr_reader = threading.Thread(target=_drain_stderr, daemon=True, name="loudnorm-stderr")
        stderr_reader.start()
        deadline = time.monotonic() + timeout_seconds
        while proc.poll() is None:
            if abort_event and abort_event.is_set():
                if supervisor is not None:
                    supervisor.stop_all(grace_seconds=1, kill_seconds=1)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                stderr_reader.join(timeout=2)
                return None
            if time.monotonic() >= deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                stderr_reader.join(timeout=2)
                _append_log("FAIL", f"timeout after {timeout_seconds}s ({input_path})")
                return None
            time.sleep(0.1)
        stderr_reader.join(timeout=2)
        stderr = "".join(stderr_chunks)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        _append_log("FAIL", f"timeout after {timeout_seconds}s ({input_path})")
        return None
    except Exception as exc:
        _append_log("FAIL", f"subprocess exception: {exc} ({input_path})")
        return None

    if proc.returncode != 0:
        # Truncar stderr para no saturar el log
        short = (stderr or "").strip().splitlines()[-3:]
        reason = " || ".join(s.strip() for s in short if s.strip())[:300]
        _append_log("FAIL", f"rc={proc.returncode} | {reason}")
        return None

    stderr = stderr or ""
    # El filtro loudnorm imprime siempre un único bloque JSON cuyo primer key
    # es ``input_i``. Un match no-greedy nos mantiene dentro de ese bloque
    # incluso si ffmpeg imprime texto adicional después.
    match = re.search(r'\{\s*"input_i".*?\}', stderr, re.DOTALL)
    if not match:
        _append_log("FAIL", "no JSON block in stderr")
        return None

    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError) as exc:
        _append_log("FAIL", f"JSON decode error: {exc}")
        return None

    required = ("input_i", "input_tp", "input_lra", "input_thresh")
    if not all(k in data for k in required):
        _append_log("FAIL", f"missing keys: {sorted(set(required) - set(data.keys()))}")
        return None

    _append_log(
        "OK",
        f"I={data['input_i']} TP={data['input_tp']} LRA={data['input_lra']} "
        f"thresh={data['input_thresh']} offset={data.get('target_offset', '?')}",
    )
    return data

