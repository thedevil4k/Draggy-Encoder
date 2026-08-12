from __future__ import annotations

import collections
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime

from src.config import get_bin_dir, resolve_codec
from src.ffmpeg.manager import get_ffprobe_path
from src.ffmpeg.process_supervisor import (
    ProcessSupervisor,
    process_group_popen_kwargs,
)


logger = logging.getLogger(__name__)


# Regex de parseo de la línea 'ffmpeg version X.Y…' que devuelve
# ``ffmpeg -version``. Se mantiene local para no introducir ciclo de
# imports con ``src.ffmpeg.manager`` (que tiene su propia copia de la
# misma regex). Si esto cambia, sincronizar los dos sitios.
_VERSION_RE = re.compile(r"ffmpeg version\s+([^\s]+)", re.IGNORECASE)


def _log_dir():
    log_dir = os.path.join(get_bin_dir(), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    return log_dir


def _append_ffmpeg_log(cmd_summary, log_tail, returncode):
    """Append a failure entry to bin/logs/ffmpeg.log for diagnostics."""
    try:
        log_path = os.path.join(_log_dir(), "ffmpeg.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] rc={returncode} cmd={cmd_summary}\n")
            for line in log_tail:
                f.write(f"  {line}\n")
            f.write("\n")
    except Exception:
        pass


def _parse_time_to_seconds(value):
    """Convierte un valor de tiempo en segundos.

    Acepta números (int/float/str) y cadenas con formato HH:MM:SS o H:MM:SS.
    Si no se puede parsear, devuelve None.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    # Intentar parseo directo como número
    try:
        return float(text)
    except ValueError:
        pass
    # Intentar formato HH:MM:SS o H:MM:SS
    parts = text.split(":")
    if len(parts) == 2:
        try:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except ValueError:
            return None
    if len(parts) == 3:
        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return None
    return None


# Aplicar codificación UTF-8 a todos los subprocesos ffmpeg/ffprobe para evitar
# UnicodeDecodeError en Windows (charmap no puede decodificar algunos bytes que
# ffmpeg imprime en su progreso). `errors="replace"` reemplaza los caracteres
# no decodificables por '?' sin abortar el proceso.
def _popen_kwargs():
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 1,
        "encoding": "utf-8",
        "errors": "replace",
    }
    kwargs.update(process_group_popen_kwargs())
    return kwargs


def _run_kwargs():
    return {
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "encoding": "utf-8",
        "errors": "replace",
        "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    }


# Polling cadence inside ``run_ffmpeg``: how often we wake up to check
# ``abort_event`` and process exit when stdout is silent. Lower values
# mean faster Abort response (the user sees FFmpeg stop immediately
# after clicking Abort), higher values mean less CPU spinning. The
# previous blocking ``for line in process.stdout`` left the abort
# button unresponsive for 5+ seconds on silent encodes (audio=copy,
# very fast NVENC, tiny clips).
_FFMPEG_POLL_INTERVAL_S = 0.25


def run_ffmpeg(ffmpeg_path, args, progress_callback=None, abort_event=None,
               on_process=None, supervisor: ProcessSupervisor | None = None):
    """Ejecuta ffmpeg con los argumentos dados y opcionalmente reporta progreso.

    Devuelve ``True`` si ffmpeg terminó con ``returncode == 0`` y ``False`` en
    caso contrario (incluyendo aborto). Cuando falla, las últimas líneas de
    salida se vuelcan a ``bin/logs/ffmpeg.log`` para diagnóstico.

    ``on_process`` is an optional callable invoked once with the
    ``subprocess.Popen`` instance right after Popen returns. The GUI
    attaches the reference so it can ``kill()`` the child directly
    during a hard abort (e.g. when the abort_event path doesn't reach
    run_ffmpeg in time — see ``_abort_running_worker`` in
    ``src/ui/main_window.py``).
    """
    cmd = [ffmpeg_path] + args
    process = subprocess.Popen(cmd, **_popen_kwargs())
    if supervisor is not None:
        supervisor.register(process, "ffmpeg")
    if on_process is not None:
        try:
            on_process(process)
        except Exception:
            pass

    log_tail = collections.deque(maxlen=40)
    # ``duration`` is wrapped in a single-element list so the reader
    # thread (which fills it once ffmpeg prints its ``Duration:`` line)
    # and the main thread (which only reads via ``_terminate_for_abort``
    # after the loop exits) share the same cell safely without
    # requiring a lock for the racy write-once / read-after pattern.
    duration = [None]

    def _reader_loop():
        """Daemon thread: consume stdout line by line.

        ``process.stdout.readline()`` blocks on every platform (Linux
        and Windows).
        A previous attempt used stdlib ``selectors`` for non-blocking
        polling, but ``selectors.DefaultSelector()`` resolves to
        ``SelectSelector`` on Windows, which only supports *sockets*,
        not pipes. ``sel.register(process.stdout, ...)`` then raises
        ``ValueError`` and the abort path silently degrades back to the
        blocking readline(). The canonical cross-platform pattern is
        a dedicated reader thread: read blocks here once, the main
        thread polls ``abort_event`` and ``process.poll()`` separately
        at ``_FFMPEG_POLL_INTERVAL_S`` cadence, and on abort we
        ``terminate()`` (which closes the subprocess's stdout pipe)
        so the reader sees EOF and exits cleanly.
        """
        try:
            for line in process.stdout:
                log_tail.append(line.rstrip())
                if progress_callback:
                    # Parse progress inline. The reader thread is the
                    # only writer to ``duration``; the main thread
                    # reads it only after the while loop drains.
                    if duration[0] is None:
                        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", line)
                        if m:
                            duration[0] = (
                                float(m.group(1)) * 3600
                                + float(m.group(2)) * 60
                                + float(m.group(3))
                            )
                    m = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
                    if m and duration[0]:
                        current = (
                            float(m.group(1)) * 3600
                            + float(m.group(2)) * 60
                            + float(m.group(3))
                        )
                        try:
                            progress_callback(min(int(current * 100 / duration[0]), 100))
                        except Exception:
                            pass
        except (ValueError, OSError):
            # pipe closed by ``terminate()`` / ``kill()`` — expected
            # during an abort, not a bug.
            pass
        except Exception:
            logger.exception("Unexpected error in ffmpeg reader thread")
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass

    reader = threading.Thread(target=_reader_loop, daemon=True, name="ffmpeg-reader")
    reader.start()

    def _terminate_for_abort():
        """Process-abort bookkeeping shared by both flag paths.

        Tries ``terminate()`` + ``wait(5)`` first; if FFmpeg didn't
        react in five seconds we ``kill()`` and ``wait(2)``. Closes
        ``process.stdout`` so the reader thread sees EOF and exits
        within its ``join(timeout=2.0)``. The outer caller
        (``CompressionWorker``) also has a ``worker._proc.kill()``
        safety net for the case where the Python wrapper dies before
        ``run_ffmpeg`` polls abort.
        """
        try:
            # Closing stdout unblocks the reader immediately if ffmpeg
            # itself hasn't yet closed the pipe end.
            try:
                process.stdout.close()
            except Exception:
                pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
        except Exception:
            logger.exception("Error during FFmpeg abort sequence")
        # Wait for the reader to finish draining whatever remains.
        try:
            reader.join(timeout=2.0)
        except Exception:
            pass
        _append_ffmpeg_log(" ".join(args[:6]) + "...", list(log_tail), -1)
        return False

    # Main thread: poll ``process.wait(timeout)`` and ``abort_event``
    # on every ``_FFMPEG_POLL_INTERVAL_S`` cycle. ``wait(timeout)``
    # either returns ffmpeg's returncode (clean exit ⇒ break the
    # loop) or raises ``TimeoutExpired`` (ffmpeg still alive ⇒ check
    # abort). Reader thread consumes stdout in parallel; closing the
    # pipe via ``process.stdout.close()``/``terminate()`` makes its
    # ``readline()`` return immediately.
    while True:
        try:
            process.wait(timeout=_FFMPEG_POLL_INTERVAL_S)
            break  # ffmpeg exited cleanly
        except subprocess.TimeoutExpired:
            pass
        if abort_event and abort_event.is_set():
            return _terminate_for_abort()

    # Drain whatever buffered lines the reader captured during ffmpeg's
    # final output flush before it closed stdout.
    try:
        reader.join(timeout=2.0)
    except Exception:
        pass
    if process.returncode != 0:
        tail_text = "\n".join(log_tail)
        cmd_summary = " ".join([str(a) for a in args[:8]]) + (" ..." if len(args) > 8 else "")
        _append_ffmpeg_log(cmd_summary, list(log_tail), process.returncode)
        logger.error("FFmpeg failed (rc=%s) for %s. Tail:\n%s", process.returncode, cmd_summary, tail_text)
        return False
    return True


def get_video_info(ffmpeg_path, video_path):
    """Devuelve información básica del vídeo usando ffprobe."""
    ffprobe = get_ffprobe_path()
    if not ffprobe or not os.path.exists(video_path):
        return {}
    try:
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration,size,bit_rate:stream=width,height,r_frame_rate,codec_name,nb_frames",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ]
        result = subprocess.run(cmd, timeout=30, **_run_kwargs())
        info = {}
        for line in (result.stdout or "").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
        return info
    except Exception:
        return {}


def health_check_ffmpeg(ffmpeg_path, timeout=10):
    """Smoke-test rápido del binario FFmpeg antes de una compresión real.

    Útil cuando el binario fue copiado a mano o viene de un mirror raro:
    podemos encontrarnos con un ffmpeg que existe pero está corrupto,
    truncado, mal extraído o sin permisos. Esta función ejecuta
    ``<ffmpeg> -version`` con timeout acotado y devuelve:

    - ``ok`` (bool): True solo si ``returncode == 0`` y parseamos una
      cadena de versión del stdout.
    - ``version`` (str | None): versión parseada por
      ``_VERSION_RE`` (compartido con ``src.ffmpeg.manager`` via
      duplicación; aquí usamos una versión local para no acoplar este
      módulo a manager).
    - ``error_msg`` (str | None): tail corto del stdout/stderr si la
      verificación falla; None si todo sale bien. El caller sólo lo usa
      para el modal “FFmpeg no responde, ¿continuar?”.

    Fail-safe: cualquier excepción se mapea a ``(False, None, str(exc))``
    para que el UI nunca vea un crash sin contexto.
    """
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        return False, None, f"Binary not found at {ffmpeg_path!r}"
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            timeout=timeout,
            **_run_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return False, None, f"Timed out after {timeout}s"
    except Exception as exc:
        return False, None, str(exc)

    text = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        tail = "\n".join(text.splitlines()[-8:]).strip() or "(no output)"
        return False, None, f"rc={result.returncode}\n{tail}"

    match = _VERSION_RE.search(text)
    if match:
        return True, match.group(1).strip(), None
    # Sin version parseada pero rc=0: binario responde pero no es ffmpeg.
    return False, None, "Binary responded but no version line was found"


def extract_frame(ffmpeg_path, video_path, time_sec, output_path,
                  abort_event=None, supervisor: ProcessSupervisor | None = None):
    """Extrae un frame sin dejar un ffmpeg bloqueado durante un abort."""
    cmd = [
        ffmpeg_path,
        "-ss", str(time_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        output_path,
    ]
    proc = None
    try:
        kwargs = _run_kwargs()
        kwargs.pop("capture_output", None)
        # This helper only needs the exit code and output image. Keeping
        # FFmpeg's verbose stderr in an undrained pipe can deadlock the
        # process on Windows once that pipe fills (the full build prints a
        # multi-kilobyte banner even for a one-frame extraction).
        kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
        kwargs.update(process_group_popen_kwargs())
        proc = subprocess.Popen(cmd, **kwargs)
        if supervisor is not None:
            supervisor.register(proc, "frame-extract")
        deadline = time.monotonic() + 60
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
                return False
            if time.monotonic() >= deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                return False
            time.sleep(0.1)
        ok = proc.returncode == 0
        if not ok:
            try:
                logger.warning("Frame extraction failed: %s", (proc.stderr.read() or "")[-512:])
            except Exception:
                pass
        return ok
    except Exception:
        return False
    finally:
        if proc is not None:
            try:
                proc.stderr.close()
            except Exception:
                pass


def build_args(input_path, output_path, settings, loudnorm_meas=None):
    """Construye los argumentos de ffmpeg a partir de los ajustes.

    Audio:
        - noaudio      → -an (sin audio)
        - copy sin boost → -c:a copy (no re-codifica)
        - aac / mp3 / copy + boost → re-codifica con loudnorm EBU R128
          (I=-16 LUFS para cine, TP=-1.5, LRA=11 dB)

    Si ``loudnorm_meas`` es un dict con input_i/input_tp/input_lra/input_thresh/
    target_offset (devuelto por ``src.ffmpeg.loudnorm.measure_loudness``), se
    aplica el modo lineal de dos pasadas para precisión broadcast (-0.5 dB de
    desviación típico vs ±3 dB del single-pass).
    """
    # Use input seeking (-ss before -i) for fast seeking on large files.
    # Use -t (duration) after the input to avoid timestamp reset surprises.
    start_time = settings.get("start_time")
    end_time = settings.get("end_time")
    args = ["-y"]
    if start_time and str(start_time).strip():
        args.extend(["-ss", str(start_time).strip()])

    args.extend(["-i", input_path])

    # When both start and end are present, encode the *duration* of the segment
    # instead of the absolute output timestamp. This works correctly regardless
    # of whether -ss is an input or output option.
    start_sec = _parse_time_to_seconds(start_time)
    end_sec = _parse_time_to_seconds(end_time)
    if end_sec is not None:
        if start_sec is not None and end_sec > start_sec:
            duration = end_sec - start_sec
            args.extend(["-t", str(duration)])
        elif start_sec is None:
            # Solo end_time: codificar desde el inicio hasta ese punto.
            args.extend(["-t", str(end_sec)])
        else:
            # end <= start: valores inválidos; fallar de forma explícita para que
            # el usuario corrija el rango de recorte.
            raise ValueError(f"Invalid trim range: start={start_time!r} must be less than end={end_time!r}")

    # Video codec
    codec = settings.get("codec", "libx264")
    real_codec, codec_extra_args = resolve_codec(codec)
    args.extend(["-c:v", real_codec])
    args.extend(codec_extra_args)

    # Bitrate / target size
    if settings.get("mode") == "Target File Size (MB)" and settings.get("target_size_mb"):
        # Calcular bitrate de video aproximado
        duration = settings.get("duration_seconds", 0)
        target_mb = float(settings["target_size_mb"])
        if duration > 0:
            total_bits = target_mb * 8 * 1024 * 1024
            # Reservar 192 kbps para el audio (mp3, aac o loudnorm).
            audio_bitrate = 192000
            video_bitrate = max(int((total_bits / duration) - audio_bitrate), 100000)
            args.extend(["-b:v", str(video_bitrate)])
        else:
            args.extend(["-b:v", f"{settings.get('bitrate', 1500)}k"])
    else:
        args.extend(["-b:v", f"{settings.get('bitrate', 1500)}k"])

    # Resolución
    resolution = settings.get("resolution")
    if resolution and resolution != "original":
        args.extend(["-vf", f"scale=-2:{resolution}"])

    # Audio con soporte de audio boost (single-pass o two-pass lineal)
    audio = settings.get("audio", "copy")
    audio_boost = bool(settings.get("audio_boost", False))

    if audio == "noaudio":
        args.extend(["-an"])
    elif audio == "copy" and not audio_boost:
        # Sin boost y Copy → preservamos audio original
        args.extend(["-c:a", "copy"])
    else:
        # Cualquier otro caso (incluido copy + boost) fuerza re-codificación para
        # poder aplicar loudnorm.
        codec_a = "libmp3lame" if audio == "mp3" else "aac"
        args.extend(["-c:a", codec_a, "-b:a", "192k"])
        if audio_boost:
            af = _build_loudnorm_filter(loudnorm_meas)
            args.extend(["-af", af])

    # Formato contenedor
    fmt = settings.get("format", "mp4")
    if fmt == "wmv":
        args.extend(["-f", "asf"])
    elif fmt == "webm":
        args.extend(["-f", "webm"])

    args.append(output_path)
    return args


def _build_loudnorm_filter(measured):
    """Construye el filtro ``-af loudnorm=...``.

    Modo two-pass (linear): usa los valores medidos del input para una
    normalización EBU R128 de precisión broadcast. Si la medición falló o no
    tiene las claves requeridas, cae al modo single-pass con valores por
    defecto (-16 / -1.5 / 11).
    """
    required = ("input_i", "input_tp", "input_lra", "input_thresh")
    if not measured or not all(isinstance(measured.get(k), (str, int, float)) for k in required):
        return "loudnorm=I=-16:TP=-1.5:LRA=11"
    try:
        return (
            "loudnorm=I=-16:TP=-1.5:LRA=11:"
            f"measured_I={float(measured['input_i']):.4f}:"
            f"measured_TP={float(measured['input_tp']):.4f}:"
            f"measured_LRA={float(measured['input_lra']):.4f}:"
            f"measured_thresh={float(measured['input_thresh']):.4f}:"
            f"offset={float(measured.get('target_offset', 0)):.4f}:"
            "linear=true"
        )
    except (ValueError, TypeError):
        return "loudnorm=I=-16:TP=-1.5:LRA=11"


