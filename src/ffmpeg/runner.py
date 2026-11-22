from __future__ import annotations

import collections
import json
import logging
import math
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
               on_process=None, supervisor: ProcessSupervisor | None = None,
               metrics_callback=None):
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

    ``metrics_callback`` es un canal APARTE y opcional para las cifras de
    throughput. No sustituye ni ensucia a ``progress_callback``: este sigue
    recibiendo un entero y solo un entero, porque es lo que todos los
    llamantes de hoy esperan. El bloque ``-progress`` de ffmpeg trae ``fps``,
    ``speed`` y ``frame`` en cada tic, y hasta ahora se leian enteros y se
    tiraban: aqui se recuperan y se pasan como un dict, a lo mucho una vez
    por segundo, que es cada cuanto una persona puede leerlos.

    Las claves llevan prefijo ``enc_`` (``enc_fps``, ``enc_speed``,
    ``enc_frame``, ``enc_out_time``) porque este canal no es el unico que
    escribe en el desplegable: el bucle de render de IA publica sus propias
    cifras con prefijo ``ai_``. Con dos fuentes escribiendo en la misma clave
    ``fps``, la que hablaba ultimo borraba a la otra y no habia forma de saber
    a quien pertenecía el numero que se veia en pantalla. Prefijar es lo que
    hace falta para que ambas convivan.
    """
    # Human-readable FFmpeg statistics are written with carriage returns and
    # are therefore buffered by ``for line in stdout`` until the encode ends.
    # Ask FFmpeg for its machine-readable progress protocol instead so audio
    # and video progress callbacks advance continuously on Windows too.
    if progress_callback or metrics_callback:
        cmd = [
            ffmpeg_path, "-hide_banner", "-nostats", "-progress", "pipe:1",
            "-stats_period", "0.1",
        ] + list(args)
    else:
        cmd = [ffmpeg_path] + list(args)
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
    last_progress = [-1]
    # Ultimas metricas de ffmpeg y cuando se publicaron por ultima vez. El
    # bloque -progress llega cada ``stats_period`` (100 ms), que para una
    # persona es ruido: se guarda lo ultimo y se emite como mucho una vez
    # por segundo.
    # ``latest`` lleva prefijo ``enc_`` en todas sus claves: lo que hay aqui
    # es siempre lo que dijo ffmpeg, nunca lo que va contando el motor de IA.
    latest = {}
    last_metrics_emit = [0.0]

    def _emit_metrics():
        """Publica lo ultimo que dijo ffmpeg, sin repetirlo cada 100 ms."""
        if not metrics_callback or not latest:
            return
        now = time.monotonic()
        if now - last_metrics_emit[0] < 1.0:
            return
        last_metrics_emit[0] = now
        try:
            metrics_callback(dict(latest))
        except Exception:
            pass

    def _emit_time_progress(current_seconds):
        if not progress_callback or duration[0] is None or not current_seconds:
            return
        try:
            pct = min(max(int(current_seconds * 100 / duration[0]), 0), 100)
            # FFmpeg can repeat a timestamp or report a small correction when
            # streams have different time bases; never move the UI backwards.
            if pct < last_progress[0]:
                return
            if pct == last_progress[0]:
                return
            last_progress[0] = pct
            progress_callback(pct)
        except Exception:
            pass

    def _parse_progress_line(line):
        """Parse both FFmpeg's progress protocol and legacy stats lines."""
        if not progress_callback:
            return
        if duration[0] is None:
            m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", line)
            if m:
                duration[0] = (
                    float(m.group(1)) * 3600
                    + float(m.group(2)) * 60
                    + float(m.group(3))
                )

        key, sep, value = line.partition("=")
        if metrics_callback and sep and key.strip() in {
            "fps", "speed", "frame", "out_time_us", "out_time_ms", "out_time",
        }:
            # Se guarda SIEMPRE, y se emite aparte mas abajo: el throttle no
            # puede comerse el dato, solo la publicacion.
            if key.strip() in ("fps", "frame"):
                try:
                    latest["enc_" + key.strip()] = float(value.strip())
                except ValueError:
                    pass
            elif key.strip() == "speed":
                texto = value.strip()
                try:
                    # ffmpeg escribe "1.25x"; el numero solo, sin la x.
                    latest["enc_speed"] = float(texto.rstrip("x"))
                except ValueError:
                    pass
            elif key.strip() == "out_time":
                latest["enc_out_time"] = _parse_time_to_seconds(value.strip())
            else:
                try:
                    # out_time_us y out_time_ms son microsegundos las dos
                    # (el nombre en "_ms" es historico).
                    latest["enc_out_time"] = float(value.strip()) / 1_000_000.0
                except ValueError:
                    pass

        if sep and key.strip() in {"out_time_us", "out_time_ms"}:
            try:
                # Both names are expressed in microseconds by FFmpeg's
                # progress protocol (the *_ms name is historical).
                _emit_time_progress(float(value.strip()) / 1_000_000.0)
            except (TypeError, ValueError):
                pass
            return
        if sep and key.strip() == "out_time":
            _emit_time_progress(_parse_time_to_seconds(value.strip()))
            return
        if sep and key.strip() == "progress":
            # Cada bloque del protocolo -progress termina aqui, tanto si dice
            # "continue" como "end". Se publica en todos, y es el throttle de
            # un segundo el que deja pasar solo los que leen. Publicar
            # solo en el "end" hacia que en un render de siete segundos saliera
            # un unico evento, al final, cuando la ventana ya habia parado el
            # muestreo: la velocidad se veria siempre a cero.
            _emit_metrics()
            if value.strip() == "end":
                if last_progress[0] < 100:
                    last_progress[0] = 100
                    try:
                        progress_callback(100)
                    except Exception:
                        pass
            return

        # Keep compatibility with callers/older FFmpeg builds that emit the
        # human-readable ``time=HH:MM:SS.xx`` form.
        m = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
        if m:
            current = (
                float(m.group(1)) * 3600
                + float(m.group(2)) * 60
                + float(m.group(3))
            )
            _emit_time_progress(current)

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
                    # The reader thread is the only writer to ``duration``;
                    # the main thread reads it only after the loop drains.
                    _parse_progress_line(line)
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
            "-show_entries", "format=duration,size,bit_rate:stream=width,height,r_frame_rate,codec_name,nb_frames,sample_aspect_ratio:stream_side_data=rotation",
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


def get_audio_streams(ffmpeg_path, video_path):
    """Return the audio tracks in ``video_path`` in FFmpeg audio order.

    Each result contains ``audio_index`` (the value used by ``-map
    0:a:<index>``), the global stream ``index`` and user-facing metadata such
    as language, title, codec and channel count. An empty list is returned on
    missing binaries, malformed media or probe errors.
    """
    ffprobe = get_ffprobe_path()
    if not ffprobe or not os.path.exists(video_path):
        return []
    try:
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "a",
            "-show_entries",
            "stream=index,codec_name,channels,channel_layout,sample_rate,bit_rate:"
            "stream_tags=language,title",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, timeout=30, **_run_kwargs())
        if result.returncode != 0:
            return []
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            return []
        tracks = []
        for audio_index, stream in enumerate(streams):
            if not isinstance(stream, dict):
                continue
            tags = stream.get("tags") or {}
            tracks.append({
                "audio_index": audio_index,
                "index": stream.get("index"),
                "codec_name": stream.get("codec_name", ""),
                "channels": stream.get("channels"),
                "channel_layout": stream.get("channel_layout", ""),
                "sample_rate": stream.get("sample_rate", ""),
                "bit_rate": stream.get("bit_rate", ""),
                "language": tags.get("language", ""),
                "title": tags.get("title", ""),
            })
        return tracks
    except Exception:
        return []


def get_subtitle_streams(ffmpeg_path, video_path):
    """Pistas de subtítulos de ``video_path``, en orden de FFmpeg.

    ``subtitle_index`` es el índice que usa ``-map 0:s:<index>``; ``index`` es
    el stream global y el resto son etiquetas para mostrarlas o copiarlas.
    Devuelve lista vacía si no hay ffprobe, el fichero no existe o falla.
    """
    ffprobe = get_ffprobe_path()
    if not ffprobe or not os.path.exists(video_path):
        return []
    try:
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index,codec_name:stream_tags=language,title",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, timeout=30, **_run_kwargs())
        if result.returncode != 0:
            return []
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            return []
        tracks = []
        for subtitle_index, stream in enumerate(streams):
            if not isinstance(stream, dict):
                continue
            tags = stream.get("tags") or {}
            tracks.append({
                "subtitle_index": subtitle_index,
                "index": stream.get("index"),
                "codec_name": stream.get("codec_name", ""),
                "language": tags.get("language", ""),
                "title": tags.get("title", ""),
            })
        return tracks
    except Exception:
        return []


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


# Sentinela de ``audio_stream_index``: -1 significa conservar TODAS las pistas
# de audio del contenedor (MP4/MKV/MOV soportan varias pistas de audio).
KEEP_ALL_AUDIO = -1


def parse_audio_stream_index(value):
    """Parsea el índice de pista de audio desde los ajustes.

    Devuelve:
        - ``KEEP_ALL_AUDIO`` (-1) si se pide conservar todas las pistas
        - un entero >= 0 para una pista concreta
        - 0 como fallback si el valor es inválido o ausente
    """
    if value is None:
        return 0
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return 0
    return idx if idx >= 0 else KEEP_ALL_AUDIO


def keeps_all_audio(value):
    """¿Esta selección pide conservar TODAS las pistas de audio?

    La respuesta tiene que ser la misma en quien mide y en quien aplica el
    filtro: si uno cuenta las pistas y el otro no, el boost se aplica a
    pistas que nadie midió.
    """
    return parse_audio_stream_index(value) == KEEP_ALL_AUDIO


def build_args(input_path, output_path, settings, loudnorm_meas=None):
    """Construye los argumentos de ffmpeg a partir de los ajustes.

    Audio:
        - noaudio      → -an (sin audio)
        - copy sin boost → -c:a copy (no re-codifica)
        - aac / mp3 / copy + boost → re-codifica con normalización EBU R128

    El boost tiene DOS modos, y el segundo altera al primero (no se apila):

      * **Casilla sola** (``audio_fx`` vacío): two-pass ``loudnorm`` lineal,
        exactamente el comportamiento histórico.
      * **Casilla + desplegable con algo activo**: la normalización se integra
        en ``audio_fx.build_effects_chain()``, que coloca el nivel donde el
        compresor trabaja, comprime/ecualiza, vuelve a normalizar y limita.

    Si ``loudnorm_meas`` es un dict con input_i/input_tp/input_lra/input_thresh/
    target_offset (devuelto por ``src.ffmpeg.loudnorm.measure_loudness``), se
    aplica el modo lineal de dos pasadas para precisión broadcast (-0.5 dB de
    desviación típico vs ±3 dB del single-pass).
    """
    # Import perezoso: audio_fx no importa runner, así que no hay ciclo, pero
    # mantiene el arranque de la app ligero.
    from src.ffmpeg import audio_fx

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

    # Relación de aspecto y resolución.
    # ``aspect_box`` (out_w, out_h, img_w, img_h) lo calcula el pipeline, que
    # es quien conoce el SAR del origen: la imagen se escala a su tamaño
    # visible, se centra con barras negras en la caja del ratio pedido y queda
    # con píxeles cuadrados. Sin ratio forzado (``aspect`` = "source", que es
    # el defecto) el vídeo sale con la proporción de siempre, y entonces la
    # resolución elegida sigue mandando: ``-2`` da la paridad y el SAR lo
    # conserva FFmpeg, con lo que el display aspect ratio no se toca.
    aspect_box = settings.get("aspect_box")
    if aspect_box:
        out_w, out_h, img_w, img_h = (int(v) for v in aspect_box)
        args.extend([
            "-vf",
            f"scale={img_w}:{img_h},"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1",
        ])
    else:
        resolution = settings.get("resolution")
        if resolution and resolution != "original":
            args.extend(["-vf", f"scale=-2:{resolution}"])

    # Audio con soporte de audio boost (single-pass o two-pass lineal)
    audio = settings.get("audio", "copy")
    audio_boost = bool(settings.get("audio_boost", False))
    audio_stream_index = parse_audio_stream_index(settings.get("audio_stream_index", 0))

    # Explicit maps are important for MKV files with several audio tracks:
    # FFmpeg's implicit stream selection otherwise silently picks the first
    # one, regardless of the track chosen in the UI. ``-1`` conserva TODAS
    # las pistas (``-map 0:a?``).
    args.extend(["-map", "0:v:0"])
    # Con varias pistas el boost va POR PISTA. Un único ``-af`` con la medición
    # de la primera se aplicaba a todas y dejaba a las demás con la ganancia
    # calculada para otra pista; con ``-filter_complex`` cada pista lleva la
    # suya. Sin mediciones por pista (una sola pista, o medición que falló)
    # esto devuelve [] y manda el ``-af`` y el mapeo de siempre.
    por_pista = []
    if audio_boost and audio != "noaudio":
        por_pista = build_per_stream_boost_args(settings, loudnorm_meas)
    if audio != "noaudio":
        if por_pista:
            args.extend(por_pista)
        elif audio_stream_index == KEEP_ALL_AUDIO:
            args.extend(["-map", "0:a?"])
        else:
            args.extend(["-map", f"0:a:{audio_stream_index}?"])

    if audio == "noaudio":
        args.extend(["-an"])
    elif audio == "copy" and not audio_boost:
        # Sin boost y Copy → preservamos audio original
        args.extend(["-c:a", "copy"])
    else:
        # Cualquier otro caso (incluido copy + boost) fuerza re-codificación para
        # poder aplicar la normalización.
        codec_a = "libmp3lame" if audio == "mp3" else "aac"
        args.extend(["-c:a", codec_a, "-b:a", "192k"])
        if audio_boost and not por_pista:
            af = build_boost_filter(settings, loudnorm_meas)
            if af:
                args.extend(["-af", af])

    # Formato contenedor
    fmt = settings.get("format", "mp4")
    if fmt == "wmv":
        args.extend(["-f", "asf"])
    elif fmt == "webm":
        args.extend(["-f", "webm"])

    args.append(output_path)
    return args


def build_join_args(join_graph, output_path, settings, loudnorm_meas=None):
    """Argumentos de FFmpeg para una timeline de Join videos.

    ``join_graph`` lo construye ``src.processing.join`` (misma forma que
    :class:`JoinGraph`) y ya trae las entradas ``-i``, el ``filter_complex``
    con trim/scale/pad/concat y, si procede, el ``loudnorm`` de la timeline.
    Aquí solo se añade la cola de codificación, reutilizando las mismas
    reglas de codec/bitrate/contenedor que el render de un solo vídeo.

    El audio SIEMPRE se recodifica cuando hay timeline: los clips pueden
    tener codecs y sample rates distintos, así que un ``-c:a copy`` no es
    posible aunque el usuario haya elegido "Copy" para un archivo único.
    """
    settings = dict(settings or {})
    args = ["-y"]
    args.extend(list(join_graph.inputs))
    args.extend([
        "-filter_complex", join_graph.filter_complex,
        "-map", join_graph.video_label,
    ])

    codec = settings.get("codec", "libx264")
    real_codec, codec_extra_args = resolve_codec(codec)
    args.extend(["-c:v", real_codec])
    args.extend(codec_extra_args)

    duration = float(getattr(join_graph, "duration", 0.0) or 0.0)
    if settings.get("mode") == "Target File Size (MB)" and settings.get("target_size_mb"):
        target_mb = float(settings["target_size_mb"])
        if duration > 0:
            total_bits = target_mb * 8 * 1024 * 1024
            video_bitrate = max(int((total_bits / duration) - 192000), 100000)
            args.extend(["-b:v", str(video_bitrate)])
        else:
            args.extend(["-b:v", f"{settings.get('bitrate', 1500)}k"])
    else:
        args.extend(["-b:v", f"{settings.get('bitrate', 1500)}k"])

    if "-pix_fmt" not in codec_extra_args:
        args.extend(["-pix_fmt", settings.get("pixel_format", "yuv420p")])

    # "Maximum GPU speed": NVENC en su preset más rápido. Sin la casilla se
    # mantiene el preset por defecto del encoder (calidad de siempre).
    if real_codec.endswith("_nvenc") and settings.get("gpu_fast"):
        args.extend(["-preset", "p1", "-tune", "ull", "-rc", "vbr", "-multipass", "1"])

    audio = settings.get("audio", "copy")
    if audio == "noaudio" or not getattr(join_graph, "has_audio", False):
        args.append("-an")
    else:
        args.extend(["-map", join_graph.audio_label])
        container = str(settings.get("format", "mp4")).lower()
        if container == "webm":
            encoder_a = "libopus"
        elif audio == "mp3":
            encoder_a = "libmp3lame"
        else:
            encoder_a = "aac"
        args.extend(["-c:a", encoder_a, "-b:a", "192k"])
        if encoder_a == "libopus":
            args.extend(["-strict", "-2"])

    fmt = settings.get("format", "mp4")
    if fmt == "wmv":
        args.extend(["-f", "asf"])
    elif fmt == "webm":
        args.extend(["-f", "webm"])

    args.append(output_path)
    return args


def build_boost_filter(settings, measured=None):
    """Filtro(s) del Audio Boost según lo que haya configurado el usuario.

    Es el ÚNICO punto donde se decide el modo del boost, para que la ruta
    estándar, la ruta IA y la previsualización se comporten igual:

      * ``measured=None`` (o una medicion que no se pueda escribir en el
        filtro) -> ``loudnorm`` de UNA pasada, que se automide pista a pista.
        Ese ``None`` es lo que significa el hueco de una lista por pista:
        "esta pista no se pudo medir de dos pasadas", no "sin audio".
      * Sin efectos activos  -> ``loudnorm`` lineal two-pass (histórico).
      * Con efectos activos  -> cadena de ``audio_fx`` (la que altera el boost).

    Devuelve una cadena de filtros lista para ``-af`` (vacía si no hay nada
    que aplicar).
    """
    from src.ffmpeg import audio_fx

    effects = settings.get("audio_fx")
    if not audio_fx.effects_are_active(effects):
        return _build_loudnorm_filter(measured)

    measured_i = None
    if isinstance(measured, dict):
        try:
            measured_i = float(measured.get(LOUDNORM_CLAVE_LOUDNESS))
        except (TypeError, ValueError):
            measured_i = None
    chain = audio_fx.build_effects_chain(
        effects,
        measured_i=measured_i,
        target_lufs=settings.get("audio_target_lufs"),
    )
    return ",".join(chain)


def build_boost_filters_per_track(settings, mediciones):
    """Un filtro de boost por pista, en el orden de las pistas.

    Es la generalización de ``build_boost_filter`` a varias mediciones: cada
    pista se normaliza con SU loudness, no con el de la primera. La cadena
    sale de ``build_boost_filter`` para que los dos modos del boost (loudnorm
    lineal y cadena de ``audio_fx``) se comporten igual por pista.

    Cada elemento de ``mediciones`` es la medicion de esa pista, y un ``None``
    significa "esa pista no se pudo medir de dos pasadas" (silencio digital,
    valores fuera de rango): esa pista recibe el loudnorm de una pasada y las
    demas conservan la suya. La lista siempre trae un elemento por pista,
    aunque sea ``None``, porque el indice es el numero de pista.
    """
    if not isinstance(mediciones, (list, tuple)) or not mediciones:
        return []
    return [build_boost_filter(settings, m) for m in mediciones]


def build_per_stream_boost_args(settings, mediciones, audio_input="0"):
    """``-filter_complex`` + mapas para aplicar un filtro distinto a cada pista.

    Devuelve la lista de argumentos para ``args.extend(...)``, o ``[]`` si no
    hay una medición por pista — entonces el llamante usa su ``-af`` de
    siempre, que es lo que quiere un archivo con una sola pista.

    ``audio_input`` es el número de entrada tal como lo escribe FFmpeg: la
    ruta estándar codifica el audio junto al vídeo (``0``) y el mux de la ruta
    IA lo toma de la segunda entrada (``1``).

    También se lleva los metadatos de cada pista (idioma, título) al stream
    de salida correspondiente: un stream filtrado no los hereda, y sin ellos
    las pistas conservadas salen como dos "Track 1" idénticas. El índice es el
    mismo que el del ``-map`` porque es el mismo orden: la pista i de salida
    sale de la pista i de entrada. Si los dos números no coincidieran, el
    grafo de filtros ya habria fallado antes con un error claro.
    """
    filtros = build_boost_filters_per_track(settings, mediciones)
    if not filtros or not all(filtros):
        return []
    etiquetas = ["[a%d]" % i for i in range(len(filtros))]
    cadena = ";".join(
        "[%s:a:%d]%s%s" % (audio_input, i, filtro, etiqueta)
        for i, (filtro, etiqueta) in enumerate(zip(filtros, etiquetas))
    )
    args = ["-filter_complex", cadena]
    for indice, etiqueta in enumerate(etiquetas):
        args.extend(["-map", etiqueta])
        args.extend(["-map_metadata:s:a:%d" % indice,
                     "%s:s:a:%d" % (audio_input, indice)])
    return args


# ---------------------------------------------------------------------------
# loudnorm: qué se le puede pasar, en qué rango y cómo se escribe
# ---------------------------------------------------------------------------
# Una fila por valor que el boost puede llevar al filtro. De esta sola tabla
# salen las dos cosas que antes vivian en sitios distintos: los parametros que
# se escriben en el ``loudnorm=`` y los rangos que ffmpeg acepta para ellos
# (comprobados contra el binario de la casa; fuera de rango, ffmpeg aborta el
# comando entero). Anadir un valor al boost es anadir UNA fila aqui, y ni el
# filtro ni el validador se pueden quedar atras el uno del otro.
#
#     clave de la medicion -> (parametro del filtro, rango, valor por defecto)
#
# Los rangos son el intervalo que el binario de la casa acepta de verdad, medido
# parametro a parametro (no de la documentacion, que no coincide) y sujeto a un
# test que lo vuelve a comprobar contra ffmpeg. Un rango MAS ESTRECHO que el de
# ffmpeg es un bug tan serio como uno mas ancho: hace que una pista que se puede
# medir bien se considere no medible y baje a una sola pasada, cambiando su nivel
# sin que nadie se entere.
#
# Un defecto ``None`` marca la clave como obligatoria: son las cuatro que
# hacen falta para que el modo de dos pasadas tenga sentido. Las demas son
# opcionales, y su defecto es lo que se escribe cuando no vienen.
LOUDNORM_MEDIDAS = {
    "input_i": ("measured_I", (-99.0, 0.0), None),
    "input_tp": ("measured_TP", (-99.0, 99.0), None),
    "input_lra": ("measured_LRA", (0.0, 99.0), None),
    "input_thresh": ("measured_thresh", (-99.0, 0.0), None),
    "target_offset": ("offset", (-99.0, 99.0), 0.0),
}

# Las claves sin defecto son las obligatorias. Se deriva de la tabla, no se
# vuelve a escribir: asi no hay una segunda lista que mantener.
LOUDNORM_CLAVES_OBLIGATORIAS = tuple(
    clave for clave, (_param, _rango, defecto) in LOUDNORM_MEDIDAS.items()
    if defecto is None
)

# La clave que lleva el loudness medido. No va al filtro: la consume la cadena
# de efectos para calcular su ganancia, tanto en el render como en la vista
# previa. Se saca de la tabla por su parametro en vez de escribirse aqui, asi que
# un renombrado se propaga solo.
try:
    LOUDNORM_CLAVE_LOUDNESS = next(
        clave for clave, (parametro, _rango, _defecto) in LOUDNORM_MEDIDAS.items()
        if parametro == "measured_I"
    )
except StopIteration:
    # Sin valor por defecto a proposito: un respaldo a otra clave perderia la
    # ganancia en silencio, que es justo lo que esto evita. Falla al importar,
    # y con una frase que diga que falta y donde se declara.
    raise RuntimeError(
        "LOUDNORM_MEDIDAS (en src/ffmpeg/runner.py, justo encima) no declara "
        "ninguna clave cuyo parametro de filtro sea 'measured_I'. De esa clave "
        "depende la ganancia de la cadena de efectos del render y de la vista "
        "previa: anade la fila del loudness medido a la tabla."
    ) from None

# El objetivo del boost, y el loudnorm de una pasada cuando no hay nada
# medido: es lo que se aplica a una pista que no se pudo medir, y por eso
# tiene que ser el mismo objetivo que el de dos pasadas.
LOUDNORM_OBJETIVO = "I=-16:TP=-1.5:LRA=11"
LOUDNORM_SIN_MEDIR = "loudnorm=" + LOUDNORM_OBJETIVO


def measurement_is_usable(medicion):
    """¿Esta medición se puede escribir en el filtro loudnorm sin que ffmpeg la rechace?

    Una pista en silencio digital mide ``input_i = -inf`` y una muy floja puede
    medir true peaks por debajo de -99: en ambos casos el valor no se puede
    escribir en el filtro, y ffmpeg no deja pasar solo esa pista, tumba el
    comando entero. Sicha medición no es "el boost falló", es "esta pista no
    se puede medir de dos pasadas", y eso tiene una respuesta que ya existe:
    ``LOUDNORM_SIN_MEDIR``, que se automide pista a pista.
    """
    if not isinstance(medicion, dict):
        return False
    for clave, (_param, (bajo, alto), defecto) in LOUDNORM_MEDIDAS.items():
        try:
            valor = float(medicion.get(clave, defecto))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(valor) or not bajo <= valor <= alto:
            return False
    return True


def _build_loudnorm_filter(measured):
    """Construye el filtro ``-af loudnorm=...``.

    Modo two-pass (linear): usa los valores medidos del input para una
    normalizacion EBU R128 de precision broadcast. Sin medicion utilizable —no
    hay, o no trae las claves obligatorias— cae al modo single-pass con los
    valores por defecto, que es ``LOUDNORM_SIN_MEDIR``.

    Los nombres de parametro y el orden salen de ``LOUDNORM_MEDIDAS``, la
    misma tabla que usa ``measurement_is_usable`` para decidir si una medicion
    entra: quien valida y quien escribe no pueden separarse.
    """
    if not isinstance(measured, dict):
        return LOUDNORM_SIN_MEDIR
    try:
        partes = ["loudnorm=" + LOUDNORM_OBJETIVO]
        for clave, (parametro, _rango, defecto) in LOUDNORM_MEDIDAS.items():
            partes.append(
                "%s=%.4f" % (parametro, float(measured.get(clave, defecto))))
        partes.append("linear=true")
    except (ValueError, TypeError):
        return LOUDNORM_SIN_MEDIR
    return ":".join(partes)


