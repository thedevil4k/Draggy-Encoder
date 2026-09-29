from __future__ import annotations

import json
import logging
import os
import queue as queue_module
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime

import numpy as np

from src.ai.registry import get_engine_meta
from src.config import get_codec_entry, get_logs_dir, resolve_codec
from src.ffmpeg.loudnorm import measure_loudness, measure_loudness_all_tracks
from src.ffmpeg.runner import (
    KEEP_ALL_AUDIO,
    _parse_time_to_seconds,
    build_args,
    build_boost_filter,
    build_per_stream_boost_args,
    extract_frame,
    get_video_info,
    health_check_ffmpeg,
    keeps_all_audio,
    parse_audio_stream_index,
    run_ffmpeg,
)
from src.ffmpeg.process_supervisor import (
    ProcessSupervisor,
    process_group_popen_kwargs,
)
from src.utils.timing import StageTimer


logger = logging.getLogger(__name__)


def _emit_phase(phase_cb, ev, phase_id, label="", pct=0, msg=""):
    """Emit a structured phase event to the (optional) phase callback.

    ``ev`` ∈ {"start", "progress", "end", "failed"}. The GUI translates
    these into per-phase rows (tasks panel). Never raises: a broken
    watcher must not break the render.
    """
    if not phase_cb:
        return
    try:
        phase_cb({
            "ev": ev,
            "id": phase_id,
            "label": label,
            "pct": int(pct),
            "msg": msg,
        })
    except Exception as exc:
        logger.warning("phase_cb error: %s", exc)


def _emit_metrics(metrics_cb, **fields):
    """Publica cifras de throughput en un dict plano. Nunca lanza.

    El canal de métricas es **plano** a propósito, y por eso se parece al que
    ya usa ``run_ffmpeg``: quien lo consume (el desplegable de la bandeja) lo
    mezcla tal cual llega, sin tener que saber de qué render salió cada
    número. Envolverlo en ``{"metric": ..., "value": ...}`` —que es lo que
    necesita una fase, porque tiene nombre e historia— aquí solo añadía una
    forma más que el consumidor tenía que adivinar.
    """
    if not metrics_cb:
        return
    try:
        metrics_cb(dict(fields))
    except Exception as exc:
        logger.debug("metrics_cb error: %s", exc)


def _get_duration_seconds(ffmpeg_path, video_path):
    info = get_video_info(ffmpeg_path, video_path)
    try:
        return float(info.get("duration", 0))
    except Exception:
        return 0


def validate_render_preflight(ffmpeg_path, input_path, output_path=None,
                              settings=None):
    """Return actionable errors that must be fixed before a render starts."""
    settings = dict(settings or {})
    errors = []
    if not ffmpeg_path or not os.path.isfile(ffmpeg_path):
        errors.append(f"FFmpeg binary is missing: {ffmpeg_path}")
    else:
        ok, _version, detail = health_check_ffmpeg(ffmpeg_path, timeout=10)
        if not ok:
            errors.append(f"FFmpeg health check failed: {detail or 'unknown error'}")

    if not input_path or not os.path.isfile(input_path):
        errors.append(f"Input video is missing: {input_path}")
    else:
        try:
            if os.path.getsize(input_path) <= 0:
                errors.append("Input video is empty.")
        except OSError as exc:
            errors.append(f"Input video cannot be inspected: {exc}")

    output_dir = None
    if output_path:
        output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        if os.path.abspath(input_path or "") == os.path.abspath(output_path):
            errors.append("Input and output paths must be different.")
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                errors.append(f"Output directory is not writable: {output_dir}")
        except OSError as exc:
            errors.append(f"Output directory is unavailable: {exc}")

    if input_path and os.path.isfile(input_path):
        info = get_video_info(ffmpeg_path, input_path)
        try:
            width, height = int(info.get("width", 0)), int(info.get("height", 0))
        except (TypeError, ValueError):
            width = height = 0
        if width <= 0 or height <= 0:
            errors.append("FFprobe could not detect a valid video stream.")
        try:
            duration = float(info.get("duration", 0) or 0)
        except (TypeError, ValueError):
            duration = 0
        if duration <= 0:
            errors.append("FFprobe could not detect a valid video duration.")

        start = _parse_time_to_seconds(settings.get("start_time"))
        end = _parse_time_to_seconds(settings.get("end_time"))
        if start is not None and start < 0:
            errors.append("Trim start cannot be negative.")
        if end is not None and end <= 0:
            errors.append("Trim end must be greater than zero.")
        if start is not None and end is not None and end <= start:
            errors.append("Trim end must be greater than trim start.")
        if duration > 0 and start is not None and start >= duration:
            errors.append("Trim start is beyond the input duration.")

        if output_dir:
            try:
                # Conservative reserve for staging, mux and temporary audio.
                required = max(64 * 1024 * 1024, os.path.getsize(input_path) * 2)
                free = shutil.disk_usage(output_dir).free
                if free < required:
                    errors.append(
                        f"Not enough free disk space (need about {required} bytes, "
                        f"available {free})."
                    )
            except OSError:
                pass

    if not str(settings.get("codec", "")).strip():
        errors.append("No video codec is selected.")
    if settings.get("format") not in (None, "mp4", "mkv", "avi", "wmv", "mov", "webm"):
        errors.append(f"Unsupported output format: {settings.get('format')}")
    return errors


def validate_render_output(ffmpeg_path, output_path, expected_duration=None):
    """Validate a staged render before it can be published."""
    if not output_path or not os.path.isfile(output_path):
        return False, "Rendered output is missing."
    try:
        if os.path.getsize(output_path) <= 0:
            return False, "Rendered output is empty."
    except OSError as exc:
        return False, f"Rendered output cannot be inspected: {exc}"
    info = get_video_info(ffmpeg_path, output_path)
    try:
        width, height = int(info.get("width", 0)), int(info.get("height", 0))
        duration = float(info.get("duration", 0) or 0)
    except (TypeError, ValueError):
        return False, "FFprobe returned invalid output metadata."
    if width <= 0 or height <= 0 or duration <= 0:
        return False, "Rendered output has no valid video stream or duration."
    if expected_duration and duration < max(0.05, expected_duration * 0.25):
        return False, "Rendered output is unexpectedly shorter than the requested clip."
    return True, ""


def _make_temp_output_path(output_path: str) -> str:
    """Create a same-directory temporary output name and reserve no handle.

    Keeping the temporary file beside the destination makes ``os.replace``
    atomic on the same filesystem and prevents a failed FFmpeg run from
    exposing a truncated file under the user's final name.
    """
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(directory, exist_ok=True)
    ext = os.path.splitext(output_path)[1] or ".tmp"
    fd, path = tempfile.mkstemp(prefix=".draggy-render-", suffix=ext, dir=directory)
    os.close(fd)
    try:
        os.remove(path)
    except OSError:
        pass
    return path


def cleanup_stale_staging(older_than_seconds=3600):
    """Remove abandoned render staging files left by a previous crash.

    Staging files are created beside their destination (``.draggy-render-*``
    / ``.drag-audio-*``); if the app died before ``os.replace`` they would
    otherwise accumulate forever. Only files older than ``older_than_seconds``
    are touched so a concurrent render in this same process is never harmed.
    """
    import glob as _glob
    import time as _time
    now = _time.time()
    removed = 0
    for pattern in (".draggy-render-*", ".drag-audio-*"):
        try:
            for path in _glob.glob(os.path.join(tempfile.gettempdir(), pattern)):
                try:
                    if now - os.path.getmtime(path) >= older_than_seconds:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass
        except Exception:
            pass
    return removed


def _cleanup_path(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not remove temporary render file: %s", path)


def _needs_audio_reencode(settings):
    """Indica si vamos a re-codificar audio (y por tanto tiene sentido medir)."""
    audio = settings.get("audio", "copy")
    audio_boost = bool(settings.get("audio_boost", False))
    if audio == "noaudio":
        return False
    if audio == "copy" and not audio_boost:
        return False
    return True


# Techo para la pasada de medición loudnorm. La medición decodifica SOLO
# audio (~65x tiempo real): un vídeo de 8 min tarda ~7 s, pero una película
# de 2 h tardaría ~2 min en serie antes de que arranque el encode. Los clips
# más largos que el techo se miden sobre una VENTANA CENTRAL representativa
# (evita intros/outros atípicos); el resto se mide exacto. Ajustable con
# ``settings["measure_cap_seconds"]``.
_MEASURE_CAP_SECONDS = 600.0  # 10 min


def _measure_bounds(start_time, end_time, duration_s, cap=_MEASURE_CAP_SECONDS):
    """Devuelve ``(ms_start, ms_end)`` en segundos para la pasada de medición.

    Si la ventana a medir cabe entera en ``cap`` se devuelve tal cual
    (medición exacta, sin pérdida de precisión). Si es más larga, se toma
    una ventana central de ``cap`` segundos para acotar el tiempo de análisis
    manteniendo una muestra representativa.
    """
    start = _parse_time_to_seconds(start_time)
    end = _parse_time_to_seconds(end_time)
    s = 0.0 if start is None else float(start)
    e = float(duration_s or 0) if end is None else float(end)
    if e <= s:
        return start_time, end_time
    window = e - s
    if window <= cap:
        return start_time, end_time
    mid = s + window / 2.0
    ms = max(s, mid - cap / 2.0)
    me = min(e, ms + cap)
    return ms, me


def _coerce_output_for_codec(output_path, settings):
    """Ajusta contenedor y extensión si el codec elegido no cabe en MP4.

    ProRes y DNxHR (y cualquier codec ``mp4_incompatible``) no se pueden
    muxear en MP4: si el usuario eligió "mp4", cambiamos el contenedor al
    preferido por el codec (mov/mkv) y reescribimos la extensión del destino
    para que FFmpeg infiera el muxer correcto en vez de fallar en silencio.
    """
    entry = get_codec_entry(settings.get("codec"))
    if not (entry and entry.get("mp4_incompatible") and settings.get("format") == "mp4"):
        return output_path, settings
    container = entry.get("container", "mkv")
    settings = dict(settings)
    settings["format"] = container
    base, _ext = os.path.splitext(output_path)
    return base + "." + container, settings


def _measure_if_needed(ffmpeg_path, input_path, settings, status_callback=None,
                       start_time=None, end_time=None, abort_event=None,
                       supervisor: ProcessSupervisor | None = None,
                       per_track=True):
    """Mide loudness solo si audio_boost=True y vamos a re-codificar audio.

    Devuelve el dict medido (modulo de loudnorm) o None si no aplica / falla.
    Para el preview se puede pasar ``start_time`` / ``end_time`` para medir
    únicamente el trozo que se va a codificar. Para clips muy largos la
    medición se acota a una ventana central (``_measure_bounds``) para no
    serializar minutos de análisis de audio antes del encode.

    Si el usuario conserva TODAS las pistas, devuelve una lista con una
    medición por pista: el boost de dos pasadas hornea los valores medidos en
    el filtro, así que medir solo la primera y aplicar ese filtro a todas deja
    a las demás con la ganancia de otra pista. Con una sola pista el resultado
    es el mismo dict de siempre, y el comando no cambia ni un byte.

    ``per_track=False`` es para quien solo muestra u oye una pista (el
    preview): ahí una medición por pista sería trabajo tirado.
    """
    if not settings.get("audio_boost"):
        return None
    if not _needs_audio_reencode(settings):
        return None
    cap = settings.get("measure_cap_seconds")
    try:
        cap = float(cap) if cap is not None else _MEASURE_CAP_SECONDS
    except (TypeError, ValueError):
        cap = _MEASURE_CAP_SECONDS
    duration_s = settings.get("duration_seconds")
    if duration_s is None:
        duration_s = _get_duration_seconds(ffmpeg_path, input_path) or 0
    ms_start, ms_end = _measure_bounds(start_time, end_time, duration_s, cap)
    indice = settings.get("audio_stream_index", 0)
    if status_callback:
        status_callback("Analyzing audio levels (first pass)...")
    try:
        if per_track and keeps_all_audio(indice):
            return measure_loudness_all_tracks(
                ffmpeg_path, input_path,
                start_time=ms_start, end_time=ms_end,
                status_callback=status_callback,
                abort_event=abort_event,
                supervisor=supervisor,
            )
        return measure_loudness(
            ffmpeg_path, input_path,
            start_time=ms_start, end_time=ms_end,
            status_callback=status_callback,
            abort_event=abort_event,
            supervisor=supervisor,
            audio_stream_index=indice,
        )
    except Exception:
        return None


def compress_video(ffmpeg_path, input_path, output_path, settings,
                   progress_callback=None, status_callback=None, abort_event=None,
                   ai_engines=None, on_process=None, phase_cb=None,
                   supervisor: ProcessSupervisor | None = None,
                   metrics_cb=None):
    """Comprime un vídeo completo. Audio boost usa two-pass loudnorm.

    ``ai_engines`` is an optional mapping ``{"Upscale": "engine_id",
    "Interpolate": "engine_id", "Colorize": "engine_id"}``. The GUI
    resolves the mapping into loaded engines for the AI path; this function
    keeps the selection in the status/log for non-AI compression callers.
    Pass ``None`` (default) for projects that don't use the AI chain.

    ``on_process`` is an optional callable invoked once with the
    ``subprocess.Popen`` from ``run_ffmpeg``. The GUI uses it to keep
    a reference to the ffmpeg child so it can ``kill()`` directly
    when the abort_event path doesn't reach run_ffmpeg in time (see
    ``_abort_running_worker`` in ``src/ui/main_window.py``).
    """
    settings = dict(settings)
    output_path, settings = _coerce_output_for_codec(output_path, settings)
    preflight_errors = validate_render_preflight(
        ffmpeg_path, input_path, output_path, settings,
    )
    if preflight_errors:
        message = "Preflight failed: " + " ".join(preflight_errors)
        if status_callback:
            status_callback(message)
        logger.error(message)
        return False
    settings["duration_seconds"] = _get_duration_seconds(ffmpeg_path, input_path)

    if ai_engines:
        try:
            lines = []
            for cat, info in ai_engines.items():
                # WorkflowWidget.get_selected_engines() returns values as
                # ``{"engine_id": "...", "model_path": "...", "scale": N}``
                # dicts. Older call sites passed plain engine_id strings.
                # Handle both shapes so we don't blow up with
                # "unhashable type: 'dict'" when CATALOG_BY_ID[eid] is
                # attempted on a dict.
                if isinstance(info, dict):
                    eid = info.get("engine_id", "") or ""
                else:
                    eid = info or ""
                if eid:
                    meta = get_engine_meta(eid) or {}
                    label = meta.get("display_name", eid)
                    lines.append(f"{cat}={label}")
                else:
                    lines.append(f"{cat}=[auto]")
            if lines and status_callback:
                status_callback("AI engines selected: " + ", ".join(lines))
            logger.info("AI engines selected for this run: %s", ai_engines)
        except Exception as exc:
            logger.warning("ai_engines plumbing failed: %s", exc)

    # El audio boost en la ruta estándar es two-pass loudnorm. La fase
    # "measure" cubre SOLO la medición (pasada 1) y termina en cuanto se
    # obtienen los valores; la pasada 2 comprime el vídeo y aplica el filtro
    # loudnorm EN EL MISMO encode, así que "Compressing video" y "Boosting
    # audio" avanzan juntas (es exactamente lo que hace ffmpeg). Antes la
    # medición iba metida dentro de la fase de audio y el usuario veía la
    # barra "en audio" hasta el final de la compresión.
    audio_boost_active = (
        bool(settings.get("audio_boost")) and _needs_audio_reencode(settings)
    )
    if audio_boost_active:
        _emit_phase(phase_cb, "start", "measure", label="Analyzing audio levels", pct=0)
        # "start" deja la fase en PENDING; un progress inicial la marca
        # RUNNING mientras dura la pasada de medición.
        _emit_phase(phase_cb, "progress", "measure", pct=0)

    _emit_phase(phase_cb, "start", "compress", label="Compressing video", pct=0)
    if audio_boost_active:
        _emit_phase(phase_cb, "start", "audio", label="Boosting audio", pct=0)

    loudnorm_meas = _measure_if_needed(
        ffmpeg_path, input_path, settings, status_callback=status_callback,
        start_time=settings.get("start_time"),
        end_time=settings.get("end_time"),
        abort_event=abort_event,
        supervisor=supervisor,
    )

    if audio_boost_active:
        _emit_phase(phase_cb, "progress", "measure", pct=100)
        _emit_phase(phase_cb, "end", "measure")

    temp_output = _make_temp_output_path(output_path)
    try:
        args = build_args(input_path, temp_output, settings, loudnorm_meas=loudnorm_meas)

        # Progreso real del encode: avanza a la vez la barra principal y las
        # filas del panel de tareas. Con audio boost, el filtro loudnorm se
        # aplica en este mismo pase, así que "Compressing video" y
        # "Boosting audio" avanzan juntos (es exactamente lo que hace
        # ffmpeg: comprimir vídeo + boostear audio a la vez).
        def _encode_progress(pct):
            if progress_callback:
                progress_callback(pct)
            _emit_phase(phase_cb, "progress", "compress", pct=pct)
            if audio_boost_active:
                _emit_phase(phase_cb, "progress", "audio", pct=pct)

        if status_callback:
            if audio_boost_active:
                status_callback("Compressing video + boosting audio...")
            else:
                status_callback("Compressing video...")

        # El kwarg de metricas solo se pasa si hay alguien escuchando. Pasarlo
        # siempre —aunque valiese None— cambiaria la forma de la llamada para
        # todos los llamantes de siempre, y hay dobles de test que comprueban
        # la firma exacta. Asi, quien no quiere metricas ve una llamada
        # identica a la de antes, y no se nota que esta opcion existe.
        extra = {"metrics_callback": metrics_cb} if metrics_cb else {}
        result = run_ffmpeg(
            ffmpeg_path,
            args,
            progress_callback=_encode_progress,
            abort_event=abort_event,
            on_process=on_process,
            supervisor=supervisor,
            **extra,
        )
        valid, detail = validate_render_output(
            ffmpeg_path,
            temp_output,
            expected_duration=settings.get("duration_seconds"),
        ) if result else (False, "FFmpeg returned a failure code.")
        if valid:
            os.replace(temp_output, output_path)
            _emit_phase(phase_cb, "progress", "compress", pct=100)
            _emit_phase(phase_cb, "end", "compress")
            if audio_boost_active:
                _emit_phase(phase_cb, "progress", "audio", pct=100)
                _emit_phase(phase_cb, "end", "audio")
            return True
        logger.error("Standard render validation failed: %s", detail)
        _emit_phase(phase_cb, "failed", "compress", msg=detail)
        if audio_boost_active:
            _emit_phase(phase_cb, "failed", "audio", msg="Video encode failed")
        return False
    finally:
        _cleanup_path(temp_output)


def generate_preview(ffmpeg_path, input_path, settings, progress_callback=None,
                     status_callback=None, abort_event=None, ai_engines=None,
                     inference_engines=None, on_process=None, on_processes=None,
                     phase_cb=None, supervisor: ProcessSupervisor | None = None,
                     preview_time=None, metrics_cb=None):
    """Genera un preview: codifica un clip corto y extrae un frame.

    Devuelve una tupla ``(original_bytes, compressed_bytes, artifacts)`` con
    los bytes de dos imágenes JPEG y un diccionario de artefactos persistentes
    (clip comprimido, ruta de origen, ffmpeg, tiempos) que el PreviewDialog
    puede usar para re-extraer frames en otros instantes sin repetir la
    codificación. El llamante DEBE eliminar ``artifacts["dir"]`` cuando ya
    no necesite los artefactos.

    ``preview_time`` en segundos absolutos: centra el clip de 5 s en ese
    instante y extrae exactamente ESE frame. Si es ``None``, comportamiento
    anterior: clip al 20 % del vídeo y frame aleatorio reproducible.

    ``ai_engines`` carries the same shape as in ``compress_video`` and is
    surfaced in the status line. When ``inference_engines`` is provided,
    the short clip is rendered through the real AI chain before the frame is
    extracted; without it the function keeps the lightweight FFmpeg-only
    preview path.
    """
    settings = dict(settings)
    preflight_errors = validate_render_preflight(
        ffmpeg_path, input_path, None, settings,
    )
    if preflight_errors:
        raise RuntimeError("Preflight failed: " + " ".join(preflight_errors))
    duration = _get_duration_seconds(ffmpeg_path, input_path)
    settings["duration_seconds"] = duration
    _emit_phase(phase_cb, "start", "preview", label="Preview clip", pct=0)

    if ai_engines:
        try:
            lines = []
            for cat, info in ai_engines.items():
                # WorkflowWidget.get_selected_engines() returns values as
                # dicts now; handle both shapes for compat with older
                # callers that passed plain engine_id strings.
                if isinstance(info, dict):
                    eid = info.get("engine_id", "") or ""
                else:
                    eid = info or ""
                if eid:
                    meta = get_engine_meta(eid) or {}
                    label = meta.get("display_name", eid)
                    lines.append(f"{cat}={label}")
                else:
                    lines.append(f"{cat}=[auto]")
            if lines and status_callback:
                status_callback("AI engines selected (preview): " + ", ".join(lines))
        except Exception as exc:
            logger.warning("ai_engines plumbing failed (preview): %s", exc)

    # Instante a comparar. Si el usuario pidió ``preview_time`` (segundos
    # absolutos) centramos el clip ahí y extraemos ESE frame; si no, clip de
    # 5 s desde el 20 % del vídeo + frame aleatorio reproducible. El parámetro
    # tiene prioridad; si llega por settings (el camino normal desde la UI)
    # se lee de ahí también.
    if preview_time is None:
        preview_time = settings.get("preview_time")
    try:
        t = float(preview_time) if preview_time is not None else None
    except (TypeError, ValueError):
        t = None

    if t is not None:
        # Clip de 5 s centrado en ``t``, recortado a [0, duration].
        start = max(0.0, t - 2.5)
        end = min(duration, t + 2.5) if duration > 0 else t + 2.5
        if end <= start:
            end = start + 5.0
            if duration > 0:
                end = min(end, duration)
        clip_length = max(0.1, end - start)
        # Frame exacto pedido (offset dentro del clip), nunca más allá del
        # final real del clip.
        frame_offset = min(max(0.0, t - start), clip_length - 0.1)
    else:
        start = max(0, duration * 0.2) if duration > 0 else 0
        end = start + min(5, max(1, duration - start))
        # Asegurar que no nos pasamos del final del vídeo.
        if duration > 0:
            end = min(end, duration)
        clip_length = max(0.1, end - start)

        # Frame aleatorio dentro del clip, evitando los extremos. Semilla
        # basada en el path para reproducibilidad entre previews.
        rng = random.Random(input_path)
        frame_offset = rng.uniform(0.1, max(0.2, clip_length - 0.1))

    # Medimos loudness SOBRE EL MISMO CLIP que se va a codificar. El preview
    # solo enseña una imagen (y el fragmento de audio lo mide su propio
    # worker), asi que medir pista por pista aqui seria tiempo tirado.
    loudnorm_meas = _measure_if_needed(
        ffmpeg_path, input_path, settings, status_callback=status_callback,
        start_time=start, end_time=end,
        abort_event=abort_event,
        supervisor=supervisor,
        per_track=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        preview_settings = dict(settings)
        # El preview sólo necesita imagen: silenciamos el audio para evitar
        # errores opacos del muxer (p.ej. un mp4 con un audio source vorbis/
        # ac3/flac que -c:a copy no puede reempaquetar). El screenshot del
        # usuario fallaba exactamente por esto.
        preview_settings["audio"] = "noaudio"
        preview_settings["audio_boost"] = False
        preview_settings["start_time"] = str(start)
        preview_settings["end_time"] = str(end)

        fmt = settings.get("format", "mp4")
        compressed_path = os.path.join(tmpdir, f"preview_compressed.{fmt}")
        if inference_engines:
            # Use the exact frame-by-frame pipeline used by Compress. The
            # preview settings already disable audio and constrain the input
            # to the short clip above, so no temporary audio track is needed.
            ok = compress_video_with_ai(
                ffmpeg_path,
                input_path,
                compressed_path,
                preview_settings,
                inference_engines=inference_engines,
                progress_callback=progress_callback,
                status_callback=status_callback,
                abort_event=abort_event,
                on_process=on_process,
                on_processes=on_processes,
                phase_cb=phase_cb,
                supervisor=supervisor,
                metrics_cb=metrics_cb,
            )
        else:
            args = build_args(input_path, compressed_path, preview_settings, loudnorm_meas=loudnorm_meas)
            logger.info("Preview encode args: %s", args)
            ok = run_ffmpeg(
                ffmpeg_path,
                args,
                progress_callback=progress_callback,
                abort_event=abort_event,
                on_process=on_process,
                supervisor=supervisor,
            )
        if not ok:
            raise RuntimeError(
                "FFmpeg failed to encode the preview clip. "
                "Check bin/logs/ffmpeg.log for the full FFmpeg output."
            )

        # Elegir un fotograma seguro dentro del clip generado.
        # Nunca pidamos un tiempo mayor que la duración real del clip.
        frame_time = min(frame_offset, clip_length - 0.1)
        compressed_frame = os.path.join(tmpdir, "compressed_frame.jpg")
        ok_comp = extract_frame(
            ffmpeg_path, compressed_path, frame_time, compressed_frame,
            abort_event=abort_event, supervisor=supervisor,
        )

        original_time = start + frame_time
        original_frame = os.path.join(tmpdir, "original_frame.jpg")
        ok_orig = extract_frame(
            ffmpeg_path, input_path, original_time, original_frame,
            abort_event=abort_event, supervisor=supervisor,
        )

        if not ok_comp:
            raise RuntimeError("Could not extract a frame from the compressed preview clip.")
        if not ok_orig:
            raise RuntimeError(f"Could not extract the original frame at t={original_time:.2f}s.")
        if not os.path.exists(compressed_frame) or not os.path.exists(original_frame):
            raise RuntimeError("Generated frame files are missing.")
        if os.path.getsize(compressed_frame) == 0 or os.path.getsize(original_frame) == 0:
            raise RuntimeError("Generated frame files are empty.")

        # Leer las imágenes como bytes; no creamos QPixmap aquí porque esta
        # función corre en un hilo secundario y QPixmap debe construirse en el
        # hilo principal de la GUI para evitar crashes silenciosos de Qt.
        with open(original_frame, "rb") as f:
            original_data = f.read()
        with open(compressed_frame, "rb") as f:
            compressed_data = f.read()
        if not original_data or not compressed_data:
            raise RuntimeError("Generated frame files could not be read.")
        # Validación básica del header JPEG.
        if not compressed_data.startswith(b"\xff\xd8") or not original_data.startswith(b"\xff\xd8"):
            raise RuntimeError("Generated frame files are not valid JPEG images.")

        # Persistir el clip comprimido en un directorio independiente
        # del tmpdir (que se elimina al salir de este bloque). El
        # PreviewDialog puede re-extraer frames en otros instantes sin
        # repetir la codificación.
        artifacts_dir = tempfile.mkdtemp(prefix="draggy-preview-artifacts-")
        persistent_clip = os.path.join(
            artifacts_dir, f"preview_clip.{fmt}",
        )
        shutil.copyfile(compressed_path, persistent_clip)
        artifacts = {
            "dir": artifacts_dir,
            "compressed_clip": persistent_clip,
            "clip_start": start,
            "clip_length": clip_length,
            "source": input_path,
            "ffmpeg": ffmpeg_path,
            "duration": duration,
            "time": original_time,
        }
        _emit_phase(phase_cb, "progress", "preview", pct=100)
        _emit_phase(phase_cb, "end", "preview")
        return original_data, compressed_data, artifacts


# ---------------------------------------------------------------------------
# AI frame-by-frame pipeline
# ---------------------------------------------------------------------------


def compress_video_with_ai(
    ffmpeg_path,
    input_path,
    output_path,
    settings,
    inference_engine=None,
    progress_callback=None,
    status_callback=None,
    abort_event=None,
    on_process=None,
    on_processes=None,
    inference_engines=None,
    phase_cb=None,
    supervisor: ProcessSupervisor | None = None,
    metrics_cb=None,
):
    """Decode -> AI chain -> encode in a single pass (frame-by-frame).

    ``inference_engine`` (legacy) is a single ``AIInferenceEngine`` instance.
    ``inference_engines`` is an ordered LIST of engines chained together:
    each decoded frame runs through every engine in order (Upscale →
    Denoise → Colorize → FaceRestore ...), so the user can stack multiple
    AI steps like video2x, plus interpolation at its chain position.

    Interpolation: at most one engine with ``_mode in ("interpolate",
    "both")`` may appear.  Frames are interpolated between consecutive
    chain outputs (after every engine that precedes the interp step),
    doubling the output frame rate.

    Chain stability: if an engine raises at runtime, it is skipped for the
    rest of the render (the frame passes through untouched) instead of
    killing the whole job; if the interpolation engine fails, frames are
    duplicated to preserve A/V cadence.

    Steps:
        1. Probe input video (resolution, fps, codec, frames).
        2. Launch FFmpeg decoder (rawvideo pipe, BGR24).
        3. Process the first frame through the chain to learn the final
           output resolution, then launch the FFmpeg encoder.
        4. For each frame: numpy -> chain -> pipe.
        5. Mux audio separately with ``-c:a copy``.

    ``on_process`` receives the encoder ``Popen`` object so the GUI can
    kill it directly on abort.
    """
    # Direct callers may not provide a supervisor (the GUI does). Always
    # create one here so decoder, encoder, audio and mux children still have
    # a common kill path when a model or pipe fails unexpectedly.
    supervisor = supervisor or ProcessSupervisor()
    output_path, settings = _coerce_output_for_codec(output_path, settings)
    preflight_errors = validate_render_preflight(
        ffmpeg_path, input_path, output_path, settings,
    )
    if preflight_errors:
        message = "Preflight failed: " + " ".join(preflight_errors)
        if status_callback:
            status_callback(message)
        logger.error(message)
        return False

    if inference_engines is not None:
        engines = [e for e in inference_engines if e is not None]
    elif inference_engine is not None:
        engines = [inference_engine]
    else:
        raise ValueError(
            "inference_engine or inference_engines is required for "
            "compress_video_with_ai"
        )
    if not engines:
        raise ValueError("compress_video_with_ai got an empty engine list")

    # Render into a staging file next to the final destination. This keeps
    # direct callers safe as well as the GUI worker: only a fully finalized
    # video is ever published under ``output_path``.
    final_output_path = output_path
    output_path = _make_temp_output_path(final_output_path)

    info = get_video_info(ffmpeg_path, input_path)
    src_w = int(info.get("width", 0))
    src_h = int(info.get("height", 0))
    # ffprobe devuelve el framerate en ``r_frame_rate`` (ej. "30000/1001"
    # ó "30/1"). Pasamos la cadena tal cual al encoder de FFmpeg (que acepta
    # la forma X/Y) y derivamos un float para matemáticas de progreso.
    raw_fps = info.get("r_frame_rate", "") or ""
    fps_unknown = not raw_fps or raw_fps.strip() in ("0/0", "0/1", "N/A")
    if fps_unknown:
        raw_fps = "30/1"
        if status_callback:
            status_callback(
                "Frame rate unknown — using 30 fps fallback for progress; "
                "the encoder will use the input stream rate."
            )
    fps_str = str(raw_fps)
    fps_value = _fps_string_to_float(raw_fps)
    duration = _get_duration_seconds(ffmpeg_path, input_path)
    if duration and fps_value:
        est_total = int(round(duration * fps_value))
    else:
        est_total = 0
    nb_frames_raw = info.get("nb_frames", "")
    try:
        nb_frames_int = int(nb_frames_raw) if nb_frames_raw else 0
    except (TypeError, ValueError):
        nb_frames_int = 0
    total_frames = nb_frames_int or est_total

    if src_w == 0 or src_h == 0:
        raise RuntimeError("Could not detect input video resolution.")

    if status_callback:
        status_callback("AI pipeline: probing video info...")
    _emit_phase(phase_cb, "progress", "probe", label="Analyzing video", pct=50)

    # Target scale: la cadena puede tener VARIOS upscalers (p.ej. 2x→4x).
    # La escala total es el producto de los factores de cada engine que
    # realmente upscalea (get_scale_factor > 1).  El ``ai_scale`` inyectado
    # por la UI (nodo Upscale) sigue siendo la fuente para el caso clásico
    # de una sola engine; si la cadena la deriva (p.ej. dos nodos de
    # upscale), el producto manda.
    raw_scale = settings.get("ai_scale")
    try:
        ui_scale = int(raw_scale) if raw_scale is not None else 1
    except (TypeError, ValueError):
        ui_scale = 1
    if ui_scale < 1:
        ui_scale = 1

    chain_scale = _compute_total_scale(engines)
    scale = chain_scale if chain_scale > 1 else ui_scale
    dst_w = src_w * scale
    dst_h = src_h * scale

    # Si el usuario eligió una resolución de export (ej. 720p), usarla como
    # tamaño final del encoder.  El frame upscaled puede ser más grande
    # (ej. 4x de 480p = 1920p) pero el output se ajusta a la resolución
    # de export.  Si es "original", usar la resolución upscaled completa.
    export_res = settings.get("resolution", "original")
    if export_res and export_res != "original":
        try:
            export_h = int(export_res)
            export_w = int(src_w * (export_h / src_h))
            # Forzar par (requerido por la mayoría de codecs).
            export_w = export_w + (export_w % 2)
            export_h = export_h + (export_h % 2)
            if export_w > 0 and export_h > 0:
                dst_w = export_w
                dst_h = export_h
        except (TypeError, ValueError):
            pass  # Mantener dst_w/dst_h del upscale

    # --- Cadena de engines -------------------------------------------------
    steps_antes, steps_despues, interp_step = _split_ai_chain(engines)
    if interp_step is not None:
        logger.info(
            "AI chain: %d engines before interp, 1 interp step (%s), %d after",
            len(steps_antes), getattr(interp_step, "engine_id", "?"), len(steps_despues),
        )
    elif len(engines) > 1:
        logger.info("AI chain: %d engines, no interpolation step", len(engines))

    # --- "Model native": medir la resolución REAL de la cadena -------------
    # Con la opción "Model native" activa, el usuario quiere la resolución
    # de salida EXACTAMENTE como la produce el modelo de upscaling (sin el
    # re-encaje del encoder a un factor elegido). Ejecutamos un frame de
    # prueba (negro, tamaño del source) por los pasos de la cadena y usamos
    # el tamaño resultante para ``-s`` del encoder. La paridad se fuerza
    # hacia abajo (la mayoría de codecs requiere dimensiones pares). Si la
    # medición falla por cualquier motivo, caemos al cálculo escalado.
    if settings.get("ai_native") and settings.get("resolution", "original") == "original":
        try:
            probe = np.zeros((src_h, src_w, 3), dtype=np.uint8)
            for _e in list(steps_antes) + list(steps_despues):
                if getattr(_e, "_failed", False):
                    continue
                try:
                    probe = _e.process_frame(probe)
                except Exception as exc:
                    logger.warning(
                        "native-probe: engine %s failed on the probe frame "
                        "(%s); using scaled resolution instead",
                        getattr(_e, "engine_id", "?"), exc,
                    )
                    raise
            if probe.ndim == 3 and probe.shape[0] > 0 and probe.shape[1] > 0:
                probe_h, probe_w = probe.shape[0], probe.shape[1]
                probe_w -= probe_w % 2
                probe_h -= probe_h % 2
                if probe_w > 0 and probe_h > 0:
                    dst_w, dst_h = probe_w, probe_h
                    logger.info(
                        "Model native: encoder sized to the model output "
                        "(%dx%d) instead of the computed scale",
                        dst_w, dst_h,
                    )
        except Exception as exc:
            logger.warning(
                "Model native probe failed (%s); falling back to the "
                "computed scaled resolution %dx%d", exc, dst_w, dst_h,
            )

    # If interpolating, double the FPS for the encoder so the output
    # plays at the correct speed (original frames + interpolated frames).
    do_interpolate = interp_step is not None
    if do_interpolate:
        try:
            fps_num, fps_den = fps_str.split("/")
            fps_num = int(fps_num)
            fps_den = int(fps_den) if fps_den else 1
            if fps_den:
                new_fps_num = fps_num * 2
                encoder_fps_str = f"{new_fps_num}/{fps_den}"
            else:
                encoder_fps_str = f"{fps_num * 2}"
        except (ValueError, AttributeError):
            encoder_fps_str = fps_str
    else:
        encoder_fps_str = fps_str

    # Output codec settings: honor the user's selected codec and bitrate
    # so NVENC/AMF/QSV are actually used when the user picked them.
    fmt = settings.get("format", "mp4")
    vcodec = settings.get("codec", "libx264")
    real_codec, codec_extra_args = resolve_codec(vcodec)
    preset = settings.get("preset", "medium")
    pixel_fmt = settings.get("pixel_format", "yuv420p")
    extra_flags = settings.get("extra_video_flags", "")

    # --- Fases (tasks panel) ---
    phase_labels = []
    phase_labels.append(("probe", "Analyzing video"))
    engine_phase_ids = {}
    for i, _e in enumerate(engines, 1):
        _eid = getattr(_e, "engine_id", "?") or "?"
        try:
            _meta = get_engine_meta(_eid) or {}
            _label = _meta.get("display_name", _eid)
        except Exception:
            _label = _eid
        phase_labels.append((f"engine-{i}", _label))
        engine_phase_ids[id(_e)] = f"engine-{i}"
    _audio_codec = settings.get("audio", "copy")
    _audio_boost = bool(settings.get("audio_boost", False))
    _has_audio_phase = (
        _audio_codec != "noaudio" and (_audio_boost or _audio_codec != "copy")
    )
    if _has_audio_phase:
        if _audio_boost:
            phase_labels.append(("measure", "Analyzing audio levels"))
        phase_labels.append(("audio", "Encoding audio"))
    phase_labels.append(("mux", "Finalizing / Mux"))
    for ph_id, ph_label in phase_labels:
        _emit_phase(phase_cb, "start", ph_id, label=ph_label, pct=0)

    # --- Audio en paralelo -------------------------------------------------
    # Para no penalizar el throughput de la GPU con la codificación de
    # audio, en un hilo daemon medimos la loudness (si boost) y codificamos
    # la pista completa a un archivo temporal con el mismo codec/filtro que
    # usaría el mux. Cuando el render termina, el mux solo hace copy del
    # archivo pre-codificado -> el vídeo NO espera a la medición.
    audio_tmp = [None]
    audio_error = [None]
    audio_thread = None
    if _has_audio_phase:
        def _audio_job():
            try:
                audio_tmp[0] = _encode_audio_parallel(
                    ffmpeg_path, input_path, settings,
                    status_callback=status_callback, phase_cb=phase_cb,
                    abort_event=abort_event,
                    supervisor=supervisor,
                )
            except Exception as exc:
                audio_error[0] = exc
                audio_tmp[0] = None
        audio_thread = threading.Thread(target=_audio_job, daemon=True)
        audio_thread.start()

    # --- FFmpeg decoder: rawvideo BGR24 pipe -----------------------------
    # ``start_time``/``end_time`` are used by the AI preview to decode only
    # its short sample. Normal full renders leave them unset.
    dec_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "warning",
    ]
    _decode_start = settings.get("start_time")
    _decode_end = settings.get("end_time")
    if _decode_start not in (None, ""):
        dec_cmd.extend(["-ss", str(_decode_start)])
    dec_cmd.extend(["-i", input_path])
    # Los campos de Trim de la UI son TEXTO ("00:01:30"). Antes se hacía
    # float() sobre ellos: con "00:01:30" lanzaba ValueError, se caía al
    # camino "sin límite" y el render seguía hasta el FINAL del vídeo
    # ignorando el End en silencio. Ahora se parsea igual que en el resto de
    # la app (_parse_time_to_seconds) y, si el valor no se entiende, se avisa.
    _decode_start_s = _time_to_seconds(_decode_start, 0.0, "start_time",
                                       status_callback)
    _decode_end_s = _time_to_seconds(_decode_end, None, "end_time",
                                     status_callback)
    if _decode_end_s is not None and _decode_end_s > _decode_start_s:
        dec_cmd.extend(["-t", str(_decode_end_s - _decode_start_s)])
    dec_cmd.extend([
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ])
    dec_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": src_w * src_h * 3 * 4,  # buffer ~4 frames
    }
    dec_kwargs.update(process_group_popen_kwargs())
    dec_proc = subprocess.Popen(dec_cmd, **dec_kwargs)
    if supervisor is not None:
        supervisor.register(dec_proc, "decoder")
    if on_processes:
        try:
            on_processes(dec_proc, "decoder")
        except Exception:
            pass

    # --- FFmpeg encoder: read rawvideo BGR24 pipe, encode to output ---
    # GPU encoders (NVENC/AMF/QSV) do not accept the generic CPU
    # "medium" preset; omitting it lets FFmpeg use the encoder default.
    is_gpu_codec = any(
        real_codec.endswith(suffix) for suffix in ("_nvenc", "_amf", "_qsv")
    )

    enc_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{dst_w}x{dst_h}",
        "-r", encoder_fps_str,
        "-i", "pipe:0",
        "-c:v", real_codec,
    ]
    if not is_gpu_codec:
        enc_cmd.extend(["-preset", preset])
    # Only apply the default yuv420p pixel format if the codec-specific
    # extra args (e.g. 10-bit NVENC) haven't already overridden it.
    if "-pix_fmt" not in codec_extra_args:
        enc_cmd.extend(["-pix_fmt", pixel_fmt])
    enc_cmd.extend(codec_extra_args)

    # Honor the user's bitrate / target-size choice. GPU encoders (NVENC,
    # AMF, QSV) need an explicit bitrate; CPU encoders also accept it.
    mode = settings.get("mode", "")
    if mode == "Target File Size (MB)" and settings.get("target_size_mb"):
        duration = _get_duration_seconds(ffmpeg_path, input_path)
        target_mb = float(settings["target_size_mb"])
        if duration > 0:
            total_bits = target_mb * 8 * 1024 * 1024
            video_bitrate = max(int((total_bits / duration) - 192000), 100000)
            enc_cmd.extend(["-b:v", str(video_bitrate)])
        else:
            enc_cmd.extend(["-b:v", f"{settings.get('bitrate', 1500)}k"])
    else:
        # Default to the bitrate value shown in the UI (kbps).
        enc_cmd.extend(["-b:v", f"{settings.get('bitrate', 1500)}k"])

    if extra_flags:
        enc_cmd.extend(extra_flags.split())

    # Container format override (same logic as build_args).
    if fmt == "wmv":
        enc_cmd.extend(["-f", "asf"])
    elif fmt == "webm":
        enc_cmd.extend(["-f", "webm"])

    enc_cmd.extend(["-an", "-y", output_path])

    # --- Render log (JSON) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = get_logs_dir()
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"render_{timestamp}.json")

    # VRAM disponible ANTES de renderizar: si va justa, el usuario se entera
    # ahora (y no cuando el motor falle en el frame 4000).
    vram_info = _vram_preflight(status_callback)

    engine_ids = [
        getattr(e, "engine_id", "?") for e in engines
    ]
    chain_name = " -> ".join(engine_ids) if engine_ids else "unknown"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_path,
        "output": final_output_path,
        "source_resolution": f"{src_w}x{src_h}",
        "ai_engine": chain_name,
        "ai_engines": engine_ids,
        "ai_scale": scale,
        "interpolation": do_interpolate,
        "export_resolution": settings.get("resolution", "original"),
        "final_resolution": f"{dst_w}x{dst_h}",
        "codec": vcodec,
        "preset": preset,
        "bitrate_kbps": settings.get("bitrate", 1500),
        "format": fmt,
        "pixel_format": pixel_fmt,
        "audio_codec": settings.get("audio", "copy"),
        "audio_boost": bool(settings.get("audio_boost", False)),
        "audio_stream_index": settings.get("audio_stream_index"),
        "audio_stream_label": settings.get("audio_stream_label", ""),
        "total_frames": total_frames,
        "fps": fps_str,
        "encoder_fps": encoder_fps_str,
        "duration_seconds": duration,
        "encoder_command": enc_cmd,
        "decoder_command": dec_cmd,
        "vram": vram_info,
        "status": "pending",
        "render_seconds": None,
        "output_size_bytes": None,
        "error": None,
    }
    _save_render_log(log_file, log_entry)

    enc_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    enc_kwargs.update(process_group_popen_kwargs())
    enc_proc = subprocess.Popen(enc_cmd, **enc_kwargs)
    if supervisor is not None:
        supervisor.register(enc_proc, "encoder")
    if on_process:
        on_process(enc_proc)
    if on_processes:
        try:
            on_processes(enc_proc, "encoder")
        except Exception:
            pass

    # Drain de stderr en background. Sin drain, el PIPE de stderr se
    # acaba llenando (especialmente con NVENC, que escribe stats) y el
    # ffmpeg se cuelga = el loop no vota abort nunca y el worker se
    # queda esperando stdout.read para siempre.
    def _drain_stderr(proc, name):
        # El Popen no se abre con ``encoding=`` porque ``stdout`` lo
        # necesitamos en BYTES (rawvideo). Aquí ``stderr.readline()``
        # devuelve ``bytes``; los decodificamos manualmente para que
        # el log sea legible en lug gar de salir ``b'frame=...\\r\\n'``.
        try:
            for line in iter(proc.stderr.readline, b""):
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="replace")
                except Exception:
                    text = repr(line)
                logger.debug("[ai-%s] %s", name, text.rstrip())
        except (ValueError, OSError):
            pass
        except Exception:
            logger.exception("Unexpected error in ai-%s stderr drain", name)

    threading.Thread(
        target=_drain_stderr, args=(dec_proc, "dec"), daemon=True,
    ).start()
    threading.Thread(
        target=_drain_stderr, args=(enc_proc, "enc"), daemon=True,
    ).start()

    def _kill_proc(proc, name):
        """Termina ffmpeg ``proc`` y espera brevemente; SIGKILL si pasa.

        3 s ``terminate()`` + 1 s ``kill()`` cubre el caso normal y el
        patológico. Sin esto, ``dec_proc.wait(timeout=10)`` del
        ``finally`` previo podía colgar el worker si ffmpeg estaba
        esperando I/O que nunca llegaba.
        """
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                pass
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass
        except Exception:
            logger.exception("Error killing %s subprocess", name)

    frame_size = src_w * src_h * 3  # BGR24
    frame_num = 0
    t_start = time.time()
    prev_front = None  # Salida del frente de la cadena (steps_antes) del frame anterior.
    # Telemetría (P0): presupuesto real por etapa. Nace de una medición donde
    # el 35 % del frame se iba en post-proceso CPU y el 18 % en PIL, sin que
    # el log lo dijera.
    stage_timer = StageTimer()

    def _engine_snapshot(engine):
        """Timings acumulados de una engine antes de llamarla (o None)."""
        try:
            return engine.get_cumulative_timing()
        except Exception:
            return None

    def _record_engine_timing(engine, snapshot, total_ms):
        """Anota el coste de una llamada a engine + sus fases internas."""
        try:
            stage_timer.add(f"engine:{getattr(engine, 'engine_id', '?')}", total_ms)
            if not snapshot:
                return
            now = engine.get_cumulative_timing()
            for key, stage in (("pre_ms", "ai_pre"), ("infer_ms", "ai_infer"),
                               ("post_ms", "ai_post")):
                delta = float(now.get(key, 0.0)) - float(snapshot.get(key, 0.0))
                if delta >= 0:
                    stage_timer.add(stage, delta)
        except Exception:
            pass

    def _apply_steps(target, remaining_steps):
        """Aplica una lista de engines en orden a un frame.

        Si un engine falla se desactiva para el resto del render (el frame
        pasa sin procesar) pero **el usuario se entera**: se avisa en la UI,
        se marca su fase como fallida y el motivo queda en el render log.
        """
        out = target
        for _e in remaining_steps:
            if getattr(_e, "_failed", False):
                continue
            try:
                snapshot = _engine_snapshot(_e)
                step_t0 = time.perf_counter()
                step_out = _e.process_frame(out)
                _record_engine_timing(_e, snapshot,
                                      (time.perf_counter() - step_t0) * 1000.0)
                if step_out is None:
                    raise RuntimeError("engine returned None")
                out = step_out
            except Exception as exc:
                failure = _classify_ai_error(exc)
                failure["engine_id"] = getattr(_e, "engine_id", "?")
                failure["frame"] = frame_num
                _e._failure_detail = failure
                _e._failed = True
                logger.error(
                    "AI engine %s failed at frame %d (%s: %s); disabling it "
                    "for the rest of the render, keeping the previous frame "
                    "output.", failure["engine_id"], frame_num, failure["kind"],
                    failure["message"],
                )
                _notify_ai_failure(_e, failure)
        return out

    def _notify_ai_failure(engine, failure):
        """Hace VISIBLE un fallo de IA en la UI (panel de tareas + estado)."""
        if status_callback:
            try:
                status_callback(
                    f"\u26a0 AI step '{failure.get('engine_id')}' disabled at "
                    f"frame {failure.get('frame')}: {failure.get('title')}. "
                    f"{failure.get('advice')}"
                )
            except Exception:
                logger.debug("status_callback failed for AI failure", exc_info=True)
        _emit_phase(
            phase_cb, "failed", engine_phase_ids.get(id(engine)) or "engine-?",
            label=str(failure.get("engine_id") or ""),
            msg=f"{failure.get('title')}: {str(failure.get('message'))[:160]}",
        )

    def _resize_frame(im, dst_w_in, dst_h_in):
        """Resize de un frame BGR uint8 a (dst_w_in, dst_h_in).

        Usa AREA para reducción (mejor calidad al bajar de escala, p.ej.
        cuando la engine es nativa x4 y el usuario pidió x2) y LINEAR para
        agrandar. Con fallbacks PIL/numpy si opencv no está disponible.
        """
        try:
            import cv2  # type: ignore
            _interp = (
                cv2.INTER_AREA
                if (dst_h_in < im.shape[0] or dst_w_in < im.shape[1])
                else cv2.INTER_LINEAR
            )
            resized = cv2.resize(im, (dst_w_in, dst_h_in), interpolation=_interp)
            if resized.ndim == 2 and dst_h_in and dst_w_in:
                resized = np.repeat(resized[..., None], 3, axis=2)
            return resized
        except ImportError:
            pass
        try:
            from PIL import Image as _PILImg
            pil = _PILImg.fromarray(im)
            pil = pil.resize((dst_w_in, dst_h_in), _PILImg.Resampling.BILINEAR)
            resized = np.array(pil)
            if resized.ndim == 2 and dst_h_in and dst_w_in:
                resized = np.repeat(resized[..., None], 3, axis=2)
            return resized
        except ImportError:
            pass
        if im.ndim == 3:
            row_idx = np.linspace(0, im.shape[0] - 1, dst_h_in, dtype=int)
            col_idx = np.linspace(0, im.shape[1] - 1, dst_w_in, dtype=int)
            return im[row_idx[:, None], col_idx[None, :], :]
        else:
            row_idx = np.linspace(0, im.shape[0] - 1, dst_h_in, dtype=int)
            col_idx = np.linspace(0, im.shape[1] - 1, dst_w_in, dtype=int)
            resized = im[row_idx[:, None], col_idx[None, :]]
            if dst_h_in and dst_w_in:
                resized = np.repeat(resized[..., None], 3, axis=2)
            return resized

    # --- P1: pre/post-proceso en GPU (apagado por defecto) -----------------
    # Se intenta DESPUÉS de conocer el tamaño final del encoder y antes del
    # bucle. Si algo no cuadra (preferencia apagada, sin CUDA, autocomprobación
    # fallida) cada engine sigue con su camino de siempre y se anota el motivo.
    try:
        gpu_post_report = _configure_gpu_post_for_chain(
            steps_antes, steps_despues, interp_step,
            dst_w, dst_h, src_w, src_h, _resize_frame, status_callback,
        )
    except Exception as exc:
        # Un fallo aquí NUNCA puede impedir el render: se sigue por el camino
        # clásico y se anota.
        logger.warning("GPU post-process setup failed: %s", exc)
        gpu_post_report = [{"engine_id": "?", "active": False,
                            "reason": f"setup-error:{type(exc).__name__}: {exc}"}]
    if gpu_post_report:
        log_entry["gpu_post"] = gpu_post_report
    try:
        log_entry["ai_execution"] = [
            _e.describe_execution() for _e in engines
            if hasattr(_e, "describe_execution")
        ]
    except Exception:
        logger.debug("describe_execution unavailable", exc_info=True)

    try:
        while True:
            frame_t0 = time.perf_counter()
            if abort_event and abort_event.is_set():
                break

            # Detect encoder death early: si el encoder muere (codec
            # inválido, disco lleno, ...) pero el decoder sigue vivo,
            # el write a enc_proc.stdin saltará con BrokenPipeError. Lo
            # detectamos también por ``poll()`` para diagnosticar mejor.
            enc_alive = enc_proc.poll() is None
            if not enc_alive:
                logger.error(
                    "Encoder died unexpectedly at frame %d (rc=%s). "
                    "Output MP4 will be incomplete. Check codec/disk.",
                    frame_num, enc_proc.returncode,
                )
                break

            # Decode raw bytes to numpy (H, W, 3) BGR.
            with stage_timer.stage("decode"):
                raw = dec_proc.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    break  # EOF or incomplete frame
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3))

# --- Cadena de AI: frente (pasos previos a la interpolación) ---
            # Aplicamos en orden todas las engines que van ANTES del paso de
            # interpolación (upscale, denoise, colorize, ...).
            cur_front = _apply_steps(frame, steps_antes)

            # --- Interpolación de frames ---
            # Interpola ENTRE el frente del frame previo y el del actual
            # (misma resolución garantizada: la salida del mismo nodo previo
            # de la cadena), y le aplica los pasos posteriores.
            interp_pending = None
            if do_interpolate and prev_front is not None:
                try:
                    with stage_timer.stage("interp"):
                        interp_before = interp_step.interpolate_frame(prev_front, cur_front)
                    interp_pending = _apply_steps(
                        np.ascontiguousarray(interp_before), steps_despues,
                    )
                except Exception as exc:
                    logger.warning(
                        "AI interpolation failed at frame %d: %s. Skipping the "
                        "interpolated frame (A/V cadence may shift).", frame_num, exc,
                    )
                    interp_pending = None
                    # Visible una sola vez (esto se repite en cada frame): el
                    # usuario tiene que saber que su vídeo NO se está
                    # interpolando, no solo enterarse por el log.
                    if not getattr(interp_step, "_failure_reported", False):
                        interp_step._failure_reported = True
                        failure = _classify_ai_error(exc)
                        failure.update({
                            "engine_id": getattr(interp_step, "engine_id", "?"),
                            "frame": frame_num,
                        })
                        interp_step._failure_detail = failure
                        _notify_ai_failure(interp_step, failure)

            # --- Frame actual: pasos posteriores ---
            upscaled = _apply_steps(np.ascontiguousarray(cur_front), steps_despues)
            if upscaled.dtype != np.uint8:
                upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

            # Guardar el frente de este frame para interpolar el siguiente.
            prev_front = cur_front

            # Ajustar tamaño al encoder y escribir.
            if upscaled.shape[0] != dst_h or upscaled.shape[1] != dst_w:
                with stage_timer.stage("resize"):
                    upscaled = _resize_frame(upscaled, dst_w, dst_h)

            frames_to_write = []
            if interp_pending is not None:
                if interp_pending.dtype != np.uint8:
                    interp_pending = np.clip(interp_pending, 0, 255).astype(np.uint8)
                if interp_pending.shape[0] != dst_h or interp_pending.shape[1] != dst_w:
                    with stage_timer.stage("resize"):
                        interp_pending = _resize_frame(interp_pending, dst_w, dst_h)
                frames_to_write.append(interp_pending)
            frames_to_write.append(upscaled)

            for out_frame in frames_to_write:
                try:
                    with stage_timer.stage("write"):
                        enc_proc.stdin.write(out_frame.tobytes())
                except (BrokenPipeError, ValueError, OSError):
                    logger.error("encoder pipe broken at frame %d", frame_num)
                    break

            frame_num += 1
            stage_timer.mark_frame()
            stage_timer.add("frame_total", (time.perf_counter() - frame_t0) * 1000.0)
            if frame_num % 500 == 0:
                _log_stage_progress(stage_timer, frame_num, total_frames)
            if progress_callback and total_frames > 0:
                pct = min(100, int(frame_num * 100 / total_frames))
                progress_callback(pct)
                # Each engine phase advances with the same frame pct; the
                # interpolation phase finishes earlier (only interp frames).
                base_pct = min(100, pct)
                for ph_id, _ph_lbl in phase_labels:
                    if ph_id == "probe":
                        _emit_phase(phase_cb, "progress", ph_id, pct=100)
                    elif ph_id.startswith("engine-"):
                        _emit_phase(phase_cb, "progress", ph_id, pct=base_pct)
            elif progress_callback and duration > 0:
                elapsed = time.time() - t_start
                pct = min(95, int((elapsed / duration) * 50))  # rough estimate
                progress_callback(pct)
            if frame_num % 30 == 0 and (metrics_cb or status_callback):
                elapsed = time.time() - t_start
                speed = frame_num / elapsed if elapsed > 0 else 0
                if metrics_cb:
                    # El MISMO numero que el texto de abajo, pero como dato.
                    # Maquetado en una frase no sirve: el desplegable de la
                    # bandeja no puede parsear una cadena que ademas esta
                    # escrita en ingles y puede cambiar de un dia para otro.
                    _emit_metrics(metrics_cb, fps=speed or None,
                                  frame=frame_num,
                                  total_frames=total_frames)
                if status_callback:
                    mode_label = "AI upscale+interp" if do_interpolate else "AI upscale"
                    status_callback(
                        f"{mode_label}: frame {frame_num}/{total_frames or '?'} "
                        f"({speed:.1f} fps)"
                    )

    except Exception as exc:
        logger.error("AI pipeline frame %d failed: %s", frame_num, exc)
        log_entry["status"] = "error"
        log_entry["error"] = str(exc)
        _save_render_log(log_file, log_entry)
        # Stop parallel audio and any child still blocked on a pipe before
        # propagating the exception. Without this, a model error could leave
        # FFmpeg alive after the worker had already reported failure.
        try:
            if abort_event is not None:
                abort_event.set()
            supervisor.stop_all(grace_seconds=1.0, kill_seconds=1.0)
        except Exception:
            logger.exception("Could not stop AI child processes after failure")
        if audio_thread is not None:
            audio_thread.join(timeout=3.0)
        if audio_tmp[0]:
            _cleanup_path(audio_tmp[0])
        _cleanup_path(output_path)
        raise
    finally:
        # Cerramos stdin del encoder para que reciba EOF y termine por
        # sí mismo si dec_proc.stdout ha llegado a EOF normal.
        try:
            if enc_proc.stdin and not enc_proc.stdin.closed:
                enc_proc.stdin.close()
        except Exception:
            pass
        try:
            if dec_proc.stdout and not dec_proc.stdout.closed:
                dec_proc.stdout.close()
        except Exception:
            pass

        # Esperar a que el ENCODER termine de escribir el ``moov atom``.
        # El bug que estaba aquí: llamábamos ``_kill_proc`` inmediatamente
        # después de cerrar stdin, lo que hacía ``terminate()`` ANTES de
        # que ffmpeg terminara de procesar el último frame y escribir el
        # moov atom del MP4. El archivo quedaba con tamaño "razonable"
        # pero sin moov → VLC no lo reproducía → el mux audio fallaba con
        # "moov atom not found". Damos al encoder tiempo de finalizar
        # naturalmente; si se cuelga, force-kill.
        encoder_returned_cleanly = False
        encoder_rc = None
        try:
            encoder_rc = enc_proc.wait(timeout=30)
            encoder_returned_cleanly = (encoder_rc == 0)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Encoder didn't finalize within 30s after EOF, force-killing "
                "(output file may be incomplete)"
            )
        except Exception:
            logger.exception("Error waiting for encoder to finalize")

        if not encoder_returned_cleanly:
            _kill_proc(enc_proc, "encoder")
        else:
            logger.debug(
                "Encoder finalized cleanly (rc=%s) before mux step", encoder_rc
            )

        # Mata el decoder. Sin esto, un decoder colgado puede dejar
        # ffmpeg corriendo como huérfano después de abort / excepción.
        _kill_proc(dec_proc, "decoder")

        # Actualizar log con resultado del render + telemetría por etapa (P0).
        log_entry["render_seconds"] = time.time() - t_start
        try:
            log_entry["stages"] = stage_timer.summary()
            log_entry["ai_timing"] = [
                {"engine_id": getattr(_e, "engine_id", "?"),
                 **_e.get_cumulative_timing()}
                for _e in engines if hasattr(_e, "get_cumulative_timing")
            ]
        except Exception:
            logger.debug("stage timing collection failed", exc_info=True)
        # Si el camino GPU se abandonó a mitad de render (tamaño inesperado,
        # fallo del grafo), el usuario tiene que enterarse: pierde velocidad
        # sin que el resultado cambie, y un aviso en el log no lo ve nadie.
        try:
            failures = [
                {"engine_id": getattr(_e, "engine_id", "?"),
                 "reason": str(getattr(_e, "_gpu_post_failure"))}
                for _e in engines
                if getattr(_e, "_gpu_post_failure", None)
            ]
            if failures:
                log_entry["gpu_post_failures"] = failures
                if status_callback:
                    for item in failures:
                        status_callback(
                            f"\u26a0 GPU post-process disabled mid-render for "
                            f"{item['engine_id']} ({item['reason']}); "
                            f"continuing on the standard CPU path."
                        )
        except Exception:
            logger.debug("gpu_post failure report failed", exc_info=True)
        # Resumen de pasos de IA que se quedaron por el camino: el render puede
        # terminar "bien" y aun así tener frames SIN procesar. Que se sepa.
        try:
            ai_failures = [
                {"engine_id": getattr(_e, "engine_id", "?"), **_e._failure_detail}
                for _e in engines
                if getattr(_e, "_failure_detail", None)
            ]
            if ai_failures:
                log_entry["ai_failures"] = ai_failures
                if status_callback:
                    names = ", ".join(str(f.get("engine_id")) for f in ai_failures)
                    status_callback(
                        f"\u26a0 {len(ai_failures)} AI step(s) disabled during "
                        f"this render ({names}). The frames after the failure "
                        f"were NOT processed by those steps."
                    )
        except Exception:
            logger.debug("ai failure report failed", exc_info=True)
        _save_render_log(log_file, log_entry)

    if abort_event and abort_event.is_set():
        _cleanup_path(output_path)
        return False

    # Gate de integridad: para nosotros un render "éxito" DEVUELVE ALGO.
    # Si nunca llegó a escribir ningún frame (encoder muerto al inicio,
    # disco lleno, codec inválido) y nadie noto el abort, aquí
    # fallamos de verdad en lugar de muxear el 0-frames y reportar True.
    if frame_num == 0:
        msg = (
            f"AI pipeline returned 0 frames for '{os.path.basename(output_path)}' "
            f"(encoder_rc={encoder_rc}); treating as failure"
        )
        logger.error(msg)
        try:
            _cleanup_path(output_path)
        except Exception:
            pass
        return False

    # --- Mux audio (passthrough o re-codificado con Audio Boost) ---
    audio_codec = settings.get("audio", "copy")
    if audio_codec != "noaudio":
        if status_callback:
            status_callback("Muxing audio...")
        _emit_phase(phase_cb, "progress", "mux", label="Finalizing / Mux", pct=10)
        # Espera al hilo de audio paralelo (normalmente ya acabado: el
        # audio se codifica mucho más rápido que el render de GPU).
        if audio_thread is not None:
            audio_thread.join(timeout=10)
        if _has_audio_phase:
            if audio_tmp[0]:
                _emit_phase(phase_cb, "end", "audio", label="Encoding audio")
            else:
                _emit_phase(phase_cb, "failed", "audio", label="Encoding audio")
        mux_ok = _mux_audio(
            ffmpeg_path, input_path, output_path, audio_codec,
            loudnorm_meas=None,
            audio_boost=bool(settings.get("audio_boost", False)),
            container_format=settings.get("format", "mp4"),
            audio_file=audio_tmp[0],
            abort_event=abort_event,
            supervisor=supervisor,
            audio_stream_index=settings.get("audio_stream_index", 0),
            settings=settings,
        )
        # Si el mux falló sin que el usuario abortase, lo surfaceamos
        # para que no parezca un éxito silencioso (la salida queda
        # sin audio y el usuario pensaría que iba bien).
        if mux_ok:
            mux_ok, mux_detail = validate_render_output(
                ffmpeg_path,
                output_path,
                expected_duration=None,
            )
            if not mux_ok:
                logger.error("Mux validation failed: %s", mux_detail)
        if not mux_ok and (not abort_event or not abort_event.is_set()):
            msg = (
                f"\u26a0 Audio mux failed \u2014 '{os.path.basename(final_output_path)}' "
                "the staged video was discarded."
            )
            if status_callback:
                status_callback(msg)
            logger.error("Audio mux failed for %s", final_output_path)
            _cleanup_path(output_path)
            return False

    # Limpieza del audio temporal pre-codificado en paralelo.
    if audio_tmp[0] and os.path.exists(audio_tmp[0]):
        try:
            os.remove(audio_tmp[0])
        except Exception:
            pass

    # Final publication happens after video encoding and optional audio mux.
    # os.replace is atomic because staging lives beside the destination.
    # Duración esperada REAL del trozo renderizado. Antes, en cuanto
    # ``start_time`` traía cualquier valor, la comprobación se saltaba entera:
    # un render truncado con Trim podía publicarse sin revisarse.
    try:
        _full_duration = _get_duration_seconds(ffmpeg_path, input_path)
        if _decode_end_s is not None and _decode_end_s > _decode_start_s:
            _expected_duration = _decode_end_s - _decode_start_s
        elif _decode_start_s:
            _expected_duration = max(0.0, _full_duration - _decode_start_s)
        else:
            _expected_duration = _full_duration
    except Exception:
        _expected_duration = None
    valid, detail = validate_render_output(
        ffmpeg_path,
        output_path,
        expected_duration=_expected_duration,
    )
    if not valid:
        logger.error("AI pipeline produced invalid staged output: %s", detail)
        _cleanup_path(output_path)
        return False
    try:
        os.replace(output_path, final_output_path)
    except Exception as exc:
        logger.error("Could not publish AI output %s: %s", final_output_path, exc)
        _cleanup_path(output_path)
        return False

    if status_callback:
        elapsed = time.time() - t_start
        chain_label = "AI chain (with interpolation)" if do_interpolate else f"AI chain ({len(engines)} steps)"
        status_callback(f"{chain_label} complete. {frame_num} frames in {elapsed:.1f}s")

    for ph_id, _ph_lbl in phase_labels:
        if ph_id == "probe":
            _emit_phase(phase_cb, "end", ph_id)
        elif ph_id.startswith("engine-"):
            _emit_phase(phase_cb, "end", ph_id)
        elif ph_id == "mux":
            _emit_phase(phase_cb, "progress", ph_id, pct=100)
            _emit_phase(phase_cb, "end", ph_id)

    # Guardar log final con estado de éxito
    log_entry["status"] = "success"
    log_entry["render_seconds"] = time.time() - t_start
    if os.path.exists(final_output_path):
        log_entry["output_size_bytes"] = os.path.getsize(final_output_path)
    _save_render_log(log_file, log_entry)

    return True


def _engine_scale(engine):
    """Best-effort scale factor of an engine (1 for non-upscalers)."""
    try:
        return max(1, int(getattr(engine, "get_scale_factor", lambda: 1)() or 1))
    except Exception:
        return 1


def _check_frame_contract(engine_id, frame, expected_shape=None):
    """Validate the BGR/HWC/uint8 contract one engine must honour.

    Every chain step must consume and produce BGR uint8 ``(H, W, 3)``
    frames. Engines whose ``get_scale_factor() > 1`` may change H/W
    (upscalers); the other steps must preserve the exact shape.
    Detecting a violation here — instead of mid-render, when the
    encoder pipe or a later engine fails — is what lets the UI show a
    clear "engine X does not match the chain contract" error.
    """
    if frame is None:
        raise RuntimeError(
            f"AI engine '{engine_id}' returned None instead of a frame."
        )
    if not isinstance(frame, np.ndarray):
        raise RuntimeError(
            f"AI engine '{engine_id}' returned {type(frame).__name__}, "
            "expected a numpy frame."
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(
            f"AI engine '{engine_id}' returned shape {frame.shape}; the chain "
            "contract requires BGR (H, W, 3) frames."
        )
    if frame.dtype != np.uint8:
        raise RuntimeError(
            f"AI engine '{engine_id}' returned dtype {frame.dtype}; the chain "
            "contract requires uint8 frames."
        )
    if expected_shape is not None and frame.shape != expected_shape:
        raise RuntimeError(
            f"AI engine '{engine_id}' changed the frame shape to {frame.shape} "
            f"(expected {expected_shape}) without being an upscaler."
        )
    return frame


def validate_ai_chain_contract(engines, src_h=None, src_w=None):
    """Probe the whole chain ONCE before the real render starts.

    Runs a small synthetic frame through every engine in chain order and
    validates the BGR/HWC/uint8 contract at each step. This catches
    incompatible chains (an engine that returns RGB, a wrong channel
    count, a non-upscaler that changes resolution, ...) before the user
    wastes minutes of GPU time, instead of failing at frame 1 of 5000.

    The probe is best-effort: engines whose ``process_frame`` fails on
    synthetic input are skipped (the render would also degrade them via
    ``_apply_steps``), and interpolation-only engines are exercised
    through their dedicated ``interpolate_frame`` path when possible.
    Raises RuntimeError only for contract violations.
    """
    steps_antes, steps_despues, interp_step = _split_ai_chain(engines)
    h = int(src_h or 64)
    w = int(src_w or 64)
    probe = np.zeros((h, w, 3), dtype=np.uint8)
    for _e in steps_antes:
        scale = _engine_scale(_e)
        expected = None if scale > 1 else probe.shape
        try:
            out = _e.process_frame(probe)
        except Exception:
            continue  # runtime degradation path covers this engine
        try:
            probe = _check_frame_contract(
                getattr(_e, "engine_id", "?"), out,
                expected_shape=expected,
            )
        except RuntimeError:
            raise
    if interp_step is not None:
        try:
            interp_out = interp_step.interpolate_frame(probe, probe)
            _check_frame_contract(
                getattr(interp_step, "engine_id", "?"), interp_out, probe.shape,
            )
        except Exception:
            pass  # interpolation errors already degrade gracefully at runtime
    for _e in steps_despues:
        scale = _engine_scale(_e)
        expected = None if scale > 1 else probe.shape
        try:
            out = _e.process_frame(probe)
        except Exception:
            continue
        try:
            probe = _check_frame_contract(
                getattr(_e, "engine_id", "?"), out,
                expected_shape=expected,
            )
        except RuntimeError:
            raise
    return True


def _compute_total_scale(engines):
    """Producto de los factores de escala de las engines que upscalean.

    Returns 1 si ninguna engine cambia la resolución. Engines en modo
    interp-only (``_mode == "interpolate"``) no cuentan porque no cambian
    el tamaño del frame.
    """
    total = 1
    for _e in engines:
        try:
            _s = int(getattr(_e, "get_scale_factor", lambda: 1)() or 1)
        except Exception:
            _s = 1
        if _s > 1 and str(getattr(_e, "_mode", "upscale") or "upscale") != "interpolate":
            total *= max(1, _s)
    return total


def _split_ai_chain(engines):
    """Clasifica una lista ordenada de engines en `(antes, despues, interp)`.

    Returns:
        steps_antes: list of engines that transform 1→1 BEFORE the (single)
            interpolation engine (upscale, denoise, colorize, ...).
        steps_despues: list of engines AFTER the interpolation engine — they
            apply BOTH to the interpolated frame and to the current frame.
        interp_step: the single engine with ``_mode in ("interpolate","both")``,
            or None. If several interp-capable engines appear, only the FIRST
            is honored; the rest are treated as passthrough transforms placed
            in ``steps_despues``.
    """
    steps_antes = []
    steps_despues = []
    interp_step = None
    for _e in engines:
        _mode = str(getattr(_e, "_mode", "upscale") or "upscale")
        is_interp = _mode in ("interpolate", "both")
        if is_interp and interp_step is None:
            interp_step = _e
            continue
        (steps_despues if interp_step is not None else steps_antes).append(_e)
    return steps_antes, steps_despues, interp_step


#: Motivos por los que el post-proceso en GPU NO se activa y que NO merecen un
#: aviso en la UI (son situaciones normales, no fallos).
_GPU_POST_SILENT_REASONS = {
    "preference-off", "not-supported", "no-cuda", "no-target-size",
    "not-loaded", "input-size-unknown", "already-native",
}

#: Pistas de que un fallo de IA es falta de memoria de GPU. Se comprueban en
#: minúsculas sobre el mensaje completo de la excepción.
_OOM_HINTS = (
    "out of memory", "outofmemory", "cuda_error_out_of_memory",
    "failed to allocate", "cannot allocate memory", "insufficient memory",
    "cudnn_status_alloc_failed", "cublas_status_alloc_failed",
    "bad_alloc", "memory allocation", "std::bad_alloc",
)

#: Por debajo de esto se avisa ANTES de renderizar (no se bloquea nada: sólo
#: se dice, para que el usuario no descubra el problema 3000 frames después).
_VRAM_LOW_MB = 1500


def _classify_ai_error(exc) -> dict:
    """Traduce una excepción de un motor de IA a algo accionable para el usuario.

    Un fallo de IA a mitad de render NO puede quedarse en el log: hasta ahora
    el pipeline capturaba el error, desactivaba esa engine y seguía, de modo
    que el vídeo terminaba "con éxito" pero con el resto de frames sin
    procesar. Aquí se decide el mensaje que verá el usuario.
    """
    raw = f"{type(exc).__name__}: {exc}"
    low = raw.lower()
    if any(hint in low for hint in _OOM_HINTS):
        return {
            "kind": "out-of-memory",
            "title": "Not enough GPU memory",
            "advice": ("Close other programs using the GPU, export at a lower "
                       "resolution, or pick a lighter model (AnimeVideo v3) / "
                       "a smaller upscale factor."),
            "message": raw,
        }
    if "no module named" in low or "dll load failed" in low or "not installed" in low:
        return {
            "kind": "runtime-missing",
            "title": "AI runtime missing or broken",
            "advice": ("Open Manage Dependencies → Runtimes & Binaries and "
                       "reinstall the runtime for this engine."),
            "message": raw,
        }
    if "cudnn" in low or "cuda" in low or "cublas" in low:
        return {
            "kind": "cuda",
            "title": "GPU/CUDA error",
            "advice": ("Update the NVIDIA driver, or switch AI Execution "
                       "Provider to CPU in Preferences to keep working "
                       "(much slower)."),
            "message": raw,
        }
    if "expects" in low and "channel" in low or "shape" in low or "got:" in low:
        return {
            "kind": "shape",
            "title": "Model does not match the chain",
            "advice": ("That AI step expects a different frame format or "
                       "resolution; remove it from the workflow or pick "
                       "another model."),
            "message": raw,
        }
    return {
        "kind": "unknown",
        "title": "AI step failed",
        "advice": ("The rest of the render continued without that step; "
                   "check the render log for the full error."),
        "message": raw,
    }


def _gpu_memory_info():
    """``(free_mb, total_mb)`` de la primera GPU NVIDIA, o ``(None, None)``.

    Usa ``nvidia-smi`` (rápido y sin cargar torch). Si no existe o falla se
    devuelve ``(None, None)``: la telemetría de VRAM es un extra, nunca un
    requisito para renderizar.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None, None
        line = (result.stdout or "").strip().splitlines()[0]
        total_str, free_str = [part.strip() for part in line.split(",")[:2]]
        return int(float(free_str)), int(float(total_str))
    except Exception as exc:
        logger.debug("nvidia-smi memory query failed: %s", exc)
        return None, None


def _vram_preflight(status_callback=None) -> dict:
    """Avisa si queda poca VRAM ANTES de empezar a renderizar.

    No bloquea ni cancela nada (un render lento es mejor que ningún render):
    sólo hace visible una situación que, si no, el usuario descubre cuando el
    motor falla a mitad de trabajo.
    """
    info: dict = {"checked": False}
    free_mb, total_mb = _gpu_memory_info()
    if free_mb is None:
        return info
    info.update({"checked": True, "free_mb": free_mb, "total_mb": total_mb})
    if free_mb < _VRAM_LOW_MB:
        info["warning"] = f"only {free_mb} MB free of {total_mb} MB"
        message = (
            f"\u26a0 Only {free_mb} MB of GPU memory free (of {total_mb} MB). "
            "The AI step may run out of memory and be disabled mid-render. "
            "Close other GPU programs or lower the export resolution."
        )
        logger.warning("Low VRAM before render: %s MB free of %s MB",
                       free_mb, total_mb)
        if status_callback:
            status_callback(message)
    return info


def _time_to_seconds(value, default, field_name="", status_callback=None):
    """Convierte un tiempo de Trim (número o ``"HH:MM:SS"``) a segundos.

    Devuelve ``default`` cuando el campo está vacío y **avisa** cuando trae un
    valor que no se entiende: interpretarlo como "sin límite" haría que el
    render continuara hasta el final del vídeo (horas de GPU) sin decirlo.
    """
    if value in (None, ""):
        return default
    seconds = _parse_time_to_seconds(value)
    if seconds is None:
        logger.warning(
            "Trim %s could not be parsed (%r); ignoring that bound for this "
            "render.", field_name or "value", value,
        )
        if status_callback:
            status_callback(
                f"\u26a0 Trim {field_name} '{value}' is not a valid time "
                f"(use HH:MM:SS or seconds); that bound was ignored."
            )
        return default
    return float(seconds)


def _last_chain_stage(steps_antes, steps_despues, interp_step):
    """Devuelve el último engine de la cadena (el que alimenta al encoder).

    Sólo ese puede reescalar a la resolución final: si un engine intermedio
    redujera el frame, el siguiente paso de la cadena trabajaría con una
    imagen ya encogida.
    """
    groups = [steps_despues, [interp_step] if interp_step else [], steps_antes]
    for group in groups:
        for engine in reversed(list(group or [])):
            if engine is not None:
                return engine
    return None


def _configure_gpu_post_for_chain(steps_antes, steps_despues, interp_step,
                                  dst_w, dst_h, src_w, src_h, resize_fn,
                                  status_callback=None):
    """Intenta activar el pre/post-proceso en GPU en cada engine (P1).

    Devuelve una lista de informes (uno por engine) apta para el JSON del
    render. Nunca lanza y nunca deja el render peor que antes: si un engine no
    puede activar el camino GPU, sigue con el suyo clásico.
    """
    report = []
    try:
        from src.ai import gpu_post
        if not gpu_post.gpu_post_enabled():
            return report
    except Exception as exc:
        logger.debug("gpu_post unavailable: %s", exc)
        return report

    last_engine = _last_chain_stage(steps_antes, steps_despues, interp_step)
    chain = [e for e in (list(steps_antes) + list(steps_despues)) if e is not None]
    if interp_step is not None:
        chain.append(interp_step)

    for engine in chain:
        # Motores de terceros / de prueba pueden no implementar el camino GPU:
        # eso no es un fallo, simplemente no está disponible.
        if not callable(getattr(engine, "configure_gpu_post", None)):
            report.append({"engine_id": getattr(engine, "engine_id", "?") or "?",
                           "active": False, "reason": "not-supported"})
            continue
        is_last = engine is last_engine
        target = (dst_w, dst_h) if is_last else (None, None)
        try:
            result = engine.configure_gpu_post(
                target[0], target[1],
                resize_fn=resize_fn if is_last else None,
                input_size=(src_w, src_h),
            )
        except Exception as exc:
            result = {"active": False,
                      "reason": f"error:{type(exc).__name__}: {exc}"}
        if not isinstance(result, dict):
            result = {"active": False, "reason": "bad-result"}

        label = getattr(engine, "engine_id", "?") or "?"
        entry = {
            "engine_id": label,
            "resize": bool(is_last),
            "active": bool(result.get("active")),
            "reason": str(result.get("reason", "")),
        }
        for key in ("precision", "cuda_graph", "max_diff", "mean_diff",
                    "converted", "cached", "out_size"):
            if key in result:
                entry[key] = result[key]
        detail = result.get("detail")
        if detail:
            entry["detail"] = str(detail)[:400]
        report.append(entry)

        if status_callback:
            if entry["active"]:
                extra = " + resize" if is_last else ""
                status_callback(
                    f"\u26a1 GPU post-process active for {label}{extra} "
                    f"({entry.get('precision', '?')}, "
                    f"diff {entry.get('max_diff', 0)}/255)"
                )
            elif entry["reason"] not in _GPU_POST_SILENT_REASONS:
                status_callback(
                    f"\u26a0 GPU post-process unavailable for {label} "
                    f"({entry['reason']}); using the standard CPU path."
                )
    if report:
        logger.info("gpu_post report: %s", report)
    return report


def _log_stage_progress(timer, frame_num, total_frames):
    """Línea INFO con el presupuesto por etapa cada N frames (P0)."""
    try:
        stages = (timer.summary() or {}).get("stages", {})
        parts = []
        for name in ("decode", "ai_pre", "ai_infer", "ai_post", "resize", "write"):
            if name in stages:
                parts.append(f"{name}={stages[name]['median_ms']:.1f}ms")
        frame_ms = stages.get("frame_total", {}).get("median_ms", 0.0)
        fps = (1000.0 / frame_ms) if frame_ms > 0 else 0.0
        logger.info(
            "stage timing @%s/%s: %s | frame=%.1fms (%.2f fps)",
            frame_num, total_frames or "?", " ".join(parts), frame_ms, fps,
        )
    except Exception:
        logger.debug("stage timing log failed", exc_info=True)


def _fps_string_to_float(value):
    """Convierte un framerate ffprobe (``"30000/1001"``, ``"30/1"``,
    ``"30"``) a un ``float`` para matemáticas de progreso.

    Devuelve ``30.0`` ante cualquier valor inválido para no romper el
    cálculo de FPS y de frames totales. La cadena original se sigue
    pasando a FFmpeg (``-r 30000/1001``) tal cual.
    """
    if not value:
        return 30.0
    text = str(value).strip()
    if not text:
        return 30.0
    if "/" in text:
        try:
            num_str, den_str = text.split("/", 1)
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return 30.0
            result = num / den
            return result if result > 0 else 30.0
        except (ValueError, TypeError):
            return 30.0
    try:
        result = float(text)
        return result if result > 0 else 30.0
    except (ValueError, TypeError):
        return 30.0


def _save_render_log(path, entry):
    """Guarda un log JSON del render actual. Silencia errores de escritura."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Could not write render log %s: %s", path, exc)


def _audio_mux_args(audio_codec, loudnorm_meas, audio_boost, container_format="mp4",
                    settings=None):
    """Construye los argumentos de audio para el mux tras el render IA.

    Replica la convención de ``build_args``: sin boost y ``copy`` → passthrough;
    con boost (o codec explícito) → re-codifica a ``libmp3lame``/``aac`` y aplica
    la normalización EBU R128.

    El filtro lo decide ``build_boost_filter`` (fuente única de verdad), que
    resuelve los DOS modos del boost: ``loudnorm`` lineal cuando el usuario solo
    marcó la casilla, y la cadena de ``audio_fx`` cuando además activó EQ,
    compresor o limitador en el desplegable.

    Excepción: si el contenedor es ``webm``, el codec de reventa es ``libopus``
    (AAC/MP3 no son válidos en WebM); ``libmp3lame`` solo si el usuario pidió
    MP3, y en ese caso el resultado no se empaqueta en webm (config inválida que
    ya rechazaría build_args).
    """
    if audio_codec == "copy" and not audio_boost:
        return ["-c:a", "copy"]
    if audio_codec == "mp3":
        enc_a = "libmp3lame"
    elif str(container_format).lower() == "webm":
        enc_a = "libopus"
    else:
        enc_a = "aac"
    args = ["-c:a", enc_a, "-b:a", "192k"]
    if audio_boost and not isinstance(loudnorm_meas, (list, tuple)):
        # Con una medición por pista el filtro lo pone quien tiene los mapas
        # (-filter_complex), porque aquí no se sabe de qué entrada sale el
        # audio: en el encode en paralelo es la 0 y en el mux la 1.
        af = build_boost_filter(settings or {}, loudnorm_meas)
        if af:
            args.extend(["-af", af])
    return args


def _encode_audio_parallel(ffmpeg_path, input_path, settings,
                           status_callback=None, phase_cb=None, abort_event=None,
                           supervisor: ProcessSupervisor | None = None):
    """Codifica la pista de audio COMPLETA a un archivo temporal.

    Corre en un hilo daemon mientras el vídeo se renderiza con la GPU.
    Devuelve la ruta del archivo temporal (con codec y loudnorm ya
    aplicados) o ``None`` si no procede / falla. El caller limpia el
    archivo con ``os.remove``.

    El extension se elige según el codec final para que el mux del
    contenedor destino pueda copiar la pista sin re-encodear.

    ``phase_cb`` (opcional) recibe eventos de fase ``audio`` (y ``measure``
    cuando hay boost) con progreso real del encode leyendo
    ``-progress pipe:1`` de ffmpeg, para que esas fases avancen en el panel
    de tareas.
    """
    audio_codec = str(settings.get("audio", "copy") or "copy")
    audio_boost = bool(settings.get("audio_boost", False))
    container_format = settings.get("format", "mp4")
    audio_stream_index = parse_audio_stream_index(settings.get("audio_stream_index", 0))
    keep_all = audio_stream_index == KEEP_ALL_AUDIO
    if audio_codec == "noaudio":
        return None

    loudnorm_meas = None
    if audio_boost:
        # El "start" de la fase measure ya lo emite el bucle inicial de
        # phase_labels (hilo principal); aquí solo la marcamos RUNNING y la
        # cerramos al terminar la pasada de medición.
        _emit_phase(phase_cb, "progress", "measure", label="Analyzing audio levels", pct=0)
        loudnorm_meas = _measure_if_needed(
            ffmpeg_path, input_path, settings, status_callback=status_callback,
            start_time=settings.get("start_time"),
            end_time=settings.get("end_time"),
            abort_event=abort_event, supervisor=supervisor,
        )
        _emit_phase(phase_cb, "progress", "measure", pct=100)
        _emit_phase(phase_cb, "end", "measure")

    mux_args = _audio_mux_args(
        audio_codec, loudnorm_meas, audio_boost, container_format, settings,
    )
    enc_a = mux_args[1]
    if keep_all:
        # Contenedor multi-pista (Matroska audio) para conservar TODAS las
        # pistas re-codificadas; .mp3/.m4a/.opus no siempre admiten N streams.
        ext = ".mka"
    elif enc_a == "libmp3lame":
        ext = ".mp3"
    elif enc_a == "libopus":
        ext = ".opus"
    else:
        ext = ".m4a"

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".drag-audio-", suffix=ext)
    try:
        os.close(tmp_fd)
    except Exception:
        pass

    start_time = settings.get("start_time")
    end_time = settings.get("end_time")
    start_sec = _parse_time_to_seconds(start_time)
    end_sec = _parse_time_to_seconds(end_time)
    cmd = [
        ffmpeg_path,
        "-hide_banner", "-loglevel", "warning",
        "-y",
    ]
    if start_sec is not None:
        cmd.extend(["-ss", str(start_sec)])
    cmd.extend(["-i", input_path])
    if end_sec is not None:
        if start_sec is not None and end_sec > start_sec:
            cmd.extend(["-t", str(end_sec - start_sec)])
        elif start_sec is None:
            cmd.extend(["-t", str(end_sec)])
    if keep_all:
        audio_map = "0:a?"
    else:
        audio_map = f"0:a:{audio_stream_index}?"
    # Con una medición por pista, cada una lleva su filtro: un único -af con
    # la medición de la primera dejaba a las demás con la ganancia de otra.
    por_pista = []
    if audio_boost:
        por_pista = build_per_stream_boost_args(settings, loudnorm_meas)
    if por_pista:
        cmd.extend(por_pista)
    else:
        cmd.extend(["-map", audio_map])
    cmd.extend([
        "-vn",
        "-threads", "0",
        "-c:a", enc_a,
        "-b:a", "192k",
    ])
    if audio_boost and not por_pista and "-af" in mux_args:
        af = mux_args[mux_args.index("-af") + 1]
        cmd.extend(["-af", af])
    if enc_a == "libopus":
        cmd.append("-strict")
        cmd.append("-2")
    cmd.append(tmp_path)

    if phase_cb is not None:
        # -progress pipe:1 emite out_time_ms=... en stdout; lo convertimos a
        # un % de la duración del input para que la fase Audio avance.
        cmd.insert(1, "-progress")
        cmd.insert(2, "pipe:1")
        cmd.insert(3, "-stats_period")
        cmd.insert(4, "0.1")

    duration_s = _get_duration_seconds(ffmpeg_path, input_path) or 0
    if start_sec is not None:
        duration_s = max(0.0, duration_s - start_sec)
    if end_sec is not None:
        if start_sec is not None and end_sec > start_sec:
            duration_s = end_sec - start_sec
        elif start_sec is None:
            duration_s = end_sec
    if phase_cb is not None:
        _emit_phase(phase_cb, "progress", "audio", label="Encoding audio", pct=0)

    audio_kwargs = {
        "stdout": subprocess.PIPE if phase_cb is not None else subprocess.DEVNULL,
        # Do not leave FFmpeg's banner/stats in an undrained pipe: the
        # Windows build can fill it before the encoder exits. Audio progress
        # comes from stdout; stderr is diagnostic-only here.
        "stderr": subprocess.DEVNULL,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    audio_kwargs.update(process_group_popen_kwargs())
    proc = subprocess.Popen(cmd, **audio_kwargs)
    if supervisor is not None:
        supervisor.register(proc, "audio")
    progress_reader = None
    progress_lines = queue_module.Queue()
    if phase_cb is not None:
        def _drain_progress():
            try:
                for line in proc.stdout:
                    progress_lines.put(line)
            except (ValueError, OSError):
                pass
            finally:
                progress_lines.put(None)

        progress_reader = threading.Thread(
            target=_drain_progress, daemon=True, name="audio-progress-reader"
        )
        progress_reader.start()

    try:
        # Consume FFmpeg's progress protocol from a reader thread. Reading
        # stdout directly in this worker made aborts block and could lose the
        # final progress lines when the process exited between polls.
        last_pct = 0 if phase_cb is not None else -1
        eof = False
        while not eof:
            if abort_event and abort_event.is_set():
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                except Exception:
                    pass
                return None
            if phase_cb is None:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            try:
                line = progress_lines.get(timeout=0.1)
            except queue_module.Empty:
                if proc.poll() is not None and progress_reader is not None:
                    # The reader will enqueue its sentinel as soon as the
                    # pipe reaches EOF; keep draining it before finishing.
                    continue
                continue
            if line is None:
                eof = True
                continue
            line = str(line).strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            seconds = None
            if key in ("out_time_us", "out_time_ms"):
                try:
                    # FFmpeg reports both historical keys in microseconds.
                    seconds = float(value) / 1_000_000.0
                except (TypeError, ValueError):
                    pass
            elif key == "out_time":
                seconds = _parse_time_to_seconds(value)
            if key == "progress" and value == "end":
                seconds = duration_s
            if seconds is not None and duration_s > 0:
                pct = min(max(int(seconds * 100 / duration_s), 0), 100)
                if pct >= last_pct and pct != last_pct:
                    last_pct = pct
                    _emit_phase(
                        phase_cb, "progress", "audio",
                        label="Encoding audio", pct=pct,
                    )
        if progress_reader is not None:
            progress_reader.join(timeout=2.0)
        if proc.poll() is None:
            proc.wait(timeout=5)
        if proc.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            if phase_cb is not None and last_pct < 100:
                _emit_phase(phase_cb, "progress", "audio",
                            label="Encoding audio", pct=100)
            return tmp_path
        # stderr is intentionally DEVNULL to keep this background path
        # deadlock-free on Windows. The process return code remains the
        # authoritative failure signal.
        stderr_tail = ""
        logger.warning(
            "Parallel audio encode failed (rc=%s): %s", proc.returncode, stderr_tail,
        )
        return None
    finally:
        try:
            proc.stderr.close()
        except Exception:
            pass
        try:
            if proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
        except Exception:
            pass
        if progress_reader is not None:
            try:
                progress_reader.join(timeout=1.0)
            except Exception:
                pass
        if (proc.returncode is None or proc.returncode != 0) and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _mux_audio(ffmpeg_path, input_path, output_path, audio_codec="copy",
               loudnorm_meas=None, audio_boost=False, container_format="mp4",
               audio_file=None, abort_event=None,
               supervisor: ProcessSupervisor | None = None,
               audio_stream_index=0, settings=None):
    """Mux audio from the original input into the upscaled output.

    Uses ``ffmpeg -i upscaled.{ext} -i original.mp4 -map 0:v -map 1:a
    -c:v copy -c:a copy -shortest`` to avoid re-encoding audio. The
    temp output uses ``tempfile.mkstemp`` with the same extension as
    the destination container: previously the code hardcoded ``.mp4``
    even when the user selected ``.mkv`` / ``.avi`` / ``.webm``,
    which caused muxer-internal ``Invalid argument`` errors.

    ``audio_file`` (optional) is a pre-encoded audio track (produced in
    parallel by ``_encode_audio_parallel``); when given it is used as
    the audio source instead of ``input_path`` and copied verbatim
    (boost/filter already baked in).

    Polls ``abort_event`` every 100 ms while ffmpeg runs so the worker
    can be aborted cleanly from the GUI instead of blocking here for
    up to 5 minutes.
    """
    out_ext = os.path.splitext(output_path)[1] or ".mp4"
    # Track temp path so el finally global lo limpia en TODA ruta de
    # salida (rc != 0, excepción, abort, input sin audio). Sin esto,
    # los ``.mux-*`` quedan en la carpeta del usuario si cualquier
    # rama escapa la limpieza explícita.
    cleanup_path = None
    try:
        try:
            tmp_fd, tmp_out = tempfile.mkstemp(
                prefix=".mux-", suffix=out_ext,
            )
            cleanup_path = tmp_out
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        except Exception:
            # Fallback a un nombre fijo si mkstemp falla (raro en Win).
            tmp_out = output_path + ".mux" + out_ext
            cleanup_path = tmp_out

        audio_stream_index = parse_audio_stream_index(audio_stream_index)
        if audio_stream_index == KEEP_ALL_AUDIO:
            source_audio_map = "1:a?"
        elif audio_file:
            source_audio_map = "1:a:0?"
        else:
            source_audio_map = f"1:a:{audio_stream_index}?"
        # Cuando el boost llega medido por pista, el audio de aquí es la
        # segunda entrada y cada pista lleva su filtro. Con ``audio_file`` el
        # filtro ya viene horneado en el temporal y aqui solo se copia.
        por_pista = []
        if not audio_file:
            por_pista = build_per_stream_boost_args(
                settings or {}, loudnorm_meas, audio_input="1")
        cmd = [
        ffmpeg_path,
        "-hide_banner", "-loglevel", "warning",
        "-i", output_path,
        "-i", (audio_file or input_path),
        "-map", "0:v:0",
        ]
        if por_pista:
            cmd.extend(por_pista)
        else:
            cmd.extend(["-map", source_audio_map])
        cmd.extend([
        "-c:v", "copy",
        ])
        if audio_file:
            # Audio pre-codificado en paralelo: solo copiamos la pista.
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(
                _audio_mux_args(
                    audio_codec, loudnorm_meas, audio_boost, container_format,
                    settings,
                )
            )
        cmd.extend(["-shortest", "-y", tmp_out])
        mux_kwargs = {
            "stdout": subprocess.DEVNULL,
            # Mux progress is polled through proc.poll(); an undrained
            # stderr PIPE can block FFmpeg before it reaches that point.
            "stderr": subprocess.DEVNULL,
            "encoding": "utf-8",
            "errors": "replace",
        }
        mux_kwargs.update(process_group_popen_kwargs())
        proc = subprocess.Popen(cmd, **mux_kwargs)
        if supervisor is not None:
            supervisor.register(proc, "mux")
        try:
            stderr_tail = ""
            try:
                while proc.poll() is None:
                    if abort_event and abort_event.is_set():
                        try:
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                try:
                                    proc.kill()
                                    proc.wait(timeout=2)
                                except Exception:
                                    pass
                        except Exception:
                            logger.exception("Error terminating ffmpeg mux")
                        if os.path.exists(tmp_out):
                            try:
                                os.remove(tmp_out)
                            except Exception:
                                pass
                        return False
                    # No leer stderr aquí: read() bloquearía hasta que
                    # FFmpeg terminase y haría imposible observar abort_event.
                    time.sleep(0.1)
                stderr_tail = ""
            except Exception:
                logger.exception("Unexpected error during mux polling")
                if os.path.exists(tmp_out):
                    try:
                        os.remove(tmp_out)
                    except Exception:
                        pass
                    return False
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.stderr.close()
            except Exception:
                pass

        if proc.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
            try:
                os.replace(tmp_out, output_path)
                cleanup_path = None  # replaced into output_path; finally won't delete
                return True
            except Exception as exc:
                logger.error(
                    "mux replace failed (%s); stderr tail: %s", exc, stderr_tail,
                )
                return False
        if stderr_tail:
            logger.error(
                "mux ffmpeg failed (rc=%s): %s", proc.returncode, stderr_tail,
            )
        return False
    finally:
        # Limpieza exhaustiva: este finally cubre cualquier ruta de
        # salida que ``return`` o excepción no haya manejado ya. Si
        # ``cleanup_path`` es None, el archivo ya fue reemplazado.
        if cleanup_path is not None and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except Exception:
                pass

