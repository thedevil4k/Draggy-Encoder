import json
import logging
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime

import numpy as np

from src.ai.registry import get_engine_meta
from src.config import get_logs_dir, resolve_codec
from src.ffmpeg.loudnorm import measure_loudness
from src.ffmpeg.runner import (
    _build_loudnorm_filter,
    build_args,
    extract_frame,
    get_video_info,
    run_ffmpeg,
)


logger = logging.getLogger(__name__)


def _get_duration_seconds(ffmpeg_path, video_path):
    info = get_video_info(ffmpeg_path, video_path)
    try:
        return float(info.get("duration", 0))
    except Exception:
        return 0


def _needs_audio_reencode(settings):
    """Indica si vamos a re-codificar audio (y por tanto tiene sentido medir)."""
    audio = settings.get("audio", "copy")
    audio_boost = bool(settings.get("audio_boost", False))
    if audio == "noaudio":
        return False
    if audio == "copy" and not audio_boost:
        return False
    return True


def _measure_if_needed(ffmpeg_path, input_path, settings, status_callback=None,
                       start_time=None, end_time=None):
    """Mide loudness solo si audio_boost=True y vamos a re-codificar audio.

    Devuelve el dict medido (modulo de loudnorm) o None si no aplica / falla.
    Para el preview se puede pasar ``start_time`` / ``end_time`` para medir
    únicamente el trozo que se va a codificar, ahorrando tiempo.
    """
    if not settings.get("audio_boost"):
        return None
    if not _needs_audio_reencode(settings):
        return None
    if status_callback:
        status_callback("Analyzing audio levels (first pass)...")
    try:
        return measure_loudness(
            ffmpeg_path, input_path,
            start_time=start_time, end_time=end_time,
            status_callback=status_callback,
        )
    except Exception:
        return None


def compress_video(ffmpeg_path, input_path, output_path, settings,
                   progress_callback=None, status_callback=None, abort_event=None,
                   ai_engines=None, on_process=None):
    """Comprime un vídeo completo. Audio boost usa two-pass loudnorm.

    ``ai_engines`` is an optional mapping ``{"Upscale": "engine_id",
    "Interpolate": "engine_id", "Colorize": "engine_id"}``. Today we
    resolve each id against ``src.ai.registry`` and emit a friendly
    status line; real inference wiring comes in a follow-up milestone.
    Pass ``None`` (default) for projects that don't use the AI chain.

    ``on_process`` is an optional callable invoked once with the
    ``subprocess.Popen`` from ``run_ffmpeg``. The GUI uses it to keep
    a reference to the ffmpeg child so it can ``kill()`` directly
    when the abort_event path doesn't reach run_ffmpeg in time (see
    ``_abort_running_worker`` in ``src/ui/main_window.py``).
    """
    settings = dict(settings)
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

    loudnorm_meas = _measure_if_needed(
        ffmpeg_path, input_path, settings, status_callback=status_callback,
        start_time=settings.get("start_time"),
        end_time=settings.get("end_time"),
    )
    args = build_args(input_path, output_path, settings, loudnorm_meas=loudnorm_meas)
    return run_ffmpeg(
        ffmpeg_path,
        args,
        progress_callback=progress_callback,
        abort_event=abort_event,
        on_process=on_process,
    )


def generate_preview(ffmpeg_path, input_path, settings, progress_callback=None,
                     status_callback=None, abort_event=None, ai_engines=None,
                     on_process=None):
    """Genera un preview: codifica un clip corto y extrae un frame aleatorio.

    Devuelve una tupla ``(original_bytes, compressed_bytes)`` con los bytes de
    dos imágenes JPEG. Los ``QPixmap`` deben construirse en el hilo principal
    de la GUI para evitar crashes de Qt; por eso no se devuelven objetos
    gráficos desde este worker.

    ``ai_engines`` carries the same shape as in ``compress_video`` and is
    surfaced in the status line so the user sees which engines were chosen
    for this preview. Real inference wiring lands in a follow-up.
    """
    settings = dict(settings)
    duration = _get_duration_seconds(ffmpeg_path, input_path)
    settings["duration_seconds"] = duration

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

    # Clip corto de hasta 5 segundos desde el 20% del vídeo, pero nunca más
    # de lo que dura el vídeo. Si falla la detección de duración usamos un
    # clip de 5 s desde el inicio.
    start = max(0, duration * 0.2) if duration > 0 else 0
    end = start + min(5, max(1, duration - start))
    # Asegurar que no nos pasamos del final del vídeo.
    if duration > 0:
        end = min(end, duration)
    clip_length = max(0.1, end - start)

    # Elegir un fotograma aleatorio dentro del clip, evitando los extremos.
    # Se usa una semilla basada en el path para que el frame sea reproducible
    # entre previews consecutivos del mismo archivo.
    rng = random.Random(input_path)
    random_frame_time = rng.uniform(0.1, max(0.2, clip_length - 0.1))

    # Medimos loudness SOBRE EL MISMO CLIP que se va a codificar.
    loudnorm_meas = _measure_if_needed(
        ffmpeg_path, input_path, settings, status_callback=status_callback,
        start_time=start, end_time=end,
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
        args = build_args(input_path, compressed_path, preview_settings, loudnorm_meas=loudnorm_meas)
        logger.info("Preview encode args: %s", args)
        ok = run_ffmpeg(
            ffmpeg_path,
            args,
            progress_callback=progress_callback,
            abort_event=abort_event,
            on_process=on_process,
        )
        if not ok:
            raise RuntimeError(
                "FFmpeg failed to encode the preview clip. "
                "Check bin/logs/ffmpeg.log for the full FFmpeg output."
            )

        # Elegir un fotograma seguro dentro del clip generado.
        # Nunca pidamos un tiempo mayor que la duración real del clip.
        frame_time = min(random_frame_time, clip_length - 0.1)
        compressed_frame = os.path.join(tmpdir, "compressed_frame.jpg")
        ok_comp = extract_frame(ffmpeg_path, compressed_path, frame_time, compressed_frame)

        original_time = start + frame_time
        original_frame = os.path.join(tmpdir, "original_frame.jpg")
        ok_orig = extract_frame(ffmpeg_path, input_path, original_time, original_frame)

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

        return original_data, compressed_data


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
    inference_engines=None,
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

    # --- FFmpeg decoder: rawvideo BGR24 pipe ---
    dec_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "warning",
        "-i", input_path,
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ]
    dec_proc = subprocess.Popen(
        dec_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=src_w * src_h * 3 * 4,  # buffer ~4 frames
    )

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

    engine_ids = [
        getattr(e, "engine_id", "?") for e in engines
    ]
    chain_name = " -> ".join(engine_ids) if engine_ids else "unknown"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_path,
        "output": output_path,
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
        "total_frames": total_frames,
        "fps": fps_str,
        "encoder_fps": encoder_fps_str,
        "duration_seconds": duration,
        "encoder_command": enc_cmd,
        "decoder_command": dec_cmd,
        "status": "pending",
        "render_seconds": None,
        "output_size_bytes": None,
        "error": None,
    }
    _save_render_log(log_file, log_entry)

    enc_proc = subprocess.Popen(
        enc_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if on_process:
        on_process(enc_proc)

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

    def _apply_steps(target, remaining_steps):
        """Aplica una lista de engines en orden a un frame."""
        out = target
        for _e in remaining_steps:
            if getattr(_e, "_failed", False):
                continue
            try:
                step_out = _e.process_frame(out)
                if step_out is None:
                    raise RuntimeError("engine returned None")
                out = step_out
            except Exception as exc:
                logger.error(
                    "AI engine %s failed at frame %d (%s); disabling it for "
                    "the rest of the render, keeping the previous frame output.",
                    getattr(_e, "engine_id", "?"), frame_num, exc,
                )
                _e._failed = True
        return out

    def _resize_frame(im, dst_w_in, dst_h_in):
        """Nearest-neighbor resize de un frame BGR uint8 (con fallbacks)."""
        try:
            import cv2  # type: ignore
            resized = cv2.resize(
                im, (dst_w_in, dst_h_in),
                interpolation=cv2.INTER_NEAREST,
            )
            if resized.ndim == 2 and dst_h_in and dst_w_in:
                resized = np.repeat(resized[..., None], 3, axis=2)
            return resized
        except ImportError:
            pass
        try:
            from PIL import Image as _PILImg
            pil = _PILImg.fromarray(im)
            pil = pil.resize((dst_w_in, dst_h_in), _PILImg.Resampling.NEAREST)
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

    try:
        while True:
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

            raw = dec_proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break  # EOF or incomplete frame

            # Decode raw bytes to numpy (H, W, 3) BGR.
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

            # --- Frame actual: pasos posteriores ---
            upscaled = _apply_steps(np.ascontiguousarray(cur_front), steps_despues)
            if upscaled.dtype != np.uint8:
                upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

            # Guardar el frente de este frame para interpolar el siguiente.
            prev_front = cur_front

            # Ajustar tamaño al encoder y escribir.
            if upscaled.shape[0] != dst_h or upscaled.shape[1] != dst_w:
                upscaled = _resize_frame(upscaled, dst_w, dst_h)

            frames_to_write = []
            if interp_pending is not None:
                if interp_pending.dtype != np.uint8:
                    interp_pending = np.clip(interp_pending, 0, 255).astype(np.uint8)
                if interp_pending.shape[0] != dst_h or interp_pending.shape[1] != dst_w:
                    interp_pending = _resize_frame(interp_pending, dst_w, dst_h)
                frames_to_write.append(interp_pending)
            frames_to_write.append(upscaled)

            for out_frame in frames_to_write:
                try:
                    enc_proc.stdin.write(out_frame.tobytes())
                except (BrokenPipeError, ValueError, OSError):
                    logger.error("encoder pipe broken at frame %d", frame_num)
                    break

            frame_num += 1
            if progress_callback and total_frames > 0:
                pct = min(100, int(frame_num * 100 / total_frames))
                progress_callback(pct)
            elif progress_callback and duration > 0:
                elapsed = time.time() - t_start
                pct = min(95, int((elapsed / duration) * 50))  # rough estimate
                progress_callback(pct)
            if status_callback and frame_num % 30 == 0:
                elapsed = time.time() - t_start
                speed = frame_num / elapsed if elapsed > 0 else 0
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

        # Actualizar log con resultado del render
        log_entry["render_seconds"] = time.time() - t_start
        _save_render_log(log_file, log_entry)

    if abort_event and abort_event.is_set():
        return False

    # Gate de integridad: para nosotros un render "éxito" DEVUELVE ALGO.
    # Si nunca llegó a escribir ningún frame (encoder muerto al inicio,
    # disco lleno, libplacebo incompat) y nadie noto el abort, aquí
    # fallamos de verdad en lugar de muxear el 0-frames y reportar True.
    if frame_num == 0:
        msg = (
            f"AI pipeline returned 0 frames for '{os.path.basename(output_path)}' "
            f"(encoder_rc={encoder_rc}); treating as failure"
        )
        logger.error(msg)
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        return False

    # --- Mux audio (passthrough o re-codificado con Audio Boost) ---
    audio_codec = settings.get("audio", "copy")
    if audio_codec != "noaudio":
        if status_callback:
            status_callback("Muxing audio...")
        loudnorm_meas = _measure_if_needed(
            ffmpeg_path, input_path, settings, status_callback=status_callback,
        )
        mux_ok = _mux_audio(
            ffmpeg_path, input_path, output_path, audio_codec,
            loudnorm_meas=loudnorm_meas,
            audio_boost=bool(settings.get("audio_boost", False)),
            container_format=settings.get("format", "mp4"),
            abort_event=abort_event,
        )
        # Si el mux falló sin que el usuario abortase, lo surfaceamos
        # para que no parezca un éxito silencioso (la salida queda
        # sin audio y el usuario pensaría que iba bien).
        if not mux_ok and (not abort_event or not abort_event.is_set()):
            msg = (
                f"\u26a0 Audio mux failed \u2014 '{os.path.basename(output_path)}' "
                "has no audio. The video stream is intact."
            )
            if status_callback:
                status_callback(msg)
            logger.warning(
                "Audio mux returned False for %s (output has no audio track)",
                output_path,
            )

    if status_callback:
        elapsed = time.time() - t_start
        chain_label = "AI chain (with interpolation)" if do_interpolate else f"AI chain ({len(engines)} steps)"
        status_callback(f"{chain_label} complete. {frame_num} frames in {elapsed:.1f}s")

    # Guardar log final con estado de éxito
    log_entry["status"] = "success"
    log_entry["render_seconds"] = time.time() - t_start
    if os.path.exists(output_path):
        log_entry["output_size_bytes"] = os.path.getsize(output_path)
    _save_render_log(log_file, log_entry)

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


def _audio_mux_args(audio_codec, loudnorm_meas, audio_boost, container_format="mp4"):
    """Construye los argumentos de audio para el mux tras el render IA.

    Replica la convención de ``build_args``: sin boost y ``copy`` → passthrough;
    con boost (o codec explícito) → re-codifica a ``libmp3lame``/``aac`` y aplica
    ``loudnorm`` (two-pass si ``loudnorm_meas`` es un dict, si no single-pass).

    Excepción: si el contenedor es ``webm``, el codec de revenida es ``libopus``
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
    if audio_boost:
        af = _build_loudnorm_filter(loudnorm_meas)
        args.extend(["-af", af])
    return args


def _mux_audio(ffmpeg_path, input_path, output_path, audio_codec="copy",
               loudnorm_meas=None, audio_boost=False, container_format="mp4",
               abort_event=None):
    """Mux audio from the original input into the upscaled output.

    Uses ``ffmpeg -i upscaled.{ext} -i original.mp4 -map 0:v -map 1:a
    -c:v copy -c:a copy -shortest`` to avoid re-encoding audio. The
    temp output uses ``tempfile.mkstemp`` with the same extension as
    the destination container: previously the code hardcoded ``.mp4``
    even when the user selected ``.mkv`` / ``.avi`` / ``.webm``,
    which caused muxer-internal ``Invalid argument`` errors.

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

        cmd = [
        ffmpeg_path,
        "-hide_banner", "-loglevel", "warning",
        "-i", output_path,
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        ]
        cmd.extend(
            _audio_mux_args(
                audio_codec, loudnorm_meas, audio_boost, container_format,
            )
        )
        cmd.extend(["-shortest", "-y", tmp_out])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
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
                    time.sleep(0.1)
                    # Capturamos stderr para diagnóstico si algo falla al final.
                    try:
                        stderr_text = proc.stderr.read() or ""
                        stderr_tail = stderr_text[-512:]
                    except Exception:
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

