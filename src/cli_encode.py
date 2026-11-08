"""Headless encode entrypoint for the installed DraggyEncoder executable.

This module is intentionally Qt-free.  It is invoked by ``main.py`` through
``DraggyEncoder.exe --encode-job INPUT OUTPUT [--model ENGINE_ID]`` and uses
the same model registry, inference factory and ``compress_video_with_ai``
pipeline as the GUI.  Keeping the entrypoint inside the executable means the
installed app, not an external Python harness, owns the whole encode.

The default engine is the installed Real-ESRGAN AnimeVideo v3 upscaler.  The
model path is resolved from the persisted install state (which lives under the
app's own data directory) and is rejected if it does not point inside the
installed models tree.  This is the path exercised by the real render, including
the AI host and its x4 scaling.

The normal GUI remains the default when the flag is absent.
"""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_ID = "upscale-realesr-animevideov3"


def _configure_temp_on_install_drive() -> None:
    """Keep temporary files next to the install, never in the user profile."""
    from src.config import configure_process_temp

    configure_process_temp()


def _parse_args(argv: list[str]) -> tuple[str, str, str | None]:
    """Parse ``--encode-job INPUT OUTPUT [--model ENGINE_ID]``.

    ``--encode-job`` is stripped because ``main.py`` passes the whole command
    line, and ``--model=ID`` is accepted for callers that prefer that form.
    """
    args = [a for a in argv if a != "--encode-job"]
    model_id = None

    if "--model" in args:
        i = args.index("--model")
        if i + 1 >= len(args):
            raise SystemExit("--model needs an engine id")
        model_id = args[i + 1]
        del args[i:i + 2]
    else:
        for i, arg in enumerate(args):
            if arg.startswith("--model="):
                model_id = arg.split("=", 1)[1].strip()
                del args[i]
                break

    if len(args) != 2:
        raise SystemExit(
            "usage: DraggyEncoder.exe --encode-job INPUT OUTPUT "
            "[--model ENGINE_ID]"
        )
    return args[0], args[1], model_id


def _select_upscale_entry(requested_model: str | None) -> dict:
    """Return the installed Upscale registry entry for this job.

    ``list_installed`` is intentional: the Upscale node is explicitly active
    for this operation, so the engine does not need the separate "enabled"
    preference used by the GUI's model list.  The persisted ``enabled`` flag
    is unrelated and leaving it untouched preserves the state the user sees.
    """
    from src.ai.registry import list_installed

    installed = list_installed("Upscale")
    if not installed:
        raise SystemExit("no installed Upscale model is registered")

    if requested_model:
        entry = next(
            (e for e in installed if e.get("engine_id") == requested_model),
            None,
        )
        if entry is None:
            available = ", ".join(
                str(e.get("engine_id") or "?") for e in installed
            )
            raise SystemExit(
                f"Upscale model is not installed: {requested_model} "
                f"(available: {available})"
            )
        return entry

    entry = next(
        (e for e in installed if e.get("engine_id") == DEFAULT_ENGINE_ID),
        installed[0],
    )
    return entry


def _resolve_model(engine_id: str) -> tuple[str, str]:
    """Resolve a model file from the persisted install state.

    Returns ``(path, inference_id)``.  ONNX is preferred because the installed
    engine already has a validated ONNX graph; a PyTorch file is a fallback for
    installations that have not been converted yet.  The path must be inside
    the app's models tree, which keeps the encode from silently loading a
    model from another drive or the system profile.
    """
    from src.ai.engines_state import get_state
    from src.ai.inference import get_model_path, pick_weight_file
    from src.ai.registry import get_engine_dir
    from src.config import get_models_dir

    state = get_state(engine_id) or {}
    if not state.get("installed"):
        raise SystemExit(f"Upscale model is not marked installed: {engine_id}")

    model_dir = state.get("install_path") or get_engine_dir(engine_id)
    if not model_dir or not os.path.isdir(model_dir):
        raise SystemExit(f"model directory is missing: {model_dir}")

    model_path = get_model_path(engine_id, "onnxruntime")
    inference_id = "onnxruntime"
    if not model_path or not os.path.isfile(model_path):
        model_path = pick_weight_file(model_dir, engine_id, recursive=True)
        inference_id = engine_id
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit(
            f"no usable weight file found for {engine_id} in {model_dir}"
        )

    models_root = os.path.abspath(get_models_dir())
    resolved = os.path.abspath(model_path)
    if resolved != models_root and not resolved.startswith(models_root + os.sep):
        raise SystemExit(
            f"model resolved outside the installed models directory: {resolved}"
        )
    return resolved, inference_id


def _engine_loaded(engine) -> bool:
    return bool(getattr(engine, "is_loaded", False))


def run_encode_cli(argv: list[str]) -> int:
    """Run one real video upscale through the production AI pipeline.

    Exit codes: 0 = output published, 2 = render or setup failure.
    """
    from src.ai.inference import create_inference_engine
    from src.config import get_logs_dir, get_models_dir
    from src.ffmpeg.manager import get_local_ffmpeg_path
    from src.processing.pipeline import compress_video_with_ai

    _configure_temp_on_install_drive()
    input_path, output_path, requested_model = _parse_args(argv)
    if not os.path.isfile(input_path):
        raise SystemExit(f"input not found: {input_path}")
    if os.path.getsize(input_path) <= 0:
        raise SystemExit(f"input is empty: {input_path}")

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    entry = _select_upscale_entry(requested_model)
    engine_id = str(entry.get("engine_id") or "")
    model_path, inference_id = _resolve_model(engine_id)

    ffmpeg = get_local_ffmpeg_path()
    if not ffmpeg or not os.path.isfile(ffmpeg):
        raise SystemExit(f"ffmpeg not found: {ffmpeg}")

    log_path = os.path.join(get_logs_dir(), "encode_cli.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    started = time.time()
    logger.info(
        "encode-cli start input=%s output=%s engine=%s inference=%s "
        "model=%s models_root=%s",
        input_path,
        output_path,
        engine_id,
        inference_id,
        model_path,
        get_models_dir(),
    )

    engine = None
    try:
        engine = create_inference_engine(inference_id, model_path)
        if engine is None or not _engine_loaded(engine):
            raise RuntimeError("the inference engine did not load")
        try:
            engine_scale = int(engine.get_scale_factor() or 4)
        except (TypeError, ValueError):
            engine_scale = 4
        if engine_scale < 1:
            engine_scale = 4
        actual_engine_id = str(getattr(engine, "engine_id", engine_id) or engine_id)
        logger.info(
            "encode-cli engine loaded engine=%s providers=%s scale=%s",
            actual_engine_id,
            getattr(engine, "get_active_providers", lambda: None)(),
            engine_scale,
        )

        settings = {
            "codec": "libx264",
            "format": "mp4",
            "resolution": "original",
            "audio": "noaudio",
            "audio_stream_index": 0,
            "bitrate": 1500,
            "preset": "ultrafast",
            "pixel_format": "yuv420p",
            "mode": "Bitrate",
            # The chain scale is normally read from the loaded ONNX graph; this
            # is the authoritative fallback for this known x4 engine and keeps
            # the encoder dimensions at exactly four times the source.
            "ai_scale": engine_scale,
        }

        ok = compress_video_with_ai(
            ffmpeg,
            input_path,
            output_path,
            settings,
            inference_engines=[engine],
            progress_callback=lambda p: logger.info("progress %s", p),
            status_callback=lambda msg: logger.info("status %s", msg),
        )
    except SystemExit:
        raise
    except Exception:
        logger.exception("encode-cli failed")
        return 2
    finally:
        if engine is not None:
            try:
                engine.unload()
            except Exception:
                pass
        try:
            from src.ai import host_client

            host_client.stop_host()
        except Exception:
            pass

    output_size = (
        os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    )
    if not ok:
        logger.error(
            "encode-cli finished ok=False elapsed=%.2fs size=%s",
            time.time() - started,
            output_size,
        )
        return 2
    if output_size <= 0:
        logger.error("encode-cli produced no output: %s", output_path)
        return 2

    logger.info(
        "encode-cli finished ok=True elapsed=%.2fs size=%s",
        time.time() - started,
        output_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_encode_cli(sys.argv[1:]))
