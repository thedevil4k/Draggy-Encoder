"""Headless encode entrypoint for the installed DraggyEncoder executable.

This module is intentionally Qt-free.  It is invoked by ``main.py`` through
``DraggyEncoder.exe --encode-job INPUT OUTPUT`` and uses the same model
registry, inference factory and ``compress_video_with_ai`` pipeline as the
GUI.  Keeping the entrypoint inside the executable means the installed app,
not an external Python harness, owns the whole encode.

The normal GUI remains the default when the flag is absent.
"""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


def _configure_temp_on_install_drive() -> None:
    """Keep temporary files next to the install, never in the user profile."""
    from src.config import configure_process_temp

    configure_process_temp()


def _parse_args(argv: list[str]) -> tuple[str, str, str | None]:
    args = [a for a in argv if a != "--encode-job"]
    model_id = None
    if "--model" in args:
        i = args.index("--model")
        if i + 1 >= len(args):
            raise SystemExit("--model needs an engine id")
        model_id = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        raise SystemExit(
            "usage: DraggyEncoder.exe --encode-job INPUT OUTPUT [--model ENGINE_ID]"
        )
    return args[0], args[1], model_id


def run_encode_cli(argv: list[str]) -> int:
    from src.ai.inference import create_inference_engine, get_model_path
    from src.ai.registry import list_enabled
    from src.config import get_logs_dir
    from src.ffmpeg.manager import get_local_ffmpeg_path
    from src.processing.pipeline import compress_video_with_ai

    _configure_temp_on_install_drive()
    input_path, output_path, requested_model = _parse_args(argv)
    if not os.path.isfile(input_path):
        raise SystemExit(f"input not found: {input_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    enabled = list_enabled("Colorize")
    if not enabled:
        raise SystemExit("no enabled Colorize model is registered")
    if requested_model:
        entry = next(
            (e for e in enabled if e.get("engine_id") == requested_model), None
        )
        if entry is None:
            raise SystemExit(f"Colorize model not enabled: {requested_model}")
    else:
        entry = enabled[0]
    engine_id = str(entry.get("engine_id") or "")
    model_path = get_model_path(engine_id, "onnxruntime")
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit(f"no ONNX weight file for {engine_id}: {model_path}")

    ffmpeg = get_local_ffmpeg_path()
    if not ffmpeg or not os.path.isfile(ffmpeg):
        raise SystemExit(f"ffmpeg not found: {ffmpeg}")

    settings = {
        "codec": "libx264",
        "format": "mp4",
        "resolution": "original",
        "audio": "copy",
        "audio_stream_index": 0,
        "bitrate": 1500,
        "preset": "medium",
        "pixel_format": "yuv420p",
        "mode": "Bitrate",
    }

    log_path = os.path.join(get_logs_dir(), "encode_cli.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    started = time.time()
    logger.info("encode-cli start input=%s output=%s engine=%s model=%s",
                input_path, output_path, engine_id, model_path)

    engine = create_inference_engine("onnxruntime", model_path)
    if engine is None:
        raise SystemExit("could not create the ONNX inference engine")
    loaded = engine.load()
    if not loaded:
        raise SystemExit("the ONNX inference engine did not load")
    logger.info("encode-cli engine loaded providers=%s",
                getattr(engine, "get_active_providers", lambda: None)())

    try:
        ok = compress_video_with_ai(
            ffmpeg,
            input_path,
            output_path,
            settings,
            inference_engines=[engine],
            progress_callback=lambda p: logger.info("progress %s", p),
            status_callback=lambda msg: logger.info("status %s", msg),
        )
    finally:
        try:
            engine.unload()
        except Exception:
            pass

    logger.info("encode-cli finished ok=%s elapsed=%.2fs size=%s",
                ok, time.time() - started,
                os.path.getsize(output_path) if os.path.exists(output_path) else 0)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(run_encode_cli(sys.argv[1:]))
