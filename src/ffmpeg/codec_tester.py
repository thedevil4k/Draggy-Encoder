import os
import subprocess
from datetime import datetime

from src.config import ALL_KNOWN_CODECS, get_bin_dir, resolve_codec


def _log_dir():
    """Directorio donde se guardan los logs de errores de codecs."""
    log_dir = os.path.join(get_bin_dir(), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    return log_dir


def _append_log(codec_id, success, error=""):
    """Anexa una línea al log de test de codecs para diagnóstico."""
    try:
        log_file = os.path.join(_log_dir(), "codec_test.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "OK" if success else "FAIL"
        line = f"[{ts}] {status} {codec_id}"
        if not success and error:
            short = (error or "").strip().splitlines()[-3:]
            line += " | " + " || ".join(s.strip() for s in short if s.strip())[:400]
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _test_codec(ffmpeg_path, codec):
    """Prueba si un codec puede codificar un frame de prueba.

    Estrategia robusta:
    - Input: testsrc2 con duración 0.2s @ 30fps (6 frames).
    - Pixel format de input explícito (yuv420p) → compatible con TODOS los encoders.
    - Bitrate explícito (-b:v 1M) → evita fallos de rate control.
    - Output a -f null - → no depende del muxer, solo prueba el encoder.
    - Pix_fmt extra (p010le para codecs 10-bit) se aplica como opción de OUTPUT.
    """
    try:
        real_codec, extra_args = resolve_codec(codec)
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "lavfi",
            "-i", "testsrc2=size=320x240:rate=30:duration=0.2",
            "-pix_fmt", "yuv420p",
            "-c:v", real_codec,
            "-b:v", "1M",
            "-frames:v", "1",
        ] + extra_args + [
            "-f", "null",
            "-",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        ok = result.returncode == 0
        _append_log(codec, ok, result.stderr if not ok else "")
        return ok
    except subprocess.TimeoutExpired:
        _append_log(codec, False, "TIMEOUT (>30s)")
        return False
    except Exception as exc:
        _append_log(codec, False, f"EXC: {exc}")
        return False


def test_all_codecs(ffmpeg_path, progress_callback=None):
    """Prueba todos los codecs conocidos y devuelve los compatibles."""
    try:
        log_file = os.path.join(_log_dir(), "codec_test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
    except Exception:
        pass

    compatible = {}
    total = sum(len(v) for v in ALL_KNOWN_CODECS.values())
    current = 0
    for device, codecs in ALL_KNOWN_CODECS.items():
        compatible[device] = []
        for codec_id, codec_name in codecs:
            ok = _test_codec(ffmpeg_path, codec_id)
            if ok:
                compatible[device].append((codec_id, codec_name))
            current += 1
            if progress_callback:
                try:
                    progress_callback(int(current * 100 / total))
                except Exception:
                    pass
    return compatible


def test_specific_codec(ffmpeg_path, codec_id):
    """Prueba un único codec (útil para el botón 'Re-test')."""
    return _test_codec(ffmpeg_path, codec_id)
