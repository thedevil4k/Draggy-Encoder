import os
import subprocess
from datetime import datetime

from src.config import ALL_KNOWN_CODECS, get_bin_dir, get_codec_entry, resolve_codec


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

    Deriva el comando de la tabla declarativa (``get_codec_entry``): usa el
    pixel format de salida del codec (8/10/12 bits), sus args extra (perfil
    ProRes/DNxHR) y el tamaño de fuente que exige (DNxHR requiere 1080p).
    La salida va a ``-f null -``, así el probe solo valida el encoder y no
    depende del muxer ni del contenedor.
    """
    try:
        entry = get_codec_entry(codec)
        real_codec, _extra = resolve_codec(codec)
        pix_fmt = (entry or {}).get("pix_fmt", "yuv420p")
        extra_args = list((entry or {}).get("extra") or [])
        size = (entry or {}).get("probe_size", "320x240")
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"testsrc2=size={size}:rate=25:duration=0.2",
            "-pix_fmt", pix_fmt,
            "-c:v", real_codec,
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
