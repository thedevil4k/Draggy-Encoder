"""Utilidades FFmpeg de un solo uso, inspiradas en la barra de Artplayer.

La app ya tiene el flujo principal (Import -> IA -> Compress) y el editor de
Join. Estas herramientas son operaciones puntuales sobre un vídeo concreto:
capturar un frame, pixelar, difuminar, estabilizar, quitar una marca de agua,
invertir, dividir, cambiar el volumen o añadir pistas de audio de otros
ficheros.

No tocan el pipeline ni el estado de la app: reciben una ruta y devuelven otra.
Así se pueden probar y usar sin arrastrar torch, Qt ni la cola de renders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.ffmpeg.runner import (
    build_boost_filter,
    get_audio_streams,
    get_video_info,
    measurement_is_usable,
)


TOOL_LABELS = {
    "screenshot": "Video screenshot",
    "pixelate": "Pixelate video",
    "blur": "Blur video",
    "stabilize": "Video stabilizer",
    "delogo": "Remove watermark",
    "reverse": "Reverse video",
    "split": "Split video",
    "volume": "Change volume",
    "multiaudio": "Add audio tracks",
}

TOOL_DESCRIPTIONS = {
    "screenshot": "Save one frame as a PNG or JPG image.",
    "pixelate": "Replace the picture with large blocks. Good for hiding faces or details.",
    "blur": "Apply a Gaussian blur to the whole frame.",
    "stabilize": "Reduce camera shake with FFmpeg's deshake filter.",
    "delogo": "Interpolate the pixels inside a rectangle to hide a logo or watermark.",
    "reverse": "Play the clip backwards, audio included when present.",
    "split": "Cut the clip into equal-length parts. Parts are re-encoded for clean cuts.",
    "volume": "Raise or lower the audio gain without re-encoding the video.",
    "multiaudio": "Keep this video and add the audio of another episode (a "
                   "whole video file is fine) as extra selectable tracks, so "
                   "one file carries every dub. Nothing has to be extracted "
                   "first: its first audio track is copied straight in. Each "
                   "added track can be loudness-normalised (EBU R128) so it "
                   "matches the volume of the one already in the video.",
}

TOOL_SUFFIXES = {
    "screenshot": "_frame",
    "pixelate": "_pixelated",
    "blur": "_blurred",
    "stabilize": "_stabilized",
    "delogo": "_clean",
    "reverse": "_reversed",
    "split": "_parts",
    "volume": "_volume",
    "multiaudio": "_multi",
}

_VIDEO_ENCODE = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "18")
_AUDIO_ENCODE = ("-c:a", "aac", "-b:a", "192k")
_MP4_FLAGS = ("-movflags", "+faststart")


@dataclass(frozen=True)
class VideoToolPlan:
    """Todo lo que necesita el worker para ejecutar una herramienta."""

    tool_id: str
    input_path: str
    output_path: str
    args: list[str] = field(default_factory=list)
    duration: float = 0.0
    has_audio: bool = False
    width: int = 0
    height: int = 0
    is_directory: bool = False
    # Opciones originales y datos con los que se construyó ``args``: el worker
    # los necesita para rehacer el comando cuando mide el loudness en dos
    # pasadas, ya fuera del hilo de la interfaz.
    options: dict = field(default_factory=dict)
    audio_track_count: int = 0
    # [(indice_de_salida, ruta, indice_de_pista)] de las pistas que el worker
    # tiene que medir antes del mux (añadidas y, opcionalmente, originales).
    normalize_sources: list[tuple[int, str, int]] = field(
        default_factory=list
    )


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _fmt_seconds(value: float) -> str:
    return f"{max(0.0, float(value)):.3f}".rstrip("0").rstrip(".") or "0"


def _video_encode_args() -> list[str]:
    return list(_VIDEO_ENCODE)


def _audio_encode_args() -> list[str]:
    return list(_AUDIO_ENCODE)


def _audio_track_entries(options: dict) -> list[dict]:
    """Normaliza ``options['tracks']`` a una lista de dicts validada."""
    raw = options.get("tracks")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Add at least one audio file first.")
    entries = []
    for item in raw:
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            raise ValueError("Invalid audio track entry.")
        path = str(item.get("path") or "").strip()
        if not path:
            raise ValueError("Every added track needs a file.")
        entries.append({
            "path": path,
            "language": str(item.get("language") or "").strip(),
            "title": str(item.get("title") or "").strip(),
        })
    return entries


def _loudnorm_for(measurements: dict, out_index: int) -> str:
    """Filtro de normalización de una pista de salida, con lo que se midió.

    Con medición usable se usa el loudnorm lineal de dos pasadas; sin ella
    (silencio digital, pista ilegible) cae a una pasada en vez de dejar la
    pista sin normalizar.
    """
    measured = measurements.get(out_index)
    if measured is None:
        measured = measurements.get(str(out_index))
    return build_boost_filter(
        {}, measured if measurement_is_usable(measured) else None
    )


def _multiaudio_container(options: dict) -> str:
    container = str(options.get("container") or "mkv").strip().lower()
    if container not in ("mkv", "mp4"):
        raise ValueError("The output container must be MKV or MP4.")
    return container


def _filtered_video_args(options: dict, *, has_audio: bool) -> list[str]:
    args = [*_video_encode_args(), *_MP4_FLAGS]
    if has_audio:
        args.extend(_audio_encode_args())
    return args


def build_tool_args(
    tool_id: str,
    input_path: str,
    output_path: str,
    options: dict | None = None,
    *,
    has_audio: bool = False,
    duration: float = 0.0,
    width: int = 0,
    height: int = 0,
    audio_track_count: int | None = None,
) -> list[str]:
    """Construye los argumentos de FFmpeg para ``tool_id``.

    Es una función pura (no ejecuta nada) para poder probar los filtros y las
    validaciones sin lanzar un render. ``output_path`` es el patrón ``%03d``
    cuando la herramienta es ``split``.
    """
    if tool_id not in TOOL_LABELS:
        raise ValueError(f"Unknown video tool: {tool_id}")
    if not input_path:
        raise ValueError("No source video selected.")
    if not output_path:
        raise ValueError("No output path selected.")
    options = dict(options or {})
    args = ["-hide_banner", "-nostdin", "-y"]

    if tool_id == "screenshot":
        stamp = _as_float(options.get("time"), 0.0)
        if stamp < 0:
            raise ValueError("The frame time cannot be negative.")
        if duration > 0 and stamp > duration:
            raise ValueError("The frame time is beyond the end of the video.")
        args.extend(["-ss", _fmt_seconds(stamp), "-i", input_path,
                     "-frames:v", "1", "-q:v", "2", output_path])
        return args

    if tool_id == "multiaudio":
        entries = _audio_track_entries(options)
        container = _multiaudio_container(options)
        normalize = bool(options.get("normalize"))
        measurements = options.get("measurements") or {}
        source_key = os.path.normcase(os.path.abspath(input_path))
        seen = set()
        for entry in entries:
            key = os.path.normcase(os.path.abspath(entry["path"]))
            if key == source_key:
                raise ValueError(
                    "The source video is already included; pick another file."
                )
            if key in seen:
                raise ValueError(
                    f"{os.path.basename(entry['path'])} was added twice."
                )
            seen.add(key)

        args.extend(["-i", input_path])
        for entry in entries:
            args.extend(["-i", entry["path"]])
        # El vídeo y las pistas que ya trae se copian; cada fichero añadido
        # aporta su primera pista de audio como una pista más.
        args.extend(["-map", "0:v:0"])
        if has_audio:
            args.extend(["-map", "0:a?"])
        # MKV admite subtítulos copiados; MP4 solo mov_text, así que allí se
        # dejan fuera en vez de hacer fallar el mux.
        if container == "mkv":
            args.extend(["-map", "0:s?", "-map", "0:t?"])
        for index in range(1, len(entries) + 1):
            args.extend(["-map", f"{index}:a:0"])
        base_tracks = (
            audio_track_count
            if audio_track_count is not None
            else (1 if has_audio else 0)
        )
        args.extend(["-map_metadata", "0", "-map_chapters", "0"])
        for offset, entry in enumerate(entries):
            out_index = base_tracks + offset
            if entry["language"]:
                args.extend([
                    f"-metadata:s:a:{out_index}",
                    f"language={entry['language']}",
                ])
            title = entry["title"] or os.path.splitext(
                os.path.basename(entry["path"])
            )[0]
            args.extend([f"-metadata:s:a:{out_index}", f"title={title}"])
            if container == "mp4":
                # En MP4 el nombre de pista visible en los reproductores sale
                # de handler_name; con solo title algunos lo muestran vacío.
                args.extend([
                    f"-metadata:s:a:{out_index}", f"handler_name={title}",
                ])
        args.extend(["-c", "copy"])
        # Un filtro no se puede aplicar a una pista copiada: las añadidas se
        # recodifican a AAC cuando se normaliza el volumen, y las que ya trae
        # el vídeo también si se pide nivelar todo el fichero. Si no, copiadas.
        if normalize and options.get("normalize_original"):
            for index in range(base_tracks):
                args.extend([
                    f"-c:a:{index}", "aac",
                    f"-b:a:{index}", "192k",
                    f"-ar:{index}", "48000",
                    f"-filter:a:{index}",
                    _loudnorm_for(measurements, index),
                ])
        if container == "mp4" or normalize:
            # Un dub en AC-3/DTS tampoco entra en MP4 sin reencodear; las
            # pistas originales (normalmente AAC) sí se copian tal cual.
            for offset in range(len(entries)):
                out_index = base_tracks + offset
                args.extend([
                    f"-c:a:{out_index}", "aac",
                    f"-b:a:{out_index}", "192k",
                    f"-ar:{out_index}", "48000",
                ])
                if normalize:
                    args.extend([
                        f"-filter:a:{out_index}",
                        _loudnorm_for(measurements, out_index),
                    ])
        if container == "mp4":
            args.extend(_MP4_FLAGS)
        args.append(output_path)
        return args

    args.extend(["-i", input_path])

    if tool_id == "pixelate":
        block = _as_int(options.get("block"), 16)
        if not 2 <= block <= 1024:
            raise ValueError("The pixel block size must be between 2 and 1024.")
        args.extend(["-vf", f"pixelize=w={block}:h={block}:mode=avg"])
        args.extend(_filtered_video_args(options, has_audio=has_audio))

    elif tool_id == "blur":
        sigma = _as_float(options.get("sigma"), 8.0)
        if not 0.1 <= sigma <= 100.0:
            raise ValueError("The blur strength must be between 0.1 and 100.")
        args.extend(["-vf", f"gblur=sigma={sigma:g}"])
        args.extend(_filtered_video_args(options, has_audio=has_audio))

    elif tool_id == "stabilize":
        args.extend(["-vf", "deshake=rx=16:ry=16:edge=mirror"])
        args.extend(_filtered_video_args(options, has_audio=has_audio))

    elif tool_id == "delogo":
        x = _as_int(options.get("x"), 10)
        y = _as_int(options.get("y"), 10)
        w = _as_int(options.get("w"), 120)
        h = _as_int(options.get("h"), 60)
        if x < 0 or y < 0:
            raise ValueError("The watermark position cannot be negative.")
        if w < 2 or h < 2:
            raise ValueError("The watermark rectangle must be at least 2 pixels.")
        if width > 0 and x + w > width:
            raise ValueError("The watermark rectangle is wider than the video.")
        if height > 0 and y + h > height:
            raise ValueError("The watermark rectangle is taller than the video.")
        args.extend(["-vf", f"delogo=x={x}:y={y}:w={w}:h={h}:show=0"])
        args.extend(_filtered_video_args(options, has_audio=has_audio))

    elif tool_id == "reverse":
        args.extend(["-vf", "reverse"])
        if has_audio:
            args.extend(["-af", "areverse"])
        args.extend(_filtered_video_args(options, has_audio=has_audio))

    elif tool_id == "volume":
        db = _as_float(options.get("db"), 0.0)
        if not -60.0 <= db <= 30.0:
            raise ValueError("The volume change must be between -60 and +30 dB.")
        if not has_audio:
            raise ValueError("This video has no audio track.")
        args.extend(["-af", f"volume={db:g}dB", "-c:v", "copy"])
        args.extend(_audio_encode_args())
        args.extend(_MP4_FLAGS)

    elif tool_id == "split":
        parts = _as_int(options.get("parts"), 2)
        if parts < 2 or parts > 50:
            raise ValueError("The number of parts must be between 2 and 50.")
        if duration <= 0:
            raise ValueError("Could not read the video duration.")
        if "%" not in output_path:
            raise ValueError("The split output must use a numbered filename pattern.")
        segment = duration / parts
        segment_text = f"{segment:.3f}"
        args.extend([
            *_video_encode_args(),
            "-force_key_frames", f"expr:gte(t,n_forced*{segment_text})",
            *_audio_encode_args(),
            "-f", "segment",
            "-segment_time", segment_text,
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            output_path,
        ])
        return args

    args.append(output_path)
    return args


def _unique_file(output_dir: str, stem: str, suffix: str, extension: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    attempt = 1
    while True:
        tail = "" if attempt == 1 else f"_{attempt}"
        candidate = os.path.join(output_dir, f"{stem}{suffix}{tail}{extension}")
        if not os.path.exists(candidate):
            return candidate
        attempt += 1


def _unique_dir(output_dir: str, stem: str, suffix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    attempt = 1
    while True:
        tail = "" if attempt == 1 else f"_{attempt}"
        candidate = os.path.join(output_dir, f"{stem}{suffix}{tail}")
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        attempt += 1


def _source_stem(input_path: str) -> str:
    stem = os.path.splitext(os.path.basename(input_path or ""))[0]
    return stem or "video"


def plan_tool(
    ffmpeg_path: str,
    tool_id: str,
    input_path: str,
    output_dir: str,
    options: dict | None = None,
) -> VideoToolPlan:
    """Inspecciona la fuente, elige un nombre libre y prepara el render."""
    if not os.path.isfile(input_path):
        raise ValueError("The selected video no longer exists.")
    if not output_dir:
        raise ValueError("Choose an output folder.")

    info = get_video_info(ffmpeg_path, input_path) or {}
    duration = max(0.0, _as_float(info.get("duration"), 0.0))
    width = max(0, _as_int(info.get("width"), 0))
    height = max(0, _as_int(info.get("height"), 0))
    audio_streams = get_audio_streams(ffmpeg_path, input_path)
    has_audio = bool(audio_streams)
    options = dict(options or {})
    stem = _source_stem(input_path)
    suffix = TOOL_SUFFIXES[tool_id]
    normalize_sources: list[tuple[int, str]] = []

    if tool_id == "screenshot":
        extension = ".jpg" if str(options.get("format", "png")).lower() in (
            "jpg", "jpeg",
        ) else ".png"
        output_path = _unique_file(output_dir, stem, suffix, extension)
        is_directory = False
    elif tool_id == "multiaudio":
        container = _multiaudio_container(options)
        entries = _audio_track_entries(options)
        for entry in entries:
            if not os.path.isfile(entry["path"]):
                raise ValueError(
                    f"{os.path.basename(entry['path'])} no longer exists."
                )
            if not get_audio_streams(ffmpeg_path, entry["path"]):
                raise ValueError(
                    f"{os.path.basename(entry['path'])} has no audio track."
                )
        if options.get("normalize"):
            # El índice es el de la pista en la SALIDA: las que ya trae el
            # vídeo van delante, y el worker mide cada pista para rehacer el
            # comando con su loudness medido. La tercera columna es la pista
            # de audio del fichero que hay que medir.
            base_count = len(audio_streams)
            if options.get("normalize_original"):
                for index in range(base_count):
                    normalize_sources.append((index, input_path, index))
            for offset, entry in enumerate(entries):
                normalize_sources.append(
                    (base_count + offset, entry["path"], 0)
                )
        output_path = _unique_file(
            output_dir, stem, suffix, ".mkv" if container == "mkv" else ".mp4"
        )
        is_directory = False
    elif tool_id == "split":
        output_path = _unique_dir(output_dir, stem, suffix)
        output_path = os.path.join(output_path, "part_%03d.mp4")
        is_directory = True
    else:
        output_path = _unique_file(output_dir, stem, suffix, ".mp4")
        is_directory = False

    args = build_tool_args(
        tool_id,
        input_path,
        output_path,
        options,
        has_audio=has_audio,
        duration=duration,
        width=width,
        height=height,
        audio_track_count=len(audio_streams),
    )
    return VideoToolPlan(
        tool_id=tool_id,
        input_path=input_path,
        output_path=output_path,
        args=args,
        duration=duration,
        has_audio=has_audio,
        width=width,
        height=height,
        is_directory=is_directory,
        options=options,
        audio_track_count=len(audio_streams),
        normalize_sources=normalize_sources,
    )
