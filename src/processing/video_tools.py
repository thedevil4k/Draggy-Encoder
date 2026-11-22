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
import re
from dataclasses import dataclass, field

from src.ffmpeg.runner import (
    build_boost_filter,
    get_audio_streams,
    get_subtitle_streams,
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
                   "first: the audio track you pick is copied straight in, "
                   "together with the subtitles that file carries. Each "
                   "added track can be loudness-normalised (EBU R128) so it "
                   "matches the volume of the one already in the video, and "
                   "carries its own delay when the dubs do not start together.",
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

# Extensiones que se consideran "episodio" al añadir una carpeta en lote. El
# audio suelto queda fuera: en una carpeta de serie, un .mka/.mp3 es un
# doblaje, no un vídeo al que añadirle pistas.
VIDEO_EXTENSIONS = (
    ".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts", ".m2ts",
    ".mpg", ".mpeg", ".wmv", ".flv",
)

# Extensiones que puede aportar un doblaje (el vídeo entero también vale).
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + (
    ".mka", ".m4a", ".aac", ".mp3", ".ac3", ".eac3", ".dts", ".flac",
    ".wav", ".opus", ".ogg",
)

# Desfase máximo que se admite por pista. Más que esto no es un desfase, es
# otro montaje del episodio.
MAX_DELAY_SECONDS = 30.0

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
    # [(indice_en_tracks, ruta, pista)] de los doblajes cuyo desfase hay que
    # medir comparando su audio con el del vídeo, antes del comando.
    delay_sources: list[tuple[int, str, int]] = field(default_factory=list)


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
        try:
            stream = int(item.get("stream") or 0)
        except (TypeError, ValueError):
            raise ValueError(
                "The audio track index of the added file is invalid."
            ) from None
        if not 0 <= stream <= 63:
            raise ValueError("The audio track index of the added file is invalid.")
        try:
            delay = float(item.get("delay") or 0.0)
        except (TypeError, ValueError):
            raise ValueError("The delay must be a number of seconds.") from None
        if not -MAX_DELAY_SECONDS <= delay <= MAX_DELAY_SECONDS:
            raise ValueError(
                f"The delay must be between -{MAX_DELAY_SECONDS:g} and "
                f"{MAX_DELAY_SECONDS:g} seconds."
            )
        entries.append({
            "path": path,
            "stream": stream,
            "language": str(item.get("language") or "").strip(),
            "title": str(item.get("title") or "").strip(),
            "delay": delay,
        })
    return entries


# Subtítulos que MP4 acepta: solo texto, y convertidos a mov_text. Un PGS o un
# subtítulo de DVD son imagen y harían fallar el mux entero.
_TEXT_SUBTITLE_CODECS = frozenset({
    "subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "eia_608",
})


def subtitle_is_text(stream: dict) -> bool:
    """¿Este subtítulo se puede meter en un MP4 como ``mov_text``?"""
    codec = str((stream or {}).get("codec_name") or "").lower()
    return codec in _TEXT_SUBTITLE_CODECS


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
    subtitle_streams: list[list[dict]] | None = None,
    base_subtitle_count: int = 0,
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
        for index, entry in enumerate(entries, start=1):
            args.extend(["-map", f"{index}:a:{entry['stream']}"])
        # -c copy va antes de los codecs por pista de los subtítulos: si fuera
        # después, el copiado genérico pisaría el mov_text y FFmpeg intentaría
        # meter un subrip en MP4.
        args.extend(["-c", "copy"])
        # Subtítulos del fichero añadido, los que traiga. En MKV se copian tal
        # cual (SRT, ASS, PGS); en MP4 solo entran los de texto y convertidos a
        # mov_text, que es lo único que el contenedor acepta.
        added_subs = subtitle_streams or []
        sub_out = base_subtitle_count if container == "mkv" else 0
        for offset in range(len(entries)):
            streams = added_subs[offset] if offset < len(added_subs) else []
            for stream in streams:
                if container == "mp4" and not subtitle_is_text(stream):
                    continue
                args.extend([
                    "-map", f"{offset + 1}:s:{stream['subtitle_index']}",
                ])
                if container == "mp4":
                    args.extend([f"-c:s:{sub_out}", "mov_text"])
                language = str(stream.get("language") or "")
                if language and language != "und":
                    args.extend([
                        f"-metadata:s:s:{sub_out}", f"language={language}",
                    ])
                title = str(stream.get("title") or "")
                if title:
                    args.extend([f"-metadata:s:s:{sub_out}", f"title={title}"])
                sub_out += 1
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
        # Un filtro no se puede aplicar a una pista copiada: las añadidas se
        # recodifican a AAC cuando se normaliza el volumen o se les pone
        # desfase, y las que ya trae el vídeo también si se pide nivelar todo
        # el fichero. Si no, copiadas.
        if normalize and options.get("normalize_original"):
            for index in range(base_tracks):
                args.extend([
                    f"-c:a:{index}", "aac",
                    f"-b:a:{index}", "192k",
                    f"-ar:{index}", "48000",
                    f"-filter:a:{index}",
                    _loudnorm_for(measurements, index),
                ])
        has_delay = any(abs(entry["delay"]) > 1e-9 for entry in entries)
        if container == "mp4" or normalize or has_delay:
            # Un dub en AC-3/DTS tampoco entra en MP4 sin reencodear; las
            # pistas originales (normalmente AAC) sí se copian tal cual.
            for offset, entry in enumerate(entries):
                out_index = base_tracks + offset
                args.extend([
                    f"-c:a:{out_index}", "aac",
                    f"-b:a:{out_index}", "192k",
                    f"-ar:{out_index}", "48000",
                ])
                # El desfase va como filtro, no como -itsoffset: con -c copy
                # el muxer de MKV recoloca los timestamps negativos y el
                # audio vuelve a ir tarde (comprobado con un episodio real).
                filters = []
                if entry["delay"] > 0:
                    filters.append(
                        f"adelay={int(round(entry['delay'] * 1000))}:all=1"
                    )
                elif entry["delay"] < 0:
                    filters.append(
                        f"atrim=start={abs(entry['delay']):.3f},"
                        f"asetpts=PTS-STARTPTS"
                    )
                if normalize:
                    filters.append(_loudnorm_for(measurements, out_index))
                if filters:
                    args.extend([
                        f"-filter:a:{out_index}", ",".join(filters),
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


# ---------------------------------------------------------------------------
# Emparejado de episodios con su doblaje (modo lote)
# ---------------------------------------------------------------------------
# El código de episodio es lo primero que se mira: dos rips del mismo capítulo
# casi nunca se llaman igual ("Family.Guy.S01E03.1080p" y
# "Padre.de.familia.S01E03.Castellano"), pero los dos llevan S01E03.
_EPISODE_SXXEXX = re.compile(
    r"(?:^|[^0-9a-z])s(\d{1,2})[ ._-]*e(\d{1,3})(?![0-9])", re.IGNORECASE
)
_EPISODE_NXNN = re.compile(r"(?:^|[^0-9])(\d{1,2})x(\d{1,3})(?![0-9])")

# Tokens que no distinguen un episodio de su doblaje: idioma del rip y ruido
# de release. Se quitan para comparar nombres cuando no hay código de episodio.
_NOISE_TOKENS = frozenset({
    "eng", "english", "ing", "spa", "spanish", "esp", "es", "cast",
    "castellano", "lat", "latino", "vose", "vos", "vo", "multi", "dual",
    "dub", "dubbed", "doblaje", "sub", "subs", "subbed", "espanol",
    "1080p", "720p", "2160p", "480p", "576p", "4k", "x264", "x265",
    "h264", "h265", "hevc", "avc", "bluray", "bdrip", "brrip", "webrip",
    "web", "webdl", "dl", "hdtv", "dvdrip", "aac", "ac3", "eac3", "dts",
    "ddp", "10bit", "hdr", "proper", "repack", "remux",
})


def episode_key(name: str) -> tuple[int, int] | None:
    """(temporada, episodio) si el nombre lo lleva, o ``None``.

    Acepta ``S01E03`` (con puntos, espacios o guiones en medio) y ``1x03``.
    """
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    match = _EPISODE_SXXEXX.search(stem) or _EPISODE_NXNN.search(stem)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def normalized_stem(name: str) -> str:
    """Nombre sin extensión, en minúsculas y sin tokens de idioma/calidad."""
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    text = re.sub(r"[^0-9a-z]+", " ", stem.lower())
    return " ".join(
        token for token in text.split() if token and token not in _NOISE_TOKENS
    )


def media_files(folder: str, extensions=MEDIA_EXTENSIONS) -> list[str]:
    """Ficheros de ``folder`` con esas extensiones, ordenados y sin carpetas."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    found = [
        os.path.join(folder, name) for name in names
        if name.lower().endswith(tuple(extensions))
    ]
    return sorted(path for path in found if os.path.isfile(path))


def match_dub(episode_path: str, dub_paths: list[str]) -> dict:
    """Busca el doblaje de ``episode_path`` dentro de ``dub_paths``.

    Devuelve ``{"dub": ruta o "", "status": texto}``. Nunca adivina: dos
    candidatos para el mismo capítulo (castellano y latino, por ejemplo) salen
    como ambiguos en vez de elegir uno al azar y producir un mux equivocado.
    """
    episode_abs = os.path.normcase(os.path.abspath(episode_path))
    candidates = [
        path for path in dub_paths
        if os.path.isfile(path)
        and os.path.normcase(os.path.abspath(path)) != episode_abs
    ]
    key = episode_key(episode_path)
    if key is not None:
        same_key = [path for path in candidates if episode_key(path) == key]
        if len(same_key) == 1:
            return {"dub": same_key[0], "status": "Matched (S%02dE%02d)" % key}
        if len(same_key) > 1:
            stem = normalized_stem(episode_path)
            same_stem = [p for p in same_key if normalized_stem(p) == stem]
            if stem and len(same_stem) == 1:
                return {
                    "dub": same_stem[0], "status": "Matched (S%02dE%02d)" % key,
                }
            return {
                "dub": "", "status": f"Ambiguous ({len(same_key)} files)",
            }
    stem = normalized_stem(episode_path)
    if stem:
        same_stem = [p for p in candidates if normalized_stem(p) == stem]
        if len(same_stem) == 1:
            return {"dub": same_stem[0], "status": "Matched (name)"}
        if len(same_stem) > 1:
            return {
                "dub": "", "status": f"Ambiguous ({len(same_stem)} files)",
            }
    return {"dub": "", "status": "No dub found"}


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
    normalize_sources: list[tuple[int, str, int]] = []
    delay_sources: list[tuple[int, str, int]] = []
    subtitle_streams: list[list[dict]] = []
    base_subtitle_count = 0

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
            name = os.path.basename(entry["path"])
            if not os.path.isfile(entry["path"]):
                raise ValueError(f"{name} no longer exists.")
            extra_audio = get_audio_streams(ffmpeg_path, entry["path"])
            if not extra_audio:
                raise ValueError(f"{name} has no audio track.")
            if entry["stream"] >= len(extra_audio):
                raise ValueError(
                    f"{name} has no audio track #{entry['stream'] + 1}."
                )
            subtitle_streams.append(
                get_subtitle_streams(ffmpeg_path, entry["path"])
            )
        if container == "mkv":
            # Los subtítulos del propio vídeo se copian, así que las pistas
            # añadidas empiezan después de los suyos.
            base_subtitle_count = len(
                get_subtitle_streams(ffmpeg_path, input_path)
            )
        if options.get("auto_delay"):
            # El desfase se mide en el worker (decodifica audio): aquí solo se
            # anota qué pista de qué doblaje hay que comparar con el vídeo.
            for index, entry in enumerate(entries):
                delay_sources.append((index, entry["path"], entry["stream"]))
        if options.get("normalize"):
            # El índice es el de la pista en la SALIDA: las que ya trae el
            # vídeo van delante, y el worker mide cada pista para rehacer el
            # comando con su loudness medido. La última columna es la pista de
            # audio del fichero que hay que medir.
            base_count = len(audio_streams)
            if options.get("normalize_original"):
                for index in range(base_count):
                    normalize_sources.append((index, input_path, index))
            for offset, entry in enumerate(entries):
                normalize_sources.append(
                    (base_count + offset, entry["path"], entry["stream"])
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
        subtitle_streams=subtitle_streams,
        base_subtitle_count=base_subtitle_count,
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
        delay_sources=delay_sources,
    )
