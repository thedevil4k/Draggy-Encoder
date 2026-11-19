"""Timeline de unión de vídeos (Join videos).

Una "timeline" es una lista ordenada de ``Clip``: cada clip apunta a un
archivo y lleva su propio recorte ``start``/``end``. El pipeline no ve N
vídeos: ve una sola fuente con geometría, fps y duración efectivas, igual
que si el usuario hubiera importado un único archivo ya montado.

El grafo FFmpeg que genera este módulo hace, por clip:

  * ``-ss start -t duration -i archivo``  (seek de entrada, rápido y exacto)
  * ``setpts`` + ``fps`` + ``scale``/``pad`` + ``setsar`` para normalizar
  * ``asetpts`` + ``aresample`` + ``aformat`` + ``apad`` para audio
  * un ``anullsrc`` de la misma duración si el clip no tiene audio
  * ``concat=n=N:v=1:a=1`` al final

Así el resto del pipeline (IA, escala, encoder, audio boost) trabaja sobre
un único flujo homogéneo sin duplicar código.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.ffmpeg.runner import get_audio_streams, get_video_info


_FPS_FALLBACK = 30.0
_MIN_CLIP_SECONDS = 0.05


def parse_seconds(value: Any, default: float = 0.0) -> float:
    """Convierte ``12.5`` o ``"00:00:12.500"`` a segundos.

    Tolera vacío/None (devuelve ``default``) y basura (también ``default``):
    la validación posterior es la que rechaza un rango incoherente.
    """
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return default
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return default
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def _ratio_to_float(expr: str, default: float = _FPS_FALLBACK) -> float:
    text = str(expr or "").strip()
    if not text or text in ("0/0", "0/1", "N/A"):
        return default
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            num_f, den_f = float(num), float(den)
            if den_f:
                return num_f / den_f
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def _even(value: int) -> int:
    value = max(2, int(value))
    return value + (value % 2)


@dataclass
class Clip:
    """Un vídeo de la timeline con su recorte y los datos del sondeo."""

    path: str
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = _FPS_FALLBACK
    fps_expr: str = "30/1"
    sar: str = "1:1"
    rotation: int = 0
    has_audio: bool = False

    @property
    def trim_duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_payload(self) -> dict:
        """Forma JSON que viaja en settings, cola y last_job."""
        return {
            "path": self.path,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "duration": round(float(self.duration), 3),
            "width": int(self.width or 0),
            "height": int(self.height or 0),
            "fps": round(float(self.fps or _FPS_FALLBACK), 6),
            "fps_expr": self.fps_expr or "30/1",
            "sar": self.sar or "1:1",
            "rotation": int(self.rotation or 0),
            "has_audio": bool(self.has_audio),
        }


@dataclass
class JoinGraph:
    """Argumentos de entrada + grafo de filtros listos para FFmpeg."""

    inputs: list[str]
    filter_complex: str
    width: int
    height: int
    fps: float
    fps_expr: str
    duration: float
    has_audio: bool

    @property
    def video_label(self) -> str:
        return "[vout]"

    @property
    def audio_label(self) -> str:
        return "[aout]" if self.has_audio else ""


def probe_clip_payload(ffmpeg_path: str, payload: dict) -> Clip:
    """Completa un payload de la UI con los datos reales del archivo."""
    path = str((payload or {}).get("path") or "")
    info = get_video_info(ffmpeg_path, path) if path else {}
    try:
        duration = float(info.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        width = int(float(info.get("width") or 0))
        height = int(float(info.get("height") or 0))
    except (TypeError, ValueError):
        width = height = 0
    fps_expr = str(info.get("r_frame_rate") or "30/1")
    fps = _ratio_to_float(fps_expr)
    try:
        rotation = int(float(str(info.get("rotation") or 0).strip() or 0))
    except (TypeError, ValueError):
        rotation = 0
    has_audio = bool(get_audio_streams(ffmpeg_path, path)) if path else False

    start = parse_seconds(payload.get("start"), 0.0)
    raw_end = payload.get("end")
    end = parse_seconds(raw_end, duration) if raw_end not in (None, "") else duration
    if duration > 0:
        start = max(0.0, min(start, max(0.0, duration - _MIN_CLIP_SECONDS)))
        end = max(0.0, min(end, duration))
    else:
        start = max(0.0, start)
        end = max(start, end)

    return Clip(
        path=path,
        start=start,
        end=end,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        fps_expr=fps_expr if fps_expr not in ("", "0/0", "N/A") else "30/1",
        sar=str(info.get("sample_aspect_ratio") or "1:1"),
        rotation=rotation,
        has_audio=has_audio,
    )


def probe_clips(ffmpeg_path: str, payloads: Iterable[dict]) -> list[Clip]:
    return [probe_clip_payload(ffmpeg_path, p) for p in (payloads or [])]


def clips_from_settings(settings: dict | None) -> list[Clip]:
    """Reconstruye la timeline desde settings/cola (JSON simple)."""
    raw = (settings or {}).get("join_clips")
    if not raw:
        return []
    clips: list[Clip] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clips.append(
            Clip(
                path=str(item.get("path") or ""),
                start=parse_seconds(item.get("start"), 0.0),
                end=parse_seconds(item.get("end"), 0.0),
                duration=parse_seconds(item.get("duration"), 0.0),
                width=int(item.get("width") or 0),
                height=int(item.get("height") or 0),
                fps=float(item.get("fps") or _FPS_FALLBACK),
                fps_expr=str(item.get("fps_expr") or "30/1"),
                sar=str(item.get("sar") or "1:1"),
                rotation=int(item.get("rotation") or 0),
                has_audio=bool(item.get("has_audio")),
            )
        )
    return clips


def validate_clips(clips: Sequence[Clip]) -> list[str]:
    """Errores accionables; lista vacía = timeline válida."""
    errors: list[str] = []
    if not clips:
        return ["No videos in the join timeline."]
    for index, clip in enumerate(clips, 1):
        label = f"Clip {index} ({os.path.basename(clip.path) or '?'})"
        if not clip.path or not os.path.exists(clip.path):
            errors.append(f"{label}: file not found.")
            continue
        if clip.duration <= 0:
            errors.append(f"{label}: could not read its duration.")
            continue
        if clip.end <= clip.start:
            errors.append(f"{label}: trim end must be greater than trim start.")
            continue
        if clip.trim_duration < max(_MIN_CLIP_SECONDS, 1.0 / max(1.0, clip.fps)):
            errors.append(f"{label}: trim is shorter than one frame.")
    return errors


def join_duration(clips: Sequence[Clip]) -> float:
    return round(sum(c.trim_duration for c in clips), 6)


def target_geometry(clips: Sequence[Clip], settings: dict | None = None) -> tuple[int, int, float, str]:
    """Tamaño y fps comunes para todos los clips.

    Si el usuario eligió una resolución de export (720p, 1080p...), esa manda
    en alto y el ancho se toma del clip más ancho para no recortar ninguno.
    Sin resolución, se usa el mayor lienzo de la timeline. Siempre par.
    """
    settings = settings or {}
    clips = [c for c in clips if c.width > 0 and c.height > 0]
    if not clips:
        return 1280, 720, _FPS_FALLBACK, "30/1"

    raw = str(settings.get("resolution") or "original")
    export_h = None
    if raw.isdigit() and int(raw) > 0:
        export_h = int(raw)

    fps = max((c.fps or _FPS_FALLBACK) for c in clips)
    fps_clip = max(clips, key=lambda c: c.fps or _FPS_FALLBACK)
    fps_expr = fps_clip.fps_expr if abs((fps_clip.fps or 0) - fps) < 1e-6 else f"{fps:.6f}"

    if export_h:
        width = max(round(c.width * (export_h / c.height)) for c in clips)
        height = export_h
    else:
        width = max(c.width for c in clips)
        height = max(c.height for c in clips)
    return _even(width), _even(height), fps, fps_expr


def _clip_input_args(clip: Clip) -> list[str]:
    duration = max(_MIN_CLIP_SECONDS, clip.trim_duration)
    return [
        "-ss", f"{clip.start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", clip.path,
    ]


def build_join_graph(clips: Sequence[Clip], settings: dict | None = None,
                     video_only: bool = False,
                     audio_only: bool = False) -> JoinGraph:
    """Construye ``-i`` por clip y el ``-filter_complex`` de unión.

    ``video_only`` / ``audio_only`` evitan dejar salidas sin conectar: el
    decoder de IA solo mapea vídeo y el encode de audio solo mapea audio, y
    FFmpeg rechaza un ``filter_complex`` con etiquetas que nadie consume.
    """
    settings = settings or {}
    clips = list(clips)
    errors = validate_clips(clips)
    if errors:
        raise ValueError(" ".join(errors))
    if video_only and audio_only:
        raise ValueError("video_only and audio_only are mutually exclusive")

    width, height, fps, fps_expr = target_geometry(clips, settings)
    audio_enabled = (
        not video_only
        and str(settings.get("audio", "copy") or "copy") != "noaudio"
    )
    video_enabled = not audio_only

    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    input_index = 0

    for i, clip in enumerate(clips):
        duration = max(_MIN_CLIP_SECONDS, clip.trim_duration)
        inputs.extend(_clip_input_args(clip))
        vidx = input_index
        input_index += 1

        if video_enabled:
            chains.append(
                f"[{vidx}:v]setpts=PTS-STARTPTS,"
                f"fps={fps_expr},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,format=yuv420p[v{i}]"
            )
            labels.append(f"[v{i}]")

        if not audio_enabled:
            continue
        if clip.has_audio:
            aidx = vidx
        else:
            inputs.extend([
                "-f", "lavfi",
                "-t", f"{duration:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            ])
            aidx = input_index
            input_index += 1
        chains.append(
            f"[{aidx}:a]asetpts=PTS-STARTPTS,"
            f"aresample=48000,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"apad=whole_dur={duration:.3f},"
            f"atrim=0:{duration:.3f}[a{i}]"
        )
        labels.append(f"[a{i}]")

    concat_inputs = "".join(labels)
    if audio_enabled and video_enabled:
        chains.append(
            f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[vout][aout0]"
        )
        # El boost va DESPUÉS del concat: una sola medición para toda la
        # timeline y un nivel coherente entre clips.
        if bool(settings.get("audio_boost")):
            from src.ffmpeg.runner import build_boost_filter

            boost = build_boost_filter(dict(settings), None)
            if boost:
                chains.append(f"[aout0]{boost}[aout]")
            else:
                chains.append("[aout0]anull[aout]")
        else:
            chains.append("[aout0]anull[aout]")
    elif audio_enabled:
        # Audio-only (encode de audio en paralelo): sin vídeo que mapear.
        chains.append(
            f"{concat_inputs}concat=n={len(clips)}:v=0:a=1[aout0]"
        )
        if bool(settings.get("audio_boost")):
            from src.ffmpeg.runner import build_boost_filter

            boost = build_boost_filter(dict(settings), None)
            chains.append(
                f"[aout0]{boost}[aout]" if boost else "[aout0]anull[aout]"
            )
        else:
            chains.append("[aout0]anull[aout]")
    else:
        chains.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=0[vout]")

    return JoinGraph(
        inputs=inputs,
        filter_complex=";".join(chains),
        width=width,
        height=height,
        fps=fps,
        fps_expr=fps_expr,
        duration=join_duration(clips),
        has_audio=audio_enabled,
    )


def select_preview_clip(clips: Sequence[Clip], seconds: float | None) -> tuple[Clip, float]:
    """Clip y offset local para un instante ABSOLUTO de la timeline."""
    if not clips:
        raise ValueError("No videos in the join timeline.")
    if seconds is None or seconds <= 0:
        return clips[0], min(0.5, max(0.0, clips[0].trim_duration / 2))
    remaining = float(seconds)
    for clip in clips:
        duration = clip.trim_duration
        if remaining < duration or clip is clips[-1]:
            offset = min(max(0.0, remaining), max(0.0, duration - 0.05))
            return clip, offset
        remaining -= duration
    return clips[-1], max(0.0, clips[-1].trim_duration - 0.05)


def timeline_name(clips: Sequence[Clip], limit: int = 60) -> str:
    if not clips:
        return "Joined"
    stem = os.path.splitext(os.path.basename(clips[0].path))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "Joined"
    return f"Joined_{stem}_{len(clips)}clips"[:limit]
