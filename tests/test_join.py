import os

import pytest

from src.ffmpeg.runner import build_join_args
from src.processing.join import (
    Clip,
    JoinGraph,
    build_join_graph,
    clips_from_settings,
    join_duration,
    parse_seconds,
    select_preview_clip,
    target_geometry,
    validate_clips,
)


def _clip(path, start=0.0, end=5.0, width=1280, height=720, fps=30.0,
          has_audio=True, duration=10.0):
    return Clip(
        path=str(path), start=start, end=end, duration=duration,
        width=width, height=height, fps=fps, fps_expr=f"{int(fps)}/1",
        has_audio=has_audio,
    )


def test_parse_seconds_accepts_numbers_and_timecodes():
    assert parse_seconds(12.5) == 12.5
    assert parse_seconds("12.5") == 12.5
    assert parse_seconds("01:30") == 90
    assert parse_seconds("00:01:30.500") == 90.5
    assert parse_seconds("") == 0
    assert parse_seconds("garbage", 7) == 7


def test_join_duration_sums_trimmed_ranges():
    clips = [_clip("a.mp4", 1, 4), _clip("b.mp4", 0, 2.5)]
    assert join_duration(clips) == pytest.approx(5.5)


def test_target_geometry_uses_export_height_and_widest_clip():
    clips = [_clip("a.mp4", width=1920, height=1080), _clip("b.mp4", width=1280, height=720)]
    width, height, fps, _ = target_geometry(clips, {"resolution": "720"})
    assert height == 720
    assert width == 1280
    assert fps == 30


def test_target_geometry_original_uses_largest_canvas():
    clips = [_clip("a.mp4", width=1281, height=721)]
    width, height, _, _ = target_geometry(clips, {"resolution": "original"})
    assert width % 2 == 0 and height % 2 == 0
    assert width >= 1281 and height >= 721


def test_build_join_graph_normalizes_and_adds_silence(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    clips = [_clip(str(a), 1, 4, width=1920, height=1080),
             _clip(str(b), 0, 2, width=1280, height=720, has_audio=False)]

    graph = build_join_graph(clips, {"resolution": "720", "audio": "copy"})
    assert isinstance(graph, JoinGraph)
    assert graph.width == 1280 and graph.height == 720
    assert "concat=n=2:v=1:a=1" in graph.filter_complex
    assert "anullsrc" in " ".join(graph.inputs)
    assert graph.filter_complex.count("scale=") == 2
    assert "apad=whole_dur=" in graph.filter_complex
    assert graph.duration == pytest.approx(5.0)


def test_build_join_graph_video_only_has_no_audio_labels(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    graph = build_join_graph([_clip(str(a))], {"audio": "copy"}, video_only=True)
    assert "[a" not in graph.filter_complex
    assert "concat=n=1:v=1:a=0" in graph.filter_complex
    assert graph.has_audio is False


def test_build_join_graph_audio_only_has_no_video_labels(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    graph = build_join_graph([_clip(str(a))], {"audio": "copy"}, audio_only=True)
    assert "[v" not in graph.filter_complex
    assert "concat=n=1:v=0:a=1" in graph.filter_complex
    assert graph.has_audio is True


def test_build_join_graph_respects_noaudio(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    graph = build_join_graph([_clip(str(a))], {"audio": "noaudio"})
    assert "concat=n=1:v=1:a=0" in graph.filter_complex
    assert "anullsrc" not in " ".join(graph.inputs)
    assert graph.has_audio is False


def test_build_join_graph_puts_loudnorm_after_concat(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    graph = build_join_graph([_clip(str(a))], {"audio": "copy", "audio_boost": True})
    assert graph.filter_complex.index("concat=") < graph.filter_complex.index("loudnorm")


def test_validate_clips_reports_missing_file_and_bad_range(tmp_path):
    missing = _clip(str(tmp_path / "nope.mp4"))
    bad = _clip(str(tmp_path / "bad.mp4"), 5, 2)
    (tmp_path / "bad.mp4").write_bytes(b"x")
    errors = validate_clips([missing, bad])
    assert any("file not found" in e for e in errors)
    assert any("greater than trim start" in e for e in errors)


def test_select_preview_clip_maps_absolute_time(tmp_path):
    a = _clip(str(tmp_path / "a.mp4"), 10, 14)   # 4 s
    b = _clip(str(tmp_path / "b.mp4"), 0, 3)
    clip, offset = select_preview_clip([a, b], 5.0)
    assert clip is b
    assert offset == pytest.approx(1.0)
    clip, offset = select_preview_clip([a, b], 0)
    assert clip is a and offset >= 0


def test_clips_from_settings_roundtrip(tmp_path):
    c = _clip(str(tmp_path / "a.mp4"), 1, 3)
    restored = clips_from_settings({"join_clips": [c.to_payload()]})
    assert len(restored) == 1
    assert restored[0].path == c.path
    assert restored[0].start == 1 and restored[0].end == 3


def test_build_join_args_maps_video_audio_and_nvenc(tmp_path):
    graph = JoinGraph(
        inputs=["-i", "a.mp4"], filter_complex="[vout]", width=1280, height=720,
        fps=30.0, fps_expr="30/1", duration=10.0, has_audio=True,
    )
    out = str(tmp_path / "out.mp4")
    args = build_join_args(graph, out, {
        "codec": "hevc_nvenc", "bitrate": 1200, "audio": "copy",
        "format": "mp4", "gpu_fast": True,
    })
    assert args[-1] == out
    assert "-filter_complex" in args
    assert args[args.index("-map")] == "-map"
    assert "[aout]" in args
    assert "aac" in args
    assert "-an" not in args
    assert "p1" in args
    assert args.count("-i") == 1


def test_build_join_args_noaudio(tmp_path):
    graph = JoinGraph(
        inputs=[], filter_complex="[vout]", width=640, height=360,
        fps=25.0, fps_expr="25/1", duration=2.0, has_audio=False,
    )
    args = build_join_args(graph, str(tmp_path / "o.mp4"), {"audio": "copy"})
    assert "-an" in args
    assert "[aout]" not in args
