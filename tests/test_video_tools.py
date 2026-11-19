"""Pruebas de las herramientas FFmpeg de un solo uso (sin lanzar FFmpeg)."""

import os

import pytest

from src.processing import video_tools


def _args(tool_id, output="out.mp4", options=None, **kwargs):
    return video_tools.build_tool_args(
        tool_id, "input.mp4", output, options or {}, **kwargs
    )


def test_pixelate_builds_filter_and_audio():
    args = _args("pixelate", options={"block": 24}, has_audio=True)
    assert "pixelize=w=24:h=24:mode=avg" in args
    assert "libx264" in args
    assert "aac" in args
    assert args[-1] == "out.mp4"


def test_screenshot_seeks_and_writes_one_frame():
    args = _args(
        "screenshot",
        output="frame.png",
        options={"time": 1.25},
    )
    assert args[:3] == ["-hide_banner", "-nostdin", "-y"]
    assert args[args.index("-ss") + 1] == "1.25"
    assert args[args.index("-frames:v") + 1] == "1"
    assert args[-1] == "frame.png"


def test_screenshot_rejects_time_after_end():
    with pytest.raises(ValueError, match="beyond"):
        _args("screenshot", options={"time": 5}, duration=4)


def test_reverse_only_reverses_audio_when_present():
    with_audio = _args("reverse", has_audio=True)
    without_audio = _args("reverse", has_audio=False)
    assert "areverse" in with_audio
    assert "areverse" not in without_audio


def test_volume_requires_audio_and_uses_audio_only_encode():
    with pytest.raises(ValueError, match="no audio"):
        _args("volume", options={"db": 3}, has_audio=False)
    args = _args("volume", options={"db": 3}, has_audio=True)
    assert "volume=3dB" in args
    assert "copy" in args
    assert "aac" in args


def test_delogo_validates_rectangle_against_source():
    with pytest.raises(ValueError, match="wider"):
        _args("delogo", options={"x": 0, "y": 0, "w": 500, "h": 20},
              width=320, height=240)
    args = _args("delogo", options={"x": 10, "y": 10, "w": 100, "h": 50},
                 width=320, height=240)
    assert "delogo=x=10:y=10:w=100:h=50:show=0" in args


def test_split_uses_equal_segment_time_and_pattern():
    args = _args(
        "split",
        output="parts/part_%03d.mp4",
        options={"parts": 4},
        duration=10,
    )
    assert "10.000" not in args
    assert args[args.index("-segment_time") + 1] == "2.500"
    assert args[-1] == "parts/part_%03d.mp4"
    assert "-f" in args


def test_plan_creates_unique_output_and_reads_metadata(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"")
    monkeypatch.setattr(video_tools, "get_video_info", lambda *_: {
        "duration": "12.0", "width": "640", "height": "360",
    })
    monkeypatch.setattr(video_tools, "get_audio_streams", lambda *_: [{"index": 1}])

    plan = video_tools.plan_tool(
        "ffmpeg", "pixelate", str(source), str(tmp_path),
        {"block": 16},
    )
    assert plan.output_path.endswith("clip_pixelated.mp4")
    assert plan.duration == 12.0
    assert plan.width == 640
    assert plan.has_audio is True
    assert os.path.dirname(plan.output_path) == str(tmp_path)


def test_plan_split_returns_numbered_pattern(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"")
    monkeypatch.setattr(video_tools, "get_video_info", lambda *_: {
        "duration": "8.0", "width": "640", "height": "360",
    })
    monkeypatch.setattr(video_tools, "get_audio_streams", lambda *_: [])

    plan = video_tools.plan_tool(
        "ffmpeg", "split", str(source), str(tmp_path), {"parts": 2},
    )
    assert plan.is_directory is True
    assert plan.output_path.endswith(os.path.join("clip_parts", "part_%03d.mp4"))


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        _args("nope")
