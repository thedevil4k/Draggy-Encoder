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


def test_multiaudio_maps_every_track_and_tags_the_new_one():
    args = _args(
        "multiaudio",
        output="out.mkv",
        options={"tracks": [{"path": "es.mkv", "language": "spa",
                             "title": "Spanish"}]},
        has_audio=True,
        audio_track_count=1,
    )
    assert args.count("-i") == 2
    assert "1:a:0" in args
    assert "0:a?" in args
    assert args[args.index("language=spa") - 1] == "-metadata:s:a:1"
    assert "copy" in args
    assert args[-1] == "out.mkv"


def test_multiaudio_mp4_reencodes_only_the_added_track():
    args = _args(
        "multiaudio",
        output="out.mp4",
        options={"container": "mp4",
                 "tracks": [{"path": "en.mkv", "language": "eng"}]},
        has_audio=True,
        audio_track_count=2,
    )
    assert args[args.index("language=eng") - 1] == "-metadata:s:a:2"
    assert args[args.index("-c:a:2") + 1] == "aac"
    assert "+faststart" in args


def test_multiaudio_rejects_duplicates_and_the_source_itself():
    with pytest.raises(ValueError, match="already included"):
        _args("multiaudio", options={"tracks": [{"path": "input.mp4"}]},
              has_audio=True)
    with pytest.raises(ValueError, match="twice"):
        _args(
            "multiaudio",
            options={"tracks": [{"path": "a.mkv"}, {"path": "a.mkv"}]},
            has_audio=True,
        )
    with pytest.raises(ValueError, match="at least one"):
        _args("multiaudio", options={"tracks": []}, has_audio=True)


def test_plan_multiaudio_uses_mkv_and_checks_added_files(tmp_path, monkeypatch):
    source = tmp_path / "ep.mp4"
    source.write_bytes(b"")
    extra = tmp_path / "ep_es.mkv"
    extra.write_bytes(b"")
    monkeypatch.setattr(video_tools, "get_video_info", lambda *_: {
        "duration": "20.0", "width": "640", "height": "360",
    })
    monkeypatch.setattr(video_tools, "get_audio_streams", lambda *_: [{"index": 1}])

    plan = video_tools.plan_tool(
        "ffmpeg", "multiaudio", str(source), str(tmp_path),
        {"tracks": [{"path": str(extra), "language": "spa"}]},
    )
    assert plan.output_path.endswith("ep_multi.mkv")
    assert plan.is_directory is False

    monkeypatch.setattr(video_tools, "get_audio_streams", lambda *_: [])
    with pytest.raises(ValueError, match="no audio"):
        video_tools.plan_tool(
            "ffmpeg", "multiaudio", str(source), str(tmp_path),
            {"tracks": [{"path": str(extra)}]},
        )


def test_multiaudio_normalize_reencodes_only_the_added_tracks():
    args = _args(
        "multiaudio",
        output="out.mkv",
        options={"normalize": True,
                 "measurements": {2: {"input_i": "-27.0", "input_tp": "-3.0",
                                      "input_lra": "7.0",
                                      "input_thresh": "-37.0",
                                      "target_offset": "0.0"}},
                 "tracks": [{"path": "es.mkv", "language": "spa"}]},
        has_audio=True,
        audio_track_count=2,
    )
    assert args[args.index("-c:a:2") + 1] == "aac"
    assert args[args.index("-ar:2") + 1] == "48000"
    filter_text = args[args.index("-filter:a:2") + 1]
    assert "loudnorm=" in filter_text
    assert "measured_I=-27.0000" in filter_text
    assert "linear=true" in filter_text
    assert "-c:a:0" not in args and "-c:a:1" not in args
    assert "copy" in args
    assert "+faststart" not in args


def test_multiaudio_normalize_can_also_level_the_original_tracks():
    args = _args(
        "multiaudio",
        output="out.mkv",
        options={"normalize": True, "normalize_original": True,
                 "measurements": {0: {"input_i": "-24.0", "input_tp": "-2.0",
                                      "input_lra": "9.0",
                                      "input_thresh": "-34.0",
                                      "target_offset": "0.0"}},
                 "tracks": [{"path": "es.mkv", "language": "spa"}]},
        has_audio=True,
        audio_track_count=1,
    )
    assert args[args.index("-c:a:0") + 1] == "aac"
    assert args[args.index("-ar:0") + 1] == "48000"
    assert "measured_I=-24.0000" in args[args.index("-filter:a:0") + 1]
    # La añadida sigue normalizándose a su vez y con su propia medición.
    assert args[args.index("-c:a:1") + 1] == "aac"
    assert "loudnorm=" in args[args.index("-filter:a:1") + 1]


def test_multiaudio_normalize_without_measurement_uses_one_pass():
    args = _args(
        "multiaudio",
        options={"normalize": True, "tracks": [{"path": "es.mkv"}]},
        has_audio=True,
    )
    assert args[args.index("-filter:a:1") + 1] == "loudnorm=I=-16:TP=-1.5:LRA=11"


def test_multiaudio_without_normalize_keeps_every_track_copied():
    args = _args(
        "multiaudio",
        options={"normalize": False, "tracks": [{"path": "es.mkv"}]},
        has_audio=True,
    )
    assert not any(arg.startswith("-filter:") for arg in args)
    assert not any(arg.startswith("-c:a:") for arg in args)


def test_plan_multiaudio_normalize_lists_each_added_track(
    tmp_path, monkeypatch
):
    source = tmp_path / "ep.mp4"
    source.write_bytes(b"")
    extra = tmp_path / "ep_es.mkv"
    extra.write_bytes(b"")
    monkeypatch.setattr(video_tools, "get_video_info", lambda *_: {
        "duration": "20.0", "width": "640", "height": "360",
    })
    monkeypatch.setattr(video_tools, "get_audio_streams", lambda *_: [{"index": 1}])

    plan = video_tools.plan_tool(
        "ffmpeg", "multiaudio", str(source), str(tmp_path),
        {"tracks": [{"path": str(extra)}], "normalize": True},
    )
    # Una pista original delante: el índice de salida de la añadida es 1.
    assert plan.normalize_sources == [(1, str(extra), 0)]
    assert plan.options["normalize"] is True
    assert plan.audio_track_count == 1

    plan = video_tools.plan_tool(
        "ffmpeg", "multiaudio", str(source), str(tmp_path),
        {"tracks": [{"path": str(extra)}], "normalize": True,
         "normalize_original": True},
    )
    # También se mide la pista propia del vídeo, que en la salida es la 0.
    assert plan.normalize_sources == [(0, str(source), 0), (1, str(extra), 0)]


def test_multiaudio_advances_the_dub_with_a_trim_filter():
    args = _args(
        "multiaudio",
        options={"tracks": [{"path": "es.mkv", "delay": -1.25}]},
        has_audio=True,
    )
    filter_text = args[args.index("-filter:a:1") + 1]
    assert filter_text == "atrim=start=1.250,asetpts=PTS-STARTPTS"
    assert args[args.index("-c:a:1") + 1] == "aac"
    assert "-itsoffset" not in args


def test_multiaudio_delays_the_dub_with_adelay():
    args = _args(
        "multiaudio",
        options={"tracks": [{"path": "es.mkv", "delay": 1.5}]},
        has_audio=True,
    )
    assert args[args.index("-filter:a:1") + 1] == "adelay=1500:all=1"


def test_multiaudio_copies_untouched_when_the_delay_is_zero():
    args = _args(
        "multiaudio",
        options={"tracks": [{"path": "es.mkv", "delay": 0}]},
        has_audio=True,
    )
    assert not any(arg.startswith("-filter:") for arg in args)
    assert not any(arg.startswith("-c:a:") for arg in args)


def test_multiaudio_rejects_delay_out_of_range():
    with pytest.raises(ValueError, match="between -30 and 30"):
        _args(
            "multiaudio",
            options={"tracks": [{"path": "es.mkv", "delay": 45}]},
            has_audio=True,
        )


def test_plan_multiaudio_auto_delay_lists_the_dubs(tmp_path, monkeypatch):
    source = tmp_path / "ep.mp4"
    source.write_bytes(b"")
    extra = tmp_path / "ep_es.mkv"
    extra.write_bytes(b"")
    monkeypatch.setattr(video_tools, "get_video_info", lambda *_: {
        "duration": "20.0", "width": "640", "height": "360",
    })
    monkeypatch.setattr(video_tools, "get_audio_streams", lambda *_: [{"index": 1}])

    plan = video_tools.plan_tool(
        "ffmpeg", "multiaudio", str(source), str(tmp_path),
        {"tracks": [{"path": str(extra)}], "auto_delay": True},
    )
    assert plan.delay_sources == [(0, str(extra), 0)]


def test_multiaudio_uses_the_chosen_audio_stream_and_copies_its_subs():
    args = _args(
        "multiaudio",
        output="out.mkv",
        options={"tracks": [{"path": "es.mkv", "stream": 1}]},
        has_audio=True,
        audio_track_count=1,
        subtitle_streams=[[{
            "subtitle_index": 0, "codec_name": "subrip",
            "language": "spa", "title": "Spanish SDH",
        }]],
        base_subtitle_count=2,
    )
    assert "1:a:1" in args
    assert "1:a:0" not in args
    assert "1:s:0" in args
    # Los subtítulos del vídeo van delante: este sale como pista 2.
    assert args[args.index("language=spa") - 1] == "-metadata:s:s:2"
    assert args[args.index("title=Spanish SDH") - 1] == "-metadata:s:s:2"
    assert "mov_text" not in args


def test_multiaudio_mp4_converts_text_subs_and_skips_bitmap_ones():
    args = _args(
        "multiaudio",
        output="out.mp4",
        options={"container": "mp4", "tracks": [{"path": "es.mkv"}]},
        has_audio=True,
        audio_track_count=1,
        subtitle_streams=[[
            {"subtitle_index": 0, "codec_name": "hdmv_pgs_subtitle"},
            {"subtitle_index": 1, "codec_name": "subrip", "language": "spa"},
        ]],
    )
    assert "1:s:0" not in args
    assert "1:s:1" in args
    assert args[args.index("-c:s:0") + 1] == "mov_text"
    assert args[args.index("language=spa") - 1] == "-metadata:s:s:0"


def test_plan_rejects_an_audio_stream_the_dub_does_not_have(
    tmp_path, monkeypatch
):
    source = tmp_path / "ep.mp4"
    source.write_bytes(b"")
    extra = tmp_path / "ep_es.mkv"
    extra.write_bytes(b"")
    monkeypatch.setattr(video_tools, "get_video_info", lambda *_: {
        "duration": "20.0", "width": "640", "height": "360",
    })
    monkeypatch.setattr(video_tools, "get_audio_streams", lambda *_: [{"index": 1}])
    monkeypatch.setattr(video_tools, "get_subtitle_streams", lambda *_: [])

    with pytest.raises(ValueError, match="#3"):
        video_tools.plan_tool(
            "ffmpeg", "multiaudio", str(source), str(tmp_path),
            {"tracks": [{"path": str(extra), "stream": 2}]},
        )


def test_episode_key_understands_sxxexx_and_1x03():
    assert video_tools.episode_key("Show.S01E03.1080p.mkv") == (1, 3)
    assert video_tools.episode_key("show s1 e3.mkv") == (1, 3)
    assert video_tools.episode_key("Padre.de.familia 1x03.mkv") == (1, 3)
    assert video_tools.episode_key("random trailer.mkv") is None


def test_match_dub_pairs_releases_with_different_names(tmp_path):
    base = tmp_path / "Family.Guy.S01E03.1080p.mkv"
    dub = tmp_path / "Padre.de.familia.S01E03.Castellano.mkv"
    other = tmp_path / "Padre.de.familia.S01E04.Castellano.mkv"
    for path in (base, dub, other):
        path.write_bytes(b"")

    result = video_tools.match_dub(
        str(base), video_tools.media_files(str(tmp_path))
    )
    assert result["dub"] == str(dub)
    assert "S01E03" in result["status"]


def test_match_dub_falls_back_to_the_normalised_name(tmp_path):
    base = tmp_path / "episodio piloto.mp4"
    dub = tmp_path / "Episodio Piloto (castellano).mkv"
    base.write_bytes(b"")
    dub.write_bytes(b"")

    result = video_tools.match_dub(
        str(base), video_tools.media_files(str(tmp_path))
    )
    assert result["dub"] == str(dub)
    assert "name" in result["status"]


def test_match_dub_refuses_ambiguity_and_reports_missing(tmp_path):
    base = tmp_path / "Show.S01E03.mkv"
    latin = tmp_path / "Show.S01E03.Latino.mkv"
    castilian = tmp_path / "Show.S01E03.Castellano.mkv"
    for path in (base, latin, castilian):
        path.write_bytes(b"")

    result = video_tools.match_dub(str(base), [str(latin), str(castilian)])
    assert result["dub"] == ""
    assert "Ambiguous" in result["status"]

    empty = tmp_path / "empty"
    empty.mkdir()
    missing = video_tools.match_dub(
        str(base), video_tools.media_files(str(empty))
    )
    assert missing["dub"] == ""
    assert missing["status"] == "No dub found"


def test_media_files_skips_folders_and_other_extensions(tmp_path):
    video = tmp_path / "a.mkv"
    audio = tmp_path / "b.mka"
    note = tmp_path / "c.txt"
    (tmp_path / "subdir").mkdir()
    for path in (video, audio, note):
        path.write_bytes(b"")

    assert video_tools.media_files(str(tmp_path)) == [str(video), str(audio)]
    assert video_tools.media_files(
        str(tmp_path), video_tools.VIDEO_EXTENSIONS
    ) == [str(video)]


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        _args("nope")
