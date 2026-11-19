from src.ai import hardware_fit, vram_calibration
from src.ai.hardware_fit import GpuProfile, annotate_models, recommend_models


def test_calibration_roundtrip_and_worst_case(tmp_path):
    path = str(tmp_path / "vram.json")
    entry = vram_calibration.record_measurement(
        "upscale-demo",
        1200,
        estimate_mb=340,
        resolution="320x180 -> 1280x720",
        backend="CUDAExecutionProvider",
        path=path,
    )
    assert entry["peak_delta_mb"] == 1200
    assert vram_calibration.lookup("upscale-demo", path=path)["peak_delta_mb"] == 1200

    # Una medida menor no debe sustituir al peor caso.
    vram_calibration.record_measurement("upscale-demo", 800, path=path)
    assert vram_calibration.lookup("upscale-demo", path=path)["peak_delta_mb"] == 1200


def test_annotate_uses_measured_value(monkeypatch):
    monkeypatch.setattr(
        vram_calibration,
        "lookup",
        lambda engine_id: {
            "peak_delta_mb": 6200,
            "resolution": "320x180 -> 1280x720",
            "measured_at_iso": "2026-11-15T20:00:00+00:00",
        }
        if engine_id == "upscale-demo"
        else None,
    )
    profile = GpuProfile(
        name="NVIDIA GeForce RTX 3060",
        vendor="NVIDIA",
        total_mb=12288,
        free_mb=10000,
        backend="CUDA",
    )
    models = [
        {
            "engine_id": "upscale-demo",
            "display_name": "Demo",
            "category": "Upscale",
            "size_mb": 64,
            "enabled": True,
        }
    ]
    annotated, _ = annotate_models(models, profile)
    assert annotated[0]["vram_source"] == "measured"
    assert annotated[0]["vram_estimate_mb"] == 6200
    assert annotated[0]["vram_fit"] == "fits"


def test_recommendation_reports_measured_source(monkeypatch):
    monkeypatch.setattr(
        vram_calibration,
        "lookup",
        lambda engine_id: {"peak_delta_mb": 6200, "resolution": "720p"}
        if engine_id == "upscale-demo"
        else None,
    )
    profile = GpuProfile(
        name="NVIDIA GeForce RTX 3060",
        vendor="NVIDIA",
        total_mb=12288,
        free_mb=10000,
        backend="CUDA",
    )
    rec = recommend_models(
        [
            {
                "engine_id": "upscale-demo",
                "display_name": "Demo",
                "category": "Upscale",
                "size_mb": 64,
                "enabled": True,
            }
        ],
        "Upscale",
        profile,
    )
    assert rec["vram_source"] == "measured"
    assert "measured on this GPU" in rec["reason"]
