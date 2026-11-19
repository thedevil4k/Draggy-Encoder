from src.ai.hardware_fit import (
    GpuProfile,
    annotate_models,
    estimate_model_vram_mb,
    parse_memory_mb,
    recommend_models,
)


def _profile(free_mb=10000, total_mb=12288, backend="CUDA"):
    return GpuProfile(
        name="NVIDIA GeForce RTX 3060",
        vendor="NVIDIA",
        total_mb=total_mb,
        free_mb=free_mb,
        backend=backend,
    )


def test_parse_memory_mb_formats():
    assert parse_memory_mb("12288 MiB") == 12288
    assert parse_memory_mb("12 GB") == 12288
    assert parse_memory_mb("~67 MB") == 67
    assert parse_memory_mb(2048) == 2048
    assert parse_memory_mb("") is None


def test_estimate_includes_weights_and_activation():
    model = {
        "engine_id": "upscale-realesrgan-x4plus",
        "category": "Upscale",
        "size_estimate": "~67 MB",
    }
    estimate = estimate_model_vram_mb(model)
    assert estimate is not None
    assert 600 <= estimate <= 1200


def test_estimate_unknown_without_size():
    assert estimate_model_vram_mb({"category": "Upscale"}) is None


def test_annotate_marks_too_large():
    models = [{"engine_id": "huge", "category": "Upscale", "size_mb": 12000}]
    annotated, profile = annotate_models(models, _profile())
    assert profile.total_mb == 12288
    assert annotated[0]["vram_fit"] == "too_large"
    assert annotated[0]["vram_estimate_mb"] > 10000


def test_recommendation_skips_model_that_does_not_fit():
    models = [
        {
            "engine_id": "huge",
            "display_name": "Huge",
            "category": "Upscale",
            "size_mb": 12000,
            "quality_score": 99,
            "enabled": True,
        },
        {
            "engine_id": "small",
            "display_name": "Small",
            "category": "Upscale",
            "size_mb": 67,
            "quality_score": 80,
            "enabled": True,
        },
    ]
    rec = recommend_models(models, "Upscale", _profile())
    assert rec["engine_id"] == "small"
    assert rec["vram_fit"] == "fits"


def test_recommendation_prefers_enabled_fitting_model():
    models = [
        {
            "engine_id": "installed",
            "display_name": "Installed",
            "category": "Upscale",
            "size_mb": 67,
            "quality_score": 95,
            "installed": True,
        },
        {
            "engine_id": "enabled",
            "display_name": "Enabled",
            "category": "Upscale",
            "size_mb": 67,
            "quality_score": 80,
            "installed": True,
            "enabled": True,
        },
    ]
    rec = recommend_models(models, "Upscale", _profile())
    assert rec["engine_id"] == "enabled"
    assert "enabled" in rec["reason"]


def test_cpu_profile_does_not_claim_fit():
    models = [{"engine_id": "x", "category": "Upscale", "size_mb": 67}]
    annotated, _ = annotate_models(models, GpuProfile(backend="CPU"))
    assert annotated[0]["vram_fit"] == "unknown"
    rec = recommend_models(models, "Upscale", GpuProfile(backend="CPU"))
    assert rec["engine_id"] == "x"


def test_card_shows_recommended_and_vram_warning(qapp):
    from PySide6.QtWidgets import QLabel

    from src.ui.model_search_result import ModelSearchResult

    card = ModelSearchResult(
        {
            "engine_id": "upscale-demo",
            "display_name": "Demo",
            "category": "Upscale",
            "vram_estimate_mb": 9000,
            "vram_fit": "too_large",
            "gpu_total_mb": 12288,
        },
        recommended=True,
    )
    texts = [label.text() for label in card.findChildren(QLabel)]
    assert any("RECOMMENDED" in text for text in texts)
    assert any("may not fit" in text for text in texts)
