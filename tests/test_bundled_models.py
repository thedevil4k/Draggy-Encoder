"""Los modelos que viajan en el instalador deben quedar listos sin HF."""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture()
def bundled_env(tmp_path, monkeypatch):
    """Aísla el store de modelos y el engines.json de la instalación real."""
    import src.ai.engines_state as state
    import src.ai.registry as registry

    models = tmp_path / "models"
    session = tmp_path / "session"
    models.mkdir()
    session.mkdir()
    monkeypatch.setattr(registry, "get_models_dir", lambda: str(models))
    monkeypatch.setattr(state, "get_session_dir", lambda: str(session))
    monkeypatch.setattr(state, "_LOAD_CACHE", {})
    return models, session


def _write_bundled(models):
    import src.ai.registry as registry

    paths = {
        "upscale-realesrgan-x4plus": ("onnx", "model.onnx"),
        "upscale-realesr-animevideov3": ("realesr-animevideov3.onnx",),
    }
    for engine_id, parts in paths.items():
        target = os.path.join(
            registry.get_engine_dir(engine_id), *parts
        )
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(b"fake onnx")


def test_seed_marks_bundled_models_installed_and_enabled(bundled_env):
    from src.ai.engines_state import get_state, seed_bundled_engines

    models, _session = bundled_env
    _write_bundled(models)

    changed = seed_bundled_engines()

    assert set(changed) == {
        "upscale-realesrgan-x4plus",
        "upscale-realesr-animevideov3",
    }
    for engine_id in changed:
        state = get_state(engine_id)
        assert state["installed"] is True
        assert state["enabled"] is True
        assert state["onnx_converted"] is True
        assert os.path.isfile(state["onnx_path"])


def test_seed_is_idempotent(bundled_env):
    from src.ai.engines_state import seed_bundled_engines

    models, _session = bundled_env
    _write_bundled(models)

    assert seed_bundled_engines()
    assert seed_bundled_engines() == []


def test_seed_respects_explicit_disable(bundled_env):
    from src.ai.engines_state import get_state, patch, seed_bundled_engines

    models, _session = bundled_env
    _write_bundled(models)
    seed_bundled_engines()

    patch("upscale-realesrgan-x4plus", enabled=False)
    assert seed_bundled_engines() == []
    assert get_state("upscale-realesrgan-x4plus")["enabled"] is False


def test_seed_enables_legacy_installed_record_without_enabled(bundled_env):
    from src.ai.engines_state import get_state, patch, seed_bundled_engines

    models, _session = bundled_env
    _write_bundled(models)
    patch("upscale-realesr-animevideov3", installed=True, enabled=None)

    assert "upscale-realesr-animevideov3" in seed_bundled_engines()
    assert get_state("upscale-realesr-animevideov3")["enabled"] is True


def test_seed_does_nothing_when_model_files_are_absent(bundled_env):
    from src.ai.engines_state import seed_bundled_engines

    assert seed_bundled_engines() == []
    assert not (bundled_env[1] / "engines.json").exists()
