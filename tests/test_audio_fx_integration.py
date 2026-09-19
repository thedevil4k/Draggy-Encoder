"""Integración de los efectos de audio con el render y los ajustes.

Fija el contrato de los DOS modos del boost de punta a punta:
  * casilla sola -> loudnorm lineal (comportamiento histórico intacto);
  * casilla + desplegable -> la cadena de efectos altera el boost.

También cubre la persistencia de los ajustes y que un cambio en el panel llegue
al diccionario de settings que consume el pipeline.
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from src.ffmpeg import audio_fx
from src.ffmpeg.runner import _build_loudnorm_filter, build_boost_filter


MEASURED = {
    "input_i": "-25.40",
    "input_tp": "-9.30",
    "input_lra": "8.70",
    "input_thresh": "-35.40",
    "target_offset": "0.0000",
}


def _effects(**overrides):
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    for key, value in overrides.items():
        cfg[key].update(value) if isinstance(value, dict) else cfg.update({key: value})
    return cfg


# ---------------------------------------------------------------------------
# build_boost_filter: el punto único donde se decide el modo
# ---------------------------------------------------------------------------


def test_checkbox_alone_uses_linear_loudnorm():
    """Sin efectos, el filtro es EXACTAMENTE el loudnorm de dos pasadas."""
    settings = {"audio_fx": audio_fx.default_effects()}
    assert build_boost_filter(settings, MEASURED) == _build_loudnorm_filter(MEASURED)


def test_missing_audio_fx_key_uses_linear_loudnorm():
    """Un settings sin la clave (llamadores antiguos) no debe cambiar nada."""
    assert build_boost_filter({}, MEASURED) == _build_loudnorm_filter(MEASURED)
    assert build_boost_filter({"audio_fx": None}, MEASURED) == _build_loudnorm_filter(MEASURED)


def test_effects_switch_the_boost_to_the_chain():
    settings = {"audio_fx": _effects(compressor={"enabled": True})}
    af = build_boost_filter(settings, MEASURED)
    assert "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=" not in af
    assert "acompressor=" in af
    assert "volume=" in af


def test_effects_without_measurement_still_builds_a_valid_chain():
    """Sin medición no hay umbral, pero la cadena debe seguir siendo válida."""
    settings = {"audio_fx": _effects(compressor={"enabled": True})}
    af = build_boost_filter(settings, None)
    assert af.count("loudnorm=") == 1, af
    assert "acompressor=" not in af


def test_boost_filter_target_lufs_override():
    settings = {"audio_fx": _effects(compressor={"enabled": True}),
                "audio_target_lufs": -20.0}
    af = build_boost_filter(settings, MEASURED)
    assert "I=-20.0" in af


def test_boost_filter_never_raises_on_garbage():
    for garbage in (None, [], "texto", 7, {"eq": "roto"}):
        assert isinstance(build_boost_filter({"audio_fx": garbage}, MEASURED), str)


# ---------------------------------------------------------------------------
# build_args: la ruta estándar re-codifica con el filtro correcto
# ---------------------------------------------------------------------------


def test_build_args_applies_effects_chain_when_enabled():
    from src.ffmpeg.runner import build_args

    settings = {
        "codec": "libx264", "audio": "copy", "audio_boost": True,
        "format": "mp4", "bitrate": 1500, "resolution": "original",
        "audio_fx": _effects(compressor={"enabled": True}),
    }
    args = build_args("in.mp4", "out.mp4", settings, loudnorm_meas=MEASURED)
    af = args[args.index("-af") + 1]
    assert "acompressor=" in af
    assert args[args.index("-c:a") + 1] == "aac"


def test_build_args_without_boost_has_no_audio_filter():
    from src.ffmpeg.runner import build_args

    settings = {
        "codec": "libx264", "audio": "copy", "audio_boost": False,
        "format": "mp4", "bitrate": 1500, "resolution": "original",
        "audio_fx": _effects(eq={"enabled": True, "bass": 4.0}),
    }
    args = build_args("in.mp4", "out.mp4", settings, loudnorm_meas=MEASURED)
    assert "-af" not in args
    assert args[args.index("-c:a") + 1] == "copy"


# ---------------------------------------------------------------------------
# Ruta IA: _audio_mux_args usa la misma decisión
# ---------------------------------------------------------------------------


def test_audio_mux_args_share_the_boost_decision():
    from src.processing.pipeline import _audio_mux_args

    plain = _audio_mux_args("copy", MEASURED, True, "mp4",
                            {"audio_fx": audio_fx.default_effects()})
    assert "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=" in plain[plain.index("-af") + 1]

    with_fx = _audio_mux_args("copy", MEASURED, True, "mp4",
                              {"audio_fx": _effects(compressor={"enabled": True})})
    assert "acompressor=" in with_fx[with_fx.index("-af") + 1]


def test_audio_mux_args_passthrough_unaffected():
    from src.processing.pipeline import _audio_mux_args

    assert _audio_mux_args("copy", None, False, "mp4", {}) == ["-c:a", "copy"]


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def test_effects_survive_a_json_round_trip(tmp_path, monkeypatch):
    """Lo que se guarda en preferencias debe volver utilizable."""
    from src import preferences

    monkeypatch.setattr(preferences, "_path",
                        lambda: str(tmp_path / "preferences.json"))
    cfg = _effects(eq={"enabled": True, "bass": 4.5},
                   compressor={"enabled": True, "ratio": 5.0},
                   limiter={"enabled": True})
    preferences.set("audio_fx", cfg)

    restored = preferences.get("audio_fx")
    assert audio_fx.effects_are_active(restored) is True
    assert restored["eq"]["bass"] == pytest.approx(4.5)
    assert restored["compressor"]["ratio"] == pytest.approx(5.0)
    assert restored["limiter"]["enabled"] is True
    # Y produce la misma cadena que antes de guardar.
    assert audio_fx.build_effects_chain(restored, -25.0) == audio_fx.build_effects_chain(
        cfg, -25.0
    )


# ---------------------------------------------------------------------------
# Panel de UI (necesita QApplication)
# ---------------------------------------------------------------------------


def test_panel_reports_inactive_by_default(qapp):
    from src.ui.audio_fx_panel import AudioFxPanel

    panel = AudioFxPanel()
    assert audio_fx.effects_are_active(panel.get_effects()) is False
    panel.deleteLater()


def test_panel_sliders_change_the_effects(qapp):
    from src.ui.audio_fx_panel import AudioFxPanel

    panel = AudioFxPanel()
    panel.eq_check.setChecked(True)
    panel.eq_bass.slider.setValue(50)          # escala x10 -> 5.0 dB
    cfg = panel.get_effects()
    assert cfg["eq"]["bass"] == pytest.approx(5.0)
    assert audio_fx.effects_are_active(cfg) is True
    panel.deleteLater()


def test_panel_reset_returns_to_standard_boost(qapp):
    from src.ui.audio_fx_panel import AudioFxPanel

    panel = AudioFxPanel()
    panel.comp_check.setChecked(True)
    panel.reset_button.click()
    assert audio_fx.effects_are_active(panel.get_effects()) is False
    panel.deleteLater()


def test_panel_preset_sets_compressor_values(qapp):
    from src.ui.audio_fx_panel import AudioFxPanel, COMPRESSOR_PRESETS

    panel = AudioFxPanel()
    panel.comp_check.setChecked(True)
    for index, (name, values) in enumerate(COMPRESSOR_PRESETS):
        panel.preset_combo.setCurrentIndex(index)
        cfg = panel.get_effects()["compressor"]
        assert cfg["ratio"] == pytest.approx(values["ratio"]), name
        assert cfg["threshold_lu"] == pytest.approx(values["threshold_lu"]), name
    panel.deleteLater()


def test_panel_emits_changed_but_not_while_loading(qapp):
    from src.ui.audio_fx_panel import AudioFxPanel

    panel = AudioFxPanel()
    hits = []
    panel.changed.connect(lambda: hits.append(1))
    panel.set_effects(_effects(compressor={"enabled": True}))
    assert hits == [], "cargar configuración no debe notificar"
    panel.set_effects(audio_fx.default_effects(), emit=True)
    assert hits == [1], "restablecer sí debe notificar"
    panel.deleteLater()


def test_panel_preview_is_debounced_and_cancellable(qapp):
    """Arrastrar un control no debe lanzar una generación por píxel."""
    from src.ui.audio_fx_panel import AudioFxPanel

    panel = AudioFxPanel()
    requests, stops = [], []
    panel.preview_requested.connect(lambda: requests.append(1))
    panel.preview_stop_requested.connect(lambda: stops.append(1))

    panel.eq_check.setChecked(True)
    panel.preview_button.click()
    assert len(requests) == 1

    panel.set_playback_state(True)
    for value in (10, 20, 30, 40, 50):
        panel.eq_bass.slider.setValue(value)
    assert len(requests) == 1, "el anti-rebote no debe disparar todavía"

    panel._preview_timer.timeout.emit()  # simula el vencimiento del temporizador
    assert len(requests) == 2

    panel.stop_button.click()
    assert stops == [1]
    assert panel.stop_button.isEnabled() is False

    # Estando parado, mover un control no reproduce nada por sorpresa.
    panel.eq_bass.slider.setValue(70)
    panel._preview_timer.timeout.emit()
    assert len(requests) == 2
    panel.deleteLater()
