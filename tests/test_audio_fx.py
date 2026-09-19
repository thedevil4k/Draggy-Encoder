"""Cadena de efectos del Audio Boost (alterar el boost, no apilar encima).

El comportamiento que se fija aquí:
  * casilla sola (desplegable cerrado o vacío) -> boost estándar, SIN efectos.
  * casilla + algo activado -> el boost cambia de modo y aplica la cadena.

Se validan además las decisiones que costaron medidas descubrir (orden del
compresor, normalización final, make-up ausente, limitador con margen).
"""
from __future__ import annotations

import copy
import inspect

import pytest

from src.ffmpeg import audio_fx


def test_closed_dropdown_leaves_boost_untouched():
    """Con el desplegable cerrado NADA cambia: el boost es el de siempre."""
    cfg = audio_fx.default_effects()
    assert audio_fx.effects_are_active(cfg) is False
    assert audio_fx.boost_uses_linear_loudnorm(cfg) is True
    assert audio_fx.build_effects_chain(cfg, measured_i=-25.0) == []
    assert audio_fx.describe_effects(cfg) == "boost estandar"


def test_enabled_but_no_option_selected_still_leaves_boost_untouched():
    """Abrir el desplegable sin activar nada tampoco altera el boost.

    Es el caso fácil de romper: ``enabled`` sería True y bastaría con eso para
    cambiar el sonido aunque el usuario no haya tocado ningún control.
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    assert audio_fx.effects_are_active(cfg) is False
    assert audio_fx.boost_uses_linear_loudnorm(cfg) is True
    assert audio_fx.build_effects_chain(cfg, measured_i=-25.0) == []


def test_flat_eq_does_not_change_the_boost():
    """EQ activado con las tres bandas a 0 no debe alterar el boost.

    Aplicarlo costaría tiempo en cada muestra sin efecto audible.
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["eq"]["enabled"] = True  # todas las bandas siguen a 0.0
    assert audio_fx.effects_are_active(cfg) is False
    assert audio_fx.build_effects_chain(cfg, measured_i=-25.0) == []


def test_eq_only_switches_the_boost_mode():
    """EQ con bandas reales -> el boost deja de ser loudnorm lineal."""
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["eq"].update(enabled=True, bass=4.0)
    assert audio_fx.effects_are_active(cfg) is True
    assert audio_fx.boost_uses_linear_loudnorm(cfg) is False
    chain = audio_fx.build_effects_chain(cfg, measured_i=-25.0)
    assert any(f.startswith("bass=") for f in chain)


def test_compressor_comes_after_the_level_shift():
    """El compresor va DESPUÉS del desplazamiento de nivel, nunca antes.

    Con el compresor antes, el material llega a -25 LUFS y un umbral como -18
    dB no se dispara: medido, el rango dinámico no cambiaba (efecto nulo).
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["compressor"].update(enabled=True, ratio=4.0)
    chain = audio_fx.build_effects_chain(cfg, measured_i=-25.0)

    gain_index = next(i for i, f in enumerate(chain) if f.startswith("volume="))
    comp_index = next(i for i, f in enumerate(chain) if f.startswith("acompressor="))
    assert gain_index < comp_index


def test_chain_is_normalised_after_compressing():
    """Tras el compresor hay una normalización: sin ella el nivel se hunde.

    Medido: un compresor 4:1 bajaba 7 LU y el archivo acababa en -23 LUFS.
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["compressor"].update(enabled=True, ratio=4.0)
    chain = audio_fx.build_effects_chain(cfg, measured_i=-25.0)

    comp_index = next(i for i, f in enumerate(chain) if f.startswith("acompressor="))
    norm_after = [f for f in chain[comp_index + 1:] if f.startswith("loudnorm=")]
    assert norm_after, chain


def test_limiter_is_last_and_has_encoder_margin():
    """El limitador cierra la cadena y su techo descuenta el margen del códec.

    Sin ese margen el archivo entregado supera el techo pedido: el boost actual
    entrega -0.20 dBTP con objetivo -1.5.
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["compressor"].update(enabled=True)
    cfg["limiter"]["enabled"] = True
    chain = audio_fx.build_effects_chain(cfg, measured_i=-25.0)

    assert chain[-1].startswith("alimiter="), chain
    ceiling = audio_fx.limiter_ceiling_db(cfg)
    assert ceiling == pytest.approx(-1.5 - audio_fx.ENCODER_PEAK_MARGIN_DB)
    assert ceiling < -1.5, "el techo del filtro debe quedar por debajo del objetivo"


def test_limiter_is_opt_in():
    """El limitador NO viene activo: solo actúa si el usuario lo activa.

    Aunque baja el pico (-0.20 dBTP de hoy a -2.2), es un cambio de sonido que
    debe pedir el usuario, no colarse por abrir el desplegable.
    """
    cfg = audio_fx.default_effects()
    assert cfg["limiter"]["enabled"] is False
    assert audio_fx.build_limiter_filters(cfg) == []
    assert audio_fx.limiter_ceiling_db(cfg) is None


def test_limiter_disables_autolevel():
    """``level=disabled`` es obligatorio: el auto-nivel rompía el loudness.

    Medido: -14.50 LUFS con auto-level frente a -16.00 sin él.
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["compressor"]["enabled"] = True
    cfg["limiter"]["enabled"] = True
    filters = audio_fx.build_limiter_filters(cfg)
    assert filters and "level=disabled" in filters[0]


def test_compressor_threshold_follows_the_material_level():
    """El umbral se deriva del nivel medido, no es un dB absoluto.

    Así el compresor trabaja tanto con una fuente a -14 como a -30 LUFS.
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["compressor"].update(enabled=True, threshold_lu=8.0)

    loud = audio_fx.compressor_threshold_db(cfg, -14.0)
    quiet = audio_fx.compressor_threshold_db(cfg, -30.0)
    assert loud == pytest.approx(-24.0)
    assert quiet == pytest.approx(-24.0), "el umbral debe ser el mismo tras el boost"


def test_compressor_skipped_without_measurement():
    """Sin medición no hay umbral con sentido: se omite en vez de inventarlo."""
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["compressor"]["enabled"] = True
    assert audio_fx.build_compressor_filters(cfg, None) == []
    chain = audio_fx.build_effects_chain(cfg, measured_i=None)
    assert not any(f.startswith("acompressor=") for f in chain)
    # La cadena sigue siendo válida (normaliza y limita).
    assert any(f.startswith("loudnorm=") for f in chain)


def test_eq_filters_are_multichannel_safe():
    """Se usan filtros globales, no ``anequalizer`` (que exige layout)."""
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["eq"].update(enabled=True, bass=3, mid=-2, treble=2)
    filters = audio_fx.build_eq_filters(cfg)
    joined = ",".join(filters)
    assert "anequalizer" not in joined
    assert "bass=" in joined and "treble=" in joined and "equalizer=" in joined


def test_sanitize_clamps_and_defaults():
    """Valores absurdos no deben llegar a FFmpeg (romperían todo el render)."""
    cfg = audio_fx.sanitize_effects({
        "enabled": True,
        "eq": {"enabled": True, "bass": 999, "mid": "no-numero", "treble": None},
        "compressor": {"enabled": True, "ratio": -5, "attack_ms": 1e9},
        "limiter": {"ceiling_dbtp": -99},
    })
    assert cfg["eq"]["bass"] == 12.0
    assert cfg["eq"]["mid"] == 0.0
    assert cfg["eq"]["treble"] == 0.0
    assert cfg["compressor"]["ratio"] == 1.0
    assert cfg["compressor"]["attack_ms"] == 200.0
    assert cfg["limiter"]["ceiling_dbtp"] == -6.0


def test_sanitize_never_raises():
    for garbage in (None, [], "texto", 42, {"eq": "roto"}, {"compressor": 7}):
        cfg = audio_fx.sanitize_effects(garbage)
        assert cfg == audio_fx.default_effects()


def test_no_makeup_gain_on_the_compressor():
    """No se usa ``makeup``: la normalización final lo sustituye.

    El makeup de ``acompressor`` sube el nivel linealmente y disparaba picos
    (medido: +4 dB de makeup -> 5.5 dBTP). La normalización posterior deja el
    nivel en el objetivo sin ese riesgo.
    """
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["compressor"].update(enabled=True, ratio=4.0)
    filters = audio_fx.build_compressor_filters(cfg, measured_i=-25.0)
    assert filters and "makeup" not in filters[0]


def test_describe_effects_mentions_each_active_block():
    cfg = audio_fx.default_effects()
    cfg["enabled"] = True
    cfg["eq"].update(enabled=True, bass=3)
    cfg["compressor"].update(enabled=True, ratio=4)
    cfg["limiter"]["enabled"] = True
    text = audio_fx.describe_effects(cfg)
    assert "boost +" in text
    assert "EQ" in text
    assert "compresor" in text
    assert "limitador" in text
