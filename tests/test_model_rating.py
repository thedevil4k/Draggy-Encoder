"""Tests del baremo de modelos y de las regresiones de arquitectura.

Cubre dos cosas que costaron un bug real cada una:

  * ``model_rating``: que las métricas sean deterministas y que el índice se
    mueva en la dirección correcta (más fidelidad/detalle -> más nota; más
    halos -> menos nota).
  * ``_RRDB`` / ``RRDBNet``: el forward tiene que ser EXACTAMENTE el oficial.
    Un ``LeakyReLU`` de más después de la suma residual (o el orden
    conv/interpolate invertido) carga los pesos sin quejarse y devuelve una
    imagen oscura y borrosa: el type-check del state_dict no lo detecta.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai import model_rating as mr  # noqa: E402


# ------------------------------------------------------------ métricas base
def test_reference_set_is_deterministic():
    first = mr.reference_pair(96, 160, 4)
    second = mr.reference_pair(96, 160, 4)
    assert [name for name, *_ in first] == [name for name, *_ in second]
    for (_, high_a, low_a, _), (_, high_b, low_b, _) in zip(first, second):
        assert np.array_equal(high_a, high_b)
        assert np.array_equal(low_a, low_b)
    # El LR es exactamente el HR promediado a bloques scale x scale.
    _, high, low, scale = first[0]
    assert low.shape[0] == high.shape[0] // scale
    assert low.shape[1] == high.shape[1] // scale


def test_psnr_and_ssim_bounds():
    _, high, _low, _scale = mr.reference_pair(64, 96, 4)[0]
    assert mr.ssim(high, high) == pytest.approx(1.0, abs=1e-6)
    assert mr.psnr(high, high) >= 98.0
    rng = np.random.default_rng(0)
    noisy = np.clip(high.astype(np.int16) + rng.integers(-20, 20, high.shape),
                    0, 255).astype(np.uint8)
    assert 0.0 < mr.ssim(high, noisy) < 1.0
    assert 0.0 < mr.psnr(high, noisy) < 60.0


def test_score_moves_in_the_right_direction():
    base = {"ssim": 0.80, "detail_gain": 1.6, "overshoot_pct": 2.0,
            "colour_drift": 1.0}
    better = dict(base, ssim=0.90, detail_gain=2.2)
    worse = dict(base, ssim=0.72, detail_gain=1.1)
    noisy = dict(base, overshoot_pct=20.0, colour_drift=12.0)
    assert (mr.score_from_metrics(better)["score"]
            > mr.score_from_metrics(base)["score"])
    assert (mr.score_from_metrics(worse)["score"]
            < mr.score_from_metrics(base)["score"])
    assert (mr.score_from_metrics(noisy)["score"]
            < mr.score_from_metrics(base)["score"])
    for result in (mr.score_from_metrics(better), mr.score_from_metrics(noisy)):
        assert 0.0 <= result["score"] <= 100.0


def test_tiers_and_labels():
    assert mr.quality_tier(10) == "básica"
    assert mr.quality_tier(80) == "máxima"
    assert mr.speed_tier(0.5) == "muy lenta"
    assert mr.speed_tier(100) == "muy rápida"
    line = mr.format_rating_line({"score": 48.5}, {"fps": 21.8})
    assert "48" in line and "21.8" in line
    assert mr.format_rating_line(None, None) == ""
    tooltip = mr.rating_tooltip({"score": 48.5, "fidelity_score": 27,
                                 "detail_score": 60, "artifacts_score": 60},
                                {"fps": 21.8, "ms_per_frame": 45.9,
                                 "device": "cuda"})
    assert "fidelidad" in tooltip and "detalle" in tooltip


# ------------------------------------------------------- medición end-to-end
class _FixedEngine:
    """Motor de prueba: reescala con bilineal y declara la escala pedida."""

    def __init__(self, scale=4, blur=False):
        self.scale = scale
        self.blur = blur
        self.calls = 0

    def get_scale_factor(self):
        return self.scale

    def process_frame(self, frame):
        self.calls += 1
        out = mr._baseline_upscale(frame, frame.shape[0] * self.scale,
                                   frame.shape[1] * self.scale)
        return out


def test_measure_quality_reports_all_scores():
    engine = _FixedEngine(scale=4)
    rating = mr.measure_quality(engine, height=96, width=160)
    assert rating is not None
    for key in ("score", "fidelity_score", "detail_score", "artifacts_score",
                "ssim", "psnr_db", "detail_gain", "overshoot_pct",
                "colour_drift", "images", "measured_at"):
        assert key in rating, key
    assert len(rating["images"]) == 5
    # Tarea común (x2) y escala nativa real, separadas a propósito.
    assert rating["task_scale"] == mr.DEFAULT_TASK_SCALE
    assert rating["scale"] == mr.DEFAULT_TASK_SCALE
    assert rating["native_scale"] == 4
    assert rating["version"] == mr.RATING_VERSION
    assert 0.0 <= rating["score"] <= 100.0


def test_task_is_the_same_for_every_native_scale():
    """Un x2 y un x4 se miden con la MISMA degradación de referencia.

    Es la condición para poder compararlos: si al x4 se le pidiera reconstruir
    desde 1/16 de los píxeles parecería peor sólo por tener una tarea más
    difícil.
    """
    small = mr.measure_quality(_FixedEngine(scale=2), height=64, width=96)
    large = mr.measure_quality(_FixedEngine(scale=4), height=64, width=96)
    assert small and large
    assert small["task_scale"] == large["task_scale"] == mr.DEFAULT_TASK_SCALE
    assert small["resolution"] == large["resolution"]
    assert [i["name"] for i in small["images"]] == [i["name"] for i in large["images"]]
    assert small["native_scale"] == 2 and large["native_scale"] == 4


def test_native_scale_comes_from_the_output_not_the_engine():
    """Si el motor declara x2 pero devuelve x4, lo que manda es la salida real."""

    class _LyingEngine(_FixedEngine):
        def get_scale_factor(self):
            return 2

    rating = mr.measure_quality(_LyingEngine(scale=4), height=64, width=96)
    assert rating is not None and rating["native_scale"] == 4


def test_measure_quality_is_safe_with_broken_engines():
    class _Exploding:
        def process_frame(self, frame):
            raise RuntimeError("boom")

    assert mr.measure_quality(_Exploding()) is None
    assert mr.measure_quality(None) is None

    class _NoneEngine:
        def process_frame(self, frame):
            return None

    assert mr.measure_quality(_NoneEngine()) is None


def test_rating_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(mr, "_ratings_path",
                        lambda: str(tmp_path / "model_ratings.json"))
    engine = _FixedEngine(scale=4)
    fingerprint = {"size": 123, "mtime_ns": 456}
    stored = mr.measure_and_store("upscale-foo", "onnx", engine,
                                  fingerprint=fingerprint, scale=4)
    assert stored is not None
    assert mr.get_rating("upscale-foo", "onnx", fingerprint) == stored
    assert mr.get_rating("upscale-foo", "onnx") == stored
    assert mr.get_rating("upscale-foo", "torch", fingerprint) is None


def test_recommendation_picks_fastest_and_best():
    ratings = {
        "rapido": ({"score": 40.0}, {"fps": 30.0}),
        "bonito": ({"score": 70.0}, {"fps": 4.0}),
    }
    line = mr.recommendation(ratings)
    assert "rapido" in line and "bonito" in line
    assert "más rápido" in line or "ir rápido" in line
    assert mr.recommendation({}) .startswith("Todavía no hay")


# ------------------------------------------------- regresiones de arquitectura
torch = pytest.importorskip("torch")


def _small_rrdb_block():
    from src.ai.architectures import _RRDB
    torch.manual_seed(0)
    block = _RRDB(8, 4)
    block.eval()
    return block


def test_rrdb_block_has_no_activation_after_the_residual():
    """Oficial: ``rdb3(rdb2(rdb1(x))) * 0.2 + x``, sin activación detrás."""
    import torch.nn.functional as F

    block = _small_rrdb_block()
    x = torch.randn(1, 8, 6, 6)
    with torch.no_grad():
        inner = block.rdb3(block.rdb2(block.rdb1(x)))
        official = inner * 0.2 + x
        buggy = F.leaky_relu(inner * 0.2 + x, 0.2)
        got = block(x)
    assert torch.allclose(got, official, atol=1e-6)
    assert not torch.allclose(got, buggy, atol=1e-4), (
        "volvió el LeakyReLU de más tras la suma residual: oscurece la imagen"
    )


def test_rrdbnet_upsamples_before_the_conv():
    """Oficial: ``conv_up(interpolate(feat))``, no al revés."""
    import torch.nn.functional as F

    from src.ai.architectures import RRDBNet

    torch.manual_seed(1)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=8,
                    num_growth=4, num_block=1, num_feat_up=8)
    model.eval()
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        feat = model.conv_first(x)
        body_feat = feat
        for block in model.body:          # nn.ModuleList: se recorre a mano
            body_feat = block(body_feat)
        body_feat = model.conv_body(body_feat)
        feat = feat + body_feat
        up = model.lrelu(model.conv_up1(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        official = model.conv_last(model.lrelu(model.conv_hr(up)))
        wrong = model.lrelu(F.interpolate(model.conv_up1(feat),
                                          scale_factor=2, mode="nearest"))
        wrong = model.conv_last(model.lrelu(model.conv_hr(wrong)))
        got = model(x)
    assert got.shape == official.shape
    assert torch.allclose(got, official, atol=1e-5)
    assert not torch.allclose(got, wrong, atol=1e-4), (
        "volvió el orden conv_up -> interpolate: borra el detalle"
    )


def test_onnx_export_version_forces_reexport(tmp_path):
    """Un .onnx sin marca (o de versión anterior) no puede reutilizarse."""
    from src.ai import inference

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"x")
    assert inference._read_export_marker(str(onnx_path)) == 0
    inference._write_export_marker(str(onnx_path))
    assert inference._read_export_marker(str(onnx_path)) == inference.ONNX_EXPORT_VERSION
    assert inference.ONNX_EXPORT_VERSION >= 2
