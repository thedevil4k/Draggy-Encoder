"""Tests de la telemetría por etapa (P0) y del envoltorio de grafo (P1).

El objetivo de estos tests es doble:
  * que la telemetría no pueda romper un render (nunca lanza, memoria acotada),
  * que el grafo con cabeza/cola haga EXACTAMENTE la misma cuenta que el
    camino clásico cuando la reescalada no cambia (comparación en CPU).

No necesitan GPU: el grafo se ejecuta con CPUExecutionProvider.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.timing import StageTimer  # noqa: E402


# --------------------------------------------------------------- StageTimer
def test_stage_timer_accumulates_and_summarises():
    timer = StageTimer()
    for value in (10.0, 20.0, 30.0, 40.0):
        timer.add("decode", value)
    timer.add("resize", 5.0)
    for _ in range(3):
        timer.mark_frame()

    summary = timer.summary()
    assert summary["frames"] == 3
    assert summary["stages"]["decode"]["n"] == 4
    assert summary["stages"]["decode"]["median_ms"] == pytest.approx(25.0)
    assert summary["stages"]["decode"]["max_ms"] == pytest.approx(40.0)
    assert summary["stages"]["decode"]["mean_ms"] == pytest.approx(25.0)
    assert summary["sum_of_medians_ms"] == pytest.approx(30.0)
    assert summary["fps_implied_by_stages"] == pytest.approx(1000 / 30.0, abs=0.01)


def test_stage_timer_ignores_garbage_without_raising():
    timer = StageTimer()
    timer.add("decode", "no soy un número")
    timer.add("decode", None)
    timer.add("decode", float("nan"))
    timer.add("decode", -5)
    assert timer.summary()["stages"] == {}

    # Las excepciones dentro del bloque medido no se tragan.
    with pytest.raises(ValueError):
        with timer.stage("boom"):
            raise ValueError("nope")
    assert timer.summary()["stages"]["boom"]["n"] == 1


def test_stage_timer_memory_is_bounded():
    timer = StageTimer(cap=100)
    for value in range(1000):
        timer.add("decode", float(value))
    assert len(timer.samples("decode")) == 100
    assert timer.summary()["stages"]["decode"]["n"] == 1000


# ---------------------------------------------------------------- gpu_post
from src.ai import gpu_post  # noqa: E402


def _tiny_model(tmp_path):
    """Modelo ONNX mínimo: un Conv 1x1 identidad (sin torch, sin DepthToSpace).

    Deliberadamente sin PixelShuffle: lo que se prueba aquí es la cabeza y la
    cola del envoltorio (cast, orden de canales, transposes, escala 0..255),
    no el layout de canales de DepthToSpace (que además depende del modo
    DCR/CRD del exportador).
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weights = np.zeros((3, 3, 1, 1), dtype=np.float32)
    for channel in range(3):
        weights[channel, channel, 0, 0] = 1.0
    node_conv = helper.make_node("Conv", ["input", "w"], ["output"],
                                 kernel_shape=[1, 1])
    graph = helper.make_graph(
        [node_conv], "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["n", 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["n", 3, "oh", "ow"])],
        [numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = os.path.join(str(tmp_path), "tiny.onnx")
    onnx.save(model, path)
    return path, 1


def _reference_math(frame, scale=1):
    """Lo que debe devolver el envoltorio: RGB->modelo->x255->clip->uint8->BGR."""
    rgb = frame[:, :, ::-1].astype(np.float32)
    big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return np.ascontiguousarray(
        np.clip(big, 0, 255).astype(np.uint8)[:, :, ::-1]
    )


def test_wrapped_graph_matches_classic_math(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    src, scale = _tiny_model(tmp_path)
    frame = gpu_post.sample_frames(16, 24)[0]

    wrapped, info = gpu_post.build_wrapped_model(
        src, out_size=None, precision="model")
    assert wrapped and os.path.isfile(wrapped), info
    assert info["reason"] if "reason" in info else True

    session = ort.InferenceSession(wrapped, providers=["CPUExecutionProvider"])
    got = session.run([session.get_outputs()[0].name],
                      {session.get_inputs()[0].name: frame[None, ...]})[0][0]
    expected = _reference_math(frame, scale)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected), "el envoltorio no reproduce la cuenta clásica"


def test_wrapped_graph_resizes_to_encoder_size(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    src, _scale = _tiny_model(tmp_path)
    frame = gpu_post.sample_frames(16, 24)[0]

    wrapped, _info = gpu_post.build_wrapped_model(
        src, out_size=(32, 20), precision="model")
    session = ort.InferenceSession(wrapped, providers=["CPUExecutionProvider"])
    got = session.run([session.get_outputs()[0].name],
                      {session.get_inputs()[0].name: frame[None, ...]})[0]
    assert got.shape == (1, 20, 32, 3)
    assert got.dtype == np.uint8


def test_wrapped_graph_fp16_keeps_io_uint8(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    src, _scale = _tiny_model(tmp_path)
    wrapped, info = gpu_post.build_wrapped_model(src, precision="fp16")
    if wrapped is None:
        pytest.skip(f"conversor fp16 no disponible: {info.get('reason')}")
    model = onnx.load(wrapped)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.UINT8
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.UINT8
    dtypes = {onnx.numpy_helper.to_array(i).dtype for i in model.graph.initializer}
    assert np.dtype(np.float16) in dtypes or np.dtype(np.float32) in dtypes


def test_build_wrapped_model_is_cached(tmp_path):
    src, _scale = _tiny_model(tmp_path)
    first, info_first = gpu_post.build_wrapped_model(src, out_size=None)
    second, info_second = gpu_post.build_wrapped_model(src, out_size=None)
    assert first == second
    assert info_second.get("cached") is True
    assert info_first.get("cached") is False


def test_self_check_rejects_flat_and_swapped_outputs():
    reference = gpu_post.sample_frames(64, 64)[0]
    frames = [reference]

    # Salida plana: debe rechazarse.
    flat = np.full_like(reference, 7)
    ok, details = gpu_post.run_self_check(
        lambda _f: flat, lambda _f: reference, frames=frames)
    assert not ok and "flat" in details["reason"]

    # Canales invertidos: las medias por canal lo delatan.
    swapped = np.ascontiguousarray(reference[:, :, ::-1])
    ok, details = gpu_post.run_self_check(
        lambda _f: swapped, lambda _f: reference, frames=frames)
    assert not ok and "channel" in details["reason"]

    # Misma imagen con otro filtro (diferencia local pequeña): debe aceptarse.
    blurred = (reference.astype(np.float32) * 0.98 + 2).astype(np.uint8)
    ok, details = gpu_post.run_self_check(
        lambda _f: blurred, lambda _f: reference, frames=frames)
    assert ok, details["reason"]


def test_evaluate_check_shape_mismatch():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 5, 3), dtype=np.uint8)
    stats = gpu_post.compare_frames(a, b)
    assert stats["shape_ok"] is False
    ok, reason = gpu_post.evaluate_check(stats)
    assert not ok and "shape" in reason


def test_precision_preference_normalisation(monkeypatch):
    import src.preferences as preferences

    cases = {
        "fp16": "fp16", "FP16": "fp16", "half": "fp16",
        "fp32": "fp32", "FULL": "fp32",
        "model": "model", "": "model", "cualquier cosa": "model",
    }
    for raw, expected in cases.items():
        monkeypatch.setattr(preferences, "get", lambda key, default=None, _v=raw: _v)
        assert gpu_post.get_precision_preference() == expected
