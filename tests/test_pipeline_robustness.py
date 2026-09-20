"""Robustez del pipeline: parsing del Trim y fallback del camino GPU.

Los dos nacen de fallos reales encontrados durante la implementación de P0/P1:

  * El campo End del Trim llega de la UI como TEXTO ("00:01:30") y el pipeline
    le hacía ``float()``: con ese formato lanzaba ValueError, se caía al camino
    "sin límite" y el render continuaba hasta el FINAL del vídeo. Ahora se
    parsea como en el resto de la app y, si no se entiende, se avisa.
  * Si el camino de post-proceso en GPU falla a mitad de render, la engine NO
    puede quedarse desactivada (eso dejaría el resto del vídeo sin upscale en
    silencio): debe volver al camino clásico en el mismo frame.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.inference import ONNXRUNTIMEInferenceEngine  # noqa: E402
from src.processing.pipeline import _time_to_seconds  # noqa: E402


# ------------------------------------------------------------------- Trim
def test_trim_accepts_hhmmss_from_the_ui():
    assert _time_to_seconds("00:00:06", None) == pytest.approx(6.0)
    assert _time_to_seconds("00:01:30", None) == pytest.approx(90.0)
    assert _time_to_seconds("1:02:03", None) == pytest.approx(3723.0)
    assert _time_to_seconds("12.5", None) == pytest.approx(12.5)
    assert _time_to_seconds(7, None) == pytest.approx(7.0)


def test_trim_empty_uses_default():
    assert _time_to_seconds(None, 3.0) == pytest.approx(3.0)
    assert _time_to_seconds("", 0.0) == pytest.approx(0.0)


def test_trim_invalid_warns_instead_of_silently_ignoring():
    messages = []
    assert _time_to_seconds("no soy una hora", 0.0, "end_time",
                            status_callback=messages.append) == pytest.approx(0.0)
    assert messages and "end_time" in messages[0]


# ----------------------------------------------- fallback del camino GPU
class _StubEngine(ONNXRUNTIMEInferenceEngine):
    """Engine sin modelo real: el camino clásico solo cuenta las llamadas."""

    def __init__(self, state, run_error=None):
        super().__init__("does-not-exist.onnx", "stub")
        self._loaded = True
        self._session = object()
        self._gpu_post = state
        self.classic_calls = 0
        self._run_error = run_error

    def process_frame(self, frame):  # camino clásico simulado
        self.classic_calls += 1
        return frame.copy()

    def _run_with_state(self, state, frame):  # usado por la autocomprobación
        if self._run_error:
            raise self._run_error
        return frame


class _FailingSession:
    def run(self, *_args, **_kwargs):
        raise RuntimeError("CUDA out of memory (simulado)")


def _frame():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_gpu_post_with_incomplete_state_falls_back():
    engine = _StubEngine({"active": True, "session": None})
    out = engine._process_frame_gpu_post(_frame())
    assert out.shape == (8, 8, 3)
    assert engine.classic_calls == 1
    assert engine.gpu_post_active() is False


def test_gpu_post_aborts_on_unexpected_input_size():
    state = {"active": True, "session": _FailingSession(),
             "input_name": "input", "output_name": "out", "input_size": (64, 64)}
    engine = _StubEngine(state)
    out = engine._process_frame_gpu_post(_frame())  # 8x8, no 64x64
    assert engine.classic_calls == 1
    assert engine.gpu_post_active() is False
    assert "input size changed" in engine._gpu_post_failure
    assert out.shape == (8, 8, 3)


def test_gpu_post_aborts_when_the_wrapped_graph_fails():
    state = {"active": True, "session": _FailingSession(),
             "input_name": "input", "output_name": "out", "input_size": (8, 8)}
    engine = _StubEngine(state)
    out = engine._process_frame_gpu_post(_frame())
    assert engine.classic_calls == 1, "debe procesar el frame por el camino clásico"
    assert engine.gpu_post_active() is False
    assert "out of memory" in engine._gpu_post_failure
    assert out.shape == (8, 8, 3)


def test_images_are_still_produced_after_the_fallback():
    """El frame siguiente ya no intenta el camino GPU y sale por el clásico."""
    state = {"active": True, "session": _FailingSession(),
             "input_name": "input", "output_name": "out", "input_size": (8, 8)}
    engine = _StubEngine(state)
    engine._process_frame_gpu_post(_frame())
    engine.process_frame(_frame())
    engine.process_frame(_frame())
    assert engine.classic_calls == 3
