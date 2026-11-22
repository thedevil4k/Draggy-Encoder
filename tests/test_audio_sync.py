"""Pruebas de la detección de desfase entre dos doblajes (sin ffmpeg)."""

import numpy as np
import pytest

from src.processing import audio_sync

RATE = audio_sync.SAMPLE_RATE


def _envelope(frames, seed):
    rng = np.random.default_rng(seed)
    raw = rng.random(frames)
    # Suavizado: una envolvente real no cambia de un valor al siguiente.
    return np.convolve(raw, np.ones(7) / 7, mode="same")


def _audio(envelope, carrier):
    frame = int(round(RATE * audio_sync.FRAME_SECONDS))
    total = envelope.size * frame
    time = np.arange(total) / RATE
    amplitude = np.repeat(envelope, frame)
    return (amplitude * np.sin(2 * np.pi * carrier * time)).astype(np.float32)


def test_envelope_has_one_value_per_frame():
    samples = np.ones(RATE, dtype=np.float32)  # 1 segundo
    envelope = audio_sync.energy_envelope(samples)
    assert envelope.size == int(round(1 / audio_sync.FRAME_SECONDS))


def test_delay_is_detected_between_two_languages():
    envelope = _envelope(600, seed=3)
    base = _audio(envelope, 440.0)

    shift = 100  # 2 segundos a 20 ms por trama
    delayed = np.concatenate([np.zeros(shift), envelope[:-shift]])
    # Misma escena (misma envolvente) en otro idioma: otra frecuencia.
    dub = _audio(delayed, 880.0)

    offset, confidence = audio_sync.estimate_offset(base, dub)
    # El doblaje va 2 s por detrás: hay que adelantarlo.
    assert offset == pytest.approx(
        -shift * audio_sync.FRAME_SECONDS, abs=0.04
    )
    assert confidence > 0.6


def test_a_dub_ahead_of_the_video_needs_a_positive_delay():
    envelope = _envelope(600, seed=11)
    shift = 50  # 1 segundo
    early = np.concatenate([envelope[shift:], np.zeros(shift)])
    base = _audio(envelope, 300.0)
    dub = _audio(early, 700.0)

    offset, confidence = audio_sync.estimate_offset(base, dub)
    assert offset == pytest.approx(
        shift * audio_sync.FRAME_SECONDS, abs=0.04
    )
    assert confidence > 0.6


def test_unrelated_audio_is_not_reported_as_a_match():
    base = _audio(_envelope(600, seed=1), 440.0)
    dub = _audio(_envelope(600, seed=2), 660.0)

    _offset, confidence = audio_sync.estimate_offset(base, dub)
    assert confidence < audio_sync.MIN_CONFIDENCE


def test_too_little_audio_is_rejected():
    with pytest.raises(ValueError, match="Not enough"):
        audio_sync.estimate_offset(
            np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32)
        )


def test_peak_frames_keeps_one_peak_per_frame():
    samples = np.array(
        [0, 1000, -32000, 200, 4000, -500, 8000, 100], dtype=np.int16
    )
    peaks = audio_sync.peak_frames(samples, 2)
    assert peaks.size == 4
    assert peaks[0] == pytest.approx(1000 / 32768, abs=1e-6)
    assert peaks[1] == pytest.approx(32000 / 32768, abs=1e-6)
    assert peaks[2] == pytest.approx(4000 / 32768, abs=1e-6)
    assert peaks[3] == pytest.approx(8000 / 32768, abs=1e-6)


def test_align_clip_delays_the_dub_with_silence():
    # 1 kHz para que una muestra sea 1 ms y el desplazamiento sea exacto.
    raw = np.arange(1, 11, dtype=np.int16)
    aligned = audio_sync.align_clip(
        raw, 0.0, 0.0, 10, 0.002, sample_rate=1000,
    )
    assert aligned.tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_align_clip_advances_the_dub_and_pads_the_tail():
    raw = np.arange(1, 11, dtype=np.int16)
    aligned = audio_sync.align_clip(
        raw, 0.0, 0.0, 10, -0.002, sample_rate=1000,
    )
    assert aligned.tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 0, 0]


def test_align_clip_reads_from_the_right_place_in_the_chunk():
    # El trozo empieza en 100 s; la ventana en 130 s con 1 s de desfase oye
    # el audio del segundo 129, que en el trozo es la muestra 29 (a 1 kHz).
    raw = np.arange(200, dtype=np.int16)
    aligned = audio_sync.align_clip(
        raw, 100.0, 130.0, 5, 1.0, sample_rate=1,
    )
    assert aligned.tolist() == [29, 30, 31, 32, 33]


def test_align_clip_without_a_dub_is_silence():
    aligned = audio_sync.align_clip(
        np.zeros(0, dtype=np.int16), 0.0, 0.0, 6, 0.0,
    )
    assert aligned.tolist() == [0, 0, 0, 0, 0, 0]
