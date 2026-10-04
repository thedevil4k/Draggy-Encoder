"""Robustez del pipeline: parsing del Trim y fallback del camino GPU.

Los dos nacen de fallos reales encontrados durante la implementación de P0/P1:

  * El campo End del Trim llega de la UI como TEXTO ("00:01:30") y el pipeline
    le hacía ``float()``: con ese formato lanzaba ValueError, se caía al camino
    "sin límite" y el render continuaba hasta el FINAL del vídeo. Ahora se
    parsea como en el resto de la app y, si no se entiende, se avisa.
  * Si el camino de post-proceso en GPU falla a mitad de render, la engine NO
    puede quedarse desactivada (eso dejaría el resto del vídeo sin upscale en
    silencio): debe volver al camino clásico en el mismo frame.
  * El SAR (píxeles no cuadrados) del origen se perdía de camino al output: el
    upscale es uniforme, así que basta con llevar el SAR al encoder para que el
    display aspect ratio del origen sobreviva al render.
  * La rotación del contenedor (vídeos de móvil) la aplica FFmpeg al decodificar,
    pero el pipeline dimensionaba con la geometría codificada: el reshape leía
    la fila cruzada y el vídeo salía girado Y estirado (+77 % de DAR).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.inference import ONNXRUNTIMEInferenceEngine  # noqa: E402
from src.processing.pipeline import (  # noqa: E402
    _fit_aspect_box,
    _pad_frame,
    _parse_aspect_ratio,
    _resolve_source_geometry,
    _time_to_seconds,
    compress_video,
    compress_video_with_ai,
    generate_preview,
)


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


# ---------------------------------------------------- geometría de la fuente
# El render trabaja con la geometría de la frame DECODIFICADA, no con la que
# declara el contenedor:
#   * el SAR (píxeles no cuadrados) tiene que viajar al output, porque el
#     encoder lee frames rawvideo y escribe SAR 1:1 si no se le dice (un 4:3
#     anamórfico salía 3:2: +12,5 % de ancho; un 16:9 salía 3:2: -15,6 %, lo
#     que el usuario llamaba "apretado");
#   * la rotación se la aplica FFmpeg al decodificar (``autorotate``), con un
#     ``transpose`` que cambia los lados e invierte el SAR (medido: 320x240
#     SAR 8:9 + rotation 90 -> 240x320 SAR 9:8). Si el render dimensiona con
#     la geometría codificada, el reshape lee la fila cruzada y el vídeo sale
#     girado y estirado.


def test_source_geometry_is_the_probe_verbatim_without_rotation():
    geo = _resolve_source_geometry(
        {"width": "720", "height": "480", "sample_aspect_ratio": "8:9"})
    assert (geo.width, geo.height, geo.sar, geo.rotation) == (
        720, 480, (8, 9), 0)

    # Sin SAR, con SAR cuadrado o con SAR que no se entiende: píxeles
    # cuadrados, que es lo que hacen de siempre los ficheros normales.
    for sar_text in ("1:1", "", "N/A", None, "0:1"):
        geo = _resolve_source_geometry(
            {"width": "320", "height": "240", "sample_aspect_ratio": sar_text})
        assert geo.sar == (1, 1), sar_text


def test_source_geometry_swaps_sides_and_inverts_sar_on_quarter_turns():
    for angle in ("90", "270", "-90"):
        geo = _resolve_source_geometry({
            "width": "320", "height": "240",
            "sample_aspect_ratio": "8:9", "rotation": angle,
        })
        assert (geo.width, geo.height, geo.sar) == (240, 320, (9, 8)), angle

    geo = _resolve_source_geometry({
        "width": "1920", "height": "1080", "rotation": "90",
    })
    assert (geo.width, geo.height, geo.sar) == (1080, 1920, (1, 1))


def test_source_geometry_normalises_what_ffprobe_reports():
    """360 es 0; 180 no cambia los lados; y lo que no se entiende es 0."""
    def rotation_of(value):
        return _resolve_source_geometry({
            "width": "320", "height": "240", "rotation": value}).rotation

    assert rotation_of("360") == 0
    assert rotation_of("180") == 180
    assert _resolve_source_geometry({
        "width": "320", "height": "240",
        "sample_aspect_ratio": "8:9", "rotation": "180",
    }).sar == (8, 9)
    for value in (None, "", "N/A"):
        assert rotation_of(value) == 0


def test_source_geometry_without_a_valid_resolution_is_an_error():
    for info in ({}, {"width": "0", "height": "480"}, {"width": "x"}):
        with pytest.raises(RuntimeError):
            _resolve_source_geometry(info)


#: Matriz de rotación de la caja ``tkhd`` de MP4 (16.16 punto fijo). Este
#: FFmpeg ya no escribe la matriz con ``-metadata:s:v rotate=``, así que para
#: tener un fichero de verdad con rotación hay que escribirla a mano.
_TKHD_MATRIX = {
    0:   (0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000),
    90:  (0x00000000, 0x00010000, 0, 0xFFFF0000, 0x00000000, 0, 0, 0, 0x40000000),
    180: (0xFFFF0000, 0, 0, 0, 0xFFFF0000, 0, 0, 0, 0x40000000),
    270: (0x00000000, 0xFFFF0000, 0, 0x00010000, 0x00000000, 0, 0, 0, 0x40000000),
}


def _tag_rotation(path, degrees):
    """Escribe ``rotation`` en la matriz de la pista de vídeo de un MP4.

    ffprobe lo lee como side data y FFmpeg lo aplica al decodificar, que es
    justo lo que tiene que reproducirse en el test. Sólo se busca la matriz
    identidad que escribe el muxer, así que el fichero tiene que ser recién
    codificado por ffmpeg.
    """
    import struct

    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    identity = struct.pack(">9I", *_TKHD_MATRIX[0])
    spot = data.find(b"tkhd")
    assert spot > 0, "el MP4 no tiene caja tkhd"
    matrix = data.find(identity, spot)
    assert matrix > 0, "no se encuentra la matriz identidad en tkhd"
    struct.pack_into(">9I", data, matrix, *_TKHD_MATRIX[(-degrees) % 360])
    with open(path, "wb") as fh:
        fh.write(data)


@pytest.fixture
def ffmpeg_pair():
    """``(ffmpeg, ffprobe)`` del checkout, o skip si no hay binarios."""
    from src.ffmpeg.manager import (
        get_ffmpeg_dir,
        get_ffprobe_path,
        get_local_ffmpeg_path,
    )

    ffmpeg = get_local_ffmpeg_path()
    if not (ffmpeg and os.path.exists(ffmpeg)):
        ffmpeg = os.path.join(get_ffmpeg_dir(), "ffmpeg.exe")
    ffprobe = get_ffprobe_path()
    if not (os.path.exists(ffmpeg) and ffprobe and os.path.exists(ffprobe)):
        pytest.skip("FFmpeg/FFprobe not available in this checkout")
    return ffmpeg, ffprobe


def _make_clip(ffmpeg, path, size="320x240", sar=None):
    import subprocess

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc2=size={size}:rate=5:duration=1"]
    if sar:
        cmd += ["-vf", f"setsar={sar}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    if subprocess.run(cmd, capture_output=True, timeout=120).returncode != 0:
        pytest.skip("could not build the test clip")


def _probe_stream(ffprobe, path):
    import json
    import subprocess

    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,sample_aspect_ratio,display_aspect_ratio"
         ":stream_side_data=rotation", "-of", "json", path],
        capture_output=True, text=True, timeout=60,
    )
    return json.loads(out.stdout)["streams"][0]


class _PassthroughUpscaler:
    """Engine x4 sin modelo: el pipeline ajusta el tamaño con su propio resize."""

    engine_id = "stub-x4"
    _mode = "upscale"

    def get_scale_factor(self):
        return 4

    def process_frame(self, frame):
        return frame.copy()


# (rotación, SAR del origen) -> (tamaño y SAR que debe tener el render). El
# origen mide 320x240, así que un cuarto de vuelta se ve 240x320 y el render lo
# tiene que respetar (x4). El SAR sale invertido porque el transpose de
# FFmpeg también lo invierte.
@pytest.mark.parametrize("rotation,sar,expected", [
    (90, None, ((960, 1280), (1, 1))),
    (-90, None, ((960, 1280), (1, 1))),
    (90, "8/9", ((960, 1280), (9, 8))),
])
def test_ai_render_follows_the_source_rotation(tmp_path, ffmpeg_pair,
                                              rotation, sar, expected):
    """El render sale con la MISMA orientación y proporción que mostraba el origen."""
    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    _make_clip(ffmpeg, src, sar=sar)
    _tag_rotation(src, rotation)

    ok = compress_video_with_ai(
        ffmpeg, src, dst,
        {"codec": "h264", "resolution": "original", "ai_scale": 4,
         "bitrate": 2000, "audio": "noaudio", "audio_boost": False,
         "format": "mp4", "preset": "veryfast", "pixel_format": "yuv420p"},
        inference_engine=_PassthroughUpscaler(),
    )
    assert ok is True and os.path.exists(dst)

    stream = _probe_stream(ffprobe, dst)
    assert (stream["width"], stream["height"]) == expected[0], stream
    num, den = (int(v) for v in str(
        stream.get("sample_aspect_ratio") or "1:1").split(":"))
    assert (num, den) == expected[1], stream
    # Los píxeles ya vienen girados de FFmpeg: el output NO debe llevar
    # rotación, o el reproductor lo giraría por segunda vez.
    assert not any("rotation" in side
                   for side in (stream.get("side_data_list") or [])), stream


@pytest.mark.parametrize("setsar,expected_dar", [("1/1", "3:2"), ("8/9", "4:3")])
def test_ai_render_keeps_the_source_display_aspect(tmp_path, ffmpeg_pair,
                                                   setsar, expected_dar):
    """Render real: el SAR del origen tiene que acabar en el fichero."""
    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    sar_num, sar_den = (int(v) for v in setsar.split("/"))
    _make_clip(ffmpeg, src, size="720x480", sar=setsar)

    messages = []
    ok = compress_video_with_ai(
        ffmpeg, src, dst,
        {"codec": "h264", "resolution": "original", "bitrate": 2000,
         "audio": "noaudio", "audio_boost": False, "format": "mp4",
         "preset": "veryfast", "pixel_format": "yuv420p", "ai_scale": 4},
        inference_engine=_PassthroughUpscaler(),
        status_callback=messages.append,
    )
    assert ok is True and os.path.exists(dst)

    stream = _probe_stream(ffprobe, dst)
    assert (stream["width"], stream["height"]) == (2880, 1920)
    # Lo que ve el reproductor: SAR x ancho/alto = DAR del origen.
    num, den = (int(v) for v in str(
        stream.get("sample_aspect_ratio") or "1:1").split(":"))
    got_dar = stream["width"] * num / (stream["height"] * den)
    assert got_dar == pytest.approx(720 * sar_num / (480 * sar_den), abs=1e-3)
    # ffprobe omite los campos DAR/SAR cuando valen 1:1; si estan, tienen que
    # ser los del origen.
    dar_field = stream.get("display_aspect_ratio")
    assert dar_field in (expected_dar, None), stream
    if sar_num != sar_den:
        # El render lo dice en voz alta: se ve en el log y en la UI.
        assert any("not square" in m and expected_dar in m
                   for m in messages), messages


# ------------------------------------------------- relación de aspecto salida
# El desplegable "Aspect Ratio" de Compression Settings tiene dos valores muy
# distintos: "source" (lo de siempre, la caja no se toca) y un ratio explícito,
# que tiene que cumplirse de verdad: caja con ESE ratio, imagen sin deformar y
# barras negras para llenar. Las dos rutas de render (IA y compresión estándar)
# parten de la misma caja, así que un 4:3 no sale estirado en ninguna.


def test_aspect_option_only_forces_a_ratio_when_it_really_is_one():
    assert _parse_aspect_ratio("16:9") == (16, 9)
    assert _parse_aspect_ratio(" 9:16 ") == (9, 16)
    # "source" y todo lo que no se entiende dejan el vídeo como estaba.
    for value in ("source", "Original", "", None, "16-9", "16", "0:9", "16:0",
                  "x:y", "1:2:3"):
        assert _parse_aspect_ratio(value) is None, value


# (tamaño VISIBLE de la imagen, altura de export, ratio) -> caja e imagen.
# "Visible" porque un 720x480 SAR 8:9 se ve 640x480: el encaje se calcula sobre
# eso, no sobre los bytes del contenedor.
@pytest.mark.parametrize("display,export_h,ratio,expected", [
    ((640, 480), 480, (16, 9), ((854, 480), (640, 480))),
    ((1920, 1080), 1080, (9, 16), ((608, 1080), (608, 342))),
    ((1920, 1080), None, (4, 3), ((1920, 1440), (1920, 1080))),
    ((1080, 1920), None, (1, 1), ((1920, 1920), (1080, 1920))),
    ((1920, 1080), 480, (18, 9), ((960, 480), (854, 480))),
    ((320, 240), None, (16, 9), ((426, 240), (320, 240))),
])
def test_forced_ratio_box_keeps_the_image_undistorted(display, export_h, ratio,
                                                      expected):
    box = _fit_aspect_box(display[0], display[1], export_h, ratio)
    assert box == expected[0] + expected[1], box
    out_w, out_h, img_w, img_h = box
    # La caja es el ratio pedido (salvo el redondeo a dimensiones pares)...
    assert out_w / out_h == pytest.approx(ratio[0] / ratio[1], abs=5e-3), box
    # ...la imagen mantiene la proporción con la que se ve el origen...
    assert img_w / img_h == pytest.approx(
        display[0] / display[1], abs=5e-3), box
    # ...y cabe entera dentro de la caja (barras, nunca recorte).
    assert img_w <= out_w and img_h <= out_h, box
    for value in box:
        assert value % 2 == 0 and value >= 2, box


def test_forced_ratio_box_keeps_the_requested_height():
    """Con resolución elegida manda la altura; el ratio da el ancho."""
    for export_h in (240, 480, 720, 1080):
        out_w, out_h, _img_w, _img_h = _fit_aspect_box(
            1920, 1080, export_h, (16, 9))
        assert out_h == export_h, export_h
        assert out_w / out_h == pytest.approx(16 / 9, abs=5e-3), export_h


def test_without_a_forced_ratio_there_is_no_box_at_all():
    """Sin ratio forzado la geometría es la de siempre: nada que recalcular."""
    assert _fit_aspect_box(1920, 1080, 480, None) is None
    # Una imagen que no se puede medir tampoco da caja (y así no se rompe).
    assert _fit_aspect_box(0, 0, 480, (16, 9)) is None


def test_padding_centres_the_image_on_black():
    frame = np.full((2, 4, 3), 255, dtype=np.uint8)
    padded = _pad_frame(frame, 10, 6)
    assert padded.shape == (6, 10, 3)
    assert padded.dtype == np.uint8
    # Centrado: la imagen cae en las filas 2-3 y las columnas 3-6.
    assert np.array_equal(padded[3, 5], [255, 255, 255])
    assert padded[0].sum() == 0 and padded[5].sum() == 0
    assert padded[2, 0].sum() == 0 and padded[3, 9].sum() == 0
    # Una imagen que ya es del tamaño de la caja sale intacta.
    assert _pad_frame(padded, 10, 6) is padded


def _first_frame_rgb(ffmpeg, path, width, height):
    """Primer frame en RGB plano, para mirar las barras de verdad."""
    import subprocess

    out = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr[-400:]
    return np.frombuffer(
        out.stdout[:width * height * 3], dtype=np.uint8).reshape(height, width, 3)


def test_ai_render_honours_a_forced_aspect_ratio(tmp_path, ffmpeg_pair):
    """Render real con 16:9 pedido sobre un 4:3: caja 16:9 e imagen 4:3."""
    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    _make_clip(ffmpeg, src)  # 320x240 (4:3), x4 = 1280x960

    messages = []
    ok = compress_video_with_ai(
        ffmpeg, src, dst,
        {"codec": "h264", "resolution": "original", "aspect": "16:9",
         "ai_scale": 4, "bitrate": 2000, "audio": "noaudio", "audio_boost": False,
         "format": "mp4", "preset": "veryfast", "pixel_format": "yuv420p"},
        inference_engine=_PassthroughUpscaler(),
        status_callback=messages.append,
    )
    assert ok is True and os.path.exists(dst)

    stream = _probe_stream(ffprobe, dst)
    out_w, out_h = stream["width"], stream["height"]
    assert out_h == 960, stream               # la altura manda (x4 de 240)
    assert out_w / out_h == pytest.approx(16 / 9, abs=5e-3), stream
    # Las barras se rellenan con píxeles cuadrados: nada de -aspect aquí.
    num, den = (int(v) for v in str(
        stream.get("sample_aspect_ratio") or "1:1").split(":"))
    assert (num, den) == (1, 1), stream
    # La imagen no está estirada: en el centro sigue habiendo dibujo y en el
    # borde de la caja, negro.
    frame = _first_frame_rgb(ffmpeg, dst, out_w, out_h)
    assert frame[out_h // 2, 0].sum() == 0, "la barra izquierda no es negra"
    assert frame[0, out_w // 2].max() > 0, "el centro no debería ser negro"
    assert any("16:9" in m for m in messages), messages


def test_ai_render_forced_ratio_undoes_anamorphic_pixels(tmp_path, ffmpeg_pair):
    """Un 4:3 anamórfico (SAR 8:9) a 16:9 sale 4:3 de verdad, no estirado."""
    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    _make_clip(ffmpeg, src, size="720x480", sar="8/9")  # se ve 640x480

    ok = compress_video_with_ai(
        ffmpeg, src, dst,
        {"codec": "h264", "resolution": "480", "aspect": "16:9", "ai_scale": 4,
         "bitrate": 2000, "audio": "noaudio", "audio_boost": False,
         "format": "mp4", "preset": "veryfast", "pixel_format": "yuv420p"},
        inference_engine=_PassthroughUpscaler(),
    )
    assert ok is True and os.path.exists(dst)

    stream = _probe_stream(ffprobe, dst)
    assert (stream["width"], stream["height"]) == (854, 480), stream
    num, den = (int(v) for v in str(
        stream.get("sample_aspect_ratio") or "1:1").split(":"))
    assert (num, den) == (1, 1), stream
    # La imagen dentro de la caja es 720x480 (4:3), no 854x480 (1.78): con el
    # SAR a 1:1 la única forma de que se vea 4:3 es respetarlo.
    frame = _first_frame_rgb(ffmpeg, dst, 854, 480)
    assert frame[240, 0].sum() == 0, "no hay barra a la izquierda"
    assert frame[240, 853].sum() == 0, "no hay barra a la derecha"
    assert frame[240, 427].max() > 0, "el centro debería tener imagen"


def test_standard_compress_honours_a_forced_aspect_ratio(tmp_path, ffmpeg_pair):
    """La ruta estándar (sin IA) cumple el mismo ratio que la de IA."""
    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    _make_clip(ffmpeg, src, size="720x480", sar="8/9")  # se ve 640x480

    ok = compress_video(
        ffmpeg, src, dst,
        {"codec": "h264", "resolution": "480", "aspect": "16:9",
         "bitrate": 2000, "audio": "noaudio", "audio_boost": False,
         "format": "mp4", "preset": "veryfast", "pixel_format": "yuv420p"},
    )
    assert ok is True and os.path.exists(dst)

    stream = _probe_stream(ffprobe, dst)
    assert (stream["width"], stream["height"]) == (854, 480), stream
    # ffprobe reduce el DAR: con ancho y alto pares sale 427:240, que es el
    # 16:9 que cabe en 480 de alto.
    dar = stream.get("display_aspect_ratio")
    if dar:
        num, den = (int(v) for v in str(dar).split(":"))
        assert num / den == pytest.approx(16 / 9, abs=5e-3), stream
    num, den = (int(v) for v in str(
        stream.get("sample_aspect_ratio") or "1:1").split(":"))
    assert (num, den) == (1, 1), stream
    frame = _first_frame_rgb(ffmpeg, dst, 854, 480)
    assert frame[240, 0].sum() == 0 and frame[240, 853].sum() == 0
    assert frame[240, 427].max() > 0


def test_standard_compress_without_a_forced_ratio_is_untouched(tmp_path,
                                                               ffmpeg_pair):
    """"Keep source aspect ratio" no cambia el vídeo: ni un filtro de más."""
    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    _make_clip(ffmpeg, src, size="720x480", sar="8/9")  # se ve 640x480

    ok = compress_video(
        ffmpeg, src, dst,
        {"codec": "h264", "resolution": "480", "aspect": "source",
         "bitrate": 2000, "audio": "noaudio", "audio_boost": False,
         "format": "mp4", "preset": "veryfast", "pixel_format": "yuv420p"},
    )
    assert ok is True and os.path.exists(dst)

    stream = _probe_stream(ffprobe, dst)
    assert (stream["width"], stream["height"]) == (720, 480), stream
    # El SAR anamórfico sobrevive, así que el reproductor lo sigue viendo 4:3.
    num, den = (int(v) for v in str(
        stream.get("sample_aspect_ratio") or "1:1").split(":"))
    assert (num, den) == (8, 9), stream
    assert stream["width"] * num / (stream["height"] * den) == pytest.approx(
        4 / 3, abs=1e-3)


def test_preview_encodes_the_same_forced_ratio_as_the_render(tmp_path,
                                                             ffmpeg_pair):
    """El preview tiene que enseñar exactamente lo que va a salir."""
    import shutil

    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    _make_clip(ffmpeg, src, size="720x480", sar="8/9")  # se ve 640x480

    _original, _compressed, artifacts = generate_preview(
        ffmpeg, src,
        {"codec": "h264", "resolution": "480", "aspect": "16:9", "bitrate": 2000,
         "audio": "noaudio", "audio_boost": False, "format": "mp4",
         "preset": "veryfast", "pixel_format": "yuv420p"},
    )
    try:
        stream = _probe_stream(ffprobe, artifacts["compressed_clip"])
        assert (stream["width"], stream["height"]) == (854, 480), stream
    finally:
        shutil.rmtree(artifacts["dir"], ignore_errors=True)
