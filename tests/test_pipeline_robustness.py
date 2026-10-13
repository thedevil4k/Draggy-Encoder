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


def test_preview_retries_the_frame_when_the_seek_lands_past_the_clip(
        tmp_path, ffmpeg_pair, monkeypatch):
    """El preview no falla cuando el instante sorteado cae tras el último frame.

    Fallo real medido: el ``-ss`` de entrada de FFmpeg descarta lo anterior al
    instante pedido, así que en un clip corto (1 s a 5 fps: el hueco entre el
    último frame y el final es de 0,2 s) el frame pedido podía no existir. De 30
    previews sobre ese mismo clip, 6 fallaban (20%) con "Could not extract a
    frame from the compressed preview clip" -- y el instante se sortea con
    semilla del path, así que le tocaba a cualquiera.
    """
    import shutil

    import src.processing.pipeline as pipeline

    ffmpeg, _ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    _make_clip(ffmpeg, src, size="720x480")  # 1 s a 5 fps

    real = pipeline.extract_frame
    attempts = []

    def flaky(ffmpeg_path, video_path, time_sec, output_path, **kwargs):
        attempts.append((os.path.basename(video_path), round(float(time_sec), 3)))
        # El primer intento sobre el clip YA codificado falla, tal y como falla
        # cuando el seek de entrada cae después de su último frame.
        if len(attempts) == 1 and float(time_sec) > 0:
            return False
        return real(ffmpeg_path, video_path, time_sec, output_path, **kwargs)

    monkeypatch.setattr(pipeline, "extract_frame", flaky)

    _original, _compressed, artifacts = generate_preview(
        ffmpeg, src,
        {"codec": "h264", "resolution": "480", "aspect": "16:9", "bitrate": 2000,
         "audio": "noaudio", "audio_boost": False, "format": "mp4",
         "preset": "veryfast", "pixel_format": "yuv420p"},
    )
    try:
        comprimido = [t for name, t in attempts
                      if name.startswith("preview_compressed")]
        original = [t for name, t in attempts if name == "in.mp4"]
        assert comprimido, attempts
        assert len(comprimido) == 2, attempts  # falló el primero: se reintentó
        assert comprimido[1] < comprimido[0], comprimido
        # El antes/después tiene que ser el MISMO momento: el frame original se
        # extrae en el instante del reintento, desplazado por el inicio del clip
        # (el 20 % de un video de 1 s).
        assert len(original) == 1, attempts
        assert original[0] - comprimido[1] == pytest.approx(0.2, abs=0.05)
    finally:
        shutil.rmtree(artifacts["dir"], ignore_errors=True)


# ------------------------------- staging del mux y limpieza de restos
# Fallo real medido: el temporal del mux nacía en %TEMP% (unidad del sistema)
# mientras el destino estaba en otra unidad (D:, E:, NAS). El ``os.replace``
# final devolvía WinError 17 (cross-device) y el mux tiraba un vídeo YA
# procesado entero. Estos tests fijan las dos mitades del arreglo: el temporal
# vive junto al destino y la limpieza barre de verdad esa carpeta.
import errno  # noqa: E402  (solo lo usa el shim de Windows de abajo)
import subprocess  # noqa: E402


class _WindowsReplaceShim:
    """``os`` con el ``os.replace`` de Windows: entre unidades, OSError.

    Cualquier ``replace`` entre carpetas distintas falla igual que en Windows
    (``WinError 17`` = ``EXDEV``), así que un test que pase con este shim
    demuestra que el mux no necesita cruzar de unidad.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def replace(self, src, dst):
        src_dir = os.path.dirname(os.path.abspath(src))
        dst_dir = os.path.dirname(os.path.abspath(dst))
        if src_dir != dst_dir:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return self._real.replace(src, dst)


class _FakeFFmpeg:
    """ffmpeg doblado: escribe en su último argumento y sale con éxito."""

    def __init__(self, command, **kwargs):
        self.command = command
        self.returncode = 0
        with open(command[-1], "wb") as fh:
            fh.write(b"muxed")

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def test_mux_stages_beside_the_destination_not_in_temp(tmp_path, monkeypatch):
    """El temporal del mux se pide para la carpeta del destino (no %TEMP%)."""
    import src.processing.pipeline as pipeline

    destino = tmp_path / "render.mp4"
    pedidos = []
    real_make = pipeline._make_temp_output_path
    monkeypatch.setattr(
        pipeline, "_make_temp_output_path",
        lambda path: (pedidos.append(path), real_make(path))[1],
    )
    monkeypatch.setattr(pipeline.subprocess, "Popen", _FakeFFmpeg)
    monkeypatch.setattr(pipeline, "os", _WindowsReplaceShim(os))

    assert pipeline._mux_audio("ffmpeg", "entrada.mp4", str(destino),
                               audio_codec="copy", settings={}) is True
    assert pedidos == [str(destino)], "el temporal no nace junto al destino"
    assert destino.read_bytes() == b"muxed"
    assert not list(tmp_path.glob(".draggy-render-*"))
    assert not list(tmp_path.glob(".mux-*"))


def _make_clip_with_audio(ffmpeg, path, size="320x240"):
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=5:duration=1",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", path]
    if subprocess.run(cmd, capture_output=True, timeout=120).returncode != 0:
        pytest.skip("could not build the test clip with audio")


def test_mux_publishes_the_audio_into_the_destination(tmp_path, ffmpeg_pair,
                                                     monkeypatch):
    """Mux real con audio y publicación sin cruzar de unidad."""
    import json

    import src.processing.pipeline as pipeline

    ffmpeg, ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    _make_clip_with_audio(ffmpeg, src)
    salida = tmp_path / "upscaled.mp4"
    encode = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-an", "-c:v", "copy", str(salida)],
        capture_output=True, timeout=120,
    )
    assert encode.returncode == 0, encode.stderr

    monkeypatch.setattr(pipeline, "os", _WindowsReplaceShim(os))
    assert pipeline._mux_audio(
        ffmpeg, src, str(salida), audio_codec="copy",
        settings={"audio": "copy", "format": "mp4"},
    ) is True

    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_type", "-of", "json", str(salida)],
        capture_output=True, text=True, timeout=60,
    )
    assert json.loads(probe.stdout)["streams"], "el mux dejó el vídeo sin audio"
    assert not list(tmp_path.glob(".draggy-render-*"))


def test_parallel_audio_stages_next_to_the_render(tmp_path, monkeypatch):
    """El audio paralelo también se escribe junto al render, no en %TEMP%."""
    import tempfile

    import src.processing.pipeline as pipeline

    monkeypatch.setattr(pipeline.subprocess, "Popen", _FakeFFmpeg)
    monkeypatch.setattr(pipeline, "_get_duration_seconds", lambda *a, **k: 10.0)
    carpeta = tmp_path / "render"
    carpeta.mkdir()
    settings = {"audio": "aac", "audio_boost": False, "format": "mp4",
                "audio_stream_index": 0}

    junto = pipeline._encode_audio_parallel("ffmpeg", "entrada.mkv", settings,
                                            staging_dir=str(carpeta))
    suelto = pipeline._encode_audio_parallel("ffmpeg", "entrada.mkv", settings)
    try:
        assert junto and os.path.dirname(junto) == str(carpeta)
        assert suelto and os.path.dirname(os.path.abspath(suelto)) ==             os.path.abspath(tempfile.gettempdir())
    finally:
        for path in (junto, suelto):
            if path and os.path.exists(path):
                os.remove(path)


def test_cleanup_stale_staging_scans_the_render_folders(tmp_path, monkeypatch):
    """Los restos junto al render se limpian (antes solo se miraba %TEMP%)."""
    import time as _time

    import src.processing.pipeline as pipeline

    salida = tmp_path / "renders"
    salida.mkdir()
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()

    def _viejo(path):
        path.write_bytes(b"x")
        os.utime(path, (_time.time() - 7200,) * 2)
        return path

    restos = [
        _viejo(salida / ".draggy-render-abc.mp4"),
        _viejo(salida / ".drag-audio-abc.m4a"),
        _viejo(temp_dir / ".drag-audio-viejo.m4a"),  # sesión anterior
    ]
    reciente = salida / ".draggy-render-nuevo.mp4"
    reciente.write_bytes(b"x")
    ajena = _viejo(tmp_path / ".draggy-render-ajena.mp4")

    monkeypatch.setattr(pipeline.tempfile, "gettempdir", lambda: str(temp_dir))
    assert pipeline.cleanup_stale_staging(directories=[str(salida)]) == 3
    assert not any(path.exists() for path in restos)
    assert reciente.exists(), "un render en curso no se toca"
    assert ajena.exists(), "no se barren carpetas que nadie pidió"


def test_gpu_post_detail_reaches_only_the_last_engine(monkeypatch):
    """El ajuste de detalle del usuario llega al ÚLTIMO paso de la cadena.

    Bug real de este bucle: el nombre ``detail`` se reutilizaba para el
    informe del motor, así que a partir del segundo paso el ajuste pedido se
    sustituía por lo que hubiera reportado el paso anterior (o por ``None``) y
    la cadena entera ignoraba la preferencia en silencio.
    """
    import src.ai.gpu_post as gpu_post
    import src.processing.pipeline as pipeline

    monkeypatch.setattr(gpu_post, "gpu_post_enabled", lambda: True)
    monkeypatch.setattr(gpu_post, "get_detail_preference", lambda: 0.8)

    class _Engine:
        def __init__(self, engine_id, reported=None):
            self.engine_id = engine_id
            self._reported = reported
            self.scales = []

        def configure_gpu_post(self, width=None, height=None, resize_fn=None,
                               input_size=None, in_scale=None):
            self.scales.append(in_scale)
            result = {"active": True, "reason": "ok"}
            if self._reported is not None:
                result["detail"] = self._reported
            return result

    primero = _Engine("paso-1", reported=1.0)
    ultimo = _Engine("paso-2")
    pipeline._configure_gpu_post_for_chain(
        [], [primero, ultimo], None, 1920, 1440, 768, 576, None,
    )

    assert primero.scales == [1.0], "un paso intermedio no se reduce"
    assert ultimo.scales == [0.8], "la preferencia no llegó al último paso"


# ------------------------- bloqueo de una cadena que no puede con el vídeo
class _FixedSizeUpscaler:
    """Motor que solo acepta frames de un tamaño concreto (modelo real: 64x64).

    Es el caso medido con el modelo de entrada fija del store: en el render
    anterior se auto-desactivaba en el frame 0 y el vídeo salía sin upscale.
    """

    engine_id = "fixed-64"
    _mode = "upscale"

    def get_scale_factor(self):
        return 2

    def process_frame(self, frame):
        h, w = frame.shape[0], frame.shape[1]
        if (h, w) != (64, 64):
            raise RuntimeError(
                f"Got invalid dimensions for input: index: 2 Got: {h} Expected: 64"
            )
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def test_chain_contract_strict_flags_an_unusable_step():
    """Modo estricto: el paso que no puede con el tamaño real bloquea.

    Sin strict (comportamiento de siempre) el paso se salta y el contrato
    pasa, que es justo lo que dejaba correr un render sin upscale.
    """
    from src.processing.pipeline import (
        AIChainUnusable, validate_ai_chain_contract,
    )

    engine = _FixedSizeUpscaler()
    assert validate_ai_chain_contract([engine], 576, 768) is True
    with pytest.raises(AIChainUnusable) as err:
        validate_ai_chain_contract([engine], 576, 768, strict=True)
    message = str(err.value)
    assert "fixed-64" in message and "768x576" in message, message
    assert "Expected: 64" in message, "el motivo del motor tiene que viajar"


def test_chain_contract_strict_ignores_a_step_already_disabled():
    """Un motor ya desactivado para el render no bloquea nada."""
    from src.processing.pipeline import validate_ai_chain_contract

    engine = _FixedSizeUpscaler()
    engine._failed = True
    assert validate_ai_chain_contract([engine], 576, 768, strict=True) is True


def test_ai_render_blocks_before_decoding_an_impossible_step(tmp_path,
                                                             ffmpeg_pair):
    """El render se corta antes de decodificar y no publica nada."""
    from src.processing.pipeline import AIChainUnusable, compress_video_with_ai

    ffmpeg, _ffprobe = ffmpeg_pair
    src = str(tmp_path / "in.mp4")
    _make_clip(ffmpeg, src, size="320x240")
    dst = str(tmp_path / "out.mp4")
    estados = []

    with pytest.raises(AIChainUnusable) as err:
        compress_video_with_ai(
            ffmpeg, src, dst,
            {"codec": "libx264", "resolution": "original", "bitrate": 1000,
             "audio": "noaudio", "format": "mp4", "preset": "veryfast",
             "pixel_format": "yuv420p"},
            inference_engines=[_FixedSizeUpscaler()],
            status_callback=estados.append,
        )

    assert "fixed-64" in str(err.value)
    assert not os.path.exists(dst), "no debe quedar salida publicada"
    assert any("AI chain blocked" in str(s) for s in estados), estados
    # Y no puede quedar NADA a medias: el bloqueo va antes de arrancar el
    # audio en paralelo y el encoder, así que no hay temporales ni procesos.
    restos = [p for p in os.listdir(str(tmp_path))
              if p.startswith((".draggy-render-", ".drag-audio-", ".mux-"))]
    assert restos == [], restos


def test_prune_render_logs_keeps_the_newest_and_touches_nothing_else(tmp_path,
                                                                   monkeypatch):
    """Los logs de render se podan por antigüedad y sin tocar el resto.

    La carpeta crecía sin límite (3109 ficheros / 21 MB medidos): uno por
    render y uno por cada test que lanza uno.
    """
    import src.processing.pipeline as pipeline

    logs = tmp_path / "logs"
    logs.mkdir()
    nombres = [f"render_2026010{d}_0000{i}.json" for i, d in enumerate(range(5))]
    for nombre in nombres:
        (logs / nombre).write_text("{}")
    (logs / "app.log").write_text("x")
    (logs / "ffmpeg.log").write_text("x")
    (logs / "loudnorm.log").write_text("x")
    (logs / "backup").mkdir()
    (logs / "backup" / "app-2026.log").write_text("x")

    monkeypatch.setattr(pipeline, "get_logs_dir", lambda: str(logs))

    assert pipeline.prune_render_logs(keep=2) == 3
    assert sorted(p.name for p in logs.glob("render_*.json")) == nombres[-2:]
    # Ni los otros logs ni la carpeta de respaldos se tocan.
    assert (logs / "app.log").exists()
    assert (logs / "ffmpeg.log").exists()
    assert (logs / "loudnorm.log").exists()
    assert (logs / "backup" / "app-2026.log").exists()

    # Poda con la carpeta ausente o sin nada que borrar: nunca lanza.
    monkeypatch.setattr(pipeline, "get_logs_dir", lambda: str(tmp_path / "nope"))
    assert pipeline.prune_render_logs() == 0
    assert pipeline.prune_render_logs(keep="basura") == 0
    assert pipeline.prune_render_logs(keep=99) == 0


def test_saving_a_render_log_prunes_the_old_ones(tmp_path):
    """Escribir el log de un render deja la carpeta acotada: los más nuevos.

    La política vive en el único sitio que escribe logs de render, así que la
    carpeta no puede crecer sin límite ni depende del arranque de la GUI.
    """
    import src.processing.pipeline as pipeline

    logs = tmp_path / "logs"
    logs.mkdir()
    # La poda actua sobre la carpeta del propio log, asi que basta escribir
    # uno mas que el limite para que el mas viejo desaparezca.
    for i in range(pipeline.RENDER_LOGS_KEEP + 1):
        pipeline._save_render_log(
            str(logs / f"render_20260101_{i:06d}.json"), {"i": i},
        )
    restantes = sorted(logs.glob("render_*.json"))
    assert len(restantes) == pipeline.RENDER_LOGS_KEEP
    nombres = [p.name for p in restantes]
    assert nombres[0] == "render_20260101_000001.json", nombres[0]
    assert nombres[-1] == f"render_20260101_{pipeline.RENDER_LOGS_KEEP:06d}.json"
