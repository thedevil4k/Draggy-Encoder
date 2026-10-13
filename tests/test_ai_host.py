"""Host de IA: el proceso que carga torch en lugar de la GUI.

Estos tests son la red que protege el objetivo: **la GUI nunca carga
torch/onnxruntime en su propio proceso** (importarlos ahí tumbaba el
instalado con 0xc0000005 en python313.dll). Cubren las cuatro cosas que
pueden romperse por separado:

* el protocolo (framing + blobs de ida y vuelta),
* la fidelidad de las ops (misma escala que la implementación de siempre),
* la fidelidad del motor (frames bit a bit iguales que in-process),
* la degradación (sin host la UI sigue viva, usando la pista del nombre).

Los dos últimos comprueban lo que de verdad importa y son los que costarían
dinero si se rompen: un frame que sale distinto o una app que muere.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from src.ai import host_client, host_protocol
from src.ai.host_client import HostError, HostUnavailable, RuntimeHost

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modelos reales del store del checkout (mismo criterio que el resto de la
# suite: si no están, el test se salta en vez de inventarse uno).
_ANIME_ONNX = os.path.join(
    "bin", "models", "Upscale", "realesr-animevideov3", "realesr-animevideov3.onnx"
)
_X4_PTH = os.path.join(
    "bin", "models", "Upscale", "realesr-animevideov3", "RealESRGAN_x4plus.pth"
)

_skip_without_onnx = pytest.mark.skipif(
    not os.path.isfile(_ANIME_ONNX), reason="el store del checkout no tiene el modelo"
)
_skip_without_pth = pytest.mark.skipif(
    not os.path.isfile(_X4_PTH), reason="el store del checkout no tiene el peso .pth"
)


# ---------------------------------------------------------------------------
# Protocolo
# ---------------------------------------------------------------------------

def test_protocol_round_trip_with_blobs():
    """Framing y blobs: lo que entra crudo sale idéntico."""
    import io

    buf = io.BytesIO()
    frame = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    meta, blob = host_protocol.encode_array(frame)
    host_protocol.write_message(
        buf, {"id": 7, "job": "engine.process", "data": {"handle": "e"}}, [blob]
    )
    buf.seek(0)

    header, blobs = host_protocol.read_message(buf)

    assert header["job"] == "engine.process"
    assert header["id"] == 7
    assert header["blob_lengths"] == [frame.nbytes]
    assert np.array_equal(host_protocol.decode_array(meta, blobs[0]), frame)


def test_protocol_rejects_huge_header():
    """Una cabecera absurda no debe reservar memoria a lo loco."""
    import io
    import struct

    buf = io.BytesIO(struct.pack(">I", host_protocol.MAX_HEADER_BYTES + 1))
    with pytest.raises(ValueError):
        host_protocol.read_message(buf)


# ---------------------------------------------------------------------------
# Host vivo
# ---------------------------------------------------------------------------

def test_ping_describes_the_host_and_shares_the_store():
    from src.runtimes import manager as rm

    host = host_client.get_host()
    info = host.ping()

    assert info["protocol"] == host_protocol.PROTOCOL_VERSION
    # El host resuelve el MISMO store que la app: si no, instalaría/leería
    # runtimes en otro sitio y no vería lo instalado.
    assert os.path.normcase(info["store"]) == os.path.normcase(
        rm.get_runtime_modules_dir()
    )
    assert int(info["pid"]) > 0
    assert host.is_running()


def test_mock_engine_moves_frames_unchanged():
    """Ida y vuelta de frames reales (blobs) por el protocolo."""
    host = host_client.get_host()
    host.ping()
    handle = host.engine_create("mock", "")

    assert host.engine_load(handle) is True
    frame = np.random.default_rng(1).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    out = host.engine_process(handle, [frame])

    assert len(out) == 1
    assert out[0].shape == frame.shape
    assert out[0].dtype == frame.dtype
    assert np.array_equal(out[0], frame)

    info = host.engine_info(handle)
    assert info["engine_id"] == "mock"
    host.engine_unload(handle)


def test_host_reports_job_errors_and_stays_alive():
    """Un job que falla devuelve error, no mata el host ni cuelga al cliente."""
    host = host_client.get_host()
    host.ping()
    pid = host.ping_info["pid"]

    with pytest.raises(HostError) as rejected:
        host.call_op("borra_todo")
    assert "op no permitida" in str(rejected.value)

    with pytest.raises(HostError):
        host.engine_create("backend-que-no-existe", "nope.pth")

    assert host.is_running()
    assert host.ping()["pid"] == pid


# ---------------------------------------------------------------------------
# Routing de la fábrica de motores
# ---------------------------------------------------------------------------

@_skip_without_onnx
def test_factory_routes_the_engine_to_the_host_by_default():
    """La inferencia corre en el host: el motor local NO se construye aquí."""
    import src.ai.inference as inf
    from src.ai.remote_engine import RemoteInferenceEngine

    engine = inf.create_inference_engine("onnxruntime", _ANIME_ONNX)

    assert isinstance(engine, RemoteInferenceEngine)
    assert engine.is_loaded is True
    assert engine.get_scale_factor() == 4
    assert engine.get_cumulative_timing()["frames"] == 0
    engine.unload()


@_skip_without_onnx
def test_factory_local_mode_when_the_host_is_disabled(monkeypatch):
    """``DRAGGY_AI_HOST=off`` (soporte/tests) usa el motor local de siempre."""
    import src.ai.inference as inf

    monkeypatch.setenv("DRAGGY_AI_HOST", "off")
    engine = inf.create_inference_engine("onnxruntime", _ANIME_ONNX)

    assert type(engine).__name__ == "ONNXRUNTIMEInferenceEngine"
    engine.unload()


def test_factory_never_loads_torch_in_a_bundle_without_host(monkeypatch, tmp_path):
    """Empaquetado y sin host: error claro, NUNCA degradar a cargar los
    runtimes nativos dentro del EXE de la GUI (el 0xc0000005)."""
    import src.ai.inference as inf

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(host_client, "host_source_root", lambda: None)

    with pytest.raises(RuntimeError) as err:
        inf.create_inference_engine("upscale-realesr-animevideov3", str(tmp_path / "m.pth"))

    assert "host de IA no está disponible" in str(err.value)
    assert "Unknown inference engine" not in str(err.value)


def test_host_disabled_by_env(monkeypatch):
    """``DRAGGY_AI_HOST=off`` es el escape hatch: ni disponible ni arrancable."""
    monkeypatch.setenv("DRAGGY_AI_HOST", "off")

    assert host_client.host_available() is False
    with pytest.raises(HostUnavailable):
        RuntimeHost().start()


# ---------------------------------------------------------------------------
# Fidelidad de las ops y del motor
# ---------------------------------------------------------------------------

@_skip_without_pth
def test_model_scale_op_matches_the_in_process_implementation():
    """La escala que mide el host es EXACTAMENTE la de siempre (torch.load)."""
    from src.ai.architectures import get_model_file_scale

    host = host_client.get_host()
    got = host.call_op("model_scale", path=_X4_PTH)

    assert int(get_model_file_scale(_X4_PTH)) == 4
    assert got == {"scale": 4, "proven": True, "hint": 4}


@_skip_without_onnx
def test_engine_round_trip_is_bit_exact_against_in_process():
    """El frame que devuelve el host es idéntico al del motor in-process."""
    from src.ai.inference import ONNXRUNTIMEInferenceEngine

    host = host_client.get_host()
    host.ping()
    frame = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)

    handle = host.engine_create("onnxruntime", _ANIME_ONNX)
    assert host.engine_load(handle) is True
    out = host.engine_process(handle, [frame])[0]

    reference = ONNXRUNTIMEInferenceEngine(_ANIME_ONNX, "onnxruntime")
    assert reference.load() is True
    ref_out = reference.process_frame(frame)

    assert out.shape == ref_out.shape
    assert np.array_equal(out, ref_out), "el host cambió los píxeles"

    info = host.engine_info(handle)
    assert info["scale_factor"] == reference.get_scale_factor()
    # La telemetría del render se sigue llenando desde el host (P0).
    assert info["timing"]["frames"] == 1
    host.engine_unload(handle)


# ---------------------------------------------------------------------------
# La GUI, sin torch en su proceso (el objetivo del host)
# ---------------------------------------------------------------------------

_GUI_PROBE = r"""
import json, sys

sys.path.insert(0, sys.argv[1])
result = {"scale": None, "proven": None, "torch": [], "error": None}
try:
    from src.ui.workflow_widget import _detect_file_scale

    scale, proven = _detect_file_scale(sys.argv[2])
    result["scale"], result["proven"] = int(scale), bool(proven)
except BaseException as exc:  # noqa: BLE001 - se reporta tal cual
    result["error"] = f"{type(exc).__name__}: {exc}"
result["torch"] = sorted(
    key for key in sys.modules
    if key == "torch" or key.startswith("torch.")
    or key == "onnxruntime" or key.startswith("onnxruntime.")
)
print(json.dumps(result))
"""


def _run_gui_probe(model_path: str, env_overrides: dict) -> dict:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("DRAGGY_AI_HOST", None)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _GUI_PROBE, REPO_ROOT, model_path],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"la sonda falló:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@_skip_without_pth
def test_gui_measures_a_real_pth_without_loading_torch():
    """End-to-end: la UI mide un .pth real y torch NO entra en su proceso."""
    result = _run_gui_probe(_X4_PTH, {})

    assert result["error"] is None, result["error"]
    assert (result["scale"], result["proven"]) == (4, True)
    assert result["torch"] == [], (
        "medir un modelo cargó torch en el proceso de la GUI: "
        f"{result['torch']}"
    )


@_skip_without_pth
def test_gui_degrades_to_the_name_hint_without_host():
    """Sin host (desactivado) la UI sigue viva: pista del nombre, sin probar."""
    result = _run_gui_probe(_X4_PTH, {"DRAGGY_AI_HOST": "off"})

    assert result["error"] is None, result["error"]
    assert (result["scale"], result["proven"]) == (4, False)
    assert result["torch"] == []


# ---------------------------------------------------------------------------
# Bundle: el host encuentra su copia de fuentes
# ---------------------------------------------------------------------------

def test_host_source_root_finds_the_bundled_copy(tmp_path, monkeypatch):
    """El spec copia ``src/`` a ``ai_src/src``: el host debe encontrarla ahí.

    Sin esa copia el Python privado del instalador no tiene el código de los
    motores (vive dentro del EXE) y el host no puede arrancar en la app
    instalada, que es justo donde importa.
    """
    bundled = tmp_path / "ai_src"
    (bundled / "src" / "ai").mkdir(parents=True)
    (bundled / "src" / "ai" / "host.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert host_client.host_source_root() == str(bundled)


def test_host_source_root_is_none_without_the_copy(tmp_path, monkeypatch):
    """Sin la copia de fuentes el host se declara no disponible (y el selftest
    del bundle falla con un mensaje claro en vez de dejar la app sin IA)."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "DraggyEncoder.exe"))

    assert host_client.host_source_root() is None
    assert host_client.host_available() is False


@_skip_without_onnx
def test_engine_errors_from_the_host_are_not_silent():
    """Un modelo que no carga devuelve el motivo del host, no un frame raro."""
    host = host_client.get_host()
    host.ping()

    with pytest.raises(HostError):
        host.engine_create("onnxruntime", str(REPO_ROOT))

    assert host.is_running()


# ---------------------------------------------------------------------------
# Post-proceso en GPU (P1): vive en el host, no en la GUI
# ---------------------------------------------------------------------------

def test_proxy_exposes_the_same_post_process_interface_as_local_engines():
    """Paridad de interfaz: el pipeline trata igual a un motor local y a este.

    Es la red que evita que un método se quede sólo en los motores locales y
    el render remoto acabe con un ``AttributeError`` o, peor, sin P1.
    """
    from src.ai.inference import AIInferenceEngine
    from src.ai.remote_engine import RemoteInferenceEngine

    proxy = RemoteInferenceEngine("mock", "")
    for name in ("configure_gpu_post", "gpu_post_active", "disable_gpu_post",
                 "get_cumulative_timing", "reset_cumulative_timing",
                 "describe_execution", "get_scale_factor", "process_frame",
                 "unload", "load"):
        assert hasattr(proxy, name), f"el proxy remoto no expone {name}()"
        assert callable(getattr(proxy, name)), f"{name} no es invocable"
    # ``gpu_post_active`` es MÉTODO en los motores locales, no propiedad.
    assert callable(AIInferenceEngine.gpu_post_active)
    assert proxy.gpu_post_active() is False
    assert proxy._gpu_post is None
    assert proxy._gpu_post_failure is None


def test_gpu_post_is_activated_inside_the_host():
    """``configure_gpu_post`` del proxy cruza el protocolo y el host manda.

    Con el motor mock (sin CUDA) se comprueba el cableado completo: el host
    construye/decide, el proxy propaga el informe y el dominio del motor
    (``active``) viene del host, no de una suposición de la GUI.
    """
    from src.ai.remote_engine import RemoteInferenceEngine

    host = host_client.get_host()
    host.ping()
    engine = RemoteInferenceEngine("mock", "")
    assert engine.load() is True

    report = engine.configure_gpu_post(960, 720, input_size=(768, 576))

    assert report["active"] is True, report
    assert report["reason"] == "ok"
    assert tuple(report["out_size"]) == (960, 720)
    assert "info" not in report, "la foto del motor no va en el informe del render"
    assert engine.gpu_post_active() is True
    # El host es la fuente de verdad: también lo dice en su propia foto.
    assert host.engine_info(engine._handle)["gpu_post_active"] is True

    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    assert np.array_equal(engine.process_frame(frame), frame)
    engine.unload()


def test_frames_carry_the_state_so_the_hot_path_asks_nothing_extra(monkeypatch):
    """Telemetría y estado de P1 viajan CON el frame (nada de IPC por frame).

    El pipeline lee la telemetría antes y después de cada llamada al motor.
    Si cada lectura fuese una petición al host, un render de 20k frames haría
    40k peticiones que no aportan ni un píxel: aquí se comprueba que no se
    hace ninguna (``engine.info`` revienta si alguien la pide).
    """
    from src.ai.remote_engine import RemoteInferenceEngine

    host = host_client.get_host()
    host.ping()
    engine = RemoteInferenceEngine("mock", "")
    assert engine.load() is True
    before = engine.get_cumulative_timing()
    assert set(before) == {"frames", "pre_ms", "infer_ms", "post_ms"}

    def _no_extra_calls(*_args, **_kwargs):
        raise AssertionError("el camino caliente pidió engine.info al host")

    monkeypatch.setattr(host_client.RuntimeHost, "engine_info", _no_extra_calls)
    engine.process_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    after = engine.get_cumulative_timing()
    assert engine._state, "la respuesta del frame tendría que traer la foto"
    assert isinstance(engine.describe_execution(), dict)
    assert set(after) == set(before)
    monkeypatch.undo()
    engine.unload()


def test_a_gpu_post_failure_reported_by_the_host_reaches_the_pipeline():
    """Si el host abandona P1 a mitad de render, el proxy lo cuenta igual.

    El pipeline lo lee dos veces: para el aviso al usuario y para el JSON del
    render. Un proxy que se quede con ``active=True`` mentiría sobre la
    velocidad y el usuario nunca sabría por qué bajó.
    """
    from src.ai.remote_engine import RemoteInferenceEngine

    engine = RemoteInferenceEngine("mock", "")
    engine._gpu_post = {"active": True, "out_size": (960, 720)}

    engine._adopt_state({"gpu_post_active": False,
                         "gpu_post_failure": "input size changed (640x480 vs 768x576)"})

    assert engine.gpu_post_active() is False
    assert engine._gpu_post is None
    assert "input size changed" in engine._gpu_post_failure


def _cuda_available() -> bool:
    try:
        from src.ai.inference import import_local_runtime

        ort = import_local_runtime("onnxruntime")
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


@_skip_without_onnx
@pytest.mark.skipif(not _cuda_available(), reason="este equipo no tiene CUDA")
def test_gpu_post_really_engages_on_the_host_with_the_installed_model(capsys):
    """Prueba real (RTX): el host activa P1 y la salida ya trae el tamaño final.

    Se salta si el equipo no tiene CUDA o el store no tiene el modelo. Cuando
    corre, comprueba lo que de verdad importa: que la marca de activo la da el
    host tras su autocomprobación y que el frame llega al pipeline YA al
    tamaño del encoder (sin el resize de CPU de después, que medido eran 40 ms
    por frame en 3072x2304 -> 960x720).
    """
    from src import preferences
    from src.ai.remote_engine import RemoteInferenceEngine

    previous = preferences.get("gpu_post_process", None)
    preferences.set("gpu_post_process", True)
    try:
        engine = RemoteInferenceEngine("onnxruntime", _ANIME_ONNX)
        try:
            assert engine.load() is True
            report = engine.configure_gpu_post(960, 720, input_size=(768, 576))
            if report.get("reason") == "no-cuda":
                pytest.skip("la sesión ONNX de este equipo acabó en CPU")
            assert report["active"] is True, report
            assert engine.gpu_post_active() is True
            assert report.get("precision") in ("fp16", "fp32", "model"), report

            frame = np.random.default_rng(3).integers(
                0, 256, (576, 768, 3), dtype=np.uint8)
            out = engine.process_frame(frame)
            assert out.shape == (720, 960, 3), (
                f"P1 activo pero el frame llegó en {out.shape[1]}x{out.shape[0]}"
            )
            assert engine.gpu_post_active() is True
        finally:
            engine.unload()
    finally:
        preferences.set("gpu_post_process", previous)
