"""End-to-end regression suite for the Draggy-Encoder AI/runtime fixes.

Covers the four guarantees requested:

1. AI Engines install detection honest about disk state + KIND_MODEL rows
   kept out of the "Runtimes & Binaries" tab.
2. Veraz runtime detection: version guard on the local store, no false
   "INSTALLED" from empty namespace packages, separated torch / onnx
   installers, and pip failures that surface the real pip output.
3. UI responsiveness: the DependenciesDialog lays out at small/large sizes
   and switches tabs without crashing.
4. GPU-accelerated rescale + compress: a dummy clip is downscaled to a
   target height and encoded with h264_nv2 (skipped if no NVENC).

Run with:  python -m pytest tests/test_draggy.py -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_engines_state(tmp_path, monkeypatch):
    """Point engines.json AND the models dir at temp space; start clean."""
    import src.ai.engines_state as es
    import src.ai.registry as reg

    monkeypatch.setattr(es, "get_session_dir", lambda: str(tmp_path))
    # Make engine folders live under tmp so tests never touch bin/models.
    monkeypatch.setattr(reg, "get_models_dir", lambda: str(tmp_path / "models"))
    es.reset_all()
    yield tmp_path
    es.reset_all()


@pytest.fixture(scope="session")
def capture_screenshots_module():
    """Carga packaging/linux/capture_screenshots.py como módulo por su ruta.

    Cargar por ruta con importlib evita la colisión con el paquete
    ``packaging`` de site-packages: ``import packaging.linux...`` podría
    resolver al paquete instalado (paquete regular gana a namespace) y
    fallar con ModuleNotFoundError. El módulo expone ``__file__`` (la ruta
    real del script) y ``main()`` para las pruebas en proceso.

    Es ``session``-scoped y se registra en ``sys.modules``: la carga (que
    importa PySide6, ~3-5s) ocurre UNA vez por suite, no una por test.
    """
    import importlib.util

    name = "capture_screenshots_under_test"
    if name in sys.modules:
        return sys.modules[name]

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(
        repo_root, "packaging", "linux", "capture_screenshots.py"
    )
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _current_cp_tag():
    return "cp%d%d-cp%d%d-win_amd64" % (
        sys.version_info.major, sys.version_info.minor,
        sys.version_info.major, sys.version_info.minor,
    )


def _foreign_cp_tag():
    """Stable incompatible tag independent of the Python running pytest."""
    return "cp999-cp999-win_amd64"


# ==--------------------------------------------------------------------==
# 1. AI Engines install detection
# ==--------------------------------------------------------------------==

def test_mark_installed_persists_and_get_state_reads_it(isolated_engines_state):
    from src.ai.engines_state import mark_installed, get_state
    from src.ai.registry import get_engine_dir

    eid = "upscale-realesrgan-x4plus"
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    open(os.path.join(target, "README.md"), "w").close()  # sentinel
    open(os.path.join(target, "RealESRGAN_x4plus.pth"), "w").close()  # pesos

    ok = mark_installed(eid, install_path=target)
    assert ok is True

    st = get_state(eid)
    assert st["installed"] is True
    assert st["install_path"] == target
    assert st["last_error"] is None


def test_get_state_clears_phantom_installed_when_dir_gone(isolated_engines_state):
    """The honest-on-disk sensor must clear stale 'installed' flags."""
    import src.ai.engines_state as es
    from src.ai.engines_state import mark_installed, get_state
    from src.ai.registry import get_engine_dir

    eid = "upscale-realesrgan-x4plus"
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    open(os.path.join(target, "README.md"), "w").close()
    open(os.path.join(target, "RealESRGAN_x4plus.pth"), "w").close()
    mark_installed(eid, install_path=target)
    assert get_state(eid)["installed"] is True

    # User deletes the model folder by hand -> phantom INSTALLED bug.
    shutil.rmtree(target)
    st = get_state(eid)
    assert st["installed"] is False
    assert st["enabled"] is False
    # The breadcrumb is persisted to disk.
    raw = es._load().get("engines", {}).get(eid, {})
    assert "missing" in (raw.get("last_error") or "").lower()


def test_get_state_invalidates_code_only_install(isolated_engines_state):
    """A git clone of a code-only repo (the old OmniSR failure mode: source
    files, no weights) creates the folder but must NOT count as installed."""
    import src.ai.engines_state as es
    from src.ai.engines_state import mark_installed, get_state
    from src.ai.registry import get_engine_dir

    eid = "upscale-omnisr-x4"
    target = get_engine_dir(eid)
    os.makedirs(os.path.join(target, "components"), exist_ok=True)
    for f in ("README.md", "train.py", "test.py", "components/OmniSR.py"):
        open(os.path.join(target, f), "w").close()
    mark_installed(eid, install_path=target)

    st = get_state(eid)
    assert st["installed"] is False
    assert st["enabled"] is False
    raw = es._load().get("engines", {}).get(eid, {})
    assert "no weight" in (raw.get("last_error") or "").lower()


def test_get_state_keeps_installed_when_weights_in_subdir(isolated_engines_state):
    """HF repos may store weights in subfolders; the weight sensor must walk."""
    from src.ai.engines_state import mark_installed, get_state
    from src.ai.registry import get_engine_dir

    eid = "upscale-realesrgan-x4plus"
    target = get_engine_dir(eid)
    os.makedirs(os.path.join(target, "weights"), exist_ok=True)
    open(os.path.join(target, "weights", "model.safetensors"), "w").close()
    mark_installed(eid, install_path=target)

    assert get_state(eid)["installed"] is True


def test_engines_state_load_cache_stays_fresh_and_returns_copies(isolated_engines_state):
    """The mtime-keyed load cache must serve fresh data after external
    edits to engines.json and never hand out a mutable shared object."""
    import json
    import src.ai.engines_state as es

    p = es._path()
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "engines": {"x": {"installed": False}}}, f)

    assert es._load()["engines"]["x"]["installed"] is False

    # Mutating a returned payload must not poison the cache.
    d = es._load()
    d["engines"]["mutated"] = {}
    assert "mutated" not in es._load()["engines"]

    # External edit (new mtime) -> cache invalidates.
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "engines": {"x": {"installed": True}}}, f)
    os.utime(p, (os.path.getmtime(p) + 5,) * 2)
    assert es._load()["engines"]["x"]["installed"] is True


def test_archive_extraction_rejects_path_traversal(tmp_path):
    """Downloaded archives must not write outside their extraction root."""
    from src.ffmpeg.downloader import _safe_extract_zip

    archive_path = tmp_path / "malicious.zip"
    outside = tmp_path / "escaped.txt"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", "owned")
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive_path, "r") as archive:
        with pytest.raises(RuntimeError, match="Unsafe archive member"):
            _safe_extract_zip(archive, str(extract_dir))
    assert not outside.exists()


def test_benchmark_cache_invalidates_when_model_changes(isolated_engines_state, tmp_path):
    """A cached timing is not reused after its weight file changes."""
    import src.ai.engines_state as es

    model = tmp_path / "model.pth"
    model.write_bytes(b"first")
    stat = model.stat()
    entry = {
        "ms_per_frame": 3.0,
        "model_fingerprint": {
            "path": os.path.abspath(model),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
    }
    assert es.set_backend_benchmark("upscale-realesrgan-x4plus", "torch", entry)
    assert es.get_backend_benchmarks("upscale-realesrgan-x4plus")["torch"]

    model.write_bytes(b"changed weights")
    os.utime(model, (model.stat().st_atime, model.stat().st_mtime + 2))
    assert es.get_backend_benchmarks("upscale-realesrgan-x4plus")["torch"] is None


def test_libplacebo_probe_cached_by_binary_mtime(tmp_path, monkeypatch):
    """has_libplacebo_support must probe ffmpeg once per (path, mtime):
    unchanged binaries never re-spawn the subprocess, a freshly
    downloaded FFmpeg full (new mtime) is re-probed."""
    import src.ffmpeg.manager as man

    fake_bin = tmp_path / "ffmpeg.exe"
    fake_bin.write_bytes(b"x")
    calls = []

    class _Result:
        stdout = "ffmpeg version 7.1 ... \n ...libplacebo...\n"
        stderr = ""
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(man.subprocess, "run", fake_run)

    assert man.has_libplacebo_support(str(fake_bin)) is True
    assert man.has_libplacebo_support(str(fake_bin)) is True
    assert len(calls) == 1

    os.utime(fake_bin, (os.path.getmtime(fake_bin) + 5,) * 2)
    assert man.has_libplacebo_support(str(fake_bin)) is True
    assert len(calls) == 2


def test_legacy_fsr_engines_are_not_catalogued_or_dispatchable():
    from src.ai.inference import create_inference_engine
    from src.ai.registry import CATALOG_BY_ID

    for eid in (
        "upscale-fsr-2x",
        "upscale-fsr-3x",
        "upscale-fsr-4x",
        "upscale-fsr-2x-v2",
        "upscale-fsr-3x-v2",
        "upscale-fsr-4x-v2",
    ):
        assert eid not in CATALOG_BY_ID
        with pytest.raises(RuntimeError):
            create_inference_engine(eid, "")


def test_fsr31_mode_routing_dispatches_correctly(monkeypatch):
    """create_inference_engine routes FSR 3.1 engine ids to the right mode."""
    import src.ai.inference as inf

    captured = {}

    class _FakeFSR31:
        def __init__(self, model_path, engine_id, mode="both"):
            self.model_path = model_path
            self.engine_id = engine_id
            self._mode = mode
            self._loaded = True
            captured["engine_id"] = engine_id
            captured["mode"] = mode

        def load(self):
            return True

    monkeypatch.setattr(inf, "FSR31Engine", _FakeFSR31)

    for eid, expected_mode in [
        ("upscale-fsr31-2x", "upscale"),
        ("upscale-fsr31-2x-interp", "interpolate"),
    ]:
        captured.clear()
        eng = inf.create_inference_engine(eid, "")
        assert isinstance(eng, _FakeFSR31)
        assert captured["mode"] == expected_mode


def test_fsr31_catalog_split_by_category():
    """FSR 3.1 upscale variants live in Upscale, interpolation variants in
    Interpolate, and no '-both' entries exist (both = activate each node)."""
    from src.ai.registry import CATALOG_BY_ID

    assert CATALOG_BY_ID["upscale-fsr31-2x"]["category"] == "Upscale"
    assert CATALOG_BY_ID["upscale-fsr31-3x"]["category"] == "Upscale"
    assert CATALOG_BY_ID["upscale-fsr31-4x"]["category"] == "Upscale"

    interp_ids = [eid for eid in CATALOG_BY_ID if eid.startswith("upscale-fsr31") and eid.endswith("-interp")]
    assert len(interp_ids) == 3
    for eid in interp_ids:
        assert CATALOG_BY_ID[eid]["category"] == "Interpolate", eid
        assert CATALOG_BY_ID[eid]["mode"] == "interpolate", eid

    both_ids = [eid for eid in CATALOG_BY_ID if eid.startswith("upscale-fsr31") and eid.endswith("-both")]
    assert both_ids == []


def test_resolve_fsr3_engine_modes():
    """Workflow Upscale/Interp node combination drives the FSR 3.1 mode."""
    from src.ui.main_window import resolve_fsr3_engine

    up_only = {"engine_id": "upscale-fsr31-2x", "model_path": None, "scale": 2, "mode": "upscale"}
    up_interp = {"engine_id": "upscale-fsr31-2x-interp", "model_path": None, "scale": 2, "mode": "interpolate"}

    # Upscale node alone -> upscale mode, engine from the Upscale node.
    eid, path, mode = resolve_fsr3_engine(up_only, None)
    assert (eid, mode) == ("upscale-fsr31-2x", "upscale")
    assert path is None

    # Interp node alone -> interpolate mode, engine from the Interp node.
    eid, path, mode = resolve_fsr3_engine(None, up_interp)
    assert (eid, mode) == ("upscale-fsr31-2x-interp", "interpolate")

    # Both nodes active with FSR 3.1 -> "both" (one engine, both functions).
    eid, path, mode = resolve_fsr3_engine(up_only, up_interp)
    assert (eid, mode) == ("upscale-fsr31-2x", "both")

    # Legacy string-only values keep working.
    eid, _path, mode = resolve_fsr3_engine("upscale-fsr31-2x", "upscale-fsr31-2x-interp")
    assert (eid, mode) == ("upscale-fsr31-2x", "both")

    # Non-FSR upscaler + no interp node -> untouched engine, upscale mode.
    eid, _path, mode = resolve_fsr3_engine({"engine_id": "upscale-realesrgan-x4plus", "model_path": "/x.pth"}, None)
    assert (eid, mode) == ("upscale-realesrgan-x4plus", "upscale")


def test_get_model_path_torch_picks_engine_matching_file(isolated_engines_state):
    """With several .pth files in the repo dir, the file matching the engine
    id wins. The old 'first in listing order' picked AI-Forever_x2plus.pth
    (a 12-channel RRDBNet) and crashed the render with 'Expected: 12'."""
    from src.ai.inference import get_model_path
    from src.ai.engines_state import mark_installed
    from src.ai.registry import get_engine_dir

    eid = "upscale-realesr-animevideov3"
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    for f in ("AI-Forever_x2plus.pth", "RealESRGAN_x4plus.pth", "realesr-animevideov3.pth"):
        open(os.path.join(target, f), "w").close()
    mark_installed(eid, install_path=target)

    got = get_model_path(eid, "torch")
    assert got == os.path.join(target, "realesr-animevideov3.pth")


def test_get_model_path_torch_tie_breaks_to_exact_match(isolated_engines_state):
    """realesr-general-x4v3 vs realesr-general-wdn-x4v3: the exact engine
    match must win over the alphabetically-first lookalike."""
    from src.ai.inference import get_model_path
    from src.ai.engines_state import mark_installed
    from src.ai.registry import get_engine_dir

    eid = "upscale-realesr-general-x4v3"
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    for f in ("realesr-general-wdn-x4v3.pth", "realesr-general-x4v3.pth"):
        open(os.path.join(target, f), "w").close()
    mark_installed(eid, install_path=target)

    got = get_model_path(eid, "torch")
    assert got == os.path.join(target, "realesr-general-x4v3.pth")


def test_get_model_path_torch_none_when_no_weights(isolated_engines_state):
    from src.ai.inference import get_model_path
    from src.ai.engines_state import mark_installed
    from src.ai.registry import get_engine_dir

    eid = "upscale-omnisr-x4"
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    mark_installed(eid, install_path=target)

    assert get_model_path(eid, "torch") is None


def test_get_state_heals_legacy_lowercase_install_path(isolated_engines_state):
    """install_path stored with wrong casing is rewritten to the canonical
    registry path. Windows tolerates it; Linux (case-sensitive) would
    report 'Model folder missing on disk' forever."""
    import src.ai.engines_state as es
    from src.ai.engines_state import get_state
    from src.ai.registry import get_engine_dir

    eid = "upscale-realcugan-pro"
    canonical = get_engine_dir(eid)
    os.makedirs(canonical, exist_ok=True)
    open(os.path.join(canonical, "Real_CUGAN_pro.pth"), "w").close()
    legacy = canonical.replace("Upscale", "upscale")
    es.patch(eid, installed=True, install_path=legacy)

    st = get_state(eid)
    assert st["installed"] is True
    assert st["install_path"] == canonical
    raw = es._load()["engines"][eid]
    assert raw["install_path"] == canonical


def test_create_inference_engine_catalog_id_falls_back_to_pytorch(tmp_path, monkeypatch):
    """A catalog engine id + .pth must never raise 'Unknown inference
    engine': it tries the PyTorch engine as the last resort."""
    import src.ai.inference as inf

    pth = tmp_path / "model.pth"
    pth.write_bytes(b"fake")

    class _FakeTorch:
        def __init__(self, model_path, engine_id):
            self.model_path = model_path
            self.engine_id = engine_id
            self._loaded = True

        def load(self):
            return True

    monkeypatch.setattr(inf, "PyTorchInferenceEngine", _FakeTorch)
    eng = inf.create_inference_engine("upscale-realesr-animevideov3", str(pth))
    assert isinstance(eng, _FakeTorch)


def test_create_inference_engine_catalog_id_without_model_raises_clear_error(monkeypatch):
    """No model file -> a clear RuntimeError, not the misleading
    'Unknown inference engine' message."""
    import src.ai.inference as inf

    with pytest.raises(RuntimeError) as ei:
        inf.create_inference_engine("upscale-realesr-animevideov3", "")
    assert "Could not load inference engine" in str(ei.value)
    assert "Unknown inference engine" not in str(ei.value)


def _inject_local_store():
    """Point sys.path at the app's local runtime store (bin/runtimes/
    ai_modules) so torch/onnxruntime resolve here just like in the app."""
    store = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bin", "runtimes", "ai_modules",
    )
    if os.path.isdir(store) and store not in sys.path:
        sys.path.insert(0, store)
    return store


@pytest.fixture()
def local_store_runtime():
    """Import torch/onnxruntime from the app store for the duration of the
    test. Only ``onnxruntime`` is evicted afterwards (torch is NOT evicted:
    re-importing torch in the same process corrupts its C internals with
    '_has_torch_function already has a docstring'). Later find_spec-based
    tests already guard torch with ``_evicted_module``."""
    store = _inject_local_store()
    yield store
    sys.path[:] = [p for p in sys.path if p != store]
    sys.modules.pop("onnxruntime", None)


def test_convert_pth_to_onnx_skips_multichannel_models(tmp_path, local_store_runtime):
    """convert_pth_to_onnx must refuse models whose input channels != 3
    instead of producing an ONNX that crashes mid-render."""
    torch = pytest.importorskip("torch")
    from src.ai.architectures import SRVGGNetCompact
    from src.ai.inference import convert_pth_to_onnx

    m = SRVGGNetCompact(num_in_ch=12, num_out_ch=12, num_feat=4, num_conv=2, upscale=4)
    pth = tmp_path / "bad12ch.pth"
    torch.save(m.state_dict(), pth)

    assert convert_pth_to_onnx(str(pth)) is None
    assert not (tmp_path / "bad12ch.onnx").exists()


def test_onnx_engine_rejects_12ch_model(tmp_path, local_store_runtime):
    """ONNX models with a fixed input channel != 3 are refused at load(),
    not half-way through a render."""
    torch = pytest.importorskip("torch")
    from src.ai.architectures import SRVGGNetCompact
    from src.ai.inference import ONNXRUNTIMEInferenceEngine

    m = SRVGGNetCompact(num_in_ch=12, num_out_ch=12, num_feat=4, num_conv=2, upscale=4)
    m.eval()
    onnx_path = tmp_path / "bad12ch.onnx"
    torch.onnx.export(
        m, torch.randn(1, 12, 64, 64), str(onnx_path), opset_version=18,
        input_names=["input"], output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,  # exporter legacy (mismo camino que convert_pth_to_onnx_ex)
    )

    eng = ONNXRUNTIMEInferenceEngine(str(onnx_path), "onnxruntime")
    assert eng.load() is False


def test_srvgg_convert_load_process_end_to_end(tmp_path, local_store_runtime):
    """3-channel SRVGG round-trip mirroring the animevideov3 flow:
    convert .pth -> .onnx, load the ONNX engine, process a real frame."""
    torch = pytest.importorskip("torch")
    import numpy as np
    from src.ai.architectures import SRVGGNetCompact
    from src.ai.inference import convert_pth_to_onnx, create_inference_engine

    m = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=4)
    pth = tmp_path / "good3ch.pth"
    torch.save(m.state_dict(), pth)

    onnx_path = convert_pth_to_onnx(str(pth))
    assert onnx_path and os.path.exists(onnx_path)

    eng = create_inference_engine("onnxruntime", onnx_path)
    assert eng.is_loaded
    out = eng.process_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    assert out.shape == (128, 128, 3)


# ==--------------------------------------------------------------------==
# 2. Runtime version guard + honest detection
# ==--------------------------------------------------------------------==

def test_wheel_tag_matches_current_and_rejects_foreign():
    from src.runtimes.manager import _wheel_tag_matches

    cur = "cp%d%d" % (sys.version_info.major, sys.version_info.minor)
    assert _wheel_tag_matches("%s-%s-win_amd64" % (cur, cur)) is True
    # Different minor -> incompatible.
    assert _wheel_tag_matches("cp999-cp999-win_amd64") is False
    # abi3 is forward-compatible.
    assert _wheel_tag_matches("cp39-abi3-win_amd64") is True
    # Pure-python wheels carry no cp tag -> always accepted.
    assert _wheel_tag_matches("py3-none-any") is True


def test_store_incompatible_when_wheel_is_for_other_python(tmp_path, monkeypatch):
    import src.runtimes.manager as rm

    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    di = os.path.join(store, "onnxruntime-1.0.dist-info")
    os.makedirs(di, exist_ok=True)
    with open(os.path.join(di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _foreign_cp_tag())
    # cp314 wheel must NOT be injected on a 3.11 interpreter.
    assert rm.store_is_compatible_with_python() is False


def test_store_compatible_with_matching_wheel(tmp_path, monkeypatch):
    import src.runtimes.manager as rm

    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    di = os.path.join(store, "onnxruntime-1.0.dist-info")
    os.makedirs(di, exist_ok=True)
    with open(os.path.join(di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _current_cp_tag())
    assert rm.store_is_compatible_with_python() is True


def test_inject_runtime_paths_purges_foreign_store_then_injects(tmp_path, monkeypatch):
    """A store built for another Python (cp999) is purged at boot and then
    injected, instead of staying 'incompatible forever'."""
    import src.runtimes.manager as rm

    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    pkg = os.path.join(store, "torch")
    os.makedirs(pkg)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("__version__ = '1.0.0-fake'\n")
    di = os.path.join(store, "torch-1.0.0-fake.dist-info")
    os.makedirs(di)
    with open(os.path.join(di, "WHEEL"), "w") as f:
        f.write("Tag: cp999-cp999-win_amd64\n")
    with open(os.path.join(di, "top_level.txt"), "w") as f:
        f.write("torch\n")

    pre = list(sys.path)
    try:
        rm.inject_runtime_paths()
        # The foreign leftovers must be gone...
        assert not os.path.exists(di)
        assert not os.path.exists(pkg)
        # ...and the store must now be compatible and injected.
        assert rm.store_is_compatible_with_python() is True
        assert store in sys.path
    finally:
        sys.path[:] = pre


def test_purge_removes_only_foreign_version_wheels(tmp_path, monkeypatch):
    """_purge_foreign_version_wheels removes wheels built for another CPython
    but keeps pure-python and current-version wheels."""
    import src.runtimes.manager as rm

    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)

    # 1) Foreign wheel (cp314 on a 3.11 interpreter) -> must be purged.
    numpy_dir = os.path.join(store, "numpy")
    os.makedirs(numpy_dir)
    with open(os.path.join(numpy_dir, "__init__.py"), "w") as f:
        f.write("__version__ = '2.5.1'\n")
    numpy_di = os.path.join(store, "numpy-2.5.1.dist-info")
    os.makedirs(numpy_di)
    with open(os.path.join(numpy_di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _foreign_cp_tag())
    with open(os.path.join(numpy_di, "top_level.txt"), "w") as f:
        f.write("numpy\n")
    # numpy.libs sibling must also go.
    os.makedirs(os.path.join(store, "numpy.libs"))

    # 2) Current-version wheel -> kept.
    cur_di = os.path.join(store, "onnxruntime-1.28.0.dist-info")
    os.makedirs(cur_di)
    with open(os.path.join(cur_di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _current_cp_tag())

    # 3) Pure-python wheel -> kept.
    py_di = os.path.join(store, "flatbuffers-25.12.19.dist-info")
    os.makedirs(py_di)
    with open(os.path.join(py_di, "WHEEL"), "w") as f:
        f.write("Tag: py3-none-any\n")

    purged = rm._purge_foreign_version_wheels()
    assert purged == 1
    assert not os.path.exists(numpy_di)
    assert not os.path.exists(numpy_dir)
    assert not os.path.exists(os.path.join(store, "numpy.libs"))
    assert os.path.isdir(cur_di)
    assert os.path.isdir(py_di)
    assert rm.store_is_compatible_with_python() is True


def test_purge_keeps_shared_package_dir_when_compatible_sibling_exists(
    tmp_path, monkeypatch
):
    """A foreign dist-info (cp314) whose package dir is ALSO used by a
    compatible dist-info (cp311, e.g. numpy) must lose only the dist-info,
    not the shared ``numpy/`` dir."""
    import src.runtimes.manager as rm

    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)

    numpy_dir = os.path.join(store, "numpy")
    os.makedirs(numpy_dir)
    with open(os.path.join(numpy_dir, "__init__.py"), "w") as f:
        f.write("__version__ = '2.4.6'\n")
    # Compatible sibling (what pip just installed).
    ok_di = os.path.join(store, "numpy-2.4.6.dist-info")
    os.makedirs(ok_di)
    with open(os.path.join(ok_di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _current_cp_tag())
    with open(os.path.join(ok_di, "top_level.txt"), "w") as f:
        f.write("numpy\n")
    # Foreign orphan (what pip left behind).
    foreign_di = os.path.join(store, "numpy-2.5.1.dist-info")
    os.makedirs(foreign_di)
    with open(os.path.join(foreign_di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _foreign_cp_tag())

    purged = rm._purge_foreign_version_wheels()
    assert purged == 1
    assert not os.path.exists(foreign_di)
    assert os.path.isdir(ok_di)
    assert os.path.isdir(numpy_dir)
    assert rm.store_is_compatible_with_python() is True


def test_install_cleans_foreign_leftovers_and_detects_runtime(tmp_path, monkeypatch):
    """Regression: pip --target can orphan a dist-info from a previous Python
    (cp314 numpy next to freshly installed cp311 onnxruntime). After
    install_onnxruntime, the orphan must be gone and the runtime detected."""
    import types
    import src.runtimes.manager as rm

    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    os.makedirs(store, exist_ok=True)

    # Orphaned foreign wheel (what pip leaves behind after reinstall).
    numpy_di = os.path.join(store, "numpy-2.5.1.dist-info")
    os.makedirs(numpy_di)
    with open(os.path.join(numpy_di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _foreign_cp_tag())
    with open(os.path.join(numpy_di, "top_level.txt"), "w") as f:
        f.write("numpy\n")

    # Fake pip: writes a real cp311 onnxruntime package + dist-info.
    def _fake_install(cmd, *a, **k):
        pkg = os.path.join(store, "onnxruntime")
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "__init__.py"), "w") as f:
            f.write(
                "def get_available_providers():\n"
                "    return ['CPUExecutionProvider']\n"
            )
        di = os.path.join(store, "onnxruntime-1.28.0.dist-info")
        os.makedirs(di, exist_ok=True)
        with open(os.path.join(di, "WHEEL"), "w") as f:
            f.write("Tag: %s\n" % _current_cp_tag())
        with open(os.path.join(di, "METADATA"), "w") as f:
            f.write(
                "Metadata-Version: 2.1\nName: onnxruntime\nVersion: 1.28.0\n"
            )

    class _Proc:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def stdout(self):
            return iter([])

        def wait(self):
            return 0

    real = rm.subprocess
    monkeypatch.setattr(
        rm,
        "subprocess",
        types.SimpleNamespace(
            Popen=lambda cmd, *a, **k: (_fake_install(cmd), _Proc())[1],
            run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
            PIPE=real.PIPE,
            STDOUT=real.STDOUT,
        ),
    )

    rm.install_onnxruntime("cpu", progress_callback=lambda v, m: None)

    assert not os.path.exists(numpy_di)
    assert rm.store_is_compatible_with_python() is True
    st = rm.onnx_status()
    assert st["installed"] is True
    assert st["version"] == "1.28.0"


def test_reinstall_worker_accepts_two_arg_progress_callback(monkeypatch):
    """Regression: the installers call ``progress_callback(percent, message)``
    with TWO args; the dialog's worker used ``lambda v:`` (one arg) -> every
    Install/Retry in the app crashed with TypeError instead of installing."""
    import src.ui.dependencies_dialog as dd

    calls = []

    def _fake_reinstall(dep_id, progress_callback=None, parent=None,
                        abort_event=None, timeout=None):
        progress_callback(33, "Installing ONNX Runtime...")
        progress_callback(100, "ONNX Runtime installed")

    monkeypatch.setattr(dd.dependencies, "reinstall", _fake_reinstall)

    worker = dd._ReinstallWorker("onnxruntime")
    ok, failed = [], []
    worker.finished_ok.connect(ok.append)
    worker.failed.connect(lambda did, msg: failed.append(msg))
    worker.run()
    assert ok == ["onnxruntime"]
    assert failed == []
    worker.deleteLater()


def test_update_chain_worker_accepts_two_arg_progress_callback(monkeypatch):
    """Same contract bug in update_chain.py: _DependencyUpdateWorker._progress
    must accept (percent, message), not just percent."""
    import src.ui.update_chain as uc

    def _fake_install(progress_callback=None, parent=None, abort_event=None):
        progress_callback(50, "Installing PyTorch...")

    worker = uc._DependencyUpdateWorker("torch", _fake_install)
    ok, failed = [], []
    worker.finished_ok.connect(lambda: ok.append(True))
    worker.failed.connect(failed.append)
    worker.run()
    assert ok == [True]
    assert failed == []
    worker.deleteLater()


def test_can_import_rejects_empty_namespace_package(monkeypatch, tmp_path):
    """An empty ``pkg/`` dir on sys.path imports as a namespace package with
    spec.origin == None. We must NOT treat that as 'installed'."""
    from src.runtimes.manager import _can_import

    pkg = tmp_path / "empty_ns_pkg"
    pkg.mkdir()  # no __init__.py -> namespace package
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _can_import("empty_ns_pkg") is False


def test_installers_are_separated_in_registry():
    """torch and onnxruntime must have independent reinstall fns."""
    import src.dependencies as dependencies
    import src.runtimes.manager as rm

    fns = {e["id"]: e.get("reinstall_fn") for e in dependencies.REGISTRY}
    assert fns["torch"] is rm.install_pytorch
    assert fns["onnxruntime"] is rm.install_onnxruntime
    assert fns["torch"] is not fns["onnxruntime"]


def test_pytorch_install_command_includes_onnx_package():
    """torch.onnx.export (torch 2.6+) needs the 'onnx' package; the
    PyTorch installer must install it alongside torch."""
    from src.runtimes.manager import _build_pytorch_command

    cmd = _build_pytorch_command("cpu")
    assert "torch" in cmd
    assert "onnx" in cmd
    # Se instala con --target (store local) y desde el índice de PyTorch.
    assert "--target" in cmd
    assert any("download.pytorch.org" in part for part in cmd)


def test_runtime_pip_failure_surfaces_pip_output(monkeypatch):
    """_run_pip must include the tail of pip's output in the exception."""
    import types
    import src.runtimes.manager as rm

    class _Proc:
        def __init__(self):
            self.stdout = iter([
                "Collecting bogus\n",
                "  ERROR: Could not find a version that satisfies the requirement\n",
                "No matching distribution found for bogus\n",
            ])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def wait(self):
            return 1

    def _fake_popen(cmd, *a, **k):
        return _Proc()

    def _fake_run(*a, **k):
        return types.SimpleNamespace(returncode=0, stdout="pip 23.0\n", stderr="")

    real = rm.subprocess
    monkeypatch.setattr(rm, "subprocess", types.SimpleNamespace(
        Popen=_fake_popen, run=_fake_run, PIPE=real.PIPE, STDOUT=real.STDOUT
    ))

    raised = False
    msg = ""
    try:
        rm._run_pip(["pip", "install", "bogus"], lambda p, m: None, 0, 100, "install")
    except RuntimeError as exc:
        raised = True
        msg = str(exc)
    assert raised
    assert "No matching distribution" in msg
    assert "pip exited with code 1" in msg


def test_download_fraction_parses_pip_progress_lines():
    from src.runtimes.manager import _download_fraction

    assert _download_fraction("  2.4/2.4 GB ? 0:01:26") == 1.0
    assert _download_fraction("  1.2/2.4 GB ? 0:00:43") == 0.5
    assert _download_fraction("  6.1/6.1 MB 73.4 MB/s 0:00:00") == 1.0
    assert _download_fraction("Collecting torch") is None
    assert _download_fraction("") is None


def test_friendly_phase_maps_pip_lines():
    from src.runtimes.manager import _friendly_phase

    assert _friendly_phase("Collecting torch") == "Preparando la descarga..."
    assert _friendly_phase("Downloading torch-2.5.1.whl") == "Descargando..."
    assert _friendly_phase("Installing collected packages: numpy, torch") == "Instalando..."
    assert _friendly_phase("Successfully installed numpy-2.4.6") == "Completando..."
    assert _friendly_phase("") == ""


def test_run_pip_hides_internals_and_tracks_download(monkeypatch):
    """Regression (UX): the progress UI must receive ONLY friendly Spanish
    phases and honest, monotonic percentages — never raw pip output lines."""
    import types
    import src.runtimes.manager as rm

    class _Proc:
        def __init__(self):
            self._lines = iter([
                "Looking in indexes: https://download.pytorch.org/whl/cpu\n",
                "Collecting torch\n",
                "Downloading torch-2.5.1%2Bcpu-cp311-cp311-win_amd64.whl (100.0 MB)\n",
                "  50.0/100.0 MB ? 0:00:30\n",
                "  100.0/100.0 MB ? 0:00:59\n",
                "Installing collected packages: numpy, torch\n",
                "Successfully installed numpy-2.4.6 torch-2.5.1\n",
            ])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def stdout(self):
            return self._lines

        def wait(self):
            return 0

    real = rm.subprocess
    monkeypatch.setattr(rm, "subprocess", types.SimpleNamespace(
        Popen=lambda *a, **k: _Proc(),
        run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="pip 26\n", stderr=""),
        PIPE=real.PIPE,
        STDOUT=real.STDOUT,
    ))

    calls = []
    rm._run_pip(
        ["pip", "install", "torch"],
        lambda p, m: calls.append((p, m)),
        5, 99, "Descargando PyTorch...",
    )

    assert calls, "progress callback must fire"
    # Nothing raw from pip may leak into the UI.
    raw = " ".join(m for _, m in calls).lower()
    for leaked in ("collecting torch", "successfully installed", "looking in indexes",
                   "pip exited", "wheel", "download.pytorch.org"):
        assert leaked not in raw, f"pip internals leaked into UI: {leaked!r}"
    # Percentages stay within the promised window and never go backwards.
    pcts = [p for p, _ in calls]
    assert pcts == sorted(pcts)
    assert all(5 <= p <= 99 for p in pcts)
    assert pcts[-1] == 99
    # The download fraction must drive the bar (100 MB -> 99%).
    assert any(p >= 95 for p in pcts)


def test_run_pip_cancels_when_abort_event_set(monkeypatch):
    """A set abort_event must kill the pip process and raise
    'cancelled by user' instead of hanging forever."""
    import threading
    import types
    import src.runtimes.manager as rm

    class _Proc:
        def __init__(self):
            self._lines = iter(["Collecting torch\n"])
            self._terminated = False

        @property
        def stdout(self):
            return self._lines

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0 if self._terminated else None

        def terminate(self):
            self._terminated = True

        def kill(self):
            self._terminated = True

    real = rm.subprocess
    monkeypatch.setattr(rm, "subprocess", types.SimpleNamespace(
        Popen=lambda *a, **k: _Proc(),
        run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="pip\n", stderr=""),
        PIPE=real.PIPE,
        STDOUT=real.STDOUT,
    ))

    abort = threading.Event()
    abort.set()
    with pytest.raises(RuntimeError, match="cancelled by user"):
        rm._run_pip(
            ["pip", "install", "torch"], None, 0, 100, "install",
            abort_event=abort,
        )


def test_run_pip_times_out_with_wall_clock(monkeypatch):
    """A wall-clock timeout must stop pip and raise a clear error instead
    of letting 'Install' spin forever on a stalled network."""
    import types
    import src.runtimes.manager as rm

    class _Proc:
        def __init__(self):
            self._lines = iter(["Collecting torch\n"])
            self._terminated = False

        @property
        def stdout(self):
            return self._lines

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0 if self._terminated else None

        def terminate(self):
            self._terminated = True

        def kill(self):
            self._terminated = True

    real = rm.subprocess
    monkeypatch.setattr(rm, "subprocess", types.SimpleNamespace(
        Popen=lambda *a, **k: _Proc(),
        run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="pip\n", stderr=""),
        PIPE=real.PIPE,
        STDOUT=real.STDOUT,
    ))

    with pytest.raises(RuntimeError, match="timed out"):
        # timeout=0 → el deadline ya está vencido al entrar en el bucle.
        rm._run_pip(
            ["pip", "install", "torch"], None, 0, 100, "install",
            timeout=0,
        )


def test_download_file_aborts_and_removes_partial(tmp_path, monkeypatch):
    """Single-stream download must honor abort_event and clean up the
    partial file instead of leaving a corrupt archive behind."""
    import threading
    import src.ffmpeg.downloader as dl

    dest = tmp_path / "partial.zip"

    class _Resp:
        def __init__(self):
            self._chunks = iter([b"x" * 16])

        def raise_for_status(self):
            return None

        @property
        def headers(self):
            return {"content-length": "32"}

        def iter_content(self, chunk_size=0):
            for chunk in self._chunks:
                yield chunk

    def _fake_get(url, **kwargs):
        return _Resp()

    monkeypatch.setattr(dl.requests, "get", _fake_get)

    abort = threading.Event()
    abort.set()
    with pytest.raises(RuntimeError, match="cancelled by user"):
        dl._download_file("https://example.com/ffmpeg.zip", str(dest), abort_event=abort)
    assert not dest.exists()


def test_deadline_check_raises_on_timeout():
    """_deadline_check must translate an expired wall-clock budget into a
    RuntimeError (cancellation is tested via the download tests)."""
    import time
    import src.ffmpeg.downloader as dl

    with pytest.raises(RuntimeError, match="timed out"):
        dl._deadline_check(None, time.monotonic() - 1)


# ==--------------------------------------------------------------------==
# 3. KIND_MODEL rows stay out of the Binaries tab (UI)
# ==--------------------------------------------------------------------==

def test_dependencies_dialog_excludes_model_rows_from_binaries_tab(qapp):
    from src import dependencies as depmod
    from src.ui.dependencies_dialog import DependenciesDialog

    dialog = DependenciesDialog()
    try:
        dialog._populate()
        kinds = [r.info.get("kind") for r in dialog._row_widgets]
        # The Binaries tab must never render a model slot.
        assert depmod.KIND_MODEL not in kinds
        assert depmod.KIND_BINARY in kinds
        assert depmod.KIND_RUNTIME in kinds
        assert len(dialog._row_widgets) == 4
    finally:
        dialog.close()


def test_list_all_still_exposes_model_slots():
    """The KIND_MODEL filter is UI-only; list_all() still sees the 4 slots."""
    from src import dependencies as depmod
    kinds = [i["kind"] for i in depmod.list_all()]
    assert kinds.count(depmod.KIND_MODEL) == 4


# ==--------------------------------------------------------------------==
# 4. UI responsiveness (resize + tab switch, no crash)
# ==--------------------------------------------------------------------==

def test_dependencies_dialog_responsive(qapp):
    from src.ui.dependencies_dialog import DependenciesDialog

    dialog = DependenciesDialog()
    dialog._check_timer.stop()  # avoid background upstream check during test
    dialog.show()
    try:
        # Large size is honoured exactly.
        dialog.resize(1200, 800)
        qapp.processEvents()
        assert dialog.width() == 1200 and dialog.height() == 800

        # A size below the 820x560 minimum is clamped, not ignored.
        dialog.resize(600, 400)
        qapp.processEvents()
        assert dialog.width() >= 820 and dialog.height() >= 560

        # Tab switch (Engines tab) must not crash.
        assert dialog.tabs.count() == 2
        dialog.tabs.setCurrentIndex(1)
        qapp.processEvents()
        assert dialog.tabs.currentIndex() == 1
        dialog.tabs.setCurrentIndex(0)
        qapp.processEvents()
    finally:
        dialog.close()


def test_preferences_dialog_diagnostics_tab(qapp, monkeypatch):
    """La pestaña Diagnostics muestra la ruta de logs y el botón abre la
    carpeta en el gestor de archivos (QDesktopServices.openUrl)."""
    from src.ui import preferences_dialog as pd

    opened = []
    monkeypatch.setattr(
        pd.QDesktopServices, "openUrl", lambda url: opened.append(url.toString())
    )

    dialog = pd.PreferencesDialog()
    try:
        assert dialog.tabs.count() == 4
        assert dialog.tabs.tabText(2) == "Diagnostics"
        assert dialog.lbl_logs_dir.text()
        assert dialog.lbl_backups.text().startswith("Crash report backups")
        dialog._open_logs_folder()
        dialog._open_backups_folder()
        assert len(opened) == 2
        assert opened[0].startswith("file:")
        assert opened[1].startswith("file:") and opened[1].endswith("backup")
    finally:
        dialog.close()


def test_preferences_dialog_about_tab(qapp):
    """La pestaña About muestra versión, arquitectura y enlaces al repo/issues."""
    from src.ui import preferences_dialog as pd
    from src.config import APP_NAME, APP_VERSION, ISSUES_URL, REPO_URL

    dialog = pd.PreferencesDialog()
    try:
        about_widget = dialog.tabs.widget(3)
        assert dialog.tabs.tabText(3) == "About"
        # Recorre las QLabel de la pestaña y verifica versión + links.
        from PySide6.QtWidgets import QLabel

        joined = "\n".join(child.text() for child in about_widget.findChildren(QLabel))
        assert APP_NAME in joined
        assert f"Version {APP_VERSION}" in joined
        assert REPO_URL in joined
        assert ISSUES_URL in joined
    finally:
        dialog.close()


# ==--------------------------------------------------------------------==
# 5. Model-engine install dispatch (mark_installed / cancel) — no network
# ==--------------------------------------------------------------------==

class _NoOpQMessageBox:
    """Stand-in for QMessageBox that never blocks the event loop."""

    class _Btn:
        pass

    StandardButton = type("StandardButton", (), {"Yes": object(), "Ok": object()})

    def question(self, *a, **k):
        return self.StandardButton.Yes

    def information(self, *a, **k):
        return

    def critical(self, *a, **k):
        return

    def warning(self, *a, **k):
        return


def test_engine_install_success_marks_installed(isolated_engines_state, qapp, monkeypatch):
    from src.ui import dependencies_dialog as dd
    from src.ai.engines_state import get_state
    from src.ai.registry import get_engine_dir

    eid = "upscale-realesrgan-x4plus"
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    open(os.path.join(target, "README.md"), "w").close()
    open(os.path.join(target, "RealESRGAN_x4plus.pth"), "w").close()

    class _FakeWorker(dd.HFDownloadWorker):
        def run(self):
            self.finished_ok.emit(self.target_dir)

    monkeypatch.setattr(dd, "is_hf_hub_available", lambda: True)
    monkeypatch.setattr(dd, "HFDownloadWorker", _FakeWorker)
    monkeypatch.setattr(dd, "QMessageBox", _NoOpQMessageBox())

    dialog = dd.DependenciesDialog()
    dialog._check_timer.stop()
    try:
        dialog._on_engine_install(eid)
        for _ in range(300):
            qapp.processEvents()
            if dialog._git_worker is None:
                break
        st = get_state(eid)
        assert st["installed"] is True
        assert st["install_path"] == target
        assert dialog._git_worker is None
    finally:
        dialog.close()


def test_engine_install_without_weights_marks_failed(isolated_engines_state, qapp, monkeypatch):
    """A download that produces no weight files (code-only repo) must NOT be
    marked installed — the render would fail later with no explanation."""
    from src.ui import dependencies_dialog as dd
    from src.ai.engines_state import get_state
    from src.ai.registry import get_engine_dir

    eid = "upscale-omnisr-x4"
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    for f in ("README.md", "train.py", "test.py"):
        open(os.path.join(target, f), "w").close()

    class _FakeWorker(dd.HFDownloadWorker):
        def run(self):
            self.finished_ok.emit(self.target_dir)

    monkeypatch.setattr(dd, "is_hf_hub_available", lambda: True)
    monkeypatch.setattr(dd, "HFDownloadWorker", _FakeWorker)
    monkeypatch.setattr(dd, "QMessageBox", _NoOpQMessageBox())

    dialog = dd.DependenciesDialog()
    dialog._check_timer.stop()
    try:
        dialog._on_engine_install(eid)
        for _ in range(300):
            qapp.processEvents()
            if dialog._git_worker is None:
                break
        st = get_state(eid)
        assert st["installed"] is False
        assert "no model weight" in (st.get("last_error") or "").lower()
        assert dialog._git_worker is None
    finally:
        dialog.close()


def test_engine_install_failure_marks_failed(isolated_engines_state, qapp, monkeypatch):
    from src.ui import dependencies_dialog as dd
    from src.ai.engines_state import get_state

    eid = "upscale-realesrgan-x4plus"

    class _FailWorker(dd.HFDownloadWorker):
        def run(self):
            self.failed.emit("simulated download boom")

    monkeypatch.setattr(dd, "is_hf_hub_available", lambda: True)
    monkeypatch.setattr(dd, "HFDownloadWorker", _FailWorker)
    monkeypatch.setattr(dd, "QMessageBox", _NoOpQMessageBox())

    dialog = dd.DependenciesDialog()
    dialog._check_timer.stop()
    try:
        dialog._on_engine_install(eid)
        for _ in range(300):
            qapp.processEvents()
            if dialog._git_worker is None:
                break
        st = get_state(eid)
        assert st["installed"] is False
        assert "simulated download boom" in (st.get("last_error") or "")
        assert dialog._git_worker is None
    finally:
        dialog.close()


def test_hf_download_worker_cancel_contract(qapp):
    """HFDownloadWorker.cancel() flips the internal cancel flag immediately."""
    from src.ai.fetcher import HFDownloadWorker

    w = HFDownloadWorker("owner/repo", "/nonexistent/target", parent=qapp)
    assert w._cancel.is_set() is False
    w.cancel()
    assert w._cancel.is_set() is True
    w.deleteLater()


# ==--------------------------------------------------------------------==
# 6. GPU NVENC rescale + compress smoke test (real encode)
# ==--------------------------------------------------------------------==

def _ffmpeg_full():
    from src.ffmpeg.manager import get_local_ffmpeg_full_path
    p = get_local_ffmpeg_full_path()
    return p if p and os.path.exists(p) else None


def _has_nvenc(ffmpeg_path):
    try:
        out = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return False
    return "h264_nvenc" in (out.stdout or "")


@pytest.fixture()
def nvenc_clip():
    """Generate a tiny test clip once; skip if NVENC is unavailable."""
    ffmpeg = _ffmpeg_full()
    if ffmpeg is None:
        pytest.skip("FFmpeg full build not present")
    if not _has_nvenc(ffmpeg):
        pytest.skip("NVENC encoder not available on this FFmpeg build/GPU")

    tmp = tempfile.mkdtemp(prefix="draggy_nvenc_test_")
    clip = os.path.join(tmp, "src.mp4")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=640x360:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        # The BtbN 'full' build is LGPL-shared (no libx264), but it ships
        # NVENC, so generate the source clip with NVENC too.
        "-c:v", "h264_nvenc", "-c:a", "aac", clip,
    ]
    rc = subprocess.run(cmd, capture_output=True, timeout=60).returncode
    if rc != 0 or not os.path.exists(clip):
        shutil.rmtree(tmp, ignore_errors=True)
        pytest.skip("could not generate baseline test clip with FFmpeg")
    yield ffmpeg, clip, tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_compress_video_rescales_to_target_height_with_nvenc(nvenc_clip):
    """compress_video() must downscale to the target height and use NVENC."""
    from src.processing.pipeline import compress_video

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "out_180.mp4")
    settings = {
        "codec": "h264_nvenc",
        "resolution": "180",
        "bitrate": "600",
        "audio": "copy",
        "audio_boost": False,
        "format": "mp4",
        "preset": "medium",
        "pixel_format": "yuv420p",
    }
    ok = compress_video(ffmpeg, src, dst, settings)
    assert ok is True
    assert os.path.exists(dst) and os.path.getsize(dst) > 0

    # Probe the output with the ffprobe that ships next to the full ffmpeg
    # (in a normal user install get_video_info() would work too; here the dev
    # checkout has no primary bin/ffmpeg/ffprobe, so probe directly).
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    assert os.path.exists(ffprobe), "ffprobe not shipped with FFmpeg full build"
    info = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,codec_name",
         "-of", "default=noprint_wrappers=1", dst],
        capture_output=True, text=True, timeout=30,
    ).stdout
    fields = {}
    for line in info.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()

    # scale=-2:180 keeps aspect ratio: 640x360 -> 320x180.
    assert fields.get("height") == "180", fields
    assert fields.get("width") == "320", fields
    assert fields.get("codec_name") == "h264", fields


def test_build_args_uses_requested_nvenc_codec():
    """build_args() must emit -c:v h264_nvenc for codec=h264_nvenc."""
    from src.ffmpeg.runner import build_args

    args = build_args("/in.mp4", "/out.mp4", {
        "codec": "h264_nvenc",
        "bitrate": "1500",
        "audio": "copy",
        "format": "mp4",
    })
    cv = args[args.index("-c:v") + 1]
    assert cv == "h264_nvenc"


def test_compress_batch_rescale_with_nvenc(nvenc_clip):
    """Batch path: rescale *several* clips to a fixed height with NVENC.

    Mirrors the '70 videos de Shin-chan' use case — same per-clip NVENC path
    run in a loop. We use a handful of small clips to keep the test fast while
    still exercising the resize + encode + probe loop.
    """
    from src.processing.pipeline import compress_video

    ffmpeg, src, tmp = nvenc_clip
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    assert os.path.exists(ffprobe)

    settings = {
        "codec": "h264_nvenc",
        "resolution": "240",
        "bitrate": "500",
        "audio": "copy",
        "audio_boost": False,
        "format": "mp4",
        "preset": "medium",
        "pixel_format": "yuv420p",
    }

    n = 5
    for i in range(n):
        dst = os.path.join(tmp, "batch_%d.mp4" % i)
        ok = compress_video(ffmpeg, src, dst, settings)
        assert ok is True, "clip %d failed" % i
        assert os.path.exists(dst) and os.path.getsize(dst) > 0

        info = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name",
             "-of", "default=noprint_wrappers=1", dst],
            capture_output=True, text=True, timeout=30,
        ).stdout
        fields = {}
        for line in info.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k.strip()] = v.strip()
        # 640x360 -> scale=-2:240 -> 426x240 (426 = 640 * 240/360, even).
        assert fields.get("height") == "240", (i, fields)
        assert fields.get("codec_name") == "h264", (i, fields)

    produced = [f for f in os.listdir(tmp) if f.startswith("batch_") and f.endswith(".mp4")]
    assert len(produced) == n


# ==--------------------------------------------------------------------==
# 7. Self-contained ("caracol"): only the app-downloaded store counts
# ==--------------------------------------------------------------------==

class _evicted_module:
    """Context manager: evicts ``name`` from sys.modules for the duration of
    the test and restores the previous state afterwards, so a fake store
    package is what actually gets imported (and real runtime caches are not
    corrupted for later tests in the same session)."""

    def __init__(self, name):
        self.name = name
        self.saved = None

    def __enter__(self):
        self.saved = sys.modules.pop(self.name, None)
        return self

    def __exit__(self, *a):
        if self.saved is not None:
            sys.modules[self.name] = self.saved
        else:
            sys.modules.pop(self.name, None)


def _make_fake_store_pkg(tmp_path, name, version, with_metadata=True):
    """Create ``name`` (torch/onnxruntime) inside ``tmp_path`` with a
    matching current-CPython WHEEL tag, so the version guard accepts it."""
    store = str(tmp_path)
    pkg = os.path.join(store, name)
    os.makedirs(pkg)
    if name == "torch":
        init = "__version__ = '%s'\n" % version
    else:
        init = (
            "def get_available_providers():\n"
            "    return ['CPUExecutionProvider']\n"
        )
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write(init)
    di = os.path.join(store, "%s-%s.dist-info" % (name, version))
    os.makedirs(di)
    with open(os.path.join(di, "WHEEL"), "w") as f:
        f.write("Tag: %s\n" % _current_cp_tag())
    if with_metadata:
        with open(os.path.join(di, "METADATA"), "w") as f:
            f.write(
                "Metadata-Version: 2.1\nName: %s\nVersion: %s\n" % (name, version)
            )
    return store


def test_torch_status_missing_when_only_system_torch(monkeypatch, tmp_path):
    """The venv/system has torch, the local store is empty: the app must
    report NOT installed (it never borrows the system runtime)."""
    import src.runtimes.manager as rm

    store = str(tmp_path / "empty_store")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    # The assertion is about the diagnostic branch, not whether this
    # checkout happens to have torch installed globally.
    monkeypatch.setattr(rm, "_can_import_system", lambda _name: True)

    st = rm.torch_status()
    assert st["installed"] is False
    assert st["local"] is False
    assert st["version"] == "—"
    assert st["system"] is True  # diagnostic only
    assert st["install_dir"] == store


def test_torch_status_installed_when_store_has_local_copy(monkeypatch, tmp_path):
    """A real package inside the local store -> INSTALLED, local=True."""
    import src.runtimes.manager as rm

    store = _make_fake_store_pkg(tmp_path, "torch", "2.0.0-fake")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.syspath_prepend(store)

    with _evicted_module("torch"):
        st = rm.torch_status()
    assert st["installed"] is True
    assert st["local"] is True
    assert st["system"] is False
    assert st["version"] == "2.0.0-fake"
    assert st["gpu"] is False
    assert st["install_dir"] == os.path.join(store, "torch")


def test_onnx_status_installed_when_store_has_local_copy(monkeypatch, tmp_path):
    import src.runtimes.manager as rm

    store = _make_fake_store_pkg(tmp_path, "onnxruntime", "1.28.0-fake")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.syspath_prepend(store)

    with _evicted_module("onnxruntime"):
        st = rm.onnx_status()
    assert st["installed"] is True
    assert st["local"] is True
    assert st["system"] is False
    assert st["version"] == "1.28.0-fake"
    assert st["gpu"] is False
    assert st["install_dir"] == os.path.join(store, "onnxruntime")


def test_dependencies_torch_row_missing_when_only_system_copy(monkeypatch, tmp_path):
    """REGISTRY row for torch: a system-only copy must be MISSING, with a
    user-friendly message (no jargon about the bundled-runtime policy)."""
    import src.dependencies as depmod
    import src.runtimes.manager as rm

    store = str(tmp_path / "empty_store")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)

    info = depmod._pytorch_status()
    assert info["status"] == depmod.STATUS_MISSING
    assert info["install_dir"] == store
    assert "Pulsa 'Instalar'" in info["detail"]
    assert info["deletable"] is False


def test_dependencies_torch_row_installed_when_local_store_copy(monkeypatch, tmp_path):
    """REGISTRY row for torch: a copy in the local store -> INSTALLED."""
    import src.dependencies as depmod
    import src.runtimes.manager as rm

    store = _make_fake_store_pkg(tmp_path, "torch", "2.0.0-fake")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.syspath_prepend(store)

    with _evicted_module("torch"):
        info = depmod._pytorch_status()
    assert info["status"] == depmod.STATUS_INSTALLED
    assert info["install_dir"] == store
    assert info["local"] is True
    assert info["deletable"] is True


# ==--------------------------------------------------------------------==
# 8. Face restore category
# ==--------------------------------------------------------------------==

def test_face_restore_catalog_is_implemented():
    """FaceRestore must have >=3 downloadable engines, all wired (not COMING
    SOON), with slug-safe ids and dirs under bin/models/FaceRestore."""
    from src.ai.registry import CATALOG, get_engine_dir

    entries = [e for e in CATALOG if e.get("category") == "FaceRestore"]
    assert len(entries) >= 3
    for e in entries:
        assert e["is_implemented"] is True, e["engine_id"]
        assert e["hf_repo_id"], e["engine_id"]
        assert e["hf_repo_id"].startswith("http") is False  # repo_id, not url
        path = get_engine_dir(e["engine_id"])
        assert "facerestore" in path, path
    ids = {e["engine_id"] for e in entries}
    assert {"facerestore-gfpgan", "facerestore-codeformer", "facerestore-gpen-512"} <= ids


def test_face_restore_hf_repos_are_real():
    """The FaceRestore repo_ids must point at existing HuggingFace repos.

    ``TencentARC/GFPGAN`` and ``sczhou/CodeFormer`` return 404 from the
    HuggingFace API — downloads failed silently even with a valid token.
    These are the verified replacements (checked via HfApi.list_repo_files)."""
    from src.ai.registry import CATALOG_BY_ID

    expected = {
        "facerestore-gfpgan": ("nlightcho/gfpgan_v14", "GFPGANv1.4.pth"),
        "facerestore-codeformer": ("wwy/codeformer", "codeformer.pth"),
        "facerestore-gpen-512": ("akhaliq/GPEN-BFR-512", "GPEN-BFR-512.pth"),
    }
    for eid, (repo, sentinel) in expected.items():
        entry = CATALOG_BY_ID[eid]
        assert entry["hf_repo_id"] == repo, eid
        assert entry["sentinel"] == sentinel, eid


def test_workflow_maps_face_restore_node(qapp):
    """The workflow has a FaceRestore node wired to the catalog category,
    and it builds fine without the icon asset (added later by the user)."""
    from src.ui.workflow_widget import NODE_CATEGORY, WorkflowWidget

    assert NODE_CATEGORY["FaceRestore"][0] == "FaceRestore"

    widget = WorkflowWidget()
    try:
        names = [s.name for s in widget.middle_steps]
        assert "FaceRestore" in names
        step = next(s for s in widget.middle_steps if s.name == "FaceRestore")
        assert step.category == "FaceRestore"
        # A missing icon file must not crash construction or the row layout.
        assert step.icon_label.pixmap() is None or step.icon_label.pixmap().isNull()
        # Active nodes report the category as engine key (pipeline contract).
        step.set_active(True)
        engines = widget.get_selected_engines()
        assert "FaceRestore" in engines
    finally:
        widget.deleteLater()


def test_workflow_scale_dropdown_offers_model_native(qapp, isolated_engines_state):
    """The Upscale node's scale dropdown offers 'Model native' first and
    carries the native flag + real model path so the pipeline can size the
    encoder to the model's own output."""
    import src.ai.engines_state as es
    from src.ai.registry import get_engine_dir
    from src.ui.workflow_widget import WorkflowWidget

    eid = "upscale-realesr-animevideov3"  # catalog declares 2x/3x/4x
    target = get_engine_dir(eid)
    os.makedirs(target, exist_ok=True)
    # Real weight file with an unambiguous x4 scale in the name.
    open(os.path.join(target, "realesr-animevideov3-x4.pth"), "w").close()
    es.mark_installed(eid, install_path=target)

    widget = WorkflowWidget()
    try:
        ups = next(s for s in widget.middle_steps if s.name == "Upscale")
        ups.set_active(True)
        ups.combo.setCurrentIndex(0)  # select the installed engine

        assert not ups.combo_scale.isHidden()
        assert ups.combo_scale.count() >= 4  # Model native + x2/x3/x4
        texts = [ups.combo_scale.itemText(i) for i in range(ups.combo_scale.count())]
        assert texts[0] == "Model native", texts
        # Default selection IS Model native -> output = exactly the model's.
        assert ups.combo_scale.currentIndex() == 0

        info = ups.get_selected_model_info()
        assert info["native"] is True
        assert info["scale"] == 4  # native scale from the catalog
        assert info["model_path"] is not None
        # Switching to an explicit scale clears the native flag.
        for i in range(ups.combo_scale.count()):
            if ups.combo_scale.itemText(i) == "2x":
                ups.combo_scale.setCurrentIndex(i)
        info2 = ups.get_selected_model_info()
        assert info2["native"] is False
        assert info2["scale"] == 2
    finally:
        widget.deleteLater()


def test_import_step_video_paths_from_mime(tmp_path):
    """video_paths_from_mime extrae solo URLs locales con extensión de
    video y que existen en disco, en orden de arrastre."""
    from PySide6.QtCore import QMimeData, QUrl
    from src.ui.workflow_widget import video_paths_from_mime

    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"x")
    text = tmp_path / "notes.txt"
    text.write_bytes(b"x")
    missing_mkv = tmp_path / "missing.mkv"

    mime = QMimeData()
    mime.setUrls([
        QUrl.fromLocalFile(str(vid)),
        QUrl.fromLocalFile(str(text)),
        QUrl("https://huggingface.co/owner/4x-UltraSharp"),
        QUrl.fromLocalFile(str(missing_mkv)),
    ])
    # QUrl.fromLocalFile normaliza a "/"; comparamos con normpath.
    out = [os.path.normpath(p) for p in video_paths_from_mime(mime)]
    assert out == [str(vid)]
    assert video_paths_from_mime(QMimeData()) == []


def test_import_step_drop_event_emits_video_dropped(qapp, tmp_path):
    """Un drop real de videos sobre la casilla Import emite
    ``video_dropped`` con las rutas válidas y rechaza archivos no-video."""
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent
    from src.ui.workflow_widget import WorkflowWidget

    vid = tmp_path / "movie.mkv"
    vid.write_bytes(b"x")

    widget = WorkflowWidget()
    try:
        got = []
        widget.video_dropped.connect(got.append)

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(vid))])
        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        widget.step_import.dropEvent(event)
        assert [os.path.normpath(p) for p in got[0]] == [str(vid)]

        # Un drop sin videos no emite nada.
        bad = QMimeData()
        bad.setUrls([QUrl.fromLocalFile(str(tmp_path / "x.txt"))])
        event_bad = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            bad,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        widget.step_import.dropEvent(event_bad)
        assert len(got) == 1  # el drop sin videos no emite nada
    finally:
        widget.deleteLater()


def test_main_window_accepts_dropped_video_paths(tmp_path, monkeypatch):
    """El handler de drag & drop de MainWindow fija la selección igual que
    el diálogo: filtra inexistentes y actualiza input_paths/input_path."""
    from src.ui.main_window import MainWindow

    vid = tmp_path / "a.mp4"
    vid.write_bytes(b"x")

    mw = MainWindow.__new__(MainWindow)
    mw.input_path = None
    mw.input_paths = []
    statuses = []
    mw.status_label = type("L", (), {"setText": lambda self, t: statuses.append(t)})()
    updates = []
    mw._update_original_bitrate = lambda p: updates.append(p)

    mw._on_video_dropped([str(vid), str(tmp_path / "nope.mkv")])
    assert mw.input_paths == [str(vid)]
    assert mw.input_path == str(vid)
    assert statuses == ["Selected: a.mp4"]
    assert updates == [str(vid)]


def test_list_all_has_face_restore_slot(isolated_engines_state):
    """The Dependencies REGISTRY exposes a FaceRestore model slot that
    reports honestly against the empty models dir."""
    from src import dependencies as depmod

    slots = {e["id"]: e for e in depmod.REGISTRY if e.get("kind") == depmod.KIND_MODEL}
    assert "face_restore_models" in slots

    info = depmod._slot_status(
        "Face restoration models",
        "FaceRestore",
        str(isolated_engines_state / "models" / "facerestore"),
        depmod.KIND_MODEL,
    )
    assert info["status"] == depmod.STATUS_MISSING
    assert info["kind"] == depmod.KIND_MODEL


# ==--------------------------------------------------------------------==
# 9. System info (OS + Python interpreter) in Hardware Info
# ==--------------------------------------------------------------------==

def test_hardware_summary_includes_os_and_python():
    """get_hardware_summary must report the OS and the running interpreter,
    on Windows and on Linux alike."""
    import platform as _platform
    from src.hardware.detector import get_hardware_summary

    info = get_hardware_summary()

    assert isinstance(info.get("os"), str) and info["os"]
    expected_prefix = (
        "Windows" if _platform.system() == "Windows" else _platform.system().lower()
    )
    assert info["os"].lower().startswith(expected_prefix.lower())
    assert info.get("machine")

    py = info.get("python", {})
    assert py.get("version") == sys.version.split()[0]
    assert py.get("bits") in (32, 64)
    assert py.get("executable") and os.path.basename(py["executable"]).lower() in (
        "python.exe",
        "python",
    )


def _run_startup_worker(monkeypatch, hardware_fn, ffmpeg_fn, codecs_fn):
    """Ejecuta StartupWorker.run() con las tres fases parcheadas y devuelve
    el dict estructurado emitido por startup_finished."""
    import src.app as appmod

    monkeypatch.setattr(appmod, "get_hardware_summary", hardware_fn)
    monkeypatch.setattr(appmod, "ensure_ffmpeg", ffmpeg_fn)
    monkeypatch.setattr(appmod, "test_all_codecs", codecs_fn)

    worker = appmod.StartupWorker()
    results = []
    worker.startup_finished.connect(results.append)
    worker.run()
    assert len(results) == 1
    return results[0]


def test_startup_worker_hardware_failure_is_non_fatal(qapp, monkeypatch):
    """Un fallo de detección de hardware NO debe abortar el arranque ni
    traducirse a 'FFmpeg Error': phase=ok con hardware_error informado."""
    def _hw():
        raise RuntimeError("nvidia-smi exploded")

    result = _run_startup_worker(
        monkeypatch,
        _hw,
        lambda progress_callback=None: "/fake/ffmpeg",
        lambda ffmpeg_path, progress_callback=None: {"CPU": [("libx264", "H.264")]},
    )
    assert result["phase"] == "ok"
    assert result["ffmpeg_path"] == "/fake/ffmpeg"
    assert result["hardware_info"] == {}
    assert result["hardware_error"]
    assert result["compatible_codecs"] == {"CPU": [("libx264", "H.264")]}


def test_startup_worker_ffmpeg_failure_is_fatal(qapp, monkeypatch):
    """Sin FFmpeg el resultado es phase='ffmpeg' y ffmpeg_path=None."""
    result = _run_startup_worker(
        monkeypatch,
        lambda: {"cpu": "x"},
        lambda progress_callback=None: None,
        lambda ffmpeg_path, progress_callback=None: {},
    )
    assert result["phase"] == "ffmpeg"
    assert result["ffmpeg_path"] is None


def test_startup_worker_codec_failure_is_non_fatal(qapp, monkeypatch):
    """Un fallo del test de codecs degrada (lista vacía) sin abortar:
    phase=ok con codecs_error informado."""
    def _codecs(ffmpeg_path, progress_callback=None):
        raise RuntimeError("codec probe crashed")

    result = _run_startup_worker(
        monkeypatch,
        lambda: {"cpu": "x"},
        lambda progress_callback=None: "/fake/ffmpeg",
        _codecs,
    )
    assert result["phase"] == "ok"
    assert result["ffmpeg_path"] == "/fake/ffmpeg"
    assert result["codecs_error"]
    assert result["compatible_codecs"] == {}


def test_startup_worker_generic_failure_reports_startup_phase(qapp, monkeypatch):
    """Un fallo no atribuible a hardware/ffmpeg/codecs se marca phase='startup'
    con el error, no como 'FFmpeg Error'."""
    def _ff(progress_callback=None):
        raise RuntimeError("unexpected boom")

    result = _run_startup_worker(
        monkeypatch,
        lambda: {"cpu": "x"},
        _ff,
        lambda ffmpeg_path, progress_callback=None: {},
    )
    assert result["phase"] == "startup"
    assert result["error"]
    assert result["ffmpeg_path"] is None


def test_on_finished_hardware_warning_not_ffmpeg_error(qapp, monkeypatch):
    """_on_finished con phase=ok y hardware_error muestra el warning
    'Hardware Detection Failed' y arranca la MainWindow, sin traducir el
    fallo a un diálogo crítico 'FFmpeg Error'."""
    import src.app as appmod
    from PySide6 import QtWidgets as qtw

    calls = []

    class _Rec:
        @staticmethod
        def critical(parent, title, text):
            calls.append(("critical", title, text))

        @staticmethod
        def warning(parent, title, text):
            calls.append(("warning", title, text))

    monkeypatch.setattr(qtw, "QMessageBox", _Rec)

    def _fake_exit(code=0):
        calls.append(("exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(appmod.sys, "exit", _fake_exit)

    class _FakeWindow:
        def show(self):
            pass

    window_obj = _FakeWindow()
    captured = {}

    def _fake_main_window(ffmpeg_path, hardware_info, compatible_codecs):
        captured["ffmpeg_path"] = ffmpeg_path
        captured["hardware_info"] = hardware_info
        captured["compatible_codecs"] = compatible_codecs
        return window_obj

    monkeypatch.setattr(appmod, "MainWindow", _fake_main_window)

    app = object.__new__(appmod.DraggyEncoderApp)

    class _Loading:
        def close(self):
            pass

    app.loading = _Loading()

    app._on_finished({
        "phase": "ok",
        "ffmpeg_path": "/fake/ffmpeg",
        "hardware_info": {},
        "compatible_codecs": {"CPU": [("libx264", "H.264")]},
        "error": None,
        "hardware_error": "nvidia-smi exploded",
        "codecs_error": None,
    })

    titles = [c[1] for c in calls]
    assert "FFmpeg Error" not in titles
    assert ("warning", "Hardware Detection Failed") in [
        (c[0], c[1]) for c in calls
    ]
    assert app.window is window_obj
    assert captured["ffmpeg_path"] == "/fake/ffmpeg"
    assert captured["compatible_codecs"] == {"CPU": [("libx264", "H.264")]}


def test_on_finished_ffmpeg_failure_still_critical(qapp, monkeypatch):
    """El caso realmente fatal (sin FFmpeg) SÍ debe mostrar 'FFmpeg Error'
    y terminar con exit 1."""
    import src.app as appmod
    from PySide6 import QtWidgets as qtw

    calls = []

    class _Rec:
        @staticmethod
        def critical(parent, title, text):
            calls.append(("critical", title, text))

    monkeypatch.setattr(qtw, "QMessageBox", _Rec)

    def _fake_exit(code=0):
        calls.append(("exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(appmod.sys, "exit", _fake_exit)

    def _fail_main_window(*a, **k):
        raise AssertionError("MainWindow must not be built")

    monkeypatch.setattr(appmod, "MainWindow", _fail_main_window)

    app = object.__new__(appmod.DraggyEncoderApp)

    class _Loading:
        def close(self):
            pass

    app.loading = _Loading()

    with pytest.raises(SystemExit):
        app._on_finished({
            "phase": "ffmpeg",
            "ffmpeg_path": None,
            "hardware_info": {},
            "compatible_codecs": {},
            "error": None,
            "hardware_error": None,
            "codecs_error": None,
        })

    assert ("critical", "FFmpeg Error") in [(c[0], c[1]) for c in calls]
    assert ("exit", 1) in calls


@pytest.fixture()
def main_module():
    """Importa main.py (activa logging centralizado + excepthook) y restaura
    el excepthook original al terminar, para no dejar el hook de GUI activo
    en el resto de la suite (podría bloquearse esperando un clic)."""
    original_excepthook = sys.excepthook
    import main as mainmod
    try:
        yield mainmod
    finally:
        sys.excepthook = original_excepthook


def _make_exc_info(message):
    try:
        raise RuntimeError(message)
    except RuntimeError:
        return sys.exc_info()


def test_main_excepthook_logs_traceback_to_app_log(main_module, monkeypatch):
    """El excepthook vuelca el traceback completo a bin/logs/app.log."""
    marker = f"boom-log-{os.getpid()}-{time.monotonic_ns()}"
    monkeypatch.setattr(sys, "argv", ["main.py", "--selftest"])  # sin diálogo
    log_path = main_module._APP_LOG_PATH
    assert log_path, "app.log path should be resolved"

    main_module._uncaught_exception(*_make_exc_info(marker))

    with open(log_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert marker in content
    assert "Unhandled exception:" in content


def test_main_excepthook_skips_dialog_in_selftest(main_module, monkeypatch):
    """En --selftest el hook solo registra; nunca muestra un diálogo
    (evita colgar el CI headless esperando un clic)."""
    from PySide6 import QtWidgets as qtw

    calls = []

    class _Rec:
        @staticmethod
        def critical(parent, title, text):
            calls.append((title, text))

    monkeypatch.setattr(qtw, "QMessageBox", _Rec)
    monkeypatch.setattr(sys, "argv", ["main.py", "--selftest"])

    main_module._uncaught_exception(*_make_exc_info("boom-selftest"))

    assert calls == []


def test_main_excepthook_shows_dialog_with_log_path(main_module, qapp, monkeypatch):
    """En modo GUI el hook muestra un diálogo crítico con la ruta del log."""
    from PySide6 import QtWidgets as qtw

    calls = []

    class _Rec:
        @staticmethod
        def critical(parent, title, text):
            calls.append((title, text))

    monkeypatch.setattr(qtw, "QMessageBox", _Rec)
    monkeypatch.setattr(sys, "argv", ["draggy"])

    main_module._uncaught_exception(*_make_exc_info("boom-gui"))

    assert len(calls) == 1
    title, text = calls[0]
    assert "Unexpected Error" in title
    assert "app.log" in text


def test_main_excepthook_crash_report_flag_opens_issue(main_module, qapp, monkeypatch):
    """Con --crash-report, tras el diálogo el hook abre el issue pre-rellenado
    pasándole el traceback completo."""
    from PySide6 import QtWidgets as qtw
    import src.crash_report as cr

    class _Rec:
        @staticmethod
        def critical(parent, title, text):
            pass

    monkeypatch.setattr(qtw, "QMessageBox", _Rec)
    monkeypatch.setattr(sys, "argv", ["draggy", "--crash-report"])
    reported = []
    monkeypatch.setattr(
        cr, "open_crash_report", lambda exc_text="": reported.append(exc_text)
    )

    main_module._uncaught_exception(*_make_exc_info("boom-crash-report"))

    assert len(reported) == 1
    assert "boom-crash-report" in reported[0]


def test_crash_report_build_issue_url(monkeypatch, tmp_path):
    """build_issue_url genera un /issues/new pre-rellenado con título, system
    context y el tail de app.log/faulthandler.log."""
    import src.crash_report as cr
    from urllib.parse import parse_qs, urlparse

    monkeypatch.setattr(cr, "get_logs_dir", lambda: str(tmp_path))
    (tmp_path / "app.log").write_text("hello crash\n" * 100, encoding="utf-8")
    (tmp_path / "faulthandler.log").write_text(
        "Fatal Python error: segfault\n", encoding="utf-8"
    )

    url = cr.build_issue_url("RuntimeError: boom")

    parsed = urlparse(url)
    assert parsed.path.endswith("/issues/new")
    qs = parse_qs(parsed.query)
    assert qs["title"][0] == "Crash report: RuntimeError: boom"
    body = qs["body"][0]
    assert "## What happened" in body
    assert "RuntimeError: boom" in body
    assert "## System" in body
    assert "App version:" in body
    assert "hello crash" in body
    assert "Fatal Python error: segfault" in body


def test_crash_report_read_tail(tmp_path):
    """_read_tail devuelve los últimos N chars y '' para ficheros ausentes."""
    import src.crash_report as cr

    p = tmp_path / "tail.txt"
    content = "abcdefghij" * 100  # 1000 chars
    p.write_text(content, encoding="utf-8")

    assert cr._read_tail(str(p), 50) == content[-50:]
    assert cr._read_tail(str(tmp_path / "missing.txt"), 50) == ""


def test_crash_report_open_calls_webbrowser(monkeypatch, tmp_path):
    """open_crash_report abre el navegador con la URL y la devuelve."""
    import src.crash_report as cr

    opened = []
    monkeypatch.setattr(cr, "get_logs_dir", lambda: str(tmp_path))
    monkeypatch.setattr(cr.webbrowser, "open", lambda url: opened.append(url))

    url = cr.open_crash_report("boom")

    assert opened == [url]
    assert url.startswith(
        "https://github.com/thedevil4k/Draggy-Encoder/issues/new"
    )


def test_log_backup_archive_file(monkeypatch, tmp_path):
    """archive_file copia a bin/logs/backup con nombre prefix-timestamp."""
    import src.log_backup as lb

    monkeypatch.setattr(lb, "get_logs_dir", lambda: str(tmp_path))

    src = tmp_path / "app.log"
    src.write_text("crash data", encoding="utf-8")

    dest = lb.archive_file(str(src), "app")

    assert dest and os.path.isfile(dest)
    assert (tmp_path / "backup").exists()
    assert open(dest, encoding="utf-8").read() == "crash data"
    assert lb.archive_file(str(tmp_path / "missing.log"), "app") is None


def test_log_backup_rotate_keeps_newest(monkeypatch, tmp_path):
    """rotate deja los últimos N respaldos (orden lexicográfico = cronológico)."""
    import src.log_backup as lb

    monkeypatch.setattr(lb, "get_logs_dir", lambda: str(tmp_path))

    backup = tmp_path / "backup"
    backup.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (backup / f"app-20240101-000000-{i:06d}.log").write_text(
            "x", encoding="utf-8"
        )

    lb.rotate("app", keep=3)

    remaining = sorted(backup.glob("app-*.log"))
    assert [p.name for p in remaining] == [
        "app-20240101-000000-000007.log",
        "app-20240101-000000-000008.log",
        "app-20240101-000000-000009.log",
    ]


def test_log_backup_rotating_handler_archives_oldest(monkeypatch, tmp_path):
    """Al rotar, BackupRotatingFileHandler archiva el app.log.N más antiguo
    (que la rotación normal borraría) en backup/ antes de perderlo."""
    import logging
    import src.log_backup as lb

    monkeypatch.setattr(lb, "get_logs_dir", lambda: str(tmp_path))

    handler = lb.BackupRotatingFileHandler(
        str(tmp_path / "app.log"),
        maxBytes=300,
        backupCount=2,
        encoding="utf-8",
        backup_prefix="app",
        keep=100,  # alto para que el test no rote los archivos a medio camino
    )
    logger = logging.getLogger("test_backup_rollover")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        for i in range(80):
            logger.info("line %04d %s", i, "x" * 60)
    finally:
        logger.removeHandler(handler)
        handler.close()

    backups = sorted((tmp_path / "backup").glob("app-*.log"))
    assert backups, "el rollover debió archivar el app.log.N más antiguo"
    content = backups[0].read_text(encoding="utf-8")
    assert "line 0000" in content


def test_log_backup_stats_and_clear(monkeypatch, tmp_path):
    """backup_stats cuenta ficheros y suma bytes; clear_backups los borra."""
    import src.log_backup as lb

    monkeypatch.setattr(lb, "get_logs_dir", lambda: str(tmp_path))

    backup = tmp_path / "backup"
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "app-1.log").write_bytes(b"x" * 100)
    (backup / "faulthandler-1.log").write_bytes(b"y" * 50)

    num, total = lb.backup_stats()
    assert num == 2
    assert total == 150

    assert lb.clear_backups() == 2
    assert lb.backup_stats() == (0, 0)
    assert lb.clear_backups() == 0  # idempotente


def test_system_summary_windows_family_by_build():
    """Windows 10 vs Windows 11 label is decided by the build number."""
    from src.hardware.detector import get_system_summary

    summary = get_system_summary()
    os_name = summary["os_name"]

    # En Windows real el nombre debe derivar del build; en Linux/macOS el
    # nombre es el del sistema. Ambas rutas dejan os_name poblado.
    assert os_name
    if summary["os_name"].startswith("Windows"):
        assert summary["os"].startswith("Windows")
        assert "(" in summary["os"]  # incluye el release p.ej. (10.0.26100)


def test_hardware_dialog_shows_os_and_python(qapp):
    """The Hardware Info dialog renders OS + Python interpreter lines."""
    from PySide6.QtWidgets import QLabel
    from src.ui.hardware_dialog import HardwareDialog

    info = {
        "cpu": {"name": "Test CPU", "cores": 4, "threads": 8},
        "gpus": [],
        "recommended_device": "CPU",
        "os": "Windows 11 (10.0.26100)",
        "machine": "AMD64",
        "python": {"version": "3.11.15", "executable": r"C:\venv\python.exe", "bits": 64},
    }
    dlg = HardwareDialog(info)
    try:
        texts = [lbl.text() for lbl in dlg.findChildren(QLabel)]
        assert any("Windows 11 (10.0.26100) · AMD64" in t for t in texts)
        # Los labels de Python usan rich text (<b>…</b>), así que comprobamos
        # los fragmentos por separado.
        assert any("Python interpreter" in t and "3.11.15 (64-bit)" in t for t in texts)
        assert any(r"C:\venv\python.exe" in t for t in texts)
    finally:
        dlg.deleteLater()


def test_hardware_dialog_shows_gpu_driver_and_vram(qapp):
    """The Hardware Info dialog renders driver, VRAM used/total, clock and
    temperature for each GPU."""
    from PySide6.QtWidgets import QLabel
    from src.ui.hardware_dialog import HardwareDialog

    info = {
        "cpu": {"name": "Test CPU", "cores": 8, "threads": 16},
        "gpus": [{
            "name": "NVIDIA GeForce RTX 4090",
            "memory": "24564 MiB",
            "vendor": "NVIDIA",
            "driver": "560.94",
            "memory_used": "3144 MiB",
            "core_clock": "2520 MHz",
            "temperature": "68 °C",
        }],
        "recommended_device": "NVIDIA GPU",
        "os": "Windows 11 (10.0.26100)",
        "machine": "AMD64",
        "python": {"version": "3.11.15", "executable": r"C:\venv\python.exe", "bits": 64},
    }
    dlg = HardwareDialog(info)
    try:
        texts = [lbl.text() for lbl in dlg.findChildren(QLabel)]
        assert any("NVIDIA GeForce RTX 4090" in t and "NVIDIA" in t for t in texts)
        assert any("VRAM: 3144 MiB / 24564 MiB" in t for t in texts)
        assert any("Driver: 560.94" in t for t in texts)
        assert any("Núcleo (max): 2520 MHz" in t for t in texts)
        assert any("Temperatura: 68 °C" in t for t in texts)
    finally:
        dlg.deleteLater()


def test_detect_gpus_windows_parses_nvidia_smi_metrics(monkeypatch):
    """_detect_gpus_windows must parse driver, VRAM used, clock and
    temperature from the nvidia-smi CSV."""
    import src.hardware.detector as hw

    smi_csv = (
        "NVIDIA GeForce RTX 4090, 560.94, 24564, 3144, 2520, 68\r\n"
        "NVIDIA GeForce GTX 1080, 560.94, 8192, 1024, 1733, 55\r\n"
    )

    def fake_run_cmd(cmd):
        if "nvidia-smi" in cmd:
            return smi_csv
        return ""  # PowerShell/wmic no aportan nada

    monkeypatch.setattr(hw, "_run_cmd", fake_run_cmd)
    gpus = hw._detect_gpus_windows()
    assert len(gpus) == 2
    g0 = gpus[0]
    assert g0["driver"] == "560.94"
    assert g0["memory"] == "24564 MiB"
    assert g0["memory_used"] == "3144 MiB"
    assert g0["core_clock"] == "2520 MHz"
    assert g0["temperature"] == "68 °C"


def test_parse_video_controller_csv_includes_driver(monkeypatch):
    """The Win32_VideoController fallback keeps the driver version."""
    import src.hardware.detector as hw

    csv_out = (
        '"Name","AdapterRAM","VideoProcessor","AdapterCompatibility",'
        '"DriverVersion","Status"\r\n'
        '"AMD Radeon RX 6700 XT","12884901888","","Advanced Micro Devices, Inc.",'
        '"31.0.14033.1010","OK"\r\n'
    )
    gpus = hw._parse_video_controller_csv(csv_out)
    assert len(gpus) == 1
    assert gpus[0]["driver"] == "31.0.14033.1010"
    assert gpus[0]["vendor"] == "AMD"
    assert gpus[0]["memory"] == "12.0 GB"


@pytest.fixture()
def isolated_user_models(tmp_path, monkeypatch, isolated_engines_state):
    """Empuja user_models.json a temp y arranca limpio (con engines_state
    ya aislado por ``isolated_engines_state``)."""
    import src.ai.user_models as um
    monkeypatch.setattr(um, "get_session_dir", lambda: str(tmp_path))
    yield tmp_path
    um._LOAD_CACHE.clear()


# ==--------------------------------------------------------------------==
# 2bis. Registro dinámico de motores de usuario (Hub)
# ==--------------------------------------------------------------------==

def test_user_models_register_get_unregister(isolated_user_models):
    from src.ai.user_models import (all_models, catalog_entries, get_model,
                                    register, unregister)

    eid = "user-upscale-foo-bar-aaaa0000"
    assert register(eid, display_name="My SR", category="Upscale",
                    hf_repo_id="foo/bar", weight_file="model.pth") is True
    rec = get_model(eid)
    assert rec["display_name"] == "My SR"
    assert rec["category"] == "Upscale"
    assert rec["weight_file"] == "model.pth"

    entries = catalog_entries()
    mine = [e for e in entries if e["engine_id"] == eid][0]
    # Misma forma que el catálogo (el pipeline/nodos la consumen igual).
    for k in ("engine_id", "display_name", "category", "hf_repo_id", "hf_url",
              "sample_shape", "license", "is_implemented", "sentinel"):
        assert k in mine, k
    assert mine["hf_repo_id"] == "foo/bar"
    assert mine["sentinel"] == "model.pth"
    assert mine["is_user"] is True

    assert unregister(eid) is True
    assert get_model(eid) is None
    assert eid not in all_models()


def test_user_models_register_rejects_bad_slug(isolated_user_models):
    from src.ai.user_models import register, all_models
    assert register("../evil", display_name="x") is False
    assert register("User MixedCase", display_name="x") is False
    assert all_models() == {}


def test_registry_fusion_list_installed_and_get_entry(isolated_user_models):
    from src.ai.user_models import register
    from src.ai.engines_state import mark_installed, get_state
    from src.ai.registry import get_entry, get_engine_meta, get_engine_dir, list_installed
    from src.ai.hf_search import user_engine_id
    import os

    eid = user_engine_id("Upscale", "bdsr/4x-UltraSharp")
    register(eid, display_name="4x UltraSharp", category="Upscale",
             hf_repo_id="bdsr/4x-UltraSharp", weight_file="model.pth", scale=4)

    # Sin pesos en disco aún no cuenta como instalado.
    assert eid not in {e["engine_id"] for e in list_installed("Upscale")}

    target = get_engine_dir(eid)
    os.makedirs(os.path.join(target, "weights"), exist_ok=True)
    open(os.path.join(target, "weights", "4x-UltraSharp.pth"), "w").close()
    mark_installed(eid, install_path=target)

    ids = {e["engine_id"] for e in list_installed("Upscale")}
    assert eid in ids

    entry = get_engine_meta(eid)
    assert entry is not None
    assert entry["display_name"] == "4x UltraSharp"
    assert get_state(eid)["installed"] is True

    # Motores de usuario siguen el layout de catálogo: models/<categoria>/...
    assert target == os.path.join(
        str(isolated_user_models), "models", "upscale",
        "bdsr-4x-ultrasharp-" + eid.rsplit("-", 1)[1])


def test_registry_fusion_get_entry_and_scale(isolated_user_models):
    from src.ai.user_models import register
    from src.ai.registry import get_entry, get_scale_factor

    eid = "user-upscale-foo-mymodel-12345678"
    register(eid, display_name="MyModel", category="Upscale",
             hf_repo_id="foo/mymodel", weight_file="m.pth", scale=3)

    entry = get_entry(eid)
    assert entry is not None
    assert entry["is_user"] is True
    assert get_scale_factor(eid) == 3  # leido de sample_shape dinámico

    # Id de catálogo sigue resolviendo igual.
    assert get_entry("upscale-omnisr-x4") is not None
    # Id desconocido -> None.
    assert get_entry("no-existe-123") is None


def test_registry_categories_include_user_categories(isolated_user_models):
    from src.ai.user_models import register
    from src.ai.registry import categories

    before = categories()
    register("user-x9-colorize-myc-99999999", display_name="My Color",
             category="Colorize", hf_repo_id="x9/myc")
    after = categories()
    assert "Colorize" in after
    # categorías ya presentes no se duplican
    assert after.count("Colorize") == 1


def test_all_states_includes_user_engines(isolated_user_models):
    from src.ai.user_models import register
    from src.ai.engines_state import all_states

    register("user-x9-cat-one-00000000", display_name="One", category="Upscale",
             hf_repo_id="x/one")
    states = all_states()
    assert "user-x9-cat-one-00000000" in states


# ==--------------------------------------------------------------------==
# 9. Hub search (HuggingFace libre) + user engine ids
# ==--------------------------------------------------------------------==


class _FakeRepoFile:
    """``RepoFile``-like object with only ``rfilename``."""

    def __init__(self, rfilename):
        self.rfilename = rfilename


class _FakeModelInfo:
    """``ModelInfo``-like object with the attributes ``search_hub_models`` reads."""

    def __init__(self, model_id, files, tags=(), pipeline_tag=None,
                 library_name=None, downloads=0, likes=0, license=None):
        self.modelId = model_id
        self.id = model_id
        self.downloads = downloads
        self.likes = likes
        self.tags = list(tags)
        self.pipeline_tag = pipeline_tag
        self.library_name = library_name
        self.lastModified = None
        self.siblings = [_FakeRepoFile(f) for f in files]
        self.cardData = type("Card", (), {"license": license})()


class _FakeHfApi:
    """Sustituye ``HfApi`` y registra las llamadas (kwargs) para asserts
    de cache y paginación cliente (``list_models`` 1.x ya no acepta
    ``offset``: la paginación se hace con islice sobre el iterable)."""

    calls: list[dict] = []

    def __init__(self, token=None):
        self.token = token

    def list_models(self, **kwargs):
        _FakeHfApi.calls.append(dict(kwargs))
        # ``islice`` pagina sobre este iterable: no hay offset server-side.
        return [
            _FakeModelInfo(
                "owner/4x-UltraSharp",
                files=["model.pth", "README.md"],
                tags=["image-to-image", "super-resolution"],
                pipeline_tag="image-to-image",
                downloads=42_000,
                license="mit",
            ),
            _FakeModelInfo(
                "owner/rife-v4.6",
                files=["flownet.pkl", "README.md"],
                pipeline_tag=None,
                downloads=8_000,
            ),
            _FakeModelInfo(
                "owner/film-vfi",
                files=["flownet.pth", "README.md"],
                pipeline_tag=None,
                downloads=15_000,
            ),
            _FakeModelInfo(
                "bigcorp/llm-7b",
                files=["pytorch_model.bin", "config.json"],
                tags=["text-generation"],
                pipeline_tag="text-generation",
                downloads=999_999,
            ),
            _FakeModelInfo(
                "mystery/random-weights",
                files=["model.safetensors", "README.md"],
                tags=["image-to-image"],
                pipeline_tag="image-to-image",
                downloads=500,
            ),
        ]


def test_user_engine_id_is_slug_safe_and_deterministic():
    from src.ai.hf_search import user_engine_id
    from src.ai.registry import _SLUG_RE

    eid = user_engine_id("Upscale", "ai-forever/Real-ESRGAN")
    assert _SLUG_RE.match(eid), eid
    assert eid.startswith("user-upscale-")
    assert eid == user_engine_id("Upscale", "ai-forever/Real-ESRGAN")
    assert eid != user_engine_id("Upscale", "ai-forever/Real-ESRGan")
    assert user_engine_id("FaceRestore", "sczhou/CodeFormer").startswith("user-facerestore-")
    assert user_engine_id("Upscale", "") != ""


def test_detect_category_heuristics():
    from src.ai.hf_search import _detect_category

    cases = [
        (("bdsr/4x-UltraSharp", ["image-to-image"], "image-to-image"), "Upscale"),
        (("owner/Real-ESRGAN", [], None), "Upscale"),
        (("owner/rife-v4.6", [], None), "Interpolate"),
        (("owner/frame-interpolation", [], None), "Interpolate"),
        (("owner/DeOldify", [], None), "Colorize"),
        (("owner/NAFNet-denoise", [], None), "Denoise"),
        (("owner/GFPGAN", [], None), "FaceRestore"),
        (("owner/codeformer", [], None), "FaceRestore"),
        (("owner/llm-7b", ["text-generation"], "text-generation"), None),
        (("owner/random", [], None), None),
    ]
    for (repo, tags, pipe), expected in cases:
        assert _detect_category(repo, tags, pipe) == expected, repo


def test_search_hub_models_filters_and_shapes(monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    from src.ai.hf_search import search_hub_models

    results = search_hub_models(category="Upscale", use_cache=False)
    ids = {r["hf_repo_id"] for r in results}

    # El repo SR de verdad entra con confianza de categoría.
    assert "owner/4x-UltraSharp" in ids
    sr = next(r for r in results if r["hf_repo_id"] == "owner/4x-UltraSharp")
    assert sr["category_confidence"] is True
    assert sr["is_user"] is True
    assert sr["engine_id"].startswith("user-upscale-")
    assert sr["downloads"] == 42_000
    assert sr["license"] == "mit"
    assert "model.pth" in sr["files"]

    # RIFE es de otra categoría -> excluido del tab Upscale.
    assert "owner/rife-v4.6" not in ids
    # LLM con .bin pero sin señal de imagen -> excluido.
    assert "bigcorp/llm-7b" not in ids
    # Repo sin categoría pero con señal de imagen -> entra "incierto".
    assert "mystery/random-weights" in ids
    unc = next(r for r in results if r["hf_repo_id"] == "mystery/random-weights")
    assert unc["category_confidence"] is False
    assert unc["detected_category"] is None


def test_search_hub_models_engine_compat_filter(monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    from src.ai.hf_search import search_hub_models

    # engine_id=torch: solo repos con pesos torch (pth/safetensors/bin).
    results = search_hub_models(category="Upscale", engine_id="torch", use_cache=False)
    ids = {r["hf_repo_id"] for r in results}
    assert "owner/4x-UltraSharp" in ids  # .pth -> torch OK
    assert "mystery/random-weights" in ids  # .safetensors -> torch OK


def test_search_hub_models_cache_no_offset_kwargs(monkeypatch):
    """La API nueva de huggingface_hub (1.x) no acepta ``offset``: la
    llamada a ``list_models`` no debe llevar ni ``offset`` ni ``limit``
    server-side (la paginación se hace en cliente con islice)."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    _FakeHfApi.calls.clear()
    from src.ai.hf_search import search_hub_models, clear_cache

    r1 = search_hub_models(category="Upscale", use_cache=True)
    r2 = search_hub_models(category="Upscale", use_cache=True)
    assert len(_FakeHfApi.calls) == 1  # segunda llamada servida del cache

    search_hub_models(category="Upscale", page=1, limit=30, use_cache=True)
    call = _FakeHfApi.calls[-1]
    assert call.get("sort") == "downloads"
    assert "-downloads" not in call.get("sort", "")
    assert "offset" not in call
    assert "limit" not in call

    clear_cache()
    search_hub_models(category="Upscale", use_cache=True)
    assert len(_FakeHfApi.calls) == 3  # tras clear_cache vuelve a llamar


def test_search_hub_models_paginates_client_side(monkeypatch):
    """Página 0/1/2 (limit=2) devuelven slices disjuntos del iterable de
    ``list_models``: la paginación funciona sin el parámetro ``offset``."""
    class _PagedApi:
        calls: list[dict] = []

        def __init__(self, token=None):
            pass

        def list_models(self, **kwargs):
            _PagedApi.calls.append(dict(kwargs))
            return [
                _FakeModelInfo("owner/4x-UltraSharp", files=["model.pth"],
                               tags=["image-to-image"], pipeline_tag="image-to-image"),
                _FakeModelInfo("owner/rife-v4.6", files=["flownet.pkl", "model.pth"]),
                _FakeModelInfo("owner/film-vfi", files=["flownet.pth"]),
                _FakeModelInfo("bigcorp/llm-7b", files=["pytorch_model.bin"],
                               tags=["text-generation"], pipeline_tag="text-generation"),
                _FakeModelInfo("mystery/random-weights", files=["model.safetensors"],
                               tags=["image-to-image"], pipeline_tag="image-to-image"),
            ]

    monkeypatch.setattr("huggingface_hub.HfApi", _PagedApi)
    _PagedApi.calls.clear()
    from src.ai.hf_search import search_hub_models

    # category=None: entran todos los repos con categoría detectada o señal.
    p0 = {r["hf_repo_id"] for r in search_hub_models(category=None, page=0, limit=2, use_cache=False)}
    p1 = {r["hf_repo_id"] for r in search_hub_models(category=None, page=1, limit=2, use_cache=False)}
    p2 = {r["hf_repo_id"] for r in search_hub_models(category=None, page=2, limit=2, use_cache=False)}

    assert p0 == {"owner/4x-UltraSharp", "owner/rife-v4.6"}
    assert p1 == {"owner/film-vfi"}
    assert p2 == {"mystery/random-weights"}
    # Sin solapes ni saltos entre páginas de distinto tamaño.
    assert p0.isdisjoint(p1) and p1.isdisjoint(p2)
    # El filtrado de categoría sigue aplicándose tras el slice.
    assert "bigcorp/llm-7b" not in p0 | p1 | p2


def test_search_hub_models_category_none_any_category(monkeypatch):
    """category=None (pestaña "All categories") no descarta los repos con
    categoría detectada; solo quedan fuera los sin categoría Y sin señal."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    from src.ai.hf_search import search_hub_models

    results = search_hub_models(category=None, use_cache=False)
    ids = {r["hf_repo_id"] for r in results}
    assert "owner/4x-UltraSharp" in ids      # Upscale detectado
    assert "owner/film-vfi" in ids           # Interpolate detectado (sin filtro)
    assert "mystery/random-weights" in ids   # incierto con señal de imagen
    assert "bigcorp/llm-7b" not in ids       # LLM sin señal -> fuera
    rife = next(r for r in results if r["hf_repo_id"] == "owner/film-vfi")
    assert rife["category"] == "Interpolate"


def test_detect_model_arch_and_scale():
    from src.ai.architectures import detect_model_arch_and_scale

    class _S:
        def __init__(self, *shape):
            self.shape = shape

    # RRDBNet x4 (Real-ESRGAN): dos conv_up -> scale 4
    rrdb = {
        "conv_first.weight": _S(64, 3, 3, 3),
        "body.0.rdb1.conv1.weight": _S(32, 64, 3, 3),
        "conv_up1.weight": _S(256, 64, 3, 3),
        "conv_up2.weight": _S(256, 64, 3, 3),
        "conv_last.weight": _S(3, 64, 3, 3),
    }
    assert detect_model_arch_and_scale(rrdb) == ("rrdb", 4)

    # SRVGGNetCompact x4 (AnimeVideo): última conv 3*4*4=48 canales
    srvgg = {
        "body.0.weight": _S(64, 3, 3, 3),
        "body.1.weight": _S(64,),
        "body.32.weight": _S(48, 64, 3, 3),
        "body.33.weight": _S(64,),
    }
    assert detect_model_arch_and_scale(srvgg) == ("srvgg", 4)

    # OmniSR x4
    omnisr = {
        "input.weight": _S(64, 3, 3, 3),
        "output.weight": _S(64, 64, 3, 3),
        "residual_layer.0.residual_layer.0.layer.0.fn.0.weight": _S(64, 64, 1, 1),
        "up.0.weight": _S(48, 64, 3, 3),
    }
    assert detect_model_arch_and_scale(omnisr) == ("omnisr", 4)

    # CUGAN x4
    cugan = {
        "unet1.conv1.conv.0.weight": _S(32, 3, 3, 3),
        "conv_final.weight": _S(48, 32, 3, 3),
    }
    assert detect_model_arch_and_scale(cugan) == ("cugan", 4)

    # Desconocidos / inválidos
    assert detect_model_arch_and_scale({"foo": 1}) is None
    assert detect_model_arch_and_scale(None) is None

    # Checkpoints envueltos (params_ema) se desempaquetan
    wrapped = {"params_ema": dict(rrdb)}
    assert detect_model_arch_and_scale(wrapped) == ("rrdb", 4)


def test_detect_arch_and_scale_file_missing_and_unknown():
    from src.ai.architectures import detect_arch_and_scale

    assert detect_arch_and_scale("") == (None, None)
    assert detect_arch_and_scale("Z:/no/such/file.pth") == (None, None)

    # Archivo real pero no es un state_dict -> (None, None) sin excepción.
    p = os.path.join(tempfile.gettempdir(), "draggy_dummy_weights.bin")
    with open(p, "wb") as f:
        f.write(b"not a weight file at all")
    try:
        assert detect_arch_and_scale(p) == (None, None)
    finally:
        os.remove(p)


# ==--------------------------------------------------------------------==
# 7. AI chain: split, scale, end-to-end ordering
# ==--------------------------------------------------------------------==


class _FakeEngine:
    """Minimal AIInferenceEngine stand-in for pure pipeline tests."""

    def __init__(self, engine_id, scale=1, mode="upscale"):
        self.engine_id = engine_id
        self._scale = scale
        self._mode = mode
        self._failed = False
        self.order_log = []

    def get_scale_factor(self):
        return self._scale

    def process_frame(self, frame):
        self.order_log.append(self.engine_id)
        if self._failed:
            raise RuntimeError(f"{self.engine_id} boom")
        if self._scale > 1:
            import numpy as _np
            return _np.repeat(_np.repeat(frame, self._scale, axis=0), self._scale, axis=1)
        return frame

    def interpolate_frame(self, prev, cur):
        self.order_log.append(f"{self.engine_id}:interp")
        return cur


def test_split_ai_chain_no_interp_preserves_order():
    from src.processing.pipeline import _split_ai_chain

    ups = _FakeEngine("upscale", 2)
    den = _FakeEngine("denoise", 1)
    col = _FakeEngine("colorize", 1)
    antes, despues, interp = _split_ai_chain([ups, den, col])
    assert interp is None
    assert [e.engine_id for e in antes] == ["upscale", "denoise", "colorize"]
    assert despues == []


def test_split_ai_chain_interp_in_middle():
    from src.processing.pipeline import _split_ai_chain

    ups = _FakeEngine("upscale", 2)
    interp = _FakeEngine("interp", 1, mode="interpolate")
    den = _FakeEngine("denoise", 1)
    antes, despues, interp_step = _split_ai_chain([ups, interp, den])
    assert interp_step is interp
    assert [e.engine_id for e in antes] == ["upscale"]
    assert [e.engine_id for e in despues] == ["denoise"]


def test_split_ai_chain_only_first_interp_honored():
    from src.processing.pipeline import _split_ai_chain

    i1 = _FakeEngine("interp1", 1, mode="interpolate")
    i2 = _FakeEngine("interp2", 1, mode="interpolate")
    antes, despues, interp_step = _split_ai_chain([i1, i2])
    assert interp_step is i1
    assert [e.engine_id for e in despues] == ["interp2"]


def test_compute_total_scale_chain_product():
    from src.processing.pipeline import _compute_total_scale

    assert _compute_total_scale([_FakeEngine("a", 2), _FakeEngine("b", 4)]) == 8
    assert _compute_total_scale([_FakeEngine("a", 2), _FakeEngine("b", 1)]) == 2
    # Interp-only engines do not change resolution -> excluded.
    assert _compute_total_scale([
        _FakeEngine("a", 2),
        _FakeEngine("i", 4, mode="interpolate"),
    ]) == 2
    assert _compute_total_scale([_FakeEngine("den", 1)]) == 1
    assert _compute_total_scale([]) == 1


def test_generate_preview_runs_the_real_ai_chain(nvenc_clip):
    """AI preview must execute the selected chain, not only FFmpeg encode."""
    from src.processing.pipeline import generate_preview

    ffmpeg, src, tmp = nvenc_clip
    engine = _FakeEngine("preview-engine", 1)
    original, processed = generate_preview(
        ffmpeg,
        src,
        {
            "audio": "noaudio",
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "800",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=[engine],
    )
    assert original.startswith(b"\xff\xd8")
    assert processed.startswith(b"\xff\xd8")
    assert engine.order_log, "the preview never called the AI engine"


def test_compress_video_with_ai_chain_order_and_scale(nvenc_clip):
    """Chain of two engines must run in order and output the total scale."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain4.mp4")
    engines = [_FakeEngine("first-2x", 2), _FakeEngine("second-2x", 2)]
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "noaudio",
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=engines,
    )
    assert ok is True
    assert os.path.exists(dst) and os.path.getsize(dst) > 0
    # Both engines ran, in chain order, and the first frame went through
    # the whole chain before the second one.
    assert engines[0].order_log
    assert engines[1].order_log
    assert engines[0].order_log[0] == "first-2x"
    assert engines[1].order_log[0] == "second-2x"
    # Interleaved: first-2x, second-2x, first-2x, second-2x, ...
    assert engines[0].order_log[:2] == ["first-2x", "first-2x"]
    assert engines[1].order_log[:2] == ["second-2x", "second-2x"]


def test_compress_video_with_ai_chain_engine_failure_degrades(nvenc_clip):
    """A failing engine is disabled mid-render; the chain still completes."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_fail.mp4")
    bad = _FakeEngine("boom-engine", 1)
    bad._failed = True  # fail on every frame from frame 0
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "noaudio",
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=[_FakeEngine("ok-engine", 1), bad],
    )
    assert ok is True
    assert os.path.exists(dst) and os.path.getsize(dst) > 0
    assert bad._failed is True


def test_compress_video_with_ai_interp_writes_double_frames(nvenc_clip):
    """Interp step doubles the output frame count (interp + current)."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_interp.mp4")
    interp_engine = _FakeEngine("interp-step", 1, mode="interpolate")
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "noaudio",
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=[interp_engine],
    )
    assert ok is True
    assert os.path.exists(dst) and os.path.getsize(dst) > 0
    # First frame has no interp; every subsequent one emits interp+current.
    assert "interp-step:interp" in interp_engine.order_log


def test_compress_video_with_ai_native_uses_model_size(nvenc_clip):
    """'Model native' (settings ai_native) must size the encoder to the
    REAL output of the chain — a 2x-scaling fake chain fed a 640x360 clip
    therefore encodes at 1280x720, NOT at the computed scale reduction."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_native.mp4")
    model = _FakeEngine("native-model", 2)
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "noaudio",
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
            # The ui picked 'Model native': the encoder must use the model's
            # own output resolution instead of ai_scale/chain product.
            "ai_native": True,
            "ai_scale": 1,
        },
        inference_engines=[model],
    )
    assert ok is True
    assert os.path.exists(dst) and os.path.getsize(dst) > 0

    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    assert os.path.exists(ffprobe)
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "default=noprint_wrappers=1", dst],
        capture_output=True, text=True, timeout=30,
    ).stdout
    fields = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    # 640x360 -> _FakeEngine(scale=2) -> 1280x720 (the model's real output).
    assert fields.get("width") == "1280", fields
    assert fields.get("height") == "720", fields


def test_compress_video_with_ai_native_probe_fallback(nvenc_clip):
    """If the native probe fails (engine error on the probe frame), the
    render falls back to the computed scaled resolution instead of dying."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_native_fallback.mp4")

    class _ExplodingProbe(_FakeEngine):
        def process_frame(self, frame):
            # First call is the probe; subsequent calls are real frames.
            if not getattr(self, "_probe_done", False):
                self._probe_done = True
                raise RuntimeError("probe boom")
            return super().process_frame(frame)

    engine = _ExplodingProbe("native-fallback", 2)
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "noaudio",
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
            "ai_native": True,
            "ai_scale": 2,
        },
        inference_engines=[engine],
    )
    assert ok is True
    assert os.path.exists(dst) and os.path.getsize(dst) > 0
    # Fallback path still wrote frames (chain completed).
    assert engine.order_log


def test_audio_mux_args_copy_passthrough_without_boost():
    """-c:a copy when boost is off (no re-encode, fastest path)."""
    from src.processing.pipeline import _audio_mux_args

    assert _audio_mux_args("copy", None, False) == ["-c:a", "copy"]


def test_audio_mux_args_boost_uses_two_pass_loudnorm():
    """Boost + measured loudness → re-encode AAC with linear loudnorm."""
    from src.processing.pipeline import _audio_mux_args

    meas = {
        "input_i": "-18.5", "input_tp": "-1.1", "input_lra": "9.2",
        "input_thresh": "-28.0", "target_offset": "0.4",
    }
    args = _audio_mux_args("copy", meas, True)
    assert args[0:4] == ["-c:a", "aac", "-b:a", "192k"]
    af = args[args.index("-af") + 1]
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in af
    assert "measured_I=-18.5000" in af
    assert "linear=true" in af


def test_audio_mux_args_boost_fallback_single_pass():
    """Boost without measured levels → single-pass loudnorm defaults."""
    from src.processing.pipeline import _audio_mux_args

    args = _audio_mux_args("copy", None, True)
    assert args[0:4] == ["-c:a", "aac", "-b:a", "192k"]
    af = args[args.index("-af") + 1]
    assert af == "loudnorm=I=-16:TP=-1.5:LRA=11"


def test_audio_mux_args_mp3_keeps_libmp3lame():
    """boost + audio=mp3 keeps the LAME encoder."""
    from src.processing.pipeline import _audio_mux_args

    args = _audio_mux_args("mp3", None, True)
    assert args[0] == "-c:a" and args[1] == "libmp3lame"


def test_audio_mux_args_webm_uses_libopus():
    """WebM container forces libopus (AAC/MP3 are invalid in WebM)."""
    from src.processing.pipeline import _audio_mux_args

    args = _audio_mux_args("copy", None, True, container_format="webm")
    assert args[0:2] == ["-c:a", "libopus"]
    assert "aac" not in args[:4]


def test_compress_video_with_ai_audio_boost_keeps_audio(nvenc_clip):
    """Audio Boost in the AI chain must re-encode audio and keep a track."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_boost.mp4")
    status_lines = []
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "copy",
            "audio_boost": True,
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=[_FakeEngine("upscale-2x", 2)],
        status_callback=status_lines.append,
    )
    assert ok is True
    assert os.path.exists(dst) and os.path.getsize(dst) > 0
    # Audio Boost must run the loudness measurement pass on the source.
    assert any("Analyzing audio levels" in line for line in status_lines), status_lines

    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    assert os.path.exists(ffprobe), "ffprobe not shipped with full build"
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1", dst],
        capture_output=True, text=True, timeout=30,
    ).stdout
    # Boost re-encodes to AAC, not a raw copy of the source track.
    assert "aac" in out, f"expected re-encoded AAC audio, got: {out!r}"
    # No leftover parallel-audio temp files must survive the mux cleanup.
    leftovers = [f for f in os.listdir(tmp) if f.startswith(".drag-audio-")]
    assert leftovers == [], leftovers


def test_ai_chain_emits_phase_order(nvenc_clip):
    """compress_video_with_ai must emit probe -> engine -> audio -> mux."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_phases.mp4")
    events = []

    def cb(ev):
        events.append(ev)

    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "copy",
            "audio_boost": True,
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=[_FakeEngine("upscale-2x", 2)],
        phase_cb=cb,
    )
    assert ok is True
    starts = [e for e in events if e["ev"] == "start"]
    ids = [e["id"] for e in starts]
    assert ids[0] == "probe", ids
    assert any(i.startswith("engine-") for i in ids), ids
    assert "audio" in ids, ids
    assert "mux" in ids, ids
    assert ids.index("probe") < ids.index("audio") < ids.index("mux")
    # Every phase that started also ended (or failed).
    for ph in ids:
        ended = [e["id"] for e in events if e["ev"] in ("end", "failed")]
        assert ph in ended, (ph, events)


def test_ai_chain_mux_phase_completes_when_no_audio(nvenc_clip):
    """Chain without audio still emits mux completion (no crash)."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_noaudio.mp4")
    events = []
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "noaudio",
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=[_FakeEngine("upscale-2x", 2)],
        phase_cb=events.append,
    )
    assert ok is True
    ended_ids = [e["id"] for e in events if e["ev"] in ("end", "failed")]
    # Finalization phase always runs (encoder finalize); audio phase does not.
    assert "mux" in ended_ids
    assert "audio" not in ended_ids
    assert any(i.startswith("engine-") for i in ended_ids)
    assert "probe" in ended_ids
    assert not [f for f in os.listdir(tmp) if f.startswith(".drag-audio-")]


def test_parallel_audio_no_boost_skips_thread(nvenc_clip):
    """copy + no boost keeps the fast passthrough (no parallel encode)."""
    from src.processing.pipeline import compress_video_with_ai

    ffmpeg, src, tmp = nvenc_clip
    dst = os.path.join(tmp, "chain_copy.mp4")
    events = []
    ok = compress_video_with_ai(
        ffmpeg,
        src,
        dst,
        {
            "audio": "copy",
            "audio_boost": False,
            "format": "mp4",
            "codec": "h264_nvenc",
            "preset": "p5",
            "bitrate": "1500",
            "pixel_format": "yuv420p",
            "resolution": "original",
        },
        inference_engines=[_FakeEngine("upscale-2x", 2)],
        phase_cb=events.append,
    )
    assert ok is True
    assert any(e["id"] == "mux" and e["ev"] == "end" for e in events)
    leftovers = [f for f in os.listdir(tmp) if f.startswith(".drag-audio-")]
    assert leftovers == []


def test_task_history_roundtrip_and_failsafe(tmp_path, monkeypatch):
    """start/update/finish/list round-trip; corrupted file degrades to []."""
    import src.task_history as th

    monkeypatch.setenv("DEBUG_TASK_HISTORY_DIR", str(tmp_path))
    # Point the module at the tmp session dir.
    monkeypatch.setattr(th, "_path", lambda: str(tmp_path / "tasks.json"))
    assert th.start_task("t1", "video.mp4", input_path="C:\\in.mp4")
    assert th.update_phase("t1", "engine-1", "Upscale A", "running", 40)
    assert th.update_phase("t1", "mux", "Mux", "running", 10)
    assert th.finish_task("t1", "success")

    tasks = th.list_tasks()
    assert len(tasks) == 1
    t1 = tasks[0]
    assert t1["status"] == "success"
    phase_ids = {p["phase_id"] for p in t1["phases"]}
    assert phase_ids == {"engine-1", "mux"}
    assert th.get_task("t1")["name"] == "video.mp4"

    # Corrupt the file: listing must not raise.
    (tmp_path / "tasks.json").write_text("{not json", encoding="utf-8")
    assert th.list_tasks() == []
    assert th.get_task("nope") is None


def test_task_history_clear(tmp_path, monkeypatch):
    import src.task_history as th

    monkeypatch.setattr(th, "_path", lambda: str(tmp_path / "tasks.json"))
    th.start_task("x", "a.mp4")
    assert th.clear() is True
    assert th.list_tasks() == []


def test_tasks_panel_widget_smoke(qapp):
    """TasksPanel renders tasks/phases and toggles without crashing."""
    from src.ui.tasks_panel import TasksPanel

    panel = TasksPanel()
    panel.get_or_create("job-1", "movie.mp4")
    card = panel._cards["job-1"]
    card.set_running()
    card.update_phase_evt({"id": "engine-1", "label": "Upscale", "ev": "start", "pct": 0})
    card.update_phase_evt({"id": "engine-1", "ev": "progress", "pct": 50})
    card.update_phase_evt({"id": "mux", "label": "Mux", "ev": "start", "pct": 0})
    card.update_phase_evt({"id": "mux", "ev": "end", "pct": 100})
    panel.mark_finished("job-1", "success")
    assert card._status == "completed"
    assert "engine-1" in card._phases and "mux" in card._phases

    # History rebuild path must not crash either.
    panel.rebuild([
        {"task_id": "old-1", "name": "old.mp4", "status": "success",
         "phases": [{"phase_id": "engine-1", "label": "Upscale",
                     "status": "done", "pct": 100}]}
    ])
    assert "old-1" in panel._cards
    assert panel._cards["old-1"]._status == "completed"

    # Log tab: append + clear without crashing.
    panel.append_log("first line")
    panel.append_log("second line")
    assert panel.log_view.toPlainText().count("\n") >= 1
    panel._on_clear_log()
    assert panel.log_view.toPlainText() == ""


def test_tasks_panel_clear_all_removes_every_card(qapp):
    """clear_all() must wipe every visible card (fixes Clear history)."""
    from src.ui.tasks_panel import TasksPanel

    panel = TasksPanel()
    for i in range(3):
        panel.get_or_create(f"job-{i}", f"file-{i}.mp4")
    assert len(panel._cards) == 3
    assert len(panel._order) == 3

    panel.clear_all()
    assert panel._cards == {}
    assert panel._order == []
    assert panel._list_layout.count() == 1  # solo queda el stretch


def test_tasks_panel_rebuild_replaces_not_duplicates(qapp):
    """rebuild() substituye la lista: no duplica tarjetas antiguas."""
    from src.ui.tasks_panel import TasksPanel

    panel = TasksPanel()
    panel.rebuild([
        {"task_id": "a", "name": "a.mp4", "status": "success", "phases": []}
    ])
    assert "a" in panel._cards
    panel.rebuild([
        {"task_id": "b", "name": "b.mp4", "status": "running", "phases": []}
    ])
    assert "a" not in panel._cards
    assert "b" in panel._cards


# ---------------------------------------------------------------------------
# 13. Task queue: persistent render queue (video2x-style batching)
# ---------------------------------------------------------------------------


def test_task_queue_roundtrip_and_failsafe(tmp_path, monkeypatch):
    """enqueue/list/set_status/first_pending round-trip; corrupt file -> []."""
    import src.task_queue as tq

    monkeypatch.setattr(tq, "_path", lambda: str(tmp_path / "queue.json"))
    qid1 = tq.enqueue(
        name="a.mp4", input_path="C:\\in\\a.mp4",
        output_path="C:\\out\\a_comp.mp4",
        settings={"codec": "h264_nvenc", "bitrate": "1500"},
        ai_engines={"Upscale": "realesrgan-x4"},
        pipeline_order=["Upscale"],
    )
    assert qid1
    qid2 = tq.enqueue(
        name="b.mp4", input_path="C:\\in\\b.mp4",
        output_path="C:\\out\\b_comp.mp4",
        settings={"codec": "libx264"},
    )
    assert qid2 != qid1

    items = tq.list_items()
    assert [i["queue_id"] for i in items] == [qid1, qid2]  # orden de encolado
    assert tq.first_pending()["queue_id"] == qid1

    assert tq.set_status(qid1, "running")
    assert tq.first_pending()["queue_id"] == qid2
    assert tq.set_status(qid1, "done")
    assert tq.first_pending()["queue_id"] == qid2

    got = tq.get_item(qid1)
    assert got["status"] == "done"
    assert got["settings"]["codec"] == "h264_nvenc"
    assert got["ai_engines"] == {"Upscale": "realesrgan-x4"}
    assert got["pipeline_order"] == ["Upscale"]

    assert tq.remove(qid2)
    assert tq.get_item(qid2) is None
    assert tq.count() == 1

    # Failsafe: archivo corrupto -> listas vacías sin excepción.
    (tmp_path / "queue.json").write_text("{not json", encoding="utf-8")
    assert tq.list_items() == []
    assert tq.first_pending() is None
    assert tq.count() == 0


def test_task_queue_items_are_independent_snapshots(tmp_path, monkeypatch):
    """Cada item conserva SU config, aunque se encole otro después."""
    import src.task_queue as tq

    monkeypatch.setattr(tq, "_path", lambda: str(tmp_path / "queue.json"))
    qid1 = tq.enqueue(
        name="a.mp4", input_path="C:\\in\\a.mp4",
        output_path="C:\\out\\a.mp4",
        settings={"codec": "h264_nvenc", "audio_boost": True},
        ai_engines={"Upscale": "eng-x2"},
    )
    qid2 = tq.enqueue(
        name="b.mp4", input_path="C:\\in\\b.mp4",
        output_path="C:\\out\\b.mp4",
        settings={"codec": "libx264", "audio_boost": False},
        ai_engines={"Upscale": "eng-x4"},
    )
    assert tq.get_item(qid1)["settings"]["codec"] == "h264_nvenc"
    assert tq.get_item(qid2)["settings"]["codec"] == "libx264"
    assert tq.get_item(qid1)["ai_engines"] == {"Upscale": "eng-x2"}
    assert tq.get_item(qid2)["ai_engines"] == {"Upscale": "eng-x4"}
    # Duplicados permitidos (mismo input/output) -> ids distintos.
    qid3 = tq.enqueue(
        name="a.mp4", input_path="C:\\in\\a.mp4",
        output_path="C:\\out\\a.mp4", settings={},
    )
    assert qid3 != qid1
    assert tq.count() == 3


def test_add_to_queue_batches_multiple_videos_same_config(tmp_path, monkeypatch):
    """_add_to_queue con varios videos seleccionados encola TODOS con la
    misma config (snapshot único de settings + cadena IA + pipeline)."""
    import src.task_queue as tq
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(tq, "_path", lambda: str(tmp_path / "queue.json"))

    v1 = tmp_path / "a.mp4"
    v2 = tmp_path / "b.mkv"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    v1.write_bytes(b"x")
    v2.write_bytes(b"x")

    mw = MainWindow.__new__(MainWindow)
    mw.input_paths = [str(v1), str(v2)]
    mw.input_path = str(v1)
    mw.output_dir = str(out_dir)
    mw.worker = None
    mw._maybe_ffmpeg_health_check = lambda: True
    mw._remember_runtime_prefs = lambda s: None
    messages = []
    mw._on_status_message = messages.append

    class _Panel:
        def rebuild_queue(self, items):
            self.items = items

    mw.tasks_panel = _Panel()

    class _SettingsPanel:
        def get_settings(self):
            return {
                "format": "mp4",
                "output_name": "",
                "codec": "h264_nvenc",
                "resolution": "original",
            }

    mw.settings_panel = _SettingsPanel()

    class _Workflow:
        def get_pipeline(self):
            return ["Import", "Upscale", "Compress"]

        def get_selected_engines(self):
            return {"Upscale": {"engine_id": "upscale-realesrgan-x4plus",
                                "model_path": None, "scale": 4}}

    mw.workflow = _Workflow()

    mw._add_to_queue()

    items = tq.list_items()
    assert len(items) == 2
    # Orden de selección preservado.
    assert items[0]["input_path"] == str(v1)
    assert items[1]["input_path"] == str(v2)
    # Misma config (mismo snapshot) en ambos.
    assert items[0]["settings"]["codec"] == "h264_nvenc"
    assert items[1]["settings"]["codec"] == "h264_nvenc"
    assert items[0]["ai_engines"] == items[1]["ai_engines"]
    assert items[0]["pipeline_order"] == items[1]["pipeline_order"] == \
        ["Import", "Upscale", "Compress"]
    # Outputs distintos por archivo.
    assert items[0]["output_path"] == str(out_dir / "a_compressed.mp4")
    assert items[1]["output_path"] == str(out_dir / "b_compressed.mp4")
    assert "Queued 2 videos" in messages[-1]


def test_output_path_batch_with_fixed_name_avoids_collision(tmp_path, monkeypatch):
    """Un output_name fijo + varios videos -> nombres por archivo."""
    from src.ui.main_window import MainWindow

    v1 = tmp_path / "one.mp4"
    v2 = tmp_path / "two.mp4"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    v1.write_bytes(b"x")
    v2.write_bytes(b"x")

    mw = MainWindow.__new__(MainWindow)
    mw.input_paths = [str(v1), str(v2)]
    mw.input_path = str(v1)
    mw.output_dir = str(out_dir)

    class _SettingsPanel:
        def get_settings(self):
            return {"format": "mkv", "output_name": "final"}

    mw.settings_panel = _SettingsPanel()

    assert mw._get_output_path(str(v1)) == str(out_dir / "final_one.mkv")
    assert mw._get_output_path(str(v2)) == str(out_dir / "final_two.mkv")
    # Sin nombre fijo: default <base>_compressed
    mw.settings_panel = _SettingsPanel()
    mw.settings_panel.get_settings = lambda: {"format": "mp4", "output_name": ""}
    assert mw._get_output_path(str(v1)) == str(out_dir / "one_compressed.mp4")


def test_tasks_panel_queue_tab_smoke(qapp):
    """Pestaña Queue: rebuild, estados, quitar y limpiar sin crashear."""
    from src.ui.tasks_panel import TasksPanel

    panel = TasksPanel()
    panel.rebuild_queue([
        {"queue_id": "q1", "name": "movie1.mp4", "status": "queued"},
        {"queue_id": "q2", "name": "movie2.mp4", "status": "queued"},
    ])
    assert "q1" in panel._queue_rows and "q2" in panel._queue_rows
    panel.set_queue_status("q1", "running")
    assert panel._queue_rows["q1"].pill.text() == "RUNNING"
    panel.set_queue_status("q1", "done")
    assert panel._queue_rows["q1"].pill.text() == "DONE ✓"

    # Quitar un item vía la API pública (UI + persistencia no-crash).
    panel.remove_queue_item("q2")
    assert "q2" not in panel._queue_rows
    assert "q1" in panel._queue_rows

    panel.clear_queue_rows()
    assert panel._queue_rows == {}
    assert panel.btn_start_queue.isEnabled() is False


# ---------------------------------------------------------------------------
# 12. Model scale: filename parsing + target scale honored by engines
# ---------------------------------------------------------------------------


def test_parse_model_scale_from_name_real_world_names():
    """Escalas extraídas de nombres reales de pesos del ecosistema."""
    from src.ui.workflow_widget import _parse_model_scale_from_name as _ps

    assert _ps("RealESRGAN_x4plus.pth") == 4
    assert _ps("realesr-animevideov3-x4.pth") == 4
    assert _ps("realesr-animevideov3-x2.pth") == 2
    assert _ps("realesr-animevideov3-x3.pth") == 3
    assert _ps("realesr-general-x4v3.pth") == 4
    assert _ps("4x-UltraSharp.pth") == 4
    assert _ps("Real_CUGAN_x2.pth") == 2
    assert _ps("realesrgan-x4plus-anime-6b.pth") == 4
    assert _ps("RealESRGAN_x4plus_anime_6B.pth") == 4
    assert _ps("Real-CUGAN_x3_no_denoise.pth") == 3
    assert _ps("realesr_animevideov3_x4_plus.pth") == 4


def test_parse_model_scale_from_name_unknown_returns_one():
    from src.ui.workflow_widget import _parse_model_scale_from_name as _ps

    assert _ps("model.safetensors") == 1
    assert _ps("SwinIR_003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L.pth") == 1
    assert _ps("") == 1


def test_pytorch_engine_get_scale_factor_honors_target(tmp_path, monkeypatch):
    """Con _target_scale menor que la nativa, get_scale_factor reporta la
    escala elegida por el usuario (x2 sobre un peso nativo x4)."""
    import src.ai.inference as inf
    from src.ai.inference import PyTorchInferenceEngine

    pth = tmp_path / "model.pth"
    pth.write_bytes(b"fake")

    # No necesitamos torch real: solo comprobamos el getter.
    class _Noop(PyTorchInferenceEngine):
        def load(self):
            self._scale = 4
            self._loaded = True
            return True

    monkeypatch.setattr(inf, "PyTorchInferenceEngine", _Noop)
    eng = inf.PyTorchInferenceEngine(str(pth), "upscale-realesr-animevideov3")
    assert eng.load() is True
    assert eng.get_scale_factor() == 4  # nativa sin target

    eng._target_scale = 2
    assert eng.get_scale_factor() == 2  # el usuario pidió x2

    eng._target_scale = 8  # target > nativa: no es válido, se queda nativa
    assert eng.get_scale_factor() == 4

    eng._target_scale = 2
    eng._scale = 2  # peso real x2 y pedido x2: sin cambio
    assert eng.get_scale_factor() == 2


def test_onnx_engine_get_scale_factor_honors_target():
    from src.ai.inference import ONNXRUNTIMEInferenceEngine

    eng = ONNXRUNTIMEInferenceEngine("", "onnxruntime")
    eng._input_shape = (1, 3, 64, 64)
    eng._output_shape = (1, 3, 256, 256)
    assert eng.get_scale_factor() == 4

    eng._target_scale = 2
    assert eng.get_scale_factor() == 2


# ---------------------------------------------------------------------------
# 13. ONNX converter: descriptive failures + architecture support gating
# ---------------------------------------------------------------------------

def test_onnx_archs_support_set():
    from src.ai.architectures import (
        ONNX_CONVERTIBLE_ARCHS,
        arch_is_onnx_convertible,
        onnx_convertible_archs,
    )

    assert ONNX_CONVERTIBLE_ARCHS == frozenset({"omnisr", "rrdb", "srvgg", "cugan"})
    assert onnx_convertible_archs() is ONNX_CONVERTIBLE_ARCHS
    assert arch_is_onnx_convertible("rrdb") is True
    assert arch_is_onnx_convertible("srvgg") is True
    assert arch_is_onnx_convertible("cugan") is True
    assert arch_is_onnx_convertible("omnisr") is True
    assert arch_is_onnx_convertible("swinir") is False
    assert arch_is_onnx_convertible("nafnet") is False
    assert arch_is_onnx_convertible("") is False
    assert arch_is_onnx_convertible(None) is False


def test_catalog_arch_lookup_static_and_user(isolated_engines_state):
    from src.ai.registry import CATALOG_ARCH, get_catalog_arch

    assert CATALOG_ARCH["upscale-realesrgan-x4plus"] == "rrdb"
    assert CATALOG_ARCH["upscale-realesr-animevideov3"] == "srvgg"
    assert CATALOG_ARCH["upscale-omnisr-x4"] == "omnisr"
    assert CATALOG_ARCH["facerestore-gfpgan"] == "gfpgan"  # no convertible
    # FSR algorítmicos: sin arquitectura conocida -> None
    assert get_catalog_arch("upscale-fsr31-2x") is None
    assert get_catalog_arch("") is None
    assert get_catalog_arch("no-existe-123") is None

    # Motores de usuario: arch detectada en user_models.json
    from src.ai.user_models import register
    register("user-upscale-foo", display_name="Foo", category="Upscale",
             hf_repo_id="x/y", arch="srvgg", scale=4)
    assert get_catalog_arch("user-upscale-foo") == "srvgg"

    # Enriquecido por iter_catalog/list_installed para consumo de la UI
    from src.ai.registry import iter_catalog, list_installed
    from src.ai.engines_state import mark_installed
    mark_installed("upscale-realesr-animevideov3")
    entries = list(iter_catalog())
    row = next(e for e in entries if e["engine_id"] == "upscale-realesr-animevideov3")
    assert row.get("arch") == "srvgg"


def test_load_checkpoint_prefers_weights_only_true(monkeypatch):
    """torch.load debe llamarse con weights_only=True por defecto, nunca
    con weights_only=False de primeras (seguridad contra pickle)."""
    torch = pytest.importorskip("torch")
    import src.ai.inference as inf

    calls = []

    def _fake_load(path, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return {"fake": "state"}

    monkeypatch.setattr(torch, "load", _fake_load)
    assert inf.load_checkpoint_state_dict("/tmp/x.pth") == {"fake": "state"}
    assert calls == [True]


def test_load_checkpoint_refuses_unsafe_without_preference(monkeypatch):
    """Un checkpoint que requiere unpickling se rechaza con un mensaje claro
    a menos que el usuario active la preferencia explícitamente."""
    torch = pytest.importorskip("torch")
    import src.ai.inference as inf

    def _fake_load(path, **kwargs):
        if kwargs.get("weights_only") is True:
            raise RuntimeError(
                "Weights only load failed. Re-running torch.load with "
                "weights_only=False will likely succeed..."
            )
        return {"unsafe": True}

    monkeypatch.setattr(torch, "load", _fake_load)
    monkeypatch.setattr(inf.preferences, "get", lambda k, d=None: False)

    with pytest.raises(RuntimeError, match="cannot be loaded safely"):
        inf.load_checkpoint_state_dict("/tmp/x.pth")


def test_load_checkpoint_retries_unsafe_when_opted_in(monkeypatch):
    """Con la preferencia activa se reintenta con weights_only=False."""
    torch = pytest.importorskip("torch")
    import src.ai.inference as inf

    calls = []

    def _fake_load(path, **kwargs):
        wo = kwargs.get("weights_only")
        calls.append(wo)
        if wo is True:
            raise RuntimeError("Weights only load failed ...")
        return {"unsafe": True}

    monkeypatch.setattr(torch, "load", _fake_load)
    monkeypatch.setattr(inf.preferences, "get", lambda k, d=None: True)

    assert inf.load_checkpoint_state_dict("/tmp/x.pth") == {"unsafe": True}
    assert calls == [True, False]


def test_weights_only_restriction_detection():
    """_is_weights_only_restriction distingue el error de weights_only de
    un fichero corrupto (que no debe ofrecer el camino inseguro)."""
    from src.ai.inference import _is_weights_only_restriction

    assert _is_weights_only_restriction(
        RuntimeError("Weights only load failed. Re-running torch.load...")
    )
    assert _is_weights_only_restriction(
        RuntimeError("Unsupported global: GLOBAL foo.bar was not allowed")
    )
    assert _is_weights_only_restriction(
        RuntimeError("please load with weights_only=False")
    )
    assert not _is_weights_only_restriction(
        RuntimeError("invalid load key, 'v'.")
    )
    assert not _is_weights_only_restriction(OSError("No such file"))


def test_load_checkpoint_propagates_corrupt_file_error(monkeypatch):
    """Un fichero corrupto (no una restricción de weights_only) se propaga
    tal cual, sin la oferta de cargar inseguro."""
    torch = pytest.importorskip("torch")
    import src.ai.inference as inf

    def _fake_load(path, **kwargs):
        raise RuntimeError("invalid load key, 'v'.")

    monkeypatch.setattr(torch, "load", _fake_load)
    with pytest.raises(RuntimeError, match="invalid load key"):
        inf.load_checkpoint_state_dict("/tmp/x.pth")


def test_convert_pth_to_onnx_ex_reports_missing_file():
    from src.ai.inference import convert_pth_to_onnx_ex

    path, err = convert_pth_to_onnx_ex("Z:/no/such/weights.pth")
    assert path is None
    assert "not found" in err


def test_convert_pth_to_onnx_ex_reports_unknown_arch(tmp_path, local_store_runtime):
    """Un state_dict de arquitectura desconocida falla con razón legible,
    no con un error criptico del exporter."""
    torch = pytest.importorskip("torch")
    from src.ai.inference import convert_pth_to_onnx_ex

    m = torch.nn.Conv2d(3, 3, 3)
    pth = tmp_path / "unknown.pth"
    torch.save(m.state_dict(), pth)

    path, err = convert_pth_to_onnx_ex(str(pth))
    assert path is None
    assert "architecture" in err.lower()
    assert not (tmp_path / "unknown.onnx").exists()


def test_convert_pth_to_onnx_ex_reports_multichannel(tmp_path, local_store_runtime):
    """Modelos de entrada != 3 canales: razón clara y sin archivo .onnx."""
    torch = pytest.importorskip("torch")
    from src.ai.architectures import SRVGGNetCompact
    from src.ai.inference import convert_pth_to_onnx_ex

    m = SRVGGNetCompact(num_in_ch=12, num_out_ch=12, num_feat=4, num_conv=2, upscale=4)
    pth = tmp_path / "bad12ch.pth"
    torch.save(m.state_dict(), pth)

    path, err = convert_pth_to_onnx_ex(str(pth))
    assert path is None
    assert "12" in err
    assert not (tmp_path / "bad12ch.onnx").exists()


def test_convert_pth_to_onnx_ex_reuses_existing_good_onnx(tmp_path, local_store_runtime):
    """Si ya existe un .onnx de 3 canales, se reutiliza sin reconvertir."""
    torch = pytest.importorskip("torch")
    from src.ai.architectures import SRVGGNetCompact
    from src.ai.inference import convert_pth_to_onnx_ex

    m = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=4, num_conv=2, upscale=4)
    m.eval()
    pth = tmp_path / "model.pth"
    torch.save(m.state_dict(), pth)
    onnx_path = tmp_path / "model.onnx"
    torch.onnx.export(
        m, torch.randn(1, 3, 64, 64), str(onnx_path), opset_version=18,
        input_names=["input"], output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,  # exporter legacy (mismo camino que convert_pth_to_onnx_ex)
    )

    path, err = convert_pth_to_onnx_ex(str(pth))
    assert path == str(onnx_path)
    assert err == ""


def test_pick_weight_file_prefers_engine_matching_weight(tmp_path):
    """El repo de Real-ESRGAN trae 9 pesos; la selección debe coger el
    realesr-animevideov3.pth y no un RRDBNet de 67 MB (el bug que hacía
    fallar/convertir mal el modelo del usuario)."""
    from src.ai.inference import pick_weight_file

    d = tmp_path / "m"
    d.mkdir()
    for f in (
        "AI-Forever_x2plus.pth", "AI-Forever_x4plus.pth",
        "RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B.pth", "RealESRNet_x4plus.pth",
        "realesr-animevideov3.pth", "realesr-general-wdn-x4v3.pth",
        "realesr-general-x4v3.pth",
    ):
        (d / f).write_bytes(b"x")

    picked = pick_weight_file(str(d), "upscale-realesr-animevideov3")
    assert picked == str(d / "realesr-animevideov3.pth")

    # Otros engine_ids del mismo repo apuntan a su propio peso.
    picked_general = pick_weight_file(str(d), "upscale-realesr-general-x4v3")
    assert picked_general == str(d / "realesr-general-x4v3.pth")


def test_pick_weight_file_root_preferred_on_ties(tmp_path):
    """A igual puntuación gana la raíz (orden estable, no el del walk)."""
    from src.ai.inference import pick_weight_file

    d = tmp_path / "m"
    (d / "sub").mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x")
    (d / "sub" / "model.safetensors").write_bytes(b"x")

    assert pick_weight_file(str(d), "upscale-foo") == str(d / "model.safetensors")
    assert pick_weight_file(str(d), "upscale-foo", recursive=False) == str(d / "model.safetensors")
    assert pick_weight_file(str(d / "nonexistent"), "upscale-foo") is None


def test_onnx_worker_finds_matching_weight(tmp_path):
    """OnnxConvertWorker._find_weight_file elige el peso que coincide con el
    engine_id y no el primer .pth del walk (orden del filesystem)."""
    from src.ai.fetcher import OnnxConvertWorker

    d = tmp_path / "m"
    d.mkdir()
    for f in (
        "AI-Forever_x2plus.pth", "RealESRGAN_x4plus.pth",
        "realesr-animevideov3.pth", "realesr-general-x4v3.pth",
    ):
        (d / f).write_bytes(b"x")

    w = OnnxConvertWorker("upscale-realesr-animevideov3", str(d), "Upscale")
    assert w._find_weight_file() == str(d / "realesr-animevideov3.pth")


class _FakeORTIn:
    def __init__(self, shape):
        self.shape = shape


class _FakeORTSess:
    def __init__(self, shape):
        self._shape = shape

    def get_inputs(self):
        return [_FakeORTIn(self._shape)]


class _FakeORT:
    def __init__(self, shape):
        self._shape = shape

    def InferenceSession(self, path, providers=None):
        return _FakeORTSess(self._shape)


def test_onnx_input_channels_ok_three_state_semantics(monkeypatch):
    """El probe devuelve True (3ch), False (fijo != 3) o None (no se pudo
    verificar: onnxruntime ausente). None NO debe borrar archivos."""
    import sys
    from src.ai.inference import _onnx_input_channels_ok

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeORT((1, 3, 64, 64)))
    assert _onnx_input_channels_ok("x.onnx") is True

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeORT((1, 12, 64, 64)))
    assert _onnx_input_channels_ok("x.onnx") is False

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert _onnx_input_channels_ok("x.onnx") is None


def test_convert_pth_to_onnx_ex_keeps_unverifiable_existing_onnx(tmp_path, monkeypatch):
    """Si el probe no puede verificar el .onnx existente (onnxruntime
    ausente) NO se borra: se reutiliza (una conversión previa lo creó).
    Este es el caso del usuario con realesr-animevideov3.onnx en disco."""
    import sys
    from src.ai.inference import convert_pth_to_onnx_ex

    pth = tmp_path / "model.pth"
    pth.write_bytes(b"not-a-real-checkpoint")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"existing-onnx")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    path, err = convert_pth_to_onnx_ex(str(pth))
    assert path == str(onnx)
    assert err == ""
    assert onnx.exists()


def test_convert_pth_to_onnx_ex_removes_proven_bad_existing_onnx(tmp_path, monkeypatch):
    """Solo cuando el probe PRUEBA que el .onnx es de canales != 3 se borra
    (y si torch no está, la reconversión falla sin dejar archivos)."""
    import sys
    from src.ai.inference import convert_pth_to_onnx_ex

    pth = tmp_path / "model.pth"
    pth.write_bytes(b"x")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeORT((1, 12, 64, 64)))

    path, err = convert_pth_to_onnx_ex(str(pth))
    assert path is None
    assert not onnx.exists()


def test_engines_state_backend_auto_default_and_set(isolated_engines_state):
    """Preferencia de motor por engine: default 'auto', persiste torch /
    onnxruntime y normaliza valores desconocidos."""
    import src.ai.engines_state as es

    assert es.get_backend("upscale-realesr-animevideov3") == "auto"
    assert es.get_state("upscale-realesr-animevideov3")["inference_backend"] == "auto"

    assert es.set_backend("upscale-realesr-animevideov3", "torch") is True
    assert es.get_backend("upscale-realesr-animevideov3") == "torch"

    assert es.set_backend("upscale-realesr-animevideov3", "ONNXRUNTIME") is True
    assert es.get_backend("upscale-realesr-animevideov3") == "onnxruntime"

    # Valores desconocidos caen a "auto".
    assert es.set_backend("upscale-realesr-animevideov3", "banana") is True
    assert es.get_backend("upscale-realesr-animevideov3") == "auto"

    # El estado en disco lo normaliza al leerlo.
    assert es.get_state("upscale-realesr-animevideov3")["inference_backend"] in (
        "auto", "torch", "onnxruntime",
    )


def test_get_model_path_torch_uses_pick_weight_file(tmp_path, monkeypatch):
    """get_model_path(..., 'torch') elige el peso que coincide con el
    engine_id usando el mismo scoring que el conversor ONNX."""
    from src.ai import engines_state as es
    from src.ai.inference import get_model_path

    d = tmp_path / "m"
    d.mkdir()
    for f in ("AI-Forever_x2plus.pth", "realesr-animevideov3.pth",
              "RealESRGAN_x4plus.pth"):
        (d / f).write_bytes(b"x")

    monkeypatch.setattr(es, "get_state", lambda eid: {
        "installed": True,
        "install_path": str(d),
        "onnx_converted": False,
        "onnx_path": None,
    })

    assert get_model_path("upscale-realesr-animevideov3", "torch") == str(
        d / "realesr-animevideov3.pth"
    )


# ---------------------------------------------------------------------------
# 14. ONNX conversion progress on the model row (background, non-blocking)
# ---------------------------------------------------------------------------


def _engine_row_dict(installed=True, converted=False, backend="auto"):
    return {
        "engine_id": "upscale-realesr-animevideov3",
        "display_name": "RealESR AnimeVideo v3 (fast, x2/x3/x4)",
        "category": "Upscale",
        "hf_url": "https://huggingface.co/leonelhs/realesrgan",
        "size_estimate": "~2.4 MB",
        "sample_shape": "1x3xHxW -> 1x3x(4H)x(4W)",
        "license": "BSD-3-Clause",
        "popularity_score": 88,
        "quality_score": 78,
        "is_implemented": True,
        "sentinel": "README.md",
        "arch": "srvgg",
        "state": {
            "installed": installed,
            "enabled": installed,
            "onnx_converted": converted,
            "onnx_path": None,
            "inference_backend": backend,
        },
    }


def test_engine_row_convert_progress_strip(qapp):
    """La franja de conversión ONNX de la fila muestra progreso, bloquea
    las acciones, emite cancelar y restaura al terminar; el error queda
    visible en rojo (sin diálogo modal)."""
    from src.ui.engine_row import EngineRow

    row = EngineRow(_engine_row_dict())
    assert row.conv_bar.isHidden()  # oculta por defecto

    cancelled = []
    row.cancel_convert_requested.connect(cancelled.append)

    row.set_converting(True, progress=15, status="Convirtiendo a ONNX...")
    assert not row.conv_bar.isHidden()
    assert row.conv_bar.value() == 15
    assert not row.conv_status.isHidden()
    assert row.conv_status.text() == "Convirtiendo a ONNX..."
    assert row.btn_cancel_convert.isEnabled()
    assert not row.btn_convert.isEnabled()
    assert not row.btn_remove.isEnabled()
    assert not row.combo_backend.isEnabled()

    row._on_cancel_convert_clicked()
    assert cancelled == ["upscale-realesr-animevideov3"]

    row.set_converting(False)
    assert row.conv_bar.isHidden()
    assert row.btn_convert.isEnabled()
    assert row.btn_remove.isEnabled()
    assert row.combo_backend.isEnabled()

    row.show_convert_error("ONNX conversion failed.\nDetail")
    assert not row.conv_status.isHidden()
    assert "#e74c3c" in row.conv_status.styleSheet()

    row.show_convert_cancelled()
    assert not row.conv_status.isHidden()


def _wait_tab_build(qapp, tab, max_iter=300):
    """Procesa eventos hasta que la construcción progresiva del tab termina.

    La construcción de EnginesTab va por chunks con QTimer (15 ms). En el
    entorno offscreen, tras decenas de tests, ``QTest.qWait`` puede reventar
    nativamente procesando la gran cantidad de widgets; un bucle de
    ``processEvents`` + ``time.sleep`` es más robusto y suficiente para
    avanzar los timers."""
    for _ in range(max_iter):
        qapp.processEvents()
        if not tab.is_building():
            return True
        time.sleep(0.01)
    return not tab.is_building()


def test_engines_tab_conversion_state_survives_repopulate(qapp, isolated_engines_state):
    """El progreso de conversión en la fila no se pierde al repoblar la
    pestaña (repintado del catálogo a mitad de conversión)."""
    import src.ai.engines_state as es

    tmp = isolated_engines_state
    d = os.path.join(tmp, "models", "Upscale", "realesr-animevideov3")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "realesr-animevideov3.pth"), "w") as f:
        f.write("x")
    es.mark_installed("upscale-realesr-animevideov3", install_path=d)

    from src.ui.engines_tab import EnginesTab

    # No se muestra la ventana: en offscreen, mostrar un árbol de ~28 filas
    # tras decenas de tests previos provoca un access violation nativo al
    # procesar eventos (la lógica que se prueba no requiere visibilidad).
    tab = EnginesTab()
    try:
        assert _wait_tab_build(qapp, tab)
        eid = "upscale-realesr-animevideov3"
        assert eid in tab._rows_by_id

        tab.set_engine_converting(eid, True, progress=10, status="Buscando los pesos...")
        row = tab._rows_by_id[eid]
        assert not row.conv_bar.isHidden()
        assert row.conv_bar.value() == 10
        assert not row.btn_convert.isEnabled()

        # Un repopulate NO debe perder el estado de conversión.
        tab.repopulate()
        assert _wait_tab_build(qapp, tab)
        new_row = tab._rows_by_id[eid]
        assert new_row is not row
        assert not new_row.conv_bar.isHidden()
        assert new_row.conv_status.text() == "Buscando los pesos..."
        assert not new_row.btn_convert.isEnabled()

        tab.set_engine_converting(eid, False)
        assert new_row.conv_bar.isHidden()
        assert tab._converting_id is None
        assert new_row.btn_convert.isEnabled()

        # Error inline persistente (sin diálogo modal).
        tab.set_engine_converting(eid, True)
        tab.set_engine_converting(eid, False)
        tab.show_engine_convert_error(eid, "ONNX conversion failed.")
        assert not tab._rows_by_id[eid].conv_status.isHidden()
        assert "#e74c3c" in tab._rows_by_id[eid].conv_status.styleSheet()
    finally:
        tab.setParent(None)
        tab.deleteLater()
        qapp.processEvents()


def test_friendly_convert_status_translates_worker_phases():
    from src.ui.dependencies_dialog import _friendly_convert_status

    assert _friendly_convert_status("Locating model weights...") == "Buscando los pesos del modelo..."
    assert _friendly_convert_status(
        "Converting to ONNX with torch.onnx.export..."
    ) == "Convirtiendo a ONNX (torch.onnx.export)..."
    assert _friendly_convert_status("export complete") == "Conversión completada."
    assert _friendly_convert_status("anything else") == "anything else"
    assert _friendly_convert_status("") == ""


# ---------------------------------------------------------------------------
# 15. Benchmarks por backend (ms/frame ONNX vs PyTorch) en el selector
# ---------------------------------------------------------------------------


def test_backend_benchmark_roundtrip_and_active_selection(isolated_engines_state):
    """Los benchmarks se guardan por backend; get_benchmark devuelve la
    medición del backend ACTIVO (torch forzado -> PyTorch, si no -> ONNX)."""
    import src.ai.engines_state as es

    eid = "upscale-realesr-animevideov3"
    onnx_b = {"ms_per_frame": 12.3, "fps": 81.3, "device": "cuda"}
    torch_b = {"ms_per_frame": 15.1, "fps": 66.2, "device": "cuda"}

    assert es.get_backend_benchmarks(eid) == {"onnx": None, "torch": None}
    assert es.set_backend_benchmark(eid, "onnx", onnx_b) is True
    assert es.set_backend_benchmark(eid, "torch", torch_b) is True

    got = es.get_backend_benchmarks(eid)
    assert got["onnx"]["ms_per_frame"] == 12.3
    assert got["torch"]["ms_per_frame"] == 15.1

    # Backend activo: auto con ambos medidos -> el más rápido (aquí ONNX
    # 12.3 < torch 15.1); torch/torch; onnxruntime -> onnx.
    es.set_backend(eid, "auto")
    assert es.get_benchmark(eid)["ms_per_frame"] == 12.3
    es.set_backend(eid, "torch")
    assert es.get_benchmark(eid)["ms_per_frame"] == 15.1
    es.set_backend(eid, "onnxruntime")
    assert es.get_benchmark(eid)["ms_per_frame"] == 12.3
    es.set_backend(eid, "auto")

    # Borrar un backend; el otro sobrevive.
    assert es.set_backend_benchmark(eid, "torch", None) is True
    assert es.get_backend_benchmarks(eid) == {"onnx": onnx_b, "torch": None}

    # Borrar el último -> la entrada completa desaparece.
    assert es.set_backend_benchmark(eid, "onnx", None) is True
    assert es.get_backend_benchmarks(eid) == {"onnx": None, "torch": None}


def test_benchmark_legacy_flat_entry_still_readable(isolated_engines_state):
    """Entradas planas legacy (un solo dict sin backend) siguen leyéndose."""
    import src.ai.engines_state as es

    eid = "upscale-realesr-animevideov3"
    flat = {"ms_per_frame": 20.0, "fps": 50.0, "device": "cpu"}
    assert es.set_benchmark(eid, flat) is True
    assert es.get_benchmark(eid)["ms_per_frame"] == 20.0
    assert es.get_backend_benchmarks(eid) == {"onnx": None, "torch": None}
    es.clear_benchmark(eid)
    assert es.get_benchmark(eid) is None


def test_benchmark_backend_returns_none_without_weights(tmp_path, monkeypatch):
    """Sin pesos medibles, benchmark_backend devuelve None sin escribir."""
    import src.ai.benchmark as bm

    assert bm.benchmark_backend("upscale-realesr-animevideov3", "onnx",
                                target_dir=str(tmp_path), use_cache=False) is None
    assert bm.benchmark_backend("upscale-realesr-animevideov3", "torch",
                                target_dir=str(tmp_path), use_cache=False) is None
    assert bm.benchmark_backend("upscale-realesr-animevideov3", "banana",
                                target_dir=str(tmp_path), use_cache=False) is None


def test_resolve_backend_auto_picks_fastest_measured(isolated_engines_state):
    """'auto' elige el backend más rápido medido cuando hay benchmarks de
    ambos; sin comparación devuelve 'auto' (heurística clásica del render)."""
    import src.ai.engines_state as es

    eid = "upscale-realesr-animevideov3"
    onnx_b = {"ms_per_frame": 12.3, "device": "cuda"}
    torch_b = {"ms_per_frame": 15.1, "device": "cuda"}

    # Sin benchmarks -> "auto": el render conserva la heurística clásica.
    assert es.resolve_backend(eid) == "auto"

    # Solo un backend medido -> "auto": no hay comparación posible.
    es.set_backend_benchmark(eid, "onnx", onnx_b)
    assert es.resolve_backend(eid) == "auto"

    # Ambos medidos: ONNX más rápido -> onnxruntime.
    es.set_backend_benchmark(eid, "torch", torch_b)
    assert es.resolve_backend(eid) == "onnxruntime"

    # Torch más rápido -> torch (y get_benchmark sigue al backend resuelto).
    es.set_backend_benchmark(eid, "torch", {"ms_per_frame": 8.0, "device": "cuda"})
    assert es.resolve_backend(eid) == "torch"
    assert es.get_benchmark(eid)["ms_per_frame"] == 8.0

    # Empate -> onnxruntime (criterio del selector: default histórico ONNX).
    es.set_backend_benchmark(eid, "torch", {"ms_per_frame": 12.3, "device": "cuda"})
    assert es.resolve_backend(eid) == "onnxruntime"

    # La preferencia explícita manda sobre los benchmarks.
    es.set_backend(eid, "torch")
    assert es.resolve_backend(eid) == "torch"
    es.set_backend(eid, "onnxruntime")
    assert es.resolve_backend(eid) == "onnxruntime"
    es.set_backend(eid, "auto")
    assert es.resolve_backend(eid) == "onnxruntime"  # vuelve a medir el más rápido


def test_recommended_backend_logic():
    from src.ui.engine_row import EngineRow

    assert EngineRow._recommended_backend(None, None) is None
    assert EngineRow._recommended_backend({"ms_per_frame": 12.3}, None) == "onnx"
    assert EngineRow._recommended_backend(None, {"ms_per_frame": 15.1}) == "torch"
    assert EngineRow._recommended_backend(
        {"ms_per_frame": 12.3}, {"ms_per_frame": 15.1}) == "onnx"
    assert EngineRow._recommended_backend(
        {"ms_per_frame": 20.0}, {"ms_per_frame": 11.5}) == "torch"
    # Empate -> ONNX (backend por defecto con .onnx).
    assert EngineRow._recommended_backend(
        {"ms_per_frame": 10.0}, {"ms_per_frame": 10.0}) == "onnx"


def test_engine_row_backend_selector_shows_benchmarks(qapp):
    """El selector muestra ms/frame medidos y la sugerencia del más rápido
    con botón 'Usar' que emite backend_changed."""
    from src.ui.engine_row import EngineRow

    eng = _engine_row_dict()
    eng["benchmarks"] = {
        "onnx": {"ms_per_frame": 12.3},
        "torch": {"ms_per_frame": 15.1},
    }
    row = EngineRow(eng)
    labels = [row.combo_backend.itemText(i) for i in range(row.combo_backend.count())]
    assert "PyTorch — 15.1 ms/f" in labels
    assert "ONNX Runtime — 12.3 ms/f" in labels

    assert not row.lbl_backend_suggest.isHidden()
    assert "ONNX Runtime" in row.lbl_backend_suggest.text()
    assert "12.3 ms/f" in row.lbl_backend_suggest.text()
    assert not row.btn_use_recommended.isHidden()
    assert row.btn_use_recommended.isEnabled()
    assert row._recommended == "onnx"

    got = []
    row.backend_changed.connect(lambda eid, b: got.append((eid, b)))
    row._on_use_recommended()
    assert got == [("upscale-realesr-animevideov3", "onnx")]

    # Sin datos: sin sugerencia y sin ms/f en los items.
    row3 = EngineRow(_engine_row_dict())
    assert row3.lbl_backend_suggest.isHidden()
    labels3 = [row3.combo_backend.itemText(i) for i in range(row3.combo_backend.count())]
    assert labels3 == ["Auto", "PyTorch", "ONNX Runtime"]

    # Solo un backend medido: sin sugerencia (no comparable), item con ms/f.
    eng4 = _engine_row_dict()
    eng4["benchmarks"] = {"onnx": {"ms_per_frame": 9.0}, "torch": None}
    row4 = EngineRow(eng4)
    assert row4.lbl_backend_suggest.isHidden()
    assert "ONNX Runtime — 9.0 ms/f" in [
        row4.combo_backend.itemText(i) for i in range(row4.combo_backend.count())
    ]


# ==--------------------------------------------------------------------==
# P0: render stability contracts
# ==--------------------------------------------------------------------==


def test_process_supervisor_stops_registered_children():
    from src.ffmpeg.process_supervisor import ProcessSupervisor

    class FakeProcess:
        pid = 999999999

        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    proc = FakeProcess()
    supervisor = ProcessSupervisor()
    supervisor.register(proc, "encoder")
    assert supervisor.stop_all(grace_seconds=0.01, kill_seconds=0.01) == []
    assert proc.terminated is True
    assert proc.poll() is not None


def test_render_preflight_rejects_bad_trim_and_accepts_valid_input(tmp_path, monkeypatch):
    import src.processing.pipeline as pipeline

    ffmpeg = tmp_path / "ffmpeg"
    source = tmp_path / "input.mp4"
    ffmpeg.write_bytes(b"binary")
    source.write_bytes(b"video")
    monkeypatch.setattr(
        pipeline, "health_check_ffmpeg",
        lambda *args, **kwargs: (True, "6.0", None),
    )
    monkeypatch.setattr(
        pipeline, "get_video_info",
        lambda *args, **kwargs: {
            "width": "640", "height": "360", "duration": "10", "r_frame_rate": "30/1",
        },
    )
    settings = {"codec": "libx264", "format": "mp4", "start_time": "8", "end_time": "4"}
    errors = pipeline.validate_render_preflight(
        str(ffmpeg), str(source), str(tmp_path / "out.mp4"), settings,
    )
    assert any("Trim end" in error for error in errors)

    settings["end_time"] = "9"
    assert pipeline.validate_render_preflight(
        str(ffmpeg), str(source), str(tmp_path / "out.mp4"), settings,
    ) == []


def test_task_queue_enforces_render_state_transitions_and_recovery(tmp_path, monkeypatch):
    import src.task_queue as queue

    monkeypatch.setattr(queue, "_path", lambda: str(tmp_path / "queue.json"))
    queue_id = queue.enqueue(
        name="clip.mp4", input_path="/tmp/input.mp4", output_path="/tmp/output.mp4",
        settings={"codec": "libx264"},
    )
    assert queue.set_status(queue_id, "done") is False
    for state in ("validating", "preparing", "running", "muxing", "publishing", "done"):
        assert queue.set_status(queue_id, state) is True
    item = queue.get_item(queue_id)
    assert item["status"] == "done"
    assert len(item["state_history"]) == 6

    interrupted = queue.enqueue(
        name="interrupted.mp4", input_path="/tmp/input2.mp4", output_path="/tmp/output2.mp4",
        settings={},
    )
    assert queue.set_status(interrupted, "running") is True
    assert queue.recover_interrupted() == 1
    recovered = queue.get_item(interrupted)
    assert recovered["status"] == "queued"
    assert recovered["recovery_count"] == 1


def test_ai_chain_contract_catches_bad_engine_output():
    """A node that returns RGB/gray or changes resolution without being an
    upscaler must be flagged by the chain contract probe."""
    import numpy as np
    from src.processing.pipeline import validate_ai_chain_contract

    class _Good:
        engine_id = "good-upscaler"
        _mode = "upscale"

        def get_scale_factor(self):
            return 2

        def process_frame(self, frame):
            return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)

    class _BadChannels:
        engine_id = "rgb-returner"
        _mode = "upscale"

        def get_scale_factor(self):
            return 1

        def process_frame(self, frame):
            return np.zeros((64, 64, 3), dtype=np.uint8)  # RGB-shaped is fine...

    class _Gray:
        engine_id = "gray-returner"
        _mode = "upscale"

        def get_scale_factor(self):
            return 1

        def process_frame(self, frame):
            return np.zeros((64, 64), dtype=np.uint8)  # H,W only -> violation

    # Valid chain passes.
    assert validate_ai_chain_contract([_Good()], 64, 64) is True
    # A gray (2D) frame violates the BGR (H, W, 3) contract.
    with pytest.raises(RuntimeError, match=r"BGR \(H, W, 3\)"):
        validate_ai_chain_contract([_BadChannels(), _Gray()], 64, 64)


def test_ai_chain_contract_allows_upscalers_but_rejects_shape_change(tmp_path, monkeypatch):
    """Upscalers may change H/W; a non-upscaler that changes resolution
    (e.g. a broken denoiser) violates the contract."""
    import numpy as np
    from src.processing.pipeline import validate_ai_chain_contract

    class _Denoise:
        engine_id = "denoise"
        _mode = "upscale"

        def get_scale_factor(self):
            return 1

        def process_frame(self, frame):
            return np.zeros((32, 32, 3), dtype=np.uint8)  # silently halves size

    # The first step halves the probe from 64x64 to 32x32; the second
    # step feeds on that smaller frame, so no shape mismatch is flagged
    # between steps. But a lone denoiser must NOT change resolution.
    with pytest.raises(RuntimeError, match="without being an upscaler"):
        validate_ai_chain_contract([_Denoise()], 64, 64)


def test_task_history_recovers_interrupted_tasks(tmp_path, monkeypatch):
    """A task left 'running' by a crash is marked interrupted at startup."""
    import src.task_history as th

    monkeypatch.setattr(th, "_path", lambda: str(tmp_path / "tasks.json"))
    th.start_task("crash-1", "video.mp4")
    assert th.get_task("crash-1")["status"] == "running"

    assert th.recover_interrupted_tasks() == 1
    t = th.get_task("crash-1")
    assert t["status"] == "interrupted"
    assert t["finished_at"] is not None
    # Already-finished tasks are untouched.
    assert th.recover_interrupted_tasks() == 0


def test_cleanup_stale_staging_removes_old_temp_files(tmp_path, monkeypatch):
    """Abandoned .drag-audio-* / .draggy-render-* temp files older than the
    threshold are removed; recent ones are kept."""
    import time as _time
    import src.processing.pipeline as pipeline

    old_file = tmp_path / ".draggy-render-old.mp4"
    fresh_file = tmp_path / ".draggy-render-fresh.mp4"
    old_file.write_bytes(b"x")
    fresh_file.write_bytes(b"x")
    old = _time.time() - 7200
    os.utime(old_file, (old, old))

    monkeypatch.setattr(
        pipeline, "tempfile",
        type("FakeTemp", (), {"gettempdir": staticmethod(lambda: str(tmp_path))})(),
    )
    assert pipeline.cleanup_stale_staging(older_than_seconds=3600) == 1
    assert not old_file.exists()
    assert fresh_file.exists()


def test_selftest_reports_qt_platform_failure(tmp_path, monkeypatch):
    """run_selftest returns 1 and writes the failure to selftest.log when Qt
    cannot create a QApplication (nonexistent QPA backend).

    With a genuinely nonexistent QT_QPA_PLATFORM, Qt calls qFatal/abort and
    kills the process before Python can raise, so this test simulates the
    failure at the exact point Qt would abort - the QApplication constructor
    - while the environment still points at a nonexistent backend.
    """
    import main as main_module
    import PySide6.QtWidgets as QtWidgets

    monkeypatch.setenv("QT_QPA_PLATFORM", "bogus_platform")
    # Keep the test hermetic: never touch the real bin/ tree.
    monkeypatch.setattr("src.config.ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr("src.config.get_logs_dir", lambda: str(tmp_path))

    class _NoQApp:
        @staticmethod
        def instance():
            return None

        def __init__(self, *args, **kwargs):
            raise RuntimeError("no Qt platform plugin could be initialized")

    monkeypatch.setattr(QtWidgets, "QApplication", _NoQApp)

    rc = main_module.run_selftest()

    assert rc == 1
    log = (tmp_path / "selftest.log").read_text(encoding="utf-8")
    assert "FAIL QApplication offscreen" in log
    assert "SELFTEST FAIL" in log


def test_selftest_passes_with_working_qt(tmp_path, monkeypatch):
    """run_selftest reports the offscreen QApplication check as OK and
    returns 0 when Qt can create a real widget (qapp fixture provided).
    """
    import main as main_module

    monkeypatch.setattr("src.config.ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr("src.config.get_logs_dir", lambda: str(tmp_path))

    rc = main_module.run_selftest()

    assert rc == 0
    log = (tmp_path / "selftest.log").read_text(encoding="utf-8")
    assert "OK   QApplication offscreen" in log
    assert "OK   MainWindow offscreen" in log
    assert "OK   DependenciesDialog offscreen" in log
    assert "SELFTEST PASS" in log


def test_main_window_offscreen_reports_failure(qapp, monkeypatch):
    """check_main_window_offscreen returns (False, reason) without raising
    when the MainWindow constructor fails."""
    import src.qt_check as qt_check
    import src.ui.main_window as main_window_module

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("widget layout exploded")

    monkeypatch.setattr(main_window_module, "MainWindow", _Boom)

    ok, detail = qt_check.check_main_window_offscreen()

    assert ok is False
    assert "widget layout exploded" in detail


def test_dependencies_dialog_offscreen_reports_failure(qapp, monkeypatch):
    """check_dependencies_dialog_offscreen returns (False, reason) without
    raising when the DependenciesDialog constructor fails."""
    import src.qt_check as qt_check
    import src.ui.dependencies_dialog as dd_module

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("dialog layout exploded")

    monkeypatch.setattr(dd_module, "DependenciesDialog", _Boom)

    ok, detail = qt_check.check_dependencies_dialog_offscreen()

    assert ok is False
    assert "dialog layout exploded" in detail


def test_run_ui_checks_all_ok(qapp):
    """run_ui_checks ejecuta las tres validaciones en orden y las reporta
    todas como 'ok' cuando Qt y las dos pantallas construyen bien."""
    import src.qt_check as qt_check

    results = qt_check.run_ui_checks()

    assert [name for name, _, _ in results] == [
        "QApplication offscreen",
        "MainWindow offscreen",
        "DependenciesDialog offscreen",
    ]
    assert all(status == "ok" for _, status, _ in results), results


def test_run_ui_checks_skips_windows_when_qapp_fails(monkeypatch):
    """Si la QApplication falla, los checks de ventanas se marcan 'skip' y
    NO se ejecutan: construirlas sin app base no aporta información."""
    import src.qt_check as qt_check

    def _qapp_boom():
        return False, "no Qt platform plugin"

    def _never_called():
        raise AssertionError("no debería ejecutarse sin QApplication")

    monkeypatch.setattr(qt_check, "check_qapplication_offscreen", _qapp_boom)
    monkeypatch.setattr(qt_check, "check_main_window_offscreen", _never_called)
    monkeypatch.setattr(qt_check, "check_dependencies_dialog_offscreen", _never_called)

    results = qt_check.run_ui_checks()

    assert results == [
        ("QApplication offscreen", "fail", "no Qt platform plugin"),
        ("MainWindow offscreen", "skip", "sin QApplication válida"),
        ("DependenciesDialog offscreen", "skip", "sin QApplication válida"),
    ]


# Se mantienen AMBAS pruebas de capture_screenshots a propósito, porque
# cubren dos capas distintas del pre-flight:
#
# 1. test_capture_screenshots_main_returns_1_and_no_pngs_on_broken_main_window
#    (en proceso): valida el CONTRATO de run_ui_checks dentro de main()
#    parcheando el check directamente. Rápida y determinista, sin entorno.
# 2. test_capture_screenshots_preflight_rejects_broken_main_window
#    (subproceso): replica EXACTAMENTE el paso del smoke job de CI (misma
#    variable DRAGGY_TEST_BROKEN_MAIN_WINDOW, mismo comando, mismo rc). Es
#    la única que valida que el nombre de la variable del workflow sigue
#    cableado al hook de src/qt_check.py: si alguien renombra la variable
#    en un sitio sin el otro, solo esta prueba (o el propio CI) lo detecta.
#
# Sustituir 2 por 1 ahorraría ~6s de una suite de ~200s (3%) y perdería la
# cobertura del cableado env-var -> hook, así que se mantienen ambas. El
# fixture compartido es session-scoped, así que la carga del módulo solo
# ocurre una vez.
def test_capture_screenshots_preflight_rejects_broken_main_window(
    tmp_path, capture_screenshots_module
):
    """capture_screenshots.py con una MainWindow rota (el hook que usa el
    smoke job de CI) debe fallar con exit 1 y mensaje claro ANTES de
    generar ningún PNG - el pre-flight sigue siendo el gate de captura.
    """
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DRAGGY_TEST_BROKEN_MAIN_WINDOW"] = "1"
    out_dir = os.path.join(str(tmp_path), "out")

    proc = subprocess.run(
        [sys.executable, capture_screenshots_module.__file__, out_dir],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert (
        "ERROR: MainWindow offscreen check failed: simulated failure (smoke test)"
        in proc.stderr
    )
    assert os.path.isdir(out_dir)
    assert not [f for f in os.listdir(out_dir) if f.endswith(".png")]


def test_capture_screenshots_main_returns_1_and_no_pngs_on_broken_main_window(
    tmp_path, monkeypatch, capsys, qapp, capture_screenshots_module
):
    """capture_screenshots.main() con una MainWindow rota (el check real de
    qt_check devuelve fail) debe devolver 1, imprimir el error del pre-flight
    y NO escribir ningún PNG - cubre el contrato de run_ui_checks dentro del
    propio script, sin depender del hook de entorno del CI.
    """
    mod = capture_screenshots_module

    # La MainWindow está rota: el check que run_ui_checks ejecuta en segundo
    # lugar falla (igual que el hook del CI, pero parcheando la función real
    # de qt_check en vez de la variable de entorno).
    monkeypatch.setattr(
        "src.qt_check.check_main_window_offscreen",
        lambda: (False, "widget layout exploded"),
    )
    out_dir = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["capture_screenshots.py", out_dir])

    rc = mod.main()
    err = capsys.readouterr().err

    assert rc == 1
    assert "ERROR: MainWindow offscreen check failed: widget layout exploded" in err
    assert os.path.isdir(out_dir)
    assert not [f for f in os.listdir(out_dir) if f.endswith(".png")]


def test_capture_screenshots_main_success_writes_expected_pngs(
    tmp_path, monkeypatch, capsys, qapp, capture_screenshots_module
):
    """Caso de éxito de capture_screenshots.main(): con los tres checks OK,
    devuelve 0 y escribe EXACTAMENTE los dos PNG esperados (workflow y
    engines), ambos con cabecera PNG real - no vacíos ni basura.
    """
    mod = capture_screenshots_module

    out_dir = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["capture_screenshots.py", out_dir])

    rc = mod.main()
    capsys.readouterr()  # descartar posibles avisos de Qt en stderr

    assert rc == 0
    files = sorted(os.listdir(out_dir))
    assert files == [
        "DraggyEncoder-screenshot-engines.png",
        "DraggyEncoder-screenshot-workflow.png",
    ]
    png_magic = b"\x89PNG\r\n\x1a\n"
    contents = {}
    for name in files:
        path = os.path.join(out_dir, name)
        with open(path, "rb") as fh:
            contents[name] = fh.read()
        assert len(contents[name]) > 0
        assert contents[name][:8] == png_magic

    # Capturan pantallas distintas (workflow vs engines), no solo que ambos
    # archivos existan: el contenido de los dos PNG debe diferir.
    assert (
        contents["DraggyEncoder-screenshot-workflow.png"]
        != contents["DraggyEncoder-screenshot-engines.png"]
    )


# ==--------------------------------------------------------------------==
# Standalone runner (works even without pytest installed)
# ==--------------------------------------------------------------------==

if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
