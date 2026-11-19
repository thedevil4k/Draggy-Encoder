"""Pack GPU offline: troceado, hashes, extracción segura e importación.

El pack es el camino para un primer arranque sin internet ni pip. Estos tests
fijan las dos garantías que lo hacen seguro:

* el reparto en assets de menos de 2 GiB (y el manifiesto que los describe),
* que un pack inválido (otro Python, hash malo, ruta ``..``) no toca el store
  que ya funcionaba.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import zipfile
from pathlib import Path

import pytest

import src.runtimes.manager as rm
from src.runtimes import offline_pack


ROOT = Path(__file__).resolve().parents[1]

# El pack solo se publica para Windows x64 (ver README). La CI de Linux corre
# esta suite: sin el skip fallaría por diseño, no por una regresión.
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="El pack GPU offline solo se distribuye para Windows x64",
)


def _load_builder():
    path = ROOT / "packaging" / "build_gpu_pack.py"
    spec = importlib.util.spec_from_file_location("draggy_build_gpu_pack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _write_pkg(root: Path, name: str, version: str, tag: str) -> None:
    package = root / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        f"__version__ = {version!r}\n", encoding="utf-8"
    )
    dist = root / f"{name}-{version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {name.replace('_', '-')}\n"
        f"Version: {version}\n",
        encoding="utf-8",
    )
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        f"Tag: {tag}-{tag}-win_amd64\n",
        encoding="utf-8",
    )


def _make_store(root: Path) -> Path:
    """Store mínimo que satisface a la sonda de GPU y al builder."""
    tag = _python_tag()
    _write_pkg(root, "torch", "9.9.9", tag)
    # ``torch.cuda`` debe existir como atributo tras ``import torch``: la sonda
    # llama a ``module.cuda.is_available()`` sin importar un submódulo.
    (root / "torch" / "__init__.py").write_text(
        "__version__ = '9.9.9'\n"
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (root / "torch" / "version.py").write_text(
        "__version__ = '9.9.9'\n", encoding="utf-8"
    )
    (root / "torch" / "lib").mkdir(parents=True, exist_ok=True)
    (root / "torch" / "lib" / "torch_cuda.dll").write_bytes(b"MZ fake cuda")

    _write_pkg(root, "onnxruntime", "9.9.9", tag)
    (root / "onnxruntime" / "__init__.py").write_text(
        "__version__ = '9.9.9'\n"
        "def get_available_providers():\n"
        "    return ['CUDAExecutionProvider', 'CPUExecutionProvider']\n",
        encoding="utf-8",
    )
    (root / "onnxruntime" / "capi").mkdir(parents=True, exist_ok=True)
    (root / "onnxruntime" / "capi" / "onnxruntime_pybind11_state.pyd").write_bytes(
        b"MZ fake ort"
    )

    _write_pkg(root, "torchvision", "9.9.9", tag)
    return root


def _build(builder, store: Path, out: Path, **overrides):
    params = dict(
        pack_id="test-pack",
        part_limit=4 * 1024 * 1024,
        python_tag=_python_tag(),
        platform_name=sys.platform,
        arch=platform.machine(),
        min_driver_branch=0,
    )
    params.update(overrides)
    return builder.build_pack(store, out, **params)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_builder_splits_parts_and_hashes_them(builder, tmp_path):
    store = _make_store(tmp_path / "store")
    # Dos blobs de 600 KiB con un presupuesto de 1 MiB: no caben juntos.
    (store / "torch" / "blob-a.bin").write_bytes(b"a" * (600 * 1024))
    (store / "torch" / "blob-b.bin").write_bytes(b"b" * (600 * 1024))

    manifest = _build(
        builder, store, tmp_path / "out",
        part_limit=1024 * 1024,
    )

    assert len(manifest["parts"]) == 2
    assert manifest["installed_bytes"] > 1200 * 1024
    for part in manifest["parts"]:
        assert part["bytes"] < offline_pack.MAX_ASSET_BYTES
        path = tmp_path / "out" / part["name"]
        assert path.stat().st_size == part["bytes"]
        assert offline_pack._sha256_file(str(path)) == part["sha256"]

    written = json.loads(
        (tmp_path / "out" / offline_pack.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert written["id"] == "test-pack"
    assert written["python_tag"] == _python_tag()
    assert len(written["parts"]) == 2
    assert "torch/lib/torch_cuda.dll" in written["critical_files"]


def test_builder_refuses_a_store_without_cuda(builder, tmp_path):
    store = _make_store(tmp_path / "store")
    (store / "torch" / "lib" / "torch_cuda.dll").unlink()

    with pytest.raises(SystemExit) as excinfo:
        _build(builder, store, tmp_path / "out")
    assert "CUDA" in str(excinfo.value)


def test_builder_refuses_wheels_for_another_python(builder, tmp_path):
    store = _make_store(tmp_path / "store")
    dist = store / "torch-9.9.9.dist-info"
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nTag: cp99-cp99-win_amd64\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit) as excinfo:
        _build(builder, store, tmp_path / "out")
    assert "incompatible" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Validación del manifiesto
# ---------------------------------------------------------------------------


def _manifest_for(part_path: Path, **overrides) -> dict:
    digest = offline_pack._sha256_file(str(part_path))
    manifest = {
        "format": offline_pack.PACK_FORMAT,
        "id": "test",
        "platform": sys.platform,
        "arch": platform.machine(),
        "python_tag": _python_tag(),
        "backend": "cuda",
        "installed_bytes": 1,
        "parts": [
            {
                "name": part_path.name,
                "bytes": part_path.stat().st_size,
                "sha256": digest,
            }
        ],
        "_manifest_dir": str(part_path.parent),
        "_manifest_path": str(part_path.parent / offline_pack.MANIFEST_NAME),
    }
    manifest.update(overrides)
    return manifest


def test_validate_rejects_foreign_python_tag(tmp_path):
    part = tmp_path / "part.zip"
    with zipfile.ZipFile(part, "w") as zf:
        zf.writestr("torch/__init__.py", "")

    with pytest.raises(offline_pack.OfflinePackError) as excinfo:
        offline_pack.validate_manifest(_manifest_for(part, python_tag="cp99"))
    assert "Python" in str(excinfo.value)


def test_validate_rejects_asset_over_two_gib(tmp_path):
    (tmp_path / "part.zip").write_bytes(b"x")
    manifest = _manifest_for(tmp_path / "part.zip")
    manifest["parts"][0]["bytes"] = offline_pack.MAX_ASSET_BYTES
    manifest["parts"][0]["name"] = "part.zip"

    with pytest.raises(offline_pack.OfflinePackError) as excinfo:
        offline_pack.validate_manifest(manifest)
    assert "2 GiB" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Importación
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    base = tmp_path / "runtimes"
    store = base / "ai_modules"
    store.mkdir(parents=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", str(store))
    monkeypatch.setattr(rm, "get_bin_dir", lambda: str(tmp_path))
    rm._invalidate_store_probe_cache()
    yield store
    # El import inserta el store temporal en sys.path; no debe sobrevivir al
    # test (la carpeta se borra con tmp_path).
    while str(store) in sys.path:
        sys.path.remove(str(store))
    rm._invalidate_store_probe_cache()


def test_import_installs_pack_and_replaces_old_store(
    builder, isolated_store, tmp_path
):
    store = isolated_store
    # Store anterior: debe desaparecer entero tras el swap.
    (store / "torch").mkdir()
    (store / "torch" / "stale.txt").write_text("old", encoding="utf-8")

    built = _make_store(tmp_path / "built")
    out = tmp_path / "pack"
    _build(builder, built, out)
    manifest_path = out / offline_pack.MANIFEST_NAME
    assert manifest_path.is_file()

    result = offline_pack.import_pack(str(manifest_path))

    assert result["imported"] is True
    assert result["already_installed"] is False
    assert (store / "torch" / "__init__.py").is_file()
    assert (store / "onnxruntime" / "__init__.py").is_file()
    assert not (store / "torch" / "stale.txt").exists()
    assert offline_pack.installed_pack_info()["id"] == "test-pack"

    # El estado que verá la app sale del store recién importado y reporta GPU.
    torch_state = rm.torch_status()
    onnx_state = rm.onnx_status()
    assert torch_state["installed"] is True
    assert torch_state["gpu"] is True
    assert onnx_state["installed"] is True
    assert onnx_state["gpu"] is True


def test_import_is_idempotent(builder, isolated_store, tmp_path):
    built = _make_store(tmp_path / "built")
    out = tmp_path / "pack"
    _build(builder, built, out)
    manifest_path = out / offline_pack.MANIFEST_NAME

    first = offline_pack.import_pack(str(manifest_path))
    assert first["imported"] is True

    # Un testigo dentro del store: si la segunda llamada reimportara, el swap
    # lo borraría.
    sentinel = isolated_store / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    second = offline_pack.import_pack(str(manifest_path))
    assert second["already_installed"] is True
    assert second["imported"] is False
    assert sentinel.is_file()


def test_import_rejects_bad_hash_without_touching_store(
    builder, isolated_store, tmp_path
):
    built = _make_store(tmp_path / "built")
    out = tmp_path / "pack"
    manifest = _build(builder, built, out)

    part = out / manifest["parts"][0]["name"]
    payload = bytearray(part.read_bytes())
    payload[-1] ^= 0xFF
    part.write_bytes(bytes(payload))

    sentinel = isolated_store / "current.txt"
    sentinel.write_text("intact", encoding="utf-8")

    with pytest.raises(offline_pack.OfflinePackError) as excinfo:
        offline_pack.import_pack(str(out / offline_pack.MANIFEST_NAME))
    assert "SHA-256" in str(excinfo.value)
    assert sentinel.is_file()
    assert not (isolated_store / "torch").exists()


def test_import_rejects_path_traversal(isolated_store, tmp_path):
    part = tmp_path / "pack" / "part01.zip"
    part.parent.mkdir(parents=True)
    with zipfile.ZipFile(part, "w") as zf:
        zf.writestr("../escaped.txt", "no")
        zf.writestr("torch/__init__.py", "")

    manifest = _manifest_for(part)
    manifest["critical_files"] = []
    (part.parent / offline_pack.MANIFEST_NAME).write_text(
        json.dumps({k: v for k, v in manifest.items() if not k.startswith("_")}),
        encoding="utf-8",
    )

    with pytest.raises(offline_pack.OfflinePackError) as excinfo:
        offline_pack.import_pack(str(part))
    assert "insegura" in str(excinfo.value)
    assert not (tmp_path / "escaped.txt").exists()


def test_find_pack_honours_env_var(tmp_path, monkeypatch):
    manifest = tmp_path / "gpu-pack" / offline_pack.MANIFEST_NAME
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(offline_pack.ENV_PACK, str(manifest))

    assert offline_pack.find_pack() == str(manifest)


def test_pack_probe_source_is_valid_python():
    """La sonda se arma por concatenación: si se rompe, el import del pack
    falla siempre con un error de sintaxis del subproceso."""
    compile(offline_pack._PACK_PROBE_SOURCE, "<pack-probe>", "exec")
    assert "DRAGGY-CAPS=" in offline_pack._PACK_PROBE_SOURCE
    assert "add_dll_directory" in offline_pack._PACK_PROBE_SOURCE


def test_pack_supported_only_windows():
    assert offline_pack.pack_supported("win32") is True
    assert offline_pack.pack_supported("linux") is False
    assert offline_pack.pack_supported("darwin") is False


def test_find_pack_is_ignored_on_unsupported_platform(tmp_path, monkeypatch):
    manifest = tmp_path / offline_pack.MANIFEST_NAME
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(offline_pack.ENV_PACK, str(manifest))
    monkeypatch.setattr(offline_pack, "pack_supported", lambda: False)

    assert offline_pack.find_pack() is None


def test_import_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(offline_pack, "pack_supported", lambda: False)
    with pytest.raises(offline_pack.OfflinePackError) as excinfo:
        offline_pack.import_pack("does-not-matter.json")
    assert "Windows x64" in str(excinfo.value)


def test_offline_mode_flag(monkeypatch):
    monkeypatch.delenv(offline_pack.ENV_OFFLINE, raising=False)
    assert offline_pack.is_offline_mode() is False
    monkeypatch.setenv(offline_pack.ENV_OFFLINE, "1")
    assert offline_pack.is_offline_mode() is True
