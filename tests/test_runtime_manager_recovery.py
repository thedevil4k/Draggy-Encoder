"""Recuperación del runtime manager: purga de sys.modules, validación por
subproceso y guard anti-re-descarga.

Contexto (bug real, 2026-08-30): el primer import de torch dentro del
proceso de la app falló por algo transitorio; Python expulsó ``torch`` de
``sys.modules`` pero conservó ``torch.autograd``; el re-import del
diagnóstico re-ejecutó ``torch/__init__.py`` y, al reutilizar el
``torch.autograd`` huérfano sin re-adjuntar el atributo al paquete, revientó
con ``has no attribute 'autograd' (circular import)`` — un error SECUNDARIO
que ocultaba la causa real. Además, con la validación fallando, cada
arranque re-descargaba ~3 GB de PyTorch aunque el store ya estuviera sano.
Estos tests fijan el comportamiento de la cura.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import pytest

# ``conftest`` expone en ``_ORIGINALS`` las funciones que su fixture
# ``isolated_runtime`` bloquea (instaladores) y dobla (resolver), para que los
# tests que las ejercitan a propósito puedan usar las originales.
import conftest as _conftest

import src.runtimes.manager as rm


# ---------------------------------------------------------------------------
# _purge_stale_module_modules
# ---------------------------------------------------------------------------


def test_purge_stale_module_modules_clears_root_and_submodules(monkeypatch):
    monkeypatch.setitem(sys.modules, "fakepkg_sub", object())
    monkeypatch.setitem(sys.modules, "fakepkg_sub.mod", object())
    monkeypatch.setitem(sys.modules, "fakepkg_sub.other", object())
    # Un paquete ajeno NO debe tocarse.
    monkeypatch.setitem(sys.modules, "unrelated_pkg", object())

    purged = rm._purge_stale_module_modules("fakepkg_sub")
    assert purged == 3
    assert "fakepkg_sub" not in sys.modules
    assert "fakepkg_sub.mod" not in sys.modules
    assert "fakepkg_sub.other" not in sys.modules
    assert "unrelated_pkg" in sys.modules


# ---------------------------------------------------------------------------
# _can_import: purga + reintento
# ---------------------------------------------------------------------------


def _fake_spec_inside(store: str, name: str):
    class _Spec:
        origin = os.path.join(store, name, "__init__.py")

    return _Spec()


def test_can_import_purges_and_retries_after_transient_failure(monkeypatch, tmp_path):
    """Un primer import fallido deja submódulos huérfanos; _can_import debe
    purgarlos y reintentar una vez en lugar de reportar 'no instalado'."""
    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: _fake_spec_inside(store, name)
    )
    monkeypatch.setitem(sys.modules, "torch", object())
    monkeypatch.setitem(sys.modules, "torch.autograd", object())

    attempts: list[str] = []

    def fake_import(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("fallo transitorio (antivirus bloqueando el .pyd)")
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert rm._can_import("torch") is True
    assert attempts == ["torch", "torch"]  # exactamente UN reintento
    # Los huérfanos fueron purgados antes del reintento (el import falso no
    # vuelve a añadirlos).
    assert "torch" not in sys.modules
    assert "torch.autograd" not in sys.modules


def test_can_import_purges_again_when_retry_also_fails(monkeypatch, tmp_path):
    """Si el reintento también falla, el proceso debe quedar limpio (sin
    huérfanos) para que el siguiente intento (o el diagnóstico) no herede
    el artefacto de re-import."""
    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: _fake_spec_inside(store, name)
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", object())
    monkeypatch.setitem(sys.modules, "onnxruntime.capi", object())

    calls = {"n": 0}

    def always_fails(name):
        calls["n"] += 1
        raise ImportError(f"boom {calls['n']}")

    monkeypatch.setattr(importlib, "import_module", always_fails)

    assert rm._can_import("onnxruntime") is False
    assert calls["n"] == 2  # intento + UN reintento, nada más
    # Purga final: el proceso queda limpio pese al doble fallo.
    assert "onnxruntime" not in sys.modules
    assert "onnxruntime.capi" not in sys.modules


def test_can_import_does_not_purge_on_success(monkeypatch, tmp_path):
    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: _fake_spec_inside(store, name)
    )
    sentinel = object()
    monkeypatch.setitem(sys.modules, "happy_pkg", sentinel)
    monkeypatch.setattr(
        importlib, "import_module", lambda name: types.ModuleType(name)
    )
    assert rm._can_import("happy_pkg") is True
    # Camino feliz intacto: sin purgas innecesarias.
    assert sys.modules.get("happy_pkg") is sentinel


# ---------------------------------------------------------------------------
# _validate_store_import_subprocess
# ---------------------------------------------------------------------------


def _fake_subprocess_ns(run_fn):
    return types.SimpleNamespace(
        run=run_fn,
        TimeoutExpired=subprocess.TimeoutExpired,
        PIPE=subprocess.PIPE,
        STDOUT=subprocess.STDOUT,
    )


def test_probe_reports_ok_when_store_imports(monkeypatch, tmp_path):
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", str(tmp_path))
    monkeypatch.setattr(
        rm, "get_embed_python_path", lambda: "C:/fake/embed-python/python.exe"
    )

    def fake_run(cmd, **kwargs):
        assert cmd[1] == "-c"
        assert cmd[2] == rm._IMPORT_PROBE_SOURCE
        assert cmd[3] == str(tmp_path)
        assert cmd[4] == "torch"
        return types.SimpleNamespace(returncode=0, stdout="DRAGGY-IMPORT-OK\n", stderr="")

    monkeypatch.setattr(rm, "subprocess", _fake_subprocess_ns(fake_run))
    ok, detail = rm._validate_store_import_subprocess("torch")
    assert ok is True
    assert detail == ""


def test_probe_reports_real_stderr_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", str(tmp_path))
    monkeypatch.setattr(
        rm, "get_embed_python_path", lambda: "C:/fake/embed-python/python.exe"
    )

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "ModuleNotFoundError: No module named 'timeit'\n"

    monkeypatch.setattr(
        rm, "subprocess", _fake_subprocess_ns(lambda *a, **k: _Proc())
    )
    ok, detail = rm._validate_store_import_subprocess("torch")
    assert ok is False
    assert "ModuleNotFoundError" in detail
    assert "timeit" in detail


def test_probe_rejects_foreign_origin(monkeypatch, tmp_path):
    """Un paquete que importa desde fuera del store (venv del sistema) NO
    cuenta: la sonda verifica el origen ('caracol')."""
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", str(tmp_path))
    monkeypatch.setattr(rm, "get_embed_python_path", lambda: None)  # dev: sys.executable

    class _Proc:
        returncode = 4
        stdout = "DRAGGY-IMPORT-FOREIGN\n"
        stderr = ""

    monkeypatch.setattr(
        rm, "subprocess", _fake_subprocess_ns(lambda *a, **k: _Proc())
    )
    ok, detail = rm._validate_store_import_subprocess("torch")
    assert ok is False
    assert "DRAGGY-IMPORT-FOREIGN" in detail


def test_probe_reports_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", str(tmp_path))
    monkeypatch.setattr(
        rm, "get_embed_python_path", lambda: "C:/fake/embed-python/python.exe"
    )

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=5)

    monkeypatch.setattr(rm, "subprocess", _fake_subprocess_ns(_timeout))
    ok, detail = rm._validate_store_import_subprocess("torch", timeout=5)
    assert ok is False
    assert "agotó 5s" in detail


# ---------------------------------------------------------------------------
# _post_install_import_check (orden de veredicto: subproceso > in-process)
# ---------------------------------------------------------------------------


def test_post_install_check_trusts_subprocess_over_polluted_process(monkeypatch):
    monkeypatch.setattr(rm, "_can_import", lambda name: False)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(
        rm, "_validate_store_import_subprocess", lambda name, timeout=180.0: (True, "")
    )
    ok, detail = rm._post_install_import_check("torch")
    assert ok is True
    assert detail == ""


def test_post_install_check_reports_subprocess_stderr_on_real_failure(monkeypatch):
    monkeypatch.setattr(rm, "_can_import", lambda name: False)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(
        rm,
        "_validate_store_import_subprocess",
        lambda name, timeout=180.0: (False, "ImportError: DLL load failed"),
    )
    monkeypatch.setattr(
        rm, "_import_error_detail", lambda name: "AttributeError: artefacto in-process"
    )
    ok, detail = rm._post_install_import_check("torch")
    assert ok is False
    assert "DLL load failed" in detail  # la causa real manda
    assert "Diagnóstico del proceso actual" in detail  # contexto al final


def test_post_install_check_falls_back_in_process_without_probe(monkeypatch):
    monkeypatch.setattr(rm, "_can_import", lambda name: False)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: False)
    monkeypatch.setattr(
        rm, "_import_error_detail", lambda name: "ModuleNotFoundError: boom"
    )
    ok, detail = rm._post_install_import_check("torch")
    assert ok is False
    assert "ModuleNotFoundError" in detail


# ---------------------------------------------------------------------------
# torch_status / onnx_status: guard anti re-descarga
# ---------------------------------------------------------------------------


def _make_pkg_dir(store: str, name: str, version: str) -> None:
    pkg = os.path.join(store, name)
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as fh:
        fh.write(f"__version__ = '{version}'\n")


def _fake_pip_stage(destination: str) -> None:
    """Simula lo que pip deja en el staging tras una instalación correcta.

    Hace falta porque el instalador valida el staging ANTES de hacer el swap:
    un doble de pip que no escriba nada haría fallar esa validación (que es
    justo lo que debe pasar cuando la instalación no produce archivos).
    """
    for name in ("torch", "torchvision", "torchaudio", "onnx", "onnxscript"):
        os.makedirs(os.path.join(destination, name), exist_ok=True)
        with open(
            os.path.join(destination, name, "__init__.py"), "w", encoding="utf-8"
        ) as fh:
            fh.write("__version__ = '0.0.0-test'\n")


def test_torch_status_trusts_valid_store_and_skips_reinstall(monkeypatch, tmp_path):
    """El bug del arranque eterno: store válido pero import in-process
    fallando → installed=True (sin re-descarga de ~3 GB por arranque)."""
    store = str(tmp_path)
    _make_pkg_dir(store, "torch", "2.13.0-fake")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(rm, "_can_import", lambda name: False)  # proceso "contaminado"
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(rm, "_subprocess_import_ok", lambda name: (True, ""))

    st = rm.torch_status()
    assert st["installed"] is True
    assert st["local"] is True


def test_torch_status_stays_missing_when_subprocess_also_fails(monkeypatch, tmp_path):
    store = str(tmp_path)
    _make_pkg_dir(store, "torch", "broken")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(rm, "_can_import", lambda name: False)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(rm, "_subprocess_import_ok", lambda name: (False, "boom"))

    st = rm.torch_status()
    assert st["installed"] is False


def test_torch_status_skips_probe_when_store_has_no_package(monkeypatch, tmp_path):
    """Store vacío (solo torch del sistema): ni siquiera se debe sondear en
    subproceso — el fallo es de ausencia, no de entorno."""
    store = str(tmp_path)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    probe_calls: list[str] = []
    monkeypatch.setattr(
        rm, "_subprocess_import_ok", lambda name: probe_calls.append(name) or (True, "")
    )

    st = rm.torch_status()
    assert st["installed"] is False
    assert probe_calls == []


def test_onnx_status_trusts_valid_store_and_skips_reinstall(monkeypatch, tmp_path):
    store = str(tmp_path)
    _make_pkg_dir(store, "onnxruntime", "1.29.0-fake")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(rm, "_can_import", lambda name: False)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(rm, "_subprocess_import_ok", lambda name: (True, ""))

    st = rm.onnx_status()
    assert st["installed"] is True


# ---------------------------------------------------------------------------
# Instaladores: el detalle de error viene del subproceso (causa real)
# ---------------------------------------------------------------------------


def test_install_pytorch_raises_with_subprocess_stderr(monkeypatch, tmp_path, isolated_runtime):
    store = str(tmp_path)
    os.makedirs(store, exist_ok=True)
    _make_pkg_dir(store, "torch", "broken")
    with open(
        os.path.join(store, "torch", "__init__.py"), "w", encoding="utf-8"
    ) as fh:
        fh.write("raise ImportError('paquete roto a propósito')\n")

    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(rm, "_preflight_connectivity", lambda: {"summary": "ok"})
    monkeypatch.setattr(rm, "_check_disk_space", lambda kind, variant: None)
    monkeypatch.setattr(rm, "_purge_foreign_version_wheels", lambda: 0)
    monkeypatch.setattr(rm, "_run_pip_checked", lambda *a, **k: None)
    monkeypatch.setattr(rm, "_purge_stale_module_modules", lambda name: 0)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    # El staging debe producir contenido válido para llegar al veredicto final.
    monkeypatch.setattr(
        rm, "_merge_missing_from_base", lambda destination: _fake_pip_stage(destination)
    )
    monkeypatch.setattr(rm, "_validate_package_in_dir", lambda name, target, timeout=180.0: (True, ""))
    # El veredicto post-instalación es lo que este test fija: el store no
    # importa aunque pip terminó bien. Se dobla directamente para no depender
    # de cómo resuelve el import en el entorno de tests.
    monkeypatch.setattr(
        rm,
        "_post_install_import_check",
        lambda name: (False, "ImportError: DLL load failed while importing torch_cpu"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _conftest._ORIGINALS["install_pytorch"]("cuda126", progress_callback=None)
    message = str(excinfo.value)
    assert "DLL load failed" in message          # stderr real del subproceso
    assert "variante CPU" in message             # hint CUDA→CPU sigue activo


def test_install_onnxruntime_raises_with_subprocess_stderr(monkeypatch, tmp_path, isolated_runtime):
    store = str(tmp_path)
    os.makedirs(store, exist_ok=True)
    _make_pkg_dir(store, "onnxruntime", "broken")
    with open(
        os.path.join(store, "onnxruntime", "__init__.py"), "w", encoding="utf-8"
    ) as fh:
        fh.write("raise ImportError('paquete roto a propósito')\n")

    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(rm, "_preflight_connectivity", lambda: {"summary": "ok"})
    monkeypatch.setattr(rm, "_check_disk_space", lambda kind, variant: None)
    monkeypatch.setattr(rm, "_purge_foreign_version_wheels", lambda: 0)
    monkeypatch.setattr(
        rm, "_install_onnxruntime_from_wheel", lambda *a, **k: None
    )
    monkeypatch.setattr(
        rm, "_merge_missing_from_base", lambda destination: _fake_pip_stage(destination)
    )
    monkeypatch.setattr(
        rm, "_validate_package_in_dir", lambda name, target, timeout=180.0: (True, "")
    )
    monkeypatch.setattr(rm, "_purge_stale_module_modules", lambda name: 0)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(
        rm,
        "_validate_store_import_subprocess",
        lambda name, timeout=180.0: (False, "ModuleNotFoundError: No module named 'x'"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _conftest._ORIGINALS["install_onnxruntime"]("cpu", progress_callback=None)
    assert "ModuleNotFoundError" in str(excinfo.value)


def test_install_pytorch_success_when_subprocess_trusts_store(monkeypatch, tmp_path, isolated_runtime):
    """Store 'contaminado' para el proceso actual pero válido (subproceso OK):
    la instalación debe REPORTAR ÉXITO, no reintentar ni re-descargar."""
    store = str(tmp_path)
    _make_pkg_dir(store, "torch", "2.13.0-fake")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(rm, "_preflight_connectivity", lambda: {"summary": "ok"})
    monkeypatch.setattr(rm, "_check_disk_space", lambda kind, variant: None)
    monkeypatch.setattr(rm, "_purge_foreign_version_wheels", lambda: 0)
    monkeypatch.setattr(rm, "_run_pip_checked", lambda *a, **k: None)
    monkeypatch.setattr(
        rm, "_merge_missing_from_base", lambda destination: _fake_pip_stage(destination)
    )
    monkeypatch.setattr(
        rm, "_validate_package_in_dir", lambda name, target, timeout=180.0: (True, "")
    )
    monkeypatch.setattr(rm, "_purge_stale_module_modules", lambda name: 0)
    monkeypatch.setattr(rm, "_can_import", lambda name: False)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(
        rm, "_validate_store_import_subprocess", lambda name, timeout=180.0: (True, "")
    )
    assert _conftest._ORIGINALS["install_pytorch"]("cuda126", progress_callback=None) == "cuda126"


# ---------------------------------------------------------------------------
# El comando pip sale del plan resuelto, no de un pin fijo
# ---------------------------------------------------------------------------


def test_pytorch_command_pins_the_resolved_plan():
    """El comando lleva EXACTAMENTE las versiones del plan.

    Antes se enviaba un pin estático (torch==2.13.0) a un índice elegido por
    heurística; cuando esa pareja no existía, pip fallaba con "Could not find
    a version that satisfies the requirement". Ahora el plan es la fuente de
    verdad y esto fija ese contrato.
    """
    plan = {
        "variant": "cuda126",
        "versions": {"torch": "2.14.0", "torchvision": "0.29.0", "torchaudio": "2.11.0"},
        "source": "index",
        "note": "",
        "attempts": [],
    }
    cmd = rm._build_pytorch_command("cuda126", plan)
    assert "torch==2.14.0" in cmd
    assert "torchvision==0.29.0" in cmd
    assert "torchaudio==2.11.0" in cmd
    assert "onnx" in cmd
    assert "onnxscript" in cmd
    assert "https://download.pytorch.org/whl/cu126" in cmd


def test_pytorch_command_falls_back_to_known_good_set_without_versions():
    """Sin versiones resueltas (plan offline) se usa el set conocido bueno,
    que es lo único que se puede hacer sin red."""
    plan = {"variant": "cuda126", "versions": {}, "source": "offline",
            "note": "", "attempts": []}
    cmd = rm._build_pytorch_command("cuda126", plan)
    assert rm._TORCH_PIN in cmd
    assert rm._TORCHVISION_PIN in cmd
    assert rm._TORCHAUDIO_PIN in cmd


def test_pytorch_command_target_can_be_overridden(tmp_path):
    """El instalador usa un directorio de staging: --target debe poder
    cambiarse o no habría validación previa al swap."""
    plan = {"variant": "cpu", "versions": {}, "source": "offline",
            "note": "", "attempts": []}
    cmd = rm._build_pytorch_command("cpu", plan, target=str(tmp_path))
    assert cmd[cmd.index("--target") + 1] == str(tmp_path)


# ---------------------------------------------------------------------------
# Resolución dinámica de variante y versión (el bug original)
#
# Síntoma real: "Could not find a version that satisfies the requirement
# torch==2.13.0" / "pip exited with code 1". Causa: se fijaba la versión a un
# pin estático y el índice se elegía con una heurística por versión de Python,
# sin comprobar que esa pareja existiera. cu121 (elegido para Python 3.9-3.12)
# no publica NINGUNA versión de ese pin.
# ---------------------------------------------------------------------------


def _real_resolver(*args, **kwargs):
    """Invoca el resolver REAL aunque el fixture lo haya doblado.

    El fixture ``isolated_runtime`` fija el plan offline para que el resto de
    la suite no haga red; estos tests existen precisamente para ejercitar la
    resolución, así que usan la función original que el propio fixture deja en
    ``conftest._ORIGINALS`` y le pasan índices doblados.
    """
    return _conftest._ORIGINALS["resolve_pytorch_plan"](*args, **kwargs)


# Réplica reducida de los índices reales: qué wheels sirve cada uno.
_FAKE_INDICES = {
    "https://download.pytorch.org/whl/cu121": {
        "torch": ["2.5.1"], "torchvision": ["0.20.1"], "torchaudio": ["2.5.1"],
    },
    "https://download.pytorch.org/whl/cu126": {
        "torch": ["2.13.0", "2.14.0"], "torchvision": ["0.28.0", "0.29.0"],
        "torchaudio": ["2.11.0"],
    },
    "https://download.pytorch.org/whl/cu130": {
        "torch": ["2.14.0"], "torchvision": ["0.29.0"], "torchaudio": ["2.11.0"],
    },
    "https://download.pytorch.org/whl/cpu": {
        "torch": ["2.14.0"], "torchvision": ["0.29.0"], "torchaudio": ["2.11.0"],
    },
}

# Qué torch exige cada torchvision/torchaudio de PyPI (no comparten número de
# versión con torch: torchaudio 2.11.0 va con torch 2.14.0).
_FAKE_TORCH_PINS = {
    ("torchvision", "0.28.0"): "2.13.0",
    ("torchvision", "0.29.0"): "2.14.0",
    ("torchaudio", "2.11.0"): "2.14.0",
    ("torchaudio", "2.10.0"): "2.13.0",
}
_FAKE_PYPI = {
    "torchvision": ["0.29.0", "0.28.0"],
    "torchaudio": ["2.11.0", "2.10.0"],
}


def _fake_index(monkeypatch, *, python_floor=None, indices=None):
    """Dobla los índices y PyPI para resolver sin red de forma determinista.

    ``indices`` permite usar una tabla concreta sin mutar la compartida (una
    mutación se filtraba a los tests siguientes).
    """
    table = _FAKE_INDICES if indices is None else indices

    def _wheels(index_url, package):
        versions = table.get(index_url, {}).get(package, [])
        # (version con sufijo local, nombre de wheel) como en el índice real,
        # con el tag del intérprete que ejecuta los tests.
        return [
            (f"{v}+x", f"{package}-{v}-{rm._CP_TAG}-{rm._CP_TAG}-win_amd64.whl")
            for v in versions
        ]

    monkeypatch.setattr(rm, "_index_wheel_list", _wheels)
    monkeypatch.setattr(
        rm, "_pypi_release_requires_python",
        lambda package, version: python_floor,
    )
    monkeypatch.setattr(
        rm, "_pypi_versions_newest_first", lambda package: _FAKE_PYPI.get(package, [])
    )

    def fake_requires(package, version, dependency):
        return _FAKE_TORCH_PINS.get((package, version))

    monkeypatch.setattr(rm, "_pypi_requires_version", fake_requires)
    monkeypatch.setattr(rm, "_gpu_preferred_variant", lambda: "cuda121")
    monkeypatch.setattr(rm, "_variant_driver_ok", lambda candidate: True)
    monkeypatch.setattr(rm, "_has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(rm, "_platform_compatible_index", lambda: True)


def test_resolver_skips_index_without_wheels_for_this_python(monkeypatch):
    """El bug original: con GPU NVIDIA y Python 3.9-3.12, la app elegía cu121 y
    pedía un torch que cu121 no publica para ese CPython.

    El resolver debe DESCARTAR ese índice (no tiene wheels usables) y resolver
    contra el siguiente candidato con set completo, en vez de mandar a pip una
    combinación imposible.
    """
    # cu121 y cu130 sin wheels usables para este intérprete (lo que ocurría de
    # verdad en Python 3.13/3.14, y con el pin congelado también en 3.9-3.12).
    # cu126 es el único que sirve un set completo, así que debe ser el elegido.
    indices = {url: {} for url in _FAKE_INDICES}
    indices["https://download.pytorch.org/whl/cu126"] = dict(_FAKE_INDICES[
        "https://download.pytorch.org/whl/cu126"
    ])
    _fake_index(monkeypatch, python_floor=None, indices=indices)
    monkeypatch.setattr(rm, "_gpu_preferred_variant", lambda: "cuda121")

    plan = _real_resolver(use_cache=False)

    assert plan["variant"] == "cuda126", plan
    # El set resuelto es coherente consigo mismo (no mezcla familias): la
    # torchvision elegida es la que declara necesitar esa torch exacta.
    assert plan["versions"]["torch"] == "2.14.0"
    assert plan["versions"]["torchvision"] == "0.29.0"
    rejections = {a["variant"]: a["reason"] for a in plan["attempts"]}
    assert "cuda121" in rejections
    assert "Python" in rejections["cuda121"]


def test_resolver_uses_pypi_when_index_lacks_a_package(monkeypatch):
    """torchaudio CUDA no siempre está en el índice de PyTorch; hay que
    buscarlo en PyPI por la torch que declara necesitar, no por número de
    versión (torchaudio 2.11.0 acompaña a torch 2.14.0)."""
    indices = dict(_FAKE_INDICES)
    indices["https://download.pytorch.org/whl/cu130"] = {
        "torch": ["2.14.0"], "torchvision": ["0.29.0"],  # sin torchaudio
    }
    monkeypatch.setattr(rm, "_FAKE_INDICES_UNUSED", indices, raising=False)

    def fake_wheel_list(index_url, package):
        versions = indices.get(index_url, {}).get(package, [])
        return [(f"{v}+x", f"{package}-{v}-{rm._CP_TAG}-{rm._CP_TAG}-win_amd64.whl")
                for v in versions]

    monkeypatch.setattr(rm, "_index_wheel_list", fake_wheel_list)
    monkeypatch.setattr(rm, "_pypi_release_requires_python", lambda p, v: None)
    monkeypatch.setattr(
        rm, "_pypi_versions_newest_first", lambda package: _FAKE_PYPI.get(package, [])
    )
    monkeypatch.setattr(
        rm, "_pypi_requires_version",
        lambda package, version, dependency: _FAKE_TORCH_PINS.get((package, version)),
    )
    monkeypatch.setattr(rm, "_gpu_preferred_variant", lambda: "cuda130")
    monkeypatch.setattr(rm, "_platform_compatible_index", lambda: True)

    plan = _real_resolver(use_cache=False)

    assert plan["variant"] == "cuda130"
    assert plan["versions"]["torch"] == "2.14.0"
    assert plan["versions"]["torchaudio"] == "2.11.0"
    assert "torchaudio desde PyPI" in plan["note"]


def test_resolver_reports_actionable_error_when_nothing_fits(monkeypatch):
    """Si ningún canal sirve, el error debe decir QUÉ se probó y por qué, no
    el escueto "pip exited with code 1"."""
    # Índices vacíos SIN mutar la tabla compartida (mutarla contaminaba los
    # tests siguientes: veían cu121 con wheels que ya no existían).
    monkeypatch.setattr(rm, "_FAKE_INDICES", {url: {} for url in _FAKE_INDICES}, raising=False)
    _fake_index(monkeypatch, python_floor=None, indices={url: {} for url in _FAKE_INDICES})
    monkeypatch.setattr(rm, "_has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(rm, "_gpu_preferred_variant", lambda: "cpu")

    with pytest.raises(rm.PytorchPlanError) as excinfo:
        _real_resolver(use_cache=False)
    message = str(excinfo.value)
    assert "No se encontró ninguna versión de PyTorch instalable" in message
    assert "Se probaron estos canales" in message
    assert "Actualiza Python" in message
    assert excinfo.value.attempts  # lista de intentos para la UI


def test_resolver_rejects_python_below_pytorch_floor(monkeypatch):
    """Por debajo del mínimo de PyTorch ni se consulta: error claro y directo.

    Se dobla ``_python_short`` (lo que usa el mensaje) y ``version_info`` en el
    módulo del manager, no en el ``sys`` global, que es singleton del proceso.
    """
    monkeypatch.setattr(rm, "_python_short", lambda: "3.9")
    monkeypatch.setattr(rm.sys, "version_info", _FakeVersionInfo(3, 9))
    with pytest.raises(rm.PytorchPlanError) as excinfo:
        _real_resolver(use_cache=False)
    assert "Python 3.9" in str(excinfo.value)
    assert "3.10" in str(excinfo.value)


class _FakeVersionInfo(tuple):
    """``sys.version_info`` de mentira con los atributos que usa el manager."""

    def __new__(cls, major: int, minor: int):
        return super().__new__(cls, (major, minor, 0, "final", 0))

    @property
    def major(self) -> int:
        return self[0]

    @property
    def minor(self) -> int:
        return self[1]


def test_resolver_prefers_newest_when_python_allows(monkeypatch):
    """En un Python moderno se instala lo más nuevo disponible, no un pin
    congelado: el resolver consulta el índice."""
    # Sin suelo de Python: todas las versiones del índice son usables.
    _fake_index(monkeypatch)
    monkeypatch.setattr(rm, "_gpu_preferred_variant", lambda: "cuda130")

    plan = _real_resolver(use_cache=False)

    assert plan["variant"] == "cuda130"
    assert plan["versions"] == {
        "torch": "2.14.0", "torchvision": "0.29.0", "torchaudio": "2.11.0",
    }
    assert rm.pytorch_requirements(plan) == [
        "torch==2.14.0", "torchvision==0.29.0", "torchaudio==2.11.0",
    ]


def test_resolver_falls_back_to_offline_set_when_network_is_down(monkeypatch):
    """Sin red no se puede resolver, pero la app no debe quedarse sin intento:
    se usa el set conocido bueno y pip dará el error real si tampoco sirve."""
    _fake_index(monkeypatch, python_floor=None, indices={url: {} for url in _FAKE_INDICES})
    monkeypatch.setattr(rm, "_gpu_preferred_variant", lambda: "cpu")
    monkeypatch.setattr(rm, "run_connectivity_probe", lambda: {"ok": False})

    plan = _real_resolver(use_cache=False)

    assert plan["source"] == "offline"
    assert plan["versions"] == {}
    assert rm.pytorch_requirements(plan) == [
        rm._TORCH_PIN, rm._TORCHVISION_PIN, rm._TORCHAUDIO_PIN,
    ]


# ---------------------------------------------------------------------------
# Parseo de los índices (URLs con %2B y fragmento #sha256)
# ---------------------------------------------------------------------------


def test_wheel_version_handles_percent_encoded_local_segment():
    """Los índices percent-codifican el '+' del segmento local. Trocear por
    '-' daba la versión equivocada ('cp313') y el resolver no encontraba
    nada: el índice parecía vacío."""
    assert rm._wheel_version(
        "torch-2.13.0%2Bcu130-cp313-cp313-win_amd64.whl"
    ) == "2.13.0+cu130"
    assert rm._wheel_version("torch-2.13.0-cp313-cp313-win_amd64.whl") == "2.13.0"
    assert rm._public_version("2.13.0+cu130") == "2.13.0"
    assert rm._version_key("2.13.0+cu130") == (2, 13, 0)
    assert rm._version_key("2.9.1") > rm._version_key("2.9.0")


def test_index_wheel_list_parses_sha256_fragment(monkeypatch):
    """El href puede ser absoluto y llevar '#sha256=...' entre el nombre y la
    comilla: ignorarlo devolvía cero wheels para páginas enteras."""
    tag = rm._CP_TAG
    html = (
        '<a href="https://download-r2.pytorch.org/whl/cu130/'
        f'torch-2.14.0%2Bcu130-{tag}-{tag}-win_amd64.whl#sha256=abc" '
        'data-upload-time="2026-01-01">torch</a>'
        '<a href="https://download-r2.pytorch.org/whl/cu130/'
        f'torch-2.14.0%2Bcu130-{tag}-{tag}-linux_x86_64.whl#sha256=def">torch</a>'
    )
    monkeypatch.setattr(rm, "_http_get_text", lambda url, timeout=30.0: html)
    rm._wheel_list_cache.clear()

    found = rm._index_wheel_list("https://download.pytorch.org/whl/cu130", "torch")

    if sys.platform == "win32":
        assert found == [(
            "2.14.0+cu130",
            f"torch-2.14.0%2Bcu130-{tag}-{tag}-win_amd64.whl",
        )]
    else:
        # El wheel de Windows nunca debe colarse en otra plataforma.
        assert all("win_amd64" not in name for _v, name in found)


# ---------------------------------------------------------------------------
# Swap del store: reemplazo sin duplicados y con reciclaje
# ---------------------------------------------------------------------------


def _write_pkg(root: str, name: str, version: str) -> str:
    pkg = os.path.join(root, name)
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as fh:
        fh.write(f"__version__ = '{version}'\n")
    dist = os.path.join(root, f"{name}-{version}.dist-info")
    os.makedirs(dist, exist_ok=True)
    with open(os.path.join(dist, "METADATA"), "w", encoding="utf-8") as fh:
        fh.write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return pkg


def test_commit_staging_replaces_packages_and_recycles_the_rest(monkeypatch, tmp_path):
    """El escenario real del store: torch viejo + numpy + dist-info huérfano
    de una instalación anterior.

    Tras instalar torch nuevo, el store debe tener EXACTAMENTE una versión de
    torch (sin metadatos duplicados), conservar numpy (que la instalación
    nueva no trae) y no perder nada más.
    """
    base = tmp_path / "runtimes"
    store = str(base / "ai_modules")
    staging = str(base / ".staging")
    aside = str(base / ".previous")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)

    # Estado anterior del store.
    _move_target = store
    _write_pkg(store, "torch", "2.13.0")
    _write_pkg(store, "numpy", "2.5.2")
    # Huérfano de una instalación previa (el bug de pip --target --upgrade).
    os.makedirs(os.path.join(store, "numpy-2.4.4.dist-info"), exist_ok=True)

    rm._move_store_to_aside()
    assert not os.path.isdir(store) or rm._store_entries(store) == set()

    # Nueva instalación (lo que pip habría dejado en staging).
    os.makedirs(staging, exist_ok=True)
    _write_pkg(staging, "torch", "2.14.0")
    _write_pkg(staging, "torchvision", "0.29.0")

    rm._commit_staging(("torch", "torchvision", "torchaudio", "onnx", "onnxscript"))

    entries = set(os.listdir(store))
    # Una sola versión de torch y solo la nueva.
    torch_dists = sorted(e for e in entries if e.startswith("torch-") and e.endswith(".dist-info"))
    assert torch_dists == ["torch-2.14.0.dist-info"], entries
    assert not os.path.exists(os.path.join(store, "torch-2.13.0.dist-info"))
    # numpy se conserva (la instalación nueva no lo trae) y su duplicado
    # huérfano también, porque no pertenece a los paquetes reemplazados.
    assert "numpy" in entries
    assert "numpy-2.5.2.dist-info" in entries
    assert _move_target == store


def test_commit_staging_removes_orphan_metadata_of_replaced_package(monkeypatch, tmp_path):
    """Los dist-info huérfanos del paquete reemplazado SÍ deben desaparecer:
    son los que hacían que la app reportara una versión que no estaba."""
    base = tmp_path / "runtimes"
    store = str(base / "ai_modules")
    staging = str(base / ".staging")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)

    _write_pkg(store, "torch", "2.13.0")
    os.makedirs(os.path.join(store, "torch-2.12.0.dist-info"), exist_ok=True)
    os.makedirs(os.path.join(store, "torch.libs"), exist_ok=True)

    rm._move_store_to_aside()
    os.makedirs(staging, exist_ok=True)
    _write_pkg(staging, "torch", "2.14.0")
    rm._commit_staging(("torch",))

    entries = set(os.listdir(store))
    assert "torch-2.12.0.dist-info" not in entries
    assert "torch-2.13.0.dist-info" not in entries
    assert "torch-2.14.0.dist-info" in entries
    # El directorio de datos compilados del paquete también se reemplaza, no
    # se mezcla con el de la versión nueva.
    assert "torch.libs" not in entries or os.path.isdir(os.path.join(store, "torch.libs"))


def test_commit_staging_without_files_is_an_error(monkeypatch, tmp_path):
    """Una instalación que no produjo archivos no debe 'tener éxito' ni
    vaciar el store: es exactamente el caso que dejaba el runtime roto."""
    base = tmp_path / "runtimes"
    store = str(base / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    _write_pkg(store, "torch", "2.13.0")

    rm._move_store_to_aside()
    with pytest.raises(RuntimeError) as excinfo:
        rm._commit_staging(("torch",))
    assert "no generó archivos" in str(excinfo.value)
    # El store anterior sigue disponible para restaurar.
    assert os.path.isdir(rm._aside_dir())


def test_cleanup_install_leftovers_removes_staging_and_previous(monkeypatch, tmp_path):
    """Restos de un cierre a mitad de swap: un staging sin validar jamás debe
    sobrevivir (podría confundirse con el store) y un store anterior completo
    tampoco, porque ya no se necesita."""
    base = tmp_path / "runtimes"
    store = str(base / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    os.makedirs(rm._staging_dir(), exist_ok=True)
    # ``.previous`` con contenido de un swap de verdad: el paquete torch y sus
    # metadatos (lo que deja el instalador al apartar el store anterior).
    _write_pkg(rm._aside_dir(), "torch", "2.13.0")

    cleaned = rm.cleanup_install_leftovers()

    assert cleaned == 2
    assert not os.path.isdir(rm._staging_dir())
    assert not os.path.isdir(rm._aside_dir())
    assert os.path.isdir(store)


def test_cleanup_install_leftovers_keeps_unrecognised_previous(monkeypatch, tmp_path):
    """``.previous`` que NO parece un store instalado no se borra.

    Guarda nacida de un fallo real: durante el desarrollo, un test con el store
    redirigido a un temporal dejó contenido ajeno en ``.previous`` y el
    arranque lo borró. El árbol ``bin/`` no está en git, así que no había
    vuelta atrás. Ante la duda, no se toca.
    """
    base = tmp_path / "runtimes"
    store = str(base / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    aside = rm._aside_dir()
    os.makedirs(os.path.join(aside, "documentos_importantes"), exist_ok=True)

    cleaned = rm.cleanup_install_leftovers()

    # ``cleaned`` cuenta los restos que YA no están (aquí solo el staging, que
    # el test no había creado, así que 0). Lo importante es que ``.previous``
    # se respeta.
    assert cleaned == 0
    assert os.path.isdir(aside), "contenido no reconocido: no debe borrarse"


# ---------------------------------------------------------------------------
# Desinstalación por runtime (el store es compartido)
# ---------------------------------------------------------------------------


def _seed_shared_store(store: str) -> None:
    """Store con PyTorch + ONNX Runtime + dependencias compartidas."""
    for name in ("torch", "torchvision", "torchaudio", "torchgen", "onnx", "onnxscript",
                 "onnx_ir", "onnxruntime", "numpy", "sympy", "protobuf", "packaging"):
        _write_pkg(store, name, "1.0.0")


def test_uninstall_runtime_only_removes_its_own_packages(monkeypatch, tmp_path):
    """Desinstalar ONNX Runtime NO debe llevarse PyTorch.

    El store es compartido y la UI ofrece un botón "Uninstall" por fila. Antes
    ambas filas compartían un borrado del store COMPLETO: pulsar Uninstall en
    ONNX (o en PyTorch) se llevaba los dos runtimes y todas las dependencias
    compartidas de golpe, sin avisar.
    """
    store = str(tmp_path / "runtimes" / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    _seed_shared_store(store)

    complete = rm.uninstall_runtime("onnxruntime")

    assert complete is True
    entries = os.listdir(store)
    assert not any(e.startswith("onnxruntime") for e in entries), entries
    # El otro runtime y lo compartido siguen intactos.
    for kept in ("torch", "torchvision", "torchaudio", "numpy", "sympy"):
        assert kept in entries, kept
    assert rm._store_has_package_dir("torch")


def _gone(name: str, entries: list[str]) -> bool:
    """True si ni el paquete ni sus metadatos están en el store.

    Compara exacto (normalizando el nombre canónico) en vez de por prefijo:
    ``startswith("onnx")`` daría positivo con ``onnxruntime``.
    """
    prefix = name.replace("-", "_")
    for entry in entries:
        if entry == name or entry == prefix:
            return False
        if entry.endswith(".dist-info"):
            head = entry[: -len(".dist-info")].split("-", 1)[0].replace("-", "_")
            if head == prefix:
                return False
    return True


def test_uninstall_runtime_onnx_also_removes_the_onnx_stack(monkeypatch, tmp_path):
    """El stack ``onnx``/``onnxscript``/``onnx_ir`` se va con CUALQUIERA de los
    dos runtimes: solo sirve para exportar ONNX desde PyTorch y para
    ejecutarlo, así que sin uno de los dos es peso muerto."""
    store = str(tmp_path / "runtimes" / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    _seed_shared_store(store)

    rm.uninstall_runtime("onnxruntime")

    entries = os.listdir(store)
    for gone in ("onnxruntime", "onnx", "onnxscript", "onnx_ir"):
        assert _gone(gone, entries), (gone, entries)
    for kept in ("torch", "torchvision", "numpy", "sympy"):
        assert not _gone(kept, entries), (kept, entries)


def test_uninstall_runtime_torch_removes_its_satellites(monkeypatch, tmp_path):
    """PyTorch se lleva sus satélites (onnx/onnxscript, que instala él mismo)
    pero respeta ONNX Runtime y las dependencias compartidas."""
    store = str(tmp_path / "runtimes" / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    _seed_shared_store(store)

    complete = rm.uninstall_runtime("torch")

    assert complete is True
    entries = os.listdir(store)
    for gone in ("torch", "torchvision", "torchaudio", "torchgen", "onnx",
                 "onnxscript", "onnx_ir"):
        assert _gone(gone, entries), (gone, entries)
    for kept in ("onnxruntime", "numpy", "sympy", "protobuf"):
        assert not _gone(kept, entries), (kept, entries)


def test_uninstall_runtime_removes_dist_info_metadata_too(monkeypatch, tmp_path):
    """Sin borrar los ``*.dist-info`` la app seguiría reportando el runtime
    como instalado (y con la versión vieja)."""
    store = str(tmp_path / "runtimes" / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    _seed_shared_store(store)
    # Dist-info huérfano de una instalación anterior del mismo paquete: el
    # nombre de la distribución (onnxruntime_gpu) no coincide con el prefijo.
    os.makedirs(os.path.join(store, "onnxruntime_gpu-1.29.0.dist-info"))

    rm.uninstall_runtime("onnxruntime")

    assert _gone("onnxruntime", os.listdir(store))
    assert _gone("onnxruntime_gpu", os.listdir(store))


def test_uninstall_runtime_unknown_id_is_rejected(monkeypatch, tmp_path):
    store = str(tmp_path / "runtimes" / "ai_modules")
    os.makedirs(store, exist_ok=True)
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)

    with pytest.raises(RuntimeError):
        rm.uninstall_runtime("no-existe")


def test_registry_uses_per_runtime_uninstall():
    """El REGISTRY (lo que ofrece la UI) no puede volver a compartir el borrado
    total del store entre las dos filas de runtime."""
    import src.dependencies as dependencies
    import src.runtimes.manager as rm

    fns = {
        e["id"]: e.get("uninstall_fn")
        for e in dependencies.REGISTRY
        if e.get("kind") == dependencies.KIND_RUNTIME
    }
    assert set(fns) == {"torch", "onnxruntime"}
    assert fns["torch"] is not fns["onnxruntime"]
    # Y ninguna de las dos borra el store completo.
    assert fns["torch"] is not rm.uninstall_local_runtimes
    assert fns["onnxruntime"] is not rm.uninstall_local_runtimes


def test_onnx_and_torch_install_uninstall_are_symmetric():
    """Lo que el desinstalador se lleva, el instalador lo repone.

    Regresión real: al desinstalar ONNX Runtime se va el stack
    ``onnx``/``onnxscript`` (peso muerto sin ningún runtime), pero su
    instalador solo bajaba ``onnxruntime``. El usuario que desinstalaba y
    reinstalaba perdía la capacidad de convertir ``.pth`` a ONNX sin ningún
    aviso: ``torch.onnx.export`` fallaba con "The Python package 'onnx' is
    missing".
    """
    import inspect

    import src.runtimes.manager as rm

    onnx_owner = {
        entry
        for entry in rm._runtime_group_prefixes("onnxruntime")
    }
    torch_owner = set(rm._runtime_group_prefixes("torch"))
    assert "onnx" in onnx_owner and "onnxscript" in onnx_owner

    # El instalador de ONNX Runtime debe instalar el mismo stack que su
    # desinstalador retira.
    source = inspect.getsource(rm._install_onnx_stack_into_staging)
    assert '"onnx", "onnxscript"' in source

    # Y el de PyTorch también (los necesita para exportar).
    pytorch_source = inspect.getsource(rm.install_pytorch)
    assert '"onnx", "onnxscript"' in pytorch_source
    assert "onnx" in torch_owner and "onnxscript" in torch_owner


# ---------------------------------------------------------------------------
# Caché de la sonda
# ---------------------------------------------------------------------------


def test_subprocess_import_ok_caches_until_signature_changes(monkeypatch, tmp_path):
    rm._invalidate_store_probe_cache()
    store = str(tmp_path)
    pkg = os.path.join(store, "torch")
    os.makedirs(pkg, exist_ok=True)
    init_py = os.path.join(pkg, "__init__.py")
    with open(init_py, "w", encoding="utf-8") as fh:
        fh.write("x = 1\n")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)

    calls: list[str] = []

    def fake_probe(name, timeout=180.0):
        calls.append(name)
        return True, ""

    monkeypatch.setattr(rm, "_validate_store_import_subprocess", fake_probe)

    assert rm._subprocess_import_ok("torch") == (True, "")
    assert rm._subprocess_import_ok("torch") == (True, "")
    assert calls == ["torch"]  # segunda llamada: caché

    # Cambia el paquete en el store → nueva firma → nueva sonda.
    os.utime(init_py, ns=(1_500_000_000_000_000_000, 1_500_000_000_000_000_000))
    rm._subprocess_import_ok("torch")
    assert calls == ["torch", "torch"]

    # Invalidación explícita (post-instalación) → nueva sonda.
    rm._invalidate_store_probe_cache("torch")
    rm._subprocess_import_ok("torch")
    assert calls == ["torch", "torch", "torch"]


# ---------------------------------------------------------------------------
# Sonda REAL contra el store real del checkout (si torch está instalado)
# ---------------------------------------------------------------------------


def test_probe_end_to_end_against_real_store_when_torch_present():
    """Valida la sonda de punta a punta con el store real del checkout:
    subproceso real, import real, verificación de origen real."""
    init_py = os.path.join(rm.RUNTIME_MODULES_DIR, "torch", "__init__.py")
    if not os.path.isfile(init_py):
        pytest.skip("el store real de este checkout no tiene torch")
    ok, detail = rm._validate_store_import_subprocess("torch", timeout=300.0)
    assert ok is True, detail
