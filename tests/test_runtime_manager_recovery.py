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


def test_install_pytorch_raises_with_subprocess_stderr(monkeypatch, tmp_path):
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
    monkeypatch.setattr(
        rm,
        "_validate_store_import_subprocess",
        lambda name, timeout=180.0: (
            False,
            "ImportError: DLL load failed while importing torch_cpu",
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        rm.install_pytorch("cuda126", progress_callback=None)
    message = str(excinfo.value)
    assert "DLL load failed" in message          # stderr real del subproceso
    assert "variante CPU" in message             # hint CUDA→CPU sigue activo


def test_install_onnxruntime_raises_with_subprocess_stderr(monkeypatch, tmp_path):
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
    monkeypatch.setattr(rm, "_purge_stale_module_modules", lambda name: 0)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(
        rm,
        "_validate_store_import_subprocess",
        lambda name, timeout=180.0: (False, "ModuleNotFoundError: No module named 'x'"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        rm.install_onnxruntime("cpu", progress_callback=None)
    assert "ModuleNotFoundError" in str(excinfo.value)


def test_install_pytorch_success_when_subprocess_trusts_store(monkeypatch, tmp_path):
    """Store 'contaminado' para el proceso actual pero válido (subproceso OK):
    la instalación debe REPORTAR ÉXITO, no reintentar ni re-descargar."""
    store = str(tmp_path)
    _make_pkg_dir(store, "torch", "2.13.0-fake")
    monkeypatch.setattr(rm, "RUNTIME_MODULES_DIR", store)
    monkeypatch.setattr(rm, "_preflight_connectivity", lambda: {"summary": "ok"})
    monkeypatch.setattr(rm, "_check_disk_space", lambda kind, variant: None)
    monkeypatch.setattr(rm, "_purge_foreign_version_wheels", lambda: 0)
    monkeypatch.setattr(rm, "_run_pip_checked", lambda *a, **k: None)
    monkeypatch.setattr(rm, "_purge_stale_module_modules", lambda name: 0)
    monkeypatch.setattr(rm, "_can_import", lambda name: False)
    monkeypatch.setattr(rm, "_can_probe_subprocess", lambda: True)
    monkeypatch.setattr(
        rm, "_validate_store_import_subprocess", lambda name, timeout=180.0: (True, "")
    )
    assert rm.install_pytorch("cuda126", progress_callback=None) == "cuda126"


# ---------------------------------------------------------------------------
# Pins defensivos del set PyTorch
# ---------------------------------------------------------------------------


def test_pytorch_command_pins_known_good_set():
    cmd = rm._build_pytorch_command("cuda126")
    assert rm._TORCH_PIN in cmd
    assert rm._TORCHVISION_PIN in cmd
    assert rm._TORCHAUDIO_PIN in cmd
    assert "onnx" in cmd
    assert "onnxscript" in cmd
    # El pin público (sin sufijo local) es lo que viaja a pip; el wheel
    # +cu126 del índice lo satisface (PEP 440: los especificadores públicos
    # ignoran el segmento local del candidato).
    assert "torch==2.13.0" == rm._TORCH_PIN


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
