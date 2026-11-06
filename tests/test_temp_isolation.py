"""Regresión: los temporales de Draggy nunca caen en ``%TEMP%`` (C:).

El instalador vive en ``D:\\DraggyEncoder`` porque el usuario tiene C: lleno.
``main.py`` y ``src.cli_encode`` deben llamar a ``configure_process_temp()``
antes de crear temporales; si esa llamada desaparece, el guard del selftest,
el host de IA o un fallback del encode vuelven a escribir en ``%TEMP%``.

Las dos sondas arrancan la ruta real de arranque de cada entrypoint en un
subproceso limpio con ``TEMP``/``TMP``/``TMPDIR`` apuntando a un directorio
centinela (en Windows, bajo C:). Después crean un temporal con prefijo
``draggy-``: si la redirección no ocurrió, el temporal aparece en el
centinela y el test falla.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE_MARKER = "TEMP_ISOLATION_PROBE="

# ``main`` ejecuta ensure_runtime_dirs() + configure_process_temp() al
# importarse, exactamente igual que al arrancar ``--selftest``. No se lanza
# el selftest entero (necesita Qt, host y el Python privado, y es lento/
# frágil fuera de un bundle): la redirección que protege sus temporales se
# fija en este arranque, antes de cualquier comprobación.
_SELFTEST_PROBE = r"""
import json
import os
import tempfile

import main  # noqa: F401  arranque real de la app

probe = tempfile.mkdtemp(prefix="draggy-selftest-")
with open(os.path.join(probe, "probe.bin"), "wb") as fh:
    fh.write(b"selftest")
print("TEMP_ISOLATION_PROBE=" + json.dumps({
    "tempdir": tempfile.gettempdir(),
    "probe": probe,
    "env": {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")},
}))
"""

# ``run_encode_cli`` configura el temporal antes de parsear argumentos. Un
# argv vacío lanza el SystemExit de usage, pero ya ha recorrido la ruta de
# arranque del encode que nos interesa.
_ENCODE_PROBE = r"""
import json
import os
import tempfile

from src.cli_encode import run_encode_cli

try:
    run_encode_cli([])
except SystemExit:
    pass

probe = tempfile.mkdtemp(prefix="draggy-encode-")
with open(os.path.join(probe, "probe.bin"), "wb") as fh:
    fh.write(b"encode")
print("TEMP_ISOLATION_PROBE=" + json.dumps({
    "tempdir": tempfile.gettempdir(),
    "probe": probe,
    "env": {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")},
}))
"""


@pytest.fixture()
def app_data_dir():
    """Datos de la app en la misma unidad que el checkout (D: aquí).

    No se usa ``tmp_path`` porque en Windows pytest lo crea bajo C: y
    precisamente queremos que la app tenga sus datos fuera de C:.
    ``tmp-*/`` ya está ignorado por git; el directorio se borra siempre.
    """
    path = REPO_ROOT / f"tmp-test-temp-isolation-{os.getpid()}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _run_probe(source: str, app_data: Path, system_temp: Path) -> dict:
    system_temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "DRAGGY_DATA_DIR": str(app_data),
        "TEMP": str(system_temp),
        "TMP": str(system_temp),
        "TMPDIR": str(system_temp),
        "QT_QPA_PLATFORM": "offscreen",
    })
    # No dejar que el entorno del runner inyecte el checkout o un host de IA
    # ya configurado: el subproceso debe arrancar como un proceso limpio.
    env.pop("PYTHONPATH", None)
    env.pop("DRAGGY_AI_HOST", None)

    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"la sonda del entrypoint falló (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    lines = [
        line for line in proc.stdout.splitlines()
        if line.startswith(_PROBE_MARKER)
    ]
    assert lines, f"la sonda no devolvió su resultado:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(lines[-1][len(_PROBE_MARKER):])


def _assert_isolated(result: dict, app_data: Path, system_temp: Path) -> None:
    app_tmp = (app_data / "bin" / "tmp").resolve()
    reported_tempdir = Path(result["tempdir"]).resolve()
    probe = Path(result["probe"]).resolve()

    assert reported_tempdir == app_tmp, (
        f"tempfile.gettempdir() apunta a {reported_tempdir}, no a {app_tmp}"
    )
    assert probe.is_relative_to(app_tmp), (
        f"el temporal de Draggy se creó fuera de bin/tmp: {probe}"
    )
    assert not app_tmp.is_relative_to(system_temp.resolve()), (
        f"bin/tmp ({app_tmp}) está dentro del temporal del sistema "
        f"({system_temp}); los datos de la app deben vivir en su unidad"
    )
    for name in ("TMP", "TEMP", "TMPDIR"):
        assert result["env"][name] is not None, f"{name} quedó sin definir"
        assert Path(result["env"][name]).resolve() == app_tmp, (
            f"{name}={result['env'][name]!r} no apunta a {app_tmp}"
        )

    leaked = sorted(
        {p for pattern in ("*draggy*", ".drag*") for p in system_temp.rglob(pattern)}
    )
    assert leaked == [], (
        f"temporales de Draggy en el temporal del sistema ({system_temp}): {leaked}"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(_SELFTEST_PROBE, id="selftest"),
        pytest.param(_ENCODE_PROBE, id="encode"),
    ],
)
def test_entrypoint_keeps_draggy_temp_out_of_the_system_temp(
    source, app_data_dir, tmp_path
):
    """Ni ``--selftest`` ni el encode usan el temporal del sistema."""
    system_temp = tmp_path / "system-temp"

    result = _run_probe(source, app_data_dir, system_temp)

    _assert_isolated(result, app_data_dir, system_temp)
