"""La GUI nunca debe cargar torch/onnxruntime en su PROPIO proceso.

Regresión que hacía que el instalador "no arrancara": importar
``src.ai.architectures`` en el proceso congelado de la GUI cargaba
torch_cpu.dll + torch_cuda.dll (cientos de MB de extensiones nativas) y
Windows mataba DraggyEncoder.exe con una violación de acceso (0xc0000005 en
python313.dll). El disparador era de lo más inocente: ``src/ui/engine_row.py``
y ``src/ui/model_search_result.py`` importaban ese módulo a nivel de módulo
solo para leer la lista de arquitecturas convertibles a ONNX.

El test corre en un SUBPROCESO limpio y con el store de runtimes del checkout
en ``sys.path`` (igual que la app al arrancar), así que si el grafo de imports
de la GUI vuelve a tocar torch, aquí se ve de verdad: el módulo aparece en
``sys.modules``. En pytest no se puede comprobar en el proceso actual porque
otros tests (los que ejercitan la conversión .pth → .onnx) sí importan torch.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Módulos que importa el arranque de la aplicación (main.py → src.app → UI) y
# los que la UI consulta al construir sus paneles. Si alguno arrastra torch u
# onnxruntime, la app instalada puede morir al arrancar sin decir nada.
_GUI_MODULES = (
    "src.config",
    "src.app",
    "src.dependencies",
    "src.ui.main_window",
    "src.ui.preferences_dialog",
    "src.ui.dependencies_dialog",
    "src.ui.engine_row",
    "src.ui.model_search_result",
    "src.ui.engines_tab",
    "src.ui.workflow_widget",
    "src.ui.tasks_panel",
    "src.ai.inference",
    "src.ai.benchmark",
    "src.ai.engines_state",
    "src.ai.hf_search",
    "src.ai.fetcher",
    "src.ai.model_rating",
    "src.processing.pipeline",
)

_PROBE = r"""
import json, sys

sys.path.insert(0, sys.argv[1])

# Como la app al arrancar: el store de runtimes del checkout queda a tiro, de
# modo que un import de torch lo cargaría DE VERDAD (no hay falso negativo).
try:
    from src.runtimes.manager import inject_runtime_paths
    inject_runtime_paths()
except Exception:
    pass

modules = json.loads(sys.argv[2])
fails = {}
for name in modules:
    try:
        __import__(name)
    except BaseException as exc:  # noqa: BLE001 - se reporta tal cual
        fails[name] = f"{type(exc).__name__}: {exc}"

heavy = sorted(
    key for key in sys.modules
    if key == "torch" or key.startswith("torch.")
    or key == "onnxruntime" or key.startswith("onnxruntime.")
)
print(json.dumps({"fails": fails, "heavy": heavy[:12]}))
"""


def _run_probe(modules) -> dict:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, REPO_ROOT, json.dumps(list(modules))],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"la sonda falló (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_gui_import_graph_does_not_pull_heavy_runtimes():
    """Importar toda la GUI no debe meter torch/onnxruntime en el proceso."""
    result = _run_probe(_GUI_MODULES)

    assert result["fails"] == {}, (
        "los módulos de la GUI no se pudieron importar: "
        f"{json.dumps(result['fails'], ensure_ascii=False, indent=2)}"
    )
    assert result["heavy"] == [], (
        "importar la GUI cargó runtimes nativos pesados en el proceso: "
        f"{result['heavy']}. La GUI debe usar src.ai.arch_specs o el host de "
        "IA, nunca src.ai.architectures (que importa torch al cargarse)."
    )


_ENGINE_PROBE = r"""
import json, sys

sys.path.insert(0, sys.argv[1])
import numpy as np

result = {"shape": None, "error": None, "torch": []}
try:
    from src.ai.inference import create_inference_engine

    engine = create_inference_engine("onnxruntime", sys.argv[2])
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    out = engine.process_frame(frame)
    result["shape"] = [int(dim) for dim in out.shape]
    engine.unload()
except BaseException as exc:  # noqa: BLE001 - se reporta tal cual
    result["error"] = f"{type(exc).__name__}: {exc}"
result["torch"] = sorted(
    key for key in sys.modules
    if key == "torch" or key.startswith("torch.")
    or key == "onnxruntime" or key.startswith("onnxruntime.")
)
print(json.dumps(result))
"""

_MODEL = os.path.join(
    REPO_ROOT, "bin", "models", "Upscale", "realesr-animevideov3",
    "realesr-animevideov3.onnx",
)


@pytest.mark.skipif(not os.path.isfile(_MODEL), reason="falta el modelo del checkout")
def test_engine_runs_in_the_host_not_in_the_gui_process():
    """El objetivo completo: inferencia real sin torch/onnxruntime en la GUI.

    Se construye el motor por la FÁBRICA (como hace la app) y se procesa un
    frame; después se comprueba que en ESTE proceso no quedó ni torch ni
    onnxruntime. Si alguien devuelve la inferencia al proceso de la GUI, este
    test falla.
    """
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("DRAGGY_AI_HOST", None)
    proc = subprocess.run(
        [sys.executable, "-c", _ENGINE_PROBE, REPO_ROOT, _MODEL],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"la sonda falló:\n{proc.stdout}\n{proc.stderr}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["error"] is None, result["error"]
    assert result["shape"] == [64, 96, 3], result["shape"]
    assert result["torch"] == [], (
        "la inferencia cargó runtimes nativos en el proceso de la GUI: "
        f"{result['torch']}"
    )


def test_arch_specs_is_torch_free():
    """``arch_specs`` existe justo para eso: la UI lo importa a nivel de módulo."""
    result = _run_probe(("src.ai.arch_specs",))

    assert result["fails"] == {}
    assert result["heavy"] == []
