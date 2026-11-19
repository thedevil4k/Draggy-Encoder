# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
import sys
import importlib.util

"""Spec de PyInstaller para Draggy Encoder (Windows/Linux, ventana sin consola).

El bundle es deliberadamente ligero: NO incluye torch/onnxruntime/optimum/
safetensors/nvidia-* (~3.5 GB si los llevara). La app los instala bajo
demanda en bin/runtimes/ai_modules con pip --target, usando el Python
privado que se coloca junto al ejecutable (embed-python/python.exe en
Windows o runtime-python/bin/python3.13 en Linux; ver
src/runtimes/manager.py:_pip_python).

El directorio de salida (dist/DraggyEncoder) se copia completo por el
instalador o se empaqueta como tar.gz y AppImage en Linux. Los bundles
portables escriben sus datos de sesión (bin/, personal/) junto al ejecutable;
los lanzadores de AppImage redirigen esos datos al perfil del usuario.
"""

# Los imports de torch/onnxruntime/safetensors en el código son perezosos
# (try/except dentro de funciones), así que excluirlos del bundle no rompe
# nada: solo desactiva la IA hasta que el usuario la instale desde la app.
# cv2/PIL solo se usan como fallback de resize en src.processing.pipeline
# (siempre bajo ImportError, con numpy como última red): cv2 se excluye por
# peso; PIL SÍ se incluye (pequeño) para que el fallback haga un resize
# NEAREST real. El stack web (fastapi/uvicorn/mcp/pydantic/boto3/...) solo
# lo importa huggingface_hub._webhooks_server (delayed, opcional): excluirlo
# recorta decenas de MB sin afectar búsqueda/descarga de modelos.
excludes = [
    "torch", "torchvision", "torchaudio",
    "onnxruntime", "onnx", "onnxscript", "optimum", "safetensors",
    "numba", "scipy", "matplotlib", "pandas", "cv2",
    "IPython", "jupyter", "notebook", "pytest",
    "tkinter", "PyQt5", "PyQt6",
    "nvidia", "triton",
    # Stack web/servidor opcional de huggingface_hub (webhooks/OAuth)
    "fastapi", "uvicorn", "starlette", "websockets", "aiohttp", "anyio",
    "mcp", "pydantic", "pydantic_settings",
    "jsonschema", "jsonschema_specifications", "referencing", "rpds",
    "boto3", "botocore", "s3transfer",
    "google", "authlib", "cryptography",
    "gradio", "duckdb", "pyarrow",
    # Pesados/irrelevantes que el barrido de stdlib de abajo encontraría
    "test", "idlelib", "turtledemo", "turtle",
]

# El bundle es el ANFITRIÓN de los runtimes de IA: torch/onnxruntime (y su
# árbol: numpy, sympy, networkx...) se instalan a demanda en
# bin/runtimes/ai_modules y se importan DESDE este proceso congelado (ver
# src/runtimes/manager.py). Esos paquetes importan módulos de la stdlib que
# el análisis estático de NUESTRO código nunca ve, y el fallo solo aparece
# en el equipo del usuario al importar el runtime ya instalado. Caso real:
# torch >= 2.9 importa `timeit` desde torch._strobelight -> "No module named
# 'timeit'" y toda la IA queda desactivada con la instalación completa.
#
# Solución: incluir la stdlib pública completa (unos MB) para que cualquier
# import diferido de los runtimes resuelva. Lo excluido explícitamente
# (tkinter, test, ...) y los módulos privados de la stdlib no vuelven por
# esta vía; los privados que hacen falta los arrastra el análisis de cada
# módulo público incluido.
_STDLIB_SKIP = {
    "__main__", "antigravity", "this", "site", "sitecustomize", "usercustomize",
}
stdlib_hiddenimports = sorted(
    name
    for name in sys.stdlib_module_names
    if not name.startswith("_")
    and name not in _STDLIB_SKIP
    and name not in set(excludes)
    and importlib.util.find_spec(name) is not None
)

# ``datas`` incluye una copia de ``src/`` en ``ai_src/src``: el host de IA
# (src/ai/host.py) corre en el Python PRIVADO del instalador (embed-python),
# que no puede importar el código embebido dentro del EXE. Esa copia de
# fuentes puras es lo que le permite ejecutar los motores (torch/onnxruntime)
# sin cargarlos en el proceso de la GUI — importarlos ahí mataba la app con
# 0xc0000005 en python313.dll. Si esta entrada desaparece, el selftest del
# bundle falla al arrancar el host y el build se para.
# Modelos que el instalador debe llevar de fábrica: los dos upscalers
# pedidos (Real-ESRGAN normal y RealESR AnimeVideo v3). Se copian SOLO los
# .onnx, no los .pth ni los modelos de usuario que haya en bin/models. La
# ruta de destino usa minúsculas porque get_engine_dir() deriva la categoría
# del engine_id ("upscale"); en Windows es equivalente, en Linux importa.
_BUNDLED_MODEL_FILES = [
    ("bin/models/Upscale/realesrgan-x4plus/onnx/model.onnx",
     "bin/models/upscale/realesrgan-x4plus/onnx"),
    ("bin/models/Upscale/realesr-animevideov3/realesr-animevideov3.onnx",
     "bin/models/upscale/realesr-animevideov3"),
]
for _src, _dest in _BUNDLED_MODEL_FILES:
    if not os.path.isfile(_src):
        raise SystemExit(
            "Bundled model missing: %s. Run `python packaging/fetch_bundled_models.py` "
            "before building; the installer promises these two upscalers." % _src
        )

_datas = [("assets", "assets"), ("src", "ai_src/src")]
_datas.extend(_BUNDLED_MODEL_FILES)

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=_datas,
    # ``src.runtimes.offline_pack`` se importa de forma diferida en el worker de
    # arranque (y desde el diálogo de dependencias). Se declara explícito para
    # que el import del pack GPU exista siempre en el bundle, incluso si el
    # análisis de imports diferidos cambia.
    hiddenimports=stdlib_hiddenimports + ["src.runtimes.offline_pack", "src.runtimes.pack_download"],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Icono del ejecutable: assets/app.ico (multi-resolución, generado por
# packaging/windows/make_icon.py desde assets/Import.png) en Windows; en Linux
# el PNG. Un .ico con 16/24/32/48/64/128/256 hace que el logo se vea nítido en
# Explorer, Settings > Apps (entrada de desinstalación) y el instalador.
#   - PYINSTALLER_ICON=NONE   : omite el paso (builds locales con antivirus
#                               agresivo que bloquea BeginUpdateResource sobre
#                               el bootloader sin firmar, WinError 225).
#   - PYINSTALLER_ICON=<path> : fuerza otro archivo de icono.
_icon = (os.environ.get("PYINSTALLER_ICON") or "").strip()
if _icon.upper() == "NONE":
    _icon = None
elif not _icon:
    ico = os.path.join("assets", "app.ico")
    png = os.path.join("assets", "Import.png")
    _icon = ico if os.path.exists(ico) else png

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DraggyEncoder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="DraggyEncoder",
)

# PyInstaller 6 coloca los ``datas`` bajo _internal/. La app resuelve los
# modelos con get_data_dir()/bin, que en un bundle es la carpeta del EXE, no
# _internal. Moverlos a la raíz deja el layout que espera la app y que copia
# Inno Setup; además evita duplicar 70 MB dentro del instalador.
try:
    _dist_root = os.path.join(DISTPATH, "DraggyEncoder")
    _models_src = os.path.join(_dist_root, "_internal", "bin", "models")
    _models_dst = os.path.join(_dist_root, "bin", "models")
    if os.path.isdir(_models_src):
        os.makedirs(os.path.dirname(_models_dst), exist_ok=True)
        shutil.copytree(_models_src, _models_dst, dirs_exist_ok=True)
        shutil.rmtree(_models_src)
except Exception as _exc:
    raise SystemExit("Could not place bundled models next to the EXE: %s" % _exc)
