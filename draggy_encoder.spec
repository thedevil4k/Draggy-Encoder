# -*- mode: python ; coding: utf-8 -*-
"""Spec de PyInstaller para Draggy Encoder (Windows, ventana sin consola).

El bundle es deliberadamente ligero: NO incluye torch/onnxruntime/optimum/
safetensors/nvidia-* (~3.5 GB si los llevara). La app los instala bajo
demanda en bin/runtimes/ai_modules con pip --target, usando el Python
embebido (embed-python/python.exe) que coloca el instalador Inno junto al
ejecutable (ver src/runtimes/manager.py:_pip_python).

El directorio de salida (dist/DraggyEncoder) se copia completo por el
instalador. La app escribe sus datos de sesión (bin/, personal/) junto al
EXE, por eso el instalador usa una ruta de usuario sin UAC.
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
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[("assets", "assets")],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

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
    # Verificación local: Defender bloquea BeginUpdateResource sobre el
    # bootloader sin firmar (WinError 225). Con PYINSTALLER_ICON=NONE se
    # omite el paso de incrustar icono para poder compilar/ejecutar el EXE
    # en máquinas con AV agresivo. Sin la variable usa el icono por defecto.
    icon=(os.environ.get("PYINSTALLER_ICON") or "").strip() or None,
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
