"""Descarga los dos ONNX que el instalador/AppImage promete de fábrica.

Los pesos no se versionan en Git: `*.onnx` está ignorado y el repo debe
seguir siendo código, no 70 MB de binarios. El CI los baja aquí antes de
PyInstaller, comprueba el SHA-256 y deja el layout que espera el spec.

Uso:
    python packaging/fetch_bundled_models.py

Si un archivo ya existe y su hash coincide, se omite la descarga. Así un
build local no vuelve a bajar 67 MB y el CI puede cachear ``bin/models`` en
el futuro. Si el hash no coincide se reemplaza de forma atómica; nunca se
deja un ``.onnx`` truncado en el bundle.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

#: (url, ruta relativa al repo, sha256 esperado, tamaño legible)
MODELS = (
    (
        "https://huggingface.co/imgdesignart/realesrgan-x4-onnx/"
        "resolve/main/onnx/model.onnx",
        "bin/models/Upscale/realesrgan-x4plus/onnx/model.onnx",
        "fa18ce70de3a55f3149d0cc898d335d2d69fca29edc0692cb362c856b2942c3f",
        "64 MB",
    ),
    (
        "https://huggingface.co/skillsafe-ai/realesr-animevideov3/"
        "resolve/main/model.onnx",
        "bin/models/Upscale/realesr-animevideov3/realesr-animevideov3.onnx",
        "78baa685a1a92cac6e14ab2af2a8ad0ef56fa48629563b1414e3a7acc54d86e2",
        "2.4 MB",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, expected_sha: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".part", dir=str(target.parent)
    )
    os.close(fd)
    temp = Path(temp_name)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DraggyEncoder-build/1.0"},
    )
    try:
        print(f"Downloading {target.name} ...", flush=True)
        last_report = 0
        with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and done - last_report >= 8 * 1024 * 1024:
                    print(f"  {done / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB", flush=True)
                    last_report = done
        actual = _sha256(temp)
        if actual != expected_sha:
            raise SystemExit(
                f"SHA-256 mismatch for {target}:\n"
                f"  expected {expected_sha}\n"
                f"  actual   {actual}"
            )
        os.replace(temp, target)
        print(f"  OK {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    for url, relative, expected_sha, size in MODELS:
        target = ROOT / relative
        if target.is_file() and _sha256(target) == expected_sha:
            print(f"Using existing {target} ({size})", flush=True)
            continue
        if target.is_file():
            print(f"Replacing {target}: hash differs", flush=True)
        _download(url, target, expected_sha)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        raise SystemExit(130)
