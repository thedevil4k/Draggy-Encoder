"""Construye el pack GPU offline de NVIDIA (partes < 2 GiB + manifiesto).

El pack es un store ya instalado (``torch``, ``torchvision``, ``torchaudio``,
``onnxruntime`` y el cierre de dependencias) troceado en ZIPs independientes.
No se versiona en Git ni se genera en el equipo del usuario: lo produce el CI
(o un desarrollador con una GPU) y se adjunta a una Release.

Uso:
    python packaging/build_gpu_pack.py --store build/gpu-store --out dist/gpu-pack

Las partes se cortan por TAMAÑO DESCOMPRIMIDO (presupuesto conservador): un
ZIP nunca ocupa más que la suma de sus archivos más una cabecera pequeña, así
que cada asset queda por debajo del límite de 2 GiB de GitHub Releases incluso
si la compresión no ayuda. El límite por defecto (1900 MiB) deja ~148 MiB de
margen para la cabecera central y miles de entradas.

Cada parte es independiente: la app las extrae en orden sobre el mismo
staging. Un asset corrupto se vuelve a descargar solo, sin recomponer un
archivo gigante ni necesitar el doble de disco.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtimes.offline_pack import (  # noqa: E402
    MANIFEST_NAME,
    MAX_ASSET_BYTES,
    PACK_FORMAT,
)

_DEFAULT_PART_MIB = 1900
_CHUNK = 4 * 1024 * 1024

#: Archivos que la app exige encontrar tras extraer. Si el store no los tiene,
#: el pack se construiría pero fallaría en el equipo del usuario.
_CRITICAL = (
    "torch/__init__.py",
    "torchvision/__init__.py",
    "onnxruntime/__init__.py",
    "onnxruntime/capi/onnxruntime_pybind11_state.pyd",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(store: Path):
    """Archivos del store, en orden estable, sin bytecode."""
    for dirpath, dirnames, filenames in os.walk(store):
        dirnames[:] = sorted(
            d for d in dirnames if d != "__pycache__" and not d.startswith(".")
        )
        for name in sorted(filenames):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                raise SystemExit(
                    f"El store contiene un symlink ({path}); el pack solo "
                    "admite archivos regulares."
                )
            yield path


def _read_metadata(store: Path) -> dict[str, str]:
    """``{nombre normalizado: versión}`` desde los ``*.dist-info``."""
    versions: dict[str, str] = {}
    for metadata in store.glob("*.dist-info/METADATA"):
        name = ""
        version = ""
        try:
            with metadata.open("r", encoding="utf-8", errors="replace") as fh:
                for _ in range(60):
                    line = fh.readline()
                    if not line or not line.strip():
                        break
                    low = line.lower()
                    if low.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                    elif low.startswith("version:"):
                        version = line.split(":", 1)[1].strip()
                    if name and version:
                        break
        except OSError:
            continue
        if name and version:
            versions[name.replace("_", "-").lower()] = version
    return versions


def _check_store_python_tag(store: Path, python_tag: str) -> None:
    """Rechaza empaquetar wheels compilados para otro CPython.

    Un pack etiquetado cp313 con un numpy cp314 dentro importa mal en el
    bundle (o no importa). Es mejor fallar al construir que publicar 2 GB
    inservibles.
    """
    expected = python_tag.removeprefix("cp")
    for wheel in store.glob("*.dist-info/WHEEL"):
        try:
            text = wheel.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip().lower().startswith("tag:"):
                continue
            for tag in line.split(":", 1)[1].split():
                if "abi3" in tag:
                    continue
                first = tag.split("-", 1)[0]
                if first.startswith("cp") and first[2:].isdigit():
                    if first[2:] != expected:
                        raise SystemExit(
                            f"{wheel} declara el tag {tag!r}, incompatible con "
                            f"{python_tag}. Reinstala el store con el Python "
                            "embebido correcto (3.13)."
                        )


def _critical_files(store: Path) -> list[str]:
    critical = []
    for relative in _CRITICAL:
        if (store / relative).is_file():
            critical.append(relative)
    # El nombre del DLL de CUDA cambia entre versiones (torch_cuda.dll,
    # torch_cuda12.dll...). Se busca en lugar de fijarlo.
    cuda_dlls = sorted(
        p.relative_to(store).as_posix()
        for p in (store / "torch" / "lib").glob("*cuda*.dll")
        if p.is_file()
    )
    if cuda_dlls:
        critical.append(cuda_dlls[0])
    else:
        raise SystemExit(
            "El store no contiene ninguna DLL de CUDA en torch/lib. "
            "¿Instalaste la variante CPU por error?"
        )
    return critical


class _PartWriter:
    """Escribe una parte y la cierra al llegar al presupuesto."""

    def __init__(self, path: Path, compression: int):
        self.path = path
        self._compression = compression
        self._zip = zipfile.ZipFile(
            path, "w", compression=compression,
            compresslevel=6 if compression == zipfile.ZIP_DEFLATED else None,
            allowZip64=True,
        )
        self.files = 0
        self.uncompressed = 0

    def add(self, source: Path, arcname: str) -> None:
        stat = source.stat()
        info = zipfile.ZipInfo(arcname, time.localtime(stat.st_mtime)[:6])
        info.compress_type = self._compression
        info.external_attr = 0o644 << 16
        with source.open("rb") as src, self._zip.open(info, "w") as dst:
            for chunk in iter(lambda: src.read(_CHUNK), b""):
                dst.write(chunk)
        self.files += 1
        self.uncompressed += stat.st_size

    def close(self) -> None:
        self._zip.close()


def build_pack(
    store: Path,
    out_dir: Path,
    *,
    pack_id: str,
    part_limit: int,
    compress: bool = True,
    python_tag: str | None = None,
    platform_name: str = "win32",
    arch: str = "amd64",
    backend: str = "cuda",
    cuda: str = "13.0",
    min_driver_branch: int = 580,
) -> dict:
    store = store.resolve()
    if not (store / "torch" / "__init__.py").is_file():
        raise SystemExit(f"{store} no parece un store con PyTorch instalado.")
    if not (store / "onnxruntime").is_dir():
        raise SystemExit(f"{store} no contiene onnxruntime.")
    if part_limit <= 0 or part_limit >= MAX_ASSET_BYTES:
        raise SystemExit(
            f"part_limit={part_limit} no es válido: debe ser > 0 y < 2 GiB."
        )

    python_tag = python_tag or f"cp{sys.version_info.major}{sys.version_info.minor}"
    _check_store_python_tag(store, python_tag)
    crit = _critical_files(store)

    files = [(p, p.relative_to(store).as_posix()) for p in _iter_files(store)]
    for source, arcname in files:
        if source.stat().st_size >= part_limit:
            raise SystemExit(
                f"{arcname} mide {source.stat().st_size} bytes y no cabe en una "
                f"parte de {part_limit}. Aumenta --part-mib hasta el límite "
                "de 2 GiB o reparte el archivo."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{pack_id}-part*.zip"):
        stale.unlink()

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    parts: list[dict] = []
    writer: _PartWriter | None = None

    try:
        for source, arcname in files:
            size = source.stat().st_size
            if writer is not None and writer.uncompressed + size > part_limit:
                writer.close()
                parts.append(_describe_part(writer))
                writer = None
            if writer is None:
                index = len(parts) + 1
                writer = _PartWriter(
                    out_dir / f"{pack_id}-part{index:02d}.zip", compression
                )
            writer.add(source, arcname)
        if writer is not None:
            writer.close()
            parts.append(_describe_part(writer))
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        raise

    if not parts:
        raise SystemExit("El store está vacío: no hay nada que empaquetar.")

    installed_bytes = sum(p["uncompressed_bytes"] for p in parts)
    manifest = {
        "format": PACK_FORMAT,
        "id": pack_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform_name,
        "arch": arch,
        "python_tag": python_tag,
        "backend": backend,
        "cuda": cuda,
        "min_driver_branch": min_driver_branch,
        "versions": _read_metadata(store),
        "installed_bytes": installed_bytes,
        "file_count": sum(p["files"] for p in parts),
        "download_bytes": sum(p["bytes"] for p in parts),
        "critical_files": crit,
        "parts": parts,
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _describe_part(writer: _PartWriter) -> dict:
    path = writer.path
    size = path.stat().st_size
    if size >= MAX_ASSET_BYTES:
        path.unlink(missing_ok=True)
        raise SystemExit(
            f"{path.name} mide {size} bytes y supera el límite de 2 GiB por "
            "asset. Reduce --part-mib."
        )
    return {
        "name": path.name,
        "bytes": size,
        "sha256": _sha256(path),
        "files": writer.files,
        "uncompressed_bytes": writer.uncompressed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Construye el pack GPU offline de Draggy Encoder."
    )
    parser.add_argument(
        "--store", required=True,
        help="Directorio ai_modules ya instalado (torch + onnxruntime).",
    )
    parser.add_argument(
        "--out", default="dist/gpu-pack",
        help="Carpeta de salida (por defecto dist/gpu-pack).",
    )
    parser.add_argument("--id", default=None, help="Identificador del pack.")
    parser.add_argument(
        "--part-mib", type=int, default=_DEFAULT_PART_MIB,
        help=f"Presupuesto descomprimido por parte en MiB "
             f"(por defecto {_DEFAULT_PART_MIB}, límite duro 2047).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Guarda sin deflate (más rápido de construir, más pesado).",
    )
    parser.add_argument("--python-tag", default=None, help="p. ej. cp313.")
    parser.add_argument("--cuda", default="13.0", help="Versión de CUDA.")
    parser.add_argument(
        "--min-driver-branch", type=int, default=580,
        help="Rama mínima del driver NVIDIA.",
    )
    args = parser.parse_args(argv)

    part_limit = args.part_mib * 1024 * 1024
    if part_limit >= MAX_ASSET_BYTES:
        parser.error("--part-mib debe dejar cada ZIP por debajo de 2 GiB")

    store = Path(args.store)
    out_dir = Path(args.out)
    python_tag = args.python_tag or (
        f"cp{sys.version_info.major}{sys.version_info.minor}"
    )
    pack_id = args.id or (
        f"nvidia-cu{args.cuda.replace('.', '')}-win-x64-{python_tag}"
    )
    manifest = build_pack(
        store,
        out_dir,
        pack_id=pack_id,
        part_limit=part_limit,
        compress=not args.no_compress,
        python_tag=python_tag,
        cuda=args.cuda,
        min_driver_branch=args.min_driver_branch,
    )

    print(f"Pack: {manifest['id']}")
    print(f"Versiones: {manifest['versions']}")
    print(
        f"Instalado: {manifest['installed_bytes'] / 1024 ** 3:.2f} GB en "
        f"{manifest['file_count']} archivos"
    )
    for part in manifest["parts"]:
        print(
            f"  {part['name']}  {part['bytes'] / 1024 ** 2:.1f} MB  "
            f"sha256={part['sha256'][:16]}..."
        )
    print(f"Manifiesto: {out_dir / MANIFEST_NAME}")
    print(
        "\nSúbelo a una Release (los assets de Releases no gastan la cuota de "
        "artifacts):\n"
        f"  gh release upload <tag> {out_dir}/*.zip {out_dir}/{MANIFEST_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
