"""Descarga del pack GPU de NVIDIA desde GitHub Releases.

Es el camino "online" del mismo pack que importa ``offline_pack``: descarga
``gpu-pack.json`` y las partes ZIP con reanudación por ``Range``, valida el
manifiesto ANTES de bajar gigabytes, verifica el SHA-256 de cada parte y deja
el manifiesto listo para que ``import_pack()`` haga la instalación real.

No duplica la lógica de instalación: extracción segura, sonda CUDA, swap y
rollback siguen viviendo en ``offline_pack``. Aquí solo se resuelve la red.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.parse

from src.config import REPO_URL
from src.runtimes import manager, offline_pack
from src.runtimes.offline_pack import (
    FREE_SPACE_MARGIN,
    MANIFEST_NAME,
    OfflinePackError,
    _check_abort,
    _emit,
    _sha256_file,
    load_manifest,
    pack_supported,
    validate_manifest,
)

#: Release rolling que publica el workflow ``build-gpu-pack.yml``.
RELEASE_TAG = "gpu-pack-v1"
RELEASE_BASE_URL = f"{REPO_URL}/releases/download/{RELEASE_TAG}"

#: Carpeta de descarga dentro de ``bin/runtimes``: misma unidad que el store
#: (el import usa rename, no copia entre volúmenes) y fuera de ``ai_modules``.
DOWNLOAD_DIR_NAME = ".gpu-pack-download"

_CHUNK = 1024 * 1024


def manifest_url() -> str:
    return f"{RELEASE_BASE_URL}/{MANIFEST_NAME}"


def part_url(name: str) -> str:
    return f"{RELEASE_BASE_URL}/{urllib.parse.quote(name)}"


def download_dir() -> str:
    return os.path.join(manager._install_base_dir(), DOWNLOAD_DIR_NAME)


def _existing_download_bytes(target_dir: str, parts: list[dict]) -> int:
    """Bytes ya presentes que no hay que volver a bajar.

    Cuenta la parte final o su ``.part`` (lo que exista) con tope en el
    tamaño declarado. Si una parte corrupta se acaba borrando, el espacio que
    libera compensa exactamente lo que ocupa el nuevo intento, así que el
    cálculo sigue siendo conservador.
    """
    total = 0
    for part in parts:
        expected = int(part["bytes"])
        name = str(part["name"])
        for candidate in (
            os.path.join(target_dir, name),
            os.path.join(target_dir, name) + ".part",
        ):
            try:
                total += min(os.path.getsize(candidate), expected)
                break
            except OSError:
                continue
    return total


def _disk_check(manifest: dict, target_dir: str) -> None:
    """Comprueba el pico de disco ANTES de descargar.

    Durante la instalación conviven las partes descargadas, el staging
    descomprimido y el store anterior (hasta el swap). ``import_pack`` ya
    valida su parte, pero aquí se suma lo que falta por descargar para no
    llenar la unidad a mitad de los 2 GB. Lo ya bajado no se cuenta dos veces:
    un reintento tras un corte debe poder reanudar aunque quede poco espacio.
    """
    parts = manifest["parts"]
    compressed = sum(int(p["bytes"]) for p in parts)
    pending = max(0, compressed - _existing_download_bytes(target_dir, parts))
    installed = int(manifest["installed_bytes"])
    store = manager.RUNTIME_MODULES_DIR
    current = offline_pack._dir_size(store)
    needed = pending + installed + current + FREE_SPACE_MARGIN
    free = offline_pack._free_bytes(target_dir)
    if free and free < needed:
        raise OfflinePackError(
            "No hay espacio para descargar e instalar el pack GPU. Hacen "
            f"falta {needed / 1024 ** 3:.1f} GB libres en la unidad del "
            f"programa y quedan {free / 1024 ** 3:.1f} GB. Libera espacio o "
            "usa 'Import GPU pack…' con un pack ya descargado."
        )


def _fetch_manifest(session, target_dir: str) -> str:
    manifest_path = os.path.join(target_dir, MANIFEST_NAME)
    url = manifest_url()
    try:
        response = session.get(url, timeout=60)
    except Exception as exc:
        raise OfflinePackError(
            "No se pudo contactar con GitHub para descargar el pack GPU: "
            f"{exc}. Comprueba la conexión y vuelve a intentarlo."
        ) from exc
    if getattr(response, "status_code", 200) == 404:
        raise OfflinePackError(
            "La Release del pack GPU todavía no tiene assets publicados "
            f"({RELEASE_TAG}). Ejecuta el workflow 'Build NVIDIA GPU Pack' "
            "o usa 'Import GPU pack…' con un pack local."
        )
    try:
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise OfflinePackError(
            f"No se pudo leer {MANIFEST_NAME} desde GitHub: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise OfflinePackError(
            f"{MANIFEST_NAME} de la Release no contiene un objeto JSON."
        )
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, manifest_path)
    return manifest_path


def _cleanup_stale_parts(target_dir: str, keep: set[str]) -> None:
    """Borra partes de un manifiesto anterior para no acumular gigabytes."""
    try:
        entries = os.listdir(target_dir)
    except OSError:
        return
    for entry in entries:
        base = entry[:-5] if entry.endswith(".part") else entry
        if base.endswith(".zip") and base not in keep:
            try:
                os.remove(os.path.join(target_dir, entry))
            except OSError:
                pass


def _download_part(
    session,
    name: str,
    part: dict,
    target_dir: str,
    done_before: int,
    total: int,
    progress_callback,
    abort_event,
) -> None:
    """Descarga una parte con reanudación y verifica su SHA-256."""
    expected = int(part["bytes"])
    digest = str(part["sha256"]).lower()
    final = os.path.join(target_dir, name)
    partial = final + ".part"

    if os.path.isfile(final):
        if os.path.getsize(final) == expected and _sha256_file(final, abort_event) == digest:
            _emit(
                progress_callback,
                (done_before + expected) / max(1, total) * 55,
                f"Parte ya descargada: {name}",
            )
            return
        try:
            os.remove(final)
        except OSError as exc:
            raise OfflinePackError(
                f"No se pudo reemplazar la parte {name!r} descargada: {exc}"
            ) from exc

    have = os.path.getsize(partial) if os.path.isfile(partial) else 0
    if have > expected:
        try:
            os.remove(partial)
        except OSError:
            pass
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    url = part_url(name)
    try:
        response = session.get(
            url, stream=True, headers=headers, timeout=(15, 60)
        )
    except Exception as exc:
        raise OfflinePackError(
            f"Fallo de red descargando {name!r}: {exc}. Lo descargado se "
            "conserva; vuelve a intentarlo para continuar donde quedó."
        ) from exc

    with response:
        status = int(getattr(response, "status_code", 200))
        if status == 416:
            if have != expected:
                try:
                    os.remove(partial)
                except OSError:
                    pass
                raise OfflinePackError(
                    f"El servidor rechazó reanudar {name!r} y lo descargado "
                    "no está completo. Se borró para empezar de cero."
                )
        else:
            if status >= 400:
                raise OfflinePackError(
                    f"GitHub respondió {status} al descargar {name!r}. "
                    f"Comprueba que la Release {RELEASE_TAG} tiene esa parte."
                )
            if have and status != 206:
                # El servidor ignoró Range: el archivo empieza de cero.
                have = 0
            content_length = str(
                response.headers.get("Content-Length", "") or ""
            )
            if content_length.isdigit() and have + int(content_length) > expected:
                raise OfflinePackError(
                    f"El servidor devolvió más bytes de los declarados para "
                    f"{name!r}. Se aborta para no corromper la parte."
                )
            try:
                with open(partial, "ab" if have else "wb") as fh:
                    for chunk in response.iter_content(_CHUNK):
                        _check_abort(abort_event)
                        if not chunk:
                            continue
                        have += len(chunk)
                        if have > expected:
                            raise OfflinePackError(
                                f"La descarga de {name!r} superó el tamaño "
                                "declarado en el manifiesto."
                            )
                        fh.write(chunk)
                        _emit(
                            progress_callback,
                            (done_before + have) / max(1, total) * 55,
                            f"Descargando {name}… "
                            f"{have / 1024 ** 2:.0f} / "
                            f"{expected / 1024 ** 2:.0f} MB",
                        )
            except OfflinePackError:
                raise
            except Exception as exc:
                raise OfflinePackError(
                    f"Fallo de red descargando {name!r}: {exc}. Lo descargado "
                    "se conserva; vuelve a intentarlo para continuar donde quedó."
                ) from exc

    if have != expected:
        raise OfflinePackError(
            f"{name!r} quedó incompleta ({have} de {expected} bytes). Vuelve "
            "a intentarlo: la descarga continúa donde quedó."
        )

    _emit(
        progress_callback,
        (done_before + expected) / max(1, total) * 55,
        f"Verificando {name}…",
    )
    actual = _sha256_file(partial, abort_event)
    if actual != digest:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise OfflinePackError(
            f"El SHA-256 de {name!r} no coincide. Se borró la parte "
            "corrupta; vuelve a descargar el pack."
        )
    os.replace(partial, final)


def download_pack(
    progress_callback=None,
    abort_event=None,
    target_dir: str | None = None,
    session=None,
) -> str:
    """Descarga el pack publicado y devuelve la ruta de ``gpu-pack.json``.

    Solo descarga y verifica; no toca el store. El llamador debe pasar la
    ruta a ``import_pack()``. Si se cancela o falla, las partes ``.part``
    quedan en disco para reanudar. ``session`` permite inyectar un cliente
    HTTP en tests.
    """
    if offline_pack.is_offline_mode():
        raise OfflinePackError(
            "DRAGGY_OFFLINE está activo: no se descarga nada por red. Usa "
            "'Import GPU pack…' con el manifiesto y las partes locales."
        )
    if not pack_supported():
        raise OfflinePackError(
            "El pack GPU offline solo está disponible para Windows x64; esta "
            "app corre en un sistema no soportado."
        )
    if session is None:
        import requests

        session = requests.Session()

    target_dir = os.path.abspath(target_dir or download_dir())
    os.makedirs(target_dir, exist_ok=True)

    _emit(progress_callback, 0, "Contactando con GitHub Releases…")
    manifest_path = _fetch_manifest(session, target_dir)
    manifest = validate_manifest(load_manifest(manifest_path))

    parts = manifest["parts"]
    _cleanup_stale_parts(target_dir, {str(p["name"]) for p in parts})
    _disk_check(manifest, target_dir)
    total = sum(int(p["bytes"]) for p in parts)
    done_before = 0

    for index, part in enumerate(parts, start=1):
        _check_abort(abort_event)
        name = str(part["name"])
        _emit(
            progress_callback,
            done_before / max(1, total) * 55,
            f"Descargando parte {index}/{len(parts)}: {name}",
        )
        _download_part(
            session,
            name,
            part,
            target_dir,
            done_before,
            total,
            progress_callback,
            abort_event,
        )
        done_before += int(part["bytes"])
        _emit(
            progress_callback,
            done_before / max(1, total) * 55,
            f"Parte {index}/{len(parts)} verificada",
        )

    _check_abort(abort_event)
    _emit(progress_callback, 60, "Pack descargado; instalando el runtime…")
    return manifest_path


def discard_downloaded_pack(manifest_path: str) -> None:
    """Borra la carpeta de descarga tras una instalación correcta.

    Solo actúa sobre la carpeta gestionada; un pack importado manualmente
    desde otro sitio no se toca.
    """
    parent = os.path.abspath(os.path.dirname(manifest_path))
    managed = os.path.abspath(download_dir())
    if parent == managed:
        shutil.rmtree(managed, ignore_errors=True)
