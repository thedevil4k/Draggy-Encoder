"""Descarga del pack GPU: reanudación, hashes y límites de red.

La descarga no instala: entrega un ``gpu-pack.json`` al mismo importador del
pack offline. Estos tests fijan lo que puede romper un fallo de red:

* que una parte a medias se reanude con ``Range`` en vez de bajar 2 GB otra vez,
* que un SHA-256 malo borre la parte corrupta,
* que la validación del manifiesto ocurra ANTES de descargar las partes,
* que cancelar conserve lo descargado y no toque el store.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import threading
from pathlib import Path

import pytest

from src.runtimes import manager, offline_pack, pack_download


pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="El pack GPU offline solo se distribuye para Windows x64",
)


class _Response:
    def __init__(self, status: int = 200, body: bytes = b"", data=None,
                 on_chunk=None):
        self.status_code = status
        self._body = body
        self._data = data
        self.on_chunk = on_chunk
        self.headers = (
            {"Content-Length": str(len(body))} if body else {}
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data

    def iter_content(self, chunk_size):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]
            if self.on_chunk is not None:
                self.on_chunk()


class _Session:
    """Cliente falso: sirve el manifiesto y una única parte."""

    def __init__(self, manifest: dict, payload: bytes, on_chunk=None):
        self.manifest = manifest
        self.payload = payload
        self.on_chunk = on_chunk
        self.manifest_requests = 0
        self.part_requests: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        if url.endswith(pack_download.MANIFEST_NAME):
            self.manifest_requests += 1
            return _Response(data=self.manifest)
        self.part_requests.append((url, kwargs))
        header = (kwargs.get("headers") or {}).get("Range", "")
        start = int(header.split("=", 1)[1].rstrip("-")) if header else 0
        return _Response(
            status=206 if start else 200,
            body=self.payload[start:],
            on_chunk=self.on_chunk,
        )


def _manifest(payload: bytes, **overrides) -> dict:
    manifest = {
        "format": offline_pack.PACK_FORMAT,
        "id": "test-download",
        "platform": "win32",
        "arch": platform.machine(),
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "backend": "cuda",
        "min_driver_branch": 0,
        "installed_bytes": len(payload),
        "download_bytes": len(payload),
        "parts": [
            {
                "name": "gpu-pack-test-part01.zip",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "uncompressed_bytes": len(payload),
            }
        ],
    }
    manifest.update(overrides)
    return manifest


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    store = tmp_path / "runtimes" / "ai_modules"
    store.mkdir(parents=True)
    monkeypatch.setattr(manager, "RUNTIME_MODULES_DIR", str(store))
    monkeypatch.setattr(
        pack_download,
        "RELEASE_BASE_URL",
        "https://example.invalid/releases/gpu-pack-v1",
    )
    return tmp_path


def test_download_fetches_part_and_reports_progress(sandbox):
    payload = b"pack-bytes" * 512
    session = _Session(_manifest(payload), payload)
    seen = []

    path = pack_download.download_pack(
        target_dir=sandbox / "dl",
        session=session,
        progress_callback=lambda pct, msg: seen.append((pct, msg)),
    )

    assert Path(path) == sandbox / "dl" / offline_pack.MANIFEST_NAME
    assert (sandbox / "dl" / "gpu-pack-test-part01.zip").read_bytes() == payload
    assert session.manifest_requests == 1
    assert len(session.part_requests) == 1
    assert session.part_requests[0][0].endswith(
        "/gpu-pack-test-part01.zip"
    )
    assert seen and seen[-1][0] >= 60


def test_download_resumes_a_partial_part_with_range(sandbox):
    payload = b"0123456789" * 400
    target = sandbox / "dl"
    target.mkdir()
    (target / "gpu-pack-test-part01.zip.part").write_bytes(payload[:1234])
    session = _Session(_manifest(payload), payload)

    pack_download.download_pack(target_dir=target, session=session)

    headers = session.part_requests[0][1]["headers"]
    assert headers == {"Range": "bytes=1234-"}
    assert (target / "gpu-pack-test-part01.zip").read_bytes() == payload
    assert not (target / "gpu-pack-test-part01.zip.part").exists()


def test_download_reuses_a_complete_verified_part(sandbox):
    payload = b"already-here" * 256
    target = sandbox / "dl"
    target.mkdir()
    (target / "gpu-pack-test-part01.zip").write_bytes(payload)
    session = _Session(_manifest(payload), payload)

    pack_download.download_pack(target_dir=target, session=session)

    assert session.part_requests == []


def test_download_deletes_a_part_with_a_bad_hash(sandbox):
    payload = b"expected" * 128
    session = _Session(_manifest(payload), b"corrupt!" * 128)
    target = sandbox / "dl"

    with pytest.raises(offline_pack.OfflinePackError, match="SHA-256"):
        pack_download.download_pack(target_dir=target, session=session)

    assert not (target / "gpu-pack-test-part01.zip").exists()
    assert not (target / "gpu-pack-test-part01.zip.part").exists()


def test_download_validates_manifest_before_fetching_parts(sandbox):
    payload = b"x" * 64
    session = _Session(_manifest(payload, platform="linux"), payload)

    with pytest.raises(offline_pack.OfflinePackError, match="Windows x64"):
        pack_download.download_pack(target_dir=sandbox / "dl", session=session)

    assert session.part_requests == []


def test_download_404_explains_the_release_is_empty(sandbox):
    class _Missing(_Session):
        def get(self, url, **kwargs):
            return _Response(status=404)

    session = _Missing(_manifest(b"x"), b"x")
    with pytest.raises(offline_pack.OfflinePackError, match="assets"):
        pack_download.download_pack(target_dir=sandbox / "dl", session=session)


def test_download_cancel_before_start_does_not_touch_the_network(sandbox):
    payload = b"z" * 4096
    session = _Session(_manifest(payload), payload)
    abort = threading.Event()
    abort.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        pack_download.download_pack(
            target_dir=sandbox / "dl", session=session, abort_event=abort
        )

    assert session.part_requests == []


def test_download_cancel_mid_part_keeps_it_for_resume(sandbox):
    payload = b"z" * (3 * 1024 * 1024)
    abort = threading.Event()
    session = _Session(_manifest(payload), payload, on_chunk=abort.set)
    target = sandbox / "dl"

    with pytest.raises(RuntimeError, match="cancelled"):
        pack_download.download_pack(
            target_dir=target, session=session, abort_event=abort
        )

    partial = target / "gpu-pack-test-part01.zip.part"
    assert partial.is_file()
    assert 0 < partial.stat().st_size < len(payload)
    assert not (target / "gpu-pack-test-part01.zip").exists()


def test_download_respects_offline_mode(sandbox, monkeypatch):
    monkeypatch.setenv(offline_pack.ENV_OFFLINE, "1")
    session = _Session(_manifest(b"x"), b"x")

    with pytest.raises(offline_pack.OfflinePackError, match="DRAGGY_OFFLINE"):
        pack_download.download_pack(target_dir=sandbox / "dl", session=session)

    assert session.manifest_requests == 0


def test_download_rejects_unsupported_platform(sandbox, monkeypatch):
    monkeypatch.setattr(pack_download, "pack_supported", lambda: False)
    session = _Session(_manifest(b"x"), b"x")

    with pytest.raises(offline_pack.OfflinePackError, match="Windows x64"):
        pack_download.download_pack(target_dir=sandbox / "dl", session=session)


def test_discard_downloaded_pack_only_removes_the_managed_folder(
    sandbox, monkeypatch
):
    managed = sandbox / "managed"
    managed.mkdir()
    manifest = managed / offline_pack.MANIFEST_NAME
    manifest.write_text("{}", encoding="utf-8")
    (managed / "part.zip").write_bytes(b"x")
    outside = sandbox / "keep.json"
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pack_download, "download_dir", lambda: str(managed))

    pack_download.discard_downloaded_pack(str(manifest))
    assert not managed.exists()

    pack_download.discard_downloaded_pack(str(outside))
    assert outside.exists()
