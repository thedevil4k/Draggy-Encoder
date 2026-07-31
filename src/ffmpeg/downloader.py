import logging
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)

from src.config import (
    FFMPEG_DOWNLOAD_URLS,
    FFMPEG_FULL_DOWNLOAD_URLS,
    get_ffmpeg_dir,
    get_ffmpeg_full_dir,
    get_ffmpeg_executable_name,
    get_ffprobe_executable_name,
    get_logs_dir,
)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Single-stream: chunk size + connect/read timeouts.
_DOWNLOAD_CHUNK_SIZE = 256 * 1024
_DOWNLOAD_CONNECT_TIMEOUT = 15
_DOWNLOAD_READ_TIMEOUT = 30
_PROGRESS_EMIT_INTERVAL_S = 0.1

# Parallel: smaller chunk per worker = smoother progress + earlier fails.
_PARALLEL_CHUNK_SIZE = 128 * 1024
_PARALLEL_DEFAULT_WORKERS = 8
_PARALLEL_MAX_WORKERS = 16
_PARALLEL_MIN_FILE_SIZE = 16 * 1024 * 1024  # only parallelize files >= 16 MB


# ---------------------------------------------------------------------------
# Polyfill del callback de progreso
# ---------------------------------------------------------------------------

def _emit(progress_callback, percent, message=""):
    """Polyfill del callback: acepta 1-arg (legado) o 2-args (nuevo).

    Si registrado para (pct) llama con un entero. Si para (pct, msg)
    llama con ambos. Cualquier excepción del callback se silencia para no
    tumbar la descarga.
    """
    if not progress_callback:
        return
    try:
        try:
            progress_callback(int(percent), message)
        except TypeError:
            progress_callback(int(percent))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Probe de capacidad (HEAD)
# ---------------------------------------------------------------------------

def _probe_url_metadata(url):
    """HEAD probe -> (total_size, supports_range).

    - total_size: Content-Length si está y es un entero positivo, si no None.
    - supports_range: True sólo si el upstream declara Accept-Ranges: bytes.

    Best-effort: cualquier error de red devuelve (None, False) para que el
    caller caiga al camino single-stream sin romper el arranque.
    """
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=(_DOWNLOAD_CONNECT_TIMEOUT, 10),
        )
        response.raise_for_status()
        total_str = response.headers.get("content-length", "") or ""
        total = int(total_str) if total_str.isdigit() else 0
        accept = (response.headers.get("accept-ranges") or "").strip().lower()
        supports_range = accept == "bytes"
        return (total if total > 0 else None, supports_range)
    except Exception:
        return None, False


# ---------------------------------------------------------------------------
# Single-stream (fallback)
# ---------------------------------------------------------------------------

def _download_file(url, dest_path, progress_callback=None):
    """Descarga single-stream con throttle cada 0.1 s.

    Bound de stall: con timeout=(15, 30) y chunks de 256 KB, una red
    totalmente muerta falla en <= 30 s por cada 256 KB (en la práctica
    cada read toma < 1 s). El caller puede interrumpir la UI con un
    mensaje claro cuando se cumpla un timeout local.
    """
    response = requests.get(
        url,
        stream=True,
        timeout=(_DOWNLOAD_CONNECT_TIMEOUT, _DOWNLOAD_READ_TIMEOUT),
    )
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    _emit(progress_callback, 0, "Downloading FFmpeg...")
    last_emit_ts = 0.0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if total > 0 and (now - last_emit_ts) >= _PROGRESS_EMIT_INTERVAL_S:
                _emit(
                    progress_callback,
                    downloaded * 100 / total,
                    "Downloading FFmpeg...",
                )
                last_emit_ts = now


# ---------------------------------------------------------------------------
# Parallel (HTTP Range)
# ---------------------------------------------------------------------------

def _download_parallel(
    url,
    dest_path,
    total_size,
    progress_callback=None,
    num_workers=_PARALLEL_DEFAULT_WORKERS,
):
    """Descarga paralela mediante HTTP Range requests.

    Estrategia:
      1. Pre-asignar ``dest_path`` con ``os.truncate(total_size)``.
      2. Dividir el espacio de bytes en ``num_workers`` rangos contiguos.
      3. Cada worker crea su propio ``requests.Session`` (connection pool
         aislado) y descarga su rango con ``Range: bytes=A-B``.
      4. Escribir con ``seek + write`` por worker. Como los rangos NO se
         solapan, no hace falta lock sobre el fichero; solo hace falta
         lock sobre el contador agregado de progreso.
      5. Cada worker emite un progreso cuando su propio porcentaje
         agregado cruza un umbral de 1 %: emisión auto-throttled sin
         coordinación lockeada.

    Pre-condiciones (el caller las garantiza tras el probe):
      - ``total_size > 0``.
      - El servidor declara ``Accept-Ranges: bytes`` (HEAD probe).

    Fail-fast:
      - El status esperado es 206 Partial Content. Si el servidor ignora
        el header Range y devuelve 200 OK con el archivo entero, lo
        escribimos solo en el rango asignado y el resto queda a ceros
        -> datos corruptos sin salvar. Por eso el chequeo es estricto.
      - Cualquier ReadTimeout o HTTP error propaga la primera excepción
        encontrada; las workers pendientes se cancelan best-effort vía
        ``ThreadPoolExecutor.shutdown(wait=False, cancel_futures=True)``
        (Python 3.9+). En 3.8 o menor, hacemos ``shutdown(wait=False)``
        y dejamos que terminen al soltar la GIL.

    Args:
        url: URL upstream.
        dest_path: ruta de destino. Será truncada y rellenada.
        total_size: tamaño total esperado (bytes) ya conocido vía HEAD.
        progress_callback: callable(pct, msg).
        num_workers: 4-8 típico; capeado a 16 para no abusar de CDNs.
    """
    if total_size <= 0:
        raise RuntimeError("_download_parallel: total_size invalido")
    if num_workers <= 1:
        return _download_file(
            url, dest_path, progress_callback=progress_callback
        )
    num_workers = min(max(num_workers, 2), _PARALLEL_MAX_WORKERS)

    # 1) Pre-asignar el fichero destino. Crea un sparse file de tamaño
    #    exacto; los workers escribirán cada uno su rango sin solapar.
    with open(dest_path, "wb") as f:
        f.truncate(total_size)

    # 2) Calcular rangos contiguos. El último worker absorbe cualquier
    #    resto para que la suma cuadre exactamente con total_size.
    each = total_size // num_workers
    remainder = total_size - each * num_workers
    ranges = []
    start = 0
    for w in range(num_workers):
        end = start + each - 1
        if w == num_workers - 1:
            end += remainder
        ranges.append((start, end))
        start = end + 1

    # 3) Contador compartido con lock (las escrituras al fichero son
    #    independientes; solo el contador agregado necesita lock).
    counter_lock = threading.Lock()
    progress_counter = {"downloaded": 0}

    def _fetch_range(range_start, range_end):
        session = requests.Session()
        try:
            response = session.get(
                url,
                headers={"Range": "bytes={}-{}".format(range_start, range_end)},
                stream=True,
                timeout=(_DOWNLOAD_CONNECT_TIMEOUT, _DOWNLOAD_READ_TIMEOUT),
            )
            response.raise_for_status()
            # Si el servidor ignora Range y devuelve 200 con el archivo
            # entero, abortamos para no corromper el sparse file.
            if response.status_code != 206:
                raise RuntimeError(
                    "Server returned status {} en respuesta a Range; "
                    "esperado 206 Partial Content.".format(response.status_code)
                )
            local_pct = -1
            with open(dest_path, "rb+") as f:
                f.seek(range_start)
                for chunk in response.iter_content(
                    chunk_size=_PARALLEL_CHUNK_SIZE
                ):
                    if not chunk:
                        continue
                    f.write(chunk)
                    with counter_lock:
                        progress_counter["downloaded"] += len(chunk)
                        cur = progress_counter["downloaded"]
                    pct_global = min(99, int(cur * 100 / total_size))
                    if pct_global != local_pct:
                        _emit(
                            progress_callback,
                            pct_global,
                            "Downloading FFmpeg...",
                        )
                        local_pct = pct_global
        finally:
            try:
                session.close()
            except Exception:
                pass

    # 4) Spawn workers.
    with ThreadPoolExecutor(
        max_workers=num_workers, thread_name_prefix="ffmpegdl"
    ) as ex:
        futures = [
            ex.submit(_fetch_range, range_start, range_end)
            for range_start, range_end in ranges
        ]
        try:
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    try:
                        ex.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        ex.shutdown(wait=False)
                    raise exc
        except Exception:
            raise


# ---------------------------------------------------------------------------
# Búsqueda de ejecutables tras extracción
# ---------------------------------------------------------------------------

def _find_executables(root_dir):
    ffmpeg = None
    ffprobe = None
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower() == get_ffmpeg_executable_name().lower():
                ffmpeg = os.path.join(dirpath, filename)
            elif filename.lower() == get_ffprobe_executable_name().lower():
                ffprobe = os.path.join(dirpath, filename)
    return ffmpeg, ffprobe


# ---------------------------------------------------------------------------
# Orquestador público
# ---------------------------------------------------------------------------

def download_ffmpeg(progress_callback=None, parent=None):
    """Descarga FFmpeg para la plataforma actual.

    Estrategia:
      1. HEAD probe -> tamaño total + soporte de Range.
      2. Si el archivo es >= 16 MB y el upstream acepta Range, descarga
         paralela con N workers (default 8). Cada worker tiene su propia
         Session (connection pool), escribe rangos contiguos no
         solapados con seek+write. Counter compartido con lock para
         reporte de progreso agregado.
      3. Si el probe falla, el archivo es pequeño o el upstream no
         soporta Range, fallback a single-stream legacy (la implementación
         previa). En la práctica para gyan.dev (Windows, ~130 MB) y
         johnvansickle.com (Linux, ~30 MB), siempre acabamos en el camino
         paralelo.

    Fases de progreso reportadas al callback (percent, message):
      * 0   -> "Connecting to upstream..."
      * 0-90 -> "Downloading FFmpeg..." (single o parallel)
      * 92  -> "Extracting <archive_type> archive..."
      * 98  -> "Installing binaries..."
      * 100 -> "FFmpeg ready"

    Devuelve la ruta absoluta al ejecutable ``ffmpeg`` dentro de
    ``bin/ffmpeg/`` (vía ``get_ffmpeg_dir()``). Lanza ``RuntimeError`` si
    la plataforma no tiene URL conocida, si la extracción falla, o si no
    se localiza el binario.
    """
    system = platform.system()
    if system not in FFMPEG_DOWNLOAD_URLS:
        raise RuntimeError("Plataforma no soportada: {}".format(system))

    info = FFMPEG_DOWNLOAD_URLS[system]
    url = info["url"]
    archive_type = info["archive_type"]

    bin_dir = get_ffmpeg_dir()
    os.makedirs(bin_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "ffmpeg_download.{}".format(archive_type))

        # Probe de capacidad (HEAD). En el caso típico (gyan.dev /
        # johnvansickle.com) devuelve un total ~130 MB / ~30 MB y
        # supports_range=True. Si devuelve None/False, fallback seguro
        # al camino single-stream.
        _emit(progress_callback, 0, "Connecting to upstream...")
        size, supports_range = _probe_url_metadata(url)

        if size is not None and size >= _PARALLEL_MIN_FILE_SIZE and supports_range:
            # Camino rápido: 8 workers en paralelo con HTTP Range. Si falla
            # (ReadTimeout, 206 mismatch por Range ignorado, excepción de
            # socket, etc.) caemos automáticamente a single-stream para no
            # mostrar un critical dialog al usuario por un fallo que
            # single-stream habría capeado sin despeinarse.
            try:
                _download_parallel(
                    url,
                    archive_path,
                    total_size=size,
                    progress_callback=progress_callback,
                    num_workers=_PARALLEL_DEFAULT_WORKERS,
                )
            except Exception as exc:
                _emit(
                    progress_callback,
                    0,
                    "Parallel download failed; switching to single-stream...",
                )
                # Importante: el archivo parcial queda inconsistente (un
                # mismatch 206 pudo haber escrito bytes en regiones del
                # sparse file que no corresponden a su rango). Borrar para
                # que single-stream empiece desde cero y el zip/tar.xz
                # extractor no explote por offsets corruptos.
                try:
                    if os.path.exists(archive_path):
                        os.remove(archive_path)
                except OSError:
                    pass
                _download_file(url, archive_path, progress_callback=progress_callback)
        else:
            # Fallback single-stream (ficheros pequeños o sin soporte Range).
            _download_file(url, archive_path, progress_callback=progress_callback)

        _emit(progress_callback, 92, "Extracting {} archive...".format(archive_type))
        extract_dir = os.path.join(tmpdir, "ffmpeg_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            if archive_type == "zip":
                with zipfile.ZipFile(archive_path, "r") as z:
                    z.extractall(extract_dir)
            elif archive_type == "tar.xz":
                with tarfile.open(archive_path, "r:xz") as t:
                    t.extractall(extract_dir)
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
            raise RuntimeError("Archive extraction failed: {}".format(exc)) from exc

        ffmpeg_path, ffprobe_path = _find_executables(extract_dir)
        if not ffmpeg_path:
            raise RuntimeError("No se encontró el ejecutable de FFmpeg tras extraer el archivo.")

        _emit(progress_callback, 98, "Installing binaries...")
        dest_ffmpeg = os.path.join(bin_dir, get_ffmpeg_executable_name())
        dest_ffprobe = os.path.join(bin_dir, get_ffprobe_executable_name())
        shutil.copy2(ffmpeg_path, dest_ffmpeg)
        if ffprobe_path:
            shutil.copy2(ffprobe_path, dest_ffprobe)

        if system != "Windows":
            try:
                os.chmod(dest_ffmpeg, 0o755)
            except OSError:
                pass
            if os.path.exists(dest_ffprobe):
                try:
                    os.chmod(dest_ffprobe, 0o755)
                except OSError:
                    pass

    _emit(progress_callback, 100, "FFmpeg ready")
    return dest_ffmpeg


def _download_and_extract_archive(url, archive_type, tmpdir, archive_name, progress_callback):
    """Download and extract an archive to a temp directory."""
    archive_path = os.path.join(tmpdir, archive_name)

    _emit(progress_callback, 0, "Connecting to FFmpeg full upstream...")
    size, supports_range = _probe_url_metadata(url)

    if size is not None and size >= _PARALLEL_MIN_FILE_SIZE and supports_range:
        try:
            _download_parallel(
                url,
                archive_path,
                total_size=size,
                progress_callback=progress_callback,
                num_workers=_PARALLEL_DEFAULT_WORKERS,
            )
        except Exception as exc:
            _emit(progress_callback, 0, "Parallel download failed; switching to single-stream...")
            try:
                if os.path.exists(archive_path):
                    os.remove(archive_path)
            except OSError:
                pass
            _download_file(url, archive_path, progress_callback=progress_callback)
    else:
        _download_file(url, archive_path, progress_callback=progress_callback)

    _emit(progress_callback, 92, "Extracting {} archive...".format(archive_type))
    extract_dir = os.path.join(tmpdir, "ffmpeg_full_extracted")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        if archive_type in ("zip", "7z"):
            try:
                with zipfile.ZipFile(archive_path, "r") as z:
                    z.extractall(extract_dir)
            except zipfile.BadZipFile:
                # Try 7z command
                import subprocess
                try:
                    subprocess.run(
                        ["7z", "x", archive_path, "-o{}".format(extract_dir), "-y"],
                        check=True,
                        capture_output=True,
                        timeout=120,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError):
                    raise RuntimeError(
                        "Cannot extract .7z archive. Please install 7-Zip or use a different FFmpeg build."
                    )
        elif archive_type == "tar.xz":
            with tarfile.open(archive_path, "r:xz") as t:
                t.extractall(extract_dir)
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        raise RuntimeError("Archive extraction failed: {}".format(exc)) from exc

    return extract_dir


def download_ffmpeg_full(progress_callback=None, parent=None):
    """Download FFmpeg full (with libplacebo) for FSR 2.0 support.

    Downloads from BtbN lgpl-shared build (smaller, includes libplacebo).
    Falls back to gpl-shared, then johnvansickle.com static build.

    Returns the path to the FFmpeg executable or raises RuntimeError.
    """
    system = platform.system()
    if system not in FFMPEG_FULL_DOWNLOAD_URLS:
        raise RuntimeError("Platform not supported for FFmpeg full: {}".format(system))

    info = FFMPEG_FULL_DOWNLOAD_URLS[system]
    primary_url = info["url"]
    archive_type = info.get("archive_type", "tar.xz")
    backup_url = info.get("backup_url")
    backup_archive_type = info.get("backup_archive_type", archive_type)
    backup_url_2 = info.get("backup_url_2")
    backup_archive_type_2 = info.get("backup_archive_type_2", archive_type)

    bin_dir = get_ffmpeg_full_dir()
    os.makedirs(bin_dir, exist_ok=True)

    # Build ordered list of (url, archive_type, label) candidates
    candidates = [
        (primary_url, archive_type, "primary (BtbN lgpl-shared)"),
    ]
    if backup_url:
        candidates.append((backup_url, backup_archive_type, "backup-1 (BtbN gpl-shared)"))
    if backup_url_2:
        candidates.append((backup_url_2, backup_archive_type_2, "backup-2 (johnvansickle.com)"))

    last_error = None
    extract_dir = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (url, atype, label) in enumerate(candidates):
            _emit(progress_callback, 0, "Downloading FFmpeg full ({})...".format(label))
            archive_name = "ffmpeg_full_{}.{}".format(idx, atype)
            try:
                extract_dir = _download_and_extract_archive(
                    url, atype, tmpdir, archive_name, progress_callback,
                )
                break
            except Exception as exc:
                last_error = exc
                _emit(progress_callback, 0, "Failed ({}): {}".format(label, exc))
                continue

        if extract_dir is None:
            raise RuntimeError(
                "All FFmpeg full download sources failed. Last error: {}".format(last_error)
            )

        ffmpeg_path, ffprobe_path = _find_executables(extract_dir)
        if not ffmpeg_path:
            raise RuntimeError("No se encontro el ejecutable de FFmpeg full tras extraer el archivo.")

        _emit(progress_callback, 98, "Installing FFmpeg full binaries...")
        # BtbN shared builds ship DLLs alongside ffmpeg.exe (typically in a
        # `bin/` subdir of the archive). Static builds have everything baked
        # into the executable. To cover both layouts — and to capture all
        # third-party deps that may sit in sibling dirs (libplacebo, vulkan,
        # shaderc, ...), we walk the whole extracted tree and copy every
        # DLL / shared lib that lives at the same level as the executables.
        ffmpeg_dir = os.path.dirname(os.path.abspath(ffmpeg_path))
        _emit(progress_callback, 99, "Installing FFmpeg full binaries + DLLs...")
        copied = 0
        # Walk one level deep around ffmpeg.exe so we cover both the
        # `bin/` directory of BtbN layouts and any sibling dir that holds
        # plugin DLLs.
        candidate_dirs = {ffmpeg_dir}
        for entry in os.listdir(ffmpeg_dir):
            sub = os.path.join(ffmpeg_dir, entry)
            if os.path.isdir(sub):
                candidate_dirs.add(sub)
        # Also walk top-level dirs of the archive (BtbN's bin/, lib/, etc.).
        for entry in os.listdir(extract_dir):
            sub = os.path.join(extract_dir, entry)
            if os.path.isdir(sub):
                candidate_dirs.add(sub)

        for src_dir in candidate_dirs:
            for entry in os.listdir(src_dir):
                src = os.path.join(src_dir, entry)
                if not os.path.isfile(src):
                    continue
                low = entry.lower()
                keep = (
                    low.endswith(".dll")
                    or ".so" in low
                    or low.endswith(".dylib")
                    or low == get_ffmpeg_executable_name().lower()
                    or low == get_ffprobe_executable_name().lower()
                )
                if not keep:
                    continue
                dst = os.path.join(bin_dir, entry)
                if os.path.exists(dst):
                    # Already present (e.g. ffmpeg.exe picked up from the
                    # first dir we walked). Skip rather than overwrite with
                    # a potentially identical copy.
                    continue
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except OSError:
                    pass

        dest_ffmpeg = os.path.join(bin_dir, get_ffmpeg_executable_name())
        dest_ffprobe = os.path.join(bin_dir, get_ffprobe_executable_name())
        # Belt-and-suspenders: make sure the EXEs are present even if the
        # listdir above missed them (e.g. older shared layouts).
        if not os.path.exists(dest_ffmpeg):
            shutil.copy2(ffmpeg_path, dest_ffmpeg)
            copied += 1
        if ffprobe_path and not os.path.exists(dest_ffprobe):
            shutil.copy2(ffprobe_path, dest_ffprobe)
            copied += 1

        if system != "Windows":
            try:
                os.chmod(dest_ffmpeg, 0o755)
            except OSError:
                pass
            if os.path.exists(dest_ffprobe):
                try:
                    os.chmod(dest_ffprobe, 0o755)
                except OSError:
                    pass

    _emit(progress_callback, 100, "FFmpeg full ready")
    return dest_ffmpeg
