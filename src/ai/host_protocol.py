"""Protocolo binario entre la GUI y el host de IA (proceso aparte).

**Por qué existe.** Cargar torch/onnxruntime DENTRO del proceso congelado de
la GUI (PyInstaller + PySide6) mata la aplicación: las extensiones nativas
(torch_cpu.dll, torch_cuda.dll) provocan una violación de acceso
(0xc0000005 en python313.dll) y Windows registra ``Application Error 1000``
sin traceback. La app "no arranca" y no hay nada que el usuario pueda hacer.

**Cómo.** Todo trabajo que necesite esos runtimes ocurre en un proceso
aparte: el *host* (``src.ai.host``), lanzado con el Python privado del
instalador (``embed-python`` en Windows, ``runtime-python`` en Linux). La GUI
solo habla este protocolo por ``stdin``/``stdout``: JSON para lo de siempre y
blobs binarios crudos para los frames (numpy ``uint8`` HWC), sin base64 ni
serialización intermedia.

Formato de un mensaje (cliente y servidor usan el mismo):

    +----------------+---------------------+------------------------+
    | 4 bytes BE     | cabecera JSON (N)   | blobs (sum(lengths))   |
    | N (cab. len)   | UTF-8, sin \n       | bytes crudos, en orden |
    +----------------+---------------------+------------------------+

La cabecera lleva ``blob_lengths`` (tamaño de cada blob) y ``blob_meta``
(dtype/forma por blob, para reconstruir el array). Un error se devuelve como
``{"ok": false, "error": "..."}``, nunca como un mensaje distinto: el cliente
tiene una única ruta de respuesta y así no hay respuestas colgadas.
"""

from __future__ import annotations

import json
import struct

#: Sube esto si cambia el formato del mensaje; el cliente rechaza un host con
#: otra versión (un host viejo y un cliente nuevo no deben hablar).
PROTOCOL_VERSION = 1

#: Cabecera más grande que esto = stream desincronizado o corrupto.
MAX_HEADER_BYTES = 32 * 1024 * 1024

_HEADER_STRUCT = struct.Struct(">I")


def pack_message(header: dict, blobs=()) -> bytes:
    """Serializa una cabecera + blobs al formato del protocolo."""
    header = dict(header)
    header["v"] = PROTOCOL_VERSION
    header["blob_lengths"] = [len(blob) for blob in blobs]
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HEADER_STRUCT.pack(len(payload)) + payload + b"".join(bytes(b) for b in blobs)


def write_message(stream, header: dict, blobs=()) -> None:
    """Escribe un mensaje completo en un stream binario (flush inmediato)."""
    stream.write(pack_message(header, blobs))
    stream.flush()


def _read_exactly(stream, count: int) -> bytes:
    """Lee exactamente ``count`` bytes o falla con ``EOFError``."""
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"EOF en el stream del host (faltaban {remaining} bytes)")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(stream) -> tuple[dict, list]:
    """Lee un mensaje completo: ``(cabecera, [blobs])``."""
    raw_len = _read_exactly(stream, _HEADER_STRUCT.size)
    (length,) = _HEADER_STRUCT.unpack(raw_len)
    if length <= 0 or length > MAX_HEADER_BYTES:
        raise ValueError(f"cabecera de {length} bytes fuera de rango")
    header = json.loads(_read_exactly(stream, length).decode("utf-8"))
    lengths = list(header.get("blob_lengths") or [])
    blobs = [_read_exactly(stream, int(n)) for n in lengths]
    return header, blobs


# ---------------------------------------------------------------------------
# Arrays (frames) como blobs
# ---------------------------------------------------------------------------

def encode_array(array) -> tuple[dict, bytes]:
    """``(meta, bytes)`` de un array numpy: crudo, C-contiguo, sin copia de más."""
    meta = {
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
    }
    return meta, array.tobytes(order="C")


def decode_array(meta: dict, blob: bytes):
    """Reconstruye el array numpy descrito por ``meta`` (importa numpy aquí)."""
    import numpy as np

    dtype = np.dtype(meta["dtype"])
    array = np.frombuffer(blob, dtype=dtype)
    shape = tuple(int(dim) for dim in meta.get("shape") or ())
    if shape:
        array = array.reshape(shape)
    # ``frombuffer`` devuelve un array de solo lectura sobre el buffer: el
    # pipeline reescribe frames (padding, fits de aspecto), así que se
    # garantiza escritura con una copia explícita solo si hace falta.
    return array if array.flags.writeable else np.array(array, copy=True)
