"""Cola persistente de renders programables (modo video2x).

Vive en ``bin/.session/queue.json`` (carpeta gitignored). Cada item es un
snapshot completo e independiente de lo que el usuario configuró al pulsar
"Add to queue": video de entrada, ajustes de compresión, cadena de IA y
ruta de salida. Se ejecuta en secuencia cuando el usuario pulsa
"Start queue" (un render a la vez, GPU única), continuando tras fallos.

Diseño:
- Fail-safe E/S: cualquier excepción de disco devuelve ``False`` / lista
  vacía; nunca rompe el flujo principal.
- Append-style opaco: reescribimos el JSON completo acotado a
  ``TASK_QUEUE_LIMIT``. Simple y suficiente.
- Duplicados permitidos (mismo video + misma salida): cada item es un
  job propio, el usuario puede repetirlo conscientemente.
- Al cargar, los items en ``running`` vuelven a ``queued`` (la app se
  cortó a mitad de un render).

API:
- ``enqueue(*, name, input_path, output_path, settings, ai_engines,
  pipeline_order) -> str`` (devuelve el nuevo id)
- ``remove(queue_id) -> bool``
- ``cancel(queue_id) -> str`` (cancela UN item: ver sus códigos)
- ``clear() -> bool``
- ``list_items(limit=None) -> list[dict]`` (orden de encolado)
- ``get_item(queue_id) -> dict | None``
- ``set_status(queue_id, status, error=None) -> bool``
- ``first_pending() -> dict | None``
- ``count() -> int``
"""

from __future__ import annotations

import json
import os
import threading
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.config import ensure_runtime_dirs, get_session_dir


TASK_QUEUE_FILENAME = "queue.json"
TASK_QUEUE_LIMIT = 500
_VALID_STATUS = {
    "queued", "validating", "preparing", "running", "muxing",
    "publishing", "done", "failed", "aborted", "recoverable",
}
# Estados en los que el item ya no renderiza nada. Es publico porque es la
# definicion unica de "terminado": la UI lo usa para decidir que filas puede
# quitar de la lista, y duplicar el set aqui garantia que los dos se
# desincronizasen el dia que se añadiera un estado.
TERMINAL_STATUS = frozenset({"done", "failed", "aborted"})
_ALLOWED_TRANSITIONS = {
    "queued": {"queued", "validating", "preparing", "running", "failed", "aborted"},
    "validating": {"validating", "preparing", "running", "failed", "aborted"},
    "preparing": {"preparing", "running", "failed", "aborted"},
    "running": {"running", "muxing", "done", "failed", "aborted"},
    "muxing": {"muxing", "publishing", "failed", "aborted"},
    "publishing": {"publishing", "done", "failed", "aborted"},
    "recoverable": {"recoverable", "queued", "failed", "aborted"},
    "done": {"done"},
    "failed": {"failed"},
    "aborted": {"aborted"},
}
# Estados en los que el item ya tiene un render en marcha detrás. Cancelar
# aquí no es borrar una fila: es parar ese render, y por eso ``cancel`` NO
# borra el item sino que lo deja marcado como ``aborted`` (quien llama es el
# responsable de detener el proceso; al cerrarse, el worker vuelve por su
# propio camino y la fila queda coherente).
_ACTIVE_STATUS = {"validating", "preparing", "running", "muxing", "publishing"}
_LOCK = threading.Lock()


def _path() -> str:
    return os.path.join(get_session_dir(), TASK_QUEUE_FILENAME)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_raw() -> list:
    """Lee el archivo crudo. Nunca lanza."""
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict)]
    except Exception:
        return []


def _save_raw(items: list) -> bool:
    """Persist the queue atomically so a crash cannot corrupt it."""
    tmp_path = None
    try:
        ensure_runtime_dirs()
        path = _path()
        fd, tmp_path = tempfile.mkstemp(
            prefix=".queue-", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path))
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        return True
    except Exception:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def _get_index(items: list, queue_id: str) -> int:
    for i, item in enumerate(items):
        if item.get("queue_id") == queue_id:
            return i
    return -1


def enqueue(*, name: str, input_path: str, output_path: str,
            settings: dict, ai_engines: Optional[dict] = None,
            pipeline_order: Optional[list] = None) -> str:
    """Añade un render a la cola. Devuelve el id generado (o "" si falla)."""
    input_path = str(input_path or "")
    if not input_path:
        return ""
    queue_id = str(uuid.uuid4())
    with _LOCK:
        items = _load_raw()
        items.append({
            "queue_id": queue_id,
            "name": str(name or os.path.basename(input_path) or input_path),
            "input_path": input_path,
            "output_path": str(output_path or ""),
            "settings": dict(settings or {}),
            "ai_engines": dict(ai_engines) if ai_engines else {},
            "pipeline_order": list(pipeline_order) if pipeline_order else [],
            "status": "queued",
            "error": "",
            "created_at": _now(),
            "finished_at": None,
        })
        items = items[-TASK_QUEUE_LIMIT:]
        if not _save_raw(items):
            return ""
    return queue_id


def list_items(limit: Optional[int] = None) -> list:
    """Items en orden de encolado (el primero es el primero en ejecutarse)."""
    items = _load_raw()
    if limit is not None:
        return items[:limit]
    return items


def get_item(queue_id: str) -> Optional[dict]:
    with _LOCK:
        for item in _load_raw():
            if item.get("queue_id") == queue_id:
                return dict(item)
    return None


def set_status(queue_id: str, status: str, error: str = "") -> bool:
    """Advance one queue item through its validated lifecycle."""
    if status not in _VALID_STATUS:
        return False
    with _LOCK:
        items = _load_raw()
        idx = _get_index(items, queue_id)
        if idx < 0:
            return False
        current = str(items[idx].get("status") or "queued")
        if status not in _ALLOWED_TRANSITIONS.get(current, {current}):
            return False
        items[idx]["status"] = str(status)
        items[idx]["error"] = str(error or "")
        items[idx].setdefault("state_history", []).append({
            "from": current,
            "to": status,
            "at": _now(),
            "error": str(error or ""),
        })
        if status in TERMINAL_STATUS:
            items[idx]["finished_at"] = _now()
        return _save_raw(items)


def recover_interrupted() -> int:
    """Move active jobs back to ``queued`` for a safe retry.

    This must be called once during application startup, not on every queue
    read: changing a live ``running`` job while the UI polls it would create
    duplicate renders.
    """
    changed = 0
    with _LOCK:
        items = _load_raw()
        for item in items:
            status = item.get("status")
            if status in {"validating", "preparing", "running", "muxing", "publishing"}:
                item["status"] = "queued"
                item["error"] = "Recovered after an interrupted render; queued for retry."
                item["recovery_count"] = int(item.get("recovery_count", 0) or 0) + 1
                item.setdefault("state_history", []).append({
                    "from": status,
                    "to": "queued",
                    "at": _now(),
                    "error": item["error"],
                })
                changed += 1
        if changed:
            _save_raw(items)
    return changed


def first_pending() -> Optional[dict]:
    """El primer item ``queued`` de la cola, o ``None`` si no hay."""
    for item in list_items():
        if item.get("status") == "queued":
            return dict(item)
    return None


def count() -> int:
    return len(list_items())


def is_active(queue_id: str) -> bool:
    """¿El item tiene un render en marcha ahora mismo?"""
    item = get_item(queue_id)
    return bool(item) and str(item.get("status")) in _ACTIVE_STATUS


def cancel(queue_id: str) -> str:
    """Cancela UN item de la cola sin tocar los demás.

    Devuelve qué pasó, para que la UI sepa si tiene que hacer algo más:

    - ``"cancelled"``: estaba esperando su turno; sale de la cola.
    - ``"aborted"``: estaba en marcha. El item queda marcado ``aborted``
      (NO se borra: el render sigue vivo hasta que quien llama lo para,
      y su cierre necesita un item al que escribirle el final).
    - ``"finished"``: ya estaba en un estado terminal; nada que cancelar.
    - ``"not_found"`` / ``"error"``: no existe o no se pudo guardar.

    Quitar la fila a pelo (``remove``) con un render en marcha dejaba al
    worker escribiendo un archivo que la app ya no contaba; por eso el
    borrado y la parada van en la misma decisión.
    """
    with _LOCK:
        items = _load_raw()
        idx = _get_index(items, queue_id)
        if idx < 0:
            return "not_found"
        current = str(items[idx].get("status") or "queued")
        if current in _ACTIVE_STATUS:
            items[idx]["status"] = "aborted"
            items[idx]["error"] = "Cancelled by user."
            items[idx]["finished_at"] = _now()
            items[idx].setdefault("state_history", []).append({
                "from": current,
                "to": "aborted",
                "at": _now(),
                "error": "Cancelled by user.",
            })
            return "aborted" if _save_raw(items) else "error"
        if current in TERMINAL_STATUS:
            return "finished"
        items.pop(idx)
        return "cancelled" if _save_raw(items) else "error"


def remove(queue_id: str) -> bool:
    with _LOCK:
        items = _load_raw()
        idx = _get_index(items, queue_id)
        if idx < 0:
            return False
        items.pop(idx)
        return _save_raw(items)


def purge_finished() -> int:
    """Borra TODOS los items ya terminados (done/failed/aborted) de golpe.

    Devuelve cuantos se fueron (0 si no habia ninguno o si no se pudo
    guardar). Los items "queued" y los que tienen un render en marcha
    (``_ACTIVE_STATUS``) se quedan intactos: quitarles la fila por la
    espalda dejaria al worker escribiendo un archivo que la app ya no
    cuenta, que es justo lo que ``cancel`` evita item a item.
    """
    with _LOCK:
        items = _load_raw()
        kept = [
            it for it in items
            if str(it.get("status") or "queued") not in TERMINAL_STATUS
        ]
        removed = len(items) - len(kept)
        if removed <= 0:
            return 0
        return removed if _save_raw(kept) else 0


def clear() -> bool:
    try:
        path = _path()
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception:
        return False