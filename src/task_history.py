"""Historial persistente de tareas de render (compresión / preview).

Vive en ``bin/.session/tasks.json`` (carpeta gitignored). Un *task* es un
job de compresión o preview que el usuario ha lanzado; guarda su estado y
el de cada fase (engine de IA, audio boost, mux) para que el panel lateral
los muestre y reabra entre sesiones.

Diseño:
- Fail-safe E/S: cualquier excepción de disco devuelve ``False`` / lista
  vacía; nunca rompe el flujo principal del render.
- Append-style opaco: reescribimos el JSON completo con la lista acotada a
  ``TASK_HISTORY_LIMIT`` (los más recientes). Simple y suficiente para un
  historial de tareas.
- El worker GUI escribe ``start_task``/``update_task_phase``/``finish_task``
  conforme avanzan las fases reales del pipeline.

API:
- ``start_task(task_id, name, mode) -> bool``
- ``update_phase(task_id, phase_id, status, pct) -> bool``
- ``finish_task(task_id, status) -> bool``
- ``list_tasks(limit=None) -> list[dict]``
- ``get_task(task_id) -> dict | None``
- ``clear() -> bool``
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Optional

from src.config import ensure_runtime_dirs, get_session_dir


TASK_HISTORY_FILENAME = "tasks.json"
TASK_HISTORY_LIMIT = 100
_LOCK = threading.Lock()


def _path() -> str:
    return os.path.join(get_session_dir(), TASK_HISTORY_FILENAME)


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


def _save_raw(tasks: list) -> bool:
    """Persist task history atomically and durably."""
    tmp_path = None
    try:
        ensure_runtime_dirs()
        path = _path()
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tasks-", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path))
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
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


def _get_index(tasks: list, task_id: str) -> int:
    for i, task in enumerate(tasks):
        if task.get("task_id") == task_id:
            return i
    return -1


def start_task(task_id: str, name: str, input_path: str = "") -> bool:
    """Crea una tarea pendiente en el historial."""
    if not task_id:
        return False
    with _LOCK:
        tasks = _load_raw()
        if _get_index(tasks, task_id) >= 0:
            return True  # idempotente
        tasks.append({
            "task_id": str(task_id),
            "name": str(name or task_id),
            "input_path": str(input_path or ""),
            "status": "running",
            "created_at": _now(),
            "finished_at": None,
            "phases": [],
        })
        tasks = tasks[-TASK_HISTORY_LIMIT:]
        return _save_raw(tasks)


def update_phase(task_id: str, phase_id: str, label: str = "",
                 status: str = "pending", pct: int = 0) -> bool:
    """Upsert a single phase ({status,pct,label}) of a task."""
    with _LOCK:
        tasks = _load_raw()
        idx = _get_index(tasks, task_id)
        if idx < 0:
            return False
        task = tasks[idx]
        phases = task.setdefault("phases", [])
        for ph in phases:
            if ph.get("phase_id") == phase_id:
                if label:
                    ph["label"] = str(label)
                ph["status"] = str(status)
                ph["pct"] = int(pct)
                task["status"] = "running"
                return _save_raw(tasks)
        phases.append({
            "phase_id": str(phase_id),
            "label": str(label or phase_id),
            "status": str(status),
            "pct": int(pct),
        })
        return _save_raw(tasks)


def recover_interrupted_tasks() -> int:
    """Mark tasks still "running" from a previous session as interrupted.

    Called once at startup, before the UI builds the task list. A task that
    was mid-render when the app crashed or was force-closed stays "running"
    forever otherwise, and the tasks panel would show it as active even
    though no worker exists.
    """
    changed = 0
    with _LOCK:
        tasks = _load_raw()
        for task in tasks:
            if task.get("status") == "running" and not task.get("finished_at"):
                task["status"] = "interrupted"
                task["finished_at"] = _now()
                changed += 1
        if changed:
            _save_raw(tasks)
    return changed


def finish_task(task_id: str, status: str) -> bool:
    """Mark a task complete: ``status`` in ("success","failed","aborted")."""
    with _LOCK:
        tasks = _load_raw()
        idx = _get_index(tasks, task_id)
        if idx < 0:
            return False
        tasks[idx]["status"] = str(status)
        tasks[idx]["finished_at"] = _now()
        return _save_raw(tasks)


def list_tasks(limit: Optional[int] = None) -> list:
    """Most-recent-first, opcionalmente acotada a ``limit``."""
    tasks = _load_raw()
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    if limit is not None:
        return tasks[:limit]
    return tasks


def get_task(task_id: str) -> Optional[dict]:
    with _LOCK:
        for task in _load_raw():
            if task.get("task_id") == task_id:
                return dict(task)
    return None


def clear() -> bool:
    try:
        path = _path()
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception:
        return False