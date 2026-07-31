"""Historial persistente de los últimos ``HISTORY_LIMIT`` updates.

Vive en ``bin/.session/update_history.json`` (carpeta gitignored). Cada
entrada registra el antes/después de una actualización para que el
usuario pueda revisarla en la pestaña **History** del diálogo de
preferencias.

Diseño:
- FIFO: ``record_update`` inserta al principio y trunca a ``HISTORY_LIMIT``.
- Tolerancia a fallos: cualquier excepción de E/S se silencia para que
  un fallo persistencia NO rompa el flujo principal de actualización.
- Schema estricto: cada entrada debe incluir los campos canonicos. Las
  claves opcionales (``old_version`` / ``new_version`` / ``old_bytes`` /
  ``new_bytes``) admiten ``None`` para instalaciones fresh o mediciones
  que fallen (p. ej. manifest corrupto).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from src.config import ensure_runtime_dirs, get_session_dir


HISTORY_FILENAME = "update_history.json"
HISTORY_LIMIT = 10


def _path() -> str:
    return os.path.join(get_session_dir(), HISTORY_FILENAME)


def _load() -> list[dict]:
    try:
        ensure_runtime_dirs()
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save(entries: list[dict]) -> bool:
    try:
        ensure_runtime_dirs()
        with open(_path(), "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _entry(
    *,
    dep_id: str,
    display_name: str,
    old_version: Optional[str],
    new_version: Optional[str],
    duration_seconds: float,
    old_bytes: Optional[int],
    new_bytes: Optional[int],
    success: bool,
) -> dict:
    """Normaliza y valida una entrada antes de persistirla."""
    from datetime import datetime, timezone
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dep_id": str(dep_id or "unknown"),
        "display_name": str(display_name or dep_id or "unknown"),
        "old_version": old_version if old_version else None,
        "new_version": new_version if new_version else None,
        "duration_seconds": float(max(0.0, duration_seconds)),
        "old_bytes": int(old_bytes) if old_bytes is not None else None,
        "new_bytes": int(new_bytes) if new_bytes is not None else None,
        "success": bool(success),
    }


def record_update(
    *,
    dep_id: str,
    display_name: str,
    old_version: Optional[str],
    new_version: Optional[str],
    duration_seconds: float,
    old_bytes: Optional[int],
    new_bytes: Optional[int],
    success: bool,
) -> Optional[dict]:
    """Persiste una nueva entrada y devuelve la entrada guardada.

    Inserta al principio (la más reciente primero) y trunca para mantener
    solo las últimas ``HISTORY_LIMIT``. Devuelve ``None`` si no se pudo
    persistir.
    """
    entry = _entry(
        dep_id=dep_id,
        display_name=display_name,
        old_version=old_version,
        new_version=new_version,
        duration_seconds=duration_seconds,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        success=success,
    )
    entries = _load()
    entries.insert(0, entry)
    if len(entries) > HISTORY_LIMIT:
        entries = entries[:HISTORY_LIMIT]
    if _save(entries):
        return entry
    return None


def get_recent(limit: int = HISTORY_LIMIT) -> list[dict]:
    """Devuelve una copia de los últimos ``limit`` registros (más recientes primero)."""
    entries = _load()
    if limit <= 0 or limit >= len(entries):
        return list(entries)
    return list(entries[:limit])


def clear() -> bool:
    """Borra el historial. Pensado para QA / debug / 'Reset' en preferencias."""
    try:
        path = _path()
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception:
        return False


def count() -> int:
    """Número de registros actualmente persistidos."""
    return len(_load())
