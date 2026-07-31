"""Snapshot del último job de compress/preview que el usuario disparó.

Almacenado en `bin/.session/last_job.json` (carpeta gitignored). Se
sobreescribe silenciosamente cada vez que el usuario hace click en
Compress o Preview en MainWindow, **antes** de lanzar el
``CompressionWorker``. La idea: si el usuario cierra la app justo
cuando tenía un job a punto de ejecutar y hay updates pendientes, la
propuesta de cierre ofrece "Update + Compress + close" y reanuda ese
job con los settings fotografiados, no los del SettingsPanel actual
(que pueden haber cambiado).

Diseño:
- Single-most-recent (no histórico). La última voluntad del usuario
  siempre gana; el historial persistente vive en ``update_history``.
- Safe-fail E/S: cualquier excepción de disco no rompe el flujo
  principal (la compresión continúa aunque no podamos persistir).
- Schema estable: ``mode``, ``input_path`` (no vacío), ``settings``
  (dict). ``output_path`` puede ser ``None`` para preview.

API:
- ``save_last_job(*, mode, input_path, output_path, settings) -> bool``
- ``load_last_job() -> dict | None``
- ``has_last_job() -> bool``
- ``clear() -> bool``
- ``count() -> int``
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from src.config import ensure_runtime_dirs, get_session_dir


LAST_JOB_FILENAME = "last_job.json"


def _path() -> str:
    return os.path.join(get_session_dir(), LAST_JOB_FILENAME)


def save_last_job(
    *,
    mode: str,
    input_path: str,
    output_path: Optional[str],
    settings: dict,
    ai_engines: Optional[dict] = None,
) -> bool:
    """Persiste el snapshot del último job. Sobre-escribe entradas previas.

    Reglas:
    - ``mode`` debe ser ``"compress"`` o ``"preview"``. Cualquier otro
      valor retorna ``False`` sin escribir: preferimos no persistir data
      dudosa en vez de contaminar el archivo.
    - ``settings`` se clona vía ``dict(...)`` para evitar aliasing con
      estructuras mutables del caller.
    - ``ai_engines`` es opcional; cuando está presente, es un dict
      ``{"Upscale": "engine_id", ...}`` que el followup "Update +
      Compress + close" usará para restaurar la elección de IA junto
      con los settings clásicos. Se clona para evitar aliasing.
    - Fail-safe: cualquier excepción (JSON, I/O) se silencia devolviendo
      ``False``. No interrumpimos el flujo principal de compresión.
    """
    if mode not in ("compress", "preview"):
        return False
    try:
        ensure_runtime_dirs()
        payload = {
            "mode": str(mode),
            "input_path": str(input_path) if input_path else "",
            "output_path": str(output_path) if output_path else None,
            "settings": dict(settings or {}),
            "ai_engines": dict(ai_engines) if ai_engines else {},
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_last_job() -> Optional[dict]:
    """Carga ``last_job.json`` o ``None`` si falta/inválido.

    Validador ligero:
    - debe ser ``dict``;
    - ``mode`` ∈ {compress, preview};
    - ``input_path``: string no vacío;
    - ``settings``: ``dict``.

    ``output_path`` se acepta como ``None`` (modo preview) sin Validarlo
    de más — el caller verifica paths antes del run.
    """
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("mode") not in ("compress", "preview"):
            return None
        input_path = data.get("input_path")
        if not isinstance(input_path, str) or not input_path:
            return None
        settings = data.get("settings")
        if not isinstance(settings, dict):
            return None
        # Tolerate older snapshots that pre-date the ai_engines field;
        # missing values become an empty dict so callers can iterate
        # without special-casing.
        if not isinstance(data.get("ai_engines"), dict):
            data["ai_engines"] = {}
        return data
    except Exception:
        return None


def has_last_job() -> bool:
    """``True`` si hay un last_job válido. No devuelve el dict."""
    try:
        return load_last_job() is not None
    except Exception:
        return False


def clear() -> bool:
    """Borra el archivo. Pensado para Reset/QA o tras falla permanente."""
    try:
        path = _path()
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception:
        return False


def count() -> int:
    """Siempre 0 o 1 (single-most-recent); sanity check."""
    try:
        return 1 if os.path.exists(_path()) else 0
    except Exception:
        return 0
