"""Registro dinámico de motores de usuario instalados desde el Hub.

El catálogo (``src.ai.registry.CATALOG``) es estático: lista literal en
código. Los modelos que el usuario encuentra con el buscador del Hub no
pueden vivir ahí, así que se registran aquí y se FUSIONAN con el catálogo
en ``registry.get_entry`` / ``registry._all_with_state`` (ver la docstring
de ``registry.py``: los puntos de entrada del catálogo son los que la UI
consume, así que fusionar en ellos hace que los motores de usuario
aparezcan en los nodos del workflow y en la pestaña AI Engines sin tocar
esas vistas).

State file layout (created on first write):

    bin/.session/user_models.json

        {
          "version": 1,
          "models": {
            "user-upscale-owner-4x-ultrasharp-a1b2c3d4": {
              "display_name": "4x UltraSharp",
              "category": "Upscale",
              "hf_repo_id": "owner/4x-UltraSharp",
              "hf_url": "https://huggingface.co/owner/4x-UltraSharp",
              "arch": "rrdb",            # detectado tras la descarga
              "scale": 4,                # detectado tras la descarga
              "weight_file": "model.pth",
              "created_at_iso": "2026-08-03T10:00:00+00:00"
            }
          }
        }

El estado installed/enabled de estos motores se guarda en
``engines_state.py`` (mismo engine_id): el sensor de disco y los
hardcodes FSR de engines_state no colisionan porque los ids de usuario
empiezan por ``user-`` (slug-safe, generado por ``hf_search.user_engine_id``).

Escrituras atómicas (temp + os.replace) y cache por mtime, igual que
``engines_state``.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from src.config import get_session_dir
from src.ai.registry import _assert_slug  # slug guard (misma política que engines_state)


_USER_MODELS_VERSION = 1
_USER_MODELS_FILENAME = "user_models.json"
_LOCK = threading.Lock()
# mtime_ns + size keyed -- ver docstring de engines_state._load.
_LOAD_CACHE: dict = {}  # path -> (mtime_ns, size, data)


def _path() -> str:
    return os.path.join(get_session_dir(), _USER_MODELS_FILENAME)


def _atomic_write(path: str, payload: dict) -> bool:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def _empty() -> dict:
    return {"version": _USER_MODELS_VERSION, "models": {}}


def _load() -> dict:
    p = _path()
    try:
        st = os.stat(p)
        mtime, size = st.st_mtime_ns, st.st_size
    except OSError:
        mtime, size = 0, 0
    cached = _LOAD_CACHE.get(p)
    if cached is not None and cached[0] == mtime and cached[1] == size:
        return copy.deepcopy(cached[2])
    try:
        if not os.path.exists(p):
            data = _empty()
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        # Preserva el corrupto (un solo model-de-usuario alterado ya es un
        # dolor de reconstruir) antes de resetar a estado vacío.
        try:
            os.replace(p, p + ".bak")
        except Exception:
            pass
        data = _empty()
    if not isinstance(data, dict):
        data = _empty()
    data["version"] = _USER_MODELS_VERSION
    data.setdefault("models", {})
    _LOAD_CACHE[p] = (mtime, size, data)
    return copy.deepcopy(data)


def _save(data: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(_path()) or ".", exist_ok=True)
    except Exception:
        pass
    return _atomic_write(_path(), data)


def is_user_engine(engine_id: str) -> bool:
    """True si el id fue generado por el buscador del Hub (prefijo ``user-``)."""
    return isinstance(engine_id, str) and engine_id.startswith("user-")


def all_models() -> dict:
    """Copia de todos los registros de motores de usuario."""
    return copy.deepcopy(_load().get("models", {}))


def get_model(engine_id: str) -> Optional[dict]:
    """Devuelve el registro crudo de un motor de usuario, o None."""
    return copy.deepcopy(_load().get("models", {}).get(engine_id))


def register(engine_id: str, **fields) -> bool:
    """Crea o actualiza un motor de usuario.

    Valida el slug (misma política que ``engines_state.patch``). El
    timestamp se actualiza siempre; el resto de campos se mergean.
    """
    if not engine_id:
        return False
    try:
        _assert_slug(engine_id)
    except ValueError as exc:
        print(f"[user_models] invalid engine_id: {exc}", file=__import__("sys").stderr)
        return False
    ok = False
    with _LOCK:
        data = _load()
        bucket = data.setdefault("models", {}).setdefault(engine_id, {})
        bucket.update(fields)
        bucket["updated_at_iso"] = datetime.now(timezone.utc).isoformat()
        ok = _save(data)
    return ok


def unregister(engine_id: str) -> bool:
    """Elimina un motor de usuario del registro (sin tocar engines_state)."""
    with _LOCK:
        data = _load()
        if engine_id not in data.get("models", {}):
            return False
        del data["models"][engine_id]
        ok = _save(data)
    if ok:
        try:
            from src.ai.engines_state import mark_uninstalled
            mark_uninstalled(engine_id)
        except Exception:
            pass
    return ok


def catalog_entries() -> list[dict]:
    """Devuelve los motores de usuario con la MISMA forma que el catálogo
    (schema de ``registry.CATALOG``) para fusionarlos en los puntos de
    entrada de ``src.ai.registry``.

    ``sentinel`` se setea al archivo de pesos detectado: la validación de
    instalación de ``dependencies_dialog._on_engine_install_ok`` comprueba
    el sentinel O ``dir_has_weights``, así que un repo con pesos en
    subcarpetas también pasa.
    """
    out = []
    for engine_id, rec in _load().get("models", {}).items():
        scale = rec.get("scale")
        out.append({
            "engine_id": engine_id,
            "display_name": rec.get("display_name", engine_id),
            "category": rec.get("category") or "Other",
            "hf_repo_id": rec.get("hf_repo_id", ""),
            "hf_url": rec.get("hf_url", ""),
            "size_estimate": "",
            "sample_shape": (
                f"1x3xHxW -> 1x3x({scale}H)x({scale}W)"
                if scale
                else "1x3xHxW -> 1x3xHxW"
            ),
            "license": rec.get("license", ""),
            "popularity_score": rec.get("popularity_score", 0),
            "quality_score": rec.get("quality_score", 0),
            "is_implemented": True,
            "sentinel": rec.get("weight_file") or "",
            "lfs_warning": False,
            "is_user": True,
        })
    return out


if __name__ == "__main__":
    print("User models path:", _path())
    print("Entries:", json.dumps(catalog_entries(), indent=2, sort_keys=True))
