"""Almacenamiento seguro de datos personales (tokens, API keys, etc.).

Los datos se guardan en ``personal/`` (carpeta en .gitignore) como
archivos JSON individuales.  Cada archivo contiene un solo valor
serializado como JSON string.

Uso::

    from src.personal import get_token, set_token

    token = get_token()          # str o ""
    set_token("hf_abc123")       # guarda en personal/hf_token.json

El directorio ``personal/`` se crea automáticamente al primer写入.
"""

from __future__ import annotations

import json
import os

from src import config


_DIR = os.path.join(config.get_data_dir(), "personal")
_HF_TOKEN_FILE = "hf_token.json"


def _ensure_dir() -> None:
    os.makedirs(_DIR, exist_ok=True)


def _read_json(filename: str) -> dict:
    path = os.path.join(_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_json(filename: str, data: dict) -> bool:
    _ensure_dir()
    path = os.path.join(_DIR, filename)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# HuggingFace token
# ---------------------------------------------------------------------------

def get_token() -> str:
    """Devuelve el token de HuggingFace o una cadena vacía."""
    return _read_json(_HF_TOKEN_FILE).get("token", "")


def set_token(token: str) -> bool:
    """Guarda el token de HuggingFace. Devuelve True si se guardó."""
    return _write_json(_HF_TOKEN_FILE, {"token": token})


def has_token() -> bool:
    """True si hay un token configurado."""
    return bool(get_token())


def migrate_from_preferences() -> None:
    """Migra el token de preferences.json a personal/ si existe.

    Se ejecuta una sola vez; si el token ya está en personal/ no hace nada.
    """
    if has_token():
        return
    try:
        from src.preferences import get as _pref_get, set as _pref_set, has as _pref_has
        old_token = _pref_get("hf_api_token", "")
        if old_token:
            set_token(old_token)
            # Limpiar la clave antigua de preferences.json
            if _pref_has("hf_api_token"):
                _pref_set("hf_api_token", "")
    except Exception:
        pass
