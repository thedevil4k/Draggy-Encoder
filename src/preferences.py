"""Preferencias persistentes del usuario.

Se almacenan en ``bin/.session/preferences.json`` (carpeta gitignored).
La API es trivial para que el resto del programa pueda consultar/guardar
valores sin importar ``json`` ni ``os`` cada vez.

Diseño:
- Lectura sincrónica al inicio (sin caché): si el archivo está corrupto o
  no existe, devuelve ``default``.
- Escritura tolerante a fallos: errores de E/S se silencian para no romper
  el flujo principal (la preferencia simplemente no se persiste).
- Semántica merge: ``set(k, v)`` no borra las demás claves.
"""

from __future__ import annotations

import json
import os

from src.config import ensure_runtime_dirs, get_session_dir


PREFS_FILENAME = "preferences.json"


def _path():
    return os.path.join(get_session_dir(), PREFS_FILENAME)


def _load():
    p = None
    try:
        ensure_runtime_dirs()
        p = _path()
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        if p and os.path.exists(p):
            try:
                os.replace(p, p + ".bak")
            except Exception:
                pass
        return {}


def _save(data):
    try:
        ensure_runtime_dirs()
        tmp = _path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _path())
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def get(key, default=None):
    """Devuelve ``preferences[key]`` o ``default`` si no existe / falla la lectura."""
    return _load().get(key, default)


def set(key, value):
    """Persiste ``preferences[key] = value``. Mantiene las demás claves."""
    data = _load()
    data[key] = value
    return _save(data)


def has(key):
    return key in _load()


def all() -> dict:
    """Devuelve una copia de todas las preferencias."""
    return dict(_load())
