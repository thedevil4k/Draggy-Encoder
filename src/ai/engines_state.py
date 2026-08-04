"""On/off, installed/converted flags for each AI engine, persisted.

State file layout (created on first write):

    bin/.session/engines.json

        {
          "version": 1,
          "engines": {
            "upscale-realesrgan-x4plus": {
              "enabled": true,
              "installed": true,
              "installed_at_iso": "2026-07-22T14:32:00+00:00",
              "install_path": "D:/DRAGGY_ENCODER/bin/models/Upscale/realesrgan-x4plus",
              "onnx_converted": false,
              "onnx_path": null,
              "last_error": null
            },
            ...
          }
        }

We store the path as derived from ``src.ai.registry.get_engine_dir``
so multiple clones never disagree about where to look for the
weights. We also store a ``last_error`` field so a failed install
attempt leaves a breadcrumb for the next UI render (without crashing
it on read).

The schema is forward-compatible: the ``version`` field lets us
migrate cleanly if we add fields later.

Atomic writes: every save goes through a temp file + ``os.replace``
so a crash mid-write cannot leave a half-written JSON that breaks
future reads.

Thread-safety: ``patch`` (and hence ``mark_installed``/``mark_onnx``
etc.) takes a module-level lock around the read-modify-write so two
simultaneous toggles can't race on disk. Reads (``get_state``,
``all_states``) need no lock because ``os.replace`` is atomic at
the filesystem level and reads return either the pre-write or
post-write state.

Slug validation: every mutation path (engines_state.patch) calls
``src.ai.registry._assert_slug`` to reject any engine_id that does
not match ``^[a-z0-9]+(?:-[a-z0-9]+)*$``. This hardens against
path traversal if/when engine_ids become user-editable.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

from src.config import ensure_runtime_dirs, get_session_dir
from src.ai.registry import _assert_slug  # slug guard for mutation paths


_STATE_VERSION = 1
_STATE_FILENAME = "engines.json"
_STATE_LOCK = threading.Lock()

# mtime-keyed parse cache: engines.json is read+parsed once per on-disk
# change. Rendering the AI Engines tab / workflow calls ``get_state``
# dozens of times per paint; without this every call re-reads the file.
_LOAD_CACHE: dict = {}  # path -> (mtime_ns, data)


# ---------------------------------------------------------------------------
# Path / IO
# ---------------------------------------------------------------------------


def _path() -> str:
    return os.path.join(get_session_dir(), _STATE_FILENAME)


def _ensure_parent_dir() -> None:
    try:
        ensure_runtime_dirs()
    except Exception:
        pass


def _atomic_write(path: str, payload: dict) -> bool:
    """Write JSON atomically (temp file + os.replace)."""
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


def _empty_state() -> dict:
    return {"version": _STATE_VERSION, "engines": {}}


def _load() -> dict:
    """Read the state file. Returns empty state if missing or corrupt.

    Results are cached by file mtime (nanoseconds) + size: under heavy
    disk I/O, a float mtime can repeat between an atomic replace and the
    next read, which would serve a stale copy. ``st_mtime_ns`` has
    nanosecond resolution so a collision requires an identical-size write
    within the same nanosecond — effectively impossible.
    Callers must not mutate the returned dict (use ``patch`` instead,
    which persists and is the only writer). Copies are returned so
    accidental mutation of a cached payload cannot poison later reads.
    """
    _ensure_parent_dir()
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
            data = _empty_state()
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        # No tiene sentido perder un vino multi-engine por un bit-flip:
        # preserva el corrupto junto al original para futura recuperación
        # manual antes de resetar a estado vacío.
        try:
            os.replace(p, p + ".bak")
        except Exception:
            pass
        data = _empty_state()
    if not isinstance(data, dict):
        data = _empty_state()
    if data.get("version") != _STATE_VERSION:
        # Future: migration hook. For now just accept and tag.
        data["version"] = _STATE_VERSION
    data.setdefault("engines", {})
    _LOAD_CACHE[p] = (mtime, size, data)
    return copy.deepcopy(data)


def _save(data: dict) -> bool:
    _ensure_parent_dir()
    return _atomic_write(_path(), data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_state(engine_id: str) -> dict:
    """Return the persisted state for ``engine_id``, or default if absent.

    ``get_state`` deliberately tolerates unknown ids: callers want to
    render the catalog in one pass without crashing on missing state.
    Slug validation is enforced in the mutation paths (``patch`` and
    the ``mark_*`` wrappers) because those write to disk.

    Default values:
        enabled            = False  (opt-in; prevents surprises)
        installed          = False
        installed_at_iso   = None
        install_path       = None   (caller computes via registry)
        onnx_converted     = False
        onnx_path          = None
        last_error         = None

    FSR engines (upscale-fsr-*) are always installed+enabled because
    they are pure-algorithmic (no model download required).
    """
    data = _load()
    raw = data.get("engines", {}).get(engine_id) or {}
    state = {
        "enabled": bool(raw.get("enabled", False)),
        "installed": bool(raw.get("installed", False)),
        "installed_at_iso": raw.get("installed_at_iso"),
        "install_path": raw.get("install_path"),
        "onnx_converted": bool(raw.get("onnx_converted", False)),
        "onnx_path": raw.get("onnx_path"),
        "last_error": raw.get("last_error"),
    }
    # FSR 1.0 engines are always available (no download needed)
    if engine_id and engine_id.startswith("upscale-fsr") and not engine_id.endswith("-v2") and "fsr31" not in engine_id:
        state["installed"] = True
        state["enabled"] = True

    # FSR 2.0 and FSR 3.0 engines require FFmpeg full with libplacebo
    if engine_id and (engine_id.endswith("-v2") or "fsr31" in engine_id):
        # Check if FFmpeg full is available
        try:
            from src.ffmpeg.manager import has_libplacebo_support, get_local_ffmpeg_full_path
            ffmpeg_full = get_local_ffmpeg_full_path()
            if os.path.exists(ffmpeg_full) and has_libplacebo_support(ffmpeg_full):
                state["installed"] = True
                state["enabled"] = True
            else:
                state["installed"] = False
                state["enabled"] = False
                state["last_error"] = "Requires FFmpeg full with libplacebo support"
        except Exception:
            state["installed"] = False
            state["enabled"] = False
            state["last_error"] = "Cannot check FFmpeg full availability"

    # Sensor veraz contra disco: si el estado dice "instalado" pero el
    # directorio del modelo ya no existe (el usuario borró/trasladó los
    # pesos a mano, o un borrado quedó a medias), reportamos installed=False
    # y corregimos engines.json para que la UI nunca muestre un INSTALLED
    # fantasma. FSR 1.0/2.0/3.1 se excluyen: no usan directorio de modelos
    # (2.0/3.1 usan el binario FFmpeg full, que ya se validó arriba).
    if (
        engine_id
        and state.get("installed")
        and not engine_id.startswith("upscale-fsr")
    ):
        try:
            from src.ai.registry import get_engine_dir
            model_dir = get_engine_dir(engine_id)
            stored = state.get("install_path")
            # Auto-reparación: instalaciones antiguas guardaron rutas en
            # minúsculas ("bin/models/upscale/...") que en Windows se
            # resuelven igual, pero en Linux (case-sensitive) rompen la
            # detección. Escribimos siempre la ruta canónica del catálogo.
            if stored and os.path.normpath(stored) != os.path.normpath(model_dir):
                try:
                    patch(engine_id, install_path=model_dir)
                except Exception:
                    pass
            if not model_dir or not os.path.isdir(model_dir):
                state["installed"] = False
                state["enabled"] = False
                state["onnx_converted"] = False
                state["onnx_path"] = None
                try:
                    patch(
                        engine_id,
                        installed=False,
                        enabled=False,
                        installed_at_iso=None,
                        install_path=None,
                        onnx_converted=False,
                        onnx_path=None,
                        last_error="Model folder missing on disk.",
                    )
                except Exception:
                    pass
            elif not dir_has_weights(model_dir):
                # El directorio existe pero solo contiene código fuente
                # (clone de un repo git sin pesos) o archivos de config.
                # Marcarlo instalado haría fallar el render con
                # "No model weights found" sin explicación.
                state["installed"] = False
                state["enabled"] = False
                state["onnx_converted"] = False
                state["onnx_path"] = None
                try:
                    patch(
                        engine_id,
                        installed=False,
                        enabled=False,
                        installed_at_iso=None,
                        install_path=None,
                        onnx_converted=False,
                        onnx_path=None,
                        last_error=(
                            "Model folder has no weight files "
                            "(only source code or configs)."
                        ),
                    )
                except Exception:
                    pass
        except Exception:
            pass
    return state


def all_states() -> dict:
    """Return a copy of the entire engines state, normalised per engine.

    Engines with no record are still present, with default values.
    Use this in the UI to render every row of the catalog in one pass.
    Incluye los motores de usuario registrados desde el Hub.
    """
    data = _load()
    out = {}
    try:
        from src.ai.registry import CATALOG
        ids = [e["engine_id"] for e in CATALOG]
    except Exception:
        ids = list(data.get("engines", {}).keys())
    try:
        from src.ai.user_models import all_models as _all_user_models
        for uid in _all_user_models():
            if uid not in ids:
                ids.append(uid)
    except Exception:
        pass
    for eid in ids:
        out[eid] = get_state(eid)
    return out


WEIGHT_EXTS = (".pth", ".pt", ".safetensors", ".onnx", ".bin", ".gguf")


def dir_has_weights(directory: str) -> bool:
    """True si ``directory`` contiene algún archivo de pesos de modelo.

    Recorre recursivamente (los repos HF suelen tener pesos en
    subcarpetas). Un directorio con solo código fuente (``.py``,
    ``README.md``, configs) devuelve False: eso NO es una instalación.
    """
    if not directory or not os.path.isdir(directory):
        return False
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(WEIGHT_EXTS):
                return True
    return False


def patch(engine_id: str, **fields) -> bool:
    """Merge ``fields`` into the persisted state for ``engine_id``.

    Slug-validates the id first; tags ``updated_at_iso``. The
    read-modify-write is serialised under ``_STATE_LOCK`` so two
    concurrent toggle clicks cannot race. After a successful save we
    emit ``state_changed`` on the AI bus so subscribers (workflow
    nodes, the dependencies dialog) re-paint without polling.
    """
    if not engine_id:
        return False
    try:
        _assert_slug(engine_id)
    except ValueError as exc:
        # Bad ids are a programming error; log to stderr but do not
        # crash the GUI - rather, the persistence silently fails.
        print(f"[engines_state] invalid engine_id: {exc}", file=sys.stderr)
        return False
    ok = False
    with _STATE_LOCK:
        data = _load()
        bucket = data.setdefault("engines", {}).setdefault(engine_id, {})
        bucket.update(fields)
        bucket["updated_at_iso"] = datetime.now(timezone.utc).isoformat()
        ok = _save(data)
    # Emit on the bus OUTSIDE the lock. Signal handlers can be slow
    # (UI repaint, file dialog) and we don't want them holding the
    # lock. Bus connections use Qt.UniqueConnection so duplicates
    # do not stack even on widget rebuilds.
    if ok:
        try:
            from src.ai.bus import bus as _state_bus
            _state_bus.state_changed.emit(engine_id)
        except Exception:
            # Bus emit failure must not break persistence.
            pass
    return ok


def mark_installed(engine_id: str, install_path: Optional[str] = None) -> bool:
    """Convenience wrapper used by the fetcher after a successful clone."""
    return patch(
        engine_id,
        installed=True,
        installed_at_iso=datetime.now(timezone.utc).isoformat(),
        install_path=install_path,
        last_error=None,
    )


def mark_uninstalled(engine_id: str) -> bool:
    """Used when the user removes weights; keeps enabled as-is."""
    return patch(
        engine_id,
        installed=False,
        installed_at_iso=None,
        install_path=None,
        onnx_converted=False,
        onnx_path=None,
        last_error=None,
    )


def mark_failed(engine_id: str, message: str) -> bool:
    """Used when the fetcher fails; leaves installed=False."""
    return patch(engine_id, installed=False, last_error=message)


# ---------------------------------------------------------------------------
# Benchmarks (ms/frame por engine, cacheado en el mismo engines.json)
# ---------------------------------------------------------------------------
#
# El ETA del render lee estos datos sin re-medir: se escriben tras una
# instalación (worker en background) o manualmente. La clave top-level
# ``benchmarks`` no interfiere con ``engines`` (get_state/all_states la
# ignoran).

def get_benchmark(engine_id: str) -> Optional[dict]:
    """Benchmark cacheado para ``engine_id``, o None.

    Forma: ``{"ms_per_frame": float, "fps": float, "resolution": [w, h],
    "device": str, "measured_at_iso": str}``.
    """
    data = _load()
    return copy.deepcopy(data.get("benchmarks", {}).get(engine_id)) or None


def set_benchmark(engine_id: str, data: dict) -> bool:
    """Almacena (o sobrescribe) el benchmark de ``engine_id``.

    ``data`` debe ser un dict JSON-serializable; si es None/{} se elimina
    la entrada. Misma política de escritura atómica + lock que ``patch``.
    """
    if not engine_id:
        return False
    try:
        _assert_slug(engine_id)
    except ValueError as exc:
        print(f"[engines_state] invalid engine_id: {exc}", file=sys.stderr)
        return False
    ok = False
    with _STATE_LOCK:
        data_all = _load()
        bucket = data_all.setdefault("benchmarks", {})
        if data:
            bucket[engine_id] = data
        else:
            bucket.pop(engine_id, None)
        ok = _save(data_all)
    return ok


def clear_benchmark(engine_id: str) -> bool:
    """Elimina el benchmark de un engine (p.ej. al desinstalar)."""
    return set_benchmark(engine_id, None)


def mark_onnx(engine_id: str, onnx_path: Optional[str] = None,
              converted: bool = True) -> bool:
    return patch(
        engine_id,
        onnx_converted=converted,
        onnx_path=onnx_path,
        last_error=None if converted else "conversion cancelled",
    )


def reset_all() -> bool:
    """Wipe the entire engines.json. Used as last-resort in dev tools."""
    return _save(_empty_state())


if __name__ == "__main__":
    print("Engines state path:", _path())
    print("Recorded:", json.dumps(all_states(), indent=2, sort_keys=True))
