"""El disco: lo que miden y lo que borran el registro y los dos managers.

Este módulo existe por una razón concreta y no por simetría.
``main.py`` llama a ``process_pending_cleanup()`` al arrancar, ANTES de
inyectar los runtimes, para poder borrar los DLL que Windows tenía
bloqueados en la sesión anterior. Esa garantía solo es real si el código
del arranque no arrastra consigo la app: hasta ahora el arranque importaba
``src.dependencies``, cuyo ámbito de módulo importa los dos managers que
precisamente hay que borrar. Que no importase torch era suerte del orden de
imports, no una decisión del sitio de la llamada.

Aquí viven dos cosas, y son la misma:

* **Medir**: ``safe_dir_size`` — de cuánto ocupa un árbol en disco. Estaba
  duplicado, casi idéntico byte a byte, en ``src.dependencies`` y en
  ``src.runtimes.manager``. Ahora hay una sola copia.
* **Borrar aunque no se pueda**: ``rmtree_force`` borra un árbol y, si
  Windows deja algo bloqueado, lo anota en
  ``bin/.session/pending_cleanup.json`` para el próximo arranque. El
  fichero guarda, además de la ruta, la marca de creación del directorio:
  si entre la desinstalación y el reinicio el usuario reinstaló el mismo
  runtime, la ruta apunta ya a la instalación NUEVA y borrarla se llevaría
  por delante lo recién puesto. Al arrancar se compara la marca y, si no
  cuadra, la entrada se descarta.

Ni el registro ni los managers saben nada de esto: ellos preguntan «borra
esto» y reciben ``True``/``False``. Que la respuesta sea «se ha programado
para el próximo arranque» lo decide este módulo, que es donde vive el
formato del fichero y sus reglas.
"""

from __future__ import annotations

import json
import os
import shutil

from src.config import ensure_runtime_dirs

# Borrado robusto + cleanup pendiente (archivos DLL bloqueados en Windows)
# ---------------------------------------------------------------------------

PENDING_CLEANUP_FILENAME = "pending_cleanup.json"


def _pending_cleanup_path():
    from src.config import get_session_dir
    return os.path.join(get_session_dir(), PENDING_CLEANUP_FILENAME)


def _path_signature(path: str) -> int:
    """Marca de creación de ``path`` (0 si no existe o no se puede leer).

    En Windows ``st_ctime_ns`` no cambia al añadir archivos dentro, así que
    sirve para distinguir "el directorio bloqueado que registramos" de "un
    directorio NUEVO que se ha re-creado en la misma ruta".
    """
    try:
        return int(os.stat(path).st_ctime_ns)
    except OSError:
        return 0


def register_pending_cleanup(path: str) -> None:
    """Registra una ruta para borrarla en el próximo arranque (best-effort).

    Windows bloquea DLLs cargadas (torch/onnxruntime ya importados). No
    podemos borrarlos en caliente; al siguiente arranque, antes de importar
    cualquier runtime, ``process_pending_cleanup()`` completa el borrado.

    Se guarda además la marca de creación del directorio. Es imprescindible:
    si el usuario desinstala (queda bloqueado), REINSTALA el mismo runtime y
    luego reinicia, el borrado pendiente apuntaría a la ruta de la instalación
    NUEVA y se la llevaría por delante. Comparando la marca se detecta que ese
    directorio ya no es el que se quería borrar y se respeta.
    """
    try:
        ensure_runtime_dirs()
        p = _pending_cleanup_path()
        data = {}
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if not isinstance(data, dict):
            data = {}
        entries = data.setdefault("paths", [])
        entry = {"path": path, "sig": _path_signature(path)}
        # Reemplaza una entrada previa de la misma ruta (re-registro).
        entries[:] = [
            e for e in entries
            if (e.get("path") if isinstance(e, dict) else e) != path
        ]
        entries.append(entry)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _pending_entry_path(entry) -> str:
    """Ruta de una entrada del fichero de limpiezas pendientes.

    Acepta el formato antiguo (una ruta suelta) y el nuevo (``{path, sig}``).
    """
    if isinstance(entry, dict):
        return str(entry.get("path") or "")
    return str(entry or "")


def _pending_entry_signature(entry) -> int:
    """Marca de creación registrada (0 si el formato no la tiene)."""
    if isinstance(entry, dict):
        try:
            return int(entry.get("sig") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _remove_empty_dir_tree(path: str) -> bool:
    """Borra ``path`` si está vacío (o solo contiene dirs vacíos).

    Los DLL bloqueados impiden borrar SUS archivos, pero el resto del árbol sí
    se puede limpiar; esto retira la carcasa vacía en vez de dejarla para
    siempre. Devuelve True si el directorio desapareció.
    """
    if not os.path.exists(path):
        return True
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            return False
        return not os.path.exists(path)
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            try:
                os.remove(os.path.join(dirpath, name))
            except OSError:
                pass
        for name in dirnames:
            try:
                os.rmdir(os.path.join(dirpath, name))
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        return False
    return True


def process_pending_cleanup() -> None:
    """Completa borrados pendientes del arranque anterior.

    DEBE llamarse antes de importar ``torch``/``onnxruntime`` (en ``main.py``
    justo después de inyectar los runtimes, antes de importarlos) para que los
    DLLs no estén todavía bloqueados. Idempotente y tolerante a errores de E/S.

    Una entrada se descarta (se considera hecha) si la ruta ya no existe o si
    el directorio que hay ahora NO es el que se registró: eso significa que el
    runtime se reinstaló y borrarlo destruiría una instalación válida.
    """
    p = _pending_cleanup_path()
    if not os.path.exists(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        data = {}
    entries = list(data.get("paths") or [])
    remaining = []
    for entry in entries:
        path = _pending_entry_path(entry)
        if not path:
            continue
        if not os.path.exists(path):
            continue  # ya no está: nada que hacer
        expected = _pending_entry_signature(entry)
        if expected and _path_signature(path) != expected:
            continue  # re-creado: es una instalación nueva, se respeta
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
        if os.path.exists(path) and not _remove_empty_dir_tree(path):
            remaining.append(entry)
    try:
        if remaining:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"paths": remaining}, f, indent=2)
        else:
            os.remove(p)
    except Exception:
        pass


def rmtree_force(path: str) -> bool:
    """Borra recursivamente ``path``; si queda algo bloqueado, lo programa
    para el próximo arranque. Devuelve True si el borrado quedó completo."""
    if not os.path.isdir(path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return not os.path.exists(path)
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
    if os.path.exists(path):
        register_pending_cleanup(path)
        return False
    return True


# ---------------------------------------------------------------------------
# Medir
# ---------------------------------------------------------------------------


def safe_dir_size(path: str) -> int:
    """Bytes que ocupa ``path`` en disco; 0 si no existe o no se puede leer."""
    try:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total
    except Exception:
        return 0
