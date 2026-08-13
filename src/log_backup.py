"""Rotación de crash reports a una carpeta de respaldo.

Cuando ``app.log`` crece demasiado, el ``RotatingFileHandler`` rota el fichero
y, al cabo de ``backupCount`` rotaciones, borra el más antiguo (``app.log.N``).
Para no perder esos crash reports, este módulo los archiva en
``bin/logs/backup/`` **antes** de que la rotación los borre, manteniendo los
últimos ``keep`` respaldos. En cada rotación también hace un snapshot de los
ficheros extra indicados (p. ej. ``faulthandler.log``, que contiene los
volcados de segfaults).

Diseño copy-only: los respaldos son **copias** (``shutil.copy2``), nunca se
truncan ni mueven los ficheros activos, así que no hay carrera con el stream
abierto del handler ni riesgo de corromper el log en uso.
"""

from __future__ import annotations

import os
import shutil
import time
from logging.handlers import RotatingFileHandler

from src.config import get_logs_dir

DEFAULT_KEEP = 10


def backup_dir():
    """Devuelve (creando si hace falta) ``bin/logs/backup``."""
    d = os.path.join(get_logs_dir(), "backup")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _timestamp():
    """Timestamp legible y único (microsegundos) para nombrar respaldos.

    La resolución de segundos colisionaría cuando varias rotaciones ocurren
    en ráfaga (mismo segundo) y ``copy2`` sobrescribiría el mismo destino;
    el sufijo de microsegundos lo evita y sigue siendo ordenable
    lexicográficamente (el orden coincide con el cronológico).
    """
    now = time.time()
    micros = int(now * 1_000_000) % 1_000_000
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{micros:06d}"


def archive_file(path, prefix, ext=".log"):
    """Copia ``path`` a ``backup/<prefix>-<timestamp><ext>``.

    Devuelve la ruta del respaldo creado o ``None`` si no hay nada que
    archivar o el copiado falla. Nunca lanza.
    """
    try:
        if not path or not os.path.isfile(path):
            return None
        dest = os.path.join(backup_dir(), f"{prefix}-{_timestamp()}{ext}")
        shutil.copy2(path, dest)
        return dest
    except OSError:
        return None


def backup_stats():
    """Devuelve ``(num_ficheros, total_bytes)`` de los respaldos en backup/.

    Tolerante a errores de E/S: cualquier fichero ilegible se salta y nunca
    lanza. Devuelve ``(0, 0)`` si la carpeta no existe o está vacía.
    """
    d = backup_dir()
    num = 0
    total = 0
    try:
        names = os.listdir(d)
    except OSError:
        return 0, 0
    for name in names:
        try:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                num += 1
                total += os.path.getsize(path)
        except OSError:
            pass
    return num, total


def clear_backups():
    """Borra todos los ficheros de backup/. Devuelve cuántos borró. Nunca lanza."""
    d = backup_dir()
    removed = 0
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    for name in names:
        try:
            os.remove(os.path.join(d, name))
            removed += 1
        except OSError:
            pass
    return removed


def rotate(prefix, keep=DEFAULT_KEEP):
    """Borra los respaldos más antiguos de ``prefix`` dejando ``keep``.

    Ordena lexicográficamente (el timestamp ``YYYYMMDD-HHMMSS`` es
    monotónico), así que los más viejos quedan al principio.
    """
    d = backup_dir()
    try:
        names = sorted(n for n in os.listdir(d) if n.startswith(f"{prefix}-"))
    except OSError:
        return
    for name in names[: max(0, len(names) - keep)]:
        try:
            os.remove(os.path.join(d, name))
        except OSError:
            pass


class BackupRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler que archiva el fichero rotado más antiguo antes de
    que ``doRollover`` lo borre, y mantiene rotada la carpeta de respaldo.

    ``backup_prefix`` nombra los respaldos en ``backup/``; ``extra_snapshots``
    es una secuencia de tuplas ``(path, prefix)`` de ficheros adicionales que
    se snapshotean en cada rotación (p. ej. ``faulthandler.log``).
    """

    def __init__(
        self,
        *args,
        backup_prefix="app",
        keep=DEFAULT_KEEP,
        extra_snapshots=(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._backup_prefix = backup_prefix
        self._keep = keep
        self._extra_snapshots = tuple(extra_snapshots)

    def doRollover(self):
        # El fichero ``baseFilename.backupCount`` es el que la rotación normal
        # borra/sobrescribe; lo archivamos primero (y rotamos los respaldos).
        if self.backupCount > 0:
            oldest = f"{self.baseFilename}.{self.backupCount}"
            archive_file(oldest, self._backup_prefix, ".log")
            for path, prefix in self._extra_snapshots:
                archive_file(path, prefix, ".log")
            rotate(self._backup_prefix, self._keep)
            for _, prefix in self._extra_snapshots:
                rotate(prefix, self._keep)
        super().doRollover()
