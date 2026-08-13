"""Cadena de auto-actualización multi-dependencia.

Este modulo contiene:

- ``_DependencyUpdateWorker``: ``QThread`` que re-instala UNA dependencia
  concreta. Es generico: recibe ``dep_id`` y ``reinstall_fn`` por el
  constructor, sin acoplarse al modulo ``src.ffmpeg``. Para FFmpeg, el
  caller sigue siendo responsable de invalidar el cache upstream antes
  de llamar a ``force_redownload`` (porque el cache vive en
  ``src.ffmpeg.upstream``, no en ``src.dependencies``).

- ``_UpdateChainController``: ``QObject`` que recorre una lista de
  dependencias pendientes en orden del ``REGISTRY`` y va arrancando un
  ``_DependencyUpdateWorker`` cada vez. Vive en el hilo principal; solo
  spawnea un thread a la vez. Al terminar toda la cadena emite
  ``chain_finished(summary)``.

- ``notify_log(kind, title, body)``: escribe un paso del proceso a
  ``bin/logs/notifications.log`` con icono ``[INFO ]`` / ``[WARN ]`` /
  ``[ERROR]``. Se usa para que el panel de dependencias pueda mostrar el
  historial de actualizaciones al abrirlo, sin depender del system tray.

Patrones usados (alineados con ``src/ui/main_window.py``):
- Workers con ``progress``, ``finished_ok``, ``failed`` y ``finished``.
- Flags de estado (``_running``, ``_current_index``) para evitar
  re-entranzas.
- Señales Qt para comunicacion thread-safe; nunca se tocan widgets
  desde el background thread.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from PySide6.QtCore import QObject, QThread, Signal


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

NOTIFICATIONS_LOG_FILENAME = "notifications.log"

_KIND_ICON = {
    "info": "[INFO ]",
    "warn": "[WARN ]",
    "error": "[ERROR]",
}


def _read_bytes(dep: dict):
    """Lee el tamano de la unidad logica de una dependencia (en bytes).

    Fuente preferida: ``dep["bytes_fn"]`` (callable opcional registrado en
    ``REGISTRY`` que mide la unidad logica: binario FFmpeg, modelo, runtime).
    Fallback: ``dep["status_fn"]()["size_bytes"]`` que reporta el tamano
    completo del ``install_dir`` (menos preciso pero siempre disponible).

    Devuelve ``None`` si ambas fallan o devuelven ``None``: el controller
    propaga ese ``None`` al paso del historial para mantener el delta
    coherente (no fuerza 0 que falsearia el resto de la entrada).
    """
    fn = dep.get("bytes_fn")
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    try:
        info = dep["status_fn"]() or {}
    except Exception:
        info = {}
    return info.get("size_bytes")


def _log_path():
    # Importacion local para no introducir ciclos en import-time.
    from src.config import get_logs_dir
    return os.path.join(get_logs_dir(), NOTIFICATIONS_LOG_FILENAME)


def notify_log(kind: str, title: str, body: str = "") -> None:
    """Escribe un paso del proceso de actualizacion al log persistente.

    ``kind`` debe ser uno de ``info`` / ``warn`` / ``error``. La funcion
    es tolerante a errores de E/S (un fallo aqui NO debe romper la cadena
    de actualizacion; si el disco esta lleno o el archivo esta bloqueado,
    el log se omite y el resultado se sigue notificando por el system tray).
    """
    icon = _KIND_ICON.get(kind, "[INFO ]")
    try:
        log_path = _log_path()
        from src.config import ensure_runtime_dirs
        ensure_runtime_dirs()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] {icon} {title}"
        if body:
            # Acotamos el cuerpo para no acabar con lineas de 8 KB si el
            # upstream devolvio un stacktrace gigantesco.
            body_clean = str(body).replace("\r", " ").replace("\n", " | ")
            if len(body_clean) > 480:
                body_clean = body_clean[:477] + "..."
            line += f": {body_clean}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("notify_log failed: %s", exc)


# ---------------------------------------------------------------------------
# Worker per-dep (generico)
# ---------------------------------------------------------------------------

class _DependencyUpdateWorker(QThread):
    """Background thread que re-instala una sola dependencia.

    Senales:
    - ``progress(int)``    : 0..100, emite mientras descarga.
    - ``status(str)``      : texto legible para status_label.
    - ``finished_ok``      : la re-instalacion termino sin lanzar excepcion.
    - ``failed(str)``      : termino con excepcion; payload es ``str(exc)``.
    - ``finished``         : siempre (signal Qt built-in al salir el thread).
    """

    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(
        self,
        dep_id: str,
        reinstall_fn: Callable[..., None],
        parent: Optional[QObject] = None,
        on_start: Optional[Callable[[], None]] = None,
        abort_event: Optional[threading.Event] = None,
    ):
        super().__init__(parent)
        self.dep_id = dep_id
        self.reinstall_fn = reinstall_fn
        self._on_start = on_start  # callable opcional, corre antes del install
        # ``request_stop`` lo setea; el instalador (pip / descarga de FFmpeg)
        # lo observa para abortar de verdad, no solo "soft cancel".
        self.abort_event = abort_event or threading.Event()

    def cancel(self) -> None:
        """Pide al instalador que pare en cuanto pueda."""
        self.abort_event.set()

    def run(self) -> None:
        try:
            if self._on_start is not None:
                try:
                    self._on_start()
                except Exception:
                    # Fallos en pre-hook NO cancelan la descarga; son solo
                    # housekeeping (ej. limpiar cache upstream de FFmpeg).
                    pass

            def _progress(v: int, _message: str = "") -> None:
                self.progress.emit(int(v))

            def _status(text: str) -> None:
                self.status.emit(str(text))

            self.status.emit(f"Updating {self.dep_id}...")
            self.reinstall_fn(
                progress_callback=_progress,
                parent=self.parent(),
                abort_event=self.abort_event,
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Chain controller (corre en hilo GUI, secuencial)
# ---------------------------------------------------------------------------

class _UpdateChainController(QObject):
    """Itera sobre una lista de deps y dispara un worker a la vez.

    Senales para el caller (todas emitidas en hilo GUI):
    - ``chain_started(total: int)``
    - ``step_started(index: int, total: int, dep_id: str, label: str)``
    - ``step_progress(int)``             : reenvia ``worker.progress``.
    - ``step_status(str)``               : reenvia ``worker.status``.
    - ``step_finished(index, dep_id, ok, message)``
    - ``chain_finished(dict)``           : ``{"ok": int, "failed": int, "total": int}``
    """

    chain_started = Signal(int)
    step_started = Signal(int, int, str, str)
    step_progress = Signal(int)
    step_status = Signal(str)
    step_finished = Signal(int, str, bool, str)
    chain_finished = Signal(dict)

    def __init__(self, deps: list[dict], parent: Optional[QObject] = None):
        super().__init__(parent)
        # ``deps`` es la lista ``pending_updatable_dependencies()``.
        # Cada elemento debe exponer: dep_id, display_name, reinstall_fn,
        # info (dict con version/display_name/etc) y entry (REGISTRY entry).
        self._deps = list(deps)
        self._index = -1
        self._worker: Optional[_DependencyUpdateWorker] = None
        self._results: list[dict] = []
        self._running = False
        # Cuando el usuario aborta la cadena (request_stop), ningun paso
        # nuevo debe arrancarse aunque queden deps por procesar.
        self._stopped = False
        # Estado por paso en vuelo para alimentar el mini-stats banner y el
        # historial persistente. La clave es ``dep_id`` (unica por paso).
        # Valores: {"pre_bytes": int|None, "pre_version": str|None,
        #          "pre_label": str, "started_monotonic": float}.
        self._step_state: dict = {}

    # -------------------- public API --------------------
    def start(self) -> None:
        """Arranca la cadena. Idempotente: si ya esta corriendo o si se
        marcó _stopped, no hace nada."""
        if self._running or self._stopped:
            return
        if not self._deps:
            self.chain_finished.emit({"ok": 0, "failed": 0, "total": 0})
            return
        self._running = True
        self._results = []
        self.chain_started.emit(len(self._deps))
        self._advance()

    def is_running(self) -> bool:
        return self._running and self._worker is not None and self._worker.isRunning()

    def current_dep_id(self) -> Optional[str]:
        if 0 <= self._index < len(self._deps):
            return self._deps[self._index].get("dep_id")
        return None

    def request_stop(self, wait_ms: int = 2000) -> None:
        """Best-effort: pide al worker que pare si sigue corriendo.

        Marca ``_stopped`` para que ningún paso nuevo arranque aunque
        el slot ``_advance`` se dispare por la señal ``finished`` del
        hilo que acaba de morir.

        Hoy ``_DependencyUpdateWorker.run()`` NO polea
        ``isInterruptionRequested()``, asi que este metodo es un soft
        cancel: el hint llega al thread pero ``force_redownload``
        (bloqueante de red) no lo consulta. El thread termina por
        cuenta propia al acabar la descarga; ``wait(wait_ms)`` solo se
        asegura de no soltar la referencia mientras Qt reapa la
        jerarquia.
        """
        self._stopped = True
        # Cade lateral: aunque el slot _advance se disparara con cola
        # llena, sin deps no hay paso posible que arrancar.
        self._deps = []
        if self._worker is None:
            return
        if not self._worker.isRunning():
            return
        try:
            self._worker.cancel()
        except Exception:
            pass
        try:
            self._worker.requestInterruption()
        except Exception:
            pass
        try:
            self._worker.wait(wait_ms)
        except Exception:
            pass

    # -------------------- internals --------------------
    def _advance(self) -> None:
        self._index += 1
        if self._stopped:
            # Cadena abortada por el usuario (request_stop): no arrancar
            # ningún paso nuevo aunque queden deps pendientes.
            self._running = False
            self.chain_finished.emit(
                {
                    "ok": sum(1 for r in self._results if r["ok"]),
                    "failed": sum(1 for r in self._results if not r["ok"]),
                    "total": len(self._results),
                    "results": list(self._results),
                    "stopped": True,
                }
            )
            return
        if self._index >= len(self._deps):
            self._running = False
            ok_count = sum(1 for r in self._results if r["ok"])
            fail_count = sum(1 for r in self._results if not r["ok"])
            self.chain_finished.emit(
                {
                    "ok": ok_count,
                    "failed": fail_count,
                    "total": len(self._results),
                    "results": list(self._results),
                }
            )
            return

        dep = self._deps[self._index]
        dep_id = dep.get("dep_id", "?")
        label = dep.get("display_name", dep_id)

        # Construimos un on_start especifico por dep. Para FFmpeg queremos
        # limpiar el cache upstream; para futuras deps (modelos, runtimes)
        # el caller puede extender esta logica desde fuera.
        on_start = self._make_on_start(dep_id)

        # Snapshot pre-install: leemos el estado de la dependencia ANTES
        # de tocarla. Esto alimenta el mini-stats del banner y el historial
        # persistente. Se ejecuta en el hilo GUI (status_fn es read-only).
        try:
            pre_info = dep["status_fn"]() or {}
        except Exception:
            pre_info = {}
        self._step_state[dep_id] = {
            "pre_version": pre_info.get("version"),
            "pre_bytes": _read_bytes(dep),
            "pre_label": label,
            "started_monotonic": time.monotonic(),
        }

        self.step_started.emit(self._index, len(self._deps), dep_id, label)
        notify_log("info", f"chain step start", f"{dep_id} ({label})")

        self._worker = _DependencyUpdateWorker(
            dep_id=dep_id,
            reinstall_fn=dep["reinstall_fn"],
            parent=self.parent(),
            on_start=on_start,
        )
        self._worker.progress.connect(self.step_progress.emit)
        self._worker.status.connect(self.step_status.emit)
        self._worker.finished_ok.connect(self._on_step_ok)
        self._worker.failed.connect(self._on_step_fail)
        # ``finished`` se emite SIEMPRE (ok o fail). Limpiamos la referencia
        # antes de avanzar para evitar condiciones de carrera.
        self._worker.finished.connect(self._on_thread_terminated)
        self._worker.start()

    def _make_on_start(self, dep_id: str) -> Optional[Callable[[], None]]:
        """Devuelve un callable para ejecutar antes de la descarga.

        Hoy solo FFmpeg tiene una accion especifica (vaciar el cache de
        upstream para que el siguiente ``is_update_available`` no diga
        'still up-to-date' tras la reinstalacion). Cuando se anadan
        runtimes/modelos, este switch se extiende.
        """
        if dep_id == "ffmpeg":
            def _clear_ffmpeg_cache() -> None:
                try:
                    from src.ffmpeg.upstream import clear_cache
                    clear_cache()
                except Exception:
                    pass

            return _clear_ffmpeg_cache
        return None

    def _finalize_step(self, ok: bool, message: str) -> None:
        """Metodo auxiliar: captura metricas post-install + persiste history.

        Lee el estado de la dependencia tras la reinstalacion (version +
        size_bytes), calcula la duracion total del step y registra la
        entrada en ``bin/.session/update_history.json``. Si la captura
        post-install falla (manifest corrupto, I/O), seguimos emitiendo
        el resultado con ``None`` en los campos faltantes para no romper
        la cadena.
        """
        dep_id = self.current_dep_id() or "?"
        dep_entry = next(
            (d for d in self._deps if d.get("dep_id") == dep_id), None
        )
        state = self._step_state.get(dep_id, {})
        started = state.get("started_monotonic")
        duration = 0.0
        if started is not None:
            duration = max(0.0, time.monotonic() - started)

        post_info = {}
        if dep_entry is not None:
            try:
                post_info = dep_entry["status_fn"]() or {}
            except Exception:
                post_info = {}

        old_version = state.get("pre_version")
        new_version = post_info.get("version") if ok else old_version
        old_bytes = state.get("pre_bytes")
        new_bytes = _read_bytes(dep_entry) if ok else old_bytes
        display_name = (
            (dep_entry or {}).get("display_name") or state.get("pre_label") or dep_id
        )

        # Persistir en history (best-effort: si falla la E/S seguimos).
        try:
            import src.update_history as update_history
            update_history.record_update(
                dep_id=dep_id,
                display_name=display_name,
                old_version=old_version,
                new_version=new_version,
                duration_seconds=duration,
                old_bytes=old_bytes,
                new_bytes=new_bytes,
                success=ok,
            )
        except Exception:
            pass

        result = {
            "dep_id": dep_id,
            "display_name": display_name,
            "ok": ok,
            "message": message,
            "old_version": old_version,
            "new_version": new_version,
            "old_bytes": old_bytes,
            "new_bytes": new_bytes,
            "duration_seconds": duration,
        }
        self._results.append(result)
        # Limpiamos el state del step consumido.
        self._step_state.pop(dep_id, None)
        self.step_finished.emit(self._index, dep_id, ok, message)

    # -------------------- slots --------------------
    def _on_step_ok(self) -> None:
        dep_id = self.current_dep_id() or "?"
        self._finalize_step(True, "")
        notify_log("info", f"chain step ok", dep_id)

    def _on_step_fail(self, msg: str) -> None:
        dep_id = self.current_dep_id() or "?"
        self._finalize_step(False, msg)
        notify_log("warn", f"chain step failed", f"{dep_id}: {msg}")

    def _on_thread_terminated(self) -> None:
        # Siempre tras finished_ok o failed. Limpiamos referencia y avanzamos
        # al siguiente paso. Usamos QTimer.singleShot(0) para evitar recursar
        # el slot mientras el thread aun esta liberando recursos.
        self._worker = None
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._advance)
