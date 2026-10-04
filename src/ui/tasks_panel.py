"""Panel lateral de tareas + log: muestra el progreso por fase de cada job.

Derecha de la interfaz, colapsable con un toggle. Dos pestañas:

- **Tasks**: una ``TaskCard`` por render pedido por el usuario (bien
  corriente, bien histórico persistido en ``src.task_history``) con una
  fila por fase (``PhaseRow``) y su mini-barra: engine de IA, audio, mux.
- **Log**: historial plano de mensajes de estado ("lo que está sucediendo
  y ha sucedido"), timestampados y con autoscroll.

Los eventos vienen del worker vía un dict plano:
``{"ev": start|progress|end|failed, "id": str, "label": str, "pct": int,
"msg": str}``. El panel no importa nada de pipeline para evitar
acoplamientos de ida.
"""

from __future__ import annotations

import os
from datetime import datetime

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Estado de fase -> (color del pill, texto corto)
_PHASE_STATUS_COLORS = {
    "running": ("#00d2d3", "#0f0f1a", "RUNNING"),
    "done":    ("#27ae60", "#ffffff", "DONE"),
    "failed":  ("#e74c3c", "#ffffff", "FAILED"),
    "pending": ("#2c3e50", "#a0a0b0", "PENDING"),
}


class PhaseBar(QWidget):
    """Mini-barra de progreso por fase (custom painted, cross-OS)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pct = 0
        self.color = QColor("#00d2c3")
        self.setFixedHeight(8)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set(self, pct: int) -> None:
        self.pct = max(0, min(100, int(pct or 0)))
        self.update()

    def paintEvent(self, event):  # noqa: D401 - Qt callback name
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            track = self.rect().adjusted(0, 1, 0, -1)
            painter.setBrush(QColor("#1f2c4c"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(track, 3, 3)
            if self.pct > 0:
                fill_w = max(1, int(track.width() * self.pct / 100.0))
                fill = track.adjusted(0, 0, fill_w - track.width(), 0)
                painter.setBrush(self.color)
                painter.drawRoundedRect(fill, 3, 3)
        finally:
            painter.end()


class PhaseRow(QWidget):
    """Label + mini-barra + pill de estado para una fase."""

    def __init__(self, phase_id, label, parent=None):
        super().__init__(parent)
        self.phase_id = phase_id
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(6)
        self.label = QLabel(label)
        self.label.setWordWrap(False)
        self.label.setStyleSheet("color: #c8c8d4; font-size: 11px;")
        self.label.setMinimumWidth(96)
        lay.addWidget(self.label, 1)
        self.bar = PhaseBar()
        lay.addWidget(self.bar, 3)
        self.status = QLabel("PENDING")
        self._paint_status("pending")
        lay.addWidget(self.status)

    def _paint_status(self, status: str):
        bg, fg, text = _PHASE_STATUS_COLORS.get(
            status, ("#2c3e50", "#a0a0b0", str(status).upper())
        )
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color: {fg}; background: {bg}; padding: 1px 6px; "
            "border-radius: 4px; font-size: 9px; font-weight: bold;"
        )

    def update_phase(self, ev: str, pct: int):
        self.bar.set(pct)
        if ev == "start":
            self._paint_status("pending")
            self.bar.set(min(pct, 100))
        elif ev == "progress":
            self._paint_status("running")
            self.bar.set(pct)
        elif ev == "end":
            self._paint_status("done")
            self.bar.set(100)
        elif ev == "failed":
            self._paint_status("failed")
            self.bar.color = QColor("#e74c3c")
            self.bar.set(min(pct, 100))
            self.bar.update()


class TaskCard(QFrame):
    """Una tarea (render en curso o del histórico) con sus fases."""

    def __init__(self, task_id, name, parent=None):
        super().__init__(parent)
        self.task_id = task_id
        self.setObjectName("TaskCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "#TaskCard { background-color: #141a30; border: 1px solid #243049; "
            "border-radius: 8px; }"
        )
        self._phases: dict[str, PhaseRow] = {}
        self._phase_order: list[str] = []
        self._status = "pending"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self.title = QLabel(name or task_id)
        self.title.setStyleSheet("color: #e8e8f0; font-weight: bold; font-size: 12px;")
        self.title.setWordWrap(True)
        outer.addWidget(self.title)

        self._status_label = QLabel("PENDING")
        self._status_label.setStyleSheet(
            "color: #a0a0b0; font-size: 10px;"
        )
        outer.addWidget(self._status_label)

        self._row_container = QWidget()
        self._row_layout = QVBoxLayout(self._row_container)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(4)
        outer.addWidget(self._row_container)
        outer.addStretch(1)

    @property
    def name(self):
        return getattr(self, "_name", self.task_id)

    @name.setter
    def name(self, value):
        self._name = value

    def add_phase(self, phase_id: str, label: str) -> PhaseRow:
        if phase_id in self._phases:
            return self._phases[phase_id]
        row = PhaseRow(phase_id, label)
        row.setVisible(True)
        self._row_layout.addWidget(row)
        self._phases[phase_id] = row
        self._phase_order.append(phase_id)
        return row

    def set_phase(self, phase_id: str, label: str = "") -> PhaseRow:
        if phase_id not in self._phases:
            return self.add_phase(phase_id, label)
        if label:
            self._phases[phase_id].label.setText(label)
        return self._phases[phase_id]

    def update_phase_evt(self, ev: dict):
        phase_id = ev.get("id", "")
        if not phase_id:
            return
        label = ev.get("label", "") or phase_id
        row = self.set_phase(phase_id, label)
        row.update_phase(ev.get("ev", "progress"), int(ev.get("pct", 0) or 0))
        if ev.get("ev") == "failed":
            self._set_task_status("failed")
        elif ev.get("ev") == "end":
            self._maybe_auto_done()

    def _set_task_status(self, status: str):
        self._status = status
        texts = {
            "running": ("RUNNING", "#00d2c3"),
            "done": ("DONE ✓", "#27ae60"),
            "failed": ("FAILED", "#e74c3c"),
            "aborted": ("ABORTED", "#e67e22"),
        }
        text, color = texts.get(status, (status.upper(), "#a0a0b0"))
        self._status_label.setText(text.upper())
        self._status_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")

    def _maybe_auto_done(self):
        if not self._phases:
            return
        any_ended = any(
            p.status.text().strip() in ("DONE", "FAILED", "ABORTED")
            for p in self._phases.values()
        )
        phases_done = all(
            p.status.text().strip() in ("DONE", "FAILED", "ABORTED")
            for p in self._phases.values()
        )
        if phases_done and any_ended:
            self._set_task_status("completed")

    def set_running(self):
        self._set_task_status("running")

    def finish(self, status: str):
        safe = "completed" if status == "success" else ("aborted" if status == "aborted" else "failed")
        self._set_task_status(safe)
        for p in self._phases.values():
            if p.status.text().strip() == "RUNNING":
                p._paint_status("done")
                p.bar.set(100)


class TasksPanel(QWidget):
    """Panel lateral colapsable: pestaña Tasks (renders del usuario) + Log.

    ``cancel_requested`` lo emite el ``✕`` de cada fila de la cola con el
    ``queue_id`` afectado. El panel NO decide qué hacer con ese item: borrar
    una fila en espera es suyo, pero parar un render en marcha es cosa de
    quien tiene al worker (MainWindow), así que el botón solo avisa.
    """

    cancel_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled = True

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.title = QLabel("Tasks")
        self.title.setStyleSheet("color: #e8e8f0; font-weight: bold; font-size: 14px;")
        header.addWidget(self.title)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #7f8fa6; font-size: 11px;")
        header.addWidget(self._count_label, 1, Qt.AlignmentFlag.AlignRight)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: none; background: transparent; }"
            "QTabBar::tab { background: #1f2c4c; color: #a0a0b0; padding: 4px 12px;"
            " border-radius: 5px 5px 0 0; margin-right: 2px; }"
            "QTabBar::tab:selected { background: #243049; color: #e8e8f0; }"
        )

        # --- Pestaña Tasks: tarjetas de render del usuario ---
        self._tasks_host = QWidget()
        tasks_lay = QVBoxLayout(self._tasks_host)
        tasks_lay.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._list = QWidget()
        self._list.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch(1)
        self.scroll.setWidget(self._list)
        tasks_lay.addWidget(self.scroll, 1)

        self.btn_clear = QPushButton("Clear history")
        self.btn_clear.setStyleSheet(
            "background-color: #1f2c4c; color: #a0a0b0; border: 1px solid #2c3e50; "
            "border-radius: 6px; padding: 4px 10px; font-size: 11px;"
        )
        tasks_lay.addWidget(self.btn_clear)
        self.tabs.addTab(self._tasks_host, "Tasks")

        # --- Pestaña Log: todo lo que sucede y ha sucedido ---
        self._log_host = QWidget()
        log_lay = QVBoxLayout(self._log_host)
        log_lay.setContentsMargins(0, 0, 0, 0)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background: #0f1424; color: #c8c8d4; border: 1px solid #243049; "
            "border-radius: 6px; font-family: Consolas, monospace; font-size: 11px; }"
        )
        log_lay.addWidget(self.log_view, 1)

        self.btn_clear_log = QPushButton("Clear log")
        self.btn_clear_log.setStyleSheet(
            "background-color: #1f2c4c; color: #a0a0b0; border: 1px solid #2c3e50; "
            "border-radius: 6px; padding: 4px 10px; font-size: 11px;"
        )
        self.btn_clear_log.clicked.connect(self._on_clear_log)
        log_lay.addWidget(self.btn_clear_log)

        self.tabs.addTab(self._log_host, "Log")

        # --- Pestaña Queue: renders programados por el usuario ---
        self._queue_host = QWidget()
        queue_lay = QVBoxLayout(self._queue_host)
        queue_lay.setContentsMargins(0, 0, 0, 0)

        self.queue_scroll = QScrollArea()
        self.queue_scroll.setWidgetResizable(True)
        self.queue_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._queue_list = QWidget()
        self._queue_list.setStyleSheet("background: transparent;")
        self._queue_layout = QVBoxLayout(self._queue_list)
        self._queue_layout.setContentsMargins(0, 0, 0, 0)
        self._queue_layout.setSpacing(6)
        self._queue_layout.addStretch(1)
        self.queue_scroll.setWidget(self._queue_list)
        queue_lay.addWidget(self.queue_scroll, 1)

        self._queue_empty = QLabel(
            "No queued jobs.\n\nSelect a video, configure the AI chain\n"
            "and compression settings, then press\n"
            "\"Add to queue\" below."
        )
        self._queue_empty.setStyleSheet(
            "color: #7f8fa6; font-size: 11px; padding: 8px;"
        )
        self._queue_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._queue_empty.setWordWrap(True)
        self._queue_empty.hide()
        queue_lay.addWidget(self._queue_empty)

        # Progreso global de la cola: cuántos jobs van completados del total
        # (no confundir con la barra del render actual de la ventana principal).
        self._queue_progress_label = QLabel("No queued jobs")
        self._queue_progress_label.setStyleSheet(
            "color: #7f8fa6; font-size: 11px;"
        )
        queue_lay.addWidget(self._queue_progress_label)
        self._queue_progress_bar = QProgressBar()
        self._queue_progress_bar.setRange(0, 1)
        self._queue_progress_bar.setValue(0)
        self._queue_progress_bar.setTextVisible(True)
        self._queue_progress_bar.setFixedHeight(10)
        self._queue_progress_bar.setStyleSheet(
            "QProgressBar { background: #1f2c4c; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: #00d2c3; border-radius: 3px; }"
        )
        self._queue_progress_bar.hide()
        queue_lay.addWidget(self._queue_progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_start_queue = QPushButton("Start queue")
        self.btn_start_queue.setStyleSheet(
            "background-color: #27ae60; color: #ffffff; border: none; "
            "border-radius: 6px; padding: 5px 10px; font-size: 11px; font-weight: bold;"
        )
        self.btn_start_queue.setEnabled(False)
        btn_row.addWidget(self.btn_start_queue, 1)
        self.btn_clear_queue = QPushButton("Clear queue")
        self.btn_clear_queue.setStyleSheet(
            "background-color: #1f2c4c; color: #a0a0b0; border: 1px solid #2c3e50; "
            "border-radius: 6px; padding: 4px 10px; font-size: 11px;"
        )
        btn_row.addWidget(self.btn_clear_queue)
        queue_lay.addLayout(btn_row)

        self.tabs.addTab(self._queue_host, "Queue")
        root.addWidget(self.tabs, 1)

        self._order: list[str] = []   # task ids (newest first)
        self._cards: dict[str, TaskCard] = {}
        self._queue_rows: dict[str, QueueRow] = {}
        self._queue_running = False

    # --- API del Log -------------------------------------------------

    def append_log(self, message: str):
        """Añade una línea timestampada al log (autoscroll al final)."""
        if not message:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {message}")
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_view.setTextCursor(cursor)

    def _on_clear_log(self):
        self.log_view.clear()

    def _sync_count(self):
        self._count_label.setText(str(len(self._cards)))

    def rebuild(self, tasks: list = None):
        """``(Re)construye la lista desde ``list_tasks()`` reemplazando todo."""
        self.clear_all()
        for task in (tasks or []):
            task_id = task.get("task_id", "")
            if not task_id:
                continue
            card = self.get_or_create(task_id, task.get("name", task_id))
            for ph in (task.get("phases") or []):
                card.set_phase(ph.get("phase_id", ""), ph.get("label", ""))
                card.update_phase_evt({"id": ph.get("phase_id"), "label": ph.get("label"),
                                       "pct": ph.get("pct", 0),
                                       "ev": "end" if ph.get("status") == "done" else "pending"})
            card.finish(task.get("status", "running"))

    def get_or_create(self, task_id: str, name: str = "") -> TaskCard:
        if task_id in self._cards:
            card = self._cards[task_id]
            if name and card.name != name:
                card.name = name
                card.title.setText(name)
            return card
        card = TaskCard(task_id, name or task_id)
        self._cards[task_id] = card
        # Newest first: insert at the top of the list (index 0).
        self._list_layout.insertWidget(0, card)
        self._order.insert(0, task_id)
        self._sync_count()
        return card

    def remove_task(self, task_id: str):
        card = self._cards.pop(task_id, None)
        if card is not None:
            self._list_layout.removeWidget(card)
            card.deleteLater()
            if task_id in self._order:
                self._order.remove(task_id)
        self._sync_count()

    def clear_all(self):
        """Elimina TODAS las tarjetas del UI (historial visible)."""
        for task_id in list(self._order):
            card = self._cards.pop(task_id, None)
            if card is not None:
                self._list_layout.removeWidget(card)
                card.deleteLater()
        self._order.clear()
        self._sync_count()

    def update_phase_event(self, task_id: str, ev: dict):
        card = self.get_or_create(task_id)
        card.set_running()
        card.update_phase_evt(ev)

    def mark_finished(self, task_id: str, status: str):
        card = self._cards.get(task_id)
        if card is None:
            return
        card.finish(status)

    # --- API de la Queue ----------------------------------------------

    def rebuild_queue(self, items: list = None):
        """Reconstruye la pestaña Queue desde ``src.task_queue.list_items()``."""
        for queue_id in list(self._queue_rows):
            self._remove_queue_row(queue_id)
        has_any = False
        count = 0
        for item in (items or []):
            queue_id = item.get("queue_id", "")
            if not queue_id:
                continue
            self._queue_row(queue_id, item.get("name", queue_id), item=item)
            self.set_queue_status(queue_id, item.get("status", "queued"))
            has_any = True
            count += 1
        self._queue_empty.setVisible(not has_any)
        # El botón Start queue debe habilitarse con jobs en cola (y la UI de
        # progreso reflejar el conteo). Antes solo se habilitaba al TERMINAR
        # una cola, así que la primera vez no se podía arrancar nada.
        self._queue_progress_label.setText(
            f"{count} job(s) queued" if has_any else "No queued jobs"
        )
        self.set_queue_running(self._queue_running, total=count)

    def _queue_row(self, queue_id: str, name: str, item: dict = None) -> QueueRow:
        row = self._queue_rows.get(queue_id)
        if row is not None:
            if name:
                row.name_label.setText(name)
            if item is not None:
                row.set_details(item)
            return row
        row = QueueRow(queue_id, name)
        row.btn_remove.clicked.connect(
            lambda _checked=False, qid=queue_id: self.cancel_requested.emit(qid)
        )
        if item is not None:
            row.set_details(item)
        # Auto-expand del primer item en marcha para que se vea sin click.
        if item is not None and item.get("status") == "running":
            row.set_expanded(True)
        self._queue_rows[queue_id] = row
        self._queue_layout.insertWidget(self._queue_layout.count() - 1, row)
        self._queue_empty.hide()
        return row

    def set_queue_status(self, queue_id: str, status: str):
        row = self._queue_rows.get(queue_id)
        if row is not None:
            row.set_status(status)
            # Al pasar a "running", auto-expand para mostrar lo que se está
            # comprimiendo ahora mismo.
            if status == "running":
                row.set_expanded(True)

    def _remove_queue_row(self, queue_id: str):
        row = self._queue_rows.pop(queue_id, None)
        if row is not None:
            self._queue_layout.removeWidget(row)
            row.deleteLater()

    def remove_queue_item(self, queue_id: str) -> str:
        """Aplica el ``✕`` de UNA fila: cancela ese item y quita la fila.

        Devuelve el resultado de ``task_queue.cancel``:

        - ``"cancelled"``: estaba esperando y sale de la cola.
        - ``"aborted"``: tenía un render en marcha. La fila NO se borra:
          se repinta como ABORTED y se queda visible, porque el render lo
          para quien llama (MainWindow) y su cierre necesita un item al que
          escribirle el final. Borrarla aquí dejaba un ffmpeg escribiendo
          un archivo que la app ya no contaba.
        - ``"finished"``: ya estaba terminado; la fila se descarta de la
          lista (no hay render que parar, sólo ocupaba sitio).
        - ``"not_found"``: fila fantasma; se descarta sin tocar el archivo.
        - ``"error"``: no se pudo guardar el cambio; la fila se queda como
          estaba, que es preferible a mostrar un estado que no se guardó.
        """
        try:
            import src.task_queue as task_queue
            result = task_queue.cancel(queue_id)
        except Exception:
            result = "error"
        if result == "aborted":
            self.set_queue_status(queue_id, "aborted")
            return result
        if result == "error":
            return result
        if result in ("finished", "not_found"):
            # ``cancel`` no toca los items terminales ni los que no existen;
            # aquí sólo se olvidan de la lista.
            try:
                import src.task_queue as task_queue
                task_queue.remove(queue_id)
            except Exception:
                pass
        self._forget_queue_row(queue_id)
        return result

    def _forget_queue_row(self, queue_id: str):
        """Quita la fila de la lista y recontesta el estado de la cola."""
        self._remove_queue_row(queue_id)
        if not self._queue_rows:
            self._queue_empty.show()
            self._queue_progress_label.setText("No queued jobs")
        else:
            pending = sum(
                1 for row in self._queue_rows.values()
                if row.pill.text() == _QUEUE_STATUS_META["queued"][0]
            )
            self._queue_progress_label.setText(f"{pending} job(s) queued")
        self.set_queue_running(self._queue_running)

    def clear_queue_rows(self):
        """Limpia SOLO la pestaña UI; no toca el archivo persistente."""
        for queue_id in list(self._queue_rows):
            self._remove_queue_row(queue_id)
        self._queue_empty.show()
        self._queue_progress_label.setText("No queued jobs")
        self.set_queue_running(self._queue_running)

    def set_queue_running(self, running: bool, total: int = 0):
        """Habilita/deshabilita Start queue según si la cola está en marcha."""
        self._queue_running = running
        self.btn_start_queue.setEnabled(
            not running and bool(self._queue_rows)
        )
        if running:
            self._queue_progress_bar.setRange(0, max(int(total), 1))
            self._queue_progress_bar.setValue(0)
            self._queue_progress_bar.show()
            self._queue_progress_label.setText(f"Queue running: 0/{total} done")
        else:
            self._queue_progress_bar.hide()

    def set_queue_progress(self, done: int, total: int):
        """Actualiza la barra global de la cola: jobs completados / total."""
        done = max(0, int(done))
        total = max(0, int(total))
        self._queue_progress_bar.setRange(0, max(total, 1))
        self._queue_progress_bar.setValue(done)
        if total and done >= total:
            self._queue_progress_label.setText("Queue finished")
        else:
            self._queue_progress_label.setText(f"Queue: {done}/{total} done")
        self._queue_progress_bar.show()


# ---------------------------------------------------------------------------
# Queue: renders programados por el usuario (cola de trabajo)
# ---------------------------------------------------------------------------

_QUEUE_STATUS_META = {
    "queued":      ("QUEUED", "#7f8fa6"),
    "validating":  ("CHECKING", "#f1c40f"),
    "preparing":   ("PREPARING", "#f1c40f"),
    "running":     ("RUNNING", "#00d2c3"),
    "muxing":      ("MUXING", "#00d2c3"),
    "publishing":  ("PUBLISHING", "#00d2c3"),
    "done":        ("DONE ✓", "#27ae60"),
    "failed":      ("FAILED", "#e74c3c"),
    "aborted":     ("ABORTED", "#e67e22"),
}
# Estados sin render detrás: su ``✕`` ya no cancela nada, sólo saca la
# fila terminada de la lista.
_QUEUE_TERMINAL_STATUS = {"done", "failed", "aborted"}


class QueueRow(QFrame):
    """Fila de la pestaña Queue: nombre + estado + quitar item.

    Colapsada muestra solo ``name|pill|✕``. Expandida (botón toggle ▸) muestra
    debajo un ``QLabel`` con rich-text y todo el snapshot del item: rutas,
    pipeline, engines de IA (modelo + escala + modo), ajustes de compresión,
    audio y timestamps.

    El ``✕`` es la acción individual sobre ese item: cancela el render en
    marcha, saca de la cola el que está en espera y descarta de la lista
    el que ya terminó (done/failed/aborted), que ya no tiene nada que
    cancelar pero sí ocupa sitio.
    """

    def __init__(self, queue_id, name, parent=None):
        super().__init__(parent)
        self.queue_id = queue_id
        self.setObjectName("QueueRow")
        self.setStyleSheet(
            "#QueueRow { background-color: #141a30; border: 1px solid #243049; "
            "border-radius: 6px; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        # Cabecera: chevron + nombre + estado + ✕
        header = QHBoxLayout()
        header.setSpacing(6)

        self.toggle_btn = QPushButton("\u25b8")  # ▸
        self.toggle_btn.setFixedSize(20, 20)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setToolTip("Show full job details")
        self.toggle_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #a0a0b0; "
            "border: none; font-size: 12px; font-weight: bold; padding: 0px; }"
            "QPushButton:hover { color: #00d2d3; }"
            "QPushButton:checked { color: #00d2d3; }"
        )
        self.toggle_btn.toggled.connect(self._set_expanded)
        header.addWidget(self.toggle_btn)

        self.name_label = QLabel(name or queue_id)
        self.name_label.setStyleSheet("color: #e8e8f0; font-size: 11px;")
        self.name_label.setWordWrap(True)
        header.addWidget(self.name_label, 1)

        self.pill = QLabel("QUEUED")
        self._paint("queued")
        header.addWidget(self.pill)

        # Cancelar SOLO este item. En espera quita la fila; en marcha para
        # el render (lo hace MainWindow, que es quien tiene al worker).
        self.btn_remove = QPushButton("\u2715")  # ✕
        self.btn_remove.setFixedWidth(26)
        self.btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_remove.setStyleSheet(
            "QPushButton { background: transparent; color: #8a93ab; "
            "border: none; font-size: 14px; }"
            "QPushButton:hover { color: #e74c3c; background: #2a1a22; "
            "border-radius: 4px; }"
        )
        self.btn_remove.setToolTip("Cancel this job")
        header.addSpacing(4)
        header.addWidget(self.btn_remove)

        outer.addLayout(header)

        # Detalle (oculto): rich-text con TODO el snapshot del item.
        self._details = QLabel("")
        self._details.setWordWrap(True)
        self._details.setTextFormat(Qt.TextFormat.RichText)
        self._details.setStyleSheet(
            "QLabel { background-color: #0f1424; color: #c8c8d4; "
            "padding: 8px; border: 1px solid #243049; border-radius: 4px; "
            "font-family: Consolas, monospace; font-size: 11px; }"
        )
        self._details.hide()
        outer.addWidget(self._details)

    def _set_expanded(self, expanded: bool):
        self.toggle_btn.setText("\u25be" if expanded else "\u25b8")  # ▾ / ▸
        self._details.setVisible(expanded)

    def set_expanded(self, expanded: bool):
        """API pública: expande/colapsa desde el panel (p.ej. auto-expand running)."""
        # Si el estado es el mismo, setChecked no emite toggled → no recursión.
        if self.toggle_btn.isChecked() != expanded:
            self.toggle_btn.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self.toggle_btn.isChecked()

    def set_details(self, item: dict):
        """Pinta el snapshot HTML del item. Llamar cuando cambia el item."""
        self._details.setText(_format_queue_item_details(item))

    def _paint(self, status: str):
        text, color = _QUEUE_STATUS_META.get(
            status, ("QUEUED", "#9f8fa6")
        )
        self.pill.setText(text)
        self.pill.setStyleSheet(
            f"color: {color}; background: #1f2c4c; padding: 1px 6px; "
            "border-radius: 4px; font-size: 9px; font-weight: bold;"
        )

    def set_status(self, status: str):
        self._paint(status)
        # El botón acompaña al estado: siempre disponible (un item ya
        # terminado también se puede quitar de la lista), pero con el
        # texto que dice qué va a hacer DE VERDAD.
        if status in _QUEUE_TERMINAL_STATUS:
            self.btn_remove.setToolTip("Remove this finished job from the list")
        elif str(status) == "queued":
            self.btn_remove.setToolTip("Cancel this job and remove it from the queue")
        else:
            self.btn_remove.setToolTip("Stop this render now")


def _format_queue_item_details(item: dict) -> str:
    """Render del snapshot completo de un item de la cola como HTML.

    Secciones, en este orden:
        - Encabezado con el nombre.
        - Input / Output (rutas largas, envueltas).
        - Pipeline (orden de nodos).
        - AI engines (categoría → engine_id, model, escala, native).
        - Compression (codec, bitrate, resolution, format, preset, etc.).
        - Audio (audio_boost target + audio_codec).
        - Timestamps + error.
    """

    def _esc(s) -> str:
        s = "" if s is None else str(s)
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = []

    name = _esc(item.get("name") or item.get("queue_id") or "")
    parts.append(f"<div style='margin-bottom:8px;'><b style='color:#e8e8f0;'>{name}</b></div>")

    input_path = item.get("input_path") or "—"
    output_path = item.get("output_path") or "—"
    parts.append(
        "<b style='color:#00d2d3;'>Input:</b> "
        f"<span style='color:#c8c8d4;'>{_esc(input_path)}</span>"
    )
    parts.append(
        "<div><b style='color:#00d2d3;'>Output:</b> "
        f"<span style='color:#c8c8d4;'>{_esc(output_path)}</span></div>"
    )

    pipeline = item.get("pipeline_order") or []
    if pipeline:
        chain = " \u2192 ".join(_esc(s) for s in pipeline)
        parts.append(
            f"<div><b style='color:#00d2d3;'>Pipeline:</b> "
            f"<span style='color:#c8c8d4;'>{chain}</span></div>"
        )

    ai_engines = item.get("ai_engines") or {}
    if ai_engines:
        parts.append("<div><b style='color:#00d2d3;'>AI engines:</b></div>")
        for cat, info in ai_engines.items():
            cat_lbl = _esc(cat)
            if isinstance(info, dict):
                eid = info.get("engine_id", "") or "auto"
                scale = info.get("scale")
                model = info.get("model_path") or ""
                native = bool(info.get("native"))
                scale_lbl = ""
                if scale and not native:
                    try:
                        scale_lbl = f" <span style='color:#27ae60;'>(x{int(scale)})</span>"
                    except (TypeError, ValueError):
                        scale_lbl = ""
                elif native:
                    scale_lbl = " <span style='color:#27ae60;'>(native)</span>"
                model_lbl = ""
                if model:
                    model_lbl = (
                        "<div style='margin-left:18px;color:#7f8fa6;'>"
                        f"model: {_esc(os.path.basename(model))}</div>"
                    )
                parts.append(
                    "<div style='margin-left:12px;'>"
                    f"\u25b8 <b>{cat_lbl}</b>: "
                    f"<code style='color:#c8c8d4;'>{_esc(eid)}</code>{scale_lbl}"
                    f"{model_lbl}</div>"
                )
            else:
                parts.append(
                    "<div style='margin-left:12px;'>"
                    f"\u25b8 <b>{cat_lbl}</b>: "
                    f"<code style='color:#c8c8d4;'>{_esc(info)}</code></div>"
                )

    settings = item.get("settings") or {}
    if settings:
        parts.append(
            "<div style='margin-top:6px;'>"
            "<b style='color:#00d2d3;'>Compression:</b></div>"
        )
        comp_keys = [
            ("codec", "Codec"),
            ("encoder", "Encoder"),
            ("preset", "Preset"),
            ("bitrate", "Bitrate"),
            ("crf", "CRF"),
            ("resolution", "Resolution"),
            ("width", "Width"),
            ("height", "Height"),
            ("fps", "FPS"),
            ("format", "Format"),
            ("container", "Container"),
            ("output_name", "Output name"),
            ("pixel_format", "Pixel format"),
            ("extra_ffmpeg", "Extra ffmpeg args"),
        ]
        for k, label in comp_keys:
            v = settings.get(k)
            if v not in (None, "", []):
                parts.append(
                    "<div style='margin-left:12px;color:#c8c8d4;'>"
                    f"\u25b8 <b>{_esc(label)}</b>: "
                    f"<code>{_esc(v)}</code></div>"
                )

    if settings:
        # The live SettingsPanel stores the codec as ``audio``; older queue
        # snapshots used ``audio_codec``. Render both so historical jobs keep
        # their complete explanation.
        audio_codec = settings.get("audio", settings.get("audio_codec"))
        audio_any = (
            "audio" in settings
            or settings.get("audio_codec")
            or settings.get("audio_bitrate")
            or settings.get("audio_stream_index") is not None
            or settings.get("normalize_audio")
            or settings.get("audio_boost")
        )
        if audio_any:
            parts.append(
                "<div style='margin-top:6px;'>"
                "<b style='color:#00d2d3;'>Audio:</b></div>"
            )
            if audio_codec == "noaudio":
                parts.append(
                    "<div style='margin-left:12px;color:#c8c8d4;'>"
                    "\u25b8 Mode: <code>disabled</code></div>"
                )
            elif audio_codec:
                parts.append(
                    "<div style='margin-left:12px;color:#c8c8d4;'>"
                    f"\u25b8 Codec: <code>{_esc(audio_codec)}</code></div>"
                )
            audio_boost = settings.get("audio_boost")
            if isinstance(audio_boost, dict):
                target = (
                    audio_boost.get("target_lufs")
                    or audio_boost.get("target")
                    or "—"
                )
                parts.append(
                    "<div style='margin-left:12px;color:#c8c8d4;'>"
                    f"\u25b8 Loudness boost: <code>{_esc(target)} LUFS</code></div>"
                )
            elif audio_boost:
                parts.append(
                    "<div style='margin-left:12px;color:#c8c8d4;'>"
                    "\u25b8 Loudness boost: <code>enabled</code></div>"
                )
            if settings.get("audio_bitrate"):
                parts.append(
                    "<div style='margin-left:12px;color:#c8c8d4;'>"
                    f"\u25b8 Bitrate: <code>{_esc(settings['audio_bitrate'])}</code></div>"
                )
            stream_index = settings.get("audio_stream_index")
            if stream_index is not None and audio_codec != "noaudio":
                try:
                    is_all = int(stream_index) == -1
                except (TypeError, ValueError):
                    is_all = False
                if is_all:
                    track_label = "All tracks"
                else:
                    try:
                        track_label = settings.get("audio_stream_label") or f"Track {int(stream_index) + 1}"
                    except (TypeError, ValueError):
                        track_label = settings.get("audio_stream_label") or str(stream_index)
                parts.append(
                    "<div style='margin-left:12px;color:#c8c8d4;'>"
                    f"\u25b8 Selected track: <code>{_esc(track_label)}</code></div>"
                )

    if item.get("created_at"):
        parts.append(
            "<div style='margin-top:6px;'>"
            f"<b style='color:#7f8fa6;'>Queued at:</b> "
            f"<span style='color:#c8c8d4;'>{_esc(item['created_at'])}</span></div>"
        )
    if item.get("finished_at"):
        parts.append(
            "<div><b style='color:#7f8fa6;'>Finished at:</b> "
            f"<span style='color:#c8c8d4;'>{_esc(item['finished_at'])}</span></div>"
        )
    if item.get("error"):
        parts.append(
            "<div><b style='color:#7f8fa6;'>Error:</b> "
            "<span style='color:#e74c3c;'>"
            f"{_esc(item['error'])}</span></div>"
        )

    return "".join(parts)