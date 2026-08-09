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

from datetime import datetime

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
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
    """Panel lateral colapsable: pestaña Tasks (renders del usuario) + Log."""

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
        root.addWidget(self.tabs, 1)

        self._order: list[str] = []   # task ids (newest first)
        self._cards: dict[str, TaskCard] = {}

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