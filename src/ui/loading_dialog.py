from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

# Each phase is defined by its display label and the progress range at which
# it becomes active.  The dialog maps the current progress percentage to one
# of these phases and renders a visual stepper.
_PHASES = [
    ("Detecting hardware…", 10),
    ("Downloading FFmpeg…", 30),
    ("Installing ONNX Runtime…", 55),
    ("Installing PyTorch…", 72),
    ("Testing codecs…", 90),
]

_DONE_LABEL = "Ready"

# Styles ------------------------------------------------------------------
_DOT_ACTIVE = (
    "QLabel {{"
    "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, "
    "    stop:0 #00d2d3, stop:1 #5f27cd);"
    "  border-radius: 6px;"
    "  min-width: 12px; max-width: 12px;"
    "  min-height: 12px; max-height: 12px;"
    "}}"
)
_DOT_DONE = (
    "QLabel {{"
    "  background: #00d2d3;"
    "  border-radius: 6px;"
    "  min-width: 12px; max-width: 12px;"
    "  min-height: 12px; max-height: 12px;"
    "}}"
)
_DOT_PENDING = (
    "QLabel {{"
    "  background: #3a4a6b;"
    "  border-radius: 6px;"
    "  min-width: 12px; max-width: 12px;"
    "  min-height: 12px; max-height: 12px;"
    "}}"
)
_TEXT_ACTIVE = "color: #ffffff; font-weight: bold;"
_TEXT_DONE = "color: #00d2d3; font-weight: bold;"
_TEXT_PENDING = "color: #6b7b9e; font-weight: normal;"


class _PhaseRow(QWidget):
    """A single row in the phase stepper: dot + label + optional checkmark."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        self._dot.setStyleSheet(_DOT_PENDING)
        layout.addWidget(self._dot)

        self._check = QLabel("✓")
        self._check.setStyleSheet("color: #00d2d3; font-weight: bold; font-size: 12px;")
        self._check.hide()
        layout.addWidget(self._check)

        self._label = QLabel(text)
        self._label.setStyleSheet(_TEXT_PENDING)
        layout.addWidget(self._label)

        layout.addStretch()

    # -- state mutators ----------------------------------------------------

    def mark_done(self):
        self._dot.setStyleSheet(_DOT_DONE)
        self._check.show()
        self._label.setStyleSheet(_TEXT_DONE)

    def mark_active(self):
        self._dot.setStyleSheet(_DOT_ACTIVE)
        self._check.hide()
        self._label.setStyleSheet(_TEXT_ACTIVE)

    def mark_pending(self):
        self._dot.setStyleSheet(_DOT_PENDING)
        self._check.hide()
        self._label.setStyleSheet(_TEXT_PENDING)


class LoadingDialog(QDialog):
    def __init__(self, title="Cargando...", message="Por favor espera", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )

        layout = QVBoxLayout(self)

        self.label = QLabel(message)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            """
            QProgressBar {
                background-color: #16213e;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                min-height: 14px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2d3, stop:1 #5f27cd);
                border-radius: 6px;
            }
            """
        )
        layout.addWidget(self.progress)

        # Phase stepper — one row per phase, stacked vertically
        self._rows: list[_PhaseRow] = []
        for phase_label, _ in _PHASES:
            row = _PhaseRow(phase_label)
            self._rows.append(row)
            layout.addWidget(row)

        self._active_idx = -1  # nothing active yet

    # -- public API --------------------------------------------------------

    def set_progress(self, value: int):
        self.progress.setValue(value)
        idx = self._phase_index_for(value)
        if idx != self._active_idx:
            self._activate_phase(idx)

    def set_message(self, message: str):
        self.label.setText(message)

    # -- internals ---------------------------------------------------------

    def _phase_index_for(self, pct: int) -> int:
        """Return the index of the currently active phase, or ``len`` when done."""
        if pct >= 100:
            return len(_PHASES)  # all done
        for i, (_, threshold) in enumerate(_PHASES):
            if pct < threshold:
                return i - 1 if i > 0 else 0
        # pct is between last threshold and 100 → last phase
        return len(_PHASES) - 1

    def _activate_phase(self, idx: int):
        """Update the stepper to show *idx* as active, everything before as done."""
        self._active_idx = idx
        for i, row in enumerate(self._rows):
            if i < idx:
                row.mark_done()
            elif i == idx:
                row.mark_active()
            else:
                row.mark_pending()

        # When all done, mark last phase as done and show "Ready"
        if idx >= len(_PHASES):
            for row in self._rows:
                row.mark_done()
            self.label.setText(_DONE_LABEL)
