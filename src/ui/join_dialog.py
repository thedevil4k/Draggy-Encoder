"""Diálogo de Join videos: ordenar clips y recortar cada uno.

La lista es la timeline: arrastrar una fila (o usar Up/Down) cambia el orden
real del render. Cada clip conserva su propio ``start``/``end``; el panel de
la derecha edita el clip seleccionado y enseña miniaturas del IN y del OUT
para elegir el corte a ojo.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.ffmpeg.runner import extract_frame
from src.processing.join import (
    Clip,
    join_duration,
    probe_clip_payload,
    validate_clips,
)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


class JoinDialog(QDialog):
    """Editor de la timeline. ``result_clips()`` devuelve los ``Clip``."""

    def __init__(self, ffmpeg_path: str, parent=None, paths=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.clips: list[Clip] = []
        self._by_id: dict[int, Clip] = {}
        self._current_id: int | None = None
        self._loading = False
        self._thumb_dir = tempfile.mkdtemp(prefix="draggy-join-thumbs-")
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.timeout.connect(self._refresh_thumbnails)

        self.setWindowTitle("Join videos")
        self.resize(940, 640)
        self._build_ui()
        if paths:
            self.add_paths(paths)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        left.addWidget(self.list_widget, 1)

        buttons = QHBoxLayout()
        btn_add = QPushButton("Add videos…")
        btn_add.clicked.connect(self._choose_files)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove_current)
        btn_up = QPushButton("Up")
        btn_up.clicked.connect(lambda: self._move_current(-1))
        btn_down = QPushButton("Down")
        btn_down.clicked.connect(lambda: self._move_current(1))
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear)
        for btn in (btn_add, btn_remove, btn_up, btn_down, btn_clear):
            buttons.addWidget(btn)
        left.addLayout(buttons)

        self.total_label = QLabel("Total: 00:00.000 · 0 clips")
        self.total_label.setStyleSheet("color: #a0a0b0;")
        left.addWidget(self.total_label)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        trim_group = QGroupBox("Trim selected clip")
        trim_layout = QVBoxLayout(trim_group)

        self.start_spin = QDoubleSpinBox()
        self.end_spin = QDoubleSpinBox()
        for spin in (self.start_spin, self.end_spin):
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setSuffix(" s")
            spin.setKeyboardTracking(False)
        self.start_spin.valueChanged.connect(self._on_start_changed)
        self.end_spin.valueChanged.connect(self._on_end_changed)

        self.start_slider = QSlider(Qt.Horizontal)
        self.end_slider = QSlider(Qt.Horizontal)
        self.start_slider.setRange(0, 10000)
        self.end_slider.setRange(0, 10000)
        self.start_slider.valueChanged.connect(self._on_start_slider)
        self.end_slider.valueChanged.connect(self._on_end_slider)

        trim_layout.addWidget(QLabel("Start"))
        trim_layout.addWidget(self.start_slider)
        trim_layout.addWidget(self.start_spin)
        trim_layout.addWidget(QLabel("End"))
        trim_layout.addWidget(self.end_slider)
        trim_layout.addWidget(self.end_spin)

        self.trim_info = QLabel("Select a clip to edit its trim.")
        self.trim_info.setStyleSheet("color: #a0a0b0;")
        trim_layout.addWidget(self.trim_info)
        right.addWidget(trim_group)

        thumbs = QHBoxLayout()
        self.thumb_in = QLabel("IN")
        self.thumb_out = QLabel("OUT")
        for label in (self.thumb_in, self.thumb_out):
            label.setFixedSize(300, 169)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background: #101828; color: #667; border: 1px solid #2c3e50;")
            thumbs.addWidget(label)
        right.addLayout(thumbs)
        right.addStretch(1)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        right.addWidget(box)
        root.addLayout(right, 4)

    # --------------------------------------------------------------- datos
    def add_paths(self, paths) -> None:
        added = 0
        for path in paths or []:
            if not path or not os.path.exists(path):
                continue
            clip = probe_clip_payload(self.ffmpeg_path, {"path": path})
            if clip.duration <= 0:
                QMessageBox.warning(
                    self, "Join videos",
                    f"Could not read the duration of:\n{path}",
                )
                continue
            self._append_clip(clip)
            added += 1
        if added:
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def _append_clip(self, clip: Clip) -> None:
        self.clips.append(clip)
        key = id(clip)
        self._by_id[key] = clip
        item = QListWidgetItem()
        item.setData(Qt.UserRole, key)
        self.list_widget.addItem(item)
        self._refresh_item(item, clip, self.list_widget.count() - 1)
        self._update_total()

    def _refresh_item(self, item: QListWidgetItem, clip: Clip, index: int) -> None:
        item.setText(
            f"{index + 1}. {os.path.basename(clip.path)}   "
            f"trim {_fmt_time(clip.start)} → {_fmt_time(clip.end)}   "
            f"({clip.trim_duration:.2f}s)"
        )

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add videos to the timeline",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.wmv *.mov *.webm *.m4v);;All Files (*)",
        )
        if paths:
            self.add_paths(paths)

    def _current_clip(self) -> Clip | None:
        if self._current_id is not None:
            return self._by_id.get(self._current_id)
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.clips):
            return self.clips[row]
        return None

    def _on_row_changed(self, row: int) -> None:
        clip = self.clips[row] if 0 <= row < len(self.clips) else None
        if clip is None:
            self._current_id = None
            return
        self._current_id = id(clip)
        self._load_trim(clip)

    def _load_trim(self, clip: Clip) -> None:
        self._loading = True
        duration = max(0.1, clip.duration)
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(0.0, duration)
            spin.setVisible(True)
        self.start_spin.setValue(min(clip.start, duration))
        self.end_spin.setValue(min(clip.end, duration))
        self.start_slider.setValue(int(round(10000 * clip.start / duration)))
        self.end_slider.setValue(int(round(10000 * clip.end / duration)))
        self.trim_info.setText(
            f"{os.path.basename(clip.path)} · source {_fmt_time(clip.duration)} · "
            f"selected {_fmt_time(clip.trim_duration)}"
        )
        self._loading = False
        self._thumb_timer.start(250)

    def _sync_sliders(self, clip: Clip) -> None:
        duration = max(0.1, clip.duration)
        self._loading = True
        self.start_slider.setValue(int(round(10000 * clip.start / duration)))
        self.end_slider.setValue(int(round(10000 * clip.end / duration)))
        self._loading = False

    def _on_start_slider(self, value: int) -> None:
        if self._loading:
            return
        clip = self._current_clip()
        if clip is None:
            return
        clip.start = min(clip.duration * value / 10000.0, max(0.0, clip.end - 0.05))
        self._loading = True
        self.start_spin.setValue(clip.start)
        self._loading = False
        self._after_trim_change(clip)

    def _on_end_slider(self, value: int) -> None:
        if self._loading:
            return
        clip = self._current_clip()
        if clip is None:
            return
        clip.end = max(clip.duration * value / 10000.0, clip.start + 0.05)
        clip.end = min(clip.end, clip.duration)
        self._loading = True
        self.end_spin.setValue(clip.end)
        self._loading = False
        self._after_trim_change(clip)

    def _on_start_changed(self, value: float) -> None:
        if self._loading:
            return
        clip = self._current_clip()
        if clip is None:
            return
        clip.start = min(max(0.0, value), max(0.0, clip.end - 0.05))
        self._sync_sliders(clip)
        self._after_trim_change(clip)

    def _on_end_changed(self, value: float) -> None:
        if self._loading:
            return
        clip = self._current_clip()
        if clip is None:
            return
        clip.end = min(max(clip.start + 0.05, value), clip.duration)
        self._sync_sliders(clip)
        self._after_trim_change(clip)

    def _after_trim_change(self, clip: Clip) -> None:
        row = self.list_widget.currentRow()
        item = self.list_widget.item(row)
        if item is not None:
            self._refresh_item(item, clip, row)
        self.trim_info.setText(
            f"{os.path.basename(clip.path)} · source {_fmt_time(clip.duration)} · "
            f"selected {_fmt_time(clip.trim_duration)}"
        )
        self._update_total()
        self._thumb_timer.start(350)

    def _update_total(self) -> None:
        total = join_duration(self.clips)
        self.total_label.setText(
            f"Total: {_fmt_time(total)} · {len(self.clips)} clip(s)"
        )

    def _on_rows_moved(self, *_args) -> None:
        order: list[Clip] = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            clip = self._by_id.get(item.data(Qt.UserRole))
            if clip is not None:
                order.append(clip)
        if len(order) != len(self.clips):
            return
        self.clips = order
        for row, clip in enumerate(self.clips):
            self._refresh_item(self.list_widget.item(row), clip, row)
        self._update_total()

    def _move_current(self, delta: int) -> None:
        row = self.list_widget.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < self.list_widget.count()):
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(target, item)
        self.list_widget.setCurrentRow(target)
        self._on_rows_moved()

    def _remove_current(self) -> None:
        row = self.list_widget.currentRow()
        if row < 0:
            return
        item = self.list_widget.takeItem(row)
        clip = self._by_id.pop(item.data(Qt.UserRole), None)
        if clip is not None and clip in self.clips:
            self.clips.remove(clip)
        for i, remaining in enumerate(self.clips):
            self._refresh_item(self.list_widget.item(i), remaining, i)
        self._update_total()
        self._current_id = None
        if self.clips:
            self.list_widget.setCurrentRow(min(row, len(self.clips) - 1))

    def _clear(self) -> None:
        self.list_widget.clear()
        self.clips.clear()
        self._by_id.clear()
        self._current_id = None
        self._update_total()

    # ------------------------------------------------------------- preview
    def _refresh_thumbnails(self) -> None:
        clip = self._current_clip()
        if clip is None:
            return
        for label, time_sec in ((self.thumb_in, clip.start), (self.thumb_out, max(clip.start, clip.end - 0.05))):
            name = "in" if label is self.thumb_in else "out"
            target = os.path.join(self._thumb_dir, f"{id(clip)}_{name}.jpg")
            try:
                ok = extract_frame(self.ffmpeg_path, clip.path, time_sec, target)
            except Exception:
                ok = False
            if ok and os.path.exists(target):
                pix = QPixmap(target)
                if not pix.isNull():
                    label.setPixmap(
                        pix.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    )
                    continue
            label.setText("no frame")

    # --------------------------------------------------------------- cierre
    def _accept(self) -> None:
        errors = validate_clips(self.clips)
        if errors:
            QMessageBox.warning(self, "Join videos", "\n".join(errors))
            return
        self.accept()

    def result_clips(self) -> list[Clip]:
        return list(self.clips)

    def closeEvent(self, event) -> None:  # noqa: N802 (API de Qt)
        shutil.rmtree(self._thumb_dir, ignore_errors=True)
        super().closeEvent(event)
