"""Editor de sincronía de un doblaje, con las dos formas de onda a la vista.

La herramienta de pistas de audio ya medía el desfase, pero ajustarlo a ciegas
es incómodo: aquí se ven las dos pistas en la misma línea de tiempo (el
doblaje desplazado por su desfase) y se puede escuchar una u otra desde el
mismo punto, cambiar de pista al vuelo y mover el doblaje con pasos pequeños.

El audio se decodifica por trozos y se reproduce con ``QAudioSink`` sobre una
fuente PCM propia: las dos pistas comparten posición de lectura, así que el
cambio A/B cae en la misma muestra y no hay dos reproductores que se puedan
desincronizar entre sí.
"""

from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
from PySide6.QtCore import (
    QIODevice,
    QPointF,
    QRectF,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.processing.audio_sync import (
    FRAME_SECONDS,
    MAX_OFFSET_SECONDS,
    PREVIEW_SAMPLE_RATE,
    PREVIEW_SECONDS,
    align_clip,
    detect_delay,
    read_pcm,
    read_waveform,
)

# Las dos pistas se mandan al altavoz en estéreo aunque se decodifiquen en
# mono: el formato mono no lo aceptan todos los dispositivos de Windows.
CHANNELS = 2


class WaveformWorker(QThread):
    """Lee la forma de onda de las dos pistas sin bloquear el diálogo."""

    progress = Signal(str)
    done = Signal(object, object)
    failed = Signal(str)

    def __init__(self, ffmpeg_path, base_path, base_stream, dub_path,
                 dub_stream, parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.base_path = base_path
        self.base_stream = int(base_stream)
        self.dub_path = dub_path
        self.dub_stream = int(dub_stream)
        self.abort_event = threading.Event()

    def cancel(self):
        self.abort_event.set()

    def run(self):
        try:
            self.progress.emit("Reading the original track…")
            base = read_waveform(
                self.ffmpeg_path, self.base_path,
                stream_index=self.base_stream,
                abort_event=self.abort_event,
            )
            if self.abort_event.is_set():
                self.failed.emit("Cancelled by user.")
                return
            self.progress.emit("Reading the added track…")
            dub = read_waveform(
                self.ffmpeg_path, self.dub_path,
                stream_index=self.dub_stream,
                abort_event=self.abort_event,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        if self.abort_event.is_set():
            self.failed.emit("Cancelled by user.")
            return
        self.done.emit(base, dub)


class SyncDetectWorker(QThread):
    """Mide el desfase mientras el editor sigue abierto."""

    done = Signal(float, float)
    failed = Signal(str)

    def __init__(self, ffmpeg_path, base_path, dub_path, dub_stream,
                 parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.base_path = base_path
        self.dub_path = dub_path
        self.dub_stream = int(dub_stream)
        self.abort_event = threading.Event()

    def cancel(self):
        self.abort_event.set()

    def run(self):
        try:
            result = detect_delay(
                self.ffmpeg_path, self.base_path, self.dub_path,
                dub_stream_index=self.dub_stream,
                abort_event=self.abort_event,
            )
        except Exception as exc:
            self.failed.emit(f"Could not compare the audio:\n{exc}")
            return
        if self.abort_event.is_set():
            self.failed.emit("Cancelled by user.")
            return
        offset = result.get("offset")
        if offset is None:
            self.failed.emit(
                result.get("reason") or "Could not detect the delay."
            )
            return
        self.done.emit(float(offset), float(result.get("confidence") or 0.0))


class WaveformView(QWidget):
    """Dos carriles de audio en la misma línea de tiempo, con cabezal."""

    seek_requested = Signal(float)

    MARGIN_LEFT = 12.0
    MARGIN_RIGHT = 12.0
    MARGIN_TOP = 26.0
    MARGIN_BOTTOM = 12.0
    LANE_GAP = 10.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.base_peaks = np.zeros(0, dtype=np.float32)
        self.dub_peaks = np.zeros(0, dtype=np.float32)
        self.frame_seconds = FRAME_SECONDS
        self.duration = 0.0
        self.position = 0.0
        self.delay = 0.0
        self.span = 30.0
        self.base_label = "Original"
        self.dub_label = "Added"
        self._dragging = False

    # ------------------------------------------------------------------ data
    def set_data(self, base, dub, duration, *, base_label="", dub_label=""):
        self.base_peaks = np.asarray(base, dtype=np.float32)
        self.dub_peaks = np.asarray(dub, dtype=np.float32)
        self.duration = max(0.0, float(duration))
        if base_label:
            self.base_label = base_label
        if dub_label:
            self.dub_label = dub_label
        self.update()

    def set_span(self, seconds: float) -> None:
        self.span = max(1.0, float(seconds))
        self.update()

    def set_position(self, seconds: float) -> None:
        self.position = max(0.0, float(seconds))
        self.update()

    def set_delay(self, seconds: float) -> None:
        self.delay = float(seconds)
        self.update()

    def view_range(self) -> tuple[float, float]:
        # Con un archivo más corto que el lapso se enseña entero, no un trozo
        # de 30 s con la mitad del carril vacío.
        if self.duration <= 0:
            return 0.0, max(1.0, self.span)
        if self.duration <= self.span:
            return 0.0, self.duration
        start = self.position - self.span / 2.0
        start = max(0.0, min(start, self.duration - self.span))
        return start, start + self.span

    # ----------------------------------------------------------------- paint
    def paintEvent(self, _event):  # noqa: N802 (API de Qt)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1020"))
        painter.setPen(QPen(QColor("#1f2c4c"), 1))
        painter.drawRoundedRect(
            QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 8, 8
        )
        left = self.MARGIN_LEFT
        top = self.MARGIN_TOP
        width = max(1.0, self.width() - left - self.MARGIN_RIGHT)
        usable = self.height() - top - self.MARGIN_BOTTOM - self.LANE_GAP
        lane_height = max(24.0, usable / 2.0)
        base_rect = QRectF(left, top, width, lane_height)
        dub_rect = QRectF(
            left, top + lane_height + self.LANE_GAP, width, lane_height
        )
        view_start, view_end = self.view_range()
        if view_end <= view_start:
            painter.end()
            return
        self._paint_ruler(painter, view_start, view_end, left, width)
        self._paint_lane(
            painter, base_rect, self.base_peaks, QColor("#00d2d3"),
            self.base_label, 0.0, view_start, view_end,
        )
        self._paint_lane(
            painter, dub_rect, self.dub_peaks, QColor("#f5a524"),
            f"{self.dub_label}  ({self.delay:+.2f} s)", self.delay,
            view_start, view_end,
        )
        self._paint_playhead(painter, view_start, view_end, left, width)
        painter.end()

    def _paint_ruler(self, painter, view_start, view_end, left, width):
        span = view_end - view_start
        step = self._tick_step(span)
        painter.setFont(QFont("Segoe UI", 7))
        tick = math.ceil(view_start / step) * step
        while tick <= view_end + 1e-6:
            x = left + (tick - view_start) / span * width
            painter.setPen(QPen(QColor("#2c3e50"), 1))
            painter.drawLine(QPointF(x, 18), QPointF(x, 25))
            painter.setPen(QColor("#8fa3b8"))
            painter.drawText(
                QRectF(x - 32, 2, 64, 15),
                Qt.AlignmentFlag.AlignCenter,
                self._clock(tick),
            )
            tick += step

    def _paint_lane(self, painter, rect, peaks, color, label, offset,
                    view_start, view_end):
        painter.fillRect(rect, QColor("#11182b"))
        middle = rect.center().y()
        painter.setPen(QPen(QColor("#243453"), 1))
        painter.drawLine(
            QPointF(rect.left(), middle), QPointF(rect.right(), middle)
        )
        if peaks.size:
            span = view_end - view_start
            pixels = max(1, int(rect.width()))
            peaks_end = peaks.size * self.frame_seconds
            painter.setPen(QPen(color, 1))
            for pixel in range(pixels):
                t0 = view_start + (pixel / pixels) * span + offset
                t1 = view_start + ((pixel + 1) / pixels) * span + offset
                if t0 >= peaks_end or t1 <= 0:
                    continue
                first = max(0, int(t0 / self.frame_seconds))
                last = min(peaks.size, int(t1 / self.frame_seconds) + 1)
                if last <= first:
                    continue
                amplitude = float(np.max(peaks[first:last]))
                if amplitude < 0.004:
                    continue
                half = (rect.height() / 2.0 - 3.0) * min(1.0, amplitude)
                x = rect.left() + pixel + 0.5
                painter.drawLine(
                    QPointF(x, middle - half), QPointF(x, middle + half)
                )
        painter.setFont(QFont("Segoe UI", 8))
        label_width = painter.fontMetrics().horizontalAdvance(label) + 12
        label_rect = QRectF(rect.left() + 6, rect.top() + 3,
                            label_width, 16)
        painter.fillRect(label_rect, QColor(11, 16, 32, 210))
        painter.setPen(color)
        painter.drawText(QPointF(rect.left() + 11, rect.top() + 15), label)

    def _paint_playhead(self, painter, view_start, view_end, left, width):
        if not view_start <= self.position <= view_end:
            return
        x = left + (self.position - view_start) / (view_end - view_start) * width
        painter.setPen(QPen(QColor("#ff5f7a"), 1.5))
        painter.drawLine(
            QPointF(x, self.MARGIN_TOP - 5),
            QPointF(x, self.height() - self.MARGIN_BOTTOM),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ff5f7a"))
        painter.drawPolygon(QPolygonF([
            QPointF(x - 5, self.MARGIN_TOP - 11),
            QPointF(x + 5, self.MARGIN_TOP - 11),
            QPointF(x, self.MARGIN_TOP - 2),
        ]))

    # ----------------------------------------------------------- interaction
    def mousePressEvent(self, event):  # noqa: N802 (API de Qt)
        if event.button() == Qt.MouseButton.LeftButton and self.duration > 0:
            self._dragging = True
            self._seek_to(event.position().x())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 (API de Qt)
        if self._dragging:
            self._seek_to(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 (API de Qt)
        self._dragging = False
        super().mouseReleaseEvent(event)

    def _seek_to(self, x: float) -> None:
        width = max(1.0, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT)
        start, end = self.view_range()
        ratio = max(0.0, min(1.0, (x - self.MARGIN_LEFT) / width))
        self.seek_requested.emit(start + ratio * (end - start))

    @staticmethod
    def _tick_step(span: float) -> float:
        for step in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
            if span / step <= 7:
                return float(step)
        return 900.0

    @staticmethod
    def _clock(seconds: float, tenths: bool = False) -> str:
        seconds = max(0.0, float(seconds))
        whole = int(seconds)
        text = f"{whole // 60}:{whole % 60:02d}"
        if tenths:
            text += f".{int((seconds - whole) * 10)}"
        return text


class _PcmStream(QIODevice):
    """PCM de dos pistas con una única posición de lectura.

    Cambiar de pista no mueve la posición: el cambio A/B se oye exactamente en
    el mismo punto de la línea de tiempo, sin que dos reproductores tengan que
    arrancar a la vez.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._buffers = {"base": b"", "dub": b""}
        self._mode = "base"
        self._position = 0

    def isSequential(self):  # noqa: N802 (API de Qt)
        return True

    def set_buffers(self, base: bytes, dub: bytes) -> None:
        with self._lock:
            self._buffers = {"base": bytes(base), "dub": bytes(dub)}

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = "dub" if mode == "dub" else "base"

    def set_seconds(self, seconds: float) -> None:
        with self._lock:
            self._position = max(
                0, int(round(float(seconds) * PREVIEW_SAMPLE_RATE))
            ) * CHANNELS * 2

    def readData(self, maxlen):  # noqa: N802 (API de Qt)
        with self._lock:
            data = self._buffers[self._mode]
            chunk = data[self._position:self._position + int(maxlen)]
            self._position += len(chunk)
        return bytes(chunk)

    def writeData(self, _data):  # noqa: N802 (API de Qt)
        return -1

    def bytesAvailable(self):  # noqa: N802 (API de Qt)
        with self._lock:
            remaining = max(
                0, len(self._buffers[self._mode]) - self._position
            )
        return remaining + super().bytesAvailable()


class AudioSyncDialog(QDialog):
    """Ver, oír y mover el desfase de un doblaje antes de unirlo."""

    def __init__(self, ffmpeg_path, base_path, dub_path, *, dub_stream=0,
                 delay=0.0, base_label="Original audio", dub_label="",
                 parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.base_path = base_path
        self.dub_path = dub_path
        self.dub_stream = int(dub_stream)
        self._delay = float(delay)
        self._duration = 0.0
        self._pos = 0.0
        self._mode = "base"
        self._playing = False
        self._chunk_start = 0.0
        self._chunk_loaded = False
        self._base_pcm = np.zeros(0, dtype=np.int16)
        self._dub_raw = np.zeros(0, dtype=np.int16)
        self._dub_raw_start = 0.0
        self._clock_origin = 0.0
        self._pos_origin = 0.0
        self._sink = None
        self._wave_worker = None
        self._detect_worker = None
        self._base_label = base_label
        self._dub_label = dub_label or os.path.basename(dub_path)

        self._stream = _PcmStream(self)
        self._stream.open(QIODevice.OpenModeFlag.ReadOnly)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_tick)

        self.setWindowTitle("Sync the added audio")
        self.resize(940, 540)
        self.setMinimumSize(720, 460)
        self._build_ui()
        self._apply_styles()
        self._start_loading()

    # -------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        title = QLabel("Sync the added audio")
        title.setObjectName("syncTitle")
        root.addWidget(title)

        source = QLabel(f"{self._base_label}   ↔   {self._dub_label}")
        source.setObjectName("syncSource")
        source.setWordWrap(True)
        root.addWidget(source)

        self.waveform = WaveformView(self)
        self.waveform.set_data(
            [], [], 0.0, base_label="Original",
            dub_label=os.path.basename(self.dub_path),
        )
        self.waveform.set_delay(self._delay)
        self.waveform.seek_requested.connect(self._seek)
        root.addWidget(self.waveform, 1)

        transport = QHBoxLayout()
        transport.setSpacing(8)
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("primaryButton")
        self.btn_play.setToolTip("Space: play or pause the current excerpt.")
        self.btn_play.clicked.connect(self._toggle_play)
        transport.addWidget(self.btn_play)

        self.btn_base = QPushButton("Original")
        self.btn_dub = QPushButton("Added dub")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for button, mode in ((self.btn_base, "base"), (self.btn_dub, "dub")):
            button.setObjectName("modeButton")
            button.setCheckable(True)
            self.mode_group.addButton(button)
            button.setProperty("syncMode", mode)
            transport.addWidget(button)
        self.btn_base.setChecked(True)
        self.btn_base.setToolTip(
            "Listen to the original track. Switching keeps the same position, "
            "so the same line can be compared in both tracks."
        )
        self.btn_dub.setToolTip(
            "Listen to the added track with its delay already applied."
        )
        self.mode_group.buttonToggled.connect(self._on_mode_toggled)

        self.position_label = QLabel("0:00.0 / 0:00.0")
        self.position_label.setObjectName("syncClock")
        transport.addSpacing(6)
        transport.addWidget(self.position_label)
        transport.addStretch(1)

        transport.addWidget(QLabel("Visible span"))
        self.zoom = QComboBox()
        for label, seconds in (("10 s", 10.0), ("20 s", 20.0),
                               ("30 s", 30.0), ("1 min", 60.0),
                               ("2 min", 120.0), ("5 min", 300.0)):
            self.zoom.addItem(label, seconds)
        self.zoom.setCurrentIndex(2)
        self.zoom.currentIndexChanged.connect(
            lambda _index: self.waveform.set_span(
                float(self.zoom.currentData() or 30.0)
            )
        )
        transport.addWidget(self.zoom)
        root.addLayout(transport)

        delay_row = QHBoxLayout()
        delay_row.setSpacing(8)
        delay_row.addWidget(QLabel("Delay of the added track"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(-MAX_OFFSET_SECONDS, MAX_OFFSET_SECONDS)
        self.delay_spin.setSingleStep(0.01)
        self.delay_spin.setDecimals(3)
        self.delay_spin.setSuffix(" s")
        self.delay_spin.setValue(self._delay)
        self.delay_spin.setToolTip(
            "Positive delays the dub, negative advances it. The waveform and "
            "the playback move with this value."
        )
        self.delay_spin.valueChanged.connect(self._set_delay)
        delay_row.addWidget(self.delay_spin)

        self.nudge_buttons = []
        for text, delta in (("−0.5", -0.5), ("−0.05", -0.05),
                            ("+0.05", 0.05), ("+0.5", 0.5)):
            button = QPushButton(text)
            button.setToolTip(f"Move the added track {delta:+.2f} s.")
            button.clicked.connect(
                lambda _checked=False, step=delta: self._set_delay(
                    self._delay + step
                )
            )
            delay_row.addWidget(button)
            self.nudge_buttons.append(button)

        self.btn_detect = QPushButton("Detect from audio")
        self.btn_detect.setToolTip(
            "Compares the energy envelope of both tracks and moves the dub to "
            "the measured offset. If the audio does not match clearly, the "
            "delay is left alone."
        )
        self.btn_detect.clicked.connect(self._detect)
        delay_row.addWidget(self.btn_detect)
        delay_row.addStretch(1)
        root.addLayout(delay_row)

        self.status = QLabel("Reading both tracks…")
        self.status.setObjectName("syncStatus")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_cancel)
        self.btn_accept = QPushButton("Use this delay")
        self.btn_accept.setObjectName("primaryButton")
        self.btn_accept.clicked.connect(self.accept)
        buttons.addWidget(self.btn_accept)
        root.addLayout(buttons)

        self._controls = [
            self.btn_play, self.btn_base, self.btn_dub, self.delay_spin,
            self.btn_detect, *self.nudge_buttons,
        ]
        self._set_controls_enabled(False)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background-color: #0f0f1a; color: #e0e0e0; }
            QLabel#syncTitle { color: #00d2d3; font-size: 19px;
                               font-weight: bold; }
            QLabel#syncSource { color: #8fa3b8; font-size: 11px; }
            QLabel#syncClock { color: #c7d2df; font-size: 12px;
                               font-family: Consolas, monospace; }
            QLabel#syncStatus { color: #8fa3b8; font-size: 10px; }
            QComboBox, QDoubleSpinBox {
                background-color: #1f2c4c; color: #e0e0e0;
                border: 1px solid #2c3e50; border-radius: 4px;
                padding: 5px 7px; min-height: 20px;
            }
            QPushButton {
                background-color: #1f2c4c; color: #e0e0e0;
                border: 1px solid #2c3e50; border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #2c3e50; }
            QPushButton:disabled { color: #5f6f7f; }
            QPushButton#primaryButton {
                background-color: #00d2d3; color: #0f0f1a;
                border: none; font-weight: bold;
            }
            QPushButton#primaryButton:hover { background-color: #26e6e7; }
            QPushButton#modeButton:checked {
                background-color: #f5a524; color: #0f0f1a;
                border-color: #f5a524; font-weight: bold;
            }
            """
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in self._controls:
            widget.setEnabled(enabled)

    # --------------------------------------------------------------- loading
    def _start_loading(self) -> None:
        self._wave_worker = WaveformWorker(
            self.ffmpeg_path, self.base_path, 0, self.dub_path,
            self.dub_stream, self,
        )
        self._wave_worker.progress.connect(self.status.setText)
        self._wave_worker.done.connect(self._on_waveforms)
        self._wave_worker.failed.connect(self._on_waveforms_failed)
        self._wave_worker.finished.connect(self._on_wave_finished)
        self._wave_worker.start()

    def _on_wave_finished(self) -> None:
        # Se suelta la referencia cuando el hilo ya ha terminado de verdad: si
        # se soltara al recibir ``done``, cerrar justo después destruiría un
        # QThread en marcha.
        self._wave_worker = None

    def _on_waveforms(self, base, dub) -> None:
        base = np.asarray(base, dtype=np.float32)
        dub = np.asarray(dub, dtype=np.float32)
        if base.size == 0:
            self._on_waveforms_failed(
                "The original track has no audio to show."
            )
            return
        self._duration = base.size * FRAME_SECONDS
        self.waveform.set_data(
            base, dub, self._duration,
            base_label="Original", dub_label=os.path.basename(self.dub_path),
        )
        self.waveform.set_delay(self._delay)
        self._pos = min(
            max(0.0, self._duration * 0.05),
            max(0.0, self._duration - 2.0),
        )
        self.waveform.set_position(self._pos)
        self._update_clock()
        if not self._load_chunk(self._pos):
            return
        self._set_controls_enabled(True)
        self.status.setText(
            "Click the waveform to move; Play then Original / Added dub to "
            "compare the same moment. Move the delay until the two waveforms "
            "line up and the lines fall together."
        )

    def _on_waveforms_failed(self, message: str) -> None:
        self.status.setText(message.replace("\n", " "))
        self._set_controls_enabled(False)
        QMessageBox.warning(self, "Sync the added audio", message)

    # ----------------------------------------------------------------- audio
    def _load_chunk(self, seconds: float) -> bool:
        if self._duration <= 0:
            return False
        start = max(0.0, min(float(seconds), max(0.0, self._duration - 1.0)))
        raw_start = max(0.0, start - MAX_OFFSET_SECONDS)
        raw_seconds = (
            PREVIEW_SECONDS + (start - raw_start) + MAX_OFFSET_SECONDS
        )
        try:
            base = read_pcm(
                self.ffmpeg_path, self.base_path, start=start,
                seconds=PREVIEW_SECONDS, stream_index=0,
            )
            raw = read_pcm(
                self.ffmpeg_path, self.dub_path, start=raw_start,
                seconds=raw_seconds, stream_index=self.dub_stream,
            )
        except Exception as exc:
            self.status.setText(f"Could not read the audio: {exc}")
            self._chunk_loaded = False
            return False
        if base.size == 0:
            self.status.setText("This video has no audio to preview.")
            self._chunk_loaded = False
            return False
        self._chunk_start = start
        self._base_pcm = base
        self._dub_raw = raw
        self._dub_raw_start = raw_start
        self._chunk_loaded = True
        self._rebuild_audio()
        return True

    @property
    def _chunk_end(self) -> float:
        return self._chunk_start + self._base_pcm.size / PREVIEW_SAMPLE_RATE

    def _rebuild_audio(self) -> None:
        if not self._chunk_loaded:
            return
        count = int(self._base_pcm.size)
        dub = align_clip(
            self._dub_raw, self._dub_raw_start, self._chunk_start,
            count, self._delay,
        )
        self._stream.set_buffers(
            self._stereo_bytes(self._base_pcm),
            self._stereo_bytes(dub),
        )

    @staticmethod
    def _stereo_bytes(samples: np.ndarray) -> bytes:
        return np.repeat(samples, CHANNELS).tobytes()

    def _ensure_sink(self) -> bool:
        if self._sink is not None:
            return True
        try:
            from PySide6.QtMultimedia import (
                QAudioFormat, QAudioSink, QMediaDevices,
            )

            audio_format = QAudioFormat()
            audio_format.setSampleRate(PREVIEW_SAMPLE_RATE)
            audio_format.setChannelCount(CHANNELS)
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            device = QMediaDevices.defaultAudioOutput()
            if device is None or device.isNull():
                raise RuntimeError("there is no audio output device")
            sink = QAudioSink(device, audio_format, self)
            # Búfer corto: con uno de un segundo, cambiar A/B seguiría sonando
            # la pista anterior durante casi un segundo.
            sink.setBufferSize(PREVIEW_SAMPLE_RATE * CHANNELS * 2 // 10)
        except Exception as exc:
            self.status.setText(
                f"Playback is not available ({exc}). The waveform and the "
                "delay controls still work."
            )
            return False
        self._sink = sink
        return True

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
            return
        if self._duration <= 0 or not self._chunk_loaded:
            return
        if self._pos >= self._duration - 0.1:
            self._seek(max(0.0, self._chunk_start))
        if not self._ensure_sink():
            return
        self._start_playback()

    def _start_playback(self) -> None:
        if not self._chunk_loaded or not (
            self._chunk_start <= self._pos < self._chunk_end
        ):
            if not self._load_chunk(self._pos):
                return
        self._stream.set_mode(self._mode)
        self._stream.set_seconds(max(0.0, self._pos - self._chunk_start))
        self._pos_origin = self._pos
        self._clock_origin = time.monotonic()
        self._sink.start(self._stream)
        self._playing = True
        self.btn_play.setText("Pause")
        self._timer.start()

    def _pause(self) -> None:
        if self._playing:
            self._pos = min(self._chunk_end, self._current_position())
        self._stop_audio()
        self.waveform.set_position(self._pos)
        self._update_clock()

    def _stop_audio(self) -> None:
        self._timer.stop()
        self._playing = False
        self.btn_play.setText("Play")
        if self._sink is not None:
            try:
                self._sink.stop()
            except Exception:
                pass

    def _current_position(self) -> float:
        if not self._playing:
            return self._pos
        return self._pos_origin + (time.monotonic() - self._clock_origin)

    def _on_tick(self) -> None:
        if not self._playing:
            return
        position = self._current_position()
        if position >= self._chunk_end - 0.05:
            if self._chunk_end >= self._duration - 0.1:
                self._pos = self._duration
                self._stop_audio()
                self.waveform.set_position(self._pos)
                self._update_clock()
                return
            self._stop_audio()
            self._pos = self._chunk_end
            if not self._load_chunk(self._pos):
                self.waveform.set_position(self._pos)
                self._update_clock()
                return
            self._start_playback()
            return
        self._pos = position
        self.waveform.set_position(position)
        self._update_clock()

    def _seek(self, seconds: float) -> None:
        was_playing = self._playing
        self._stop_audio()
        target = max(0.0, min(float(seconds), self._duration))
        if (
            not self._chunk_loaded
            or target < self._chunk_start
            or target > self._chunk_end - 0.5
        ):
            self._load_chunk(target)
        self._pos = target
        self.waveform.set_position(target)
        self._update_clock()
        if was_playing:
            self._start_playback()

    # ----------------------------------------------------------------- delay
    def _set_delay(self, value: float) -> None:
        value = max(-MAX_OFFSET_SECONDS, min(MAX_OFFSET_SECONDS, float(value)))
        self._delay = value
        if abs(self.delay_spin.value() - value) > 1e-9:
            self.delay_spin.blockSignals(True)
            self.delay_spin.setValue(value)
            self.delay_spin.blockSignals(False)
        self.waveform.set_delay(value)
        if self._chunk_loaded:
            self._rebuild_audio()
        self.status.setText(f"Delay {value:+.3f} s.")

    def _on_mode_toggled(self, button, checked: bool) -> None:
        if not checked:
            return
        mode = button.property("syncMode") or "base"
        self._mode = "dub" if mode == "dub" else "base"
        self._stream.set_mode(self._mode)

    def _detect(self) -> None:
        if self._detect_worker is not None or self._duration <= 0:
            return
        self.btn_detect.setEnabled(False)
        self.status.setText("Comparing both tracks…")
        worker = SyncDetectWorker(
            self.ffmpeg_path, self.base_path, self.dub_path,
            self.dub_stream, self,
        )
        worker.done.connect(self._on_detected)
        worker.failed.connect(self._on_detect_failed)
        worker.finished.connect(self._on_detect_finished)
        self._detect_worker = worker
        worker.start()

    def _on_detect_finished(self) -> None:
        self._detect_worker = None

    def _on_detected(self, offset: float, confidence: float) -> None:
        self.btn_detect.setEnabled(True)
        self._set_delay(offset)
        self.status.setText(
            f"Detected {offset:+.2f} s (confidence {confidence:.2f}). "
            "Press Play and switch Original / Added dub to check it."
        )

    def _on_detect_failed(self, message: str) -> None:
        self.btn_detect.setEnabled(True)
        self.status.setText(message.replace("\n", " "))

    def _update_clock(self) -> None:
        self.position_label.setText(
            f"{WaveformView._clock(self._pos, True)} / "
            f"{WaveformView._clock(self._duration)}"
        )

    # --------------------------------------------------------------- closing
    def delay(self) -> float:
        return float(self._delay)

    def keyPressEvent(self, event):  # noqa: N802 (API de Qt)
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play()
            event.accept()
            return
        super().keyPressEvent(event)

    def _stop_workers(self) -> bool:
        busy = False
        for worker in (self._wave_worker, self._detect_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                if not worker.wait(4000):
                    busy = True
        return not busy

    def accept(self) -> None:
        self._stop_audio()
        if not self._stop_workers():
            self.status.setText("Still closing the audio readers…")
            return
        super().accept()

    def reject(self) -> None:
        self._stop_audio()
        if not self._stop_workers():
            self.status.setText("Still closing the audio readers…")
            return
        super().reject()

    def closeEvent(self, event):  # noqa: N802 (API de Qt)
        self._stop_audio()
        if not self._stop_workers():
            event.ignore()
            self.status.setText("Still closing the audio readers…")
            return
        super().closeEvent(event)
