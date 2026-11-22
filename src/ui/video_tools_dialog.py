"""Diálogo de herramientas de vídeo de un solo uso.

La lista de la izquierda imita la barra de herramientas de Artplayer; cada
entrada construye un comando FFmpeg independiente y lo ejecuta en un QThread
para que la ventana siga respondiendo.

No es el pipeline de compresión: no toca la cola, ni el códec elegido, ni la
cadena de IA. Sirve para resolver una cosa concreta y volver al flujo normal.
"""

from __future__ import annotations

import os
import threading

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ffmpeg.loudnorm import measure_loudness
from src.ffmpeg.runner import (
    get_audio_streams,
    get_video_info,
    measurement_is_usable,
    run_ffmpeg,
)
from src.processing.video_tools import (
    TOOL_DESCRIPTIONS,
    TOOL_LABELS,
    build_tool_args,
    plan_tool,
)
from src.utils.helpers import parse_time_to_seconds


class VideoToolWorker(QThread):
    """Ejecuta un ``VideoToolPlan`` y publica progreso y resultado."""

    progress = Signal(int)
    done = Signal(str)
    failed = Signal(str)
    stage = Signal(str)

    def __init__(self, ffmpeg_path, plan, parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.plan = plan
        self.abort_event = threading.Event()

    def cancel(self):
        self.abort_event.set()

    def _display_path(self) -> str:
        if self.plan.is_directory:
            return os.path.dirname(self.plan.output_path)
        return self.plan.output_path

    def _normalize_args(self) -> list[str] | None:
        """Mide el loudness de cada pista añadida y rehace el comando.

        La medición (paso 1 de loudnorm) se hace aquí, en el hilo del worker:
        en un episodio dura decenas de segundos y hacerlo al preparar el plan
        congelaría el diálogo. Devuelve ``None`` si el usuario aborta.
        """
        measurements = {}
        total = len(self.plan.normalize_sources)
        for position, (out_index, path, stream_index) in enumerate(
            self.plan.normalize_sources, start=1
        ):
            self.stage.emit(
                f"Measuring loudness ({position}/{total}): "
                f"{os.path.basename(path)}…"
            )
            measured = measure_loudness(
                self.ffmpeg_path,
                path,
                audio_stream_index=stream_index,
                abort_event=self.abort_event,
            )
            if self.abort_event.is_set():
                return None
            measurements[out_index] = (
                measured if measurement_is_usable(measured) else None
            )
        self.stage.emit("Adding the tracks…")
        return build_tool_args(
            self.plan.tool_id,
            self.plan.input_path,
            self.plan.output_path,
            {**self.plan.options, "measurements": measurements},
            has_audio=self.plan.has_audio,
            duration=self.plan.duration,
            width=self.plan.width,
            height=self.plan.height,
            audio_track_count=self.plan.audio_track_count,
        )

    def run(self):
        args = self.plan.args
        if self.plan.normalize_sources:
            try:
                normalized = self._normalize_args()
            except Exception as exc:
                self.failed.emit(f"Could not measure the audio levels:\n{exc}")
                return
            if normalized is None:
                self.failed.emit("Cancelled by user.")
                return
            args = normalized
        try:
            ok = run_ffmpeg(
                self.ffmpeg_path,
                args,
                progress_callback=self.progress.emit,
                abort_event=self.abort_event,
            )
        except Exception as exc:
            self.failed.emit(f"Could not run FFmpeg:\n{exc}")
            return

        if self.abort_event.is_set():
            self.failed.emit("Cancelled by user.")
            return
        if not ok:
            self.failed.emit(
                "FFmpeg could not finish this tool.\n\n"
                "Check bin/logs/ffmpeg.log for the last lines of the error."
            )
            return

        target = self._display_path()
        if self.plan.is_directory:
            has_parts = (
                os.path.isdir(target)
                and any(name.endswith(".mp4") for name in os.listdir(target))
            )
            if not has_parts:
                self.failed.emit("FFmpeg finished but no parts were created.")
                return
        elif not os.path.isfile(self.plan.output_path):
            self.failed.emit("FFmpeg finished but the output file is missing.")
            return

        self.done.emit(target)


class VideoToolsDialog(QDialog):
    """Pequeña suite de utilidades FFmpeg para el vídeo seleccionado."""

    def __init__(self, ffmpeg_path, input_path, output_dir, parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.input_path = input_path
        self.output_dir = output_dir or os.path.dirname(input_path) or os.getcwd()
        self.worker = None
        self._tool_ids = list(TOOL_LABELS.keys())
        self._panels = {}
        # Ficheros cuyo audio se añade al vídeo actual (herramienta multiaudio).
        self._audio_tracks: list[dict] = []
        self._info = get_video_info(ffmpeg_path, input_path) or {}
        self._duration = self._float_info("duration")
        self._width = int(self._float_info("width"))
        self._height = int(self._float_info("height"))
        self._has_audio = bool(get_audio_streams(ffmpeg_path, input_path))

        self.setWindowTitle("Video tools")
        self.resize(880, 590)
        self.setMinimumSize(760, 500)
        self._build_ui()
        self._apply_styles()
        self._on_tool_changed(0)

    # ------------------------------------------------------------------ data
    def _float_info(self, key) -> float:
        try:
            return float(self._info.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _resolution_text(self) -> str:
        if self._width and self._height:
            return f"{self._width} × {self._height}"
        return "unknown resolution"

    def _source_text(self) -> str:
        name = os.path.basename(self.input_path)
        bits = [name, self._resolution_text()]
        if self._duration > 0:
            bits.append(f"{self._duration:.1f}s")
        if self._has_audio:
            bits.append("audio")
        return "  ·  ".join(bits)

    # -------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        title = QLabel("Video tools")
        title.setObjectName("toolsTitle")
        root.addWidget(title)

        source = QLabel(self._source_text())
        source.setObjectName("toolsSource")
        source.setWordWrap(True)
        root.addWidget(source)

        body = QHBoxLayout()
        body.setSpacing(14)

        self.tool_list = QListWidget()
        self.tool_list.setObjectName("toolList")
        self.tool_list.setFixedWidth(218)
        self.tool_list.setSpacing(2)
        for tool_id in self._tool_ids:
            self.tool_list.addItem(TOOL_LABELS[tool_id])
        self.tool_list.currentRowChanged.connect(self._on_tool_changed)
        body.addWidget(self.tool_list)

        right = QVBoxLayout()
        right.setSpacing(8)
        self.description = QLabel()
        self.description.setObjectName("toolDescription")
        self.description.setWordWrap(True)
        right.addWidget(self.description)

        self.stack = QStackedWidget()
        for tool_id in self._tool_ids:
            panel = self._build_tool_panel(tool_id)
            self._panels[tool_id] = panel
            self.stack.addWidget(panel)
        right.addWidget(self.stack, 1)

        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        output_row.addWidget(QLabel("Output folder"))
        self.output_edit = QLineEdit(self.output_dir)
        self.output_edit.setReadOnly(True)
        output_row.addWidget(self.output_edit, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._choose_output_dir)
        output_row.addWidget(btn_browse)
        right.addLayout(output_row)

        self.output_hint = QLabel(
            "A new filename is created from the source name; existing files are never overwritten."
        )
        self.output_hint.setObjectName("toolHint")
        self.output_hint.setWordWrap(True)
        right.addWidget(self.output_hint)

        self.progress = QProgressBarLike()
        right.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.status = QLabel("Ready")
        self.status.setObjectName("toolStatus")
        self.status.setWordWrap(True)
        buttons.addWidget(self.status, 1)

        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._abort)
        buttons.addWidget(self.btn_abort)

        self.btn_run = QPushButton("Run tool")
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self._run)
        buttons.addWidget(self.btn_run)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        buttons.addWidget(self.btn_close)
        right.addLayout(buttons)

        body.addLayout(right, 1)
        root.addLayout(body, 1)

    def _build_tool_panel(self, tool_id: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        if tool_id == "screenshot":
            self.shot_time = QLineEdit(
                "00:00:01" if self._duration <= 0 or self._duration >= 1 else "0"
            )
            self.shot_time.setPlaceholderText("HH:MM:SS, MM:SS or seconds")
            form.addRow("Frame at", self.shot_time)
            self.shot_format = QComboBox()
            self.shot_format.addItems(["PNG", "JPG"])
            form.addRow("Image format", self.shot_format)

        elif tool_id == "pixelate":
            self.pixel_block = QSpinBox()
            self.pixel_block.setRange(2, 128)
            self.pixel_block.setSingleStep(4)
            self.pixel_block.setValue(16)
            self.pixel_block.setSuffix(" px")
            form.addRow("Block size", self.pixel_block)

        elif tool_id == "blur":
            self.blur_sigma = QDoubleSpinBox()
            self.blur_sigma.setRange(0.5, 50.0)
            self.blur_sigma.setSingleStep(1.0)
            self.blur_sigma.setDecimals(1)
            self.blur_sigma.setValue(8.0)
            form.addRow("Strength (sigma)", self.blur_sigma)

        elif tool_id == "stabilize":
            note = QLabel(
                "Uses deshake with mirrored edges. It reduces shake but cannot "
                "recover motion blur or rolling shutter."
            )
            note.setWordWrap(True)
            note.setObjectName("toolHint")
            form.addRow(note)

        elif tool_id == "delogo":
            self.logo_x = QSpinBox()
            self.logo_y = QSpinBox()
            self.logo_w = QSpinBox()
            self.logo_h = QSpinBox()
            for spin, maximum in (
                (self.logo_x, max(1, self._width - 2)),
                (self.logo_y, max(1, self._height - 2)),
                (self.logo_w, max(2, self._width)),
                (self.logo_h, max(2, self._height)),
            ):
                spin.setRange(0, maximum)
                spin.setSingleStep(2)
            self.logo_w.setMinimum(2)
            self.logo_h.setMinimum(2)
            self.logo_x.setValue(10 if self._width > 20 else 0)
            self.logo_y.setValue(10 if self._height > 20 else 0)
            self.logo_w.setValue(max(2, min(self._width or 120, 120)))
            self.logo_h.setValue(max(2, min(self._height or 60, 60)))
            form.addRow("X", self.logo_x)
            form.addRow("Y", self.logo_y)
            form.addRow("Width", self.logo_w)
            form.addRow("Height", self.logo_h)
            note = QLabel(
                f"Source: {self._resolution_text()}. Increase the rectangle slightly "
                "past the logo so the edges do not remain visible."
            )
            note.setWordWrap(True)
            note.setObjectName("toolHint")
            form.addRow(note)

        elif tool_id == "reverse":
            note = QLabel(
                "The whole clip is buffered before being written. Audio is "
                "reversed too when the source has an audio track."
            )
            note.setWordWrap(True)
            note.setObjectName("toolHint")
            form.addRow(note)

        elif tool_id == "split":
            self.split_parts = QSpinBox()
            self.split_parts.setRange(2, 50)
            self.split_parts.setValue(2)
            form.addRow("Parts", self.split_parts)
            note = QLabel(
                "Parts are re-encoded with forced keyframes so the cuts land "
                "at the requested time. The duration is split into equal parts."
            )
            note.setWordWrap(True)
            note.setObjectName("toolHint")
            form.addRow(note)

        elif tool_id == "volume":
            self.volume_db = QDoubleSpinBox()
            self.volume_db.setRange(-60.0, 30.0)
            self.volume_db.setSingleStep(1.0)
            self.volume_db.setDecimals(1)
            self.volume_db.setValue(3.0)
            self.volume_db.setSuffix(" dB")
            form.addRow("Gain", self.volume_db)
            note = QLabel(
                "The video stream is copied; only the audio is re-encoded. "
                "The source has no audio track." if not self._has_audio else
                "The video stream is copied; only the audio is re-encoded."
            )
            note.setWordWrap(True)
            note.setObjectName("toolHint")
            if not self._has_audio:
                note.setStyleSheet("color: #f39c12;")
            form.addRow(note)

        elif tool_id == "multiaudio":
            self.multi_container = QComboBox()
            self.multi_container.addItem(
                "MKV (tracks copied; added ones re-encoded if normalised)", "mkv"
            )
            self.multi_container.addItem("MP4 (extra tracks re-encoded to AAC)", "mp4")
            form.addRow("Output format", self.multi_container)

            self.multi_language = QComboBox()
            for label, code in (
                ("Spanish", "spa"),
                ("English", "eng"),
                ("French", "fra"),
                ("German", "deu"),
                ("Italian", "ita"),
                ("Portuguese", "por"),
                ("Japanese", "jpn"),
                ("No language tag", ""),
            ):
                self.multi_language.addItem(label, code)
            form.addRow("Language of new tracks", self.multi_language)

            self.multi_normalize = QComboBox()
            self.multi_normalize.addItem("Copy the added tracks as they are", False)
            self.multi_normalize.addItem(
                "Normalise loudness (EBU R128, −16 LUFS)", True
            )
            self.multi_normalize.setToolTip(
                "Same EBU R128 loudnorm as the main render's Audio Boost, applied "
                "to the added tracks only. The level is measured on each added "
                "file, so no dB value has to be worked out by hand."
            )
            form.addRow("Volume of added tracks", self.multi_normalize)

            self.multi_normalize_original = QCheckBox(
                "Also normalise the video's own tracks"
            )
            self.multi_normalize_original.setEnabled(False)
            self.multi_normalize_original.setToolTip(
                "Levels every audio track of the file to the same target, so "
                "switching language keeps the same volume. The video's own "
                "tracks are re-encoded to AAC; the picture is still copied."
            )
            self.multi_normalize.currentIndexChanged.connect(
                lambda _index: self.multi_normalize_original.setEnabled(
                    bool(self.multi_normalize.currentData())
                )
            )
            form.addRow(self.multi_normalize_original)

            self.multi_tracks = QListWidget()
            self.multi_tracks.setObjectName("trackList")
            self.multi_tracks.setMinimumHeight(88)
            form.addRow("Added audio", self.multi_tracks)

            track_buttons = QHBoxLayout()
            track_buttons.setSpacing(8)
            self.btn_add_audio = QPushButton("Add dubbed episode…")
            self.btn_add_audio.clicked.connect(self._add_audio_files)
            self.btn_remove_audio = QPushButton("Remove selected")
            self.btn_remove_audio.clicked.connect(self._remove_audio_file)
            track_buttons.addWidget(self.btn_add_audio)
            track_buttons.addWidget(self.btn_remove_audio)
            track_buttons.addStretch(1)
            form.addRow(track_buttons)

            note = QLabel(
                "Pick the other episode as it is: its first audio track is taken "
                "directly, with no need to extract or convert anything. The video "
                "and the tracks it already has are copied without re-encoding, so "
                "use the same episode so the tracks stay in sync. With loudness "
                "normalisation each added file is measured first and only those "
                "tracks are re-encoded, all the dubs landing on the same target "
                "(-16 LUFS, -1.5 dBTP) instead of the raw level of each rip. "
                "Tick the box to level the video's own tracks too, so switching "
                "language does not change the volume."
            )
            note.setWordWrap(True)
            note.setObjectName("toolHint")
            form.addRow(note)

        layout.addLayout(form)
        layout.addStretch(1)
        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background-color: #0f0f1a; color: #e0e0e0; }
            QLabel#toolsTitle { color: #00d2d3; font-size: 20px; font-weight: bold; }
            QLabel#toolsSource { color: #8fa3b8; font-size: 11px; }
            QLabel#toolDescription { color: #e0e0e0; font-size: 13px; }
            QLabel#toolHint { color: #8fa3b8; font-size: 10px; }
            QLabel#toolStatus { color: #a0a0b0; font-size: 11px; }
            QListWidget#toolList {
                background-color: #16213e; border: 1px solid #2c3e50;
                border-radius: 8px; padding: 4px; outline: 0;
            }
            QListWidget#toolList::item {
                padding: 9px 8px; border-radius: 6px; color: #c7d2df;
            }
            QListWidget#toolList::item:selected {
                background-color: #00d2d3; color: #0f0f1a; font-weight: bold;
            }
            QListWidget#toolList::item:hover:!selected { background-color: #1f2c4c; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #1f2c4c; color: #e0e0e0;
                border: 1px solid #2c3e50; border-radius: 4px;
                padding: 5px 7px; min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #00d2d3;
            }
            QPushButton {
                background-color: #1f2c4c; color: #e0e0e0;
                border: 1px solid #2c3e50; border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #2c3e50; }
            QListWidget#trackList {
                background-color: #1f2c4c; color: #e0e0e0;
                border: 1px solid #2c3e50; border-radius: 4px; padding: 4px;
            }
            QPushButton:disabled { color: #5f6f7f; }
            QPushButton#primaryButton {
                background-color: #00d2d3; color: #0f0f1a;
                border: none; font-weight: bold;
            }
            QPushButton#primaryButton:hover { background-color: #26e6e7; }
            QFrame#progressTrack { background-color: #1f2c4c; border-radius: 3px; }
            QFrame#progressFill { background-color: #00d2d3; border-radius: 3px; }
            """
        )

    # --------------------------------------------------------------- actions
    def _on_tool_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._tool_ids):
            return
        self.stack.setCurrentIndex(row)
        tool_id = self._tool_ids[row]
        self.description.setText(TOOL_DESCRIPTIONS[tool_id])
        self.status.setText("Ready")
        self.progress.setValue(0)

    def _choose_output_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_edit.text()
        )
        if chosen:
            self.output_edit.setText(chosen)

    def _current_tool_id(self) -> str:
        row = max(0, self.tool_list.currentRow())
        return self._tool_ids[row]

    def _add_audio_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add the dubbed episode (video or audio)",
            self.output_dir,
            "Video or audio (*.mkv *.mp4 *.m4v *.mov *.avi *.webm *.ts *.m2ts "
            "*.mka *.m4a *.aac *.mp3 *.ac3 *.eac3 *.dts *.flac *.wav *.opus "
            "*.ogg);;Video (*.mkv *.mp4 *.m4v *.mov *.avi *.webm *.ts *.m2ts);;"
            "Audio (*.mka *.m4a *.aac *.mp3 *.ac3 *.eac3 *.dts *.flac *.wav "
            "*.opus *.ogg);;All files (*)",
        )
        if not paths:
            return
        code = self.multi_language.currentData() or ""
        label = self.multi_language.currentText() if code else "No language tag"
        source_key = os.path.normcase(os.path.abspath(self.input_path))
        added = 0
        skipped = []
        for path in paths:
            key = os.path.normcase(os.path.abspath(path))
            if key == source_key:
                continue
            if any(
                os.path.normcase(os.path.abspath(track["path"])) == key
                for track in self._audio_tracks
            ):
                continue
            # Un vídeo entero también vale: se usa su primera pista de audio,
            # sin extraer ni convertir nada antes.
            streams = get_audio_streams(self.ffmpeg_path, path)
            if not streams:
                skipped.append(os.path.basename(path))
                continue
            first = streams[0]
            detail_bits = [str(first.get("codec_name") or "audio")]
            channels = first.get("channels")
            if channels:
                detail_bits.append(f"{channels}ch")
            source_language = str(first.get("language") or "")
            if source_language and source_language != "und":
                detail_bits.append(source_language)
            detail = " ".join(detail_bits)
            self._audio_tracks.append({
                "path": path,
                "language": code,
                "title": self.multi_language.currentText() if code else "",
            })
            self.multi_tracks.addItem(
                f"{os.path.basename(path)}  ·  {detail}  ·  {label}"
            )
            added += 1
        if skipped:
            QMessageBox.warning(
                self,
                "Add audio tracks",
                "These files have no audio track and were skipped:\n\n"
                + "\n".join(skipped),
            )
        if added:
            self.status.setText(f"{len(self._audio_tracks)} audio track(s) to add.")
        elif not skipped:
            self.status.setText("Those files are already in the list.")

    def _remove_audio_file(self) -> None:
        row = self.multi_tracks.currentRow()
        if row < 0 or row >= len(self._audio_tracks):
            return
        self.multi_tracks.takeItem(row)
        self._audio_tracks.pop(row)
        self.status.setText(f"{len(self._audio_tracks)} audio track(s) to add.")

    def _options(self) -> dict:
        tool_id = self._current_tool_id()
        if tool_id == "screenshot":
            text = self.shot_time.text().strip()
            try:
                stamp = parse_time_to_seconds(text) if text else 0.0
            except (TypeError, ValueError):
                raise ValueError("Frame time must be seconds or MM:SS / HH:MM:SS.")
            return {
                "time": stamp,
                "format": self.shot_format.currentText().lower(),
            }
        if tool_id == "pixelate":
            return {"block": self.pixel_block.value()}
        if tool_id == "blur":
            return {"sigma": self.blur_sigma.value()}
        if tool_id == "delogo":
            return {
                "x": self.logo_x.value(),
                "y": self.logo_y.value(),
                "w": self.logo_w.value(),
                "h": self.logo_h.value(),
            }
        if tool_id == "split":
            return {"parts": self.split_parts.value()}
        if tool_id == "volume":
            return {"db": self.volume_db.value()}
        if tool_id == "multiaudio":
            if not self._audio_tracks:
                raise ValueError("Add at least one audio file first.")
            return {
                "tracks": [dict(track) for track in self._audio_tracks],
                "container": self.multi_container.currentData() or "mkv",
                "normalize": bool(self.multi_normalize.currentData()),
                "normalize_original": (
                    bool(self.multi_normalize.currentData())
                    and self.multi_normalize_original.isChecked()
                ),
            }
        return {}

    def _set_running(self, running: bool) -> None:
        self.tool_list.setEnabled(not running)
        self.output_edit.setEnabled(not running)
        self.btn_run.setEnabled(not running)
        self.btn_abort.setEnabled(running)
        self.btn_close.setEnabled(not running)
        for name in (
            "multi_tracks",
            "multi_container",
            "multi_language",
            "multi_normalize",
            "multi_normalize_original",
            "btn_add_audio",
            "btn_remove_audio",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not running)
        # El check de nivelar las pistas originales solo tiene sentido con la
        # normalización activa, también al volver del render.
        if getattr(self, "multi_normalize_original", None) is not None:
            self.multi_normalize_original.setEnabled(
                not running and bool(self.multi_normalize.currentData())
            )

    def _run(self) -> None:
        if self.worker is not None:
            return
        if not self.output_edit.text().strip():
            QMessageBox.warning(self, "Video tools", "Choose an output folder.")
            return
        try:
            plan = plan_tool(
                self.ffmpeg_path,
                self._current_tool_id(),
                self.input_path,
                self.output_edit.text().strip(),
                self._options(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Video tools", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Video tools", f"Could not prepare this tool:\n{exc}"
            )
            return

        self.progress.setValue(0)
        self.status.setText("Running…")
        self._set_running(True)
        self.worker = VideoToolWorker(self.ffmpeg_path, plan, self)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.stage.connect(self.status.setText)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _abort(self) -> None:
        if self.worker is not None:
            self.status.setText("Stopping…")
            self.worker.cancel()
            self.btn_abort.setEnabled(False)

    def _on_done(self, path: str) -> None:
        self._set_running(False)
        self.worker = None
        self.progress.setValue(100)
        self.status.setText(f"Done: {path}")
        QMessageBox.information(
            self, "Video tools", f"Finished successfully:\n\n{path}"
        )

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.worker = None
        self.progress.setValue(0)
        self.status.setText(message.replace("\n", " "))
        if message != "Cancelled by user.":
            QMessageBox.warning(self, "Video tools", message)

    def closeEvent(self, event) -> None:  # noqa: N802 (API de Qt)
        if self.worker is not None:
            event.ignore()
            self.status.setText("Abort the running tool before closing.")
            return
        super().closeEvent(event)


class QProgressBarLike(QFrame):
    """Barra fina sin el texto blanco de QProgressBar sobre fondo oscuro."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("progressTrack")
        self.setFixedHeight(6)
        self._value = 0
        self._fill = QFrame(self)
        self._fill.setObjectName("progressFill")
        self._fill.setGeometry(0, 0, 0, 6)

    def setValue(self, value: int) -> None:
        self._value = max(0, min(100, int(value)))
        width = int(self.width() * self._value / 100)
        self._fill.setGeometry(0, 0, width, self.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (API de Qt)
        super().resizeEvent(event)
        self.setValue(self._value)
