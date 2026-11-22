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
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ffmpeg.loudnorm import measure_loudness
from src.ffmpeg.runner import (
    get_audio_streams,
    get_subtitle_streams,
    get_video_info,
    measurement_is_usable,
    run_ffmpeg,
)
from src.processing.audio_sync import detect_delay
from src.ui.audio_sync_dialog import AudioSyncDialog
from src.processing.video_tools import (
    TOOL_DESCRIPTIONS,
    TOOL_LABELS,
    VIDEO_EXTENSIONS,
    build_tool_args,
    match_dub,
    media_files,
    plan_tool,
)
from src.utils.helpers import parse_time_to_seconds


def _execute_plan(ffmpeg_path, plan, abort_event, on_progress, on_stage):
    """Mide el loudness si toca, ejecuta el plan y comprueba la salida.

    Devuelve ``(ruta_final, error)``; con éxito el error es una cadena vacía.
    Lo usan tanto el worker de un solo vídeo como el de lote, para que las dos
    rutas midan, ejecuten y validen exactamente igual.
    """
    args = plan.args
    options = dict(plan.options or {})

    if plan.delay_sources:
        tracks = [dict(track) for track in (options.get("tracks") or [])]
        total = len(plan.delay_sources)
        for position, (index, path, stream_index) in enumerate(
            plan.delay_sources, start=1
        ):
            on_stage(
                f"Detecting delay ({position}/{total}): "
                f"{os.path.basename(path)}…"
            )
            result = detect_delay(
                ffmpeg_path,
                plan.input_path,
                path,
                dub_stream_index=stream_index,
                abort_event=abort_event,
            )
            if abort_event.is_set():
                return "", "Cancelled by user."
            offset = result.get("offset")
            if offset is None:
                # Sin una medición fiable se deja el desfase que hubiera: es
                # mejor un mux sin tocar que uno movido a ciegas.
                on_stage(result.get("reason") or "No delay detected.")
                continue
            if index < len(tracks):
                tracks[index]["delay"] = offset
            on_stage(
                f"Delay {offset:+.2f} s "
                f"(confidence {float(result.get('confidence') or 0.0):.2f})"
            )
        options["tracks"] = tracks

    if plan.normalize_sources:
        measurements = {}
        total = len(plan.normalize_sources)
        for position, (out_index, path, stream_index) in enumerate(
            plan.normalize_sources, start=1
        ):
            on_stage(
                f"Measuring loudness ({position}/{total}): "
                f"{os.path.basename(path)}…"
            )
            measured = measure_loudness(
                ffmpeg_path,
                path,
                audio_stream_index=stream_index,
                abort_event=abort_event,
            )
            if abort_event.is_set():
                return "", "Cancelled by user."
            measurements[out_index] = (
                measured if measurement_is_usable(measured) else None
            )
        options["measurements"] = measurements
        on_stage("Adding the tracks…")

    if plan.delay_sources or plan.normalize_sources:
        args = build_tool_args(
            plan.tool_id,
            plan.input_path,
            plan.output_path,
            options,
            has_audio=plan.has_audio,
            duration=plan.duration,
            width=plan.width,
            height=plan.height,
            audio_track_count=plan.audio_track_count,
        )

    ok = run_ffmpeg(
        ffmpeg_path,
        args,
        progress_callback=on_progress,
        abort_event=abort_event,
    )
    if abort_event.is_set():
        return "", "Cancelled by user."
    if not ok:
        return "", (
            "FFmpeg could not finish this tool.\n\n"
            "Check bin/logs/ffmpeg.log for the last lines of the error."
        )

    target = (
        os.path.dirname(plan.output_path) if plan.is_directory
        else plan.output_path
    )
    if plan.is_directory:
        has_parts = (
            os.path.isdir(target)
            and any(name.endswith(".mp4") for name in os.listdir(target))
        )
        if not has_parts:
            return "", "FFmpeg finished but no parts were created."
    elif not os.path.isfile(target):
        return "", "FFmpeg finished but the output file is missing."
    return target, ""


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

    def run(self):
        try:
            target, error = _execute_plan(
                self.ffmpeg_path,
                self.plan,
                self.abort_event,
                self.progress.emit,
                self.stage.emit,
            )
        except Exception as exc:
            self.failed.emit(f"Could not run FFmpeg:\n{exc}")
            return
        if error:
            self.failed.emit(error)
            return
        self.done.emit(target)


class DelayDetectWorker(QThread):
    """Mide el desfase entre el vídeo y un doblaje sin bloquear el diálogo."""

    done = Signal(float, float)
    failed = Signal(str)

    def __init__(self, ffmpeg_path, base_path, dub_path, parent=None,
                 stream_index=0):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.base_path = base_path
        self.dub_path = dub_path
        self.stream_index = int(stream_index)
        self.abort_event = threading.Event()

    def cancel(self):
        self.abort_event.set()

    def run(self):
        try:
            result = detect_delay(
                self.ffmpeg_path,
                self.base_path,
                self.dub_path,
                dub_stream_index=self.stream_index,
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
        self.done.emit(
            float(offset), float(result.get("confidence") or 0.0)
        )


class BatchAudioWorker(QThread):
    """Une el doblaje de varios episodios en serie, con estado por fila."""

    row_status = Signal(int, str)
    stage = Signal(str)
    progress = Signal(int)
    done = Signal(int, int)
    failed = Signal(str)

    def __init__(self, ffmpeg_path, rows, output_dir, options, track_meta,
                 parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.rows = rows
        self.output_dir = output_dir
        self.options = options
        self.track_meta = track_meta
        self.abort_event = threading.Event()

    def cancel(self):
        self.abort_event.set()

    def run(self):
        total = max(1, len(self.rows))
        done_count = 0
        failed_count = 0
        for index, row in enumerate(self.rows):
            if self.abort_event.is_set():
                self.failed.emit("Cancelled by user.")
                return
            name = os.path.basename(row["episode"])
            self.stage.emit(f"Episode {index + 1}/{total}: {name}")
            self.row_status.emit(index, "Preparing…")
            options = {
                **self.options,
                "tracks": [{**self.track_meta, "path": row["dub"]}],
            }
            try:
                plan = plan_tool(
                    self.ffmpeg_path,
                    "multiaudio",
                    row["episode"],
                    self.output_dir,
                    options,
                )
            except Exception as exc:
                self.row_status.emit(index, f"Failed: {exc}")
                failed_count += 1
                continue

            def _on_progress(value, _index=index):
                self.progress.emit(
                    int((_index + max(0, min(100, value)) / 100) / total * 100)
                )

            try:
                target, error = _execute_plan(
                    self.ffmpeg_path,
                    plan,
                    self.abort_event,
                    _on_progress,
                    self.stage.emit,
                )
            except Exception as exc:
                error = str(exc)
                target = ""
            if self.abort_event.is_set():
                self.row_status.emit(index, "Cancelled")
                self.failed.emit("Cancelled by user.")
                return
            if error:
                first_line = error.splitlines()[0] if error else "Failed"
                self.row_status.emit(index, f"Failed: {first_line}")
                failed_count += 1
            else:
                self.row_status.emit(
                    index, f"Done: {os.path.basename(target)}"
                )
                done_count += 1
        self.progress.emit(100)
        self.done.emit(done_count, failed_count)


class VideoToolsDialog(QDialog):
    """Pequeña suite de utilidades FFmpeg para el vídeo seleccionado."""

    def __init__(self, ffmpeg_path, input_path, output_dir, parent=None):
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.input_path = input_path
        self.output_dir = output_dir or os.path.dirname(input_path) or os.getcwd()
        self.worker = None
        self._detect_worker = None
        self._loading_delay = False
        self._tool_ids = list(TOOL_LABELS.keys())
        self._panels = {}
        # Ficheros cuyo audio se añade al vídeo actual (herramienta multiaudio).
        self._audio_tracks: list[dict] = []
        # Estado del modo lote: episodios elegidos, doblaje de cada uno y
        # carpeta única de doblajes (vacía = buscar en la carpeta de cada uno).
        self._batch_episodes: list[str] = []
        self._batch_rows: list[dict] = []
        self._batch_dub_dir = ""
        self._batch_table_rows: list[int] = []
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

            self.multi_batch = QCheckBox(
                "Batch: merge a whole folder or several episodes"
            )
            self.multi_batch.setToolTip(
                "Each episode is paired with its dub by S01E03 (or by identical "
                "name) and the settings above are applied to all of them."
            )
            self.multi_batch.toggled.connect(self._toggle_batch)
            form.addRow(self.multi_batch)

            self.multi_single_box = QWidget()
            single_box = QVBoxLayout(self.multi_single_box)
            single_box.setContentsMargins(0, 0, 0, 0)
            single_box.setSpacing(8)

            self.multi_tracks = QListWidget()
            self.multi_tracks.setObjectName("trackList")
            self.multi_tracks.setMinimumHeight(88)
            single_box.addWidget(self.multi_tracks)

            track_buttons = QHBoxLayout()
            track_buttons.setSpacing(8)
            self.btn_add_audio = QPushButton("Add dubbed episode…")
            self.btn_add_audio.clicked.connect(self._add_audio_files)
            self.btn_remove_audio = QPushButton("Remove selected")
            self.btn_remove_audio.clicked.connect(self._remove_audio_file)
            track_buttons.addWidget(self.btn_add_audio)
            track_buttons.addWidget(self.btn_remove_audio)
            track_buttons.addStretch(1)
            single_box.addLayout(track_buttons)

            stream_row = QHBoxLayout()
            stream_row.setSpacing(8)
            stream_row.addWidget(QLabel("Audio track of the added file"))
            self.multi_stream = QComboBox()
            self.multi_stream.setToolTip(
                "Which audio track of the added file is copied, for rips with "
                "more than one (commentary, audio description…). Its "
                "subtitles come along too."
            )
            stream_row.addWidget(self.multi_stream, 1)
            self.multi_stream.setEnabled(False)
            single_box.addLayout(stream_row)

            delay_row = QHBoxLayout()
            delay_row.setSpacing(8)
            delay_row.addWidget(QLabel("Delay of the selected track"))
            self.multi_delay = QDoubleSpinBox()
            self.multi_delay.setRange(-30.0, 30.0)
            self.multi_delay.setSingleStep(0.05)
            self.multi_delay.setDecimals(3)
            self.multi_delay.setValue(0.0)
            self.multi_delay.setSuffix(" s")
            self.multi_delay.setToolTip(
                "Positive delays the dub, negative advances it. Pick a track "
                "in the list to change its own delay."
            )
            delay_row.addWidget(self.multi_delay)
            self.btn_detect_delay = QPushButton("Detect from audio")
            self.btn_detect_delay.setToolTip(
                "Compares the energy envelope of both files and fills the delay "
                "with the measured offset. It is not a guess: if the audio does "
                "not match clearly, nothing is changed."
            )
            self.btn_detect_delay.clicked.connect(self._detect_track_delay)
            delay_row.addWidget(self.btn_detect_delay)
            self.btn_sync_editor = QPushButton("Sync editor…")
            self.btn_sync_editor.setToolTip(
                "Shows both waveforms on the same timeline, lets you listen "
                "to the original or the added track from the same point and "
                "move the dub while you check the sync."
            )
            self.btn_sync_editor.clicked.connect(self._open_sync_editor)
            delay_row.addWidget(self.btn_sync_editor)
            delay_row.addStretch(1)
            single_box.addLayout(delay_row)

            self.multi_tracks.currentRowChanged.connect(self._load_track_details)
            self.multi_stream.currentIndexChanged.connect(self._store_track_stream)
            self.multi_delay.valueChanged.connect(self._store_track_delay)
            self.multi_delay.setEnabled(False)
            self.btn_detect_delay.setEnabled(False)
            self.btn_sync_editor.setEnabled(False)

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
            single_box.addWidget(note)
            form.addRow(self.multi_single_box)

            self.multi_batch_box = QWidget()
            batch_box = QVBoxLayout(self.multi_batch_box)
            batch_box.setContentsMargins(0, 0, 0, 0)
            batch_box.setSpacing(8)

            batch_buttons = QHBoxLayout()
            batch_buttons.setSpacing(8)
            self.btn_batch_episodes = QPushButton("Add episodes…")
            self.btn_batch_episodes.clicked.connect(self._batch_add_files)
            self.btn_batch_folder = QPushButton("Add folder…")
            self.btn_batch_folder.clicked.connect(self._batch_add_folder)
            self.btn_batch_dubs = QPushButton("Dub folder…")
            self.btn_batch_dubs.clicked.connect(self._batch_choose_dubs)
            self.btn_batch_clear = QPushButton("Clear")
            self.btn_batch_clear.clicked.connect(self._batch_clear)
            for button in (
                self.btn_batch_episodes,
                self.btn_batch_folder,
                self.btn_batch_dubs,
                self.btn_batch_clear,
            ):
                batch_buttons.addWidget(button)
            batch_buttons.addStretch(1)
            batch_box.addLayout(batch_buttons)

            batch_delay_row = QHBoxLayout()
            batch_delay_row.setSpacing(8)
            batch_delay_row.addWidget(QLabel("Delay for every episode"))
            self.batch_delay = QDoubleSpinBox()
            self.batch_delay.setRange(-30.0, 30.0)
            self.batch_delay.setSingleStep(0.05)
            self.batch_delay.setDecimals(3)
            self.batch_delay.setValue(0.0)
            self.batch_delay.setSuffix(" s")
            self.batch_delay.setToolTip(
                "Positive delays the dub, negative advances it."
            )
            batch_delay_row.addWidget(self.batch_delay)
            self.batch_auto_delay = QCheckBox(
                "Detect the delay of each episode"
            )
            self.batch_auto_delay.setToolTip(
                "Compares the audio of each episode with its dub and uses the "
                "measured offset, which is the only thing that catches a dub "
                "with a longer intro. Episodes whose audio does not match "
                "clearly keep the delay above."
            )
            self.batch_auto_delay.toggled.connect(
                lambda checked: self.batch_delay.setEnabled(not checked)
            )
            batch_delay_row.addWidget(self.batch_auto_delay)
            batch_delay_row.addStretch(1)
            batch_box.addLayout(batch_delay_row)

            self.batch_hint = QLabel(
                "Dubs are looked up in each episode's own folder. Add a dub "
                "folder to search somewhere else."
            )
            self.batch_hint.setWordWrap(True)
            self.batch_hint.setObjectName("toolHint")
            batch_box.addWidget(self.batch_hint)

            self.batch_table = QTableWidget(0, 3)
            self.batch_table.setObjectName("batchTable")
            self.batch_table.setHorizontalHeaderLabels(
                ["Episode", "Dub", "Status"]
            )
            self.batch_table.verticalHeader().setVisible(False)
            self.batch_table.setEditTriggers(
                QAbstractItemView.EditTrigger.NoEditTriggers
            )
            self.batch_table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.batch_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
            self.batch_table.setMinimumHeight(150)
            batch_box.addWidget(self.batch_table)
            self.multi_batch_box.setVisible(False)
            form.addRow(self.multi_batch_box)

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
            QTableWidget#batchTable {
                background-color: #1f2c4c; color: #e0e0e0;
                border: 1px solid #2c3e50; border-radius: 4px;
                gridline-color: #2c3e50;
                alternate-background-color: #16213e;
            }
            QTableWidget#batchTable::item { padding: 3px; }
            QTableWidget#batchTable QHeaderView::section {
                background-color: #16213e; color: #8fa3b8;
                border: none; padding: 4px;
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
        self._update_run_label()

    def _update_run_label(self) -> None:
        # Etiqueta del botón: "Run batch" solo con el lote activo en audio.

        batch_box = getattr(self, "multi_batch", None)
        is_batch = (
            batch_box is not None
            and batch_box.isChecked()
            and self._current_tool_id() == "multiaudio"
        )
        if getattr(self, "btn_run", None) is not None:
            self.btn_run.setText("Run batch" if is_batch else "Run tool")

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
            track = {
                "path": path,
                "stream": 0,
                "streams": streams,
                "subtitles": get_subtitle_streams(self.ffmpeg_path, path),
                "language": code,
                "title": self.multi_language.currentText() if code else "",
                "delay": 0.0,
                "detail": self._stream_detail(streams[0]),
                "lang_label": label,
            }
            self._audio_tracks.append(track)
            self.multi_tracks.addItem(self._track_text(track))
            added += 1
        if skipped:
            QMessageBox.warning(
                self,
                "Add audio tracks",
                "These files have no audio track and were skipped:\n\n"
                + "\n".join(skipped),
            )
        if added:
            self.multi_tracks.setCurrentRow(len(self._audio_tracks) - 1)
            self.status.setText(f"{len(self._audio_tracks)} audio track(s) to add.")
        elif not skipped:
            self.status.setText("Those files are already in the list.")

    def _remove_audio_file(self) -> None:
        row = self.multi_tracks.currentRow()
        if row < 0 or row >= len(self._audio_tracks):
            return
        self.multi_tracks.takeItem(row)
        self._audio_tracks.pop(row)
        self._load_track_details(self.multi_tracks.currentRow())
        self.status.setText(f"{len(self._audio_tracks)} audio track(s) to add.")

    # ------------------------------------------------- track stream / delay
    def _stream_detail(self, stream: dict) -> str:
        stream = stream or {}
        bits = [str(stream.get("codec_name") or "audio")]
        channels = stream.get("channels")
        if channels:
            bits.append(f"{channels}ch")
        language = str(stream.get("language") or "")
        if language and language != "und":
            bits.append(language)
        return " ".join(bits)

    def _track_text(self, track: dict) -> str:
        parts = [os.path.basename(track["path"])]
        if track.get("detail"):
            parts.append(str(track["detail"]))
        if track.get("lang_label"):
            parts.append(str(track["lang_label"]))
        subtitles = track.get("subtitles") or []
        if subtitles:
            parts.append(
                "1 subtitle" if len(subtitles) == 1
                else f"{len(subtitles)} subtitles"
            )
        delay = float(track.get("delay") or 0.0)
        if abs(delay) > 1e-9:
            parts.append(f"{delay:+.2f} s")
        return "  ·  ".join(parts)

    def _load_track_details(self, row: int) -> None:
        valid = 0 <= row < len(self._audio_tracks)
        self._loading_delay = True
        try:
            self.multi_stream.blockSignals(True)
            self.multi_stream.clear()
            streams = []
            chosen = 0
            if valid:
                streams = self._audio_tracks[row].get("streams") or []
                chosen = int(self._audio_tracks[row].get("stream") or 0)
            for position, stream in enumerate(streams):
                index = int(stream.get("audio_index", position))
                text = f"{position + 1} · {self._stream_detail(stream)}"
                if stream.get("title"):
                    text += f" · {stream['title']}"
                self.multi_stream.addItem(text, index)
                if index == chosen:
                    self.multi_stream.setCurrentIndex(position)
            self.multi_stream.setCurrentIndex(
                max(0, self.multi_stream.currentIndex())
            )
            self.multi_stream.blockSignals(False)
            self.multi_delay.setValue(
                float(self._audio_tracks[row].get("delay") or 0.0)
                if valid else 0.0
            )
        finally:
            self._loading_delay = False
        self.multi_stream.setEnabled(valid and len(streams) > 1)
        self.multi_delay.setEnabled(valid)
        self.btn_detect_delay.setEnabled(valid)
        self.btn_sync_editor.setEnabled(valid and self._has_audio)

    def _store_track_stream(self, position: int) -> None:
        if self._loading_delay:
            return
        row = self.multi_tracks.currentRow()
        if row < 0 or row >= len(self._audio_tracks):
            return
        if position < 0:
            return
        stream_index = self.multi_stream.itemData(position)
        if stream_index is None:
            return
        track = self._audio_tracks[row]
        track["stream"] = int(stream_index)
        for stream in track.get("streams") or []:
            if int(stream.get("audio_index", -1)) == int(stream_index):
                track["detail"] = self._stream_detail(stream)
                break
        item = self.multi_tracks.item(row)
        if item is not None:
            item.setText(self._track_text(track))

    def _store_track_delay(self, value: float) -> None:
        if self._loading_delay:
            return
        row = self.multi_tracks.currentRow()
        if row < 0 or row >= len(self._audio_tracks):
            return
        self._audio_tracks[row]["delay"] = float(value)
        item = self.multi_tracks.item(row)
        if item is not None:
            item.setText(self._track_text(self._audio_tracks[row]))

    def _open_sync_editor(self) -> None:
        """Abre el editor de formas de onda de la pista seleccionada."""
        row = self.multi_tracks.currentRow()
        if row < 0 or row >= len(self._audio_tracks):
            QMessageBox.warning(
                self, "Add audio tracks",
                "Select the added track you want to sync first.",
            )
            return
        if not self._has_audio:
            QMessageBox.warning(
                self, "Add audio tracks",
                "This video has no audio track to compare with.",
            )
            return
        track = self._audio_tracks[row]
        dialog = AudioSyncDialog(
            self.ffmpeg_path,
            self.input_path,
            track["path"],
            dub_stream=int(track.get("stream") or 0),
            delay=float(track.get("delay") or 0.0),
            base_label="Original track of this video",
            dub_label=os.path.basename(track["path"]),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        track["delay"] = float(dialog.delay())
        item = self.multi_tracks.item(row)
        if item is not None:
            item.setText(self._track_text(track))
        if self.multi_tracks.currentRow() == row:
            self._load_track_details(row)
        self.status.setText(f"Delay set to {track['delay']:+.3f} s.")

    def _detect_track_delay(self) -> None:
        row = self.multi_tracks.currentRow()
        if row < 0 or row >= len(self._audio_tracks):
            QMessageBox.warning(
                self, "Add audio tracks",
                "Select the added track whose delay you want to measure.",
            )
            return
        if self._detect_worker is not None:
            return
        track = self._audio_tracks[row]
        self.btn_detect_delay.setEnabled(False)
        self.multi_delay.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.status.setText(
            f"Comparing the audio with {os.path.basename(track['path'])}…"
        )
        self._detect_worker = DelayDetectWorker(
            self.ffmpeg_path,
            self.input_path,
            track["path"],
            self,
            stream_index=int(track.get("stream") or 0),
        )
        self._detect_worker.done.connect(
            lambda offset, confidence, row=row: self._on_delay_detected(
                row, offset, confidence
            )
        )
        self._detect_worker.failed.connect(self._on_delay_failed)
        self._detect_worker.finished.connect(self._detect_worker.deleteLater)
        self._detect_worker.start()

    def _on_delay_detected(self, row: int, offset: float,
                           confidence: float) -> None:
        self._detect_worker = None
        self.btn_abort.setEnabled(False)
        self._load_track_details(self.multi_tracks.currentRow())
        if 0 <= row < len(self._audio_tracks):
            self._audio_tracks[row]["delay"] = float(offset)
            item = self.multi_tracks.item(row)
            if item is not None:
                item.setText(self._track_text(self._audio_tracks[row]))
            if self.multi_tracks.currentRow() == row:
                self._load_track_details(row)
        self.status.setText(
            f"Measured delay {offset:+.2f} s (confidence {confidence:.2f})."
        )

    def _on_delay_failed(self, message: str) -> None:
        self._detect_worker = None
        self.btn_abort.setEnabled(False)
        self._load_track_details(self.multi_tracks.currentRow())
        self.status.setText(message.replace("\n", " "))

    # ------------------------------------------------------------ batch mode
    def _toggle_batch(self, enabled: bool) -> None:
        self.multi_single_box.setVisible(not enabled)
        self.multi_batch_box.setVisible(enabled)
        self._update_run_label()
        if enabled and not self._batch_episodes and os.path.isfile(
            self.input_path or ""
        ):
            self._batch_episodes.append(self.input_path)
            self._refresh_batch()
        self.progress.setValue(0)
        self.status.setText("Ready")

    def _batch_start_dir(self) -> str:
        if self._batch_dub_dir:
            return self._batch_dub_dir
        if self._batch_episodes:
            return os.path.dirname(self._batch_episodes[0])
        return self.output_edit.text() or self.output_dir

    def _batch_add_files(self) -> None:
        filter_text = "Video files (" + " ".join(
            "*" + extension for extension in VIDEO_EXTENSIONS
        ) + ");;All files (*)"
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add episodes", self._batch_start_dir(), filter_text
        )
        if paths:
            self._extend_batch_episodes(paths)

    def _batch_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Add every episode in a folder", self._batch_start_dir()
        )
        if not folder:
            return
        files = media_files(folder, VIDEO_EXTENSIONS)
        if not files:
            QMessageBox.warning(
                self, "Add audio tracks", "That folder has no video files."
            )
            return
        self._extend_batch_episodes(files)

    def _batch_choose_dubs(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Folder with the dubbed episodes",
            self._batch_start_dir(),
        )
        if folder:
            self._batch_dub_dir = folder
            self._refresh_batch()

    def _batch_clear(self) -> None:
        self._batch_episodes = []
        self._refresh_batch()

    def _extend_batch_episodes(self, paths) -> None:
        seen = {
            os.path.normcase(os.path.abspath(path))
            for path in self._batch_episodes
        }
        for path in paths:
            key = os.path.normcase(os.path.abspath(path))
            if key in seen or not os.path.isfile(path):
                continue
            self._batch_episodes.append(path)
            seen.add(key)
        self._refresh_batch()

    def _set_batch_cell(self, row: int, column: int, text: str) -> None:
        item = self.batch_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.batch_table.setItem(row, column, item)
        item.setText(text)

    def _refresh_batch(self) -> None:
        # Una sola lectura de cada carpeta aunque haya veinte episodios en ella.
        cache = {}
        rows = []
        for episode in self._batch_episodes:
            folder = self._batch_dub_dir or os.path.dirname(episode)
            if folder not in cache:
                cache[folder] = media_files(folder)
            rows.append({"episode": episode, **match_dub(episode, cache[folder])})
        self._batch_rows = rows
        self.batch_table.setRowCount(len(rows))
        ready = 0
        for row, item in enumerate(rows):
            ready += 1 if item["dub"] else 0
            self._set_batch_cell(row, 0, os.path.basename(item["episode"]))
            self._set_batch_cell(
                row,
                1,
                os.path.basename(item["dub"]) if item["dub"] else "—",
            )
            self._set_batch_cell(row, 2, item["status"])
        where = self._batch_dub_dir or "each episode's own folder"
        self.batch_hint.setText(
            f"Dubs are looked up in {where}: {ready} of {len(rows)} episode(s) "
            f"matched. Pairing uses S01E03, or the name when there is no "
            f"episode number; ambiguous cases are left out. Each dub contributes "
            f"its first audio track and the subtitles it carries."
        )
        self.status.setText(
            f"{ready} episode(s) ready to merge." if rows
            else "Add episodes to start."
        )

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
            if not self._audio_tracks and not self.multi_batch.isChecked():
                raise ValueError("Add at least one audio file first.")
            return {
                **self._multi_base_options(),
                "tracks": [dict(track) for track in self._audio_tracks],
            }
        return {}

    def _multi_base_options(self) -> dict:
        """Ajustes compartidos por todos los episodios del lote."""
        normalize = bool(self.multi_normalize.currentData())
        return {
            "container": self.multi_container.currentData() or "mkv",
            "normalize": normalize,
            "normalize_original": (
                normalize and self.multi_normalize_original.isChecked()
            ),
            "auto_delay": bool(
                self.multi_batch.isChecked()
                and self.batch_auto_delay.isChecked()
            ),
        }

    def _multi_track_meta(self) -> dict:
        code = self.multi_language.currentData() or ""
        return {
            # En el lote no hay selector por episodio: se usa la primera pista
            # de audio de cada doblaje y se traen sus subtítulos.
            "stream": 0,
            "language": code,
            "title": self.multi_language.currentText() if code else "",
            "delay": (
                0.0 if self.batch_auto_delay.isChecked()
                else float(self.batch_delay.value())
            ),
        }

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
            "multi_batch",
            "btn_batch_episodes",
            "btn_batch_folder",
            "btn_batch_dubs",
            "btn_batch_clear",
            "batch_table",
            "batch_delay",
            "batch_auto_delay",
            "multi_delay",
            "multi_stream",
            "btn_detect_delay",
            "btn_sync_editor",
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
        if not running:
            # El desfase vuelve a depender de la pista seleccionada y de si el
            # lote lo detecta solo, no de que el render haya terminado.
            if getattr(self, "multi_delay", None) is not None:
                self._load_track_details(self.multi_tracks.currentRow())
            if getattr(self, "batch_delay", None) is not None:
                self.batch_delay.setEnabled(
                    not self.batch_auto_delay.isChecked()
                )

    def _run(self) -> None:
        if self.worker is not None:
            return
        if not self.output_edit.text().strip():
            QMessageBox.warning(self, "Video tools", "Choose an output folder.")
            return
        if self._current_tool_id() == "multiaudio" and self.multi_batch.isChecked():
            self._run_batch()
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

    def _run_batch(self) -> None:
        rows = [row for row in self._batch_rows if row["dub"]]
        if not rows:
            QMessageBox.warning(
                self,
                "Add audio tracks",
                "No episode has a matching dub yet. Add the episodes and, if "
                "they live somewhere else, the folder with the dubs.",
            )
            return
        # Las filas que se procesan son un subconjunto de la tabla: se guarda a
        # qué fila real corresponde cada trabajo para ir pintando el estado.
        self._batch_table_rows = [self._batch_rows.index(row) for row in rows]
        self.progress.setValue(0)
        self.status.setText("Running…")
        self._set_running(True)
        self.worker = BatchAudioWorker(
            self.ffmpeg_path,
            rows,
            self.output_edit.text().strip(),
            self._multi_base_options(),
            self._multi_track_meta(),
            self,
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.stage.connect(self.status.setText)
        self.worker.row_status.connect(self._on_batch_row)
        self.worker.done.connect(self._on_batch_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_batch_row(self, position: int, text: str) -> None:
        if 0 <= position < len(self._batch_table_rows):
            self._set_batch_cell(self._batch_table_rows[position], 2, text)

    def _on_batch_done(self, done_count: int, failed_count: int) -> None:
        self._set_running(False)
        self.worker = None
        self.progress.setValue(100)
        self.status.setText(f"{done_count} done, {failed_count} failed.")
        QMessageBox.information(
            self,
            "Add audio tracks",
            f"{done_count} episode(s) finished, {failed_count} failed.\n\n"
            f"Output folder:\n{self.output_edit.text().strip()}",
        )

    def _abort(self) -> None:
        if self._detect_worker is not None:
            self.status.setText("Stopping…")
            self._detect_worker.cancel()
            self.btn_abort.setEnabled(False)
            return
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
        if self._detect_worker is not None:
            event.ignore()
            self.status.setText("Wait for the delay detection to finish.")
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
