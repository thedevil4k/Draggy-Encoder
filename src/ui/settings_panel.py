from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.config import AUDIO_OPTIONS, FORMATS, RESOLUTIONS
from src.hardware.codecs import get_available_devices_with_codecs


class SettingsPanel(QWidget):
    def __init__(self, hardware_info=None, compatible_codecs=None, parent=None):
        super().__init__(parent)
        self.hardware_info = hardware_info or {}
        self.compatible_codecs = compatible_codecs or {}
        self.available_devices = self._build_available_devices()
        self._setup_ui()

    def refresh(self, hardware_info, compatible_codecs):
        """Reconstruye el panel con hardware/compatibilidad actualizados."""
        self.hardware_info = hardware_info or {}
        self.compatible_codecs = compatible_codecs or {}
        self.available_devices = self._build_available_devices()

        # Repoblar combo de Device
        current_device = self.device_combo.currentText()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for device_name in self.available_devices.keys():
            self.device_combo.addItem(device_name)
        # Intentar mantener el device seleccionado si sigue existiendo
        idx = self.device_combo.findText(current_device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.device_combo.blockSignals(False)
        self._on_device_changed()


    def _build_available_devices(self):
        """Construye el mapa de dispositivos/codecs filtrado por compatibilidad con FFmpeg."""
        devices = get_available_devices_with_codecs(self.hardware_info)
        if not self.compatible_codecs:
            return devices

        filtered = {}
        # Mapear nombres de device a familia para buscar en compatible_codecs
        family_map = {
            "CPU": "CPU",
            "NVIDIA GPU": "NVIDIA GPU",
            "AMD GPU": "AMD GPU",
            "Intel GPU": "Intel GPU",
        }
        for device_name, codecs in devices.items():
            family = None
            for key in family_map:
                if device_name.startswith(key):
                    family = family_map[key]
                    break
            if not family:
                filtered[device_name] = codecs
                continue

            allowed = set()
            for family_name in [family, device_name]:
                for codec_id, codec_name in self.compatible_codecs.get(family_name, []):
                    allowed.add(codec_id)

            filtered_codecs = [(cid, cname) for cid, cname in codecs if cid in allowed]
            if filtered_codecs:
                filtered[device_name] = filtered_codecs
        return filtered

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        group = QGroupBox("Compression Settings")
        group.setStyleSheet(
            "QGroupBox { color: #00d2d3; font-weight: bold; border: 1px solid #2c3e50; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        grid = QGridLayout(group)
        grid.setColumnStretch(1, 1)
        grid.setSpacing(8)

        # Mode
        self.mode_label = QLabel("Mode")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Target Bitrate (kbps)", "Target File Size (MB)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        grid.addWidget(self.mode_label, 0, 0)
        grid.addWidget(self.mode_combo, 0, 1)

        # Bitrate / Target size
        self.value_label = QLabel("Bitrate (kbps)")
        self.value_spin = QSpinBox()
        self.value_spin.setRange(100, 100000)
        self.value_spin.setValue(1500)
        self.value_slider = QSlider(Qt.Orientation.Horizontal)
        self.value_slider.setRange(100, 50000)
        self.value_slider.setValue(1500)
        self.value_slider.valueChanged.connect(self.value_spin.setValue)
        self.value_spin.valueChanged.connect(self.value_slider.setValue)

        value_layout = QHBoxLayout()
        value_layout.addWidget(self.value_spin)
        value_layout.addWidget(self.value_slider)
        grid.addWidget(self.value_label, 1, 0)
        grid.addLayout(value_layout, 1, 1)

        self.original_bitrate_label = QLabel("Original video bitrate: -")
        self.original_bitrate_label.setStyleSheet("color: #5f6f7f; font-size: 10px;")
        grid.addWidget(self.original_bitrate_label, 2, 1)

        self.ai_bitrate_label = QLabel("")
        self.ai_bitrate_label.setStyleSheet("color: #00cccc; font-size: 10px;")
        grid.addWidget(self.ai_bitrate_label, 3, 1)

        # Output Name
        self.output_name_label = QLabel("Output Name")
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Leave empty for original name")
        grid.addWidget(self.output_name_label, 4, 0)
        grid.addWidget(self.output_name_edit, 4, 1)

        # Resolution
        self.resolution_label = QLabel("Resolution")
        self.resolution_combo = QComboBox()
        for key, text in RESOLUTIONS:
            self.resolution_combo.addItem(text, key)
        grid.addWidget(self.resolution_label, 5, 0)
        grid.addWidget(self.resolution_combo, 5, 1)

        # Device
        self.device_label = QLabel("Device")
        self.device_combo = QComboBox()
        for device_name in self.available_devices.keys():
            self.device_combo.addItem(device_name)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        grid.addWidget(self.device_label, 6, 0)
        grid.addWidget(self.device_combo, 6, 1)

        # Codec
        self.codec_label = QLabel("Codec")
        self.codec_combo = QComboBox()
        grid.addWidget(self.codec_label, 7, 0)
        grid.addWidget(self.codec_combo, 7, 1)

        # Export format
        self.format_label = QLabel("Export")
        self.format_combo = QComboBox()
        for key, text in FORMATS:
            self.format_combo.addItem(text, key)
        grid.addWidget(self.format_label, 8, 0)
        grid.addWidget(self.format_combo, 8, 1)

        # Audio
        self.audio_label = QLabel("Audio")
        self.audio_combo = QComboBox()
        for key, text in AUDIO_OPTIONS:
            self.audio_combo.addItem(text, key)
        self.audio_combo.currentIndexChanged.connect(self._on_audio_changed)
        grid.addWidget(self.audio_label, 9, 0)
        grid.addWidget(self.audio_combo, 9, 1)

        # Audio boost
        self.audio_boost_check = QCheckBox("Audio Boost (normalize volume)")
        self.audio_boost_check.setChecked(False)
        grid.addWidget(self.audio_boost_check, 10, 1)

        # Trim
        self.trim_check = QCheckBox("Trim (only export a portion of the video)")
        self.trim_check.toggled.connect(self._on_trim_toggled)
        grid.addWidget(self.trim_check, 11, 1)

        self.start_edit = QLineEdit("00:00:00")
        self.end_edit = QLineEdit("00:00:00")
        self.start_edit.setEnabled(False)
        self.end_edit.setEnabled(False)
        trim_layout = QHBoxLayout()
        trim_layout.addWidget(QLabel("Start"))
        trim_layout.addWidget(self.start_edit)
        trim_layout.addWidget(QLabel("End"))
        trim_layout.addWidget(self.end_edit)
        grid.addLayout(trim_layout, 12, 1)

        layout.addWidget(group)
        layout.addStretch()

        self._on_device_changed()
        self._on_mode_changed()

    def _on_mode_changed(self):
        self.value_spin.valueChanged.disconnect(self.value_slider.setValue)
        self.value_slider.valueChanged.disconnect(self.value_spin.setValue)
        mode = self.mode_combo.currentText()
        if "Bitrate" in mode:
            self.value_label.setText("Bitrate (kbps)")
            self.value_spin.setRange(100, 100000)
            self.value_spin.setValue(1500)
            self.value_slider.setRange(100, 50000)
            self.value_slider.setValue(1500)
        else:
            self.value_label.setText("Target Size (MB)")
            self.value_spin.setRange(1, 100000)
            self.value_spin.setValue(500)
            self.value_slider.setRange(1, 10000)
            self.value_slider.setValue(500)
        self.value_spin.valueChanged.connect(self.value_slider.setValue)
        self.value_slider.valueChanged.connect(self.value_spin.setValue)

    def _on_device_changed(self):
        device = self.device_combo.currentText()
        self.codec_combo.clear()
        for codec_id, codec_name in self.available_devices.get(device, []):
            self.codec_combo.addItem(codec_name, codec_id)

    def _on_audio_changed(self):
        # Audio Boost can be applied whenever we are going to re-encode audio.
        # It is disabled only when audio is completely disabled (-an).
        audio = self.audio_combo.currentData()
        self.audio_boost_check.setEnabled(audio != "noaudio")

    def _on_trim_toggled(self, checked):
        self.start_edit.setEnabled(checked)
        self.end_edit.setEnabled(checked)

    def set_original_bitrate(self, bitrate_text):
        self.original_bitrate_label.setText(f"Original video bitrate: {bitrate_text}")

    def set_ai_bitrate(self, text):
        self.ai_bitrate_label.setText(text)

    def get_settings(self):
        return {
            "mode": self.mode_combo.currentText(),
            "bitrate": self.value_spin.value(),
            "target_size_mb": self.value_spin.value(),
            "output_name": self.output_name_edit.text().strip(),
            "resolution": self.resolution_combo.currentData(),
            "device": self.device_combo.currentText(),
            "codec": self.codec_combo.currentData(),
            "format": self.format_combo.currentData(),
            "audio": self.audio_combo.currentData(),
            "audio_boost": self.audio_boost_check.isChecked(),
            "trim": self.trim_check.isChecked(),
            "start_time": self.start_edit.text() if self.trim_check.isChecked() else None,
            "end_time": self.end_edit.text() if self.trim_check.isChecked() else None,
        }

    def set_device(self, device_name):
        # Buscar un device que contenga el nombre recomendado (CPU, NVIDIA, AMD, Intel)
        for i in range(self.device_combo.count()):
            text = self.device_combo.itemText(i)
            if device_name in text:
                self.device_combo.setCurrentIndex(i)
                return
        # Fallback al primero
        if self.device_combo.count() > 0:
            self.device_combo.setCurrentIndex(0)

    def set_codec(self, codec_id):
        """Selecciona el codec por ID (data del QComboBox). Idempotente.

        Llamado por ``MainWindow`` cuando hay una preferencia ``last_codec``
        vigente y queremos restaurar el codec seleccionado en sesiones
        previas. Si el codec ya no existe en la familia del device actual
        (p. ej. cambio de GPU con drivers actualizados), no hace nada y
        deja el primer codec disponible de la familia.
        """
        if not codec_id:
            return
        for i in range(self.codec_combo.count()):
            if self.codec_combo.itemData(i) == codec_id:
                self.codec_combo.setCurrentIndex(i)
                return
        # No encontrado: no tocamos el indice. El codec actual (que
        # ``_on_device_changed`` dejo seleccionado) sigue siendo valido
        # y no es invasivo sobrescribir con el primero.
        return
