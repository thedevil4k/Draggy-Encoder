"""Panel desplegable de Audio Boost avanzado (EQ, compresor, limitador).

Diseño de la interacción:

* La fila de cabecera tiene la flecha de despliegue, al lado de la casilla de
  Audio Boost. Cerrado no se ve nada más y el boost se comporta como siempre.
* Dentro, cada bloque (EQ / Compresor / Limitador) tiene su propia casilla. Si
  no se activa ninguna, el boost sigue siendo el de siempre: abrir el
  desplegable no cambia el sonido.
* Los controles emiten ``changed`` con la configuración lista para
  ``src.ffmpeg.audio_fx``, y ``preview_requested`` cuando el usuario quiere
  escuchar el resultado mientras ajusta.

La lógica de valores (rangos, acotado, valores por defecto) vive en
``audio_fx``; este widget solo presenta controles y no decide nada por su
cuenta, para que la validación sea una sola.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.ffmpeg import audio_fx


# Presets de compresión. Los umbrales son LU por debajo del nivel del material
# (audio_fx los convierte a dB tras el boost), así que funcionan con cualquier
# fuente sin tener que calibrarlos a mano.
COMPRESSOR_PRESETS = (
    ("Suave (mantiene la dinámica)", {"threshold_lu": 12.0, "ratio": 2.0,
                                      "attack_ms": 30.0, "release_ms": 400.0}),
    ("Medio (recomendado)", {"threshold_lu": 8.0, "ratio": 3.0,
                             "attack_ms": 20.0, "release_ms": 250.0}),
    ("Fuerte (sonido denso)", {"threshold_lu": 6.0, "ratio": 6.0,
                              "attack_ms": 10.0, "release_ms": 180.0}),
    ("Aplastado (voz/radio)", {"threshold_lu": 5.0, "ratio": 12.0,
                               "attack_ms": 5.0, "release_ms": 120.0}),
)


class _SliderRow(QWidget):
    """Fila etiqueta + deslizador + valor numérico (patrón de un mando)."""

    value_changed = Signal(float)

    def __init__(self, label, minimum, maximum, value, suffix="", decimals=1,
                 step=0.5, parent=None):
        super().__init__(parent)
        self._scale = 10 ** decimals
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label = QLabel(label)
        self.label.setMinimumWidth(96)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(minimum * self._scale))
        self.slider.setMaximum(int(maximum * self._scale))
        self.slider.setSingleStep(int(step * self._scale))
        self.slider.setPageStep(int((maximum - minimum) * self._scale / 10) or 1)
        self.slider.setValue(int(value * self._scale))
        layout.addWidget(self.slider, 1)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(value)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(84)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, raw):
        value = raw / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.value_changed.emit(value)

    def _on_spin(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self._scale))
        self.slider.blockSignals(False)
        self.value_changed.emit(float(value))

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float) -> None:
        """Fija el valor SIN emitir (lo usan presets y la restauración).

        No emite a propósito: quien lo llama (``_on_preset``, ``set_effects``)
        ya decide si debe notificar. Emitir aquí encadenaría señales durante la
        carga de la configuración guardada y guardaría estados intermedios.
        """
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(float(value))
        self.slider.setValue(int(float(value) * self._scale))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def set_enabled(self, enabled: bool) -> None:
        self.slider.setEnabled(enabled)
        self.spin.setEnabled(enabled)
        self.label.setEnabled(enabled)


class AudioFxPanel(QWidget):
    """Bloque de efectos avanzados del Audio Boost."""

    changed = Signal()
    preview_requested = Signal()
    preview_stop_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._previewing = False
        self._build()

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 4, 4, 6)
        root.setSpacing(6)

        hint = QLabel(
            "Estos ajustes cambian el propio Audio Boost: la normalización se\n"
            "recalcula para dejar trabajar al ecualizador y al compresor."
        )
        hint.setStyleSheet("color: #7f8c9b; font-size: 10px;")
        root.addWidget(hint)

        # --- Ecualizador -------------------------------------------------
        self.eq_check = QCheckBox("Ecualizador")
        root.addWidget(self.eq_check)
        self.eq_bass = _SliderRow("Graves", -12, 12, 0.0, " dB")
        self.eq_mid = _SliderRow("Medios", -12, 12, 0.0, " dB")
        self.eq_treble = _SliderRow("Agudos", -12, 12, 0.0, " dB")
        for row in (self.eq_bass, self.eq_mid, self.eq_treble):
            root.addWidget(row)
        root.addWidget(self._separator())

        # --- Compresor ---------------------------------------------------
        comp_header = QHBoxLayout()
        self.comp_check = QCheckBox("Compresor")
        comp_header.addWidget(self.comp_check)
        comp_header.addStretch(1)
        comp_header.addWidget(QLabel("Preajuste"))
        self.preset_combo = QComboBox()
        for name, _values in COMPRESSOR_PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentIndex(1)
        self.preset_combo.setToolTip(
            "Ajustes de partida. Puedes afinar cada control después."
        )
        comp_header.addWidget(self.preset_combo)
        root.addLayout(comp_header)

        self.comp_threshold = _SliderRow("Umbral", 0.0, 24.0, 8.0, " LU")
        self.comp_ratio = _SliderRow("Proporción", 1.0, 20.0, 3.0, ":1")
        self.comp_attack = _SliderRow("Ataque", 0.1, 200.0, 20.0, " ms", decimals=1, step=5.0)
        self.comp_release = _SliderRow("Liberación", 10.0, 2000.0, 250.0, " ms", decimals=0, step=20.0)
        for row in (self.comp_threshold, self.comp_ratio,
                    self.comp_attack, self.comp_release):
            root.addWidget(row)
        root.addWidget(self._separator())

        # --- Limitador ---------------------------------------------------
        lim_header = QHBoxLayout()
        self.lim_check = QCheckBox("Limitador de pico")
        lim_header.addWidget(self.lim_check)
        lim_header.addStretch(1)
        self.lim_ceiling = _SliderRow("Techo", -6.0, 0.0, -1.5, " dBTP")
        lim_header.addWidget(self.lim_ceiling, 1)
        root.addLayout(lim_header)

        lim_note = QLabel(
            "El limitador recorta los picos que superan el techo. El ajuste "
            "real lleva un margen extra porque el codificador añade picos entre "
            "muestras."
        )
        lim_note.setWordWrap(True)
        lim_note.setStyleSheet("color: #7f8c9b; font-size: 10px;")
        root.addWidget(lim_note)

        # --- Acciones ----------------------------------------------------
        actions = QHBoxLayout()
        self.preview_button = QPushButton("Escuchar efecto")
        self.preview_button.setToolTip(
            "Reproduce un fragmento del audio con estos ajustes aplicados."
        )
        self.stop_button = QPushButton("Parar")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Detiene la escucha y la generación en curso.")
        self.reset_button = QPushButton("Restablecer")
        actions.addWidget(self.preview_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.reset_button)
        actions.addStretch(1)
        root.addLayout(actions)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #00cccc; font-size: 10px;")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        # --- Conexiones --------------------------------------------------
        self.eq_check.toggled.connect(self._on_any_toggle)
        self.comp_check.toggled.connect(self._on_any_toggle)
        self.lim_check.toggled.connect(self._on_any_toggle)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        for row in (self.eq_bass, self.eq_mid, self.eq_treble,
                    self.comp_threshold, self.comp_ratio,
                    self.comp_attack, self.comp_release, self.lim_ceiling):
            row.value_changed.connect(self._on_value_changed)
        self.preview_button.clicked.connect(self.preview_requested.emit)
        self.stop_button.clicked.connect(self._on_stop)
        self.reset_button.clicked.connect(self._on_reset)

        # Anti-rebote del control que se está arrastrando: sin esto, mover un
        # deslizador lanzaría una generación de audio por cada píxel.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(500)
        self._preview_timer.timeout.connect(self._request_preview_if_live)

        self._sync_enabled()
        self._update_summary()

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #2a2f36;")
        return line

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    def _on_any_toggle(self, _checked=False):
        self._sync_enabled()
        self._emit_changed()

    def _on_preset(self, index):
        if self._loading or not (0 <= index < len(COMPRESSOR_PRESETS)):
            return
        _name, values = COMPRESSOR_PRESETS[index]
        self.comp_threshold.set_value(values["threshold_lu"])
        self.comp_ratio.set_value(values["ratio"])
        self.comp_attack.set_value(values["attack_ms"])
        self.comp_release.set_value(values["release_ms"])
        self._emit_changed()

    def _on_value_changed(self, _value=0.0):
        self._emit_changed()

    def _on_reset(self):
        # emit=True: restablecer es una acción del usuario, así que el cambio
        # debe persistirse (sin esto la configuración vieja seguía guardada).
        self.set_effects(audio_fx.default_effects(), emit=True)
        self.reset_requested.emit()

    def _sync_enabled(self):
        """Habilita los controles de cada bloque según su casilla."""
        for row in (self.eq_bass, self.eq_mid, self.eq_treble):
            row.set_enabled(self.eq_check.isChecked())
        for row in (self.comp_threshold, self.comp_ratio,
                    self.comp_attack, self.comp_release):
            row.set_enabled(self.comp_check.isChecked())
        self.preset_combo.setEnabled(self.comp_check.isChecked())
        self.lim_ceiling.set_enabled(self.lim_check.isChecked())

    def _emit_changed(self):
        if self._loading:
            return
        self._update_summary()
        self.changed.emit()
        # Si el usuario ya estaba escuchando, se le devuelve el audio con el
        # ajuste nuevo tras el anti-rebote: es el bucle "muevo y oigo" de un
        # plugin. Si no estaba escuchando, no se reproduce nada por sorpresa.
        if self._previewing:
            self._preview_timer.start()

    def _request_preview_if_live(self):
        if self._previewing:
            self.preview_requested.emit()

    def _on_stop(self):
        """Detiene la escucha y sale del modo "en vivo"."""
        self._previewing = False
        self._preview_timer.stop()
        self.preview_stop_requested.emit()
        self.set_playback_state(False)
        self.set_status("Escucha detenida.")

    def set_playback_state(self, playing: bool) -> None:
        """Refleja en la UI si hay audio sonando (lo llama la ventana)."""
        self._previewing = bool(playing)
        self.stop_button.setEnabled(bool(playing))
        self.preview_button.setText("Escuchar de nuevo" if playing else "Escuchar efecto")

    def _update_summary(self):
        cfg = self.get_effects()
        if not audio_fx.effects_are_active(cfg):
            self.status_label.setText(
                "Sin efectos: el boost se aplica como siempre."
            )
            return
        self.status_label.setText(
            f"Se aplicará: {audio_fx.describe_effects(cfg)}"
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def get_effects(self) -> dict:
        """Configuración actual, lista para ``src.ffmpeg.audio_fx``."""
        return audio_fx.sanitize_effects({
            "enabled": (
                self.eq_check.isChecked()
                or self.comp_check.isChecked()
                or self.lim_check.isChecked()
            ),
            "eq": {
                "enabled": self.eq_check.isChecked(),
                "bass": self.eq_bass.value(),
                "mid": self.eq_mid.value(),
                "treble": self.eq_treble.value(),
            },
            "compressor": {
                "enabled": self.comp_check.isChecked(),
                "threshold_lu": self.comp_threshold.value(),
                "ratio": self.comp_ratio.value(),
                "attack_ms": self.comp_attack.value(),
                "release_ms": self.comp_release.value(),
            },
            "limiter": {
                "enabled": self.lim_check.isChecked(),
                "ceiling_dbtp": self.lim_ceiling.value(),
            },
        })

    def set_effects(self, effects, emit: bool = False) -> None:
        """Carga una configuración.

        ``emit=False`` (por defecto) no notifica: es lo que necesita la
        restauración al abrir la ventana, para no guardar estados intermedios.
        ``emit=True`` lo usa "Restablecer", que sí debe persistir el cambio.
        """
        cfg = audio_fx.sanitize_effects(effects)
        self._loading = True
        try:
            self.eq_check.setChecked(bool(cfg["eq"]["enabled"]))
            self.eq_bass.set_value(cfg["eq"]["bass"])
            self.eq_mid.set_value(cfg["eq"]["mid"])
            self.eq_treble.set_value(cfg["eq"]["treble"])

            self.comp_check.setChecked(bool(cfg["compressor"]["enabled"]))
            self.comp_threshold.set_value(cfg["compressor"]["threshold_lu"])
            self.comp_ratio.set_value(cfg["compressor"]["ratio"])
            self.comp_attack.set_value(cfg["compressor"]["attack_ms"])
            self.comp_release.set_value(cfg["compressor"]["release_ms"])

            self.lim_check.setChecked(bool(cfg["limiter"]["enabled"]))
            self.lim_ceiling.set_value(cfg["limiter"]["ceiling_dbtp"])
        finally:
            self._loading = False
        self._sync_enabled()
        self._update_summary()
        if emit:
            self.changed.emit()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text or "")
