"""Ventana de información de hardware, con la redetección y los códecs dentro.

Antes la redetección era un botón de la barra superior y el resultado había que
ir a verlo a este diálogo. Ahora el botón vive **aquí**, junto a los datos que
actualiza: al pulsarlo se vuelve a detectar hardware y códecs, y esta misma
ventana se refresca en el sitio (sin cerrarse) mostrando la lista de códecs
disponibles por dispositivo (CPU y cada GPU).
"""

from __future__ import annotations

import html
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.dialog_utils import fit_to_screen, wrap_scrollable
from src.utils.helpers import is_windows_elevated


class HardwareDialog(QDialog):
    """Datos de hardware + códecs disponibles, con botón de redetección.

    ``compatible_codecs`` es el mapa ``{dispositivo: [(codec_id, nombre), ...]}``
    que produce la detección (``CPU``, ``NVIDIA GPU (...)``, ``AMD GPU (...)``,
    ``Intel GPU (...)``). ``redetect_cb`` es la función que lanza la
    redetección (la del programa, en segundo plano): este diálogo no la
    implementa, sólo la pide y se refresca cuando llega el resultado vía
    ``refresh()``.
    """

    def __init__(self, hardware_info, compatible_codecs=None, parent=None,
                 redetect_cb=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Info")
        self.setMinimumSize(520, 360)
        self._hardware_info = hardware_info or {}
        self._compatible_codecs = compatible_codecs or {}
        self._redetect_cb = redetect_cb
        self._busy = False
        self._build_ui()
        self._populate()
        fit_to_screen(self, preferred_width=620, preferred_height=840,
                      min_width=520, min_height=360)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(10)
        self.area = wrap_scrollable(self.content)
        outer.addWidget(self.area, stretch=1)

        bottom = QHBoxLayout()
        self.btn_redetect = QPushButton("Re-detect Hardware && Codecs")
        self.btn_redetect.setToolTip(
            "Volver a detectar CPU, GPU y los códecs que acepta FFmpeg en este "
            "equipo.\nActualiza esta ventana y la lista de códecs del panel de "
            "ajustes."
        )
        self.btn_redetect.clicked.connect(self._on_redetect)
        self.btn_redetect.setEnabled(self._redetect_cb is not None)
        bottom.addWidget(self.btn_redetect)

        self.lbl_busy = QLabel("")
        self.lbl_busy.setStyleSheet("color: #a0a0b0; font-size: 10px;")
        bottom.addWidget(self.lbl_busy)
        bottom.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        bottom.addWidget(self.btn_close)
        outer.addLayout(bottom)

    def _add(self, widget):
        self.content_layout.addWidget(widget)

    def _label(self, text, word_wrap=True):
        label = QLabel(text)
        label.setWordWrap(word_wrap)
        return label

    def _clear_content(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    # ------------------------------------------------------------ contenido
    def _populate(self):
        self._clear_content()
        info = self._hardware_info

        title = QLabel("Hardware Information")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00d2d3; font-size: 16px; font-weight: bold;")
        self._add(title)

        # --- OS ---
        os_text = info.get("os") or "Unknown OS"
        machine = info.get("machine") or ""
        if machine:
            os_text += f" · {machine}"
        self._add(self._label(f"<b>OS:</b> {html.escape(str(os_text))}"))

        # --- Python interpreter ---
        python = info.get("python", {}) or {}
        py_version = python.get("version") or (sys.version.split()[0] if sys.version else "?")
        py_bits = python.get("bits")
        py_text = f"<b>Python interpreter:</b> {html.escape(str(py_version))}"
        if py_bits:
            py_text += f" ({py_bits}-bit)"
        self._add(self._label(py_text))

        py_exe = python.get("executable") or sys.executable
        if py_exe:
            # Campo de sólo lectura en vez de etiqueta: una ruta larga sin
            # espacios donde partir (C:\Users\...\python.exe) fijaba ella sola
            # el ancho mínimo del diálogo (~680 px) y sacaba barra horizontal.
            # Así se puede seleccionar y copiar, y no impone tamaño.
            exe_edit = QLineEdit(str(py_exe))
            exe_edit.setReadOnly(True)
            exe_edit.setFrame(False)
            exe_edit.setCursorPosition(0)
            exe_edit.setToolTip(str(py_exe))
            exe_edit.setStyleSheet(
                "QLineEdit { color: #a0a0b0; font-size: 10px;"
                " background: transparent; border: none; }"
            )
            self._add(exe_edit)

        # --- Elevación / UIPI (solo relevante en Windows) ---
        if sys.platform == "win32":
            if is_windows_elevated():
                elev_text = (
                    "<b>Elevation:</b> Administrator — drag &amp; drop from "
                    "Explorer is blocked (UIPI)"
                )
            else:
                elev_text = (
                    "<b>Elevation:</b> Standard user (drag &amp; drop from "
                    "Explorer works)"
                )
        else:
            elev_text = "<b>Elevation:</b> N/A (not Windows)"
        self._add(self._label(elev_text))

        cpu = info.get("cpu", {}) or {}
        cpu_text = f"<b>CPU:</b> {html.escape(str(cpu.get('name', 'Unknown')))}<br>"
        cpu_text += (f"<b>Cores:</b> {cpu.get('cores', '?')}  "
                     f"<b>Threads:</b> {cpu.get('threads', '?')}")
        self._add(self._label(cpu_text))

        gpus = info.get("gpus", []) or []
        if gpus:
            self._add(self._label("<b>GPUs detected:</b>"))
            for gpu in gpus:
                text = f"• {html.escape(str(gpu.get('name', 'Unknown')))} ({html.escape(str(gpu.get('vendor', 'Unknown')))})"
                memory = gpu.get("memory", "")
                memory_used = gpu.get("memory_used", "")
                if memory:
                    if memory_used:
                        text += f"<br>&nbsp;&nbsp;VRAM: {memory_used} / {memory}"
                    else:
                        text += f"<br>&nbsp;&nbsp;VRAM: {memory}"
                for key, prefix in (("driver", "Driver"), ("core_clock", "Núcleo (max)"),
                                    ("temperature", "Temperatura")):
                    value = gpu.get(key, "")
                    if value:
                        text += f"<br>&nbsp;&nbsp;{prefix}: {html.escape(str(value))}"
                self._add(self._label(text))
        else:
            self._add(self._label("No GPUs detected."))

        recommended = info.get("recommended_device", "CPU")
        self._add(self._label(f"<b>Recommended device:</b> {html.escape(str(recommended))}"))

        self._populate_codecs()
        self.content_layout.addStretch()

    def _populate_codecs(self):
        """Lista de códecs disponibles, agrupada por dispositivo."""
        codecs = self._compatible_codecs or {}
        self._add(self._label(
            "<b>Codecs available on this machine:</b>"
        ))
        if not codecs:
            self._add(self._label(
                "<span style='color:#a0a0b0'>Not detected yet — press "
                "<i>Re-detect Hardware &amp; Codecs</i>.</span>"
            ))
            return

        # Sólo cuentan los dispositivos con códecs: la detección incluye
        # entradas vacías (p. ej. "AMD GPU" en un equipo NVIDIA) y contarlas
        # daría un recuento engañoso.
        listed = [(device, [c for c in (entries or []) if c])
                  for device, entries in codecs.items()]
        listed = [(device, entries) for device, entries in listed if entries]
        if not listed:
            self._add(self._label(
                "<span style='color:#a0a0b0'>No codec could be verified with "
                "the installed FFmpeg.</span>"
            ))
            return

        total = sum(len(entries) for _device, entries in listed)
        self._add(self._label(
            f"<span style='color:#a0a0b0'>{total} codec(s) across "
            f"{len(listed)} device(s). GPU entries are hardware encoders; "
            f"CPU entries are software.</span>"
        ))

        for device, entries in listed:
            self._add(self._label(
                f"<b>{html.escape(self._device_label(device))}</b> "
                f"<span style='color:#7f8fa6'>— {len(entries)} codec(s)</span>"
            ))
            for codec_id, codec_name in entries:
                detail = self._codec_detail(codec_id)
                self._add(self._label(
                    f"&nbsp;&nbsp;• {html.escape(str(codec_name))}{detail}"
                ))

    def _device_label(self, device) -> str:
        """Nombre del dispositivo, con el modelo de GPU si se conoce.

        La detección de códecs devuelve claves genéricas ("NVIDIA GPU"); el
        modelo está en la información de hardware, así que se añade aquí para
        que la lista diga a qué GPU pertenece cada grupo.
        """
        name = str(device)
        if "(" in name:
            return name
        vendor = name.split()[0] if name.split() else name
        if vendor.upper() == "CPU":
            return name
        for gpu in (self._hardware_info.get("gpus") or []):
            if str(gpu.get("vendor", "")).strip().lower() == vendor.strip().lower():
                model = str(gpu.get("name") or "").strip()
                if model:
                    return f"{vendor} GPU ({model})"
                break
        return name

    @staticmethod
    def _codec_detail(codec_id) -> str:
        """Sufijo discreto con encoder y formato de píxel del códec."""
        try:
            from src.config import get_codec_entry
            entry = get_codec_entry(codec_id) or {}
        except Exception:
            entry = {}
        parts = [str(entry.get(key)) for key in ("encoder", "pix_fmt")
                 if entry.get(key)]
        if not parts:
            return ""
        return ("  <span style='color:#7f8fa6'>"
                + html.escape(" · ".join(parts)) + "</span>")

    # --------------------------------------------------------------- estado
    def refresh(self, hardware_info=None, compatible_codecs=None) -> None:
        """Repinta el diálogo con datos nuevos (sin cerrarlo)."""
        if hardware_info is not None:
            self._hardware_info = hardware_info or {}
        if compatible_codecs is not None:
            self._compatible_codecs = compatible_codecs or {}
        self._populate()

    def set_busy(self, busy: bool, message: str = "") -> None:
        """Bloquea el botón mientras dura la redetección."""
        self._busy = bool(busy)
        self.btn_redetect.setEnabled(not self._busy and self._redetect_cb is not None)
        self.btn_redetect.setText(
            "Re-detecting…" if self._busy else "Re-detect Hardware && Codecs"
        )
        self.lbl_busy.setText(message if self._busy else "")

    def _on_redetect(self):
        if self._busy or self._redetect_cb is None:
            return
        self.set_busy(True, "Re-detecting hardware and codecs…")
        try:
            self._redetect_cb()
        except Exception as exc:
            self.set_busy(False)
            QMessageBox.warning(self, "Re-detect Error", f"Could not re-detect: {exc}")
