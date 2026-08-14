import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from src.utils.helpers import is_windows_elevated


class HardwareDialog(QDialog):
    def __init__(self, hardware_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Info")
        self.setMinimumWidth(460)
        self._setup_ui(hardware_info)

    def _setup_ui(self, hardware_info):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Hardware Information")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00d2d3; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # --- OS ---
        os_text = hardware_info.get("os") or "Unknown OS"
        machine = hardware_info.get("machine") or ""
        if machine:
            os_text += f" · {machine}"
        layout.addWidget(QLabel(f"<b>OS:</b> {os_text}"))

        # --- Python interpreter ---
        python = hardware_info.get("python", {}) or {}
        py_version = python.get("version") or (sys.version.split()[0] if sys.version else "?")
        py_bits = python.get("bits")
        py_text = f"<b>Python interpreter:</b> {py_version}"
        if py_bits:
            py_text += f" ({py_bits}-bit)"
        layout.addWidget(QLabel(py_text))

        py_exe = python.get("executable") or sys.executable
        if py_exe:
            exe_lbl = QLabel(py_exe)
            exe_lbl.setWordWrap(True)
            exe_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            exe_lbl.setStyleSheet("color: #a0a0b0; font-size: 10px;")
            layout.addWidget(exe_lbl)

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
        layout.addWidget(QLabel(elev_text))

        cpu = hardware_info.get("cpu", {})
        cpu_text = f"<b>CPU:</b> {cpu.get('name', 'Unknown')}<br>"
        cpu_text += f"<b>Cores:</b> {cpu.get('cores', '?')}  <b>Threads:</b> {cpu.get('threads', '?')}"
        layout.addWidget(QLabel(cpu_text))

        gpus = hardware_info.get("gpus", [])
        if gpus:
            layout.addWidget(QLabel("<b>GPUs detected:</b>"))
            for gpu in gpus:
                name = gpu.get("name", "Unknown")
                vendor = gpu.get("vendor", "Unknown")
                memory = gpu.get("memory", "")
                memory_used = gpu.get("memory_used", "")
                driver = gpu.get("driver", "")
                core_clock = gpu.get("core_clock", "")
                temperature = gpu.get("temperature", "")
                text = f"• {name} ({vendor})"
                if memory:
                    if memory_used:
                        text += f"<br>&nbsp;&nbsp;VRAM: {memory_used} / {memory}"
                    else:
                        text += f"<br>&nbsp;&nbsp;VRAM: {memory}"
                if driver:
                    text += f"<br>&nbsp;&nbsp;Driver: {driver}"
                if core_clock:
                    text += f"<br>&nbsp;&nbsp;Núcleo (max): {core_clock}"
                if temperature:
                    text += f"<br>&nbsp;&nbsp;Temperatura: {temperature}"
                label = QLabel(text)
                label.setWordWrap(True)
                layout.addWidget(label)
        else:
            layout.addWidget(QLabel("No GPUs detected."))

        recommended = hardware_info.get("recommended_device", "CPU")
        layout.addWidget(QLabel(f"<b>Recommended device:</b> {recommended}"))

        layout.addStretch()
