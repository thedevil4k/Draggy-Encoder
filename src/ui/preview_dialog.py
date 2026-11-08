import os
import tempfile

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from src.ffmpeg.runner import extract_frame
from src.utils.helpers import seconds_to_time


class ComparisonWidget(QLabel):
    """Muestra original y comprimido superpuestos con divisor arrastrable y zoom."""

    DIVIDER_HIT_RADIUS = 8          # px tolerados al pulsar sobre el divisor
    MIN_ZOOM = 0.5
    MAX_ZOOM = 4.0
    ZOOM_STEP = 1.15

    def __init__(self, original, compressed, parent=None):
        super().__init__(parent)
        self.original = original
        self.compressed = compressed
        self.division = 0.5           # divisor en [0..1]
        self.zoom = 1.0               # factor de zoom
        self.pan_x = 0                # offset de paneo (px)
        self.pan_y = 0

        self._dragging_divider = False
        self._dragging_pan = False
        self._drag_start_pos = None
        self._drag_start_pan = (0, 0)

        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0f0f1a;")
        self.setMouseTracking(True)

    # -------------------- API pública --------------------
    def set_division(self, value):
        """Acepta valor en [0..100] (slider) o [0..1] (fracción)."""
        if value > 1.0:
            value = value / 100.0
        self.division = max(0.0, min(1.0, value))
        self.update()
        self._sync_external_slider()

    def set_images(self, original, compressed):
        """Sustituye las imágenes comparadas (p.ej. al cambiar de frame)."""
        self.original = original
        self.compressed = compressed
        self.update()

    def set_zoom(self, zoom):
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(zoom)))
        self._clamp_pan()
        self.update()
        self._sync_external_zoom()

    def reset_view(self):
        """Restaura la vista a su layout inicial.

        Centra el divisor (50 % en el rango del slider), resetea el zoom
        a 1.0 (fit-to-window) y pone pan a cero. Actualiza el painter y
        emite los cambios a los sliders externos para que el UI cuadre
        con el estado interno.
        """
        self.division = 0.5
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update()
        self._sync_external_slider()
        self._sync_external_zoom()

    # -------------------- Sincronización con controles --------------------
    def _sync_external_slider(self):
        parent = self.parent()
        if parent and hasattr(parent, "slider"):
            slider = parent.slider
            slider.blockSignals(True)
            slider.setValue(int(self.division * 100))
            slider.blockSignals(False)

    def _sync_external_zoom(self):
        parent = self.parent()
        if parent and hasattr(parent, "zoom_slider"):
            slider = parent.zoom_slider
            slider.blockSignals(True)
            slider.setValue(int(self.zoom * 100))
            slider.blockSignals(False)

    # -------------------- Geometría --------------------
    def _base_scaled_size(self):
        """Tamaño base manteniendo aspecto (zoom=1, sin pan)."""
        if self.width() <= 0 or self.height() <= 0:
            return 0, 0
        ow, oh = self.original.width(), self.original.height()
        cw, ch = self.compressed.width(), self.compressed.height()
        ref_w, ref_h = (ow, oh) if (ow * oh) >= (cw * ch) else (cw, ch)
        if ref_w == 0 or ref_h == 0:
            return max(1, self.width()), max(1, self.height())
        ratio = min(self.width() / ref_w, self.height() / ref_h)
        return max(1, int(ref_w * ratio)), max(1, int(ref_h * ratio))

    def _image_rect(self):
        bw, bh = self._base_scaled_size()
        disp_w = int(bw * self.zoom)
        disp_h = int(bh * self.zoom)
        base_x = (self.width() - bw) // 2
        base_y = (self.height() - bh) // 2
        offset_x = int((bw - disp_w) / 2 + self.pan_x)
        offset_y = int((bh - disp_h) / 2 + self.pan_y)
        return base_x + offset_x, base_y + offset_y, disp_w, disp_h

    def _clamp_pan(self):
        bw, bh = self._base_scaled_size()
        if bw <= 0 or bh <= 0:
            return
        disp_w = int(bw * self.zoom)
        disp_h = int(bh * self.zoom)
        margin_x = max(50, abs((disp_w - bw) // 2))
        margin_y = max(50, abs((disp_h - bh) // 2))
        self.pan_x = max(-margin_x, min(margin_x, self.pan_x))
        self.pan_y = max(-margin_y, min(margin_y, self.pan_y))

    # -------------------- Eventos --------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        y = event.position().y()
        ix, iy, iw, ih = self._image_rect()
        on_image = (ix <= x <= ix + iw) and (iy <= y <= iy + ih)
        if not on_image:
            return

        split_x = ix + int(iw * self.division)
        if abs(x - split_x) <= self.DIVIDER_HIT_RADIUS:
            self._dragging_divider = True
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self._update_division_from_x(x)
        elif self.zoom > 1.05:
            # paneo sólo cuando hay zoom de sobra
            self._dragging_pan = True
            self._drag_start_pos = event.position()
            self._drag_start_pan = (self.pan_x, self.pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging_divider:
            self._update_division_from_x(event.position().x())
        elif self._dragging_pan and self._drag_start_pos is not None:
            dx = event.position().x() - self._drag_start_pos.x()
            dy = event.position().y() - self._drag_start_pos.y()
            self.pan_x = self._drag_start_pan[0] + dx
            self.pan_y = self._drag_start_pan[1] + dy
            self._clamp_pan()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_divider or self._dragging_pan:
                self._dragging_divider = False
                self._dragging_pan = False
                self.unsetCursor()

    def leaveEvent(self, event):
        if not (self._dragging_divider or self._dragging_pan):
            self.unsetCursor()

    def wheelEvent(self, event):
        # Ctrl + rueda -> zoom anclado al cursor
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                factor = self.ZOOM_STEP
            elif delta < 0:
                factor = 1.0 / self.ZOOM_STEP
            else:
                factor = 1.0

            if factor != 1.0:
                ix, iy, iw, ih = self._image_rect()
                cx = event.position().x()
                cy = event.position().y()
                if iw > 0 and ih > 0:
                    rel_x = (cx - ix) / iw
                    rel_y = (cy - iy) / ih
                else:
                    rel_x = rel_y = 0.5

                bw, bh = self._base_scaled_size()
                new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom * factor))
                self.zoom = new_zoom
                nx, ny, nw, nh = self._image_rect()
                base_x = (self.width() - bw) // 2
                base_y = (self.height() - bh) // 2
                target_x = cx - rel_x * nw
                target_y = cy - rel_y * nh
                cur_x = base_x + int((bw - nw) / 2) + self.pan_x
                cur_y = base_y + int((bh - nh) / 2) + self.pan_y
                self.pan_x += (target_x - cur_x)
                self.pan_y += (target_y - cur_y)
                self._clamp_pan()
                self.update()
                self._sync_external_zoom()
            event.accept()
        else:
            super().wheelEvent(event)

    def _update_division_from_x(self, x):
        ix, _iy, iw, _ih = self._image_rect()
        if iw <= 0:
            return
        self.division = max(0.0, min(1.0, (x - ix) / iw))
        self.update()
        self._sync_external_slider()

    # -------------------- Pintado --------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#0f0f1a"))

        if self.original.isNull() or self.compressed.isNull():
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No se pudieron cargar las imágenes de preview",
            )
            return

        bw, bh = self._base_scaled_size()
        ix, iy, iw, ih = self._image_rect()
        if bw <= 0 or bh <= 0 or iw <= 0 or ih <= 0:
            return

        # Pixmaps base (fit) -> luego reescalado al tamaño mostrado (zoomed)
        base_orig = self.original.scaled(
            bw, bh,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        base_comp = self.compressed.scaled(
            bw, bh,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        shown_orig = base_orig.scaled(
            iw, ih,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        shown_comp = base_comp.scaled(
            iw, ih,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Original completo (izquierda)
        painter.drawPixmap(ix, iy, shown_orig)

        # Comprimido solo en la parte derecha del divisor
        split_x = ix + int(iw * self.division)
        clip_w = max(0, (ix + iw) - split_x)
        if clip_w > 0:
            painter.setClipRect(split_x, iy, clip_w, ih)
            painter.drawPixmap(ix, iy, shown_comp)

        # Línea divisoria
        painter.setClipping(False)
        pen = painter.pen()
        pen.setColor(QColor("#00d2d3"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(split_x, iy, split_x, iy + ih)

        # Punto central visible para indicar que el divisor es arrastrable
        painter.setBrush(QColor("#00d2d3"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(split_x, iy + ih / 2.0), 6, 6)

        # Etiquetas "ORIGINAL" (izquierda) y "COMPRESSED" (derecha)
        self._draw_side_label(painter, "ORIGINAL", ix + 10, iy + 10, align_right=False)
        self._draw_side_label(
            painter,
            "COMPRESSED",
            (ix + iw) - 10,
            iy + 10,
            align_right=True,
        )

    def _draw_side_label(self, painter, text, x_anchor, y_anchor, align_right):
        """Etiqueta con backdrop semitransparente para que se lea sobre la imagen."""
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad_x = 10
        pad_y = 6
        text_w = metrics.horizontalAdvance(text) + pad_x * 2
        text_h = metrics.height() + pad_y * 2
        if align_right:
            rect = QRectF(x_anchor - text_w, y_anchor, text_w, text_h)
        else:
            rect = QRectF(x_anchor, y_anchor, text_w, text_h)

        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawPath(path)

        painter.setPen(QColor("#ffffff"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class PreviewDialog(QDialog):
    def __init__(self, original, compressed, info_text, parent=None, artifacts=None):
        super().__init__(parent)
        self.setWindowTitle("Compression Preview")
        self.setMinimumSize(900, 700)
        self.artifacts = artifacts or None
        self._single_frame = bool(
            self.artifacts and self.artifacts.get("single_frame")
        )
        self._frame_origin = (
            float(self.artifacts.get("clip_start", 0) or 0)
            if self.artifacts else 0.0
        )
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(300)
        self._regen_timer.timeout.connect(self._regenerate_frames)
        self._setup_ui(original, compressed, info_text)

    def _setup_ui(self, original, compressed, info_text):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Compression Preview")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00d2d3; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.comparison = ComparisonWidget(original, compressed)
        layout.addWidget(self.comparison, stretch=1)

        # Slider del divisor (mantiene sincronía con el arrastre directo)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.comparison.set_division)
        layout.addWidget(self.slider)

        # Selector de instante: re-extrae frames del clip ya codificado
        # (sin repetir la cadena IA) y del origen en el nuevo instante.
        # La preview de IA solo renderiza un frame, así que no hay clip que
        # recorrer: se oculta el selector y se usa el campo "Frame at" para
        # pedir otro instante.
        if self.artifacts and not self._single_frame:
            clip_start = float(self.artifacts.get("clip_start", 0) or 0)
            clip_length = float(self.artifacts.get("clip_length", 5) or 5)
            frame_time = float(self.artifacts.get("time", 0) or 0)
            self.frame_slider = QSlider(Qt.Orientation.Horizontal)
            self.frame_slider.setRange(0, max(10, int(round(clip_length * 10))))
            self.frame_slider.setValue(
                max(0, min(
                    self.frame_slider.maximum(),
                    int(round((frame_time - clip_start) * 10)),
                ))
            )
            self.frame_slider.valueChanged.connect(self._on_frame_slider)
            self.frame_time_label = QLabel("")
            self.frame_time_label.setStyleSheet(
                "color: #00d2d3; font-size: 11px;"
            )
            self.frame_time_label.setFixedWidth(120)
            frame_row = QHBoxLayout()
            frame_caption = QLabel("Frame")
            frame_caption.setStyleSheet("color: #a0a0b0; font-size: 11px;")
            frame_row.addWidget(frame_caption)
            frame_row.addWidget(self.frame_slider, stretch=1)
            frame_row.addWidget(self.frame_time_label)
            layout.addLayout(frame_row)
            self._update_frame_time_label()

        # Zoom + reset
        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Zoom")
        zoom_label.setStyleSheet("color: #a0a0b0;")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(
            int(self.comparison.MIN_ZOOM * 100),
            int(self.comparison.MAX_ZOOM * 100),
        )
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset_view)

        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(self.zoom_slider, stretch=1)
        zoom_row.addWidget(self.btn_reset)
        layout.addLayout(zoom_row)

        info_label = QLabel(info_text or "")
        info_label.setStyleSheet("color: #a0a0b0; font-size: 12px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        if self.artifacts and self._single_frame:
            hint_text = (
                "La preview de IA compara un solo fotograma. Arrastra la línea "
                "vertical para comparar; cambia 'Frame at' y vuelve a lanzar "
                "Preview para otro instante. Ctrl + rueda para zoom."
            )
        elif self.artifacts:
            hint_text = (
                "Arrastra la línea vertical para comparar; usa el deslizador "
                "de frame para elegir el instante. Ctrl + rueda para zoom."
            )
        else:
            hint_text = (
                "Arrastra la línea vertical para comparar. "
                "Ctrl + rueda del ratón para hacer zoom sobre la imagen."
            )
        hint = QLabel(hint_text)
        hint.setStyleSheet("color: #00d2d3; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Botones
        btn_layout = QHBoxLayout()
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        self.btn_save_original = QPushButton("Save Original")
        self.btn_save_original.clicked.connect(
            lambda: self._save_image(self.comparison.original)
        )
        self.btn_save_compressed = QPushButton("Save Compressed")
        self.btn_save_compressed.clicked.connect(
            lambda: self._save_image(self.comparison.compressed)
        )

        btn_layout.addWidget(self.btn_save_original)
        btn_layout.addWidget(self.btn_save_compressed)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def _on_zoom_slider(self, value):
        self.comparison.set_zoom(value / 100.0)

    def _on_reset_view(self):
        self.comparison.reset_view()

    # -------------------- Selector de instante --------------------

    def _on_frame_slider(self, value):
        self._update_frame_time_label()
        # Debounce: la extracción de frames corre al soltar el deslizador
        # (o 300 ms después del último movimiento) para no lanzar ffmpeg
        # a cada tick.
        self._regen_timer.start()

    def _update_frame_time_label(self):
        if not hasattr(self, "frame_slider"):
            return
        t = self._frame_origin + self.frame_slider.value() / 10.0
        self.frame_time_label.setText(seconds_to_time(t))

    def _regenerate_frames(self):
        """Re-extrae el frame original (del origen) y el comprimido (del
        clip ya codificado con la cadena IA) en el instante elegido."""
        artifacts = self.artifacts
        if not artifacts:
            return
        try:
            ffmpeg = artifacts.get("ffmpeg")
            source = artifacts.get("source")
            clip = artifacts.get("compressed_clip")
            clip_start = float(artifacts.get("clip_start", 0))
            clip_length = float(artifacts.get("clip_length", 5))
            # El slider está acotado a la ventana del clip (no a la duración
            # completa del vídeo): antes se podía pedir el minuto 10 y se
            # comparaba contra el último frame del clip de 5 s.
            t = clip_start + self.frame_slider.value() / 10.0
            out_dir = artifacts.get("dir")
            orig_jpg = os.path.join(out_dir, f"frame_orig_{t:.1f}.jpg")
            comp_jpg = os.path.join(out_dir, f"frame_comp_{t:.1f}.jpg")
            # El instante dentro del clip comprimido (offset respecto al
            # inicio del clip), recortado para no pasarse del final.
            comp_t = max(0.0, min(t - clip_start, clip_length - 0.05))
            if not extract_frame(ffmpeg, source, t, orig_jpg):
                return
            if not extract_frame(ffmpeg, clip, comp_t, comp_jpg):
                return
            opm = QPixmap()
            cpm = QPixmap()
            opm.load(orig_jpg)
            cpm.load(comp_jpg)
            if opm.isNull() or cpm.isNull():
                return
            self.comparison.set_images(opm, cpm)
        except Exception:
            # Fallo de extracción: mantener los frames anteriores.
            pass

    def _save_image(self, pixmap):
        dest, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", "frame.jpg", "Images (*.jpg *.png)"
        )
        if dest:
            pixmap.save(dest)
