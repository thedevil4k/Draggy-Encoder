"""Genera ``assets/app.ico`` (icono multi-resolución) desde ``assets/Import.png``.

Windows necesita un .ico con varias resoluciones (16/24/32/48/64/128/256) para
que el logo se vea nítido en TODOS los sitios: Explorador, Settings > Apps
(entrada de desinstalación), el instalador y el desinstalador. PyInstaller
convertía antes el PNG de 400x400 a un .ico de UNA sola resolución, que en
algunos contextos de Windows acaba mostrando un icono genérico.

Usa PySide6 (dependencia de la app y del entorno de build) en vez de Pillow,
así funciona tanto en el runner de CI como en un checkout local.

Uso:
    python packaging/windows/make_icon.py

Salida: ``assets/app.ico`` (se regenera cada vez; es un asset versionable).
"""

from __future__ import annotations

import os
import struct

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QImage

# Tamaños recomendados por Inno Setup y el shell de Windows.
SIZES = [256, 128, 64, 48, 32, 24, 16]


def _project_root() -> str:
    # El script vive en packaging/windows/; el directorio del script está 2
    # niveles por debajo de la raíz del repo.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _to_png_bytes(image: QImage) -> bytes:
    """Serializa un QImage a bytes PNG en memoria."""
    buffer = QByteArray()
    device = QBuffer(buffer)
    device.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(device, "PNG"):
        raise RuntimeError("No se pudo codificar el frame a PNG")
    device.close()
    return bytes(buffer)


def build_ico(source_path: str, output_path: str) -> None:
    source = QImage(source_path)
    if source.isNull():
        raise SystemExit(f"No se pudo cargar el icono fuente: {source_path}")

    # ARGB32 garantiza canal alfa y una conversión estable al escalar.
    source = source.convertToFormat(QImage.Format.Format_ARGB32)

    frames: list[tuple[int, bytes]] = []
    for size in SIZES:
        scaled = source.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Si el source no fuera cuadrado, centrar sobre lienzo cuadrado.
        if scaled.width() != size or scaled.height() != size:
            canvas = QImage(size, size, QImage.Format.Format_ARGB32)
            canvas.fill(Qt.GlobalColor.transparent)
            from PySide6.QtGui import QPainter
            painter = QPainter(canvas)
            painter.drawImage(
                (size - scaled.width()) // 2,
                (size - scaled.height()) // 2,
                scaled,
            )
            painter.end()
            scaled = canvas
        frames.append((size, _to_png_bytes(scaled)))

    # ICONDIR (6 bytes) + ICONDIRENTRY (16 bytes cada uno) + PNGs.
    header = struct.pack("<HHH", 0, 1, len(frames))
    offset = 6 + 16 * len(frames)
    directory = bytearray()
    images = bytearray()
    for size, png in frames:
        # 0 en width/height significa 256.
        w = 0 if size == 256 else size
        h = 0 if size == 256 else size
        directory += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(png), offset)
        images += png
        offset += len(png)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(header + directory + images)


def main() -> None:
    root = _project_root()
    source_path = os.path.join(root, "assets", "Import.png")
    output_path = os.path.join(root, "assets", "app.ico")
    build_ico(source_path, output_path)
    print(f"Icono generado: {output_path}")


if __name__ == "__main__":
    main()
