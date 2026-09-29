# La matriz de NVENC de NVIDIA, transcrita. ORÁCULO DE VERIFICACIÓN, no estado
# de la aplicación: vive aquí, en ``tests/``, y no en ``src/`` a propósito, para
# que su ubicación no prometa lo que no hace. La app NO lee este fichero; su
# único importador es el contraste de ``tests/test_ffmpeg_build_versions.py``.
#
# Por qué aquí y no en ``src/``: el dueño del dato en tiempo de ejecución es
# ``src/hardware/codecs.py``, que decide por reconocimiento de nombre (el
# hardware llega como "NVIDIA GeForce RTX 3060 Laptop GPU", "GA102" o
# "GeForce GTX 980Ti" pegado, un espacio que no se puede acotar). Una tabla
# ordenada por nombre de producto no puede sustituir a ese reconocimiento, así
# que los dos no pueden ser el mismo dueño: uno afirma y el otro comprueba que
# lo afirmado siga siendo lo que dice NVIDIA. Poner el oráculo junto al código
# que verifica sugería que la app lo leía, y no lo hace.
#
# Actualizar esto NO es reescribir código: se copia la fila nueva aquí y el
# contraste dice si el mapa la reclama o si le falta. El test corre sin red;
# esta tabla es la copia local de la fuente, con su fecha, para que el contraste
# sea reproducible.
#
# Fuente: la página oficial de NVIDIA, pestaña "NVENC - Encoding" (las cinco
# sub-tablas: Consumer, Professional, Server, DGX y Jetson/IGX). De cada fila
# interesan cuatro columnas, las únicas que el programa puede afirmar:
# "H.264 (AVC) YUV 4:2:0", "H.265 (HEVC) 4K YUV 4:2:0", "HEVC 10-bit support"
# y "AV1 YUV 4:2:0". Las de 4:2:2 / 4:4:4 / lossless / B-frames no se copian
# porque el programa no ofrece esos códecs.
#
# Las filas donde NVIDIA agrupa varios productos ("GeForce GTX 750 / GTX 950 /
# 960") están partidas aquí en un producto por fila, que es como las ve el
# programa; el agrupamiento original se conserva en los comentarios de sección.

SOURCE = {
    "url": "https://developer.nvidia.com/video-encode-decode-support-matrix",
    "tab": "NVENC - Encoding",
    "consulted": "2026-09-25",
    "columnas": (
        "H.264 (AVC) YUV 4:2:0, H.265 (HEVC) 4K YUV 4:2:0, "
        "HEVC 10-bit support, AV1 YUV 4:2:0"
    ),
}

# Cada fila: (producto, familia, motores NVENC, h264, hevc, hevc10, av1).
# "motores" es la columna "Total # of NVENC" de la tabla.
FILAS = (
    # ---- Consumer: Blackwell, 9ª generación (las cinco, AV1 incluido) -------
    ("GeForce RTX 5050", "Blackwell", 1, True, True, True, True),
    ("GeForce RTX 5060", "Blackwell", 1, True, True, True, True),
    ("GeForce RTX 5060 Ti", "Blackwell", 1, True, True, True, True),
    ("GeForce RTX 5070", "Blackwell", 1, True, True, True, True),
    ("GeForce RTX 5070 Ti", "Blackwell", 1, True, True, True, True),
    ("GeForce RTX 5080", "Blackwell", 2, True, True, True, True),
    ("GeForce RTX 5090", "Blackwell", 3, True, True, True, True),
    ("GeForce RTX 5070 Laptop", "Blackwell", 1, True, True, True, True),
    ("GeForce RTX 5070 Ti Laptop", "Blackwell", 1, True, True, True, True),
    ("GeForce RTX 5080 Laptop", "Blackwell", 2, True, True, True, True),
    ("GeForce RTX 5090 Laptop", "Blackwell", 3, True, True, True, True),

    # ---- Consumer: Ada Lovelace, 8ª generación (AV1 incluido) --------------
    ("GeForce RTX 4050 Laptop", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4060 Laptop", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4060", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4060TI", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4070 Laptop", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4070", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4070 Super", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4070 TI", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4070 TI Super", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4080 Laptop", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4080 16GB", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4080 Super", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4090 Laptop", "Ada", 1, True, True, True, True),
    ("GeForce RTX 4090", "Ada", 1, True, True, True, True),

    # ---- Consumer: Ampere, 7ª generación ----------------------------------
    ("GeForce RTX 2050", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3050", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3050 Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3050 Ti Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3060", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3060 Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3060 Ti", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3060 Ti Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3070", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3070 Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3070 Ti", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3070 Ti Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3080", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3080 Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3080 Ti", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3080 Ti Laptop", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3090", "Ampere", 1, True, True, True, False),
    ("GeForce RTX 3090 TI", "Ampere", 1, True, True, True, False),
    ("GeForce MX570", "Ampere", 1, True, True, True, False),
    ("GeForce MX570 A", "Ampere", 0, False, False, False, False),

    # ---- Consumer: Turing, 6ª y 7ª generación ------------------------------
    ("GeForce MX450", "Turing", 0, False, False, False, False),
    ("GeForce MX550", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1630", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1650 Max-Q", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1650 GDDR5", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1650 GDDR6", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1650 SUPER", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1660", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1660 Ti", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1660 Ti Max-Q", "Turing", 1, True, True, True, False),
    ("GeForce GTX 1660 SUPER", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2060", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2060 (TU104)", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2060 SUPER", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2070", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2070 Max-Q", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2070 SUPER", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2080", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2080 Max-Q", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2080 SUPER", "Turing", 1, True, True, True, False),
    ("GeForce RTX 2080Ti", "Turing", 1, True, True, True, False),
    ("Titan RTX", "Turing", 1, True, True, True, False),

    # ---- Consumer: Pascal, 6ª generación -----------------------------------
    ("GeForce MX150", "Pascal", 0, False, False, False, False),
    ("GeForce MX 330", "Pascal", 0, False, False, False, False),
    ("GeForce MX350", "Pascal", 0, False, False, False, False),
    ("GeForce GT 1030", "Pascal", 0, False, False, False, False),
    ("GeForce GTX 1050", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1050 Ti", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1060", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1070M", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1080M", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1070", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1070 Ti", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1080", "Pascal", 1, True, True, True, False),
    ("GeForce GTX 1080 Ti", "Pascal", 1, True, True, True, False),
    ("GeForce GTX Titan Xp", "Pascal", 1, True, True, True, False),
    ("Titan V", "Volta", 1, True, True, True, False),

    # ---- Consumer: Maxwell 2ª (H.264 + HEVC 8-bit, sin Main10) ------------
    # Publicadas en tres líneas: "GeForce GTX 750 / GTX 950 / 960",
    # "Geforce GTX 965M > 980M / 980MX" y "GeForce GTX 960 Ti / 970 / 980".
    ("GeForce GTX 750", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 950", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 960", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 965M", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 980M", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 980MX", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 960 Ti", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 970", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 980", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX 980Ti", "Maxwell 2", 1, True, True, False, False),
    ("GeForce GTX Titan X", "Maxwell 2", 1, True, True, False, False),

    # ---- Consumer: Maxwell 1ª (sólo H.264) ---------------------------------
    ("GeForce GTX 745", "Maxwell 1", 1, True, False, False, False),
    ("GeForce GTX 750 Ti", "Maxwell 1", 1, True, False, False, False),
    ("GeForce 845M", "Maxwell 1", 1, True, False, False, False),
    ("GeForce 940M", "Maxwell 1", 1, True, False, False, False),
    ("GeForce 940MX", "Maxwell 1", 1, True, False, False, False),
    ("GeForce 945M", "Maxwell 1", 1, True, False, False, False),
    ("GeForce GTX 850A", "Maxwell 1", 1, True, False, False, False),
    ("GeForce GTX 960A", "Maxwell 1", 1, True, False, False, False),
    ("GeForce GTX 850M", "Maxwell 1", 1, True, False, False, False),
    ("GeForce GTX 960M", "Maxwell 1", 1, True, False, False, False),
    ("GeForce 830A", "Maxwell 1", 0, False, False, False, False),
    ("GeForce 830M", "Maxwell 1", 0, False, False, False, False),
    ("GeForce 920MX", "Maxwell 1", 0, False, False, False, False),
    ("GeForce 945A", "Maxwell 1", 0, False, False, False, False),
    ("GeForce MX110", "Maxwell 1", 0, False, False, False, False),
    ("GeForce MX130", "Maxwell 1", 0, False, False, False, False),

    # ---- Professional: Blackwell y Ada (AV1) -------------------------------
    ("NVIDIA RTX PRO 500 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 1000 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 2000 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 3000 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 4000 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 4000 Blackwell SFF Edition", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 4500 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 5000 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX PRO 6000 Blackwell", "Blackwell", 1, True, True, True, True),
    ("NVIDIA RTX 500 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 1000 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 2000 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 3000 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 3500 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 4000 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 4000 SFF Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 4500 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 5000 Ada", "Ada", 1, True, True, True, True),
    ("NVIDIA RTX 6000 Ada", "Ada", 1, True, True, True, True),

    # ---- Professional: RTX A-series (Ampere) -------------------------------
    ("NVIDIA RTX A500 Laptop", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A1000 Laptop", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A2000 Laptop", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A3000", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A400", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A1000", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A2000", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A4000", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A4500", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A5000", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A5500", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A6000", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A4000 Laptop", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A4500 Laptop", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A5000 Laptop", "Ampere", 1, True, True, True, False),
    ("NVIDIA RTX A5500 Laptop", "Ampere", 1, True, True, True, False),

    # ---- Professional: serie T y Quadro T (Turing) -------------------------
    ("NVIDIA T400", "Turing", 1, True, True, True, False),
    ("NVIDIA T500", "Turing", 0, False, False, False, False),
    ("NVIDIA T550 Laptop", "Turing", 1, True, True, True, False),
    ("NVIDIA T600", "Turing", 1, True, True, True, False),
    ("NVIDIA T1000", "Turing", 1, True, True, True, False),
    ("NVIDIA T600 Laptop", "Turing", 1, True, True, True, False),
    ("NVIDIA T1200 Laptop", "Turing", 1, True, True, True, False),
    ("Quadro T1000", "Turing", 1, True, True, True, False),
    ("Quadro T2000", "Turing", 1, True, True, True, False),
    ("Quadro T2000 Max-Q", "Turing", 1, True, True, True, False),

    # ---- Professional: Quadro RTX (Turing) ---------------------------------
    ("Quadro RTX 3000", "Turing", 1, True, True, True, False),
    ("Quadro RTX 3000 Max-Q", "Turing", 1, True, True, True, False),
    ("Quadro RTX 4000", "Turing", 1, True, True, True, False),
    ("Quadro RTX 4000 Max-Q", "Turing", 1, True, True, True, False),
    ("Quadro RTX 5000", "Turing", 1, True, True, True, False),
    ("Quadro RTX 6000", "Turing", 1, True, True, True, False),
    ("Quadro RTX 8000", "Turing", 1, True, True, True, False),

    # ---- Professional: Quadro P (Pascal) -----------------------------------
    ("Quadro P500", "Pascal", 0, False, False, False, False),
    ("Quadro P520", "Pascal", 0, False, False, False, False),
    ("Quadro P400", "Pascal", 1, True, True, True, False),
    ("Quadro P600", "Pascal", 1, True, True, True, False),
    ("Quadro P620", "Pascal", 1, True, True, True, False),
    ("Quadro P1000", "Pascal", 1, True, True, True, False),
    ("Quadro P2000", "Pascal", 1, True, True, True, False),
    ("Quadro P2200", "Pascal", 1, True, True, True, False),
    ("Quadro P3200", "Pascal", 1, True, True, True, False),
    ("Quadro P4200", "Pascal", 1, True, True, True, False),
    ("Quadro P4000", "Pascal", 1, True, True, True, False),
    ("Quadro P5000", "Pascal", 1, True, True, True, False),
    ("Quadro P5200", "Pascal", 1, True, True, True, False),
    ("Quadro P6000", "Pascal", 1, True, True, True, False),
    # El GP100 es Pascal "big": su fila dice "HEVC 4K YUV 4:2:0: YES" pero
    # "HEVC 10-bit support: NO", igual que el Tesla P100.
    ("Quadro GP100", "Pascal big", 3, True, True, False, False),

    # ---- Professional: Volta y Maxwell ------------------------------------
    ("Quadro GV100", "Volta", 3, True, True, True, False),
    ("Quadro M2000", "Maxwell 2", 1, True, True, False, False),
    ("Quadro M2200M", "Maxwell 2", 1, True, True, False, False),
    ("Quadro M3000", "Maxwell 2", 1, True, True, False, False),
    ("Quadro M4000", "Maxwell 2", 1, True, True, False, False),
    ("Quadro M5000", "Maxwell 2", 1, True, True, False, False),
    ("Quadro M5500", "Maxwell 2", 1, True, True, False, False),
    ("Quadro M6000", "Maxwell 2", 1, True, True, False, False),
    ("Quadro M500M", "Maxwell 1", 0, False, False, False, False),
    ("Quadro M520M", "Maxwell 1", 0, False, False, False, False),
    ("Quadro M600M", "Maxwell 1", 1, True, False, False, False),
    ("Quadro M620M", "Maxwell 1", 1, True, False, False, False),
    ("Quadro M1000M", "Maxwell 1", 1, True, False, False, False),
    ("Quadro M1200M", "Maxwell 1", 1, True, False, False, False),
    ("Quadro M2000M", "Maxwell 1", 1, True, False, False, False),
    ("Quadro K620", "Maxwell 1", 1, True, False, False, False),
    ("Quadro K1200", "Maxwell 1", 1, True, False, False, False),
    ("Quadro K2200", "Maxwell 1", 1, True, False, False, False),

    # ---- Server (Data Center) ----------------------------------------------
    ("NVIDIA L4", "Ada", 2, True, True, True, True),
    ("NVIDIA L40", "Ada", 3, True, True, True, True),
    ("NVIDIA L40S", "Ada", 3, True, True, True, True),
    ("NVIDIA A100", "Ampere", 0, False, False, False, False),
    ("NVIDIA A30", "Ampere", 0, False, False, False, False),
    ("NVIDIA A40", "Ampere", 1, True, True, True, False),
    ("NVIDIA A16", "Ampere", 4, True, True, True, False),
    ("NVIDIA A10", "Ampere", 1, True, True, True, False),
    ("NVIDIA A2", "Ampere", 1, True, True, True, False),
    ("NVIDIA HGX B200", "Blackwell", 0, False, False, False, False),
    ("Tesla T4", "Turing", 1, True, True, True, False),
    ("Tesla V100", "Volta", 3, True, True, True, False),
    ("Tesla P4", "Pascal", 2, True, True, True, False),
    ("Tesla P6", "Pascal", 2, True, True, True, False),
    ("Tesla P40", "Pascal", 2, True, True, True, False),
    # Tesla P100: Pascal "big", sin HEVC 10-bit (como el Quadro GP100).
    ("Tesla P100", "Pascal big", 3, True, True, False, False),
    ("Tesla M4", "Maxwell 2", 1, True, True, False, False),
    ("Tesla M40", "Maxwell 2", 2, True, True, False, False),
    ("Tesla M6", "Maxwell 2", 2, True, True, False, False),
    ("Tesla M60", "Maxwell 2", 4, True, True, False, False),
    ("Tesla M10", "Maxwell 1", 4, True, False, False, False),

    # ---- DGX (chasis) ------------------------------------------------------
    ("DGX Spark", "Blackwell", 1, True, True, True, True),
    ("DGX A100", "Ampere", 0, False, False, False, False),
    ("DGX-1", "Volta", 24, True, True, True, False),
    ("DGX-2", "Volta", 48, True, True, True, False),
    # "DGX Station" aparece dos veces en la tabla: la generación Blackwell con
    # 0 motores y la Volta con 4. Chasis, en cualquier caso.
    ("DGX Station", "Volta", 12, True, True, True, False),

    # ---- Jetson / IGX (módulos de desarrollo) -----------------------------
    ("Jetson T2000", "Thor", 1, True, True, True, True),
    ("Jetson T3000", "Thor", 1, True, True, True, True),
    ("Jetson T4000", "Thor", 1, True, True, True, True),
    ("Jetson T5000", "Thor", 1, True, True, True, True),
    ("IGX T5000", "Thor", 1, True, True, True, True),
    ("IGX T7000", "Thor", 1, True, True, True, True),
)

# Filas que NVIDIA ya no publica en NVENC pero que el programa sigue
# reconociendo, y las que publica con capacidades que chocan consigo mismas.
# El valor es (códecs que se afirman, motivo). Con una decisión presente, la
# tabla manda menos que el motivo: por eso la razón vive aquí, en el dato, y no
# en un comentario del código ni en un test.
DECISIONES = {
    # La tabla de NVENC no publica ninguna de estas tres: sólo aparecen en la
    # de decodificación, como Maxwell 2ª gen. Sin fila que leer, se les da el
    # veredicto de su familia, que sí publica 1 motor con H.264 y HEVC 4:2:0.
    "GeForce GTX 910M": (("h264_nvenc", "hevc_nvenc"),
                          "no aparece en la tabla de NVENC; se hereda el "
                          "veredicto de su familia (Maxwell 2ª, 1 motor)"),
    "GeForce GTX 920M": (("h264_nvenc", "hevc_nvenc"),
                          "no aparece en la tabla de NVENC; se hereda el "
                          "veredicto de su familia (Maxwell 2ª, 1 motor)"),
    "GeForce GTX 920A": (("h264_nvenc", "hevc_nvenc"),
                          "no aparece en la tabla de NVENC; se hereda el "
                          "veredicto de su familia (Maxwell 2ª, 1 motor)"),
    # Kepler. La tabla actual ya no publica ninguna GeForce anterior al GTX
    # 745 (que NVIDIA clasifica como Maxwell 1ª gen): dejó de listarlos al
    # terminar el soporte con la rama R470. Eso es una omisión por fin de vida,
    # no una fila de 0 motores, así que no se retira una afirmación que nadie
    # ha refutado: se mantiene el H.264, que es una afirmación de qué códecs
    # existen, no de rendimiento. El aviso de fin de soporte lo dice aparte.
    "GeForce GTX 760": (("h264_nvenc",),
                               "Kepler: NVIDIA ya no publica esta fila; se "
                               "mantiene la afirmación de códec (omisión por "
                               "fin de vida, no 0 motores)"),
    "GeForce GTX 770": (("h264_nvenc",),
                               "Kepler: NVIDIA ya no publica esta fila; se "
                               "mantiene la afirmación de códec (omisión por "
                               "fin de vida, no 0 motores)"),
    "GeForce GTX 780": (("h264_nvenc",),
                               "Kepler: NVIDIA ya no publica esta fila; se "
                               "mantiene la afirmación de códec (omisión por "
                               "fin de vida, no 0 motores)"),
    # NVIDIA se contradice consigo misma: la 940MX sale con 1 motor en
    # "Geforce 845M / 940M / 940MX / 945M" y con 0 en "GeForce GTX 920MX -
    # 940MX". Igual la 945M, en la primera línea y en "GeForce 830M > 945M".
    # Cuando la fuente no se pone de acuerdo, no se afirma.
    "GeForce 940MX": ((), "NVIDIA publica 1 motor en una línea y 0 en otra"),
    "GeForce GTX 940MX": ((), "NVIDIA publica 1 motor en una línea y 0 en otra"),
    "GeForce 945M": ((), "NVIDIA publica 1 motor en una línea y 0 en otra"),
    "GeForce GTX 945M": ((), "NVIDIA publica 1 motor en una línea y 0 en otra"),
    # La RTX 2070 Ti existe como producto, pero la tabla de NVENC no la
    # publica: solo aparecen la 2070, la 2070 Max-Q y la 2070 SUPER. Mismo
    # caso que la 910M/920M/920A: se hereda el veredicto de la familia, que si
    # publica 1 motor, y la herencia queda aqui escrita.
    "GeForce RTX 2070 Ti": (("h264_nvenc", "hevc_nvenc", "hevc_nvenc_10bit"),
                             "no aparece en la tabla de NVENC; se hereda el "
                             "veredicto de su familia (Turing, 1 motor)"),
    # Dos productos que la tabla no publica con ese nombre suelto, solo con
    # una variante del mismo chip. Se afirma lo que dice la variante.
    "GeForce GTX 1650": (("h264_nvenc", "hevc_nvenc", "hevc_nvenc_10bit"),
                         "NVIDIA no publica una fila para la GTX 1650 sin "
                         "sufijo; publica sus variantes (Max-Q, GDDR5, GDDR6, "
                         "SUPER) con 1 motor, y es el mismo chip"),
    "GeForce RTX 4080": (("h264_nvenc", "hevc_nvenc", "hevc_nvenc_10bit",
                          "av1_nvenc", "av1_nvenc_10bit"),
                         "NVIDIA no publica una fila para la RTX 4080 sin "
                         "sufijo; publica la variante 16GB con 1 motor, y es "
                         "el mismo chip"),
    # La tabla publica DOS GPUs distintas con el mismo nombre "GTX Titan X":
    # la Pascal de "GeForce GTX Titan X/ Titan Xp" (con Main10) y la Maxwell 2ª
    # de una línea propia (sin él). Sin más dato no se puede elegir, y elegir
    # mal sería prometer un códec que no está.
    "GeForce GTX Titan X": ((), "el mismo nombre aparece en dos filas con "
                            "capacidades distintas (Pascal y Maxwell 2ª)"),
    # Filas con encoder que no pueden ser la GPU de un PC.
    "DGX Spark": ((), "chasis de centro de datos, no una GPU de PC"),
    "DGX A100": ((), "chasis de centro de datos, no una GPU de PC"),
    "DGX-1": ((), "chasis de centro de datos, no una GPU de PC"),
    "DGX-2": ((), "chasis de centro de datos, no una GPU de PC"),
    "DGX Station": ((), "chasis de centro de datos, no una GPU de PC"),
    "Jetson T2000": ((), "módulo de desarrollo, no aparece como GPU de un PC"),
    "Jetson T3000": ((), "módulo de desarrollo, no aparece como GPU de un PC"),
    "Jetson T4000": ((), "módulo de desarrollo, no aparece como GPU de un PC"),
    "Jetson T5000": ((), "módulo de desarrollo, no aparece como GPU de un PC"),
    "IGX T5000": ((), "módulo de desarrollo, no aparece como GPU de un PC"),
    "IGX T7000": ((), "módulo de desarrollo, no aparece como GPU de un PC"),
    "NVIDIA HGX B200": ((), "acelerador de centro de datos, no una GPU de PC"),
    "NVIDIA B200": ((), "acelerador de centro de datos, no una GPU de PC"),
    "NVIDIA H100 SXM": ((), "acelerador de centro de datos, no una GPU de PC"),
    "NVIDIA H100 PCIe": ((), "acelerador de centro de datos, no una GPU de PC"),
    "NVIDIA GB200": ((), "acelerador de centro de datos, no una GPU de PC"),
}

# Nombres que el programa tiene que reconocer y que NVIDIA no publica con esa
# grafía: son las formas en que un WMI o ``nvidia-smi`` reportan una GPU, o la
# grafía con espacio de un nombre que la tabla escribe pegado. Cada uno apunta
# a la fila de la que sale, así que el contraste no espera un criterio nuevo
# para ellos sino el de su fila.
ESCRITURAS = {
    "GeForce GTX 920MX": "GeForce 920MX",
    "GeForce GTX 830A": "GeForce 830A",
    "GeForce GTX 830M": "GeForce 830M",
    "GeForce GTX 945A": "GeForce 945A",
    "GeForce 850A": "GeForce GTX 850A",
    "GeForce 850M": "GeForce GTX 850M",
    "GeForce 960A": "GeForce GTX 960A",
    "GeForce 960M": "GeForce GTX 960M",
    "GeForce GTX 845M": "GeForce 845M",
    "GeForce GTX 940M": "GeForce 940M",
    "GeForce GTX 980 Ti": "GeForce GTX 980Ti",
    "GeForce MX 110": "GeForce MX110",
    "GeForce MX 130": "GeForce MX130",
    "GeForce MX 150": "GeForce MX150",
    "GeForce MX 350": "GeForce MX350",
    "GeForce MX 450": "GeForce MX450",
    "GeForce MX 550": "GeForce MX550",
    "GeForce MX 570": "GeForce MX570",
    "GeForce MX 570 A": "GeForce MX570 A",
    "GeForce RTX 4060 Ti": "GeForce RTX 4060TI",
    "GeForce RTX 4070 Ti": "GeForce RTX 4070 TI",
    "GeForce RTX 4070 SUPER": "GeForce RTX 4070 Super",
    "GeForce RTX 4070 Ti SUPER": "GeForce RTX 4070 TI Super",
    "GeForce RTX 4080 SUPER": "GeForce RTX 4080 Super",
    "NVIDIA RTX 2000 Ada Generation": "NVIDIA RTX 2000 Ada",
    "NVIDIA RTX 4000 SFF Ada Generation": "NVIDIA RTX 4000 SFF Ada",
    "NVIDIA RTX 5000 Ada Generation": "NVIDIA RTX 5000 Ada",
    "NVIDIA RTX 6000 Ada Generation": "NVIDIA RTX 6000 Ada",
    "NVIDIA RTX PRO 2000 Blackwell Generation": "NVIDIA RTX PRO 2000 Blackwell",
    "NVIDIA RTX PRO 6000 Blackwell Max-Q": "NVIDIA RTX PRO 6000 Blackwell",
    "GeForce RTX 2080 Ti": "GeForce RTX 2080Ti",
    "NVIDIA GeForce GTX 760": "GeForce GTX 760",
    "NVIDIA GeForce GTX 770": "GeForce GTX 770",
    "NVIDIA GeForce GTX 780": "GeForce GTX 780",
    "NVIDIA GeForce RTX 3060": "GeForce RTX 3060",
    "NVIDIA GeForce GTX 1650": "GeForce GTX 1650",
    "NVIDIA GeForce RTX 4090": "GeForce RTX 4090",
    "GeForce RTX 3060 Laptop GPU": "GeForce RTX 3060 Laptop",
    "GeForce RTX 3060 Ti Laptop GPU": "GeForce RTX 3060 Ti Laptop",
    "GeForce RTX 4090 Laptop GPU": "GeForce RTX 4090 Laptop",
    "GeForce RTX A2000 12GB": "NVIDIA RTX A2000",
    "GeForce GTX 1050 Ti": "GeForce GTX 1050 Ti",
    "GeForce GTX 1070 Ti": "GeForce GTX 1070 Ti",
    # Codigos de chip: es como algunos WMI reportan la GPU en vez del nombre
    # comercial. Heredan la fila de la arquitectura a la que pertenecen.
    "GM107": "GeForce GTX 750 Ti",
    "GM204": "GeForce GTX 980",
    "GM206": "Tesla M4",
    "GP102": "GeForce GTX 1080",
    "GP104": "GeForce GTX 1070",
    "GP107": "GeForce GTX 1050",
    "GA102": "GeForce RTX 3090",
    "GA104": "GeForce RTX 3070",
    "GA107": "GeForce RTX 3050",
    "TU104": "GeForce RTX 2060 (TU104)",
    "TU117": "GeForce GTX 1650",
    "AD102": "GeForce RTX 4090",
    "AD104": "GeForce RTX 4080",
    "GB202": "GeForce RTX 5090",
    "GB204": "GeForce RTX 5080",
}

# Lectura de la tabla: qué códecs debe afirmar el mapa para una fila, y si esa
# fila se declara como "sin ningún motor NVENC". Es lo único que traduce la
# tabla a lo que el programa dice, y vive aquí para que el mapa y el dato no
# puedan separarse sin que se note.
def expected_for(gpu_name):
    """(codecs esperados, sin ningun motor NVENC) para un nombre de GPU.

    Se resuelve primero la grafia (una escritura puede llevar a una decision),
    luego la decision si la hay, y si no la fila de la tabla.
    """
    name = str(gpu_name or "")
    producto = ESCRITURAS.get(name, name)
    if producto in DECISIONES:
        return tuple(DECISIONES[producto][0]), False
    row = _FILA_POR_NOMBRE.get(producto)
    if row is None:
        raise KeyError(name)
    _name, _family, motores, h264, hevc, hevc10, av1 = row
    if motores == 0:
        return (), True
    codecs = []
    if h264:
        codecs.append("h264_nvenc")
    if hevc:
        codecs.append("hevc_nvenc")
    if hevc10:
        codecs.append("hevc_nvenc_10bit")
    if av1:
        codecs.append("av1_nvenc")
        codecs.append("av1_nvenc_10bit")
    return tuple(codecs), False


def all_names():
    """Todos los nombres que el contraste tiene que cubrir, sin repetir."""
    names = ([row[0] for row in FILAS] + list(ESCRITURAS)
             + list(DECISIONES))
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


_FILA_POR_NOMBRE = {row[0]: row for row in FILAS}
