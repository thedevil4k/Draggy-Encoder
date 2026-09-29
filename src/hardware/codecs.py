import re

from src.config import ALL_KNOWN_CODECS, codecs_for_vendor


# Fuente de verdad del mapa de NVIDIA: la matriz oficial que NVIDIA publica y
# mantiene actualizada en
# https://developer.nvidia.com/video-encode-decode-support-matrix
# (pestaña "NVENC - Encoding"). Lo que se afirma de cada familia es lo que dice
# esa tabla: su columna "Total # of NVENC" y, si es mayor que cero, qué códecs
# codifica. No se infiere nada de la arquitectura ni del nombre.
#
# MX_NVENC_SIN_ENCODER son los modelos que esa matriz publica con CERO motores.
# Distinguirlos de "no lo conozco" importa porque lo que ve el usuario cambia:
# aquí no hay encoder, no falta información. Sólo entran las filas que la
# matriz dice CERO sin contradecirse; las contradictorias (940MX, 945M) y las
# que ya no publica (Kepler) no se afirman ni se niegan, y su motivo está
# escrito en el test de contraste.
MX_NVENC_SIN_ENCODER = (
    # Pascal, 0 NVENC en la matriz
    r"GeForce\s*MX\s*150\b", r"GeForce\s*MX\s*330\b", r"GeForce\s*MX\s*350\b",
    r"GeForce\s*GT\s*1030\b",
    # El \b no es adorno: sin él, "Quadro P500" se comía a la P5000, que sí
    # tiene NVENC (Pascal 6ª gen en la matriz).
    r"Quadro\s*P\s*500\b", r"Quadro\s*P\s*520\b",
    # Turing, 0 NVENC en la matriz
    r"GeForce\s*MX\s*450\b", r"NVIDIA\s*T500\b",
    # Ampere, 0 NVENC en la matriz (variante sin encoder)
    r"GeForce\s*MX\s*570\s*A\b",
    # Ampere de centro de datos: la matriz les da 0 motores (A100, A30). El ancla
    # evita que "A30" se coma a una hypothetical A3000, que sí codifica.
    r"NVIDIA\s*A(?:100|30)\b",
    # Maxwell 1ª gen, 0 NVENC en la matriz
    r"GeForce\s*MX\s*110\b", r"GeForce\s*MX\s*130\b",
    r"Quadro\s*M\s*(?:500|520)M\b",
    r"GeForce\s*(?:GTX\s+)?(?:830A|830M|945A|920MX)\b",
)


def nvidia_has_no_encoder(gpu_name):
    """True si NVIDIA publica esta GPU sin ningún motor NVENC.

    No es "no lo sé": es que la matriz oficial le da 0 motores y todas las
    columnas de códec a NO. Para estas GPUs la codificación por GPU no existe,
    así que la versión de FFmpeg no decide nada y la lista de códecs que
    devuelve ``get_nvidia_codecs`` es vacía a propósito.
    """
    return _match_any(gpu_name or "", MX_NVENC_SIN_ENCODER)


def _match_any(text, patterns):
    """Devuelve True si text coincide con alguno de los patrones."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def get_nvidia_codecs(gpu_name):
    """Devuelve los codecs NVENC compatibles según el modelo de GPU NVIDIA.

    Cubre todos los formatos comunes:
    - Desktop:  "NVIDIA GeForce RTX 3060"
    - Laptop:   "NVIDIA GeForce RTX 3060 Laptop GPU"
    - Workstation: "NVIDIA RTX A2000", "Quadro RTX 4000"
    - Genéricos: "NVIDIA GeForce RTX 4060 Ti", "RTX 4070 SUPER"

    Lo que se afirma de cada familia sale de la matriz oficial de NVIDIA (ver
    ``MX_NVENC_SIN_ENCODER``): nada se deduce de la arquitectura ni del año.
    Las GPUs que NVIDIA publica sin encoder devuelven lista vacía, que es lo
    que las saca de la lista de dispositivos con códecs.
    """
    name = gpu_name or ""
    codecs = []

    # NVIDIA publica estas familias con 0 motores NVENC: no hay nada que
    # ofrecer, y prometer "h264/…/hevc" aquí sería inventarlo.
    if nvidia_has_no_encoder(name):
        return []

    # ── Arquitecturas NVIDIA ──
    # Kepler / Maxwell 1st (GTX 700, GTX 750 Ti, Quadro M…M y K…): solo H.264
    is_kepler_or_maxwell1st = _match_any(name, [
        # GTX 745, 750 Ti y 760/770/780 (la nota Kepler, de is_legacy_nvidia).
        # La GTX 750 a secas cae aquí también, pero la reclama además
        # is_maxwell2nd, que le añade el HEVC que la matriz sí le da.
        r"GTX\s*7\d{2}",
        r"GM107",
        # Quadro M…M de Maxwell 1ª gen: la matriz les da 1 NVENC con H.264 y
        # "H.265 (HEVC) 4K YUV 4:2:0: NO" (M600M, M620M, M1000M, M1200M, M2000M).
        r"Quadro\s*M\s*(?:1000|1200|2000|600|620)M\b",
        r"Quadro\s*K\s*(?:620|1200|2200)\b",
        # Portátiles Maxwell 1ª gen que la matriz agrupa sin prefijo "GTX"
        # ("GeForce 845M / 940M" y "GTX 850M > 960M", "GTX 850A > 960A").
        r"GeForce\s*(?:GTX\s+)?(?:845|850|940|960)M\b",
        r"GeForce\s*(?:GTX\s+)?(?:850|960)A\b",
        r"Tesla\s*M10\b",
    ])

    # Maxwell 2nd (GTX 900, GTX 950, GTX 960, GTX 970, GTX 980, GTX 980 Ti,
    # y las Quadro M de Maxwell: M2000, M2200M, M3000, M4000, M5000, M5500, M6000)
    is_maxwell2nd = _match_any(name, [
        # Los números van uno a uno porque un "GTX 9xx" a secas no decide: la
        # matriz publica "GTX 960" (Maxwell 2ª, HEVC en YES) y "GTX 960M / 960A
        # / 940M / 940MX / 945M" (Maxwell 1ª, HEVC en NO) como productos
        # distintos. Sin ``\b`` porque la matriz escribe "GTX 980Ti" pegado: el
        # lookahead acepta el "Ti" y rechaza la letra de 1ª gen que va detrás.
        r"GTX\s*(?:950|960|970|980)(?![0-9])(?![MXA])",  # 950, 960, 960 Ti, 970, 980, 980 Ti
        r"GTX\s*965M\b",
        r"GTX\s*980M(?:X)?\b",
        # Herencia declarada: "GeForce GTX 910M / 920M / 920A". La tabla de NVENC
        # no publica estas tres (solo la de decodificación, como Maxwell 2ª), así
        # que se les da el veredicto de su familia, que sí publica 1 motor. Por
        # eso se nombran una a una y no entran en el patrón de arriba.
        r"(?<![A-Za-z])GTX\s*9(?:10|20)[MA]\b",
        r"GM20[046]",
        # Quadro M de Maxwell 2ª gen. El sufijo "M" no basta para decidir: la
        # M2200M es de 2ª gen (GM206) y la M2000M de 1ª, así que se nombran los
        # números tal como los publica la matriz en vez de usar un patrón amplio.
        r"Quadro\s*M\s*(?:2200M|[2-6]\d{3}(?!M))",
        # Tesla M4 (GM206), M6, M40, M60: Maxwell 2ª gen.
        r"Tesla\s*M(?:60|40|6|4)\b",
        # La GTX 750 sin "Ti" es Maxwell 2ª (matriz: "GeForce GTX 750 / GTX
        # 950 / 960", 1 NVENC, HEVC 4:2:0 YES y 10-bit NO). El lookahead deja
        # fuera a la 750 Ti, que sí es Maxwell 1ª.
        r"GTX\s*750(?!\s*Ti)",
    ])

    # Pascal (GTX 10xx, Quadro Pxxx, Tesla Pxx, GP10x). El GT 1030 y los Quadro
    # P500/P520 ya no entran aquí: la matriz de NVIDIA les da 0 NVENC y
    # ``nvidia_has_no_encoder`` los ha descartado antes.
    is_pascal = _match_any(name, [
        r"GTX\s*10\d{2}",          # GTX 1050, 1060, 1070, 1080
        r"Quadro\s*P\d{3,4}",      # Quadro P400, P620, P1000, P2000, P4000, P6000
        r"(?<![A-Za-z])P\d{3,4}\b",  # P400, P2000, P4000 (formato standalone)
        r"GP10[247]",
        # Quadro GP100: Pascal "big", con su excepción de Main10 más abajo.
        r"GP100\b",
        # Tesla P4 / P6 / P40 / P100. El nombre va suelto ("Tesla P4"), así que
        # no sirve el patrón de arriba, que necesita 3-4 dígitos.
        r"Tesla\s*P(?:100|40|4|6)\b",
        # Titan Xp: Pascal "big" de 3 GB, 2 NVENC, HEVC 4:2:0 y 10-bit (matriz:
        # "GeForce GTX Titan X/ Titan Xp"). Sólo la Xp: la "GTX Titan X" a secas
        # es ambigua, porque la matriz publica ese nombre dos veces con
        # capacidades distintas. Ver MATRIZ_NO_RECONOCIDAS.
        r"(?<![A-Za-z])Titan\s*Xp\b",
    ])

    # Turing (GTX 16xx, RTX 20xx, Quadro RTX xxxx, serie T, Tesla T4, Titan RTX)
    is_turing = _match_any(name, [
        r"GTX\s*16\d{2}",          # GTX 1650, 1660, 1660 Ti
        # La matriz publica "GeForce RTX 2060 / 2070" y "GeForce RTX 2060
        # (TU104)", y ninguna fila "GTX 2060": el patrón es sólo RTX a propósito.
        # Sin ``\b`` porque la matriz escribe la 2080 Ti pegada ("2080Ti").
        r"RTX\s*20[6-9]\d(?![0-9])",  # RTX 2060, 2070, 2080, 2080 Ti / 2080Ti
        r"Quadro\s*RTX\s*\d{4}",   # Quadro RTX 4000, 5000, 6000, 8000
        r"TU\d{3}",                # TU102, TU104, TU106, TU116, TU117
        r"GeForce\s*MX\s*550",     # Turing 6ª gen, 1 NVENC en la matriz
        # Serie T de workstation: T400, T550, T600, T1000 y T1200, todas con
        # 1 NVENC y HEVC 10-bit en la matriz. El T500 no entra porque la misma
        # tabla le da 0 motores (nvidia_has_no_encoder lo aparta antes), y el
        # ``(?<![A-Za-z])`` evita que esto se pegue a un "RTX 4000" cualquiera.
        r"(?<![A-Za-z])T\s*(?:400|550|600|1000|1200)(?![0-9])",
        # Quadro T2000 (y su Max-Q): la matriz publica su NVENC en la fila
        # "NVIDIA T1000 / T1000 8GB", que es ese mismo board.
        r"Quadro\s*T2000\b",
        r"Titan\s*RTX\b",
        r"Tesla\s*T4\b",
    ])

    # Volta (Quadro GV100, Tesla V100, Titan V): 3 motores NVENC, H.264 + HEVC
    # 8/10-bit, sin AV1. La matriz no publica ningún GeForce Volta, así que
    # esta lista son nombres enteros, no una familia.
    is_volta = _match_any(name, [
        r"Quadro\s*GV100\b",
        r"Tesla\s*V100\b",
        r"Titan\s*V\b",
    ])

    # Ampere (RTX 30xx, RTX 2050, RTX A-series, datacenter A10/A16/A2/A40, MX 570)
    is_ampere = _match_any(name, [
        r"RTX\s*30\d{2}",          # RTX 3050, 3060, 3070, 3080, 3090
        r"RTX\s*2050\b",           # RTX 2050 = GA107, Ampere en la matriz
        r"RTX\s*A\d{3,4}\b",       # RTX A400 … A6000 (3 y 4 cifras)
        r"GA10",                    # GA102, GA103, GA104, GA106, GA107
        r"GeForce\s*MX\s*570(?!A)", # Ampere 7ª gen, 1 NVENC (la variante "A" no)
        # A10 / A16 / A2 / A40 del centro de datos: 1 NVENC, sin AV1. El A100 y
        # el A30 no entran aquí porque su fila dice 0 motores.
        r"(?<![A-Za-z])A(?:16|40|10|2)\b",
    ])

    # Ada Lovelace / Blackwell (RTX 40xx, 50xx, RTX xxxx Ada)
    #
    # Las workstation Quadro RTX llevan un número de 4 cifras que choca con el
    # de las GeForce Blackwell: "Quadro RTX 5000" es Turing en la matriz, pero
    # ``RTX\s*50\d{2}`` lo habría tomado por un RTX 5080 y le habría prometido
    # AV1. Las workstation tienen prioridad, que es como lo lista NVIDIA.
    is_quadro_rtx = _match_any(name, [
        r"Quadro\s*RTX\s*\d{4}",   # Quadro RTX 3000, 4000, 5000, 6000, 8000
    ])
    is_ada = not is_quadro_rtx and _match_any(name, [
        r"RTX\s*40\d{2}",          # RTX 4060, 4070, 4080, 4090
        r"RTX\s*50\d{2}",          # RTX 5060, 5070, 5080, 5090 (Blackwell)
        r"RTX\s*\d{3,4}\s*Ada",    # RTX 500 Ada … RTX 6000 Ada (3 o 4 cifras)
        r"AD10",                    # AD102, AD103, AD104, AD106, AD107
        r"GB20[24]",               # GB202, GB204 (Blackwell)
        # Centro de datos Ada: L4, L40, L40S.
        r"(?:^|\s)L(?:40S|40|4)\b",
        # Port Blackwell de workstation: "RTX PRO 6000 Blackwell …".
        r"RTX\s*PRO\s*\d{3,4}",
    ])

    # H.264 con NVENC + HEVC a partir de Maxwell 2ª (Matriz: "H.265 4K YUV
    # 4:2:0" en NO para Kepler y Maxwell 1ª, en YES desde Maxwell 2ª en
    # adelante). Volta va con el mismo grupo que Pascal/Turing: su fila dice
    # H.264 YES, HEVC 4K 4:2:0 YES, HEVC 10-bit YES, AV1 NO.
    con_hevc = is_maxwell2nd or is_pascal or is_turing or is_volta or is_ampere or is_ada
    # Main10: Pascal y posteriores, con la excepción de los dos "Pascal big".
    con_hevc_10bit = is_pascal or is_turing or is_volta or is_ampere or is_ada

    has_nvenc = is_kepler_or_maxwell1st or con_hevc

    # H.264 NVENC: disponible en todas las GPUs con NVENC (Kepler+).
    if has_nvenc:
        codecs.append(("h264_nvenc", "H.264 NVENC (8-bit)"))

    # HEVC 8-bit: Maxwell 2nd gen+ (GTX 900 series), Pascal, Volta, Turing, Ampere, Ada
    if con_hevc:
        codecs.append(("hevc_nvenc", "HEVC NVENC (8-bit)"))

    # HEVC 10-bit: Pascal y posteriores.
    # Maxwell 2ª gen NO: su NVENC codifica HEVC 4:2:0 8-bit y nada más (la
    # matriz de NVIDIA marca "HEVC 10-bit support: NO" en GM200/GM204/GM206).
    #
    # Excepción de Pascal: el GP100 (Quadro) y el P100 (Tesla) son los dos
    # "Pascal big" y su matriz dice "HEVC 4K YUV 4:2:0: YES" pero "HEVC 10-bit
    # support: NO".
    sin_hevc_10bit = _match_any(name, [
        r"(?<![A-Za-z])P100\b",     # Tesla P100
        r"(?<![A-Za-z])GP100\b",    # Quadro GP100
    ])
    if con_hevc_10bit and not sin_hevc_10bit:
        codecs.append(("hevc_nvenc_10bit", "HEVC NVENC (10-bit)"))

    # AV1 8-bit & 10-bit: Ada Lovelace / Blackwell
    if is_ada:
        codecs.append(("av1_nvenc", "AV1 NVENC (8-bit)"))
        codecs.append(("av1_nvenc_10bit", "AV1 NVENC (10-bit)"))

    return codecs


def is_legacy_nvidia(gpu_name):
    """True si la GPU es de una generación sin soporte de driver reciente.

    NVIDIA cerró el soporte de Kepler (GTX 600/700, GM107) con la última rama
    de drivers R470: una actualización de driver puede haberle quitado NVENC.
    Se usa para avisar (no para decidir) en el selector de versión de FFmpeg.
    """
    return _match_any(gpu_name or "", [
        r"GTX\s*[67]\d{2}",   # GTX 600/700
        r"GT\s*6\d{2}",       # GT 600/700
        r"GM[0-9]{3}",        # Maxwell 1st / Kepler
    ])


def get_amd_codecs(gpu_name):
    """Devuelve los codecs AMF compatibles según el modelo de GPU AMD."""
    name = gpu_name or ""
    codecs = []

    # RDNA3: RX 7000 series
    is_rdna3 = _match_any(name, [r"RX\s*7\d{3}[XT]?", r"Radeon\s*.*\s7\d{3}"])
    # RDNA2: RX 6000 series
    is_rdna2 = _match_any(name, [r"RX\s*6\d{3}[XT]?", r"Radeon\s*.*\s6\d{3}"])
    # RDNA1: RX 5000 series
    is_rdna1 = _match_any(name, [r"RX\s*5\d{3}[XT]?", r"Radeon\s*.*\s5\d{3}"])
    # Vega / Polaris / GCN (RX 400/500 series, Vega, Radeon R7/R9/RX)
    is_gcn = _match_any(name, [r"RX\s*[45]\d{3}", r"RX\s*Vega", r"Vega", r"Polaris", r"Radeon\s*(R7|R9|RX)"])

    if is_rdna3 or is_rdna2 or is_rdna1 or is_gcn:
        codecs.append(("h264_amf", "H.264 AMF"))
        codecs.append(("hevc_amf", "HEVC AMF"))

    # AV1: solo RDNA3
    if is_rdna3:
        codecs.append(("av1_amf", "AV1 AMF"))

    return codecs


def get_intel_codecs(device_name):
    """Devuelve los codecs QuickSync compatibles según el modelo de GPU Intel."""
    name = device_name or ""
    codecs = []

    # Arc / Alchemist (DG2)
    is_arc = _match_any(name, [r"Arc\s*", r"Alchemist", r"DG2"])
    # Meteor Lake / Arrow Lake / Raptor Lake (13th-14th gen+)
    is_meteor_plus = _match_any(name, [r"i[3579]-1[45]\d{3}", r"i[3579]-1[34]\d{3}", r"Meteor", r"Arrow"])
    # Tiger Lake / Alder Lake (11th-12th gen)
    is_tiger_alder = _match_any(name, [r"i[3579]-1[12]\d{3}", r"Tiger", r"Alder"])
    # Skylake hasta Ice Lake (6th-10th gen)
    is_skylake_plus = _match_any(name, [r"i[3579]-\d{4,5}", r"Skylake", r"Kaby", r"Coffee", r"Ice", r"Comet", r"Rocket"])
    # Iris / UHD Graphics (integradas). HD 530+ y similares son Skylake+ y soportan HEVC.
    is_iris_uhd = _match_any(name, [r"Iris", r"UHD\s*Graphics", r"HD\s*Graphics"])
    # Si el nombre incluye un número de modelo de GPU Intel (ej. 530, 620, 630, 770) o es UHD/Iris, asumir HEVC.
    has_gpu_model = _match_any(name, [r"HD\s*Graphics\s*\d{3}", r"Iris\s*.*\s*\d{3}", r"UHD\s*Graphics\s*\d{3}", r"UHD\s*Graphics", r"Iris\s"])

    # H.264: casi universal en QuickSync
    if is_arc or is_meteor_plus or is_tiger_alder or is_skylake_plus or is_iris_uhd:
        codecs.append(("h264_qsv", "H.264 QuickSync"))

    # HEVC: Skylake (6th gen)+, UHD/Iris/HD Graphics con modelo numérico, o Arc.
    if is_arc or is_meteor_plus or is_tiger_alder or is_skylake_plus or has_gpu_model:
        codecs.append(("hevc_qsv", "HEVC QuickSync"))

    # AV1 encode: Arc / Meteor Lake+
    if is_arc or is_meteor_plus:
        codecs.append(("av1_qsv", "AV1 QuickSync"))

    return codecs


def get_cpu_codecs():
    """Codecs software disponibles en CPU (derivados de la tabla declarativa).

    Incluye 8/10/12-bit y lossless/mezzanine (FFV1, ProRes, DNxHR); el probe
    de compatibilidad de FFmpeg filtra los que no estén compilados.
    """
    return codecs_for_vendor("cpu")


def get_available_devices_with_codecs(hardware_info):
    """Devuelve un diccionario {device_name: [(codec_id, codec_name)]} según el hardware."""
    devices = {"CPU": get_cpu_codecs()}

    for gpu in hardware_info.get("gpus", []):
        vendor = gpu.get("vendor", "")
        name = gpu.get("name", "")
        if not name:
            continue

        if vendor == "NVIDIA":
            codecs = get_nvidia_codecs(name)
            if codecs:
                devices[f"NVIDIA GPU ({name})"] = codecs
        elif vendor == "AMD":
            codecs = get_amd_codecs(name)
            if codecs:
                devices[f"AMD GPU ({name})"] = codecs
        elif vendor == "Intel":
            codecs = get_intel_codecs(name)
            if codecs:
                devices[f"Intel GPU ({name})"] = codecs

    return devices
