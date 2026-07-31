import re

from src.config import ALL_KNOWN_CODECS


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
    """
    name = gpu_name or ""
    codecs = []

    # ── Arquitecturas NVIDIA ──
    # Kepler / Maxwell 1st (GTX 700, GTX 750 Ti): solo H.264
    is_kepler_or_maxwell1st = _match_any(name, [
        r"GTX\s*7\d{2}",           # GTX 760, 770, 780 (Kepler) y 750/750 Ti (Maxwell 1st)
        r"GM107",                   # GM107 = Maxwell 1st
    ])

    # Maxwell 2nd (GTX 900, GTX 950, GTX 960, GTX 970, GTX 980, GTX 980 Ti)
    is_maxwell2nd = _match_any(name, [
        r"GTX\s*9\d{2}",           # GTX 950, 960, 970, 980
        r"GM20[046]",              # GM200, GM204, GM206
    ])

    # Pascal (GTX 10xx, GT 1030, Quadro Pxxx, Tesla Pxx)
    is_pascal = _match_any(name, [
        r"GTX\s*10\d{2}",          # GTX 1050, 1060, 1070, 1080
        r"GT\s*10\d{2}",           # GT 1030
        r"Quadro\s*P\d{3,4}",      # Quadro P400, P620, P1000, P2000, P4000, P6000
        r"\bP\d{3,4}\b",           # P400, P2000, P4000 (formato standalone)
        r"GP10[247]",              # GP102, GP104, GP107
    ])

    # Turing (GTX 16xx, RTX 20xx, Quadro RTX xxxx)
    is_turing = _match_any(name, [
        r"GTX\s*16\d{2}",          # GTX 1650, 1660, 1660 Ti
        r"RTX\s*20\d{2}",          # RTX 2060, 2070, 2080, 2080 Ti
        r"Quadro\s*RTX\s*\d{4}",   # Quadro RTX 4000, 5000, 6000, 8000
        r"TU\d{3}",                # TU102, TU104, TU106, TU116, TU117
    ])

    # Ampere (RTX 30xx, RTX A-series workstation)
    is_ampere = _match_any(name, [
        r"RTX\s*30\d{2}",          # RTX 3050, 3060, 3070, 3080, 3090
        r"RTX\s*A\d{4}",           # RTX A2000, A4000, A5000, A6000
        r"GA10",                    # GA102, GA103, GA104, GA106, GA107
    ])

    # Ada Lovelace / Blackwell (RTX 40xx, 50xx, RTX xxxx Ada)
    is_ada = _match_any(name, [
        r"RTX\s*40\d{2}",          # RTX 4060, 4070, 4080, 4090
        r"RTX\s*50\d{2}",          # RTX 5060, 5070, 5080, 5090 (Blackwell)
        r"RTX\s*\d{4}\s*Ada",      # RTX 4000 Ada, RTX 6000 Ada
        r"AD10",                    # AD102, AD103, AD104, AD106, AD107
        r"GB20[24]",               # GB202, GB204 (Blackwell)
    ])

    has_nvenc = is_kepler_or_maxwell1st or is_maxwell2nd or is_pascal or is_turing or is_ampere or is_ada

    # H.264 NVENC: disponible en todas las GPUs con NVENC (Kepler+).
    if has_nvenc:
        codecs.append(("h264_nvenc", "H.264 NVENC (8-bit)"))

    # HEVC 8-bit: Maxwell 2nd gen+ (GTX 900 series), Pascal, Turing, Ampere, Ada
    if is_maxwell2nd or is_pascal or is_turing or is_ampere or is_ada:
        codecs.append(("hevc_nvenc", "HEVC NVENC (8-bit)"))

    # HEVC 10-bit: Pascal y posteriores
    if is_pascal or is_turing or is_ampere or is_ada:
        codecs.append(("hevc_nvenc_10bit", "HEVC NVENC (10-bit)"))

    # AV1 8-bit & 10-bit: Ada Lovelace / Blackwell
    if is_ada:
        codecs.append(("av1_nvenc", "AV1 NVENC (8-bit)"))
        codecs.append(("av1_nvenc_10bit", "AV1 NVENC (10-bit)"))

    return codecs


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
    """Codecs software siempre disponibles en CPU."""
    return [
        ("libx264", "H.264 (CPU)"),
        ("libx265", "HEVC (CPU)"),
        ("libsvtav1", "AV1 SVT (CPU)"),
        ("libaom-av1", "AV1 AOM (CPU)"),
    ]


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
