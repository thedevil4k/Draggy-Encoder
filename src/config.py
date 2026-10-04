import os
import sys
import platform

APP_NAME = "DRAGGY ENCODER"
APP_VERSION = "v1"

# Identidad del proyecto para la pestaña About y el flag --crash-report.
REPO_OWNER = "thedevil4k"
REPO_NAME = "Draggy-Encoder"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
ISSUES_URL = f"{REPO_URL}/issues"
NEW_ISSUE_URL = f"{ISSUES_URL}/new"


def get_base_dir():
    """Devuelve el directorio base de la aplicación."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir():
    """Directorio escribible para modelos, runtimes, logs y preferencias.

    Los bundles portables mantienen los datos junto al ejecutable. Los
    paquetes instalados en Linux (Debian/AppImage) pueden vivir en /opt o en
    un montaje de solo lectura, así que sus lanzadores establecen
    ``DRAGGY_DATA_DIR`` y usan el perfil del usuario.
    """
    override = os.environ.get("DRAGGY_DATA_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if getattr(sys, "frozen", False) and sys.platform.startswith("linux") and (
        os.environ.get("APPIMAGE") or os.environ.get("APPDIR")
    ):
        data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
        return os.path.join(data_home, "draggy-encoder")
    return get_base_dir()


def get_bin_dir():
    """Directorio raíz donde se almacenan todos los assets externos:
    binarios, modelos, runtimes, logs. Reside junto al ejecutable en bundles
    portables y en el perfil del usuario cuando se instala en Linux.
    """
    return os.path.join(get_data_dir(), "bin")


def get_ffmpeg_dir():
    """Subcarpeta dedicada a FFmpeg (ffmpeg.exe + ffprobe.exe + manifest)."""
    return os.path.join(get_bin_dir(), "ffmpeg")


def get_models_dir():
    """Subcarpeta dedicada a modelos de IA (placeholder hasta integración)."""
    return os.path.join(get_bin_dir(), "models")


def get_runtimes_dir():
    """Subcarpeta dedicada a runtimes de IA (placeholder hasta integración)."""
    return os.path.join(get_bin_dir(), "runtimes")


def get_logs_dir():
    """Subcarpeta dedicada a logs de FFmpeg y otros procesos."""
    return os.path.join(get_bin_dir(), "logs")


def get_session_dir():
    """Subcarpeta para estado de sesión (gitignored).

    Aquí viven las preferencias del usuario (``preferences.json``) y otros
    state efímeros que NO deberían versionarse.
    """
    return os.path.join(get_bin_dir(), ".session")


def ensure_runtime_dirs():
    """Crea todas las subcarpetas internas de bin/ si no existen."""
    for d in (
        get_ffmpeg_dir(),
        get_models_dir(),
        get_runtimes_dir(),
        get_logs_dir(),
        get_session_dir(),
    ):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


def get_ffmpeg_executable_name():
    return "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"


def get_ffprobe_executable_name():
    return "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"


def get_asset_path(filename):
    """Resuelve la ruta de un asset teniendo en cuenta PyInstaller."""
    if getattr(sys, "frozen", False):
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base_path = get_base_dir()
    return os.path.join(base_path, "assets", filename)


# ---------------------------------------------------------------------------
# Tabla declarativa de codecs de vídeo — fuente de verdad única.
#
# Cada entrada describe TODO lo que necesita el encoder: el encoder real de
# FFmpeg, el pixel format de salida (8/10/12 bits), el modo de control de
# tasa y el contenedor preferido. De aquí derivan ``ALL_KNOWN_CODECS`` (para
# el probe de compatibilidad), ``resolve_codec`` (para build_args y la ruta
# IA) y el catálogo de CPU.
#
#   pix_fmt          : pixel format de salida que el encoder necesita
#   rate_mode        : "crf" | "bitrate" | "lossless"
#   container        : contenedor preferido para ese codec
#   mp4_incompatible : True si el codec NO puede muxearse en MP4
#   extra            : args extra del encoder (perfil ProRes/DNxHR, etc.)
#   probe_size       : tamaño de la fuente del probe (DNxHR exige 1080p)
# ---------------------------------------------------------------------------
VIDEO_CODEC_TABLE = [
    # --- CPU: H.264 / H.265 / AV1 en 8/10/12 bits ---
    {"id": "libx264", "label": "H.264 (CPU)", "encoder": "libx264", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "cpu"},
    {"id": "libx264_10bit", "label": "H.264 10-bit (CPU)", "encoder": "libx264", "pix_fmt": "yuv420p10le", "rate_mode": "crf", "container": "mkv", "vendor": "cpu"},
    {"id": "libx265", "label": "HEVC (CPU)", "encoder": "libx265", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "cpu"},
    {"id": "libx265_10bit", "label": "HEVC 10-bit (CPU)", "encoder": "libx265", "pix_fmt": "yuv420p10le", "rate_mode": "crf", "container": "mkv", "vendor": "cpu"},
    {"id": "libx265_12bit", "label": "HEVC 12-bit (CPU)", "encoder": "libx265", "pix_fmt": "yuv420p12le", "rate_mode": "crf", "container": "mkv", "vendor": "cpu"},
    {"id": "libsvtav1", "label": "AV1 SVT (CPU)", "encoder": "libsvtav1", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "cpu"},
    {"id": "libsvtav1_10bit", "label": "AV1 10-bit (CPU)", "encoder": "libsvtav1", "pix_fmt": "yuv420p10le", "rate_mode": "crf", "container": "mkv", "vendor": "cpu"},
    {"id": "libaom-av1", "label": "AV1 AOM (CPU)", "encoder": "libaom-av1", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "cpu"},
    # --- Lossless / intermedio ---
    {"id": "ffv1", "label": "FFV1 (lossless)", "encoder": "ffv1", "pix_fmt": "yuv420p", "rate_mode": "lossless", "container": "mkv", "mp4_incompatible": True, "vendor": "cpu"},
    {"id": "prores_ks", "label": "ProRes 422", "encoder": "prores_ks", "pix_fmt": "yuv422p10le", "rate_mode": "lossless", "container": "mov", "mp4_incompatible": True, "extra": ["-profile:v", "2"], "vendor": "cpu"},
    {"id": "dnxhr_hq", "label": "DNxHR HQ", "encoder": "dnxhd", "pix_fmt": "yuv422p", "rate_mode": "bitrate", "container": "mkv", "mp4_incompatible": True, "extra": ["-profile:v", "dnxhr_hq"], "probe_size": "1920x1080", "vendor": "cpu"},
    # --- NVIDIA NVENC ---
    {"id": "h264_nvenc", "label": "H.264 NVENC (8-bit)", "encoder": "h264_nvenc", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "nvidia"},
    {"id": "hevc_nvenc", "label": "HEVC NVENC (8-bit)", "encoder": "hevc_nvenc", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "nvidia"},
    {"id": "hevc_nvenc_10bit", "label": "HEVC NVENC (10-bit)", "encoder": "hevc_nvenc", "pix_fmt": "p010le", "rate_mode": "crf", "container": "mkv", "vendor": "nvidia"},
    {"id": "av1_nvenc", "label": "AV1 NVENC (8-bit)", "encoder": "av1_nvenc", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "nvidia"},
    {"id": "av1_nvenc_10bit", "label": "AV1 NVENC (10-bit)", "encoder": "av1_nvenc", "pix_fmt": "p010le", "rate_mode": "crf", "container": "mkv", "vendor": "nvidia"},
    # --- AMD AMF ---
    {"id": "h264_amf", "label": "H.264 AMF", "encoder": "h264_amf", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "amd"},
    {"id": "hevc_amf", "label": "HEVC AMF", "encoder": "hevc_amf", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "amd"},
    {"id": "av1_amf", "label": "AV1 AMF", "encoder": "av1_amf", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "amd"},
    # --- Intel QSV ---
    {"id": "h264_qsv", "label": "H.264 QuickSync", "encoder": "h264_qsv", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "intel"},
    {"id": "hevc_qsv", "label": "HEVC QuickSync", "encoder": "hevc_qsv", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "intel"},
    {"id": "av1_qsv", "label": "AV1 QuickSync", "encoder": "av1_qsv", "pix_fmt": "yuv420p", "rate_mode": "crf", "container": "mp4", "vendor": "intel"},
]

_CODEC_BY_ID = {entry["id"]: entry for entry in VIDEO_CODEC_TABLE}


def get_codec_entry(codec_id):
    """Devuelve la entrada declarativa del codec (o None si no existe)."""
    return _CODEC_BY_ID.get(codec_id)


def get_codec_label(codec_id):
    """Etiqueta legible de un codec (fallback: el propio id)."""
    entry = _CODEC_BY_ID.get(codec_id)
    return entry["label"] if entry else codec_id


def resolve_codec(codec_id):
    """Mapea un ID lógico de codec al encoder real y sus argumentos extra.

    Deriva de la tabla declarativa: aplica el pixel format de salida propio
    del codec (8/10/12 bits) y los args extra declarados (p. ej. el perfil de
    ProRes/DNxHR). Mantiene la firma histórica ``(encoder, extra_args)``.
    """
    entry = _CODEC_BY_ID.get(codec_id)
    if entry is None:
        return (codec_id, [])
    extra = []
    if entry.get("pix_fmt"):
        extra.extend(["-pix_fmt", entry["pix_fmt"]])
    if entry.get("extra"):
        extra.extend(entry["extra"])
    return (entry["encoder"], extra)


def codecs_for_vendor(vendor):
    """Lista ``(id, label)`` de codecs de un vendor ("cpu"/"nvidia"/"amd"/"intel")."""
    return [(e["id"], e["label"]) for e in VIDEO_CODEC_TABLE if e.get("vendor") == vendor]


ALL_KNOWN_CODECS = {
    "CPU": codecs_for_vendor("cpu"),
    "NVIDIA GPU": codecs_for_vendor("nvidia"),
    "AMD GPU": codecs_for_vendor("amd"),
    "Intel GPU": codecs_for_vendor("intel"),
}

FORMATS = [
    ("mp4", "MP4"),
    ("mkv", "MKV"),
    ("avi", "AVI"),
    ("wmv", "WMV"),
    ("mov", "MOV"),
    ("webm", "WebM"),
]

AUDIO_OPTIONS = [
    ("copy", "Copy (original)"),
    ("aac", "AAC"),
    ("mp3", "MP3 - LAME (192k)"),
    ("noaudio", "No Audio"),
]

RESOLUTIONS = [
    ("original", "Original"),
    ("4320", "8K (4320p)"),
    ("2160", "4K (2160p)"),
    ("1440", "2K (1440p)"),
    ("1080", "1080p (FHD)"),
    ("720", "720p (HD)"),
    ("480", "480p (SD)"),
]

# Relación de aspecto de la SALIDA. "source" es lo de siempre: el vídeo
# conserva la proporción con la que se grabó (incluidos los píxeles no
# cuadrados de un DVD anamórfico). Los demás valores obligan a esa proporción:
# la imagen no se deforma y lo que sobra se rellena con barras negras, así que
# la altura elegida en Resolution manda y el ratio da el ancho.
ASPECT_RATIOS = [
    ("source", "Keep source aspect ratio"),
    ("1:1", "1:1 (square)"),
    ("4:3", "4:3"),
    ("16:9", "16:9"),
    ("9:16", "9:16 (vertical)"),
    ("18:9", "18:9"),
]

# URLs de descarga de FFmpeg por defecto (builds estáticas GPL).
# Windows: gyan.dev dejó de publicar builds "full" en zip (ahora solo .7z,
# que nuestro extractor no soporta) y su build "essentials" NO incluye
# libplacebo ni SVT-AV1. Usamos el build estático gpl de BtbN
# (libplacebo + x264/x265 + SVT-AV1 + NVENC/AMF/CUDA, todo en ffmpeg.exe)
# con fallback al essentials de gyan.dev si el primario falla.
FFMPEG_DOWNLOAD_URLS = {
    "Windows": {
        "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "archive_type": "zip",
        "backup_url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "backup_archive_type": "zip",
    },
    "Linux": {
        "url": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        "archive_type": "tar.xz",
    },
}

# Cada cuántas horas se consulta el upstream para chequear actualizaciones.
# El resultado se cachea en ``bin/ffmpeg/upstream_check_cache.json`` para no
# martillear el servidor en cada apertura del panel.
UPSTREAM_CHECK_CACHE_HOURS = 24
