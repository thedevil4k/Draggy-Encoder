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


def get_ffmpeg_full_dir():
    """Subcarpeta dedicada a FFmpeg completo con libplacebo (FSR 3.1)."""
    return os.path.join(get_bin_dir(), "ffmpeg-full")


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
        get_ffmpeg_full_dir(),
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


# Lista maestra de codecs agrupados por familia de dispositivo.
# Se usa tanto para la detección de hardware como para probar compatibilidad con FFmpeg.
def resolve_codec(codec_id):
    """Mapea un ID lógico de codec al encoder real de FFmpeg y sus argumentos extra."""
    mapping = {
        "hevc_nvenc_10bit": ("hevc_nvenc", ["-pix_fmt", "p010le"]),
        "av1_nvenc_10bit": ("av1_nvenc", ["-pix_fmt", "p010le"]),
    }
    return mapping.get(codec_id, (codec_id, []))


ALL_KNOWN_CODECS = {
    "CPU": [
        ("libx264", "H.264 (CPU)"),
        ("libx265", "HEVC (CPU)"),
        ("libsvtav1", "AV1 SVT (CPU)"),
        ("libaom-av1", "AV1 AOM (CPU)"),
    ],
    "NVIDIA GPU": [
        ("h264_nvenc", "H.264 NVENC (8-bit)"),
        ("hevc_nvenc", "HEVC NVENC (8-bit)"),
        ("hevc_nvenc_10bit", "HEVC NVENC (10-bit)"),
        ("av1_nvenc", "AV1 NVENC (8-bit)"),
        ("av1_nvenc_10bit", "AV1 NVENC (10-bit)"),
    ],
    "AMD GPU": [
        ("h264_amf", "H.264 AMF"),
        ("hevc_amf", "HEVC AMF"),
        ("av1_amf", "AV1 AMF"),
    ],
    "Intel GPU": [
        ("h264_qsv", "H.264 QuickSync"),
        ("hevc_qsv", "HEVC QuickSync"),
        ("av1_qsv", "AV1 QuickSync"),
    ],
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

# URLs de descarga de FFmpeg (builds estáticas)
FFMPEG_DOWNLOAD_URLS = {
    "Windows": {
        "url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "archive_type": "zip",
    },
    "Linux": {
        "url": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        "archive_type": "tar.xz",
    },
}

# URLs de descarga de FFmpeg completo con libplacebo (FSR 3.1 support)
# libplacebo is LGPL so it's included in BOTH gpl and lgpl BtbN variants.
# Strategy: try smaller lgpl-shared first, then larger gpl-shared, then johnvansickle.com.
FFMPEG_FULL_DOWNLOAD_URLS = {
    "Windows": {
        # Primary: BtbN lgpl-shared ~67 MB (libplacebo is LGPL, so it's included)
        "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip",
        "archive_type": "zip",
        # Backup 1: BtbN gpl-shared ~76 MB (full GPL set, libplacebo present)
        "backup_url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip",
        "backup_archive_type": "zip",
        # Backup 2: johnvansickle.com static ~42 MB (libplacebo enabled)
        "backup_url_2": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        "backup_archive_type_2": "tar.xz",
    },
    "Linux": {
        # Primary: BtbN lgpl-shared ~52 MB
        "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl-shared.tar.xz",
        "archive_type": "tar.xz",
        # Backup 1: BtbN gpl-shared ~57 MB
        "backup_url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl-shared.tar.xz",
        "backup_archive_type": "tar.xz",
        # Backup 2: johnvansickle.com static ~42 MB
        "backup_url_2": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        "backup_archive_type_2": "tar.xz",
    },
}

# Cada cuántas horas se consulta el upstream para chequear actualizaciones.
# El resultado se cachea en ``bin/ffmpeg/upstream_check_cache.json`` para no
# martillear el servidor en cada apertura del panel.
UPSTREAM_CHECK_CACHE_HOURS = 24
