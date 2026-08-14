import os
import re
import subprocess
import platform
import sys


def is_windows_elevated():
    """True si el proceso corre elevado (como administrador) en Windows.

    Con UAC activo, un administrador que lanza el proceso SIN elevación
    recibe un token filtrado e ``IsUserAnAdmin`` devuelve False; solo
    devuelve True cuando el proceso tiene privilegios reales de
    administrador, que es exactamente el caso en el que UIPI bloquea el
    drag & drop desde el Explorador hacia la app. En otras plataformas
    devuelve False.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def detect_linux_container():
    """Detecta si la app corre empaquetada en Flatpak o como AppImage.

    Devuelve ``"flatpak"``, ``"appimage"`` o ``None``. Flatpak SI es un
    sandbox: los arrastres de archivos desde el gestor del host no cruzan
    la frontera del sandbox (drag & drop de videos limitado). AppImage NO
    está aislado (se monta y ejecuta sin sandbox), por lo que el drag &
    drop funciona con normalidad; aun así se detecta para poder distinguir
    ambos modos de empaquetado.
    """
    if (
        os.environ.get("FLATPAK_ID")
        or os.environ.get("FLATPAK_SANDBOX_DIR")
        or os.path.exists("/.flatpak-info")
    ):
        return "flatpak"
    if os.environ.get("APPIMAGE") or os.environ.get("APPDIR"):
        return "appimage"
    return None


def run_command(cmd, timeout=30, capture_output=True):
    """Ejecuta un comando y devuelve (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            timeout=timeout,
            shell=True,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def parse_time_to_seconds(time_str):
    """Convierte HH:MM:SS o MM:SS a segundos."""
    parts = time_str.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:
        return parts[0]
    return 0.0


def seconds_to_time(seconds):
    """Convierte segundos a HH:MM:SS."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_size(size_bytes):
    """Formatea bytes a KB/MB/GB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
