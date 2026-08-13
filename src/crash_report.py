"""Generación de issues de GitHub pre-rellenados para reportar crashes.

El flag ``--crash-report`` abre un issue de GitHub con el traceback y el
final de ``app.log`` y ``faulthandler.log`` ya pegados en el body. GitHub no
admite adjuntos vía URL, así que el texto se incluye dentro del body y se
recorta a un tamaño acotado para no reventar el límite de longitud de la URL.

Este módulo solo importa ``os/sys/platform`` y ``src.config`` (que a su vez
solo usa os/sys/platform), así que es seguro usarlo desde el ``sys.excepthook``
incluso cuando PySide6 todavía no ha podido importarse.
"""

from __future__ import annotations

import os
import platform
import sys
import urllib.parse
import webbrowser

from src.config import APP_VERSION, NEW_ISSUE_URL, get_logs_dir

# Límites del body. Cuanto más texto, más crece la URL (urllib.quote ~3x los
# bytes no-ASCII); estos valores dejan el body en ~8-10 KB, aceptable para el
# endpoint /issues/new sin arriesgar un 414.
_APP_LOG_TAIL = 4000
_FAULTHANDLER_LOG_TAIL = 2000


def _read_tail(path, max_chars):
    """Lee los últimos ``max_chars`` caracteres de ``path``.

    Devuelve ``""`` si el fichero no existe o no se puede leer. Lee en modo
    binario y decodifica de forma tolerante para no fallar con logs a medio
    escribir (utf-8 roto tras un corte de energía).
    """
    try:
        if not path or not os.path.isfile(path):
            return ""
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > max_chars:
                fh.seek(size - max_chars)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _logs_dir_paths():
    logs_dir = get_logs_dir()
    return {
        "dir": logs_dir,
        "app_log": os.path.join(logs_dir, "app.log"),
        "faulthandler_log": os.path.join(logs_dir, "faulthandler.log"),
    }


def _system_context():
    return (
        "- OS: {os_name}\n"
        "- Python: {python} ({exe})\n"
        "- Frozen (PyInstaller): {frozen}\n"
        "- App version: {version}\n"
    ).format(
        os_name=platform.platform(),
        python=sys.version.split()[0],
        exe=sys.executable or "?",
        frozen=bool(getattr(sys, "frozen", False)),
        version=APP_VERSION,
    )


def _summary_line(exc_text):
    """Última línea no vacía del traceback, recortada, para el título."""
    for line in reversed((exc_text or "").splitlines()):
        line = line.strip()
        if line:
            return line[:120]
    return ""


def build_issue_url(exc_text=""):
    """Devuelve la URL del issue pre-rellenado (no abre nada)."""
    paths = _logs_dir_paths()
    app_tail = _read_tail(paths["app_log"], _APP_LOG_TAIL)
    faul_tail = _read_tail(paths["faulthandler_log"], _FAULTHANDLER_LOG_TAIL)

    summary = _summary_line(exc_text)
    title = f"Crash report{f': {summary}' if summary else ''}"

    sections = [
        "## What happened",
        "",
        "```",
        (exc_text or "_(no exception traceback provided)_").strip(),
        "```",
        "",
        "## System",
        "",
        _system_context(),
        "",
        "## Logs (tail)",
        "",
        f"Logs directory: `{paths['dir']}`",
        "",
    ]
    if app_tail:
        sections += ["### app.log", "", "```", app_tail, "```", ""]
    if faul_tail:
        sections += ["### faulthandler.log", "", "```", faul_tail, "```", ""]
    if not app_tail and not faul_tail:
        sections += ["_(no log files found yet)_", ""]

    body = "\n".join(sections)
    query = urllib.parse.urlencode({"title": title, "body": body})
    return f"{NEW_ISSUE_URL}?{query}"


def open_crash_report(exc_text=""):
    """Abre el issue pre-rellenado en el navegador y devuelve la URL.

    Nunca lanza: si el navegador no se puede abrir (headless, sin display) o
    stderr no existe (EXE windowed), imprime la URL a stderr como fallback y
    el llamador igualmente recibe la URL para mostrarla/copiarla.
    """
    url = build_issue_url(exc_text)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        print(url, file=sys.stderr)
    except Exception:
        pass
    return url
