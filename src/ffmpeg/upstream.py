"""Comprobación periódica del upstream de FFmpeg para detectar builds nuevas.

Estrategia:
- Hacemos un HEAD sobre la URL de descarga para extraer ``Last-Modified``,
  ``Content-Length`` y ``ETag``. Esos tres campos, juntos, forman la huella
  que usamos para comparar lo que tenemos instalado con lo que ofrece el
  upstream hoy.
- Cacheamos el resultado en ``bin/ffmpeg/upstream_check_cache.json`` con
  ``UPSTREAM_CHECK_CACHE_HOURS`` de frescura. Si el cache es reciente no
  volvemos a martillear el upstream.
- Cuando el usuario reinstala, ``force_redownload()`` toma una huella fresca
  (``force=True``) y la graba en ``manifest.json`` junto a la versión local.
- Cualquier fallo de red preserva el cache anterior con un campo
  ``last_error`` para diagnóstico, en lugar de romper el panel.

Comparación:
- ``is_update_available(manifest, fingerprint)`` devuelve True/False/None.
  None significa "no tengo datos suficientes para decidir" (cache + red
  fallaron a la vez); el panel entonces muestra "—" en lugar de "Update
  available".
"""

from __future__ import annotations

import json
import os
import platform as _platform
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from src.config import (
    FFMPEG_DOWNLOAD_URLS,
    UPSTREAM_CHECK_CACHE_HOURS,
    ensure_runtime_dirs,
    get_ffmpeg_dir,
)


CACHE_FILENAME = "upstream_check_cache.json"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path():
    return os.path.join(get_ffmpeg_dir(), CACHE_FILENAME)


def read_cache():
    """Devuelve el contenido cacheado o ``{}`` si no existe / está corrupto."""
    try:
        with open(_cache_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_cache(data):
    """Persiste el cache. Devuelve True si escribió, False en error de E/S."""
    try:
        ensure_runtime_dirs()
        with open(_cache_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def clear_cache():
    """Borra el cache. Llamar tras una instalación/actualización exitosa."""
    try:
        path = _cache_path()
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_http_date(value):
    """Parse RFC 1123 / 850 / asctime date. Returns aware datetime or None."""
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _cache_age_hours(cache):
    cached_at = cache.get("checked_at")
    if not cached_at:
        return float("inf")
    try:
        dt = datetime.fromisoformat(cached_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return float("inf")


def _iso(dt):
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Red
# ---------------------------------------------------------------------------

_USER_AGENT = "DRAGGY-ENCODER-UpstreamMonitor/1.0"


def _head(url, timeout=15):
    """HEAD con fallback a GET-stream-1 si el servidor rechaza HEAD.

    Ante un fallo de red (timeout, DNS, conexión rehusada...) devuelve un
    objeto mínimo con ``status_code=0`` y ``text`` con la causa accionable.
    """
    headers = {"User-Agent": _USER_AGENT}

    class _ErrorResponse:
        status_code = 0
        headers = {}
        text = ""

    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=timeout, headers=headers
        )
        if resp.status_code in (403, 405, 501):
            # Algunos servidores no implementan HEAD; recogemos solo headers.
            resp = requests.get(
                url, allow_redirects=True, timeout=timeout,
                headers=headers, stream=True,
            )
            try:
                resp.close()
            except Exception:
                pass
        return resp
    except Exception as exc:
        err = _ErrorResponse()
        err.text = str(exc) or exc.__class__.__name__
        return err


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def _platform_name(name=None):
    return name or _platform.system()


def get_active_url(platform_name=None):
    info = FFMPEG_DOWNLOAD_URLS.get(_platform_name(platform_name))
    return info.get("url") if info else None


def get_fingerprint(platform_name=None, force=False):
    """Devuelve la huella upstream (dict) para la plataforma actual.

    Si el cache tiene menos de ``UPSTREAM_CHECK_CACHE_HOURS`` se devuelve
    directamente. Si no, se hace un HEAD fresco y se actualiza el cache.
    En caso de fallo de red se devuelve el último cache con ``last_error``.

    Política de freshness ante errores: si la última ejecución terminó con
    ``last_error`` y fue hace menos de 1 hora, conservamos el cache para
    evitar martillear al upstream durante un microcorte. Pasada esa hora,
    reintentamos.
    """
    ensure_runtime_dirs()
    url = get_active_url(platform_name)
    if not url:
        return {}

    cache = read_cache()
    age = _cache_age_hours(cache)
    if (
        not force
        and cache.get("url") == url
        and age < UPSTREAM_CHECK_CACHE_HOURS
        and not cache.get("last_error")
    ):
        return cache
    if (
        not force
        and cache.get("url") == url
        and cache.get("last_error")
        and age < 1.0
    ):
        return cache

    resp = _head(url)
    if resp.status_code >= 400:
        # Red caída / upstream caído → conservar cache, anotar error con
        # mensaje accionable (excepción subyacente si la hay).
        cache["url"] = url
        cache["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        reason = getattr(resp, "text", "") or ""
        if not reason.strip():
            reason = f"HEAD failed (status={getattr(resp, 'status_code', 'no-response')})"
        cache["last_error"] = reason
        write_cache(cache)
        return cache

    last_modified_hdr = resp.headers.get("Last-Modified")
    last_modified_dt = _parse_http_date(last_modified_hdr)
    content_length_hdr = resp.headers.get("Content-Length")
    fingerprint = {
        "url": url,
        "last_modified_header": last_modified_hdr,
        "last_modified_iso": _iso(last_modified_dt) if last_modified_dt else None,
        "content_length": int(content_length_hdr) if content_length_hdr else None,
        "etag": resp.headers.get("ETag"),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_cache(fingerprint)
    return fingerprint


# ---------------------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------------------

def is_update_available(local_manifest, fingerprint):
    """True si la build local parece más antigua que la del upstream.

    Criterios (cualquiera basta para declarar True):
      - ``etag`` difiere (identidad semántica del contenido).
      - ``content_length`` difiere (rebuild típico cambia tamaño aunque el
        upstream_revisión_hash no esté disponible).
      - ``last_modified`` del upstream es estrictamente posterior al nuestro.

    Devuelve ``None`` si no hay datos suficientes (cache vacío + red caída).
    """
    fingerprint = fingerprint or {}
    stored_lm = (local_manifest or {}).get("upstream_last_modified")
    stored_cl = (local_manifest or {}).get("upstream_content_length")
    stored_etag = (local_manifest or {}).get("upstream_etag")

    upstream_lm = fingerprint.get("last_modified_iso")
    upstream_cl = fingerprint.get("content_length")
    upstream_etag = fingerprint.get("etag")

    # Sin datos upstream → no podemos decidir
    if not (upstream_lm or upstream_cl or upstream_etag):
        return None

    # Sin huella de instalación Y tenemos huella upstream → conservador: Sí,
    # salvo que sepamos que el upstream estuvo caído (en ese caso no mentimos).
    if stored_lm is None and stored_cl is None and stored_etag is None:
        if fingerprint.get("last_error"):
            return None  # network down + sin referencia → no sabemos
        return True

    # ETag / Content-Length arriba del todo: si la build se reconstruyó con
    # el mismo timestamp (poco habitual pero posible) estos dos son los
    # discriminantes primarios.
    if (
        stored_etag
        and upstream_etag
        and stored_etag != upstream_etag
    ):
        return True
    if (
        stored_cl is not None
        and upstream_cl is not None
        and stored_cl != upstream_cl
    ):
        return True

    # Sólo después de los anteriores miramos fechas: upstream estricto más
    # reciente. Si es más antiguo (rollback) sin cambios de tamaño/etag,
    # mantenemos el False.
    if stored_lm and upstream_lm:
        try:
            def _parse(iso):
                if iso.endswith("Z"):
                    iso = iso[:-1] + "+00:00"
                return datetime.fromisoformat(iso)
            delta = _parse(upstream_lm) - _parse(stored_lm)
            if delta.total_seconds() > 0:
                return True
        except Exception:
            pass

    return False


def freshness_label(cache):
    """Texto humano para mostrar junto al valor 'Last check: …' del panel."""
    age = _cache_age_hours(cache)
    if age == float("inf"):
        return "never"
    if age < 1:
        return f"{int(age * 60)} min ago"
    if age < 24:
        return f"{int(age)} h ago"
    return f"{int(age / 24)} d ago"
