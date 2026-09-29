import json
import logging
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)

from src.config import (
    FFMPEG_DOWNLOAD_URLS,
    ensure_runtime_dirs,
    get_bin_dir,
    get_ffmpeg_dir,
    get_ffmpeg_executable_name,
    get_ffprobe_executable_name,
    get_logs_dir,
)
from src import fsops
from src.ffmpeg.builds import (
    DEFAULT_BUILD_ID,
    build_mirrors,
    build_label,
    descripcion_version,
    get_active_url,
    get_build,
    get_preferred_build_id,
    nothing_to_offer_reason,
)
from src.ffmpeg.downloader import download_ffmpeg
from src.ffmpeg.upstream import (
    get_fingerprint,
    is_update_available,
    read_cache as read_upstream_cache,
)

MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_local_ffmpeg_path():
    """Ruta al binario ffmpeg dentro de la carpeta propia del programa."""
    return os.path.join(get_ffmpeg_dir(), get_ffmpeg_executable_name())


def get_local_ffprobe_path():
    """Ruta al binario ffprobe dentro de la carpeta propia del programa."""
    return os.path.join(get_ffmpeg_dir(), get_ffprobe_executable_name())


def get_manifest_path():
    return os.path.join(get_ffmpeg_dir(), MANIFEST_FILENAME)


def find_system_ffmpeg():
    """Busca ffmpeg en el PATH del sistema (sólo diagnóstico, ya no se usa)."""
    return shutil_which("ffmpeg")


def find_system_ffprobe():
    return shutil_which("ffprobe")


def shutil_which(name):
    try:
        import shutil
        return shutil.which(name)
    except Exception:
        return None


def _emit(progress_callback, percent, message=""):
    """Wrapper fail-safe del callback de progreso.

    Acepta callbacks 1-arg (compat legacy de ``codec_tester``) ó 2-args
    (nuevo contrato de ``download_ffmpeg``). El caller decide qué fase está
    corriendo; aquí sólo nos aseguramos de no propagar excepciones del
    callback al flujo de descarga.
    """
    if not progress_callback:
        return
    try:
        try:
            progress_callback(int(percent), message)
        except TypeError:
            # Compat: callback legacy de 1 argumento.
            progress_callback(int(percent))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Manifest (provenance + version)
# ---------------------------------------------------------------------------

def read_manifest():
    """Lee el manifest.json o devuelve {} si no existe / está corrupto."""
    path = get_manifest_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_manifest(data):
    """Escribe el manifest al disco. Silencioso en caso de error de E/S."""
    ensure_runtime_dirs()
    path = get_manifest_path()
    try:
        payload = dict(data)
        payload.setdefault("id", "ffmpeg")
        payload.setdefault("installed_at", datetime.now().isoformat(timespec="seconds"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"ffmpeg version\s+([^\s]+)", re.IGNORECASE)
# La línea del major va separada por espacios: ffmpeg imprime los números
# alineados ("libavutil      61.  8.100"), no "61.8.100".
_LIBAVUTIL_RE = re.compile(
    r"^libavutil\s+(\d+)\.\s*(\d+)\.\s*(\d+)", re.MULTILINE
)


def _parse_ffmpeg_version_text(text):
    """Extrae la versión del bloque devuelto por ``ffmpeg -version``."""
    if not text:
        return None
    match = _VERSION_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: primera palabra tras el primer salto de línea
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    parts = first_line.split()
    if len(parts) >= 3 and parts[0].lower() == "ffmpeg":
        return parts[2]
    return None


def parse_ffmpeg_build(text):
    """Lee QUÉ BUILD ES lo que hay instalado, según su ``-version``.

    Dos hechos, y no uno. El primero es el token que imprime ffmpeg
    (``9.0.2`` en una release, ``N-126905-gb87602a63a-20260927`` en una
    nightly de master), que identifica la compilación pero NO dice la
    versión: es un ``git describe``. El segundo es el major de libavutil,
    que es lo que de verdad sitúa esa build en una línea de FFmpeg (9.x,
    10.x, ...) y lo que el usuario quiere leer.

    Devuelve ``{"version": ..., "libavutil_major": ...}`` o ``None`` si del
    texto no se puede sacar una versión. Lo interpreta ``builds``; aquí sólo
    se lee.
    """
    version = _parse_ffmpeg_version_text(text)
    if not version:
        return None
    match = _LIBAVUTIL_RE.search(text or "")
    return {
        "version": version,
        "libavutil_major": int(match.group(1)) if match else None,
    }


def get_ffmpeg_build_info(ffmpeg_path=None):
    """Ejecuta ``-version`` UNA vez y devuelve qué build es la instalada.

    Único sitio que lanza el binario para leer su versión: lo que salga de
    ahí pasa por :func:`parse_ffmpeg_build`, así que el token y el major se
    leen siempre del mismo sitio y no pueden discrepar entre sí.
    """
    path = ffmpeg_path or get_local_ffmpeg_path()
    if not path or not os.path.exists(path):
        return None
    try:
        result = subprocess.run(
            [path, "-version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        return parse_ffmpeg_build(text)
    except Exception:
        return None


def get_ffmpeg_version(ffmpeg_path=None):
    """La cadena de versión tal cual la imprime el binario.

    Capa fina sobre :func:`get_ffmpeg_build_info`: mismo valor, mismo
    ``None`` si el binario falta o no responde, y sin una segunda vuelta al
    proceso. Lo que el usuario LEE de esa versión lo compone ``builds``.
    """
    info = get_ffmpeg_build_info(ffmpeg_path)
    return info["version"] if info else None


def get_install_info(ffmpeg_path=None):
    """Devuelve un dict con el estado actual de la instalación de FFmpeg.

    ``build_id`` / ``build_label`` describen lo INSTALADO (se leen del manifest).
    La comparación con el upstream, en cambio, es contra la build ELEGIDA: es la
    que instalarían el botón y el próximo arranque, así que un "Actualizar"
    nunca puede anunciar una build y descargar otra.
    """
    ffmpeg = ffmpeg_path or get_local_ffmpeg_path()
    ffprobe = get_local_ffprobe_path()
    manifest = read_manifest()
    # Qué versión se instaló (el manifest lo guarda desde que el usuario puede
    # elegirla; las instalaciones antiguas, o las traídas a mano, se leen como
    # la de siempre, que es lo que hay en bin/ffmpeg).
    build_id = str(manifest.get("build_id") or DEFAULT_BUILD_ID)
    build = get_build(build_id)
    binary_missing = not (ffmpeg and os.path.exists(ffmpeg))
    # Una sola vuelta al binario y se guardan las DOS cosas que hacen falta
    # para decir qué versión es: el token y el major de libavutil. Si el
    # binario no responde queda ``build_info`` a None y se usa lo que
    # guardó el manifest, como hasta ahora.
    build_info = None if binary_missing else get_ffmpeg_build_info(ffmpeg)
    version = (build_info or {}).get("version") or manifest.get("version")
    if binary_missing:
        status = "missing"
    elif version is None:
        status = "error"
    else:
        status = "installed"

    # Comprobar contra el upstream (lee cache, sin red). El panel se encarga de
    # refrescar la cache cada 24h desde su QTimer; aquí solo leemos.
    #
    # La huella sólo sirve para la build INSTALADA: si el cache pertenece a otra
    # URL (el usuario cambió de versión y aún no se ha comprobado la nueva) no
    # son comparables y devolvemos None ("no lo sabemos") en vez de inventarse
    # un "update available".
    fingerprint = read_upstream_cache()
    active_url = get_active_url(platform.system())
    if fingerprint and active_url and fingerprint.get("url") != active_url:
        fingerprint = {}
    update_available = is_update_available(manifest, fingerprint)

    return {
        "id": "ffmpeg",
        "display_name": "FFmpeg (encoder suite)",
        "kind": "binary",
        "status": status,
        "version": version or "unknown",
        # Lo que el usuario LEE de esa versión: "9.0.2" en una release,
        # "9.x-dev (master)" en una nightly. Lo compone ``builds`` porque
        # es el único que sabe qué major de libavutil es qué línea; la fila
        # sólo lo pinta. Sin major conocido vuelve a ser el token.
        "version_text": descripcion_version(build_info) or version or "unknown",
        "install_dir": get_ffmpeg_dir(),
        "ffmpeg_path": ffmpeg,
        "ffprobe_path": ffprobe if os.path.exists(ffprobe) else None,
        "manifest_path": get_manifest_path(),
        "source_url": manifest.get("source_url"),
        "installed_at": manifest.get("installed_at"),
        "build_id": build["id"],
        # El rótulo lo compone el catálogo (``builds.build_label``), que
        # es quien sabe si ese id es una build real: si no lo es, el rótulo
        # avisa en vez de estar nombrando otra build sin decirlo.
        "build_label": build_label(build_id),
        # Por qué aquí no se puede instalar, o ``None`` si sí. Viaja en los
        # datos para que la fila no tenga que averiguarlo por su cuenta: sin
        # mirrors el botón de instalar se ofrece muerto y con el motivo a la
        # vista, en vez de vivo y prometiendo una descarga que no existe.
        "nothing_to_install": nothing_to_offer_reason(platform.system()),
        "preferred_build_id": get_preferred_build_id(),
        "system_path": find_system_ffmpeg(),
        "ffmpeg_exists": not binary_missing,
        "ffprobe_exists": os.path.exists(ffprobe) if ffprobe else False,
        # Datos de upstream (para la columna 'Latest' / 'Update available')
        "latest_fingerprint": fingerprint,
        "update_available": update_available,
        "last_check_iso": fingerprint.get("checked_at"),
        "upstream_error": fingerprint.get("last_error"),
    }


# ---------------------------------------------------------------------------
# Instalación / re-instalación
# ---------------------------------------------------------------------------

def _build_install_manifest(ffmpeg_path, source_label):
    """Construye un payload de manifest con la huella upstream en el momento
    de la instalación. Si la red está caída seguimos siendo capaces de escribir
    un manifest válido con sólo los campos locales.

    La build instalada es, por definición, la que estaba elegida: el manifest
    guarda su id, la URL de la que salió y la huella *de esa URL*, para que el
    chequeo de actualizaciones compare contra el upstream correcto.

    ``get_preferred_build_id()`` se relee aquí, y no se le pasa la build desde
    el que llama, a propósito: ninguno de los tres llamantes la tiene (los tres
    pasan por ``download_ffmpeg``, que la resuelve por su cuenta), y pasarla
    añadiría un segundo camino por el que el id viaja en vez de quitar una
    lectura. Es la misma preferencia, leída por su dueño, en la misma pila y en
    el mismo instante que la descarga.
    """
    ensure_runtime_dirs()
    build = get_build(get_preferred_build_id())
    mirrors = build_mirrors(build["id"], platform.system())
    source_url, archive_type = mirrors[0] if mirrors else (None, None)
    # Tomamos una huella fresca para saber qué hay hoy en el upstream; si esto
    # falla (sin red) seguimos con el cache anterior o un dict vacío.
    try:
        fp = get_fingerprint(force=True)
    except Exception:
        fp = read_upstream_cache() or {}
    if fp and source_url and fp.get("url") != source_url:
        # La huella es de otra URL: no nos sirve para comparar.
        fp = {}

    payload = {
        "id": "ffmpeg",
        "version": get_ffmpeg_version(ffmpeg_path) or "unknown",
        "build_id": build["id"],
        # OJO: aquí el rótulo NO es ``builds.build_label``. A propósito. El
        # manifest graba lo que se instaló de verdad, y lo que se instaló es
        # ``build["id"]``, ya saneado: si la preferencia estaba corrupta o no era
        # instalable aquí, aquí se escribió la build de siempre. Ponerle el
        # texto de aviso ("la versión guardada 'X' no está en el catálogo")
        # dentro del manifest contaminaría un registro de hechos con una
        # explicación dirigida al usuario, que es cosa de la interfaz
        # (``builds.chosen_build_label``), no del dato. Por eso aquí basta el
        # nombre plano de lo instalado.
        "build_label": build.get("label") or build["id"],
        "source_url": source_url,
        "archive_type": archive_type,
        "source": source_label,
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "upstream_last_modified": fp.get("last_modified_iso"),
        "upstream_content_length": fp.get("content_length"),
        "upstream_etag": fp.get("etag"),
        "upstream_url": fp.get("url") or source_url,
    }
    return payload


def _migrate_legacy_layout():
    """Migra el layout plano antiguo (``bin/ffmpeg.exe``) al nuevo (``bin/ffmpeg/...``).

    Estrategia copy → verify → delete para que un fallo a mitad de camino
    deje los originales intactos y nunca perdamos una instalación manual del
    usuario. El manifest sólo se escribe si AMBOS binarios están en la nueva
    ruta; en caso contrario la migración se aborta sin tocar nada, dejando
    que ``ensure_ffmpeg`` caiga al camino de descarga limpio.
    """
    try:
        old_ffmpeg = os.path.join(get_bin_dir(), get_ffmpeg_executable_name())
        old_ffprobe = os.path.join(get_bin_dir(), get_ffprobe_executable_name())
        new_ffmpeg = get_local_ffmpeg_path()
        new_ffprobe = get_local_ffprobe_path()
        # Idempotente: si el nuevo layout ya tiene ffmpeg o ffprobe, no tocar.
        if (
            not os.path.exists(old_ffmpeg)
            or os.path.exists(new_ffmpeg)
            or os.path.exists(new_ffprobe)
        ):
            return False
        ensure_runtime_dirs()

        # 1) Copiar ambos a la nueva ubicación
        shutil.copy2(old_ffmpeg, new_ffmpeg)
        if os.path.exists(old_ffprobe):
            shutil.copy2(old_ffprobe, new_ffprobe)

        # 2) Verificar que la copia aterrizó antes de borrar nada
        if not (os.path.exists(new_ffmpeg) and os.path.exists(new_ffprobe)):
            return False

        # 3) Manifest sólo si la copia es completa, con huella upstream
        payload = _build_install_manifest(new_ffmpeg, "migrated-from-flat-layout")
        wrote = write_manifest(payload)
        if not wrote:
            return False

        # 4) Borrar originales (best-effort; el estado "nuevo" ya es válido)
        for p in (old_ffmpeg, old_ffprobe):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

        # 5) Permisos Linux
        if platform.system() != "Windows":
            for p in (new_ffmpeg, new_ffprobe):
                try:
                    if os.path.exists(p):
                        os.chmod(p, 0o755)
                except OSError:
                    pass
        return True
    except Exception as exc:
        logger.warning("Legacy FFmpeg migration failed: %s", exc)
        return False


def ensure_ffmpeg(parent=None, progress_callback=None,
                  abort_event=None, timeout=None):
    """Asegura que FFmpeg esté disponible dentro de la carpeta del programa.

    Args:
        parent: QWidget padre (futuras integraciones con QFileDialog).
        progress_callback: callable(percent, message) -> None. Recibe
            updates durante la descarga y la extracción. Compatible con
            callbacks de 1 argumento vía ``_emit``.
        abort_event: threading.Event opcional para cancelar la descarga.
        timeout: presupuesto total de pared en segundos (opcional).

    Qué versión se descarga lo decide la preferencia ``ffmpeg_build_id``:
    la elegida en el panel de dependencias, o la de siempre si nunca se
    eligió ninguna (el comportamiento de toda la vida).

    Reglas (alineadas con el requisito del usuario):
      1. Si existe el binario local -> se usa, sin fallback al sistema.
      2. Si no existe -> se descarga automáticamente a ``bin/ffmpeg/``.
      3. No se consulta nunca el PATH del sistema para el flujo normal; sólo
         se expone a través de ``get_install_info()`` como diagnóstico.

    Devuelve la ruta al ejecutable o ``None`` si no se pudo obtener.
    """
    # Migración puntual del layout antiguo (bin/ffmpeg.exe -> bin/ffmpeg/ffmpeg.exe)
    _migrate_legacy_layout()

    local = get_local_ffmpeg_path()
    if os.path.exists(local) and os.path.exists(get_local_ffprobe_path()):
        # Refrescar manifest si no existe (binario traído manualmente).
        if not os.path.exists(get_manifest_path()):
            write_manifest(_manifest_from_binary(local))
        _emit(progress_callback, 100, "FFmpeg ready (cached)")
        return local

    ensure_runtime_dirs()
    try:
        path = download_ffmpeg(
            parent=parent, progress_callback=progress_callback,
            abort_event=abort_event, timeout=timeout,
        )
    except Exception as exc:
        # Registrar el error en bin/logs/ffmpeg_download.log para diagnóstico.
        try:
            log_dir = get_logs_dir()
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "ffmpeg_download.log"), "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] ensure_ffmpeg download failed: {exc}\n")
        except Exception:
            pass
        logger.error("Error descargando FFmpeg: %s", exc)
        return None
    # Escribir manifest con datos de la descarga + huella upstream
    write_manifest(_build_install_manifest(path, "first-download"))
    # ``download_ffmpeg`` ya emite 100%/Ready al terminar; no re-emitimos.
    return path


def _install_staged(staging_dir, ffmpeg_dir, names):
    """Pone en su sitio lo descargado sin dejar la instalación a medias.

    Se aparta cada archivo viejo con un nombre temporal y sólo después se mueve
    el nuevo encima; si algo falla se deshace y se relanza. No se borra nada sin
    copia: hasta el último ``os.replace`` hay una instalación completa en
    ``ffmpeg_dir``. Los archivos que el staging no trae (un ffprobe que no venga
    en el paquete, p. ej.) se dejan como estaban.
    """
    replaced = []  # (destino, copia del anterior, había anterior)
    try:
        for name in names:
            new_path = os.path.join(staging_dir, name)
            if not os.path.exists(new_path):
                continue
            target = os.path.join(ffmpeg_dir, name)
            backup = target + ".old"
            had_old = os.path.exists(target)
            if had_old:
                # Si el archivo está en uso (Windows) falla aquí, y aún no se ha
                # tocado nada: la instalación actual sigue intacta.
                os.replace(target, backup)
            replaced.append((target, backup, had_old))
            os.replace(new_path, target)
    except Exception:
        for target, backup, had_old in reversed(replaced):
            try:
                if os.path.exists(target):
                    os.remove(target)
            except OSError:
                pass
            if had_old and os.path.exists(backup):
                try:
                    os.replace(backup, target)
                except OSError:
                    pass
        raise
    finally:
        for _target, backup, _had_old in replaced:
            try:
                if os.path.exists(backup):
                    os.remove(backup)
            except OSError:
                pass


def force_redownload(progress_callback=None, parent=None, abort_event=None,
                     timeout=None):
    """Reinstala la build elegida (o la de siempre si nunca se eligió ninguna).

    La reinstalación no puede dejar el programa peor que como estaba: se descarga
    a un staging aparte, se comprueba que el binario nuevo arranca y sólo entonces
    se sustituye el anterior. Si la descarga falla, se agota el ``timeout`` o el
    usuario cancela, la instalación que ya funcionaba sigue ahí, con su manifiesto.

    Lanza ``RuntimeError`` si no se pudo dejar la nueva en su sitio (el caller
    debe traducirlo al UI). ``abort_event`` cancela la descarga; ``timeout``
    acota el tiempo total.
    """
    ffmpeg_dir = get_ffmpeg_dir()
    ensure_runtime_dirs()
    staging = os.path.join(ffmpeg_dir, ".staging-{}".format(os.getpid()))
    try:
        staged_ffmpeg = download_ffmpeg(
            progress_callback=progress_callback, parent=parent,
            abort_event=abort_event, timeout=timeout, dest_dir=staging,
        )
        if not staged_ffmpeg or not os.path.exists(staged_ffmpeg):
            raise RuntimeError("Re-download finished but FFmpeg binary is missing.")
        # Un binario corrupto o incompleto no llega a sustituir al que funciona.
        from src.ffmpeg.runner import health_check_ffmpeg
        ok, _version, error = health_check_ffmpeg(staged_ffmpeg)
        if not ok:
            raise RuntimeError(
                "La build descargada no arranca en este equipo: {}"
                .format(error or "health check fallido")
            )
        payload = _build_install_manifest(staged_ffmpeg, "user-reinstall")
        with open(os.path.join(staging, MANIFEST_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        _install_staged(staging, ffmpeg_dir, (
            get_ffmpeg_executable_name(), get_ffprobe_executable_name(),
            MANIFEST_FILENAME,
        ))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return os.path.join(ffmpeg_dir, get_ffmpeg_executable_name())


def _manifest_from_binary(ffmpeg_path):
    """Genera un manifest mínimo cuando el binario ya estaba presente sin manifest."""
    system = platform.system()
    info = FFMPEG_DOWNLOAD_URLS.get(system, {})
    return {
        "id": "ffmpeg",
        "version": get_ffmpeg_version(ffmpeg_path) or "unknown",
        "build_id": DEFAULT_BUILD_ID,
        "source_url": info.get("url"),
        "source": "manual-copy",
    }


def uninstall(parent=None):
    """Borra la instalación local de FFmpeg (binarios + manifest + cache).

    La app lo re-descarga automáticamente en el próximo arranque
    (``app.py`` → ``ensure_ffmpeg``). Si algún archivo está bloqueado
    (render en curso), ``fsops.rmtree_force`` lo programa para
    el próximo arranque.
    """
    ffmpeg_dir = get_ffmpeg_dir()
    if os.path.isdir(ffmpeg_dir):
        fsops.rmtree_force(ffmpeg_dir)


# ---------------------------------------------------------------------------
# Compatibilidad con código existente
# ---------------------------------------------------------------------------

def get_ffprobe_path():
    """Devuelve la ruta a ffprobe local; busca en el dir de ffmpeg y en PATH."""
    # 1. Intentar con el ffmpeg estándar (bin/ffmpeg/ffprobe.exe)
    local = get_local_ffprobe_path()
    if os.path.exists(local):
        return local
    # 2. Último recurso: buscar en PATH del sistema
    system = find_system_ffprobe()
    if system:
        return system
    return None
