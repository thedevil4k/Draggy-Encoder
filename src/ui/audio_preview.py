"""Vista previa audible de los efectos (escuchar mientras se ajusta).

Genera un fragmento corto del audio de la pista elegida CON los ajustes
actuales aplicados y lo reproduce con QtMultimedia. Ese es el bucle que hace
que ajustar la compresión se sienta como un plugin de audio: mueves, escuchas,
vuelves a mover.

Decisiones de diseño:

* El clip se genera con ``ffmpeg`` aplicando exactamente la misma cadena que
  usará el render (``audio_fx.build_effects_chain``), midiendo antes el nivel
  del fragmento. Así lo que se oye es lo que se va a obtener, no una
  aproximación.
* Se sintetiza a AAC para que sea rápido y para incluir el efecto del
  codificador en el pico, igual que en la salida final.
* Un solo objeto de reproducción reutilizado: cada ajuste reemplaza el clip
  anterior, y se cancela cualquier generación en curso para que arrastrar un
  control no acumule procesos de ffmpeg.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time

from PySide6.QtCore import QObject, QUrl, Signal

from src.ffmpeg import audio_fx
from src.ffmpeg.loudnorm import measure_loudness
from src.ffmpeg.process_supervisor import process_group_popen_kwargs
from src.ffmpeg.runner import (
    LOUDNORM_CLAVE_LOUDNESS,
    _parse_time_to_seconds,
    _run_kwargs,
)

logger = logging.getLogger(__name__)


# Duración del fragmento de escucha. Corto para que el ciclo ajustar-escuchar
# sea rápido, largo suficiente para juzgar la compresión (la compresión se
# aprecia en la dinámica, y eso necesita unos segundos de material).
CLIP_SECONDS = 8.0

# Margen desde el inicio: los primeros segundos suelen ser silencio, logos o
# una intro que no representa el audio de la película.
DEFAULT_OFFSET_S = 60.0


class AudioPreviewEngine(QObject):
    """Genera y reproduce el fragmento con efectos aplicados."""

    status = Signal(str)          # texto para la UI
    failed = Signal(str)          # error legible
    ready = Signal(str)           # ruta del clip generado

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = None
        self._audio_output = None
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._current_path: str | None = None
        self._clip_dir = tempfile.mkdtemp(prefix=".drag-fx-preview-")

    # ------------------------------------------------------------------
    # Reproducción
    # ------------------------------------------------------------------

    def _ensure_player(self):
        """Crea el reproductor de QtMultimedia en el hilo de la GUI.

        ``QMediaPlayer`` no se puede instanciar desde otro hilo, así que solo
        se toca desde aquí (el motor emite señales y la GUI las recibe).
        """
        if self._player is not None:
            return self._player
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(1.0)
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        self._player.errorOccurred.connect(self._on_player_error)
        return self._player

    def _on_player_error(self, _error, error_string=""):
        if error_string:
            self.failed.emit(f"No se pudo reproducir el fragmento: {error_string}")

    def stop(self):
        """Detiene la reproducción y cancela la generación en curso."""
        self._cancel.set()
        try:
            if self._player is not None:
                self._player.stop()
        except Exception:
            pass
        self._cleanup_clip()

    def _cleanup_clip(self):
        path = self._current_path
        self._current_path = None
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def cleanup(self):
        """Libera recursos (al cerrar la ventana)."""
        self.stop()
        try:
            import shutil

            shutil.rmtree(self._clip_dir, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Generación
    # ------------------------------------------------------------------

    def play(self, ffmpeg_path, input_path, effects, *, start_time=None,
             audio_stream_index=0, clip_seconds=CLIP_SECONDS,
             target_lufs=None) -> None:
        """Genera (en segundo plano) y reproduce el fragmento con efectos.

        Llamar de nuevo mientras hay una generación en curso la cancela: es lo
        que ocurre de forma natural al arrastrar un control.
        """
        if not ffmpeg_path or not input_path or not os.path.exists(input_path):
            self.failed.emit("No hay archivo de audio que previsualizar.")
            return

        cfg = audio_fx.sanitize_effects(effects)
        if not audio_fx.effects_are_active(cfg):
            self.status.emit(
                "Sin efectos activos: activa el ecualizador, el compresor o el "
                "limitador para poder escucharlos."
            )
            return

        self._cancel.set()  # cancela la generación previa
        self._cancel = threading.Event()
        cancel = self._cancel
        self.status.emit("Preparando el fragmento...")

        self._worker = threading.Thread(
            target=self._generate_and_play,
            args=(cancel, ffmpeg_path, input_path, cfg, start_time,
                  audio_stream_index, clip_seconds, target_lufs),
            daemon=True,
            name="audio-fx-preview",
        )
        self._worker.start()

    def _generate_and_play(self, cancel, ffmpeg_path, input_path, cfg,
                           start_time, audio_stream_index, clip_seconds,
                           target_lufs):
        try:
            offset = self._pick_offset(ffmpeg_path, input_path, start_time)
            if cancel.is_set():
                return

            # Se mide EL FRAGMENTO (no el archivo entero): el compresor
            # necesita el nivel real de lo que se va a oír.
            self.status.emit("Analizando el fragmento...")
            measured = measure_loudness(
                ffmpeg_path, input_path,
                timeout_seconds=120,
                start_time=offset, end_time=offset + clip_seconds,
                audio_stream_index=audio_stream_index,
                abort_event=cancel,
            )
            if cancel.is_set():
                return
            measured_i = None
            if isinstance(measured, dict):
                try:
                    # La clave del loudness sale de la declaración de
                    # ``runner`` (LOUDNORM_MEDIDAS), no de aquí: es la misma
                    # que lee el render, y si se leen de dos maneras el
                    # preview puede sonar a una cosa y el render hacer otra.
                    measured_i = float(measured.get(LOUDNORM_CLAVE_LOUDNESS))
                except (TypeError, ValueError):
                    measured_i = None

            chain = audio_fx.build_effects_chain(
                cfg, measured_i=measured_i, target_lufs=target_lufs
            )
            if not chain:
                self.status.emit("Nada que previsualizar con esta configuración.")
                return

            out_path = os.path.join(
                self._clip_dir, f"fx-{int(time.time() * 1000)}.m4a"
            )
            self.status.emit("Aplicando efectos...")
            ok = self._render_clip(
                cancel, ffmpeg_path, input_path, offset, clip_seconds,
                audio_stream_index, ",".join(chain), out_path
            )
            if cancel.is_set() or not ok:
                if not cancel.is_set():
                    self.failed.emit("No se pudo generar el fragmento de prueba.")
                return

            self._cleanup_clip()
            self._current_path = out_path
            self.status.emit(
                f"Reproduciendo {clip_seconds:.0f}s con: "
                f"{audio_fx.describe_effects(cfg)}"
            )
            self.ready.emit(out_path)
        except Exception as exc:  # pragma: no cover - defensivo
            logger.warning("vista previa de audio falló: %s", exc)
            if not cancel.is_set():
                self.failed.emit(f"Error en la vista previa: {exc}")

    def _pick_offset(self, ffmpeg_path, input_path, start_time):
        """Punto del archivo del que tomar el fragmento.

        Si el usuario ya recortó, se usa el inicio del recorte. Si no, se salta
        el primer minuto para evitar intros y silencios iniciales.
        """
        start = _parse_time_to_seconds(start_time)
        if start:
            return float(start)
        duration = 0.0
        try:
            from src.ffmpeg.runner import get_video_info

            info = get_video_info(ffmpeg_path, input_path) or {}
            duration = float(info.get("duration", 0) or 0)
        except Exception:
            duration = 0.0
        if duration and duration <= DEFAULT_OFFSET_S + CLIP_SECONDS:
            return 0.0
        return DEFAULT_OFFSET_S

    def _render_clip(self, cancel, ffmpeg_path, input_path, offset,
                     clip_seconds, audio_stream_index, filter_chain, out_path):
        """Genera el clip aplicando la cadena. Devuelve True si salió bien."""
        index = 0 if audio_stream_index in (None, -1) else audio_stream_index
        cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostats", "-y"]
        # Seek de entrada (rápido) + duración: el fragmento no necesita
        # posiciones absolutas.
        if offset:
            cmd += ["-ss", f"{float(offset):.3f}"]
        cmd += ["-i", input_path]
        cmd += ["-t", f"{float(clip_seconds):.3f}"]
        cmd += ["-map", f"0:a:{index}?", "-vn"]
        cmd += ["-af", filter_chain]
        cmd += ["-c:a", "aac", "-b:a", "192k", out_path]

        kwargs = _run_kwargs()
        kwargs.pop("capture_output", None)
        kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
        kwargs.update(process_group_popen_kwargs())
        proc = subprocess.Popen(cmd, **kwargs)
        try:
            while proc.poll() is None:
                if cancel.is_set():
                    _terminate(proc)
                    return False
                time.sleep(0.05)
        finally:
            if proc.poll() is None:
                _terminate(proc)
        return (
            proc.returncode == 0
            and os.path.exists(out_path)
            and os.path.getsize(out_path) > 0
        )


def _terminate(proc) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
