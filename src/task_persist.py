"""Un solo dueño de lo que el progreso del render le debe al disco.

El hilo de la GUI anotaba el historial de tareas una vez por fase y por
fotograma. Cada anotación releía el historial entero y lo reescribía entero con
``fsync``: 19 escrituras por segundo de render, con el bucle de eventos de la
GUI retrasado en proporción directa a la latencia del disco. Con el disco al
100 % —que es cuando pasa, porque el render está escribiendo los fotogramas— de
ahí el "no responde" que ve Windows.

Aquí no se cambia lo que la GUI dice: la barra, el estado y las tarjetas los
sigue pintando el hilo de la GUI, evento a evento y en el mismo orden. Lo que
se cambia es **dónde** se persiste. La GUI anota en memoria —que es
instantánea— y un vigilante lleva eso al disco.

La forma es un vigilante que espera en una condición, y no un sondeo. No es
estética: con un sondeo, ``_wake`` se puede perder entre el momento en que el
hilo se retira y el momento en que llega trabajo, y entonces la nota se queda
sin vigilante y ``flush`` se come el tiempo de espera entero. Medido, pasaba
en el 0,5 % de los volcados, con la GUI parada.

Con una condición no hay ventana, por dos razones distintas. La primera es que
el vigilante no se retira nunca: siempre hay alguien escuchando, así que "hay un
vigilante" es un dato (``_watching``) y no una suposición. La segunda es que
las dos preguntas —¿hay trabajo? y ¿está todo en disco?— se responden bajo el
**mismo lock**, y tomar el trabajo y marcar que hay una escritura en curso
ocurren en la misma toma del lock, sin hueco entre medias. Por eso lo que
``flush`` espera no se puede volver cierto en el hueco que dejó el vigilante.

Tres reglas, y las tres se pueden equivocar:

* **No perder ni mentir, sin reintentar a lo tonto.** Si una escritura falla,
  no se recuerda como hecha —igual que la cola— y la vuelve a pedir el
  siguiente evento de esa fase. Un vigilante que reintentara solo, en bucle,
  contra un disco que no vuelve, se comia un nucleo entero: medido, 2,98 s de
  CPU en 3,0 s y casi 8.000 escrituras por segundo. Reintentar no es recordarse
  sin mas; es un bucle, y aqui no hace falta porque el render no para de
  mandar eventos de la fase que falló.

  Lo que sí se recuerda es **que** se perdió, en ``_lost``, y eso es lo que
  convierte un ``flush`` —"no queda nada pendiente"— en un veredicto que se
  puede contar: ``drain``. Un disco lento no es una pérdida: el volcado sigue
  en curso y aterriza solo. Un disco que rechaza la escritura sí lo es.
* **Fundir progreso.** Si dos eventos llegan antes de que el vigilante llegue a
  ellos, se escribe el último, y la posición en el diccionario es la de la
  primera vez, que es el orden en que las emite el worker. Un intermedio que
  nadie llegó a leer no estaba persistido: lo que se lee desde fuera es el
  archivo, no la cola de entrada. Un cierre no necesita trato especial porque
  el worker lo manda siempre como último evento de su fase, así que "gana el
  último" ya es "gana el cierre"; se comprobó que darle su propio camino no
  cambia nada en ningún caso que el worker pueda producir, y por eso no está.
* **Comparar contra lo escrito, no contra lo pedido.** El deduplicador de la
  cola recuerda la última firma que **se escribió**. Si recuerda la que se
  pidió, un solo fallo de escritura —disco lleno a mitad de un render— deja el
  estado atrasado para siempre, porque los reintentos idénticos se descartan
  como repetidos. Las fases no tienen ese problema: cada evento vuelve a
  pedirlas.
* **La cadena de la cola no se funde nunca**, porque ``task_queue`` valida las
  transiciones. Saltarse ``muxing`` dejaría un ``publishing`` inválido y el
  item atascado sin que nadie lo dijera. Se tiran los repetidos consecutivos
  —diez "muxing" son uno— y los distintos se conservan, en orden.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


# Cómo se llama en el historial un evento de fase. Estaba copiado en los dos
# manejadores de la ventana; su dueño es la tabla.
_PHASE_STATUS = {
    "start": "pending",
    "progress": "running",
    "end": "done",
    "failed": "failed",
}

# Espera máxima de un volcado. Es una red, no el diseño: el vigilante no se
# pierde y no hay nada que esperar. Lo que no puede ser son 10 s, porque
# ``flush`` se llama en el hilo de la GUI al terminar un render, al terminar su
# preview y al cerrar, y un atasco de ese tamaño es justo el "no responde" que
# esto vino a quitar.
FLUSH_TIMEOUT = 2.0

# Cada cuánto se reintenta una fase que no se pudo escribir. No es un sondeo:
# la espera sigue despertando en cuanto hay trabajo, así que no se pierde
# ninguna señal; lo único que acota es el ritmo con que se insiste contra un


class _PhaseWriter:
    """Anota fases en memoria y las escribe desde un vigilante aparte."""

    def __init__(self):
        self._cv = threading.Condition()
        self._phases = {}           # clave -> último valor, en orden de 1ª vez
        self._queue = []            # cambios de la cola, en orden
        self._queue_written = None  # última firma que se ESCRIBIÓ
        self._close = None          # (task_id, status) pendiente de cerrar
        self._lost = set()          # lo pedido y no escrito. No se reintenta.
        self._writing = False       # hay un volcado en curso
        self._watching = False      # el vigilante existe. Dato, no suposición.

    def _idle(self):
        """¿Queda algo en la mano? Lo que el vigilante espera y lo que espera flush.

        ``_writing`` está dentro a propósito: entre tomar el trabajo y
        terminarlo los diccionarios ya están vacíos, y sin esto un volcado
        parecería hecho antes de haberlo hecho.

        Cierra la lista y no solo las fases: un volcado que lleva un cierre
        dentro no está ocioso hasta que ese cierre también se escribió.
        """
        return not (self._phases or self._queue or self._close or self._writing)

    def _pending(self):
        return bool(self._phases or self._queue or self._close)

    # ------------------------------------------------------------------
    # Lo que llama el hilo de la GUI. Nada de disco, nada de esperar.
    # ------------------------------------------------------------------
    def note_render_event(self, task_id, ev, queue_active=False):
        """Un evento de fase de un render: historial y, si toca, cola.

        El reparto entre historial y cola es de aquí, no de quien llama: la
        ventana solo tiene que decir si hay una cola en marcha.
        """
        self.note_phase_event(task_id, ev)
        if not queue_active:
            return
        kind = str(ev.get("ev") or "")
        if kind == "failed":
            self.note_queue_status(task_id, "failed", error=ev.get("msg", ""))
        elif str(ev.get("id", "")) == "mux" and kind in ("start", "progress"):
            self.note_queue_status(task_id, "muxing")

    def note_phase_event(self, task_id, ev):
        """Anota un evento de fase tal cual lo emite el worker."""
        if not task_id:
            return
        kind = str(ev.get("ev") or "")
        self.note_phase(
            task_id,
            ev.get("id", ""),
            label=ev.get("label", ""),
            status=_PHASE_STATUS.get(kind, "pending"),
            pct=int(ev.get("pct", 0) or 0),
        )

    def note_phase(self, task_id, phase_id, label="", status="pending", pct=0):
        """Deja esta fase pendiente de escribir, sustituyendo a la anterior."""
        if not task_id:
            return
        key = (str(task_id), str(phase_id))
        value = {"label": str(label or ""), "status": str(status),
                 "pct": int(pct)}
        with self._cv:
            # Un cierre llega sin etiqueta —el worker no la manda— y la que se
            # ve en el archivo es la del progreso anterior. Sin arrastrarla, el
            # "mux" se guardaría como "Finalizing / Mux" durante el render y
            # como "mux" al terminar.
            if not value["label"] and key in self._phases:
                value["label"] = self._phases[key]["label"]
            self._phases[key] = value
            self._cv.notify()
        self._start()

    def note_queue_status(self, queue_id, status, error=""):
        """Pide un estado para un item de la cola.

        Se descarta si ya está escrito o si ya está esperando; en cualquier
        otro caso se anota. Comparar contra lo escrito es lo que hace que un
        fallo de escritura se recupere en el siguiente evento en vez de dejar
        el item atrasado para siempre.
        """
        if not queue_id:
            return
        sig = (str(queue_id), str(status), str(error or ""))
        with self._cv:
            if self._queue_written == sig or sig in self._queue:
                return
            self._queue.append(sig)
            self._cv.notify()
        self._start()

    def flush(self, timeout=FLUSH_TIMEOUT):
        """Espera a que el vigilante no tenga nada en la mano.

        Se llama en tres sitios, y en los tres tiene sentido esperar: al
        terminar una tarea, al terminar su preview, y al cerrar la aplicación.
        Ahí no hay nada más que hacer, y perder la escritura sería perder datos.

        Devuelve ``True`` si al terminar **no queda nada pendiente**. Y eso es
        todo lo que significa, asi que no hay que confundirlo con "todo esta en
        disco": una escritura que se rechazó se dejó de recordar —como la
        cola— y ya no está pendiente. Para eso está ``drain``.

        El límite es corto a propósito. Si el vigilante se atasca —disco
        colgado, un ``fsync`` que no vuelve— la GUI se rinde en ``timeout`` en
        vez de quedarse mirando el reloj diez segundos. Y rendirse aquí no es
        tirar el trabajo: lo que estaba en curso sigue en curso y aterriza
        cuando el disco responda.
        """
        with self._cv:
            return self._cv.wait_for(self._idle, timeout)

    def take_unwritten(self, task_id=None, timeout=FLUSH_TIMEOUT):
        """Espera lo acotado y **se queda** lo perdido: ¿había algo sin avisar?

        El nombre dice lo que hace y lo que contesta, que es lo que le faltaba
        a ``drain``: devuelve ``True`` **cuando algo se perdió**, y se lleva lo
        perdido para no volver a contarlo. Devolver ``True`` en un método cuyo
        nombre no lo anuncia ya costó un bug: el cierre avisaba justo cuando no
        había nada que avisar.

        A diferencia de ``flush``, esto tiene efecto: pregunta y olvida. Es el
        único dueño de la decisión de "ya te lo he dicho", y por eso una
        pérdida se avisa **una vez**. Sin esto, una fase que se rechazó a mitad
        de un render y que nadie volvió a pedir se volvía a anunciar en cada
        cierre posterior de la aplicación, que es la forma más cansona de
        estar roto.

        Avisa quien pregunta: si el que pregunta no dice nada, la pérdida se
        queda callada, y eso no puede ser.

        **El tiempo agotado no es una pérdida.** Un volcado que sigue en curso
        va a terminar, y tratar la espera como fallo convertía un render
        terminado en una tarea que se quedaba en "running" para siempre, con
        el aviso de que "el programa se cerró antes de terminar" sobre un
        render que sí había terminado. Así que el veredicto solo mira
        ``_lost``: lo que el disco rechazó de verdad.

        Cada clave de ``_lost`` es ``(dueño, detalle)``, y el dueño es siempre
        el primer campo: la tarea a la que pertenece lo que se perdió. Se
        compara ese campo y no "a ver si el id aparece en la clave", que
        confundía un ``task_id`` con un ``phase_id`` que casaran por valor.

        ``task_id`` acota a una tarea; sin él, se lleva todo lo pendiente de
        avisar.
        """
        self.flush(timeout)
        with self._cv:
            if task_id is None:
                perdidas = self._lost
                self._lost = set()
            else:
                perdidas = {k for k in self._lost if k[0] == task_id}
                self._lost -= perdidas
        return bool(perdidas)

    def finish_task(self, task_id, status):
        """Cierra una tarea en disco: las fases primero, el cierre después.

        El orden es el contrato, y por eso vive aquí y no escrito en el
        manejador: si el archivo dice "success" tiene que decir también hasta
        dónde llegó. El cierre se encola como trabajo más del vigilante, que lo
        escribe detrás de las fases del mismo volcado, así que el orden no
        depende de que una espera acierte.

        Se devuelve si el cierre quedó escrito, y **el aviso va aquí**: este es
        el único sitio que sabe qué se perdió y de quién. Quien llama no
        necesita el veredicto —el panel no va a preguntar— y por eso no lo
        usa: aquí solo se cierra.
        """
        if not task_id:
            return True
        with self._cv:
            self._close = (str(task_id), str(status))
            self._cv.notify()
        self._start()
        if self.take_unwritten(task_id):
            logger.warning(
                "No se pudo guardar la tarea %s en disco (%s): el progreso "
                "pendiente no llego a escribirse.", task_id, status)
            return False
        return True

    # ------------------------------------------------------------------
    # El vigilante
    # ------------------------------------------------------------------
    def _start(self):
        """Se asegura de que hay un vigilante. ``_watching`` es el dato.

        No se deduce de ``Thread.is_alive()``: un hilo que ya ha decidido
        retirarse sigue vivo, y confundir eso con "hay alguien vigilando" es
        lo que dejaba notas sin escribir. Como el vigilante no se retira, el
        dato se pone una vez y es cierto para siempre.
        """
        with self._cv:
            if self._watching:
                return
            self._watching = True
            threading.Thread(
                target=self._run, daemon=True, name="task-persist-writer").start()

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(self._pending)
                phases, queue, close = self._take()
                self._writing = True
            # Fuera del lock: anotar no puede esperar a que acabe un fsync.
            resultado = self._write(phases, queue, close)
            with self._cv:
                self._lost.difference_update(k for k, ok in resultado if ok)
                self._lost.update(k for k, ok in resultado if not ok)
                self._writing = False
                self._cv.notify_all()

    def _take(self):
        """Vacía lo pendiente. Se llama con el lock tomado."""
        phases = list(self._phases.items())
        self._phases = {}
        queue = list(self._queue)
        self._queue = []
        close = self._close
        self._close = None
        return phases, queue, close

    def _write(self, phases, queue, close):
        """Vuelca lo tomado. Devuelve ``(clave, escrito)`` para cada cosa.

        El trato es el de la cola: **solo se da por escrito lo que se
        escribió de verdad**. Lo que falló no se recuerda ni se reintenta
        aqui; se anota en ``_lost`` y lo vuelve a pedir el siguiente evento de
        esa fase. Un vigilante que reintentara en bucle quemaria un núcleo
        entero contra un disco que no vuelve.
        """
        # Fases antes que cola antes que cierre: el mismo orden que tenía
        # cuando cada una escribía por su cuenta dentro del manejador.
        resultado = []
        for (task_id, phase_id), value in phases:
            try:
                import src.task_history as task_history
                ok = bool(task_history.update_phase(
                    task_id, phase_id, label=value["label"],
                    status=value["status"], pct=value["pct"]))
            except Exception:
                ok = False
            resultado.append(((task_id, phase_id), ok))
        for queue_id, status, error in queue:
            escrito = False
            try:
                import src.task_queue as task_queue
                escrito = bool(task_queue.set_status(queue_id, status,
                                                     error=error))
            except Exception:
                escrito = False
            # Solo se recuerda si se escribió. Si falló, el deduplicador no lo
            # da por hecho y el siguiente evento lo intentará otra vez.
            if escrito:
                with self._cv:
                    self._queue_written = (str(queue_id), str(status),
                                           str(error or ""))
            resultado.append(((str(queue_id), ("queue", str(status))), escrito))
        if close is not None:
            try:
                import src.task_history as task_history
                ok = bool(task_history.finish_task(close[0], close[1]))
            except Exception:
                ok = False
            resultado.append(((close[0], ("close",)), ok))
        return resultado

phase_writer = _PhaseWriter()
