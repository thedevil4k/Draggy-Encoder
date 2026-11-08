import {
  ALL_FORMATS,
  BlobSource,
  BufferTarget,
  CanvasSource,
  Input,
  Mp4OutputFormat,
  Output,
  Quality,
  VideoSampleSink,
  type InputVideoTrack,
  type VideoSample,
} from 'mediabunny';
import type { Settings, Step } from './pipeline';
import { frameSteps } from './pipeline';
import type { GpuPipeline } from './gpu';

/** Mediabunny: lectura perezosa del archivo del usuario (no se sube a ningún sitio). */

export interface SourceInfo {
  name: string;
  width: number;
  height: number;
  fps: number;
  duration: number;
  frames: number;
}

/** Tope duro para no tumbar la pestaña: x4 sobre 4K ya son 16K. */
const MAX_DIMENSION = 7680;

export async function openVideo(file: File): Promise<{
  info: SourceInfo;
  track: InputVideoTrack;
}> {
  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });

  if (!(await input.canRead())) {
    throw new Error('No he podido leer el contenedor de este archivo. Prueba con MP4, WebM o MKV (H.264/VP9/AV1).');
  }

  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('El archivo no tiene ninguna pista de vídeo.');

  if (!(await track.canDecode())) {
    throw new Error('El navegador no puede decodificar el códec de este vídeo. Prueba a convertirlo a H.264 antes.');
  }

  const width = await track.getDisplayWidth();
  const height = await track.getDisplayHeight();
  const duration = await track.computeDuration();
  const stats = await track.computePacketStats(90);
  const fps = stats.averagePacketRate > 0 ? stats.averagePacketRate : 30;

  return {
    info: { name: file.name, width, height, fps, duration, frames: Math.round(duration * fps) },
    track,
  };
}

/**
 * Fotograma inicial. Se pide con el iterador y no con `getSample(0)`: el
 * primer paquete casi nunca empieza exactamente en 0 (suele ser 0,033 s) y
 * `getSample(0)` devuelve null si el instante es anterior al primer fotograma.
 */
export async function firstFrame(track: InputVideoTrack): Promise<ImageBitmap> {
  const sink = new VideoSampleSink(track);
  const iterator = sink.samples();
  const first = await iterator.next();
  await iterator.return(undefined);
  const sample = first.value;
  if (!sample) throw new Error('No hay fotogramas en la pista de vídeo.');

  const canvas = new OffscreenCanvas(sample.displayWidth, sample.displayHeight);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Sin contexto 2D para la vista previa.');
  sample.draw(ctx, 0, 0);
  const bitmap = canvas.transferToImageBitmap();
  sample.close();
  return bitmap;
}

async function sampleAt(track: InputVideoTrack, time: number): Promise<VideoSample> {
  const sink = new VideoSampleSink(track);
  const duration = await track.computeDuration();
  const at = Math.max(0, Math.min(time, Math.max(0, duration - 0.001)));

  const direct = await sink.getSample(at);
  if (direct) return direct;

  let fallback: VideoSample | null = null;
  const iterator = sink.samples();
  for await (const sample of iterator) {
    fallback = sample;
    if (sample.timestamp >= at) break;
  }
  await iterator.return(undefined);
  if (!fallback) throw new Error('No hay ningún fotograma en ese instante.');
  return fallback;
}

/** Un fotograma antes/después de la cadena, para el diálogo Preview. */
export async function previewFrame(
  track: InputVideoTrack,
  time: number,
  steps: Step[],
  gpu: GpuPipeline,
): Promise<{ before: ImageBitmap; after: ImageBitmap; width: number; height: number; time: number }> {
  const passes = frameSteps(steps);
  const sample = await sampleAt(track, time);
  const width = sample.displayWidth;
  const height = sample.displayHeight;

  const source = new OffscreenCanvas(width, height);
  const sourceCtx = source.getContext('2d');
  if (!sourceCtx) throw new Error('Sin contexto 2D para el fotograma.');
  sample.draw(sourceCtx, 0, 0);
  const stamp = sample.timestamp;
  sample.close();

  const beforeCanvas = new OffscreenCanvas(width, height);
  const beforeCtx = beforeCanvas.getContext('2d');
  if (!beforeCtx) throw new Error('Sin contexto 2D para el fotograma original.');
  beforeCtx.drawImage(source, 0, 0);

  const out = passes.length
    ? await gpu.process(source, width, height, passes)
    : { canvas: source, width, height };

  const afterCanvas = new OffscreenCanvas(out.width, out.height);
  const afterCtx = afterCanvas.getContext('2d');
  if (!afterCtx) throw new Error('Sin contexto 2D para el fotograma procesado.');
  afterCtx.imageSmoothingQuality = 'high';
  afterCtx.drawImage(out.canvas, 0, 0, out.width, out.height);

  return {
    before: beforeCanvas.transferToImageBitmap(),
    after: afterCanvas.transferToImageBitmap(),
    width: out.width,
    height: out.height,
    time: stamp,
  };
}

export interface JobHooks {
  onProgress(done: number, total: number): void;
  onNote(message: string): void;
  isCancelled(): boolean;
}

/** Tamaño final: cadena + tope de resolución, siempre par para H.264/HEVC. */
function outputSize(width: number, height: number, steps: Step[], settings: Settings) {
  const scale = steps.reduce((acc, s) => acc * (s.type === 'upscale' && s.enabled ? s.factor : 1), 1);
  let w = Math.round(width * scale);
  let h = Math.round(height * scale);

  if (settings.maxHeight > 0 && h > settings.maxHeight) {
    const k = settings.maxHeight / h;
    w = Math.round(w * k);
    h = settings.maxHeight;
  }

  w += w % 2;
  h += h % 2;
  return { width: Math.max(2, w), height: Math.max(2, h) };
}

/**
 * Recorre los fotogramas del vídeo, aplica el flujo en la GPU y escribe un MP4.
 * `CanvasSource` se queda con una copia del lienzo en cada `add()`, así que el
 * lienzo de salida es fijo y solo cambia su contenido.
 */
export async function runJob(
  track: InputVideoTrack,
  steps: Step[],
  settings: Settings,
  gpu: GpuPipeline,
  hooks: JobHooks,
): Promise<Blob> {
  const passes = frameSteps(steps);

  const width = await track.getDisplayWidth();
  const height = await track.getDisplayHeight();
  const { width: outWidth, height: outHeight } = outputSize(width, height, steps, settings);

  if (Math.max(outWidth, outHeight) > MAX_DIMENSION) {
    throw new Error(
      `La salida sería de ${outWidth}×${outHeight}. El límite de esta versión web es ${MAX_DIMENSION}px; baja el factor de reescalado.`,
    );
  }

  const sink = new VideoSampleSink(track);
  const stats = await track.computePacketStats(90);
  const fps = stats.averagePacketRate > 0 ? stats.averagePacketRate : 30;
  const total = Math.max(1, Math.round((await track.computeDuration()) * fps));

  hooks.onNote(`Output ${outWidth}×${outHeight} · ${settings.bitrateKbps} kbps · ~${total} frames`);

  const inputCanvas = new OffscreenCanvas(width, height);
  const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: false });
  if (!inputCtx) throw new Error('Sin contexto 2D para el lienzo de entrada.');

  const outCanvas = new OffscreenCanvas(outWidth, outHeight);
  const outCtx = outCanvas.getContext('2d');
  if (!outCtx) throw new Error('Sin contexto 2D para el lienzo de salida.');
  outCtx.imageSmoothingQuality = 'high';

  const output = new Output({ format: new Mp4OutputFormat(), target: new BufferTarget() });
  const canvasSource = new CanvasSource(outCanvas, {
    codec: settings.codec,
    quality: new Quality({ bitrate: settings.bitrateKbps * 1000 }),
  });
  output.addVideoTrack(canvasSource, { frameRate: fps });
  await output.start();

  let done = 0;
  let time = 0;

  for await (const sample of sink.samples()) {
    if (hooks.isCancelled()) {
      await output.cancel();
      throw new DOMException('Cancelado por el usuario', 'AbortError');
    }

    inputCtx.clearRect(0, 0, width, height);
    sample.draw(inputCtx, 0, 0);

    const processed = passes.length
      ? await gpu.process(inputCanvas, width, height, passes)
      : { canvas: inputCanvas, width, height };

    outCtx.clearRect(0, 0, outWidth, outHeight);
    if (processed.width === outWidth && processed.height === outHeight) {
      outCtx.drawImage(processed.canvas, 0, 0);
    } else {
      outCtx.drawImage(processed.canvas, 0, 0, outWidth, outHeight);
    }

    const duration = sample.duration > 0 ? sample.duration : 1 / fps;
    await canvasSource.add(time, duration);
    time += duration;

    sample.close();
    done++;
    hooks.onProgress(done, total);
  }

  if (done === 0) throw new Error('No se ha podido decodificar ningún fotograma.');

  await output.finalize();
  const buffer = output.target.buffer;
  if (!buffer) throw new Error('El codificador no produjo ningún archivo.');
  return new Blob([buffer], { type: 'video/mp4' });
}
