/**
 * El flujo de trabajo, con la misma forma que la cadena de la app de
 * escritorio: nodos que se activan/desactivan y un paso final de compresión.
 * Aquí los motores son shaders WGSL, así que solo existen los dos pasos que
 * el navegador puede hacer de verdad.
 */

export type StepType = 'upscale' | 'sharpen';

interface StepBase {
  id: string;
  enabled: boolean;
}

export interface UpscaleStep extends StepBase {
  type: 'upscale';
  /** x2, x3 o x4 */
  factor: 2 | 3 | 4;
}

export interface SharpenStep extends StepBase {
  type: 'sharpen';
  /** 0 = sin efecto, 1.5 = fuerte */
  amount: number;
}

export type Step = UpscaleStep | SharpenStep;

/** Ajustes del panel "Compression Settings". */
export interface Settings {
  mode: 'bitrate';
  /** kbps */
  bitrateKbps: number;
  /** Altura máxima de salida; 0 = la que salga de la cadena. */
  maxHeight: number;
  codec: 'avc' | 'hevc';
}

export const DEFAULT_SETTINGS: Settings = {
  mode: 'bitrate',
  bitrateKbps: 8000,
  maxHeight: 0,
  codec: 'avc',
};

let counter = 0;
const nextId = () => `s${++counter}`;

export function makeStep(type: StepType): Step {
  switch (type) {
    case 'upscale':
      return { id: nextId(), type, enabled: true, factor: 2 };
    case 'sharpen':
      return { id: nextId(), type, enabled: false, amount: 0.45 };
  }
}

/** Pasos que tocan píxeles, en orden. */
export function frameSteps(steps: Step[]): Array<UpscaleStep | SharpenStep> {
  return steps.filter((s): s is UpscaleStep | SharpenStep => s.enabled);
}

/** Factor total de reescalado de la cadena activa. */
export function totalScale(steps: Step[]): number {
  return frameSteps(steps).reduce((acc, s) => acc * (s.type === 'upscale' ? s.factor : 1), 1);
}
