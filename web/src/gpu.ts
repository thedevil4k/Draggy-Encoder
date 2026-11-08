import type { SharpenStep, Step, UpscaleStep } from './pipeline';

/**
 * Motor WebGPU: un único shader de pantalla completa que hace reescalado
 * bicúbico (Catmull-Rom separable) y, opcionalmente, máscara de desenfoque
 * para el realce. Cada paso del flujo es una pasada de render sobre una
 * textura; la salida de una es la entrada de la siguiente.
 */

const SHADER = /* wgsl */ `
struct Params {
  outSize: vec2f,
  amount: f32,
  _pad: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

fn cubic(x: f32) -> f32 {
  let a = abs(x);
  if (a < 1.0) { return 1.5 * a * a * a - 2.5 * a * a + 1.0; }
  if (a < 2.0) { return -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0; }
  return 0.0;
}

fn texel() -> vec2f {
  return 1.0 / vec2f(textureDimensions(srcTex));
}

fn bicubic(uv: vec2f) -> vec3f {
  let ts = texel();
  let p = uv / ts - 0.5;
  let base = floor(p);
  let f = p - base;
  var acc = vec3f(0.0);
  var wsum = 0.0;
  for (var j = -1; j <= 2; j = j + 1) {
    for (var i = -1; i <= 2; i = i + 1) {
      let w = cubic(f.x - f32(i)) * cubic(f.y - f32(j));
      let coord = (base + vec2f(f32(i), f32(j)) + 0.5) * ts;
      acc = acc + textureSampleLevel(srcTex, samp, coord, 0.0).rgb * w;
      wsum = wsum + w;
    }
  }
  return acc / wsum;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  // Triángulo de pantalla completa
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / params.outSize;
  var c = bicubic(uv);

  if (params.amount > 0.0) {
    let ts = texel();
    let blur = (
      bicubic(uv + vec2f(ts.x, 0.0)) +
      bicubic(uv - vec2f(ts.x, 0.0)) +
      bicubic(uv + vec2f(0.0, ts.y)) +
      bicubic(uv - vec2f(0.0, ts.y))
    ) * 0.25;
    c = c + (c - blur) * params.amount;
  }

  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

export interface GpuInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  maxBufferSize: number;
  maxTextureDimension2D: number;
  format: GPUTextureFormat;
  features: string[];
}

export class GpuPipeline {
  private bindGroupLayout: GPUBindGroupLayout;

  private constructor(
    private readonly device: GPUDevice,
    private readonly module: GPUShaderModule,
    private readonly sampler: GPUSampler,
    private readonly format: GPUTextureFormat,
    readonly info: GpuInfo,
  ) {
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
  }

  static async create(): Promise<GpuPipeline> {
    if (!('gpu' in navigator) || !navigator.gpu) {
      throw new Error(
        'Este navegador no expone WebGPU. Usa Chrome o Edge actualizados (Firefox en Windows también sirve).',
      );
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('WebGPU está presente pero no hay adaptador: GPU bloqueada o driver en lista negra.');
    }
    const device = await adapter.requestDevice();
    device.lost.then((detail) => {
      console.error('[webgpu] dispositivo perdido:', detail.reason, detail.message);
    });

    const info: GpuInfo = {
      vendor: adapter.info?.vendor ?? 'desconocido',
      architecture: adapter.info?.architecture ?? 'desconocida',
      device: adapter.info?.device ?? '',
      description: adapter.info?.description ?? '',
      maxBufferSize: adapter.limits.maxBufferSize,
      maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
      format: navigator.gpu.getPreferredCanvasFormat(),
      features: [...adapter.features].sort(),
    };

    const module = device.createShaderModule({ code: SHADER });
    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    return new GpuPipeline(device, module, sampler, navigator.gpu.getPreferredCanvasFormat(), info);
  }

  /**
   * Aplica los pasos de píxel en orden. Devuelve el lienzo final y su tamaño.
   */
  async process(
    source: OffscreenCanvas | HTMLCanvasElement,
    width: number,
    height: number,
    steps: Array<UpscaleStep | SharpenStep>,
  ): Promise<{ canvas: OffscreenCanvas; width: number; height: number }> {
    let current: OffscreenCanvas | HTMLCanvasElement = source;
    let w = width;
    let h = height;

    const pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: { module: this.module, entryPoint: 'vs' },
      fragment: {
        module: this.module,
        entryPoint: 'fs',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    for (const step of steps) {
      const factor = step.type === 'upscale' ? step.factor : 1;
      const amount = step.type === 'sharpen' ? step.amount : 0;

      const tw = Math.max(1, Math.round(w * factor));
      const th = Math.max(1, Math.round(h * factor));

      const target = new OffscreenCanvas(tw, th);
      const ctx = target.getContext('webgpu');
      if (!ctx) throw new Error('No se pudo crear el contexto WebGPU del lienzo intermedio.');
      ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });

      const texture = this.device.createTexture({
        size: [w, h, 1],
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.device.queue.copyExternalImageToTexture({ source: current }, { texture }, [w, h]);

      const uniforms = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(uniforms, 0, new Float32Array([tw, th, amount, 0]));

      const bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: texture.createView() },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: uniforms } },
        ],
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ctx.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      this.device.queue.submit([encoder.finish()]);

      texture.destroy();
      uniforms.destroy();

      current = target;
      w = tw;
      h = th;
    }

    await this.device.queue.onSubmittedWorkDone();
    return { canvas: current as OffscreenCanvas, width: w, height: h };
  }

  destroy(): void {
    this.device.destroy();
  }
}

export function stepsSummary(steps: Step[]): string {
  return steps.map((s) => s.type).join(' → ') || '(vacío)';
}
