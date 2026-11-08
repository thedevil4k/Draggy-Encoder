import './style.css';
import { GpuPipeline } from './gpu';
import { openVideo, previewFrame, runJob, type SourceInfo } from './media';
import {
  DEFAULT_SETTINGS,
  makeStep,
  type Settings,
  type SharpenStep,
  type Step,
  type UpscaleStep,
} from './pipeline';
import type { InputVideoTrack } from 'mediabunny';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Falta #${id} en el HTML`);
  return el as T;
};

/* ------------------------------------------------------------------- dom */

const fileInput = $<HTMLInputElement>('file');
const nodeImport = $('node-import');
const importInfo = $('import-info');
const nodeUpscale = $('node-upscale');
const nodeSharpen = $('node-sharpen');
const upscaleSelect = $<HTMLSelectElement>('upscale-factor');
const sharpenSelect = $<HTMLSelectElement>('sharpen-amount');
const upscaleStatus = $('upscale-status');
const sharpenStatus = $('sharpen-status');
const gpuBadge = $('gpu-badge');

const setBitrate = $<HTMLInputElement>('set-bitrate');
const setBitrateSlider = $<HTMLInputElement>('set-bitrate-slider');
const setResolution = $<HTMLSelectElement>('set-resolution');
const setDevice = $<HTMLSelectElement>('set-device');
const setCodec = $<HTMLSelectElement>('set-codec');
const setName = $<HTMLInputElement>('set-name');
const originalBitrate = $('original-bitrate');

const statusEl = $('status');
const bar = $('bar');
const btnCompress = $<HTMLButtonElement>('btn-compress');
const btnPreview = $<HTMLButtonElement>('btn-preview');
const btnQueue = $<HTMLButtonElement>('btn-queue');
const btnAbort = $<HTMLButtonElement>('btn-abort');
const btnStartQueue = $<HTMLButtonElement>('btn-start-queue');
const btnClearQueue = $<HTMLButtonElement>('btn-clear-queue');
const btnClearFinished = $<HTMLButtonElement>('btn-clear-finished');
const taskList = $<HTMLUListElement>('task-list');
const taskEmpty = $('task-empty');
const previewTimeInput = $<HTMLInputElement>('preview-time');
const btnDependencies = $<HTMLButtonElement>('btn-dependencies');
const btnOutput = $<HTMLButtonElement>('btn-output');
const btnHardware = $<HTMLButtonElement>('btn-hardware');
const btnEngines = $<HTMLButtonElement>('btn-engines');
const btnPreferences = $<HTMLButtonElement>('btn-preferences');
const nodeCompress = $('node-compress');

const mGpu = $('m-gpu');
const mRend = $('m-rend');
const mEnc = $('m-enc');

const modal = $('modal');
const modalTitle = $('modal-title');
const modalBody = $('modal-body');
const modalClose = $<HTMLButtonElement>('modal-close');

const previewModal = $('preview-modal');
const previewClose = $<HTMLButtonElement>('preview-close');
const previewBefore = $<HTMLCanvasElement>('preview-before');
const previewAfter = $<HTMLCanvasElement>('preview-after');
const previewTimeLabel = $('preview-time-label');
const previewSizeLabel = $('preview-size-label');

/* ----------------------------------------------------------------- estado */

interface Loaded {
  file: File;
  info: SourceInfo;
  track: InputVideoTrack;
}

type TaskStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

interface Task {
  id: number;
  name: string;
  file: File;
  steps: Step[];
  settings: Settings;
  outputName: string;
  status: TaskStatus;
  pct: number;
  url?: string;
  size?: number;
  note?: string;
}

interface Prefs {
  autoDownload: boolean;
  remember: boolean;
}

const PREFS_KEY = 'draggy.web.prefs';
const SETTINGS_KEY = 'draggy.web.settings';

function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) return { autoDownload: false, remember: true, ...(JSON.parse(raw) as Partial<Prefs>) };
  } catch {
    /* localStorage puede estar bloqueado */
  }
  return { autoDownload: false, remember: true };
}

const prefs: Prefs = loadPrefs();
let gpu: GpuPipeline | null = null;
let encoderReady = false;
let avcSupported = false;
let hevcSupported = false;
let loaded: Loaded | null = null;
let files: File[] = [];
let busy = false;
let cancel = false;
let taskSeq = 0;

const steps: Step[] = [makeStep('upscale'), makeStep('sharpen')];
const settings: Settings = { ...DEFAULT_SETTINGS };
const tasks: Task[] = [];

/* ---------------------------------------------------------------- helpers */

function setStatus(text: string): void {
  statusEl.textContent = text;
}

function setProgress(ratio: number): void {
  bar.style.width = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
}

function persistPrefs(): void {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    /* sin persistencia no se rompe nada */
  }
}

function persistSettings(): void {
  if (!prefs.remember) return;
  try {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({
        settings,
        steps: steps.map((step) =>
          step.type === 'upscale'
            ? { type: 'upscale', enabled: step.enabled, factor: step.factor }
            : { type: 'sharpen', enabled: step.enabled, amount: step.amount },
        ),
      }),
    );
  } catch {
    /* idem */
  }
}

function restoreSettings(): void {
  if (!prefs.remember) return;
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const data = JSON.parse(raw) as {
      settings?: Partial<Settings>;
      steps?: Array<Record<string, unknown>>;
    };
    if (data.settings) {
      const bitrate = Number(data.settings.bitrateKbps);
      if (Number.isFinite(bitrate)) settings.bitrateKbps = Math.max(100, Math.min(50_000, Math.round(bitrate)));
      const height = Number(data.settings.maxHeight);
      if ([0, 480, 720, 1080].includes(height)) settings.maxHeight = height;
      if (data.settings.codec === 'avc' || data.settings.codec === 'hevc') settings.codec = data.settings.codec;
    }
    for (const saved of data.steps ?? []) {
      const step = steps.find((candidate) => candidate.type === saved.type);
      if (!step) continue;
      if (typeof saved.enabled === 'boolean') step.enabled = saved.enabled;
      if (step.type === 'upscale') {
        const factor = Number(saved.factor);
        if (factor === 2 || factor === 3 || factor === 4) step.factor = factor;
      }
      if (step.type === 'sharpen') {
        const amount = Number(saved.amount);
        if (Number.isFinite(amount)) step.amount = Math.max(0, Math.min(1.5, amount));
      }
    }
  } catch {
    /* datos viejos: se ignoran */
  }
}

const upscaleStep = (): UpscaleStep | undefined =>
  steps.find((step): step is UpscaleStep => step.type === 'upscale');
const sharpenStep = (): SharpenStep | undefined =>
  steps.find((step): step is SharpenStep => step.type === 'sharpen');

function renderNodes(): void {
  const up = upscaleStep();
  const sh = sharpenStep();

  nodeUpscale.classList.toggle('active', Boolean(up?.enabled));
  nodeSharpen.classList.toggle('active', Boolean(sh?.enabled));
  upscaleSelect.disabled = !up?.enabled;
  sharpenSelect.disabled = !sh?.enabled;

  if (up) {
    upscaleSelect.value = String(up.factor);
    upscaleStatus.textContent = up.enabled ? `${up.factor}x · Catmull-Rom` : 'inactive';
  }
  if (sh) {
    sharpenSelect.value = String(sh.amount);
    sharpenStatus.textContent = sh.enabled ? `Unsharp · ${sh.amount.toFixed(2)}` : 'inactive';
  }

  const ready = Boolean(loaded && gpu && encoderReady && !busy);
  btnCompress.disabled = !ready;
  btnPreview.disabled = !ready;
  btnQueue.disabled = Boolean(!loaded || busy);
  btnAbort.disabled = !busy;
}

function syncControls(): void {
  setBitrate.value = String(settings.bitrateKbps);
  setBitrateSlider.value = String(settings.bitrateKbps);
  setResolution.value = String(settings.maxHeight);
  setCodec.value = settings.codec;
}

/* -------------------------------------------------------------- ajustes */

function applyBitrate(value: number): void {
  const kbps = Math.max(100, Math.min(50_000, Math.round(value || 0)));
  settings.bitrateKbps = kbps;
  setBitrate.value = String(kbps);
  setBitrateSlider.value = String(kbps);
  persistSettings();
}

setBitrate.addEventListener('input', () => applyBitrate(Number(setBitrate.value)));
setBitrateSlider.addEventListener('input', () => applyBitrate(Number(setBitrateSlider.value)));
setResolution.addEventListener('change', () => {
  settings.maxHeight = Number(setResolution.value) || 0;
  persistSettings();
});
setCodec.addEventListener('change', () => {
  settings.codec = setCodec.value === 'hevc' ? 'hevc' : 'avc';
  persistSettings();
});

/* ---------------------------------------------------------------- nodos */

nodeUpscale.addEventListener('click', (event) => {
  if ((event.target as HTMLElement).tagName === 'SELECT') return;
  const step = upscaleStep();
  if (!step) return;
  step.enabled = !step.enabled;
  renderNodes();
  persistSettings();
});

nodeSharpen.addEventListener('click', (event) => {
  if ((event.target as HTMLElement).tagName === 'SELECT') return;
  const step = sharpenStep();
  if (!step) return;
  step.enabled = !step.enabled;
  renderNodes();
  persistSettings();
});

upscaleSelect.addEventListener('click', (event) => event.stopPropagation());
upscaleSelect.addEventListener('change', () => {
  const step = upscaleStep();
  if (!step) return;
  step.factor = Number(upscaleSelect.value) as 2 | 3 | 4;
  renderNodes();
  persistSettings();
});

sharpenSelect.addEventListener('click', (event) => event.stopPropagation());
sharpenSelect.addEventListener('change', () => {
  const step = sharpenStep();
  if (!step) return;
  step.amount = Number(sharpenSelect.value);
  renderNodes();
  persistSettings();
});

nodeCompress.addEventListener('click', () => {
  if (!btnCompress.disabled) btnCompress.click();
});

/* ------------------------------------------------------------------ gpu */

const VideoEncoderCtor = (
  globalThis as unknown as {
    VideoEncoder?: {
      isConfigSupported(config: Record<string, unknown>): Promise<{ supported?: boolean }>;
    };
  }
).VideoEncoder;

async function codecSupported(codec: 'avc' | 'hevc'): Promise<boolean> {
  if (!VideoEncoderCtor) return false;
  try {
    const result = await VideoEncoderCtor.isConfigSupported({
      codec: codec === 'avc' ? 'avc1.4d002a' : 'hvc1.1.6.L93.B0',
      width: 1280,
      height: 720,
      bitrate: 4_000_000,
      framerate: 30,
    });
    return Boolean(result.supported);
  } catch {
    return false;
  }
}

async function initGpu(): Promise<void> {
  try {
    gpu = await GpuPipeline.create();
    const { vendor, architecture, device } = gpu.info;
    const shortName = device || architecture || 'GPU';
    gpuBadge.textContent = `${vendor} ${shortName}`.trim();
    mGpu.textContent = `${vendor} ${architecture}`.trim();
    setDevice.replaceChildren(new Option(`${vendor} ${architecture}`.trim(), 'webgpu'));
  } catch (error) {
    gpu = null;
    gpuBadge.textContent = `WebGPU unavailable: ${(error as Error).message}`;
    setDevice.replaceChildren(new Option('No WebGPU adapter', 'none'));
  }

  avcSupported = await codecSupported('avc');
  hevcSupported = await codecSupported('hevc');
  encoderReady = avcSupported || hevcSupported;

  setCodec.replaceChildren();
  if (avcSupported) setCodec.append(new Option('H.264 (AVC)', 'avc'));
  if (hevcSupported) setCodec.append(new Option('HEVC (H.265)', 'hevc'));
  if (!encoderReady) setCodec.append(new Option('No encoder available', 'none'));

  if (settings.codec === 'hevc' && !hevcSupported) settings.codec = 'avc';
  if (settings.codec === 'avc' && !avcSupported && hevcSupported) settings.codec = 'hevc';
  setCodec.value = settings.codec;

  if (!encoderReady) setStatus('No WebCodecs video encoder available in this browser');

  renderNodes();
}

/* --------------------------------------------------------------- entrada */

async function loadFile(file: File): Promise<void> {
  setStatus('Reading video…');
  setProgress(0);
  try {
    const opened = await openVideo(file);
    loaded = { file, ...opened };
    const { info } = opened;
    const extra = files.length > 1 ? ` · ${files.length} videos` : '';
    importInfo.textContent = `${info.name} · ${info.width}×${info.height} · ${info.duration.toFixed(1)} s${extra}`;
    const totalBytes = files.reduce((acc, item) => acc + item.size, 0);
    const mbps = info.duration > 0 ? (totalBytes * 8) / info.duration / 1e6 : 0;
    originalBitrate.textContent = `Original video bitrate: ~${mbps.toFixed(1)} Mbps`;
    setStatus(`Selected: ${info.name}`);
  } catch (error) {
    loaded = null;
    importInfo.textContent = 'No video';
    originalBitrate.textContent = 'Original video bitrate: —';
    setStatus('Could not read that file');
    console.error(error);
  }
  renderNodes();
}

const VIDEO_EXT = /\.(mp4|mkv|avi|wmv|mov|webm|m4v)$/i;

async function acceptFiles(picked: File[]): Promise<void> {
  const videos = picked.filter((file) => file.type.startsWith('video/') || VIDEO_EXT.test(file.name));
  if (!videos.length) {
    setStatus('That file is not a video');
    return;
  }
  files = videos;
  await loadFile(videos[0]);
}

nodeImport.addEventListener('click', () => fileInput.click());
nodeImport.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  const picked = Array.from(fileInput.files ?? []);
  fileInput.value = '';
  if (picked.length) void acceptFiles(picked);
});

let dragDepth = 0;
window.addEventListener('dragenter', (event) => {
  event.preventDefault();
  dragDepth += 1;
  nodeImport.classList.add('over');
});
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) nodeImport.classList.remove('over');
});
window.addEventListener('drop', (event) => {
  event.preventDefault();
  dragDepth = 0;
  nodeImport.classList.remove('over');
  const dropped = Array.from(event.dataTransfer?.files ?? []);
  if (dropped.length) void acceptFiles(dropped);
});

/* ---------------------------------------------------------------- tareas */

function makeTask(file: File, status: TaskStatus): Task {
  const task: Task = {
    id: ++taskSeq,
    name: file.name,
    file,
    steps: steps.map((step) => ({ ...step })),
    settings: { ...settings },
    outputName: setName.value.trim(),
    status,
    pct: 0,
  };
  tasks.push(task);
  return task;
}

function fileNameFor(task: Task): string {
  const base = task.name.replace(/\.[^.]+$/, '');
  return `${task.outputName || `${base}-draggyweb`}.mp4`;
}

function triggerDownload(task: Task): void {
  if (!task.url) return;
  const link = document.createElement('a');
  link.href = task.url;
  link.download = fileNameFor(task);
  document.body.append(link);
  link.click();
  link.remove();
}

function renderTasks(): void {
  taskList.replaceChildren();

  if (!tasks.length) {
    taskEmpty.hidden = false;
    taskList.append(taskEmpty);
    return;
  }

  for (const task of tasks) {
    const item = document.createElement('li');
    item.className = 'task-card';
    item.dataset.id = String(task.id);

    const top = document.createElement('div');
    top.className = 'task-top';

    const name = document.createElement('span');
    name.className = 'task-name';
    name.textContent = task.name;
    name.title = task.name;

    const pill = document.createElement('span');
    pill.className = `pill ${task.status}`;
    pill.textContent = task.status.toUpperCase();

    top.append(name, pill);

    if (task.status !== 'running') {
      const remove = document.createElement('button');
      remove.className = 'task-remove';
      remove.textContent = '✕';
      remove.title = 'Remove from the list';
      remove.addEventListener('click', () => {
        const at = tasks.indexOf(task);
        if (at >= 0) tasks.splice(at, 1);
        renderTasks();
      });
      top.append(remove);
    }

    const track = document.createElement('div');
    track.className = 'task-bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(task.pct * 100)}%`;
    track.append(fill);

    const meta = document.createElement('div');
    meta.className = 'task-meta';

    const detail = document.createElement('span');
    const pct = Math.round(task.pct * 100);
    if (task.status === 'done' && task.size) detail.textContent = `${(task.size / 1024 ** 2).toFixed(2)} MB`;
    else if (task.status === 'failed' || task.status === 'cancelled') detail.textContent = task.note || task.status;
    else if (task.note) detail.textContent = task.note;
    else detail.textContent = `${pct}%`;
    meta.append(detail);

    if (task.url) {
      const link = document.createElement('a');
      link.href = task.url;
      link.download = fileNameFor(task);
      link.textContent = 'Download';
      meta.append(link);
    }

    item.append(top, track, meta);
    taskList.append(item);
  }
}

function updateTaskCard(task: Task): void {
  const item = taskList.querySelector<HTMLLIElement>(`li[data-id="${task.id}"]`);
  if (!item) return;
  const fill = item.querySelector<HTMLElement>('.task-bar > i');
  const detail = item.querySelector<HTMLElement>('.task-meta > span');
  const pct = Math.round(task.pct * 100);
  if (fill) fill.style.width = `${pct}%`;
  if (detail) detail.textContent = task.note || `${pct}%`;
}

/* --------------------------------------------------------------- render */

async function runTask(task: Task): Promise<void> {
  if (!gpu) {
    task.status = 'failed';
    task.note = 'WebGPU unavailable';
    renderTasks();
    return;
  }

  busy = true;
  cancel = false;
  task.status = 'running';
  task.pct = 0;
  task.note = '';
  renderTasks();
  renderNodes();
  setProgress(0);
  setStatus(`Compressing ${task.name}…`);

  const started = performance.now();
  let lastRate = 0;

  try {
    const opened = loaded && loaded.file === task.file ? loaded : await openVideo(task.file);
    const blob = await runJob(opened.track, task.steps, task.settings, gpu, {
      onProgress: (done, total) => {
        task.pct = total > 0 ? done / total : 0;
        const elapsed = (performance.now() - started) / 1000;
        const rate = done / Math.max(elapsed, 0.001);
        lastRate = rate;
        const left = Math.max(0, total - done) / Math.max(rate, 0.001);
        setProgress(task.pct);
        setStatus(`Frame ${done}/${total} · ${rate.toFixed(1)} fps · ETA ~${Math.round(left)} s`);
        mRend.textContent = `${rate.toFixed(1)} fps`;
        mEnc.textContent = `${elapsed.toFixed(1)} s`;
        updateTaskCard(task);
      },
      onNote: (message) => {
        task.note = message;
      },
      isCancelled: () => cancel,
    });

    task.url = URL.createObjectURL(blob);
    task.size = blob.size;
    task.status = 'done';
    task.pct = 1;
    setProgress(1);
    const seconds = (performance.now() - started) / 1000;
    setStatus(`Done in ${seconds.toFixed(1)} s · ${(blob.size / 1024 ** 2).toFixed(2)} MB`);
    if (prefs.autoDownload) triggerDownload(task);
  } catch (error) {
    const err = error as Error;
    setProgress(0);
    if (err.name === 'AbortError') {
      task.status = 'cancelled';
      task.note = 'Cancelled';
      setStatus('Cancelled');
    } else {
      task.status = 'failed';
      task.note = err.message;
      setStatus('Failed');
      console.error(err);
    }
  } finally {
    busy = false;
    cancel = false;
    mRend.textContent = lastRate ? `${lastRate.toFixed(1)} fps` : '—';
    renderTasks();
    renderNodes();
  }
}

btnCompress.addEventListener('click', async () => {
  if (!loaded || busy) return;
  const task = makeTask(loaded.file, 'queued');
  renderTasks();
  await runTask(task);
});

btnQueue.addEventListener('click', () => {
  if (!loaded) return;
  const list = files.length ? files : [loaded.file];
  for (const file of list) makeTask(file, 'queued');
  renderTasks();
  setStatus(`${list.length} task${list.length === 1 ? '' : 's'} queued`);
});

btnStartQueue.addEventListener('click', async () => {
  if (busy || !gpu) return;
  for (;;) {
    const next = tasks.find((task) => task.status === 'queued');
    if (!next) break;
    await runTask(next);
  }
  setStatus('Queue finished');
});

btnClearQueue.addEventListener('click', () => {
  for (let i = tasks.length - 1; i >= 0; i -= 1) {
    if (tasks[i].status === 'queued') tasks.splice(i, 1);
  }
  renderTasks();
});

btnClearFinished.addEventListener('click', () => {
  for (let i = tasks.length - 1; i >= 0; i -= 1) {
    if (tasks[i].status === 'done' || tasks[i].status === 'failed' || tasks[i].status === 'cancelled') {
      tasks.splice(i, 1);
    }
  }
  renderTasks();
});

btnAbort.addEventListener('click', () => {
  if (!busy) return;
  cancel = true;
  setStatus('Aborting…');
});

/* ---------------------------------------------------------------- preview */

function parseTime(raw: string): number | null {
  const text = raw.trim();
  if (!text) return null;
  if (/^\d+(\.\d+)?$/.test(text)) return Number(text);
  const parts = text.split(':').map(Number);
  if (parts.some((part) => !Number.isFinite(part))) return null;
  return parts.reduce((acc, part) => acc * 60 + part, 0);
}

btnPreview.addEventListener('click', async () => {
  if (!loaded || !gpu || busy) return;
  const fallback = Math.min(loaded.info.duration * 0.25, Math.max(0, loaded.info.duration - 0.05));
  const at = parseTime(previewTimeInput.value) ?? fallback;
  setStatus('Rendering preview…');

  try {
    const result = await previewFrame(loaded.track, at, steps, gpu);
    const pairs: Array<[HTMLCanvasElement, ImageBitmap]> = [
      [previewBefore, result.before],
      [previewAfter, result.after],
    ];
    for (const [canvas, bitmap] of pairs) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d')?.drawImage(bitmap, 0, 0);
      bitmap.close();
    }
    previewTimeLabel.textContent = `${result.time.toFixed(2)} s`;
    previewSizeLabel.textContent = `${result.width}×${result.height}`;
    previewModal.hidden = false;
    setStatus('Preview ready');
  } catch (error) {
    setStatus('Preview failed');
    console.error(error);
  }
});

/* ---------------------------------------------------------------- modales */

function openModal(title: string, build: (body: HTMLElement) => void): void {
  modalTitle.textContent = title;
  modalBody.replaceChildren();
  build(modalBody);
  modal.hidden = false;
}

function closeModals(): void {
  modal.hidden = true;
  previewModal.hidden = true;
}

function badgeKind(kind: 'ok' | 'bad' | 'warn' | 'neutral'): string {
  return kind === 'neutral' ? 'badge' : `badge ${kind}`;
}

function row(
  title: string,
  meta: string,
  detail: string,
  badge: string,
  kind: 'ok' | 'bad' | 'warn' | 'neutral',
): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'row';

  const main = document.createElement('div');
  main.className = 'row-main';

  const heading = document.createElement('div');
  heading.className = 'row-title';
  heading.textContent = title;
  main.append(heading);

  if (meta) {
    const sub = document.createElement('div');
    sub.className = 'row-meta';
    sub.textContent = meta;
    main.append(sub);
  }

  if (detail) {
    const path = document.createElement('div');
    path.className = 'row-path';
    path.textContent = detail;
    main.append(path);
  }

  const pill = document.createElement('span');
  pill.className = badgeKind(kind);
  pill.textContent = badge;

  wrap.append(main, pill);
  return wrap;
}

function note(text: string): HTMLElement {
  const paragraph = document.createElement('p');
  paragraph.className = 'muted small';
  paragraph.textContent = text;
  return paragraph;
}

btnDependencies.addEventListener('click', () => {
  openModal('Dependencies', (body) => {
    body.append(
      row(
        'WebGPU',
        'GPU compute for the image chain',
        gpu ? `${gpu.info.vendor} ${gpu.info.architecture} · ${gpu.info.format}` : 'Chrome or Edge with WebGPU',
        gpu ? 'OK' : 'MISSING',
        gpu ? 'ok' : 'bad',
      ),
      row(
        'WebCodecs · H.264',
        'Video encoder (hardware when available)',
        avcSupported ? 'avc1 detected' : 'no encoder exposed',
        avcSupported ? 'OK' : 'MISSING',
        avcSupported ? 'ok' : 'bad',
      ),
      row(
        'WebCodecs · HEVC',
        'Optional encoder, same pipeline',
        hevcSupported ? 'hvc1 detected' : 'not exposed by this browser',
        hevcSupported ? 'OK' : 'MISSING',
        hevcSupported ? 'ok' : 'warn',
      ),
      row('mediabunny', 'MP4 muxing on top of WebCodecs', 'bundled with this page', 'OK', 'ok'),
      note(
        'In the browser there is no FFmpeg, PyTorch or ONNX Runtime to download: the chain runs on WebGPU shaders and WebCodecs muxes the MP4.',
      ),
    );
  });
});

btnOutput.addEventListener('click', () => {
  openModal('Output', (body) => {
    body.append(
      row(
        'Downloads',
        'Your browser decides where a download goes; this page cannot pick a folder.',
        'Chrome / Edge → Settings → Downloads',
        'FIXED',
        'neutral',
      ),
      note('Every finished render keeps a Download link in the Tasks panel, so you can save it again at any time.'),
    );
  });
});

btnHardware.addEventListener('click', () => {
  openModal('Hardware Info', (body) => {
    if (!gpu) {
      body.append(row('WebGPU', 'No adapter available', '', 'UNAVAILABLE', 'bad'));
      return;
    }

    const list = document.createElement('dl');
    list.className = 'kv';
    const entries: Array<[string, string]> = [
      ['Vendor', gpu.info.vendor],
      ['Architecture', gpu.info.architecture],
      ['Device', gpu.info.device || '—'],
      ['Description', gpu.info.description || '—'],
      ['Canvas format', gpu.info.format],
      ['Max buffer size', `${(gpu.info.maxBufferSize / 1024 ** 2).toFixed(0)} MB`],
      ['Max texture 2D', `${gpu.info.maxTextureDimension2D} px`],
      ['Features', gpu.info.features.length ? gpu.info.features.join(', ') : 'none'],
    ];
    for (const [key, value] of entries) {
      const term = document.createElement('dt');
      term.textContent = key;
      const detail = document.createElement('dd');
      detail.textContent = value;
      list.append(term, detail);
    }
    body.append(list);

    body.append(
      row(
        'H.264 encoder',
        'WebCodecs',
        avcSupported ? 'available' : 'not available',
        avcSupported ? 'OK' : 'MISSING',
        avcSupported ? 'ok' : 'bad',
      ),
      row(
        'HEVC encoder',
        'WebCodecs',
        hevcSupported ? 'available' : 'not available',
        hevcSupported ? 'OK' : 'MISSING',
        hevcSupported ? 'warn' : 'bad',
      ),
    );
  });
});

btnEngines.addEventListener('click', () => {
  openModal('AI Engines', (body) => {
    body.append(
      row('Bicubic Catmull-Rom', 'UPSCALE · 2x / 3x / 4x', 'built-in WGSL shader · no weight files', 'BUILT-IN', 'ok'),
      row('Unsharp mask', 'SHARPEN · amount 0.25 – 1.20', 'built-in WGSL shader · no weight files', 'BUILT-IN', 'ok'),
      note(
        'This build runs its engines as shaders, so nothing is downloaded and nothing is stored on disk. The desktop app’s ONNX / PyTorch catalog is not part of the browser build.',
      ),
    );
  });
});

function prefRow(label: string, value: boolean, onChange: (next: boolean) => void): HTMLElement {
  const wrap = document.createElement('label');
  wrap.className = 'pref';

  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = value;
  input.addEventListener('change', () => onChange(input.checked));

  const text = document.createElement('span');
  text.textContent = label;

  wrap.append(input, text);
  return wrap;
}

btnPreferences.addEventListener('click', () => {
  openModal('Preferences', (body) => {
    body.append(
      prefRow('Auto-download when a render finishes', prefs.autoDownload, (value) => {
        prefs.autoDownload = value;
        persistPrefs();
      }),
      prefRow('Remember settings between sessions', prefs.remember, (value) => {
        prefs.remember = value;
        persistPrefs();
        if (value) persistSettings();
      }),
      note('Preferences live in this browser’s localStorage. Nothing leaves your computer.'),
    );
  });
});

modalClose.addEventListener('click', () => {
  modal.hidden = true;
});
previewClose.addEventListener('click', () => {
  previewModal.hidden = true;
});
modal.addEventListener('click', (event) => {
  if (event.target === modal) modal.hidden = true;
});
previewModal.addEventListener('click', (event) => {
  if (event.target === previewModal) previewModal.hidden = true;
});
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!modal.hidden || !previewModal.hidden) {
    closeModals();
    return;
  }
  if (busy) {
    cancel = true;
    setStatus('Aborting…');
  }
});

/* ------------------------------------------------------------------ boot */

restoreSettings();
syncControls();
renderNodes();
renderTasks();
void initGpu();
