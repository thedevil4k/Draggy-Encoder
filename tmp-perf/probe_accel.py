"""Read-only acceleration probe for DRAGGY ENCODER's AI inference stage.

Measures, on the real installed runtimes and the real model:
  1. ORT baseline exactly as src/ai/inference.py builds it (fp32, CUDA EP)
  2. ORT with tuned provider options
  3. ORT + CUDA graph + IO binding
  4. ORT fp16 model (manually converted copy, written to tmp-perf only)
  5. ORT fp16 + CUDA graph
  6. TensorRT EP fp16 (engine cache in tmp-perf only)
  7. PyTorch: fp32 vs autocast-fp16 vs true fp16 (channels_last on/off)
  8. Where every ORT node actually runs (CPU fallback hunt)
Nothing is written outside tmp-perf/.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tmp-perf")
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.runtimes.manager import (  # noqa: E402
    import_local_runtime,
    inject_runtime_paths,
    ensure_runtime_dlls_on_path,
)

inject_runtime_paths()
ensure_runtime_dlls_on_path()

import onnx  # noqa: E402
from onnx import helper, numpy_helper, TensorProto  # noqa: E402

ort = import_local_runtime("onnxruntime")

MODEL = os.path.join(ROOT, "bin", "models", "upscale",
                     "realesr-animevideov3", "realesr-animevideov3.onnx")
PTH = os.path.join(ROOT, "bin", "models", "upscale",
                   "realesr-animevideov3", "realesr-animevideov3.pth")
FP16_MODEL = os.path.join(OUT, "animevideov3_fp16.onnx")

print("onnxruntime", ort.__version__, "| providers:", ort.get_available_providers())
m = onnx.load(MODEL)
print("ops:", sorted({n.op_type for n in m.graph.node}))
print("inputs :", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("outputs:", [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
init_types = {numpy_helper.to_array(i).dtype for i in m.graph.initializer}
print("initializer dtypes:", init_types)

H, W = 480, 720
rng = np.random.default_rng(0)
frame = (rng.random((H, W, 3)) * 255).astype(np.uint8)


def to_input(fr, dtype=np.float32):
    rgb = np.ascontiguousarray(fr[:, :, ::-1])
    x = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    x = np.ascontiguousarray(x)
    return x.astype(dtype) if dtype != np.float32 else x


X = to_input(frame)
RESULTS = {}


def bench(fn, iters=15, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts)


def report(label, fn, **kw):
    try:
        med, best = bench(fn, **kw)
        print("  %-42s %8.1f ms  (%5.2f fps)  best %7.1f ms" % (label, med, 1000 / med, best))
        RESULTS[label] = round(med, 2)
        return med
    except Exception as exc:  # noqa: BLE001
        print("  %-42s FAILED: %s: %s" % (label, type(exc).__name__, str(exc)[:160]))
        return None


def build(providers, so=None, tag=""):
    t0 = time.perf_counter()
    s = ort.InferenceSession(MODEL, sess_options=so, providers=providers)
    dt = (time.perf_counter() - t0) * 1000.0
    print("  [%s] session built in %.0f ms | active: %s" % (tag, dt, s.get_providers()))
    return s


print("\n=== 1. ORT baseline: exactly what src/ai/inference.py does (fp32) ===")
sess = build(["CUDAExecutionProvider", "CPUExecutionProvider"], tag="fp32")
in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
report("ORT CUDA fp32 (app today)", lambda: sess.run([out_name], {in_name: X}))

print("\n=== 2. ORT fp32 + tuned provider options ===")
CUDA_OPTS = {
    "cudnn_conv_algo_search": "HEURISTIC",
    "arena_extend_strategy": "kSameAsRequested",
    "do_copy_in_default_stream": "1",
}
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess2 = build([("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"], so, "fp32+tuned")
in2, out2 = sess2.get_inputs()[0].name, sess2.get_outputs()[0].name
report("ORT CUDA fp32 + tuned options", lambda: sess2.run([out2], {in2: X}))

print("\n=== 3. ORT fp32 + CUDA graph + IO binding ===")
so3 = ort.SessionOptions()
so3.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess3 = build(
    [("CUDAExecutionProvider", {**CUDA_OPTS, "enable_cuda_graph": "1"}), "CPUExecutionProvider"],
    so3, "fp32+graph")
in3, out3 = sess3.get_inputs()[0].name, sess3.get_outputs()[0].name
io3 = sess3.io_binding()
io3.bind_cpu_input(in3, X)
io3.bind_output(out3, "cuda")
report("ORT CUDA fp32 + cuda_graph + iobinding", lambda: sess3.run_with_iobinding(io3))

# ---------------------------------------------------------------- fp16 model
print("\n=== 4. fp16 ONNX (equivalent to ncnn-vulkan's fp16 shaders) ===")


def convert_fp16(src, dst):
    g = onnx.load(src).graph
    new_init = []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32:
            init = numpy_helper.from_array(arr.astype(np.float16), init.name)
        new_init.append(init)
    del g.initializer[:]
    g.initializer.extend(new_init)
    # Cast graph inputs fp32 -> fp16
    for i in list(g.input):
        cast_out = i.name + "_fp16"
        g.node.insert(0, helper.make_node("Cast", [i.name], [cast_out],
                                          to=TensorProto.FLOAT16))
        for n in g.node:
            for k, nm in enumerate(n.input):
                if nm == i.name:
                    n.input[k] = cast_out
    # Cast graph outputs fp16 -> fp32
    for o in list(g.output):
        cast_in = o.name + "_fp32"
        g.node.append(helper.make_node("Cast", [o.name], [cast_in],
                                       to=TensorProto.FLOAT))
        for n in g.node:
            for k, nm in enumerate(n.input):
                if nm == o.name and n.op_type == "Cast" and n.output[0] == cast_in:
                    continue
        o.name = cast_in
    del g.value_info[:]
    onnx.save(onnx.ModelProto(graph=g), dst)
    return dst


try:
    convert_fp16(MODEL, FP16_MODEL)
    print("  wrote", FP16_MODEL, os.path.getsize(FP16_MODEL), "bytes")
    m16 = onnx.load(FP16_MODEL)
    bad = [i.name for i in m16.graph.initializer
           if numpy_helper.to_array(i).dtype not in (np.float16, np.int64, np.int32, np.uint8)]
    print("  non-fp16 initializers left:", bad)
    ref = sess.run([out_name], {in_name: X})[0]
    for label, prov, tag in (
        ("ORT CUDA fp16", [("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"], "fp16"),
        ("ORT CUDA fp16 + cuda_graph + iobinding",
         [("CUDAExecutionProvider", {**CUDA_OPTS, "enable_cuda_graph": "1"}), "CPUExecutionProvider"], "fp16+graph"),
    ):
        s16 = ort.InferenceSession(FP16_MODEL, providers=prov)
        i16, o16 = s16.get_inputs()[0].name, s16.get_outputs()[0].name
        out = s16.run([o16], {i16: X})[0]
        diff = float(np.abs(out - ref).max())
        rel = float(np.abs(out - ref).mean())
        print("  [%s] max|diff| vs fp32 = %.5f  mean|diff| = %.6f" % (tag, diff, rel))
        if "graph" in tag:
            io = s16.io_binding()
            io.bind_cpu_input(i16, X)
            io.bind_output(o16, "cuda")
            report(label, lambda s=s16, i=io: s.run_with_iobinding(i))
        else:
            report(label, lambda s=s16, a=i16, b=o16: s.run([b], {a: X}))
            fp16_sess, fp16_io = s16, (i16, o16)
except Exception as exc:  # noqa: BLE001
    print("  fp16 conversion/bench failed:", type(exc).__name__, exc)

# ------------------------------------------------------------------- TensorRT
print("\n=== 5. TensorRT EP fp16 (trt_fp16_enable) ===")
try:
    trt_cache = os.path.join(OUT, "trt_cache")
    os.makedirs(trt_cache, exist_ok=True)
    trt_opts = {
        "trt_fp16_enable": "1",
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": trt_cache,
        "trt_max_workspace_size": str(2 * 1024 * 1024 * 1024),
        "trt_profile_min_shapes": f"{in_name}:1x3x{H}x{W}",
        "trt_profile_opt_shapes": f"{in_name}:1x3x{H}x{W}",
        "trt_profile_max_shapes": f"{in_name}:1x3x{H}x{W}",
        "trt_builder_optimization_level": "3",
    }
    t0 = time.perf_counter()
    strt = ort.InferenceSession(
        MODEL, providers=[("TensorrtExecutionProvider", trt_opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print("  TRT session (engine build included) %.0f s | active: %s" % (time.perf_counter() - t0, strt.get_providers()))
    it, ot = strt.get_inputs()[0].name, strt.get_outputs()[0].name
    report("ORT TensorRT fp16 (build cached)", lambda: strt.run([ot], {it: X}))
    # second session = pure engine-cache load time
    t0 = time.perf_counter()
    strt2 = ort.InferenceSession(
        MODEL, providers=[("TensorrtExecutionProvider", trt_opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print("  second session (from engine cache): %.1f s" % (time.perf_counter() - t0))
    report("ORT TensorRT fp16 (cached session)", lambda: strt2.run([ot], {it: X}))
except Exception as exc:  # noqa: BLE001
    print("  TensorRT failed: %s: %s" % (type(exc).__name__, str(exc)[:300]))

# ------------------------------------------------------------------ profiling
print("\n=== 6. ORT node placement (CPU fallback hunt) ===")
try:
    so6 = ort.SessionOptions()
    so6.profile_file_prefix = os.path.join(OUT, "ort_profile")
    s6 = ort.InferenceSession(
        MODEL, sess_options=so6,
        providers=[("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"])
    a, b = s6.get_inputs()[0].name, s6.get_outputs()[0].name
    s6.run([b], {a: X})
    p = s6.end_profiling()
    with open(p, "r", encoding="utf-8") as fh:
        prof = json.load(fh)
    agg = {}
    for ev in prof:
        if ev.get("cat") != "Node":
            continue
        agg.setdefault(ev.get("args", {}).get("provider", "?"), [0, 0.0])
        agg[ev["args"].get("provider", "?")][0] += 1
        agg[ev["args"].get("provider", "?")][1] += ev.get("dur", 0)
    for k, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print("   %-26s nodes=%4d total=%8.2f ms" % (k, n, d / 1000.0))
    slow = sorted((e for e in prof if e.get("cat") == "Node"), key=lambda e: -e.get("dur", 0))[:10]
    print("   slowest nodes:")
    for e in slow:
        print("     %8.3f ms %-38s %s" % (e["dur"] / 1000.0, e.get("name", "")[:38],
                                          e.get("args", {}).get("provider", "?")))
except Exception as exc:  # noqa: BLE001
    print("  profiling failed:", type(exc).__name__, exc)

# --------------------------------------------------------------------- PyTorch
print("\n=== 7. PyTorch path (the app's other backend) ===")
try:
    import torch  # noqa: E402
    from src.ai.architectures import build_model_from_state_dict  # noqa: E402
    from src.ai.inference import load_checkpoint_state_dict  # noqa: E402

    dev = torch.device("cuda")
    sd = load_checkpoint_state_dict(PTH)
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    model = build_model_from_state_dict(sd).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print("  torch", torch.__version__, "| params %.2f M | cudnn %s" % (n_par / 1e6, torch.backends.cudnn.version()))
    t = torch.from_numpy(X).to(dev)

    def run_fp32():
        with torch.no_grad():
            model(t)

    def run_autocast():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(t)

    model16 = build_model_from_state_dict(sd).to(dev).eval().half()
    t16 = t.half()

    def run_fp16():
        with torch.no_grad():
            model16(t16)

    model16c = build_model_from_state_dict(sd).to(dev).eval().half().to(memory_format=torch.channels_last)
    t16c = t16.to(memory_format=torch.channels_last)

    def run_fp16_cl():
        with torch.no_grad():
            model16c(t16c)

    torch.backends.cudnn.benchmark = True
    for lbl, fn in (("torch fp32", run_fp32), ("torch autocast fp16 (app today)", run_autocast),
                    ("torch true fp16", run_fp16), ("torch fp16 + channels_last", run_fp16_cl)):
        report(lbl, fn)

    print("  --- full frame round trip (infer + CPU post-process + .cpu()) ---")
    def full_fp16():
        with torch.no_grad():
            o = model16(t16)
        (o.squeeze(0).clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    report("torch fp16 + post-process + H2D/D2H", full_fp16)
except Exception as exc:  # noqa: BLE001
    import traceback
    print("  torch probe failed:", type(exc).__name__, str(exc)[:300])
    traceback.print_exc()

print("\n=== SUMMARY (input %dx%d BGR, animevideov3 native x4) ===" % (W, H))
for k, v in sorted(RESULTS.items(), key=lambda kv: kv[1]):
    print("  %-46s %8.1f ms  %5.2f fps" % (k, v, 1000 / v))
with open(os.path.join(OUT, "probe_results.json"), "w", encoding="utf-8") as fh:
    json.dump(RESULTS, fh, indent=2, sort_keys=True)
print("saved tmp-perf/probe_results.json")
