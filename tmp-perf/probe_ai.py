"""Read-only diagnostics: ONNX Runtime / CUDA throughput for the upscale model.

Only reads the installed runtimes + model. Writes nothing but an ORT profile
JSON inside tmp-perf/. Does not touch the app's session state.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.runtimes.manager import (  # noqa: E402
    import_local_runtime,
    inject_runtime_paths,
    ensure_runtime_dlls_on_path,
)

inject_runtime_paths()
ensure_runtime_dlls_on_path()

ort = import_local_runtime("onnxruntime")
print("=" * 78)
print("onnxruntime", ort.__version__, "file:", ort.__file__)
print("providers:", ort.get_available_providers())
print("=" * 78)

MODEL = os.path.join(ROOT, "bin", "models", "upscale",
                     "realesr-animevideov3", "realesr-animevideov3.onnx")
print("model:", MODEL, os.path.getsize(MODEL), "bytes")

import onnx  # noqa: E402

m = onnx.load(MODEL)
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("op types:", ops)
for i in m.graph.input:
    print("  input :", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("  output:", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])

# Real frame size used by the last render: 720x480 source.
W, H = 720, 480
frame = (np.random.rand(H, W, 3) * 255).astype(np.uint8)


def to_input(fr):
    rgb = np.ascontiguousarray(fr[:, :, ::-1])
    x = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return np.ascontiguousarray(x)


X = to_input(frame)


def measure(sess, x, iters=15, warmup=3, iobinding=None):
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    for _ in range(warmup):
        if iobinding is not None:
            sess.run_with_iobinding(iobinding)
        else:
            sess.run([out_name], {in_name: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        if iobinding is not None:
            sess.run_with_iobinding(iobinding)
        else:
            sess.run([out_name], {in_name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts)


def build(providers, so=None, **kw):
    t0 = time.perf_counter()
    s = ort.InferenceSession(MODEL, sess_options=so, providers=providers, **kw)
    return s, (time.perf_counter() - t0) * 1000.0


print("-" * 78)
print("A. baseline (exactly what src/ai/inference.py does: providers, no options)")
sess_a, load_ms = build(["CUDAExecutionProvider", "CPUExecutionProvider"])
print("   built in %.0f ms, active providers: %s" % (load_ms, sess_a.get_providers()))
med, best = measure(sess_a, X)
print("   median %.1f ms/frame  (%.2f fps)   best %.1f ms" % (med, 1000 / med, best))

print("-" * 78)
print("B. SessionOptions tuned (HEURISTIC algo search, 1 thread, same arena)")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
cuda_opts = {
    "cudnn_conv_algo_search": "HEURISTIC",
    "arena_extend_strategy": "kSameAsRequested",
    "do_copy_in_default_stream": "1",
}
sess_b, load_ms = build(
    [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"], so
)
print("   built in %.0f ms, active: %s" % (load_ms, sess_b.get_providers()))
med_b, best_b = measure(sess_b, X)
print("   median %.1f ms/frame  (%.2f fps)   best %.1f ms" % (med_b, 1000 / med_b, best_b))

print("-" * 78)
print("C. CUDA graph + IOBinding (preallocated device buffers)")
so2 = ort.SessionOptions()
so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
cuda_opts2 = dict(cuda_opts)
cuda_opts2["enable_cuda_graph"] = "1"
sess_c, load_ms = build(
    [("CUDAExecutionProvider", cuda_opts2), "CPUExecutionProvider"], so2
)
print("   built in %.0f ms, active: %s" % (load_ms, sess_c.get_providers()))
in_name = sess_c.get_inputs()[0].name
out_name = sess_c.get_outputs()[0].name
io = sess_c.io_binding()
io.bind_cpu_input(in_name, X)
io.bind_output(out_name, "cuda")
med_c, best_c = measure(sess_c, X, iobinding=io)
print("   median %.1f ms/frame  (%.2f fps)   best %.1f ms" % (med_c, 1000 / med_c, best_c))

print("-" * 78)
print("D. CPU-side pre/post processing cost of one frame (numpy only)")
ts = []
for _ in range(15):
    t0 = time.perf_counter()
    y = sess_a.run([out_name], {in_name: X})[0]
    t_run = time.perf_counter() - t0
    t0 = time.perf_counter()
    o = np.squeeze(y, axis=0)
    o = np.transpose(o, (1, 2, 0))
    o = np.clip(o * 255.0, 0, 255).astype(np.uint8)
    o = np.ascontiguousarray(o[:, :, ::-1])
    ts.append((time.perf_counter() - t0) * 1000.0)
print("   post-process median %.2f ms  (output shape %s)" % (statistics.median(ts), o.shape))
ts = []
for _ in range(15):
    t0 = time.perf_counter()
    to_input(frame)
    ts.append((time.perf_counter() - t0) * 1000.0)
print("   pre-process  median %.2f ms" % statistics.median(ts))
try:
    import cv2
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        cv2.resize(o, (1080, 720), interpolation=cv2.INTER_AREA)
        ts.append((time.perf_counter() - t0) * 1000.0)
    print("   cv2 downscale 1440x960->1080x720 median %.2f ms" % statistics.median(ts))
except ImportError:
    print("   cv2 not importable from this interpreter")

print("-" * 78)
print("E. ORT profiling: which provider runs each node (CPU fallback hunt)")
so3 = ort.SessionOptions()
prof_dir = os.path.join(ROOT, "tmp-perf")
so3.profile_file_prefix = os.path.join(prof_dir, "ort_profile")
sess_p, _ = build([("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"], so3)
sess_p.run([out_name], {in_name: X})
prof_path = sess_p.end_profiling()
print("   profile:", prof_path)
try:
    with open(prof_path, "r", encoding="utf-8") as fh:
        prof = json.load(fh)
    nodes = {}
    for ev in prof:
        if ev.get("cat") == "Node" and ev.get("name", "").startswith("_"):
            continue
        if ev.get("cat") != "Node":
            continue
        p = ev.get("args", {}).get("provider", "?")
        nodes.setdefault(p, []).append(ev.get("dur", 0))
    for p, durs in sorted(nodes.items(), key=lambda kv: -sum(kv[1])):
        print("   provider %-24s nodes=%4d total=%8.1f ms" % (p, len(durs), sum(durs) / 1000.0))
    slow = sorted(
        (e for e in prof if e.get("cat") == "Node"),
        key=lambda e: -e.get("dur", 0),
    )[:12]
    print("   slowest nodes:")
    for e in slow:
        print("     %8.2f ms  %-40s %s" % (
            e.get("dur", 0) / 1000.0, e.get("name", "")[:40],
            e.get("args", {}).get("provider", "?")))
except Exception as exc:  # noqa: BLE001
    print("   could not parse profile:", exc)

print("=" * 78)
print("SUMMARY ms/frame @%dx%d (720x480 source, model native x4)" % (W, H))
print("  baseline          : %7.1f ms  (%.2f fps)" % (med, 1000 / med))
print("  tuned options     : %7.1f ms  (%.2f fps)" % (med_b, 1000 / med_b))
print("  cuda graph + iob  : %7.1f ms  (%.2f fps)" % (med_c, 1000 / med_c))
print("=" * 78)
