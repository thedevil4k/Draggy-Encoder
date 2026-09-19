"""Second acceleration probe: the fp16 question + torch tuning, measured through
the app's own engine classes where possible.

Writes only inside tmp-perf/.
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
ort = import_local_runtime("onnxruntime")
import onnx  # noqa: E402

MODEL = os.path.join(ROOT, "bin", "models", "upscale",
                     "realesr-animevideov3", "realesr-animevideov3.onnx")
PTH = os.path.join(ROOT, "bin", "models", "upscale",
                   "realesr-animevideov3", "realesr-animevideov3.pth")
FP16 = os.path.join(OUT, "animevideov3_fp16.onnx")

H, W = 480, 720
rng = np.random.default_rng(1)
BGR = (rng.random((H, W, 3)) * 255).astype(np.uint8)
R = {}

print("=== fp16 conversion ===")
try:
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m32 = onnx.load(MODEL)
    m16 = convert_float_to_float16(m32, keep_io_types=True)
    onnx.save(m16, FP16)
    print("  onnxruntime.transformers.float16 OK ->", os.path.getsize(FP16), "bytes")
    print("  io kept:", [(i.name, i.type.tensor_type.elem_type) for i in m16.graph.input],
          [(o.name, o.type.tensor_type.elem_type) for o in m16.graph.output])
except Exception as exc:  # noqa: BLE001
    print("  ORT fp16 helper failed (%s: %s); trying manual" % (type(exc).__name__, exc))
    from onnx import numpy_helper, helper, TensorProto
    m32 = onnx.load(MODEL)
    g = m32.graph
    for i, init in enumerate(g.initializer):
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32:
            g.initializer[i].CopyFrom(numpy_helper.from_array(a.astype(np.float16), init.name))
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(a.t)
                    if arr.dtype == np.float32:
                        a.t.CopyFrom(numpy_helper.from_array(arr.astype(np.float16)))
    onnx.save(m32, FP16)
    print("  manual fp16 written (Resize may need fp32 scales)")

IN = ["CPUExecutionProvider"]


def sess(path, providers, so=None):
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def bench(label, fn, iters=15, warmup=3, store=True):
    try:
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(ts)
        print("  %-46s %8.2f ms  %6.2f fps" % (label, med, 1000 / med))
        if store:
            R[label] = round(med, 2)
        return med
    except Exception as exc:  # noqa: BLE001
        print("  %-46s FAILED %s: %s" % (label, type(exc).__name__, str(exc)[:150]))
        return None


CUDA_OPTS = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
X = np.ascontiguousarray(np.expand_dims(
    np.transpose(np.ascontiguousarray(BGR[:, :, ::-1]), (2, 0, 1)).astype(np.float32) / 255.0, 0))

print("\n=== A. ONNX Runtime: fp32 vs fp16, at the model's raw level ===")
s32 = sess(MODEL, [("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"])
i32, o32 = s32.get_inputs()[0].name, s32.get_outputs()[0].name
bench("ORT fp32 (raw run)", lambda: s32.run([o32], {i32: X}))
ref = s32.run([o32], {i32: X})[0]

try:
    s16 = sess(FP16, [("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"])
    i16, o16 = s16.get_inputs()[0].name, s16.get_outputs()[0].name
    got = s16.run([o16], {i16: X})[0]
    print("  fp16 vs fp32: max|diff|=%.4f  mean|diff|=%.5f (0-255 scale)" %
          (float(np.abs(got - ref).max()), float(np.abs(got - ref).mean())))
    bench("ORT fp16 (raw run)", lambda: s16.run([o16], {i16: X}))
    so_g = ort.SessionOptions()
    so_g.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s16g = sess(FP16, [("CUDAExecutionProvider", {**CUDA_OPTS, "enable_cuda_graph": "1"}), "CPUExecutionProvider"], so_g)
    io = s16g.io_binding()
    io.bind_cpu_input(s16g.get_inputs()[0].name, X)
    io.bind_output(s16g.get_outputs()[0].name, "cuda")
    bench("ORT fp16 + cuda_graph + iobinding",
          lambda: s16g.run_with_iobinding(io))
    # batch of 4 through the same session (dynamic batch dim)
    X4 = np.ascontiguousarray(np.repeat(X, 4, axis=0))
    bench("ORT fp16 batch=4 (ms per batch of 4)", lambda: s16.run([o16], {i16: X4}), iters=8)
    bench("ORT fp32 batch=4 (ms per batch of 4)", lambda: s32.run([o32], {i32: X4}), iters=8)
except Exception as exc:  # noqa: BLE001
    print("  fp16 bench failed:", type(exc).__name__, str(exc)[:200])

print("\n=== B. through the app's real engine class (process_frame) ===")
from src.ai.inference import ONNXRUNTIMEInferenceEngine  # noqa: E402

eng = ONNXRUNTIMEInferenceEngine(MODEL, "onnxruntime")
print("  load fp32:", eng.load(), "providers:", eng.get_active_providers())
bench("app ONNXRUNTIMEInferenceEngine fp32 process_frame",
      lambda: eng.process_frame(BGR))
try:
    eng16 = ONNXRUNTIMEInferenceEngine(FP16, "onnxruntime")
    print("  load fp16:", eng16.load())
    out = eng16.process_frame(BGR)
    bench("app engine fp16 process_frame (converted model)",
          lambda: eng16.process_frame(BGR))
except Exception as exc:  # noqa: BLE001
    print("  app engine fp16 failed:", type(exc).__name__, str(exc)[:200])

print("\n=== C. PyTorch: where the time really goes (with cuda sync each iter) ===")
try:
    import torch  # noqa: E402
    from src.ai.architectures import build_model_from_state_dict  # noqa: E402
    from src.ai.inference import load_checkpoint_state_dict  # noqa: E402

    dev = torch.device("cuda")
    sd = load_checkpoint_state_dict(PTH)
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    t32 = torch.from_numpy(X).to(dev)

    def mk(half=False, cl=False):
        mdl = build_model_from_state_dict(sd).to(dev).eval()
        if half:
            mdl = mdl.half()
        if cl:
            mdl = mdl.to(memory_format=torch.channels_last)
        return mdl

    def synced(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def bench_t(label, fn, iters=12, warmup=3):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            ts = [synced(fn) for _ in range(iters)]
            med = statistics.median(ts)
            print("  %-46s %8.2f ms  %6.2f fps" % (label, med, 1000 / med))
            R[label] = round(med, 2)
            return med
        except Exception as exc:  # noqa: BLE001
            print("  %-46s FAILED %s: %s" % (label, type(exc).__name__, str(exc)[:150]))
            return None

    m32 = mk()
    bench_t("torch fp32 (cudnn.benchmark=False)", lambda: m32(t32))
    torch.backends.cudnn.benchmark = True
    bench_t("torch fp32 (cudnn.benchmark=True)", lambda: m32(t32))
    torch.backends.cudnn.benchmark = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        bench_t("torch autocast fp16 (app today, benchmark=False)", lambda: m32(t32))
    torch.backends.cudnn.benchmark = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        bench_t("torch autocast fp16 (benchmark=True)", lambda: m32(t32))
    m16 = mk(half=True)
    t16 = t32.half()
    bench_t("torch fp16 NCHW (benchmark=True)", lambda: m16(t16))
    m16c = mk(half=True, cl=True)
    t16c = t16.to(memory_format=torch.channels_last)
    bench_t("torch fp16 channels_last (benchmark=True)", lambda: m16c(t16c))

    def pre_post():
        rgb = np.ascontiguousarray(BGR[:, :, ::-1])
        x = np.ascontiguousarray(np.expand_dims(
            np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0, 0))
        tt = torch.from_numpy(x).to(dev).half().to(memory_format=torch.channels_last)
        with torch.no_grad():
            o = m16c(tt)
        o = (o.squeeze(0).clamp(0, 1) * 255.0).to(torch.uint8)
        o = o.permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(o[:, :, ::-1])

    bench_t("torch fp16 cl + full frame round trip (app-shaped)", pre_post)
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print("  torch section failed:", type(exc).__name__, str(exc)[:200])

print("\n=== C2. the app's PyTorch engine end to end (fp16 autocast, GPU-side cast) ===")
try:
    from src.ai.inference import PyTorchInferenceEngine  # noqa: E402
    engt = PyTorchInferenceEngine(PTH, "torch")
    print("  load torch engine:", engt.load())
    bench("app PyTorchInferenceEngine process_frame (x4)",
          lambda: engt.process_frame(BGR))
except Exception as exc:  # noqa: BLE001
    print("  torch engine failed:", type(exc).__name__, str(exc)[:200])

print("\n=== C3. post-processing cost at the model's REAL x4 output (2880x1920) ===")
out32 = (rng.random((1, 3, 1920, 2880)).astype(np.float32))


def post_ort():
    o = np.squeeze(out32, axis=0)
    o = np.transpose(o, (1, 2, 0))
    o = np.clip(o * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(o[:, :, ::-1])


bench("post-process fp32 2880x1920 (ORT path today)", post_ort)
u8 = post_ort()
try:
    import cv2  # noqa: E402
    bench("cv2 INTER_AREA 2880x1920 -> 1080x720 (uint8)",
          lambda: cv2.resize(u8, (1080, 720), interpolation=cv2.INTER_AREA))
    bench("cv2 INTER_LINEAR 2880x1920 -> 1080x720 (uint8)",
          lambda: cv2.resize(u8, (1080, 720), interpolation=cv2.INTER_LINEAR))
except ImportError:
    print("  cv2 not available")
else:
    big32 = (u8 * 255).astype(np.float32)
    bench("post-process fp32 2880x1920 + cv2 AREA (app today, total)",
          lambda: cv2.resize(post_ort(), (1080, 720), interpolation=cv2.INTER_AREA))

print("\n=== C4. prototype: head/tail inside the ONNX graph (uint8 in, uint8 out) ===")
try:
    from onnx import TensorProto, helper, numpy_helper

    def build_io_model(src_fp16, dst):
        mdl = onnx.load(src_fp16)
        g = mdl.graph
        src_in, src_out = g.input[0].name, g.output[0].name
        raw_in, raw_out = src_in + "_u8", src_out + "_u8"
        # head: uint8 -> fp16 -> NCHW -> * (1/255)
        const_inv = numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np.float16), "inv255")
        const_255 = numpy_helper.from_array(np.array(255.0, dtype=np.float16), "mul255")
        const_0 = numpy_helper.from_array(np.array(0.0, dtype=np.float16), "clipmin")
        const_m = numpy_helper.from_array(np.array(255.0, dtype=np.float16), "clipmax")
        g.initializer.extend([const_inv, const_255, const_0, const_m])
        head = [
            helper.make_node("Cast", [src_in], ["h_f16"], to=TensorProto.FLOAT16),
            helper.make_node("Transpose", ["h_f16"], ["h_nhwc"], perm=[0, 3, 1, 2]),
            helper.make_node("Mul", ["h_nhwc", "inv255"], [raw_in]),
        ]
        # rewrite the original graph's first node input
        for n in g.node:
            for k, nm in enumerate(n.input):
                if nm == src_in:
                    n.input[k] = raw_in
        tail = [
            helper.make_node("Mul", [src_out, "mul255"], ["t_scaled"]),
            helper.make_node("Clip", ["t_scaled", "clipmin", "clipmax"], ["t_clipped"]),
            helper.make_node("Cast", ["t_clipped"], ["t_u8"], to=TensorProto.UINT8),
            helper.make_node("Transpose", ["t_u8"], [raw_out], perm=[0, 2, 3, 1]),
        ]
        for n in g.node:
            for k, nm in enumerate(n.input):
                if nm == src_out:
                    n.input[k] = src_out  # unchanged
        g.node.extend(head + tail)
        del g.value_info[:]
        del g.input[:]
        g.input.extend([helper.make_tensor_value_info(
            src_in, TensorProto.UINT8, ["batch", "height", "width", 3])])
        del g.output[:]
        g.output.extend([helper.make_tensor_value_info(
            raw_out, TensorProto.UINT8, ["batch", "out_h", "out_w", 3])])
        onnx.save(mdl, dst)
        return dst

    io_model = build_io_model(FP16, os.path.join(OUT, "animevideov3_fp16_io.onnx"))
    sio = sess(io_model, [("CUDAExecutionProvider", {**CUDA_OPTS, "enable_cuda_graph": "1"}), "CPUExecutionProvider"])
    print("  io model:", sio.get_inputs()[0].shape, "->", sio.get_outputs()[0].shape)
    ib = sio.io_binding()
    ib.bind_cpu_input(sio.get_inputs()[0].name, BGR)
    ib.bind_output(sio.get_outputs()[0].name, "cuda")
    res = sio.run_with_iobinding(ib)
    ib.synchronize_outputs()
    got_u8 = ib.get_outputs()[0].numpy()
    ref_u8 = np.ascontiguousarray(np.clip(np.transpose(np.squeeze(ref, 0), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)[:, :, ::-1])
    print("  io model shape:", got_u8.shape, "max|diff| vs fp32 CPU post =", int(np.abs(got_u8.astype(int) - ref_u8.astype(int)).max()))
    bench("graph-head/tail fp16 uint8->uint8 (whole frame)", lambda: sio.run_with_iobinding(ib))
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print("  io-model prototype failed:", type(exc).__name__, str(exc)[:300])

print("\n=== D. node placement (is anything silently on CPU?) ===")
try:
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.abspath(os.path.join(OUT, "prof"))
    sp = sess(MODEL, [("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"], so)
    a, b = sp.get_inputs()[0].name, sp.get_outputs()[0].name
    sp.run([b], {a: X})
    path = sp.end_profiling()
    print("  profile:", path)
    with open(path, encoding="utf-8") as fh:
        prof = json.load(fh)
    agg = {}
    for ev in prof:
        if ev.get("cat") != "Node":
            continue
        p = ev.get("args", {}).get("provider", "?")
        agg.setdefault(p, [0, 0.0])
        agg[p][0] += 1
        agg[p][1] += ev.get("dur", 0)
    for k, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print("   %-30s nodes=%4d total=%8.2f ms" % (k, n, d / 1000.0))
    for e in sorted((e for e in prof if e.get("cat") == "Node"), key=lambda e: -e.get("dur", 0))[:8]:
        print("     %8.3f ms %-40s %s" % (e["dur"] / 1000.0, e.get("name", "")[:40],
                                          e.get("args", {}).get("provider", "?")))
except Exception as exc:  # noqa: BLE001
    print("  profiling failed:", type(exc).__name__, str(exc)[:200])

print("\n=== SUMMARY (single frame %dx%d BGR -> x4 model) ===" % (W, H))
for k, v in sorted(R.items(), key=lambda kv: kv[1]):
    print("  %-46s %8.2f ms  %6.2f fps" % (k, v, 1000 / v))
with open(os.path.join(OUT, "probe_results2.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=2, sort_keys=True)
