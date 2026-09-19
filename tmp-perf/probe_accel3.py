"""Third probe: can we move the per-frame colour/scale math out of numpy?

Prototypes (all built inside tmp-perf/, nothing else touched):
  T1  fp32 model + in-graph head/tail   -> uint8 BGR in, uint8 BGR out (x4 size)
  T2  T1 + in-graph downscale to 1080x720 -> uint8 BGR out at the encoder size
  T3  fp16 model, keep_io_types=False, same head/tail
  T4  torch fp16 + channels_last + GPU resize to 1080x720 (ideal torch path)
  plus PIL vs ffmpeg downscale cost, and the app engine as reference.
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
    import_local_runtime, inject_runtime_paths, ensure_runtime_dlls_on_path,
)
inject_runtime_paths()
ensure_runtime_dlls_on_path()
ort = import_local_runtime("onnxruntime")
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

MODEL = os.path.join(ROOT, "bin", "models", "upscale", "realesr-animevideov3", "realesr-animevideov3.onnx")
PTH = os.path.join(ROOT, "bin", "models", "upscale", "realesr-animevideov3", "realesr-animevideov3.pth")
FP16 = os.path.join(OUT, "animevideov3_fp16.onnx")
H, W, TW, TH = 480, 720, 1080, 720
rng = np.random.default_rng(7)
BGR = (rng.random((H, W, 3)) * 255).astype(np.uint8)
R = {}
OPTS = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}


def bench(label, fn, iters=12, warmup=3):
    try:
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(ts)
        print("  %-52s %8.2f ms  %6.2f fps" % (label, med, 1000 / med))
        R[label] = round(med, 2)
        return med
    except Exception as exc:  # noqa: BLE001
        print("  %-52s FAILED %s: %s" % (label, type(exc).__name__, str(exc)[:130]))
        return None


def make_io_model(src_model, dst, out_sizes=None, half=False):
    """Wrap a model: uint8 BGR NHWC in -> [Mul/Clip/Cast/(Resize)/Transpose] -> uint8 BGR NHWC out."""
    mdl = onnx.load(src_model)
    g = mdl.graph
    src_in, src_out = g.input[0].name, g.output[0].name
    dt = np.float16 if half else np.float32
    tt = TensorProto.FLOAT16 if half else TensorProto.FLOAT
    g.initializer.extend([
        numpy_helper.from_array(np.array(1.0 / 255.0, dtype=dt), "w_inv255"),
        numpy_helper.from_array(np.array(255.0, dtype=dt), "w_255"),
        numpy_helper.from_array(np.array(0.0, dtype=dt), "w_min"),
        numpy_helper.from_array(np.array(255.0, dtype=dt), "w_max"),
    ])
    if out_sizes is not None:
        g.initializer.append(numpy_helper.from_array(
            np.array([1, 3, out_sizes[1], out_sizes[0]], dtype=np.int64), "w_out_sizes"))
    head = [
        helper.make_node("Cast", [src_in], ["h_f"], to=tt),
        helper.make_node("Transpose", ["h_f"], ["h_chw"], perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["h_chw", "w_inv255"], ["h_norm"]),
    ]
    for n in g.node:
        for k, nm in enumerate(n.input):
            if nm == src_in:
                n.input[k] = "h_norm"
    tail = [
        helper.make_node("Mul", [src_out, "w_255"], ["t_mul"]),
        helper.make_node("Clip", ["t_mul", "w_min", "w_max"], ["t_clip"]),
    ]
    last = "t_clip"
    if out_sizes is not None:
        tail.append(helper.make_node("Resize", [last, "", "", "w_out_sizes"], ["t_rs"],
                                     mode="linear", coordinate_transformation_mode="half_pixel"))
        last = "t_rs"
    tail += [
        helper.make_node("Cast", [last], ["t_u8"], to=TensorProto.UINT8),
        helper.make_node("Transpose", ["t_u8"], ["t_out"], perm=[0, 2, 3, 1]),
    ]
    g.node.extend(head + tail)
    del g.value_info[:]
    del g.input[:]
    g.input.extend([helper.make_tensor_value_info(src_in, TensorProto.UINT8, ["n", "h", "w", 3])])
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("t_out", TensorProto.UINT8, ["n", "oh", "ow", 3])])
    onnx.save(mdl, dst)
    return dst


def io_session(path, size_hw, use_cuda_graph=True):
    providers = [("CUDAExecutionProvider", {**OPTS, "enable_cuda_graph": "1"} if use_cuda_graph else OPTS),
                 "CPUExecutionProvider"]
    s = ort.InferenceSession(path, providers=providers)
    ib = s.io_binding()
    buf = np.zeros((1, size_hw[0], size_hw[1], 3), dtype=np.uint8)
    ib.bind_cpu_input(s.get_inputs()[0].name, buf)
    ib.bind_output(s.get_outputs()[0].name, "cuda")
    return s, ib, buf


print("=== reference: app engine today ===")
from src.ai.inference import ONNXRUNTIMEInferenceEngine, PyTorchInferenceEngine  # noqa: E402
eng = ONNXRUNTIMEInferenceEngine(MODEL, "onnxruntime")
eng.load()
bench("app ORT engine fp32 process_frame (x4 out)", lambda: eng.process_frame(BGR))
app_out = eng.process_frame(BGR)
print("    app engine output shape:", app_out.shape)

try:
    import cv2
    have_cv2 = True
except ImportError:
    have_cv2 = False
print("  cv2 available:", have_cv2)
if not have_cv2:
    from PIL import Image as PILImage
    bench("PIL BILINEAR 2880x1920 -> 1080x720 (app fallback)",
          lambda: np.asarray(PILImage.fromarray(app_out).resize((TW, TH), PILImage.Resampling.BILINEAR)))

print("\n=== T1. fp32 + in-graph head/tail (uint8 in, uint8 x4 out) ===")
try:
    p1 = make_io_model(MODEL, os.path.join(OUT, "m_fp32_io.onnx"), out_sizes=None)
    s1, ib1, buf1 = io_session(p1, (H, W))
    print("  io:", s1.get_inputs()[0].shape, "->", s1.get_outputs()[0].shape)
    o1 = s1.run_with_iobinding(ib1)
    ib1.synchronize_outputs()
    print("  out shape:", ib1.get_outputs()[0].numpy().shape,
          "| max diff vs app engine:", int(np.abs(ib1.get_outputs()[0].numpy().astype(int) - app_out.astype(int)).max()))
    bench("T1 ORT fp32 uint8-in/uint8-out (x4)", lambda: s1.run_with_iobinding(ib1))
except Exception as exc:  # noqa: BLE001
    print("  T1 failed:", type(exc).__name__, str(exc)[:200])

print("\n=== T2. + downscale to 1080x720 INSIDE the graph (GPU) ===")
try:
    p2 = make_io_model(MODEL, os.path.join(OUT, "m_fp32_io_720.onnx"), out_sizes=(TW, TH))
    s2, ib2, buf2 = io_session(p2, (H, W))
    print("  io:", s2.get_inputs()[0].shape, "->", s2.get_outputs()[0].shape)
    s2.run_with_iobinding(ib2)
    ib2.synchronize_outputs()
    got = ib2.get_outputs()[0].numpy()
    if not have_cv2:
        ref = np.asarray(PILImage.fromarray(app_out).resize((TW, TH), PILImage.Resampling.BILINEAR))
    else:
        ref = cv2.resize(app_out, (TW, TH), interpolation=cv2.INTER_LINEAR)
    print("  out shape:", got.shape, "| max diff vs CPU linear downscale:",
          int(np.abs(got.astype(int) - ref.astype(int)).max()))
    bench("T2 ORT fp32 uint8-in/uint8-out + GPU resize", lambda: s2.run_with_iobinding(ib2))
    bench("T2 + tobytes write buffer", lambda: (s2.run_with_iobinding(ib2), ib2.synchronize_outputs(),
                                                ib2.get_outputs()[0].numpy().tobytes()))
except Exception as exc:  # noqa: BLE001
    print("  T2 failed:", type(exc).__name__, str(exc)[:200])

print("\n=== T3. fp16 (keep_io_types=False) + head/tail ===")
try:
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m16 = convert_float_to_float16(onnx.load(MODEL), keep_io_types=False)
    onnx.save(m16, FP16)
    for tag, kwargs in (("T3a fp16 raw", None), ("T3b fp16 head/tail", "io")):
        if kwargs is None:
            s = ort.InferenceSession(FP16, providers=[("CUDAExecutionProvider", OPTS), "CPUExecutionProvider"])
            i, o = s.get_inputs()[0].name, s.get_outputs()[0].name
            X = np.ascontiguousarray(np.expand_dims(
                np.transpose(np.ascontiguousarray(BGR[:, :, ::-1]), (2, 0, 1)).astype(np.float16) / 255.0, 0))
            bench(tag, lambda s=s, i=i, o=o, X=X: s.run([o], {i: X}))
        else:
            p3 = make_io_model(FP16, os.path.join(OUT, "m_fp16_io_720.onnx"), out_sizes=(TW, TH), half=True)
            s3, ib3, buf3 = io_session(p3, (H, W))
            s3.run_with_iobinding(ib3)
            bench("T3b fp16 uint8-in/uint8-out + GPU resize", lambda: s3.run_with_iobinding(ib3))
except Exception as exc:  # noqa: BLE001
    print("  T3 failed:", type(exc).__name__, str(exc)[:250])

print("\n=== T4. torch fp16 channels_last + GPU resize (ideal torch path) ===")
try:
    import torch  # noqa: E402
    from src.ai.architectures import build_model_from_state_dict  # noqa: E402
    from src.ai.inference import load_checkpoint_state_dict  # noqa: E402
    dev = torch.device("cuda")
    sd = load_checkpoint_state_dict(PTH)
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    mdl = build_model_from_state_dict(sd).to(dev).eval().half().to(memory_format=torch.channels_last)

    def torch_full(use_cl=True, resize_gpu=True):
        # BGR uint8 -> GPU fp16 NCHW (+channels_last) -> model -> resize -> uint8 NHWC BGR
        t = torch.from_numpy(np.ascontiguousarray(BGR[:, :, ::-1])).to(dev)
        t = t.permute(2, 0, 1).unsqueeze(0).half().div_(255.0)
        if use_cl:
            t = t.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            o = mdl(t)
        o = o.clamp_(0, 1).mul_(255.0)
        if resize_gpu:
            o = torch.nn.functional.interpolate(o, size=(TH, TW), mode="bilinear", align_corners=False)
        o = o.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()[0]
        return o

    out = torch_full()
    print("  out shape:", out.shape, "dtype", out.dtype)
    bench("T4 torch fp16 cl + GPU resize + uint8 D2H", torch_full)
    bench("T4b same but NCHW (no channels_last)", lambda: torch_full(use_cl=False))
    bench("T4c torch fp16 cl, no GPU resize (x4 uint8 D2H)",
          lambda: torch_full(resize_gpu=False))
except Exception as exc:  # noqa: BLE001
    print("  T4 failed:", type(exc).__name__, str(exc)[:250])

print("\n=== SUMMARY ===")
for k, v in sorted(R.items(), key=lambda kv: kv[1]):
    print("  %-52s %8.2f ms  %6.2f fps" % (k, v, 1000 / v))
with open(os.path.join(OUT, "probe_results3.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=2, sort_keys=True)
