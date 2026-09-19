"""Batching throughput on the fp16 in-graph model built by probe_accel3."""
import statistics
import sys
import time

sys.path.insert(0, r"D:\Draggy-Encoder")
import numpy as np
from src.runtimes.manager import (
    import_local_runtime, inject_runtime_paths, ensure_runtime_dlls_on_path,
)

inject_runtime_paths()
ensure_runtime_dlls_on_path()
ort = import_local_runtime("onnxruntime")
P = r"D:\Draggy-Encoder\tmp-perf\m_fp16_io_720.onnx"
s = ort.InferenceSession(
    P, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                  "CPUExecutionProvider"])
name = s.get_inputs()[0].name
out = s.get_outputs()[0].name
print("io:", s.get_inputs()[0].shape, "->", s.get_outputs()[0].shape)
for b in (1, 2, 4, 8):
    x = np.zeros((b, 480, 720, 3), dtype=np.uint8)
    io = s.io_binding()
    io.bind_cpu_input(name, x)
    io.bind_output(out, "cuda")
    for _ in range(3):
        s.run_with_iobinding(io)
    ts = []
    for _ in range(8):
        t0 = time.perf_counter()
        s.run_with_iobinding(io)
        ts.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(ts)
    print("  batch=%d: %8.2f ms/batch = %6.2f ms/frame (%5.1f fps)" % (b, med, med / b, 1000 * b / med))
