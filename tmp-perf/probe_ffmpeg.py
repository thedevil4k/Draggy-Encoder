"""Stage-by-stage throughput of the non-AI half of the frame pipeline.

Mimics src/processing/pipeline.py: ffmpeg decode -> bgr24 pipe -> python ->
bgr24 pipe -> ffmpeg encode (hevc_nvenc), with and without the AI step, so we
can see how many ms/frame are NOT inference.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.path.join(ROOT, "bin", "ffmpeg", "ffmpeg.exe")
SRC = r"C:\Users\thedevil4k\Downloads\SHIN CHAN EN LA ISLA DEL TESORO.mkv"
OUT = os.path.join(ROOT, "tmp-perf")
os.makedirs(OUT, exist_ok=True)

SRC_W, SRC_H = 720, 480
DST_W, DST_H = 1080, 720
N = 200  # frames to process per test
SECONDS = N / 25.0


def run(label, cmd, frames=N, stdin=None, stdout=None, feed=None):
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=stdin, stdout=stdout,
                         stderr=subprocess.DEVNULL)
    done = 0
    try:
        if feed is not None:
            for i in range(frames):
                p.stdin.write(feed)
            p.stdin.close()
            p.wait()
            done = frames
    finally:
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=5)
        except Exception:
            pass
    dt = time.perf_counter() - t0
    print("  %-52s %6.2f s  -> %6.1f fps (input-relative)" % (label, dt, done / dt if dt else 0))
    return dt


print("ffmpeg:", FFMPEG)
print("input :", SRC)
print("frames per test:", N, "(%.1f s of video at 25 fps)\n" % SECONDS)

print("=== A. decode only (this is what the pipeline pays before any AI) ===")
run("software decode -> bgr24 -> NUL",
    [FFMPEG, "-hide_banner", "-v", "error", "-t", str(SECONDS), "-i", SRC,
     "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-y", "NUL"])
run("software decode -> yuv420p -> NUL (no color convert)",
    [FFMPEG, "-hide_banner", "-v", "error", "-t", str(SECONDS), "-i", SRC,
     "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-y", "NUL"])
run("software decode -> bgr24 -> file (disk write)",
    [FFMPEG, "-hide_banner", "-v", "error", "-t", str(SECONDS), "-i", SRC,
     "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-y",
     os.path.join(OUT, "decode_dump.raw")])
run("decode -> scale to 1080x720 + bgr24 -> NUL",
    [FFMPEG, "-hide_banner", "-v", "error", "-t", str(SECONDS), "-i", SRC,
     "-map", "0:v:0", "-vf", f"scale={DST_W}:{DST_H}", "-f", "rawvideo",
     "-pix_fmt", "bgr24", "-y", "NUL"])

print("\n=== B. encode only (hevc_nvenc 1600k, bgr24 stdin) ===")
frame_dst = bytes(DST_W * DST_H * 3)
run("bgr24 %dx%d -> hevc_nvenc" % (DST_W, DST_H),
    [FFMPEG, "-hide_banner", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
     "-s", f"{DST_W}x{DST_H}", "-r", "25/1", "-i", "pipe:0",
     "-c:v", "hevc_nvenc", "-b:v", "1600k", "-an", "-y",
     os.path.join(OUT, "enc_test.mp4")],
    stdin=subprocess.PIPE, feed=frame_dst)
frame_src = bytes(SRC_W * SRC_H * 3)
run("bgr24 %dx%d -> hevc_nvenc (no scale)" % (SRC_W, SRC_H),
    [FFMPEG, "-hide_banner", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
     "-s", f"{SRC_W}x{SRC_H}", "-r", "25/1", "-i", "pipe:0",
     "-c:v", "hevc_nvenc", "-b:v", "1600k", "-an", "-y",
     os.path.join(OUT, "enc_test2.mp4")],
    stdin=subprocess.PIPE, feed=frame_src)

print("\n=== C. full loop without AI (decode -> python -> encode) ===")
frame_size = SRC_W * SRC_H * 3
dec = subprocess.Popen(
    [FFMPEG, "-hide_banner", "-v", "error", "-t", str(SECONDS), "-i", SRC,
     "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
enc = subprocess.Popen(
    [FFMPEG, "-hide_banner", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
     "-s", f"{SRC_W}x{SRC_H}", "-r", "25/1", "-i", "pipe:0",
     "-c:v", "hevc_nvenc", "-b:v", "1600k", "-an", "-y",
     os.path.join(OUT, "loop_test.mp4")],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.perf_counter()
n = 0
while True:
    raw = dec.stdout.read(frame_size)
    if not raw or len(raw) < frame_size:
        break
    enc.stdin.write(raw)
    n += 1
enc.stdin.close()
enc.wait()
dec.stdout.close()
dec.wait()
dt = time.perf_counter() - t0
print("  %-52s %6.2f s  -> %6.1f fps (frames=%d)" % ("decode -> memcpy -> encode", dt, n / dt, n))

print("\n=== D. what the AI loop does to the frame in python (720x480 -> 1440x960) ===")
import numpy as np  # noqa: E402

raw = dec_bytes = None
dec = subprocess.Popen(
    [FFMPEG, "-hide_banner", "-v", "error", "-t", "1", "-i", SRC,
     "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
raw = dec.stdout.read(frame_size)
dec.stdout.close()
dec.wait()
frame = np.frombuffer(raw, dtype=np.uint8).reshape((SRC_H, SRC_W, 3))
big = np.zeros((960, 1440, 3), dtype=np.uint8)
t0 = time.perf_counter()
for _ in range(50):
    out = np.ascontiguousarray(big[:, :, ::-1])
    inp = np.ascontiguousarray(np.transpose(out, (2, 0, 1)).astype(np.float32) / 255.0)
    inp = np.ascontiguousarray(np.expand_dims(inp, 0))
print("  %-52s %8.2f ms/frame" % ("pre-process (BGR->RGB->NCHW fp32)", (time.perf_counter() - t0) / 50 * 1000))
t0 = time.perf_counter()
for _ in range(50):
    o = np.squeeze(big, axis=0)
    o = np.transpose(o, (1, 2, 0))
    o = np.clip(o * 255.0, 0, 255).astype(np.uint8)
    o = np.ascontiguousarray(o[:, :, ::-1])
print("  %-52s %8.2f ms/frame" % ("post-process (NCHW fp32->BGR uint8)", (time.perf_counter() - t0) / 50 * 1000))
try:
    import cv2  # noqa: E402
    src_img = np.zeros((960, 1440, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(50):
        cv2.resize(src_img, (DST_W, DST_H), interpolation=cv2.INTER_AREA)
    print("  %-52s %8.2f ms/frame" % ("cv2 INTER_AREA 1440x960 -> 1080x720", (time.perf_counter() - t0) / 50 * 1000))
except ImportError:
    print("  cv2 not available in this interpreter")
t0 = time.perf_counter()
for _ in range(50):
    big.tobytes()
print("  %-52s %8.2f ms/frame" % ("tobytes() 1440x960x3 (pipe write buffer)", (time.perf_counter() - t0) / 50 * 1000))
