"""ffmpeg pipes: decode any container to linear-RGB float frames, encode back to H.264 MP4."""
from __future__ import annotations
import json, subprocess
import numpy as np
from render import srgb_to_lin, lin_to_srgb


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out); s = j["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "fps": num / den,
            "duration": float(j["format"].get("duration", 0))}


def read_frames(path, max_height=720, max_seconds=30, size=None, fps=None):
    """size=(W, H) forces an exact frame size (scale to cover, centre-crop) and fps resamples:
    how a cross-echo clip is fitted to the dry clip, frame for frame."""
    info = probe(path)
    W, H = info["width"], info["height"]
    if size:
        W, H = size
        vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}" + (f",fps={fps}" if fps else "")
    else:
        if H > max_height:                     # v1 caps the frame buffer, not the user
            W, H = int(round(W * max_height / H / 2)) * 2, max_height
        vf = f"scale={W}:{H}"
    info.update(width=W, height=H)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(max_seconds), "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def gen():
        n = W * H * 3
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield srgb_to_lin(np.frombuffer(buf, np.uint8).reshape(H, W, 3) / 255.0)
        proc.wait()
    return info, gen()


class Writer:
    def __init__(self, path, W, H, fps, crf=18):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", path],
                                  stdin=subprocess.PIPE)
        self.n = 0

    def write(self, lin):
        self.p.stdin.write((lin_to_srgb(lin) * 255 + 0.5).astype(np.uint8).tobytes()); self.n += 1

    def close(self):
        self.p.stdin.close(); self.p.wait()
        return self.n
