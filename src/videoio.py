"""ffmpeg pipes: decode any container to linear-RGB float frames, encode back to H.264 MP4."""
from __future__ import annotations
import json, os, re, shutil, subprocess
import numpy as np
from render import srgb_to_lin, lin_to_srgb


def _ffmpeg():
    """System ffmpeg if present (dev), else the static build shipped in the imageio-ffmpeg wheel (Vercel)."""
    exe = shutil.which("ffmpeg")
    if exe and not os.environ.get("RCV_BUNDLED_FFMPEG"):
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()


def _probe_ffmpeg(path):
    """ffprobe-free probe: parse `ffmpeg -i` (the bundled build ships ffmpeg only)."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    v = re.search(r"Stream #[^\n]*Video:[^\n]*?(\d{2,5})x(\d{2,5})[^\n]*?([\d.]+) (?:fps|tbr)", err)
    if not v:
        raise ValueError("no video stream")
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else 0.0
    return {"width": int(v.group(1)), "height": int(v.group(2)), "fps": float(v.group(3)), "duration": dur}


def probe(path):
    if not shutil.which("ffprobe") or os.environ.get("RCV_BUNDLED_FFMPEG"):
        return _probe_ffmpeg(path)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out); s = j["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "fps": num / den,
            "duration": float(j["format"].get("duration", 0))}


def read_frames(path, max_height=720, max_seconds=30, size=None, fps=None, max_fps=None):
    """size=(W, H) forces an exact frame size (scale to cover, centre-crop) and fps resamples:
    how a cross-echo clip is fitted to the dry clip, frame for frame. max_fps caps the rate
    (serverless: 60 fps phone clips are resampled to 30, halving the work)."""
    info = probe(path)
    if max_fps and not fps and info["fps"] > max_fps + 0.5:
        fps = max_fps
    W, H = info["width"], info["height"]
    if size:
        W, H = size
        vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}" + (f",fps={fps}" if fps else "")
    else:
        if H > max_height:                     # v1 caps the frame buffer, not the user
            W, H = int(round(W * max_height / H / 2)) * 2, max_height
        vf = f"scale={W}:{H}" + (f",fps={fps}" if fps else "")
    info.update(width=W, height=H, **({"fps": float(fps)} if fps else {}))
    cmd = [FFMPEG, "-v", "error", "-i", path, "-t", str(max_seconds), "-vf", vf,
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
        self.p = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", path],
                                  stdin=subprocess.PIPE)
        self.n = 0

    def write(self, lin):
        self.p.stdin.write((lin_to_srgb(lin) * 255 + 0.5).astype(np.uint8).tobytes()); self.n += 1

    def close(self):
        self.p.stdin.close(); self.p.wait()
        return self.n
