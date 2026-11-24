"""Face tracking: find the speaker in a clip and plan a smooth 9:16 crop that follows them.

How it works
  1. Sample the clip a few times per second and detect faces (OpenCV YuNet model).
  2. Split the clip into camera shots (a hard cut resets the camera instantly).
  3. Inside each shot: if the speaker barely moves, lock the crop (steady, like a tripod);
     if they move, follow with a smoothed path (like a camera operator).
  4. Write one crop position per video frame for FFmpeg (sendcmd file).

Returns None when tracking is not a good idea (no faces, or a wide shot with two people far
apart) - the caller then falls back to the blurred-background layout so nobody is cut off.
"""
import re
import subprocess
from pathlib import Path
import numpy as np
from .config import ROOT, FFMPEG

MODEL = ROOT / "models" / "yunet.onnx"
SAMPLE_FPS = 5


def available():
    try:
        import cv2  # noqa: F401
        return MODEL.exists()
    except Exception:
        return False


def _pick_face(faces, prev_cx):
    """Main face = biggest; among similarly big faces prefer the one we were already following."""
    areas = faces[:, 2] * faces[:, 3]
    big = faces[areas >= 0.6 * areas.max()]
    if prev_cx is not None and len(big) > 1:
        centers = big[:, 0] + big[:, 2] / 2
        return big[np.argmin(np.abs(centers - prev_cx))], len(big)
    return faces[np.argmax(areas)], len(big)


def probe(video: Path):
    """Width, height, fps of a video using FFmpeg itself (works for every codec, incl. AV1)."""
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True,
                         text=True, encoding="utf-8", errors="ignore").stderr
    m = re.search(r"Stream #.*Video:.*?(\d{3,5})x(\d{3,5})", out)
    if not m:
        raise RuntimeError("could not read video size")
    f = re.search(r"([\d.]+) fps", out)
    return int(m.group(1)), int(m.group(2)), float(f.group(1)) if f else 25.0


def analyse(video: Path, start: float, end: float):
    """-> (samples, W, H, fps).  samples: list of dict(t, cx, cy, fh, cut, wide)"""
    import cv2
    W, H, fps = probe(video)
    dw = 640
    dh = int(round(H * dw / W)) // 2 * 2
    scale = W / dw
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (dw, dh), 0.6, 0.3, 50)
    crop_w = H * 9 / 16
    proc = subprocess.Popen(
        [FFMPEG, "-nostdin", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
         "-i", str(video), "-an", "-vf", f"fps={SAMPLE_FPS},scale={dw}:{dh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = dw * dh * 3

    samples, prev_tiny, prev_cx, i = [], None, None, 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        small = np.frombuffer(buf, dtype=np.uint8).reshape(dh, dw, 3)
        t = i / SAMPLE_FPS
        i += 1
        tiny = cv2.cvtColor(cv2.resize(small, (32, 18), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
        cut = prev_tiny is not None and float(np.abs(tiny - prev_tiny).mean()) > 22
        prev_tiny = tiny
        if cut:
            prev_cx = None
        _, faces = det.detect(small)
        s = {"t": t, "cx": None, "cy": None, "fh": None, "cut": cut, "wide": False}
        if faces is not None and len(faces):
            faces = faces[faces[:, 3] > dh * 0.06]              # ignore tiny background faces
        if faces is not None and len(faces):
            f, n_big = _pick_face(faces, prev_cx)
            s["cx"] = float(f[0] + f[2] / 2) * scale
            s["cy"] = float(f[1] + f[3] / 2) * scale
            s["fh"] = float(f[3]) * scale
            prev_cx = s["cx"] / scale
            if n_big > 1:                                       # two+ similar faces: are they too far apart?
                areas = faces[:, 2] * faces[:, 3]
                big = faces[areas >= 0.6 * areas.max()]
                cxs = (big[:, 0] + big[:, 2] / 2) * scale
                s["wide"] = bool((cxs.max() - cxs.min()) > crop_w * 0.75)
        samples.append(s)
    proc.stdout.close()
    proc.wait()
    return samples, W, H, fps


def plan(samples, W, H):
    """Turn raw detections into a smooth list of (t, x_left) keyframes. None = don't track."""
    if not samples:
        return None
    crop_w = int(H * 9 / 16) // 2 * 2
    found = [s for s in samples if s["cx"] is not None]
    if len(found) < 0.35 * len(samples):
        return None                                  # mostly no face (slides, b-roll)
    if sum(s["wide"] for s in samples) > 0.5 * len(samples):
        return None                                  # two people side by side: keep them both (blur layout)

    # split into shots at hard cuts
    shots, cur = [], []
    for s in samples:
        if s["cut"] and cur:
            shots.append(cur)
            cur = []
        cur.append(s)
    shots.append(cur)

    keys, last_x = [], W / 2
    for shot in shots:
        xs = np.array([s["cx"] if s["cx"] is not None else np.nan for s in shot], dtype=float)
        if np.all(np.isnan(xs)):
            path = np.full(len(shot), W / 2 if len(shot) > SAMPLE_FPS else last_x)   # no face in this shot
        else:
            idx = np.arange(len(xs))
            ok = ~np.isnan(xs)
            xs = np.interp(idx, idx[ok], xs[ok])                 # fill gaps
            if len(xs) >= 5:                                     # remove one-off detection glitches
                pad = np.pad(xs, 2, mode="edge")
                xs = np.array([np.median(pad[k:k + 5]) for k in range(len(xs))])
            if xs.max() - xs.min() < 0.14 * crop_w:
                path = np.full(len(xs), float(np.median(xs)))    # steady shot: lock the camera
            else:
                win = max(3, int(SAMPLE_FPS * 1.0) | 1)          # ~1 s moving average
                pad = np.pad(xs, win // 2, mode="edge")
                path = np.convolve(pad, np.ones(win) / win, mode="valid")
                # dead zone: ignore small drifts so the frame doesn't 'breathe'
                out, cam = [], path[0]
                for p in path:
                    if abs(p - cam) > 0.05 * crop_w:
                        cam += (p - cam) * 0.5
                    out.append(cam)
                path = np.array(out)
        for s, cx in zip(shot, path):
            x = int(round(min(max(cx - crop_w / 2, 0), W - crop_w)))
            keys.append((s["t"], x, s["cut"]))
        last_x = float(path[-1])
    return {"crop_w": crop_w, "crop_h": H // 2 * 2, "keys": keys,
            "face_ratio": len(found) / len(samples), "shots": len(shots)}


def write_sendcmd(track, fps, duration, path: Path):
    """One 'crop x' command per frame: linear between keyframes, instant jump on a cut."""
    keys = track["keys"]
    ts = np.array([k[0] for k in keys])
    lines, n = [], int(duration * fps) + 2
    for f in range(n):
        t = f / fps
        j = int(np.searchsorted(ts, t, side="right")) - 1
        j = max(0, min(j, len(keys) - 1))
        x = keys[j][1]
        if j + 1 < len(keys) and not keys[j + 1][2]:             # glide unless next key is a new shot
            t0, t1 = keys[j][0], keys[j + 1][0]
            if t1 > t0:
                x = keys[j][1] + (keys[j + 1][1] - keys[j][1]) * (t - t0) / (t1 - t0)
        lines.append(f"{t:.3f} crop x {int(round(x))};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return keys[0][1]


def track_clip(video: Path, start: float, end: float):
    """-> (track dict, fps) or (None, reason)"""
    if not available():
        return None, "face tracking not installed"
    samples, W, H, fps = analyse(video, start, end)
    track = plan(samples, W, H)
    if track is None:
        return None, "no single speaker to follow"
    track["fps"] = fps
    return track, "ok"
