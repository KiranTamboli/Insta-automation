"""'Reelo' - the app's own cartoon host, animated to the narration.

The mouth opens with the loudness of the voice, the eyes blink, and the body bobs while talking.
Want your own character instead? Put PNG files (transparent background, same size) in
assets/character/:  closed.png  open.png  (optional: half.png, blink.png)
"""
import math
import subprocess
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .config import ROOT, FFMPEG

SIZE = (420, 500)            # canvas of the overlay (w, h)
CUSTOM = ROOT / "assets" / "character"


# ---------------------------------------------------------------- drawing the built-in character
def _draw(mouth: int, blink: bool):
    """mouth: 0 closed smile, 1 small, 2 medium, 3 wide. Drawn at 2x then reduced for smooth edges."""
    S = 2
    w, h = 380 * S, 440 * S
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # soft shadow under the body
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([70 * S, 400 * S, 310 * S, 432 * S], fill=(0, 0, 0, 110))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(8 * S)))

    # antenna with a play-button tip
    d.line([190 * S, 78 * S, 190 * S, 30 * S], fill=(255, 255, 255, 255), width=7 * S)
    d.ellipse([168 * S, 4 * S, 212 * S, 48 * S], fill=(255, 176, 58, 255), outline=(255, 255, 255, 255), width=5 * S)
    d.polygon([(184 * S, 16 * S), (184 * S, 36 * S), (201 * S, 26 * S)], fill=(255, 255, 255, 255))

    # body: rounded blob with a vertical pink -> orange gradient
    body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    top, bot = np.array([255, 61, 127]), np.array([255, 140, 66])
    t = np.clip((np.arange(h) - 70 * S) / (340 * S), 0, 1)[:, None]
    col = (top + (bot - top) * t).astype(np.uint8)                       # one colour per row
    rgba = np.concatenate([np.repeat(col[:, None, :], w, axis=1), np.full((h, w, 1), 255, np.uint8)], axis=2)
    grad = Image.fromarray(rgba, "RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([40 * S, 70 * S, 340 * S, 410 * S], radius=130 * S, fill=255)
    body.paste(grad, (0, 0), mask)
    img.alpha_composite(body)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([40 * S, 70 * S, 340 * S, 410 * S], radius=130 * S, outline=(255, 255, 255, 255), width=6 * S)
    # face plate
    d.rounded_rectangle([78 * S, 120 * S, 302 * S, 330 * S], radius=95 * S, fill=(255, 246, 240, 255))
    # cheeks
    d.ellipse([92 * S, 250 * S, 132 * S, 276 * S], fill=(255, 196, 204, 255))
    d.ellipse([248 * S, 250 * S, 288 * S, 276 * S], fill=(255, 196, 204, 255))

    # eyes
    ink = (40, 30, 50, 255)
    for cx in (140, 240):
        if blink:
            d.arc([(cx - 24) * S, 182 * S, (cx + 24) * S, 214 * S], 200, 340, fill=ink, width=7 * S)
        else:
            d.ellipse([(cx - 24) * S, 168 * S, (cx + 24) * S, 222 * S], fill=ink)
            d.ellipse([(cx - 10) * S, 176 * S, (cx + 4) * S, 192 * S], fill=(255, 255, 255, 255))

    # mouth
    if mouth == 0:
        d.arc([160 * S, 246 * S, 220 * S, 290 * S], 20, 160, fill=ink, width=7 * S)
    else:
        oh = {1: 16, 2: 30, 3: 44}[mouth]
        ow = {1: 40, 2: 50, 3: 58}[mouth]
        box = [(190 - ow / 2) * S, 258 * S, (190 + ow / 2) * S, (258 + oh) * S]
        d.ellipse(box, fill=(120, 30, 50, 255))
        if mouth >= 2:                                    # tongue
            d.ellipse([(190 - ow / 3) * S, (258 + oh * 0.55) * S, (190 + ow / 3) * S, (258 + oh + 4) * S],
                      fill=(255, 110, 130, 255))
    # little hand waving a mic
    d.ellipse([300 * S, 290 * S, 352 * S, 342 * S], fill=(255, 150, 70, 255), outline=(255, 255, 255, 255), width=5 * S)
    d.rounded_rectangle([318 * S, 236 * S, 336 * S, 300 * S], radius=8 * S, fill=(50, 50, 60, 255))
    d.ellipse([308 * S, 210 * S, 346 * S, 248 * S], fill=(80, 80, 95, 255), outline=(255, 255, 255, 255), width=3 * S)
    return img.resize((w // S, h // S), Image.LANCZOS)


def _poses():
    """-> dict[(mouth 0-3, blink bool)] = RGBA image."""
    if (CUSTOM / "closed.png").exists() and (CUSTOM / "open.png").exists():
        load = lambda n: Image.open(CUSTOM / n).convert("RGBA")
        closed, opened = load("closed.png"), load("open.png")
        half = load("half.png") if (CUSTOM / "half.png").exists() else opened
        blink = load("blink.png") if (CUSTOM / "blink.png").exists() else closed
        base = {0: closed, 1: half, 2: opened, 3: opened}
        poses = {}
        for m, im in base.items():
            poses[(m, False)] = im
            poses[(m, True)] = blink if m == 0 else im
        return poses
    return {(m, b): _draw(m, b) for m in range(4) for b in (False, True)}


def _fit(img):
    """Scale a pose into the overlay canvas, leaving room to bob."""
    cw, ch = SIZE[0], SIZE[1] - 24
    r = min(cw / img.width, ch / img.height)
    return img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.LANCZOS)


# ---------------------------------------------------------------- animation
def mouth_levels(wav: Path, fps: int):
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    n = int(sr / fps)
    frames = len(data) // n
    rms = np.sqrt((data[: frames * n].reshape(frames, n) ** 2).mean(axis=1)) if frames else np.zeros(1)
    speech = rms[rms > rms.max() * 0.05] if rms.max() > 0 else rms
    q1, q2, q3 = (np.percentile(speech, p) for p in (30, 60, 85)) if len(speech) else (1, 2, 3)
    lv = np.where(rms < rms.max() * 0.06, 0, np.where(rms < q1, 1, np.where(rms < q2, 2, np.where(rms < q3, 2, 3))))
    # smooth: never hold a mouth state for less than 2 frames (looks jittery otherwise)
    for i in range(1, len(lv) - 1):
        if lv[i] != lv[i - 1] and lv[i] != lv[i + 1]:
            lv[i] = lv[i - 1]
    return lv


def render_character(wav: Path, out: Path, duration: float, fps: int = 30):
    """Animated character with transparent background -> .mov (QuickTime Animation, keeps alpha)."""
    poses = {k: _fit(v) for k, v in _poses().items()}
    lv = mouth_levels(wav, fps)
    total = int(math.ceil(duration * fps))
    rng = np.random.default_rng(7)
    blinks, t = set(), 1.2
    while t < duration:                                   # blink every 2.5-4.5 s, 4 frames long
        f0 = int(t * fps)
        blinks.update(range(f0, f0 + 4))
        t += float(rng.uniform(2.5, 4.5))
    cmd = [FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(fps), "-i", "-", "-c:v", "qtrle", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    talk = 0.0
    for f in range(total):
        m = int(lv[f]) if f < len(lv) else 0
        talk = talk * 0.9 + (1.0 if m else 0.0) * 0.1     # how 'into it' the character is right now
        pose = poses[(m, f in blinks)]
        bob = math.sin(f / fps * 2 * math.pi / 0.9) * 7 * talk
        tilt = math.sin(f / fps * 2 * math.pi / 2.3) * 2.2 * talk
        im = pose.rotate(tilt, resample=Image.BICUBIC, expand=False) if abs(tilt) > 0.2 else pose
        canvas = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        x = (SIZE[0] - im.width) // 2
        y = SIZE[1] - im.height - 4 - int(round(12 + bob))
        if f < 8:                                         # pop in
            s = 0.6 + 0.4 * (f + 1) / 8
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            x, y = (SIZE[0] - im.width) // 2, SIZE[1] - im.height - 16
        canvas.alpha_composite(im, (max(0, x), max(0, y)))
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("could not render the character")
    return out


def preview(path: Path):
    """Save one picture of the character (for the dashboard / README)."""
    _draw(2, False).save(path)
    return path
