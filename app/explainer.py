"""Explainer reel: our character tells the viewer what the whole video is about.

  1. AI (Gemini, free) reads the video and writes a short voice-over script, split into 'beats';
     for each beat it also names the moment in the video that should play behind it.
  2. A free AI voice reads each beat (edge-tts), giving exact word timings for captions.
  3. Still pictures from the matching moments (preferring ones without a talking face, so nobody's
     lips move out of sync with the voice) are shown as framed cards with a slow zoom.
  4. Our animated character talks in the corner; captions follow the voice word by word.
Nothing to write, record or pay for.
"""
import os
import subprocess
from pathlib import Path
from . import narrator, character
from .config import FFMPEG, REEL_LAYOUT
from .editor import W, H, ASS_HEADER, _ts, _esc

CHARACTER_NAME = os.getenv("CHARACTER_NAME", "Reelo")

SCRIPT_PROMPT = """You are {name}, a friendly, energetic explainer character on Instagram. You watch long videos
so your followers don't have to, and you explain in under a minute what the video actually says.

Video title: {title}
{profile}
Transcript as numbered lines:  [S<id> @<start seconds> | <length>s] text
{transcript}

Write the voice-over for ONE reel that explains this video to someone who has never seen it.
Language: {language}{language_note}

Structure, as 6 to 8 beats:
  1. HOOK - one punchy line that makes people stop scrolling (a bold claim or a question from the video).
  2. CONTEXT - who is speaking and what the video is about ("In this video, <speaker> explains ...").
  3-6. KEY POINTS - the 3 or 4 most important ideas, each with one concrete detail from the video
       (a number, an example, a name, a short quote). Plain words, like telling a friend.
  LAST. TAKEAWAY - the one thing to remember, then "Follow for more" (in the same language).
Rules:
- 110 to 150 words in total; each beat 8 to 26 words, 1-2 short spoken sentences.
- Only facts that are in the transcript. Never invent numbers, names or claims.
- Spoken style: contractions, short sentences, no emoji, no hashtags, no stage directions.
- For every beat, "show_id" = the line S<id> whose moment in the video best illustrates that beat
  (the speaker saying it, or the story being told). Use different lines for different beats.
- "label" = 1 to 3 words shown on screen during the beat (e.g. "The real problem", "Tip #1").

Return ONLY JSON:
{{"hook_title": "on-screen title, max 7 words",
  "caption": "Instagram caption, 1-2 lines, ends with a question",
  "hashtags": "#5 #to #8 #relevant #hashtags",
  "beats": [{{"say": "...", "show_id": 12, "label": "..."}}]}}"""


def write_script(sents, title, profile, voice, status=None):
    from . import ai
    from .viral import _sent_lines
    lang = narrator.language_of(voice)
    note = (" - write in Devanagari script, simple everyday Hindi (English words people use daily are fine)"
            if lang == "Hindi" else "")
    if status:
        status(f"{CHARACTER_NAME} is writing the explainer script…")
    res = ai.ask_json(SCRIPT_PROMPT.format(name=CHARACTER_NAME, title=title, profile=profile or "",
                                           transcript=_sent_lines(sents, 0, len(sents) - 1)[:400_000],
                                           language=lang, language_note=note))
    if not isinstance(res, dict) or not res.get("beats"):
        raise RuntimeError("the AI did not return a script")
    beats, used = [], set()
    for b in res["beats"][:9]:
        say = " ".join(str(b.get("say", "")).split())
        if len(say.split()) < 3:
            continue
        try:
            sid = max(0, min(int(b.get("show_id", 0)), len(sents) - 1))
        except (TypeError, ValueError):
            sid = None
        if sid is None or sid in used:                     # missing / repeated: spread evenly instead
            sid = min(len(sents) - 1, int(len(sents) * (len(beats) + 0.5) / 8))
        used.add(sid)
        beats.append({"say": say, "sid": sid, "label": str(b.get("label", ""))[:24]})
    if len(beats) < 3:
        raise RuntimeError("the script was too short")
    return res, beats


# ---------------------------------------------------------------- picture background (no lip-sync problems)
CARD_Y = 360                     # top of the picture card
FPS = 30


def _frames_around(source: Path, t0: float, t1: float, step=1.0, w=320):
    """Small frames every `step` seconds between t0 and t1 -> list of (time, BGR array)."""
    import numpy as np
    from .tracker import probe
    vw, vh, _ = probe(source)
    h = int(round(vh * w / vw)) // 2 * 2
    p = subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-ss", f"{t0:.2f}", "-t", f"{t1 - t0:.2f}",
                        "-i", str(source), "-an", "-vf", f"fps={1 / step},scale={w}:{h}", "-f", "rawvideo",
                        "-pix_fmt", "bgr24", "-"], capture_output=True)
    n = w * h * 3
    return [(t0 + (i + 0.5) * step, np.frombuffer(p.stdout[i * n:(i + 1) * n], np.uint8).reshape(h, w, 3))
            for i in range(len(p.stdout) // n)]


def _score_frames(frames):
    """Higher = better background picture: sharp, not dark, and preferably WITHOUT a talking face."""
    import cv2
    from .tracker import MODEL
    det = None
    out = []
    for t, img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright, sharp = float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if bright < 28 or gray.std() < 12:                  # black / blank / fade frame
            continue
        face = 0.0
        if MODEL.exists():
            if det is None:
                det = cv2.FaceDetectorYN.create(str(MODEL), "", (img.shape[1], img.shape[0]), 0.6, 0.3, 20)
            _, faces = det.detect(img)
            if faces is not None and len(faces):
                face = float((faces[:, 2] * faces[:, 3]).max()) / (img.shape[0] * img.shape[1])
        score = min(sharp, 1500) / 300 + (4.0 if face == 0 else -2.0 - 12 * face)
        out.append((score, t, face))
    return out


def pick_pictures(source: Path, beats, sents, vlen):
    """For each beat choose 1-2 still moments near what it talks about, avoiding the speaker's face."""
    chosen = []
    for b in beats:
        center = sents[b["sid"]]["start"]
        t0, t1 = max(0.0, center - 8), min(vlen - 0.5, center + 14)
        cands = sorted(_score_frames(_frames_around(source, t0, t1)), reverse=True)
        want = 1 if b["dur"] < 6.5 else 2
        picks = []
        for score, t, face in cands:
            if all(abs(t - u) >= 3 for u in chosen + picks):
                picks.append(t)
            if len(picks) >= want:
                break
        if not picks:                                       # nothing usable: any frame at the moment itself
            picks = [min(vlen - 0.5, center + 0.5)]
        b["pics"] = sorted(picks)
        chosen += picks


def _card_size(vw, vh, max_w=960, max_h=820):
    r = min(max_w / vw, max_h / vh)
    return int(vw * r) // 2 * 2, int(vh * r) // 2 * 2


def render_background(source: Path, beats, out: Path, status=None):
    """All beats' pictures -> one silent 1080x1920 video. Each picture is a framed card with a slow
    zoom/pan over a blurred copy of itself; 6-frame crossfade between pictures."""
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
    from .tracker import probe
    vw, vh, _ = probe(source)
    cw, ch = _card_size(vw, vh)
    cx = (W - cw) // 2
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw - 1, ch - 1], radius=28, fill=255)
    frame_png = out.with_name("shot.png")
    proc = subprocess.Popen([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "20", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    prev, k = None, 0
    try:
        for b in beats:
            shots = b["pics"]
            seg = b["dur"] / len(shots)
            for t in shots:
                subprocess.run([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(source),
                                "-frames:v", "1", str(frame_png)], check=True)
                pic = Image.open(frame_png).convert("RGB")
                # backdrop: the same picture, blurred and darkened, filling the screen
                r = max(W / pic.width, H / pic.height)
                bg = pic.resize((int(pic.width * r) + 1, int(pic.height * r) + 1), Image.BILINEAR)
                bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
                bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(36))).enhance(0.55)
                # white frame + soft shadow around the card
                deco = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ImageDraw.Draw(sh).rounded_rectangle([cx - 6, CARD_Y + 18, cx + cw + 6, CARD_Y + ch + 30], radius=36, fill=(0, 0, 0, 150))
                deco.alpha_composite(sh.filter(ImageFilter.GaussianBlur(18)))
                ImageDraw.Draw(deco).rounded_rectangle([cx - 8, CARD_Y - 8, cx + cw + 7, CARD_Y + ch + 7], radius=34,
                                                       fill=(255, 255, 255, 235))
                base = bg.convert("RGBA")
                base.alpha_composite(deco)
                base = base.convert("RGB")
                src = pic.resize((int(cw * 1.25), int(ch * 1.25)), Image.LANCZOS)
                n = max(1, int(round(seg * FPS)))
                zoom_in = k % 2 == 0
                pan = (1 if k % 4 < 2 else -1) * 0.04 * src.width
                for f in range(n):
                    u = f / max(1, n - 1)
                    e = u * u * (3 - 2 * u)                  # ease in-out
                    z = 1.0 + 0.10 * (e if zoom_in else 1 - e)
                    w2, h2 = src.width / z, src.height / z
                    mx = min(max(src.width / 2 + pan * (e - 0.5), w2 / 2), src.width - w2 / 2)   # never show an edge
                    my = src.height / 2
                    box = (mx - w2 / 2, my - h2 / 2, mx + w2 / 2, my + h2 / 2)
                    card = src.transform((cw, ch), Image.EXTENT, box, Image.BICUBIC)
                    img = base.copy()
                    img.paste(card, (cx, CARD_Y), mask)
                    if prev is not None and f < 6:          # crossfade from the previous picture
                        img = Image.blend(prev, img, (f + 1) / 7)
                    proc.stdin.write(img.tobytes())
                    if f == n - 1:
                        prev = img
                k += 1
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("background render failed")
    finally:
        frame_png.unlink(missing_ok=True)
    return CARD_Y + ch


def _captions(beats, title, path: Path, per=3, credit="", credit_y=0):
    head = ASS_HEADER.replace(",520,1\n", ",600,1\n", 1)       # captions a bit higher: the character sits below
    head = head.replace("Style: Hook,", "Style: Label,Arial Black,52,&H00FFFFFF,&H0000FFFF,&H00000000,&HC83D1FFF,1,0,0,0,100,100,0,0,3,14,0,8,60,60,190,1\nStyle: Hook,")
    if any("ऀ" <= ch <= "ॿ" for b in beats for w in b["words"] for ch in w["word"]):
        head = head.replace("Arial Black", "Nirmala UI")
    head = head.replace("Style: Hook,", "Style: Src,Arial,30,&H00E6E6E6,&H00E6E6E6,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,2,1,8,80,80,0,1\nStyle: Hook,")
    lines = [head]
    if credit and credit_y:
        total = beats[-1]["t1"] + 0.6
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(total)},Src,,0,0,0,,{{\\an8\\pos(540,{credit_y + 22})}}{_esc(credit[:70])}\n")
    if title:
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(min(3.2, beats[0]['t1']))},Hook,,0,0,0,,{_esc(title.upper())}\n")
    for k, b in enumerate(beats):
        if b["label"] and k > 0:
            lines.append(f"Dialogue: 1,{_ts(b['t0'])},{_ts(b['t1'])},Label,,0,0,0,,{_esc(b['label'].upper())}\n")
        ws = b["words"]
        for i in range(0, len(ws), per):
            chunk = ws[i:i + per]
            cs = b["t0"] + chunk[0]["start"]
            ce = b["t0"] + (ws[i + per]["start"] if i + per < len(ws) else max(chunk[-1]["end"], b["dur"] - 0.05))
            parts = [f"{{\\k{max(1, int(round((w['end'] - w['start']) * 100)))}}}{_esc(w['word']).upper()}" for w in chunk]
            lines.append(f"Dialogue: 0,{_ts(cs)},{_ts(ce)},Cap,,0,0,0,,{{\\fad(50,30)}}{' '.join(parts)}\n")
    path.write_text("".join(lines), encoding="utf-8")


def make_explainer(source: Path, out: Path, segments, title, profile="", voice=None, layout=None,
                   status=None, progress=None):
    """-> dict with title/caption/hashtags/reason/length, or raises."""
    from .viral import build_sentences
    voice = voice or os.getenv("NARRATOR_VOICE") or narrator.DEFAULT_VOICE
    layout = layout or REEL_LAYOUT
    say = status or (lambda m: None)
    prog = progress or (lambda p: None)
    sents, _, _ = build_sentences(segments)
    res, beats = write_script(sents, title, profile, voice, say)
    prog(10)

    work = out.parent / f"{out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    GAP = 0.18
    say(f"{CHARACTER_NAME} is recording the voice-over…")
    t, engine = 0.0, "edge"
    for k, b in enumerate(beats):
        b["words"], b["speech"], engine = narrator.speak(b["say"], voice, work / f"v{k}.wav")
        b["dur"] = b["speech"] + GAP
        b["t0"], b["t1"] = t, t + b["dur"]
        t += b["dur"]
        prog(10 + 25 * (k + 1) / len(beats))
    total = t

    lst = work / "voice.txt"
    lst.write_text("".join(f"file 'v{k}.wav'\n" for k in range(len(beats))), encoding="utf-8")
    pads = "".join(f"[{k}:a]apad=pad_dur={GAP}[a{k}];" for k in range(len(beats)))
    ins = sum([["-i", f"v{k}.wav"] for k in range(len(beats))], [])
    subprocess.run([FFMPEG, "-nostdin", "-y", "-loglevel", "error", *ins, "-filter_complex",
                    pads + "".join(f"[a{k}]" for k in range(len(beats))) + f"concat=n={len(beats)}:v=0:a=1[o]",
                    "-map", "[o]", "-ar", "44100", "-ac", "1", "voice.wav"], cwd=str(work), check=True)

    say(f"Animating {CHARACTER_NAME}…")
    character.render_character(work / "voice.wav", work / "char.mov", total)
    prog(50)

    say("Choosing pictures from the video…")
    vlen = max(x["end"] for x in sents)
    pick_pictures(source, beats, sents, vlen)
    prog(62)
    say("Building the picture background…")
    card_bottom = render_background(source, beats, work / "bg.mp4")
    prog(88)

    say("Putting the explainer reel together…")
    _captions(beats, res.get("hook_title") or title, work / "cap.ass", credit=f"From: {title}", credit_y=card_bottom)
    HOLD = 0.6
    end = total + HOLD
    vf = (f"[0:v]tpad=stop_mode=clone:stop_duration=2[bg];"
          f"[bg][1:v]overlay=x=(W-w)/2:y=H-h-30:eof_action=pass:format=auto,subtitles=cap.ass,"
          f"fade=t=out:st={end - 0.45:.2f}:d=0.45[v];[2:a]apad=pad_dur={HOLD},afade=t=out:st={end - 0.3:.2f}:d=0.3[a]")
    r = subprocess.run([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-i", "bg.mp4", "-i", "char.mov",
                        "-i", "voice.wav", "-filter_complex", vf, "-map", "[v]", "-map", "[a]", "-t", f"{end:.2f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
                        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-movflags", "+faststart", str(out.resolve())],
                       cwd=str(work), capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError("final render failed: " + r.stderr[-300:])
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    prog(100)
    return {"title": res.get("hook_title") or title, "caption": res.get("caption") or "",
            "hashtags": res.get("hashtags") or "", "length": end, "engine": engine,
            "reason": f"🎙️ Explainer by {CHARACTER_NAME} · {len(beats)} beats · "
                      + (beats[0]['say'][:80] if beats else "")}
