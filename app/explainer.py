"""Explainer reel: our character tells the viewer what the whole video is about.

  1. AI (Gemini, free) reads the video and writes a short voice-over script, split into 'beats';
     for each beat it also names the moment in the video that should play behind it.
  2. A free AI voice reads each beat (edge-tts), giving exact word timings for captions.
  3. The matching moments are cut from the video (face-tracked, muted) as the background.
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


def _render_broll(source: Path, out: Path, start: float, dur: float, layout: str):
    """A muted 9:16 background piece of exactly `dur` seconds."""
    track = None
    if layout == "track":
        from . import tracker
        try:
            track, _ = tracker.track_clip(source, start, start + dur)
        except Exception:
            track = None
    if track:
        from . import tracker
        cmd_file = out.with_suffix(".cmd")
        x0 = tracker.write_sendcmd(track, track["fps"], dur, cmd_file)
        vf = (f"sendcmd=f={cmd_file.name},crop=w={track['crop_w']}:h={track['crop_h']}:x={x0}:y=0,"
              f"scale={W}:{H}:flags=lanczos,unsharp=5:5:0.6,setsar=1,fps=30")
    elif layout == "crop":
        vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps=30"
    else:
        vf = (f"split=2[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"boxblur=25:5,eq=brightness=-0.15[bgb];[fg]scale={W}:-2[fgs];"
              f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30")
    r = subprocess.run([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(source.resolve()),
                        "-t", f"{dur:.3f}", "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                        "-pix_fmt", "yuv420p", out.name],
                       cwd=str(out.parent), capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError("background clip failed: " + r.stderr[-300:])
    out.with_suffix(".cmd").unlink(missing_ok=True)


def _captions(beats, title, path: Path, per=3):
    head = ASS_HEADER.replace(",520,1\n", ",600,1\n", 1)       # captions a bit higher: the character sits below
    head = head.replace("Style: Hook,", "Style: Label,Arial Black,52,&H00FFFFFF,&H0000FFFF,&H00000000,&HC83D1FFF,1,0,0,0,100,100,0,0,3,14,0,8,60,60,190,1\nStyle: Hook,")
    if any("ऀ" <= ch <= "ॿ" for b in beats for w in b["words"] for ch in w["word"]):
        head = head.replace("Arial Black", "Nirmala UI")
    lines = [head]
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

    say("Cutting the matching moments from the video…")
    vlen = max(s["end"] for s in sents)
    for k, b in enumerate(beats):
        st = max(0.0, min(sents[b["sid"]]["start"] - 0.2, vlen - b["dur"] - 0.5))
        _render_broll(source, work / f"b{k}.mp4", st, b["dur"], layout)
        prog(50 + 35 * (k + 1) / len(beats))
    (work / "bg.txt").write_text("".join(f"file 'b{k}.mp4'\n" for k in range(len(beats))), encoding="utf-8")
    subprocess.run([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "bg.txt",
                    "-c", "copy", "bg.mp4"], cwd=str(work), check=True)

    say("Putting the explainer reel together…")
    _captions(beats, res.get("hook_title") or title, work / "cap.ass")
    HOLD = 0.6
    end = total + HOLD
    vf = (f"[0:v]tpad=stop_mode=clone:stop_duration=2[bg];"
          f"[bg][1:v]overlay=x=W-w-24:y=H-h-36:eof_action=pass:format=auto,subtitles=cap.ass,"
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
