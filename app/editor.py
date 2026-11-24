"""Cut a clip, reframe to 9:16 (1080x1920) and burn in bold animated-style captions."""
import subprocess
from pathlib import Path
from .config import REEL_LAYOUT

W, H = 1080, 1920

ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,84,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,3,2,60,60,520,1
Style: Hook,Arial Black,64,&H0000E5FF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,5,2,8,60,60,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(text):
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(segments, start, end, hook_title, path: Path, words_per_chunk=3):
    words = []
    for s in segments:
        if s["end"] < start or s["start"] > end:
            continue
        if s.get("words"):
            words += [w for w in s["words"] if start <= w["start"] < end]
        else:  # no word timing -> spread segment evenly
            toks = s["text"].split()
            if not toks:
                continue
            step = (s["end"] - s["start"]) / len(toks)
            words += [{"start": s["start"] + i * step, "end": s["start"] + (i + 1) * step, "word": t}
                      for i, t in enumerate(toks) if start <= s["start"] + i * step < end]

    header = ASS_HEADER
    if any("\u0900" <= ch <= "\u097f" for w in words for ch in w["word"]):
        header = header.replace("Arial Black", "Nirmala UI")   # Windows font with Hindi (Devanagari) glyphs
    lines = [header]
    if hook_title:
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(min(3.5, end - start))},Hook,,0,0,0,,{_esc(hook_title.upper())}\n")
    for i in range(0, len(words), words_per_chunk):
        chunk = words[i:i + words_per_chunk]
        cs = chunk[0]["start"] - start
        ce = (words[i + words_per_chunk]["start"] if i + words_per_chunk < len(words) else chunk[-1]["end"]) - start
        # highlight each word in yellow as it is spoken (karaoke \k tags in centiseconds)
        parts = []
        for w in chunk:
            dur = max(1, int(round((w["end"] - w["start"]) * 100)))
            parts.append(f"{{\\k{dur}}}{_esc(w['word']).upper()}")
        text = "{\\fad(60,40)}" + " ".join(parts)
        lines.append(f"Dialogue: 0,{_ts(cs)},{_ts(ce)},Cap,,0,0,0,,{text}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def render_reel(source: Path, out: Path, start: float, end: float, segments, hook_title="", layout=None,
                speech_end=None, progress=None, tail=True, pre_track=None):
    """tail=False renders a middle piece of a stitched reel (no freeze/fade at the end).
    pre_track: a face track computed earlier (dict), or False to force no tracking."""
    layout = layout or REEL_LAYOUT
    out.parent.mkdir(parents=True, exist_ok=True)
    speech_end = min(speech_end or end, end)
    ass = build_ass(segments, start, speech_end, hook_title, out.with_suffix(".ass"))
    dur = end - start
    HOLD = 0.5 if tail else 0.0                           # freeze the last frame in silence, then fade out
    total = dur + HOLD
    if tail:
        vfade = (f",tpad=stop_mode=clone:stop_duration={HOLD},"
                 f"fade=t=out:st={max(0.0, total - 0.45):.2f}:d=0.45")
        afilter = (f"[0:a]afade=t=in:d=0.06,afade=t=out:st={max(0.0, dur - 0.1):.2f}:d=0.1,"
                   f"apad=pad_dur={HOLD}[a]")
    else:
        vfade = ""
        afilter = f"[0:a]afade=t=in:d=0.05,afade=t=out:st={max(0.0, dur - 0.07):.2f}:d=0.07[a]"

    track = None
    if layout == "track":                                 # follow the speaker's face
        from . import tracker
        if pre_track is not None:
            track, why = (pre_track or None), "pieces of this reel need the same layout"
        else:
            try:
                track, why = tracker.track_clip(source, start, end)
            except Exception as e:
                track, why = None, str(e)
        if track is None:
            print(f"[editor] face tracking skipped ({why}) - using blurred background")
            layout = "blur"

    if layout == "track":
        cmdfile = out.with_suffix(".cmd")
        from . import tracker
        x0 = tracker.write_sendcmd(track, track["fps"], dur, cmdfile)
        vf = (f"[0:v]sendcmd=f={cmdfile.name},crop=w={track['crop_w']}:h={track['crop_h']}:x={x0}:y=0,"
              f"scale={W}:{H}:flags=lanczos,unsharp=5:5:0.6,setsar=1,subtitles={ass.name}{vfade}[v]")
    elif layout == "crop":
        vf = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,"
              f"subtitles={ass.name}{vfade}[v]")
    else:  # blurred background + full video in the middle
        vf = (f"[0:v]split=2[bg][fg];"
              f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5,eq=brightness=-0.15[bgb];"
              f"[fg]scale={W}:-2[fgs];"
              f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,subtitles={ass.name}{vfade}[v]")

    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(source.resolve()),
        "-filter_complex", vf + ";" + afilter,
        "-map", "[v]", "-map", "[a]", "-t", f"{total:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        out.name,
    ]
    # run inside the output folder so the subtitles path needs no escaping (Windows-safe)
    proc = subprocess.Popen(cmd, cwd=str(out.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="ignore")
    for line in proc.stdout:
        if progress and line.startswith("out_time_us=") and line.strip()[12:].isdigit():
            progress(min(100.0, int(line.strip()[12:]) / 1e6 / total * 100))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg failed: " + proc.stderr.read()[-400:])
    return layout


def render_stitched(source: Path, out: Path, parts, segments, hook_title="", layout=None, progress=None):
    """Summary reel: several moments from across the video joined with jump cuts.
    parts = [{"start", "end", "speech_end"}, ...] in playing order."""
    layout = layout or REEL_LAYOUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks = [None] * len(parts)
    if layout == "track":                                 # all pieces must share one look
        from . import tracker
        try:
            tracks = [tracker.track_clip(source, p["start"], p["end"])[0] for p in parts]
        except Exception as e:
            print(f"[editor] face tracking skipped ({e})")
            tracks = [None] * len(parts)
        if any(t is None for t in tracks):
            layout, tracks = "blur", [None] * len(parts)
    lens = [p["end"] - p["start"] for p in parts]
    whole, done, names = sum(lens), 0.0, []
    try:
        for i, p in enumerate(parts):
            piece = out.with_name(f"{out.stem}_p{i}.mp4")
            names.append(piece)
            render_reel(source, piece, p["start"], p["end"], segments, hook_title if i == 0 else "", layout,
                        speech_end=p.get("speech_end"), tail=(i == len(parts) - 1),
                        pre_track=(tracks[i] if layout == "track" else False),
                        progress=(lambda pct, d=done, l=lens[i]: progress((d + l * pct / 100) / whole * 97)) if progress else None)
            done += lens[i]
        lst = out.with_name(f"{out.stem}_list.txt")
        lst.write_text("".join(f"file '{n.name}'\n" for n in names), encoding="utf-8")
        names.append(lst)
        r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", lst.name, "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                            "-movflags", "+faststart", out.name],
                           cwd=str(out.parent), capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if r.returncode != 0:
            raise RuntimeError("FFmpeg join failed: " + r.stderr[-300:])
    finally:
        for n in names:                                   # remove the temporary pieces
            for f in (n, n.with_suffix(".ass"), n.with_suffix(".cmd")):
                if f != out and f.exists() and f.name.startswith(out.stem + "_"):
                    f.unlink()
    if progress:
        progress(100)
    return layout
