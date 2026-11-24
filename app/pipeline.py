"""YouTube link -> viral clips -> rendered 9:16 reels -> (optional) auto schedule."""
import json
import threading
import time
import traceback
from datetime import datetime, timedelta
from . import db
from .config import REELS_DIR, REEL_LAYOUT
from .downloader import download, fetch_captions, load_heatmap
from .transcribe import extract_audio, transcribe, loudness_per_second
from .viral import find_viral_moments
from .editor import render_reel

_last_write = {}


def _status(job_id, status, message="", progress=None, force=True):
    """Update job state. Progress-only updates are throttled to ~2 per second."""
    now = time.time()
    if not force and now - _last_write.get(job_id, 0) < 0.5:
        return
    _last_write[job_id] = now
    if progress is None:
        db.x("UPDATE jobs SET status=?, message=? WHERE id=?", (status, message, job_id))
    else:
        db.x("UPDATE jobs SET status=?, message=?, progress=MAX(COALESCE(progress,0),?) WHERE id=?",
             (status, message, round(progress, 1), job_id))
    if force:
        print(f"[job {job_id}] {status} {message}")


def run_job(job_id, url, num_clips, schedule_start=None, interval_hours=None, layout=None):
    try:
        # 1) download: 0-30%
        _status(job_id, "downloading", "Connecting to YouTube…", 1)
        video, title = download(url, job_id, progress=lambda p, info: _status(
            job_id, "downloading", f"Downloading video… {p:.0f}%  ({info})", 1 + p * 0.29, force=False))
        db.x("UPDATE jobs SET title=? WHERE id=?", (title, job_id))

        # 2) captions / transcription: 30-40% (or 30-60% when we have to transcribe)
        _status(job_id, "transcribing", "Preparing audio…", 30)
        wav = video.with_name("audio.wav")
        if not wav.exists() or wav.stat().st_mtime < video.stat().st_mtime:
            wav = extract_audio(video)
        loud = loudness_per_second(wav)
        _status(job_id, "transcribing", "Reading YouTube captions…", 33)
        segments, lang = fetch_captions(url, video.parent)
        render_from = 45
        if segments:
            _status(job_id, "transcribing", f"Got captions ({lang}) - {len(segments)} lines", 40)
        else:
            _status(job_id, "transcribing", "No YouTube captions - listening to the audio (slower)…", 33)
            segments = transcribe(wav, progress=lambda pct: _status(
                job_id, "transcribing", f"Listening to the audio… {pct:.0f}%", 33 + pct * 0.27, force=False))
            render_from = 63
        (video.parent / "transcript.json").write_text(json.dumps(segments), encoding="utf-8")

        # 3) pick moments
        _status(job_id, "finding", "Finding the most viral moments…", render_from - 3)
        heatmap = load_heatmap(video.parent)
        clips, method = find_viral_moments(segments, loud, title, num_clips, heatmap=heatmap,
                                           status=lambda m: _status(job_id, "finding", m, render_from - 3))
        if not clips:
            raise RuntimeError("No speech found - couldn't pick clips")

        # 4) render: render_from-100%
        share = (100 - render_from) / len(clips)
        for i, c in enumerate(clips, 1):
            base = render_from + share * (i - 1)
            label = f"Editing reel {i} of {len(clips)}"
            _status(job_id, "editing", f"{label}… finding the speaker" if (layout or REEL_LAYOUT) == "track"
                    else f"{label}…", base)
            out = REELS_DIR / str(job_id) / f"reel_{i}.mp4"
            render_reel(video, out, c["start"], c["end"], segments, c["title"], layout,
                        speech_end=c.get("speech_end"),
                        progress=lambda p, b=base, l=label: _status(
                            job_id, "editing", f"{l}… {p:.0f}%", b + share * p / 100 * 0.98, force=False))
            sched, status = None, "draft"
            if schedule_start:
                sched = (schedule_start + timedelta(hours=(interval_hours or 24) * (i - 1))).isoformat(timespec="minutes")
                status = "scheduled"
            db.x("""INSERT INTO clips(job_id,path,start,end,title,caption,hashtags,score,reason,status,scheduled_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                 (job_id, str(out), c["start"], c["end"], c["title"], c["caption"], c["hashtags"],
                  c["score"], c["reason"], status, sched))

        from . import ai
        picked = {"claude": "Claude AI pick", "gemini": "Gemini AI pick"}.get(
            method, "basic pick - the AI did not answer this time" if ai.provider() else "basic pick")
        _status(job_id, "done", f"{len(clips)} reels ready ({picked}{' + most-replayed data' if heatmap else ''})", 100)
    except Exception as e:
        traceback.print_exc()
        _status(job_id, "error", str(e)[:500])


def start_job(url, num_clips=3, schedule_start: datetime | None = None, interval_hours=24, layout=None):
    job_id = db.x("INSERT INTO jobs(url,num_clips,progress) VALUES(?,?,0)", (url, num_clips))
    threading.Thread(target=run_job, args=(job_id, url, num_clips, schedule_start, interval_hours, layout),
                     daemon=True).start()
    return job_id
