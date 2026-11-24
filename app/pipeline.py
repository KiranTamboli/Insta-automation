"""YouTube link -> viral clips -> rendered 9:16 reels -> (optional) auto schedule."""
import json
import threading
import time
import traceback
from datetime import datetime, timedelta
from . import db
from .config import REELS_DIR, REEL_LAYOUT, DEFAULT_HASHTAGS as DEFAULT_TAGS
from .downloader import download, fetch_captions, load_heatmap
from .transcribe import extract_audio, transcribe, loudness_per_second, energy_fine
from .viral import find_viral_moments, understand_video
from .editor import render_reel, render_stitched

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


LENGTHS = {"short": (15, 40), "standard": (25, 60), "story": (40, 90)}   # seconds (min, max)


class _Progress:
    """Overall progress when the explainer reel is made at the same time as the other reels."""
    def __init__(self, job_id):
        self.job_id, self.lock = job_id, threading.Lock()
        self.main, self.side, self.side_weight, self.render_from = 0.0, 0.0, 0.0, 45.0

    def overall(self):
        w = self.side_weight
        return self.render_from + (100 - self.render_from) * ((1 - w) * self.main + w * self.side)

    def set_main(self, frac, status, message):
        with self.lock:
            self.main = max(self.main, min(1.0, frac))
            p = self.overall()
        _status(self.job_id, status, message, p, force=False)

    def set_side(self, frac=None, message=None):
        with self.lock:
            if frac is not None:
                self.side = max(self.side, min(1.0, frac))
            p = self.overall()
        if message is not None:
            db.x("UPDATE jobs SET side=? WHERE id=?", (message, self.job_id))
        db.x("UPDATE jobs SET progress=MAX(COALESCE(progress,0),?) WHERE id=?", (round(min(p, 99.5), 1), self.job_id))


def _save_clip(job_id, i, path, start, end, info, schedule_start, interval_hours):
    sched, status = None, "draft"
    if schedule_start:
        sched = (schedule_start + timedelta(hours=(interval_hours or 24) * (i - 1))).isoformat(timespec="minutes")
        status = "scheduled"
    db.x("""INSERT INTO clips(job_id,path,start,end,title,caption,hashtags,score,reason,status,scheduled_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
         (job_id, str(path), start, end, info["title"], info["caption"], info["hashtags"] or DEFAULT_TAGS,
          info.get("score", 100.0), info["reason"], status, sched))


def run_job(job_id, url, num_clips, schedule_start=None, interval_hours=None, layout=None, length=None,
            summary=True, explainer=True, voice=None):
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
        fine = energy_fine(wav)
        say = lambda m: _status(job_id, "finding", m, render_from - 3)
        profile, summary_plan = understand_video(segments, title, fine, say) if summary else ("", None)
        from . import ai
        do_explainer = bool(explainer and ai.provider())
        extra = (1 if summary_plan else 0) + (1 if do_explainer else 0)
        if extra:
            db.x("UPDATE jobs SET num_clips=? WHERE id=?", (num_clips + extra, job_id))

        # the explainer reel starts NOW and runs alongside everything below, so all reels finish together
        prog = _Progress(job_id)
        prog.render_from = render_from
        side_result = {}
        side_thread = None
        if do_explainer:
            from .explainer import make_explainer, CHARACTER_NAME
            n_main = num_clips + (1 if summary_plan else 0)
            prog.side_weight = 1.4 / (n_main + 1.4)          # the explainer is a bit more work than one clip
            ex_out = REELS_DIR / str(job_id) / "reel_explainer.mp4"

            def side_job():
                try:
                    prog.set_side(0.0, f"{CHARACTER_NAME} is getting ready…")
                    side_result["info"] = make_explainer(
                        video, ex_out, segments, title, profile, voice, layout,
                        status=lambda m: prog.set_side(None, m),
                        progress=lambda p: prog.set_side(p / 100, None))
                    side_result["info"]["path"] = ex_out
                    prog.set_side(1.0, "")
                except Exception as e:
                    traceback.print_exc()
                    print(f"[job {job_id}] explainer reel skipped: {e}")
                    side_result["error"] = str(e)
                    prog.set_side(1.0, "")
            side_thread = threading.Thread(target=side_job, daemon=True)
            side_thread.start()
        clips, method = find_viral_moments(segments, loud, title, num_clips, heatmap=heatmap,
                                           fine=fine, limits=LENGTHS.get(length or "story"), profile=profile,
                                           status=lambda m: _status(job_id, "finding", m, render_from - 3))
        if not clips:
            raise RuntimeError("No speech found - couldn't pick clips")

        # 4) render: render_from-100%
        n_total = len(clips) + (1 if summary_plan else 0) + (1 if do_explainer else 0)
        n_main = len(clips) + (1 if summary_plan else 0)
        for i, c in enumerate(clips, 1):
            label = f"Editing reel {i} of {n_main}"
            prog.set_main((i - 1) / n_main, "editing", f"{label}… finding the speaker"
                          if (layout or REEL_LAYOUT) == "track" else f"{label}…")
            _status(job_id, "editing", f"{label}…")
            out = REELS_DIR / str(job_id) / f"reel_{i}.mp4"
            render_reel(video, out, c["start"], c["end"], segments, c["title"], layout,
                        speech_end=c.get("speech_end"),
                        progress=lambda p, k=i, l=label: prog.set_main((k - 1 + p / 100 * 0.98) / n_main,
                                                                       "editing", f"{l}… {p:.0f}%"))
            sched, status = None, "draft"
            if schedule_start:
                sched = (schedule_start + timedelta(hours=(interval_hours or 24) * (i - 1))).isoformat(timespec="minutes")
                status = "scheduled"
            db.x("""INSERT INTO clips(job_id,path,start,end,title,caption,hashtags,score,reason,status,scheduled_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                 (job_id, str(out), c["start"], c["end"], c["title"], c["caption"], c["hashtags"],
                  c["score"], c["reason"], status, sched))

        if summary_plan:                                   # the extra reel: the whole video in under a minute
            try:
                i = n_main
                label = "Editing the summary reel"
                _status(job_id, "editing", f"{label}…")
                out = REELS_DIR / str(job_id) / f"reel_{i}_summary.mp4"
                parts = summary_plan["parts"]
                render_stitched(video, out, parts, segments, summary_plan["title"], layout,
                                progress=lambda p, l=label: prog.set_main((n_main - 1 + p / 100 * 0.98) / n_main,
                                                                          "editing", f"{l}… {p:.0f}%"))
                length_s = sum(p["end"] - p["start"] for p in parts) + 0.5
                sched, status = None, "draft"
                if schedule_start:
                    sched = (schedule_start + timedelta(hours=(interval_hours or 24) * (i - 1))).isoformat(timespec="minutes")
                    status = "scheduled"
                db.x("""INSERT INTO clips(job_id,path,start,end,title,caption,hashtags,score,reason,status,scheduled_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                     (job_id, str(out), parts[0]["start"], parts[0]["start"] + length_s, summary_plan["title"],
                      summary_plan["caption"], summary_plan["hashtags"], 100.0, summary_plan["reason"], status, sched))
            except Exception as e:
                traceback.print_exc()
                print(f"[job {job_id}] summary reel skipped: {e}")
                n_total -= 1
                db.x("UPDATE jobs SET num_clips=? WHERE id=?", (n_total, job_id))
        prog.set_main(1.0, "editing", "Other reels ready…")

        if side_thread is not None:                         # wait for the explainer running alongside
            if side_thread.is_alive():
                _status(job_id, "editing", f"Finishing {CHARACTER_NAME}'s explainer reel…")
            side_thread.join()
            if "info" in side_result:
                info = side_result["info"]
                _save_clip(job_id, n_total, info["path"], 0.0, info["length"], info, schedule_start, interval_hours)
            else:
                n_total -= 1
                db.x("UPDATE jobs SET num_clips=? WHERE id=?", (n_total, job_id))
            db.x("UPDATE jobs SET side='' WHERE id=?", (job_id,))

        picked = {"claude": "Claude AI pick", "gemini": "Gemini AI pick"}.get(
            method, "basic pick - the AI did not answer this time" if ai.provider() else "basic pick")
        _status(job_id, "done", f"{n_total} reels ready ({picked}{' + most-replayed data' if heatmap else ''})", 100)
    except Exception as e:
        traceback.print_exc()
        _status(job_id, "error", str(e)[:500])


def start_job(url, num_clips=3, schedule_start: datetime | None = None, interval_hours=24, layout=None, length=None,
              summary=True, explainer=True, voice=None):
    job_id = db.x("INSERT INTO jobs(url,num_clips,progress) VALUES(?,?,0)", (url, num_clips))
    threading.Thread(target=run_job, args=(job_id, url, num_clips, schedule_start, interval_hours, layout, length,
                                           summary, explainer, voice), daemon=True).start()
    return job_id
