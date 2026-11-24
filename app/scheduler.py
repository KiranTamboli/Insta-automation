"""Background scheduler: every minute, post any reel whose scheduled time has arrived."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from . import db
from .config import TIMEZONE
from .instagram import publish_reel, is_configured

TZ = ZoneInfo(TIMEZONE)
_sched = None


def now_local():
    return datetime.now(TZ).replace(tzinfo=None)


def post_clip(clip_id):
    rows = db.q("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not rows:
        return
    c = rows[0]
    db.x("UPDATE clips SET status='posting', error=NULL WHERE id=?", (clip_id,))
    try:
        caption = f"{c['caption'] or ''}\n\n{c['hashtags'] or ''}".strip()
        media_id = publish_reel(Path(c["path"]), caption)
        db.x("UPDATE clips SET status='posted', ig_media_id=? WHERE id=?", (media_id, clip_id))
        print(f"[scheduler] posted clip {clip_id} -> {media_id}")
    except Exception as e:
        db.x("UPDATE clips SET status='failed', error=? WHERE id=?", (str(e)[:500], clip_id))
        print(f"[scheduler] clip {clip_id} failed: {e}")


def tick():
    if not is_configured():
        return
    due = db.q("SELECT id FROM clips WHERE status='scheduled' AND scheduled_at <= ? ORDER BY scheduled_at",
               (now_local().isoformat(timespec="minutes"),))
    for row in due:
        post_clip(row["id"])


def start():
    global _sched
    if _sched:
        return
    # reels stuck in 'posting' after a crash go back to the queue
    db.x("UPDATE clips SET status='scheduled' WHERE status='posting'")
    _sched = BackgroundScheduler(timezone=TZ)
    _sched.add_job(tick, "interval", seconds=60, id="poster", max_instances=1, coalesce=True)
    _sched.start()
