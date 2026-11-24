import asyncio
import os
import threading
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from . import db, scheduler
from .config import REEL_LAYOUT, TIMEZONE
from . import ai
from .instagram import is_configured, verify
from .pipeline import start_job

app = FastAPI(title="Insta Reel Automation")
TEMPLATE = Path(__file__).parent / "templates" / "index.html"


def _quiet_disconnects(loop, context):
    """Windows prints a scary traceback whenever the browser stops a video stream - ignore just that."""
    if isinstance(context.get("exception"), (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return
    loop.default_exception_handler(context)


@app.on_event("startup")
async def _startup():
    asyncio.get_running_loop().set_exception_handler(_quiet_disconnects)
    db.init()
    scheduler.start()


@app.get("/", response_class=HTMLResponse)
def index():
    return TEMPLATE.read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    ig = {"configured": is_configured()}
    if ig["configured"]:
        try:
            ig["username"] = verify().get("username")
        except Exception as e:
            ig["error"] = str(e)
    return {"instagram": ig, "ai": ai.provider(), "timezone": TIMEZONE,
            "layout": REEL_LAYOUT, "now": scheduler.now_local().isoformat(timespec="minutes")}


class AIKeyIn(BaseModel):
    provider: str      # gemini | claude
    key: str


@app.post("/api/ai-key")
def set_ai_key(k: AIKeyIn):
    name = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(k.provider)
    if not name or len(k.key.strip()) < 20:
        raise HTTPException(400, "Paste the full API key")
    old = os.environ.get(name, "")
    ai.save_key(name, k.key)
    ai.save_key("AI_PROVIDER", k.provider)
    try:
        prov, model = ai.test()
        return {"ok": True, "provider": prov, "model": model}
    except Exception as e:
        ai.save_key(name, old)
        raise HTTPException(400, f"That key didn't work: {str(e)[:200]}")


class JobIn(BaseModel):
    url: str
    num_clips: int = 3
    auto_schedule: bool = False
    start_at: str | None = None        # "2026-11-04T18:30" local time
    interval_hours: float = 24
    layout: str | None = None


@app.post("/api/jobs")
def create_job(j: JobIn):
    if "youtu" not in j.url:
        raise HTTPException(400, "Please paste a YouTube link")
    start = None
    if j.auto_schedule:
        start = datetime.fromisoformat(j.start_at) if j.start_at else scheduler.now_local()
    job_id = start_job(j.url.strip(), max(1, min(j.num_clips, 10)), start, j.interval_hours, j.layout)
    return {"id": job_id}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: int):
    db.x("DELETE FROM jobs WHERE id=?", (jid,))
    return {"ok": True}


@app.get("/api/jobs")
def jobs():
    return db.q("SELECT * FROM jobs ORDER BY id DESC LIMIT 30")


@app.get("/api/clips")
def clips():
    return db.q("SELECT * FROM clips ORDER BY COALESCE(scheduled_at,'9999'), id DESC LIMIT 200")


class ClipPatch(BaseModel):
    caption: str | None = None
    hashtags: str | None = None
    scheduled_at: str | None = None
    status: str | None = None          # 'scheduled' or 'draft'


@app.patch("/api/clips/{cid}")
def update_clip(cid: int, p: ClipPatch):
    for field in ("caption", "hashtags", "scheduled_at", "status"):
        val = getattr(p, field)
        if val is not None:
            if field == "status" and val not in ("draft", "scheduled"):
                raise HTTPException(400, "status must be draft or scheduled")
            db.x(f"UPDATE clips SET {field}=? WHERE id=?", (val, cid))
    if p.status == "scheduled":
        row = db.q("SELECT scheduled_at FROM clips WHERE id=?", (cid,))
        if row and not row[0]["scheduled_at"]:
            raise HTTPException(400, "Pick a date/time first")
    return db.q("SELECT * FROM clips WHERE id=?", (cid,))[0]


@app.post("/api/clips/{cid}/post-now")
def post_now(cid: int):
    if not is_configured():
        raise HTTPException(400, "Add IG_ACCESS_TOKEN to .env and restart")
    threading.Thread(target=scheduler.post_clip, args=(cid,), daemon=True).start()
    return {"ok": True}


@app.delete("/api/clips/{cid}")
def delete_clip(cid: int):
    row = db.q("SELECT path FROM clips WHERE id=?", (cid,))
    if row:
        Path(row[0]["path"]).unlink(missing_ok=True)
        db.x("DELETE FROM clips WHERE id=?", (cid,))
    return {"ok": True}


@app.get("/media/{cid}")
def media(cid: int):
    row = db.q("SELECT path FROM clips WHERE id=?", (cid,))
    if not row or not Path(row[0]["path"]).exists():
        raise HTTPException(404)
    return FileResponse(row[0]["path"], media_type="video/mp4")
