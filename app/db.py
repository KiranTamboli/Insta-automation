import sqlite3
import threading
from contextlib import contextmanager
from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    title TEXT,
    status TEXT DEFAULT 'queued',      -- queued, downloading, transcribing, finding, editing, done, error
    message TEXT,
    num_clips INTEGER DEFAULT 3,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    path TEXT,
    start REAL,
    end REAL,
    title TEXT,
    caption TEXT,
    hashtags TEXT,
    score REAL,
    reason TEXT,
    status TEXT DEFAULT 'draft',       -- draft, scheduled, posting, posted, failed
    scheduled_at TEXT,                 -- ISO local time
    ig_media_id TEXT,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


@contextmanager
def conn():
    with _lock:
        c = sqlite3.connect(DB_PATH)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
        if "progress" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN progress REAL DEFAULT 0")
        if "step" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN step TEXT")
        # tidy up on every start: drop failed jobs, and jobs left half-finished when the app was closed
        c.execute("DELETE FROM jobs WHERE status IN ('error','queued','downloading','transcribing','finding','editing')"
                  " AND id NOT IN (SELECT DISTINCT job_id FROM clips WHERE job_id IS NOT NULL)")
        c.execute("UPDATE jobs SET status='done', message='Stopped early - some reels were made'"
                  " WHERE status IN ('error','queued','downloading','transcribing','finding','editing')")


def q(sql, args=()):
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def x(sql, args=()):
    with conn() as c:
        cur = c.execute(sql, args)
        return cur.lastrowid
