"""Download a YouTube video (up to 1080p H.264) + its captions, with reuse of earlier downloads."""
import json
import re
from pathlib import Path
import yt_dlp
from .config import DOWNLOAD_DIR, FFMPEG

VIDEO_EXTS = (".mp4", ".mkv", ".webm")


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url.strip()


def _find_cached(vid):
    for d in DOWNLOAD_DIR.iterdir():
        f = d / "url.txt"
        if d.is_dir() and f.exists() and f.read_text(encoding="utf-8").strip() == vid:
            src = [p for p in d.glob("source.*") if p.suffix in VIDEO_EXTS and p.stat().st_size > 0]
            if src:
                info = d / "info.json"
                title = json.loads(info.read_text(encoding="utf-8")).get("title") if info.exists() else None
                return src[0], title or "video"
    return None


def download(url: str, job_id: int, progress=None) -> tuple[Path, str]:
    vid = video_id(url)
    cached = _find_cached(vid)
    if cached:
        print(f"[download] reusing earlier download {cached[0]}")
        return cached

    out_dir = DOWNLOAD_DIR / str(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        # H.264 first (decodes fast everywhere), up to 1080p so face-tracked crops stay sharp
        "format": ("bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/bv*[height<=1080][ext=mp4]+ba[ext=m4a]/"
                   "b[height<=1080][ext=mp4]/b[height<=1080]/b"),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "ffmpeg_location": FFMPEG,
        "quiet": True,
        "no_warnings": True,
    }
    parts = []                               # video + audio are downloaded as two files

    def hook(d):
        if not progress or d.get("status") != "downloading":
            return
        fn = d.get("filename")
        if fn not in parts:
            parts.append(fn)
        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
        if total:
            frac = d.get("downloaded_bytes", 0) / total
            overall = min(0.99, (len(parts) - 1 + frac) / max(2, len(parts)))
            speed = d.get("speed") or 0
            progress(overall * 100, f"{total / 1e6:.0f} MB · {speed / 1e6:.1f} MB/s" if speed else f"{total / 1e6:.0f} MB")
    opts["progress_hooks"] = [hook]
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = sorted([p for p in out_dir.glob("source.*") if p.suffix in VIDEO_EXTS],
                   key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("Download failed - no file produced")
    title = info.get("title") or "video"
    (out_dir / "url.txt").write_text(vid, encoding="utf-8")
    _save_heatmap(info, out_dir)
    (out_dir / "info.json").write_text(json.dumps({"title": title, "language": info.get("language")}), encoding="utf-8")
    return files[0], title


def _save_heatmap(info, out_dir: Path):
    """YouTube's 'Most replayed' graph (only exists for videos with enough views)."""
    hm = (info or {}).get("heatmap")
    if hm:
        (out_dir / "heatmap.json").write_text(json.dumps(hm), encoding="utf-8")


def load_heatmap(out_dir: Path):
    f = out_dir / "heatmap.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except ValueError:
        return None


# ------------------------------------------------------------------ captions
def choose_caption(info):
    """Pick captions in the language actually SPOKEN in the video.

    YouTube offers auto-translated captions in many languages; their timings are loose and the words
    don't match the audio, so cuts made from them land mid-speech. Order of preference:
    human-made captions in the original language > auto captions in the original language > English."""
    subs = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    auto = info.get("automatic_captions") or {}
    orig = next((k[:-5] for k in auto if k.endswith("-orig")), None) or (info.get("language") or "").split("-")[0]
    if orig:
        k = next((k for k in subs if k.split("-")[0] == orig), None)
        if k:
            return k, False
        for k in (f"{orig}-orig", orig):
            if k in auto:
                return k, True
    k = next((k for k in subs if k.startswith("en")), None) or next(iter(subs), None)
    if k:
        return k, False
    for k in ("en-orig", "en"):
        if k in auto:
            return k, True
    return None, False


def fetch_captions(url: str, out_dir: Path):
    """Grab YouTube's own captions (manual or auto) as word-timed segments. Returns [] if none."""
    for old in out_dir.glob("subs.*.json3"):
        old.unlink()
    base = {"skip_download": True, "noplaylist": True, "quiet": True, "no_warnings": True, "ignoreerrors": True}
    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False) or {}
        _save_heatmap(info, out_dir)
        lang, is_auto = choose_caption(info)
        if not lang:
            return [], None
        opts = {**base, "writesubtitles": not is_auto, "writeautomaticsub": is_auto, "subtitleslangs": [lang],
                "subtitlesformat": "json3", "outtmpl": str(out_dir / "subs.%(ext)s")}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as e:
        print(f"[captions] could not fetch: {e}")
        return [], None
    files = sorted(out_dir.glob("subs.*.json3"))
    if not files:
        return [], None
    segs = parse_json3(files[0])
    return segs, f"{lang}{' auto' if is_auto else ''}"


def parse_json3(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = []
    for ev in data.get("events", []):
        if "segs" not in ev:
            continue
        t0 = ev.get("tStartMs", 0) / 1000
        dur = ev.get("dDurationMs", 0) / 1000
        parts = [(s.get("utf8", ""), s.get("tOffsetMs")) for s in ev["segs"]]
        text = " ".join("".join(p for p, _ in parts).split())
        if not text:
            continue
        raw.append((t0, t0 + dur, parts, text))

    segs = []
    for i, (t0, t1, parts, text) in enumerate(raw):
        if i + 1 < len(raw):
            t1 = min(t1, raw[i + 1][0]) if raw[i + 1][0] > t0 else t1
        t1 = max(t1, t0 + 0.2)
        words = []
        if any(o is not None for _, o in parts[1:]):           # auto captions: per-word offsets
            for w, o in parts:
                w = w.strip()
                if w:
                    words.append({"start": t0 + (o or 0) / 1000, "end": None, "word": w})
        else:                                                   # manual captions: spread evenly
            toks = text.split()
            step = (t1 - t0) / max(1, len(toks))
            words = [{"start": t0 + k * step, "end": None, "word": t} for k, t in enumerate(toks)]
        for k, w in enumerate(words):
            w["end"] = words[k + 1]["start"] if k + 1 < len(words) else t1
        segs.append({"start": t0, "end": t1, "text": text, "words": words})
    return segs
