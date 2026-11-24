import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATA_DIR = ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
REELS_DIR = DATA_DIR / "reels"
DB_PATH = DATA_DIR / "insta.db"
for d in (DATA_DIR, DOWNLOAD_DIR, REELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _ensure_ffmpeg():
    """Use system FFmpeg if present, otherwise the copy bundled with the imageio-ffmpeg package."""
    import shutil, subprocess

    def works(path):
        try:
            return subprocess.run([path, "-version"], capture_output=True, timeout=20).returncode == 0
        except Exception:
            return False

    found = shutil.which("ffmpeg")
    if found and works(found):
        return found
    bin_dir = ROOT / "bin"
    exe = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not exe.exists():
        import imageio_ffmpeg
        bin_dir.mkdir(exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), exe)
        exe.chmod(0o755)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(exe)


FFMPEG = _ensure_ffmpeg()


def env(name, default=""):
    v = os.getenv(name)
    return v if v not in (None, "") else default


IG_USER_ID = env("IG_USER_ID")
IG_ACCESS_TOKEN = env("IG_ACCESS_TOKEN")
IG_GRAPH_HOST = env("IG_GRAPH_HOST", "graph.instagram.com")
IG_API_VERSION = env("IG_API_VERSION", "v21.0")

ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY")
CLAUDE_MODEL = env("CLAUDE_MODEL", "claude-sonnet-5-5")

WHISPER_MODEL = env("WHISPER_MODEL", "base")
REEL_LAYOUT = env("REEL_LAYOUT", "track")
MIN_CLIP = float(env("MIN_CLIP_SECONDS", "20"))
MAX_CLIP = float(env("MAX_CLIP_SECONDS", "55"))
DEFAULT_HASHTAGS = env("DEFAULT_HASHTAGS", "#reels #viral #trending #explore")

TIMEZONE = env("TIMEZONE", "Asia/Kolkata")
PORT = int(env("PORT", "8000"))
