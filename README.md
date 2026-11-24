# Reel Studio: YouTube → Viral Reels → Instagram (scheduled)

Paste a YouTube link and the app will:
1. **Download** the video (yt-dlp, up to 1080p)
2. **Transcribe** it with word-level timing (faster-whisper, runs locally)
3. **Find the viral moments**, either with **Claude AI** (if you add an Anthropic key) or a built-in scorer (loudness, hook words, speech pace)
4. **Edit each moment into a Reel**: 1080×1920, blurred-background or center-crop layout, a hook title for the first 3 seconds, and bold word-by-word captions
5. **Schedule and auto-post** to Instagram using the official Graph API. Videos upload straight from your PC, so you don't need any hosting.

## Required "plugins" (dependencies)
| Package | What it does |
|---|---|
| `yt-dlp` | Downloads YouTube videos |
| `faster-whisper` | Speech-to-text with word timestamps (for captions and clip picking) |
| `anthropic` | (optional) Claude picks the best clips and writes captions and hashtags |
| `fastapi`, `uvicorn`, `jinja2`, `python-multipart` | Local web dashboard |
| `apscheduler` | Posts reels at the scheduled time |
| `requests` | Instagram Graph API calls |
| `numpy`, `python-dotenv` | Audio energy analysis, config |
| **FFmpeg** (system app) | Cutting, reframing, caption burn-in |

## Setup (Windows)
1. Install **Python 3.10+** from python.org. Tick "Add Python to PATH".
2. Double-click **`setup.bat`**. It installs FFmpeg (via winget) and all Python packages, and creates `.env`.
3. Open `.env` and fill in:
   - `IG_USER_ID` and `IG_ACCESS_TOKEN` (see below)
   - `ANTHROPIC_API_KEY` (optional, for much better clip picks and captions)
4. Double-click **`start.bat`**. The dashboard opens at http://127.0.0.1:8000

Mac/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cp .env.example .env && python run.py`

## Getting Instagram API access
1. Switch your Instagram account to a **Professional (Business or Creator)** account.
2. Go to https://developers.facebook.com/apps, create an app, and add the **Instagram** product.
3. Choose **"Instagram API with Instagram Login"** (the simplest option, no Facebook Page needed). Set `IG_GRAPH_HOST=graph.instagram.com`.
   - Or use Facebook Login with a linked Facebook Page and keep `graph.facebook.com`.
4. Add your Instagram account as a tester, then generate a token with the `instagram_business_content_publish` (or `instagram_content_publish`) permission.
5. Convert it to a **long-lived token** (valid 60 days, refreshable). Put the token and your IG user ID in `.env`.
6. Restart `start.bat`. The top bar should show `Instagram: @yourname`.

Instagram limits accounts to about 50 API-published posts per 24 hours. Reels must be 3 to 90 seconds; the app makes 20 to 55 second clips by default (set `MIN_CLIP_SECONDS` and `MAX_CLIP_SECONDS` in `.env`).

## Using it
- **Make reels:** paste the link, choose how many reels and a layout, then click *Make reels*.
- **Auto-schedule option:** tick *Auto-schedule*, pick the first post time and the gap (for example every 1 day). Every reel from that video is queued automatically.
- **Manual control:** each reel card lets you preview, edit the caption and hashtags, set a time, and *Schedule*, *Unschedule*, *Post now* or *Delete*.
- The scheduler checks every minute. **Keep `start.bat` running** (and your PC awake) for scheduled posts to go out.

## Files
```
app/downloader.py   yt-dlp download
app/transcribe.py   audio extraction, loudness, Whisper
app/viral.py        Claude / heuristic viral-moment finder
app/editor.py       FFmpeg 9:16 render + karaoke captions
app/instagram.py    Graph API resumable Reel upload + publish
app/scheduler.py    background poster
app/pipeline.py     ties it all together
app/main.py         web API + dashboard
data/               downloads, rendered reels, sqlite db
```

## Important
Only repost videos **you own or have permission to use**. Reposting other creators' YouTube content can lead to copyright strikes, takedowns or a banned Instagram account, and downloading may break YouTube's Terms of Service.
