"""Publish a Reel with the official Instagram Graph API (Content Publishing).

Uses the *resumable upload* flow, so the video is uploaded straight from your PC -
no public URL / hosting needed.
  1. create container   POST /{ig_user_id}/media  (media_type=REELS, upload_type=resumable)
  2. upload bytes       POST https://rupload.facebook.com/ig-api-upload/{ver}/{container_id}
  3. wait for FINISHED  GET  /{container_id}?fields=status_code
  4. publish            POST /{ig_user_id}/media_publish
"""
import time
from pathlib import Path
import requests
from .config import IG_USER_ID, IG_ACCESS_TOKEN, IG_GRAPH_HOST, IG_API_VERSION


class InstagramError(RuntimeError):
    pass


def _base():
    return f"https://{IG_GRAPH_HOST}/{IG_API_VERSION}"


def _check(r):
    try:
        data = r.json()
    except ValueError:
        raise InstagramError(f"HTTP {r.status_code}: {r.text[:300]}")
    if r.status_code >= 400 or "error" in data:
        err = data.get("error", data)
        raise InstagramError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    return data


_uid_cache = None


def _uid():
    """Numeric IG user ID: from .env, or looked up automatically from the token."""
    global _uid_cache
    if IG_USER_ID and IG_USER_ID.isdigit():
        return IG_USER_ID
    if _uid_cache:
        return _uid_cache
    fields = "user_id,username" if "instagram" in IG_GRAPH_HOST else "id,username"
    r = requests.get(f"{_base()}/me", params={"fields": fields, "access_token": IG_ACCESS_TOKEN}, timeout=30)
    d = _check(r)
    _uid_cache = str(d.get("user_id") or d.get("id"))
    print(f"[instagram] using IG user ID {_uid_cache} (@{d.get('username')})")
    return _uid_cache


def is_configured():
    return bool(IG_ACCESS_TOKEN)


def verify():
    r = requests.get(f"{_base()}/{_uid()}", params={"fields": "username", "access_token": IG_ACCESS_TOKEN}, timeout=30)
    d = _check(r)
    d["ig_user_id"] = _uid()
    return d


def publish_reel(video: Path, caption: str, share_to_feed=True, timeout_s=600) -> str:
    if not is_configured():
        raise InstagramError("Set IG_ACCESS_TOKEN in .env first")

    # 1) container
    r = requests.post(f"{_base()}/{_uid()}/media", data={
        "media_type": "REELS",
        "upload_type": "resumable",
        "caption": caption[:2200],
        "share_to_feed": "true" if share_to_feed else "false",
        "access_token": IG_ACCESS_TOKEN,
    }, timeout=60)
    container_id = _check(r)["id"]

    # 2) upload bytes
    size = video.stat().st_size
    with open(video, "rb") as f:
        r = requests.post(
            f"https://rupload.facebook.com/ig-api-upload/{IG_API_VERSION}/{container_id}",
            headers={"Authorization": f"OAuth {IG_ACCESS_TOKEN}", "offset": "0", "file_size": str(size)},
            data=f, timeout=600,
        )
    _check(r)

    # 3) wait for processing
    deadline = time.time() + timeout_s
    while True:
        r = requests.get(f"{_base()}/{container_id}",
                         params={"fields": "status_code,status", "access_token": IG_ACCESS_TOKEN}, timeout=30)
        st = _check(r)
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise InstagramError(f"Instagram processing failed: {st.get('status')}")
        if time.time() > deadline:
            raise InstagramError("Timed out waiting for Instagram to process the video")
        time.sleep(8)

    # 4) publish
    r = requests.post(f"{_base()}/{_uid()}/media_publish",
                      data={"creation_id": container_id, "access_token": IG_ACCESS_TOKEN}, timeout=60)
    return _check(r)["id"]
