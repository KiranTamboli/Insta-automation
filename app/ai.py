"""One small interface over the AI models the app can use.

  - Google Gemini  (free API key from https://aistudio.google.com/apikey)  -> GEMINI_API_KEY
  - Anthropic Claude (paid, best quality)                                  -> ANTHROPIC_API_KEY

Keys can be pasted in the dashboard (saved to .env) or written in .env by hand.
"""
import json
import os
import re
import time
import requests
from .config import ROOT

ENV_FILE = ROOT / ".env"
_gemini_model_cache = None


def _key(name):
    return os.getenv(name, "").strip()


def provider():
    pref = _key("AI_PROVIDER").lower()
    if pref in ("claude", "gemini") and _key("ANTHROPIC_API_KEY" if pref == "claude" else "GEMINI_API_KEY"):
        return pref
    if _key("ANTHROPIC_API_KEY"):
        return "claude"
    if _key("GEMINI_API_KEY"):
        return "gemini"
    return None


def save_key(name, value):
    """Write/replace KEY=value in .env and apply it immediately."""
    value = value.strip()
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [l for l in lines if not l.strip().startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[name] = value
    global _gemini_model_cache
    _gemini_model_cache = None


# ------------------------------------------------------------------ Gemini
API = "https://generativelanguage.googleapis.com/v1beta"
_think_cfg = {}          # model -> thinking config that the model accepted


def _gemini_models(key):
    """Models to try, best first: GEMINI_MODEL if set, else the newest stable 'flash', then a fast 'lite' backup."""
    global _gemini_model_cache
    if _key("GEMINI_MODEL"):
        return [_key("GEMINI_MODEL")]
    if _gemini_model_cache:
        return _gemini_model_cache
    r = requests.get(f"{API}/models", headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
    if r.status_code >= 400:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"Google says: {msg}")
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    text_ok = [n for n in names if "flash" in n and not any(x in n for x in ("image", "tts", "live", "audio", "native"))]
    if not text_ok:
        if not names:
            raise RuntimeError("Your Gemini key can't use any text models")
        text_ok = names

    def ver(n):                                   # prefer stable over preview/experimental, then newest
        nums = re.findall(r"\d+(?:\.\d+)?", n)
        return ("preview" not in n and "exp" not in n, float(nums[0]) if nums else 0, "latest" in n)
    full = sorted([n for n in text_ok if "lite" not in n], key=ver)
    lite = sorted([n for n in text_ok if "lite" in n], key=ver)
    order = ([full[-1]] if full else []) + ([lite[-1]] if lite else [])
    _gemini_model_cache = order or [sorted(text_ok, key=ver)[-1]]
    print(f"[ai] Gemini models: {_gemini_model_cache}")
    return _gemini_model_cache


def _gemini_call(key, model, prompt, read_timeout):
    """One streamed request. Streaming keeps the connection alive while the model writes a long answer."""
    base = {"temperature": 0.3, "responseMimeType": "application/json"}
    # newer Gemini models 'think' before answering; keep that short so a long transcript doesn't time out
    options = [_think_cfg[model]] if model in _think_cfg else [
        {"thinkingConfig": {"thinkingLevel": "low"}}, {"thinkingConfig": {"thinkingBudget": 1024}}, {}]
    last = None
    for extra in options:
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {**base, **extra}}
        r = requests.post(f"{API}/models/{model}:streamGenerateContent", params={"alt": "sse"},
                          headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                          json=body, timeout=(20, read_timeout), stream=True)
        if r.status_code == 400 and extra:        # this model doesn't know that thinking option - try the next
            last = r.text[:300]
            r.close()
            continue
        if r.status_code >= 400:
            err = RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
            err.status = r.status_code
            raise err
        _think_cfg[model] = extra
        out = []
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                chunk = json.loads(line[5:].strip())
                for part in chunk["candidates"][0]["content"]["parts"]:
                    if not part.get("thought"):
                        out.append(part.get("text", ""))
            except (ValueError, KeyError, IndexError):
                continue
        text = "".join(out)
        if not text.strip():
            raise RuntimeError("Gemini returned an empty answer")
        return text
    raise RuntimeError(f"Gemini rejected the request: {last}")


def _ask_gemini(prompt):
    key = _key("GEMINI_API_KEY")
    errors = []
    for model in _gemini_models(key):
        for attempt in range(3):
            try:
                t0 = time.time()
                text = _gemini_call(key, model, prompt, read_timeout=300)
                print(f"[ai] {model} answered in {time.time() - t0:.0f}s")
                return text
            except requests.exceptions.Timeout:
                errors.append(f"{model}: no answer in time")
                break                                       # slow model: go to the faster backup
            except requests.exceptions.RequestException as e:
                errors.append(f"{model}: connection problem ({type(e).__name__})")
                time.sleep(5)
            except RuntimeError as e:
                errors.append(f"{model}: {str(e)[:160]}")
                if getattr(e, "status", 0) in (429, 500, 503) and attempt < 2:
                    time.sleep(15 * (attempt + 1))          # busy / rate limited: wait and retry
                    continue
                break
    raise RuntimeError("; ".join(errors[-3:]) or "Gemini unavailable")


# ------------------------------------------------------------------ Claude
def _ask_claude(prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
    msg = client.messages.create(model=_key("CLAUDE_MODEL") or "claude-sonnet-5-5", max_tokens=6000,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def ask_json(prompt):
    """Send a prompt, return parsed JSON (list or dict)."""
    p = provider()
    if not p:
        raise RuntimeError("No AI key configured")
    text = _ask_claude(prompt) if p == "claude" else _ask_gemini(prompt)
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    return json.loads(m.group(1) if m else text)


def test():
    """Quick check that the configured key works. Returns (provider, model)."""
    p = provider()
    if p == "gemini":
        key = _key("GEMINI_API_KEY")
        models = _gemini_models(key)
        _gemini_call(key, models[0], 'Reply with the JSON {"ok": true}', read_timeout=90)   # real round trip
        return p, models[0]
    if p == "claude":
        _ask_claude("Reply with OK")
        return p, _key("CLAUDE_MODEL") or "claude-sonnet-5-5"
    return None, None
