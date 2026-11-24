"""Free AI voice for the explainer reel.

Main engine: Microsoft Edge 'read aloud' voices via the edge-tts package - free, no key, natural
Indian-English and Hindi voices, and it reports when each word is spoken (used for captions).
Backup when offline: the voice built into Windows (pyttsx3) - more robotic, but it always works.
"""
import asyncio
import subprocess
from pathlib import Path
from .config import FFMPEG

VOICES = {
    "en-IN-NeerjaNeural": ("English", "Neerja (female, Indian English)"),
    "en-IN-PrabhatNeural": ("English", "Prabhat (male, Indian English)"),
    "en-US-AndrewNeural": ("English", "Andrew (male, US English)"),
    "en-US-AvaNeural": ("English", "Ava (female, US English)"),
    "hi-IN-SwaraNeural": ("Hindi", "Swara (female, Hindi)"),
    "hi-IN-MadhurNeural": ("Hindi", "Madhur (male, Hindi)"),
}
DEFAULT_VOICE = "en-IN-NeerjaNeural"

# Only for running inside a network that inspects HTTPS (set by the test harness, unused on a normal PC)
SSL_CONTEXT = None
PROXY = None


def language_of(voice):
    return VOICES.get(voice, ("English", ""))[0]


def _to_wav(src: Path, dst: Path):
    subprocess.run([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "44100", str(dst)],
                   check=True)


def _duration(path: Path):
    import wave
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


async def _edge(text, voice, mp3: Path):
    import edge_tts
    if SSL_CONTEXT is not None:
        import edge_tts.communicate as cm
        cm._SSL_CTX = SSL_CONTEXT
    c = edge_tts.Communicate(text, voice, rate="+6%", boundary="WordBoundary", proxy=PROXY)
    words = []
    with open(mp3, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append({"start": ch["offset"] / 1e7, "end": (ch["offset"] + ch["duration"]) / 1e7,
                              "word": ch["text"]})
    return words


def _windows_voice(text, wav: Path):
    import pyttsx3
    eng = pyttsx3.init()
    eng.setProperty("rate", 175)
    eng.save_to_file(text, str(wav))
    eng.runAndWait()


def _spread(text, dur):
    toks = text.split()
    step = dur / max(1, len(toks))
    return [{"start": i * step, "end": (i + 1) * step, "word": t} for i, t in enumerate(toks)]


def speak(text: str, voice: str, out_wav: Path):
    """Text -> wav file. Returns (word timings, seconds, engine name)."""
    mp3 = out_wav.with_suffix(".mp3")
    try:
        words = asyncio.run(_edge(text, voice, mp3))
        if mp3.stat().st_size < 1000:
            raise RuntimeError("no audio received")
        _to_wav(mp3, out_wav)
        mp3.unlink(missing_ok=True)
        dur = _duration(out_wav)
        return (words or _spread(text, dur)), dur, "edge"
    except Exception as e:
        print(f"[voice] online voice failed ({str(e)[:120]}) - trying the Windows voice")
        mp3.unlink(missing_ok=True)
    raw = out_wav.with_name(out_wav.stem + "_raw.wav")
    _windows_voice(text, raw)
    _to_wav(raw, out_wav)
    raw.unlink(missing_ok=True)
    dur = _duration(out_wav)
    return _spread(text, dur), dur, "windows"
