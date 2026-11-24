"""Audio extraction + speech-to-text with word-level timestamps (faster-whisper)."""
import subprocess
import wave
from pathlib import Path
import numpy as np
from .config import WHISPER_MODEL

_model = None


def extract_audio(video: Path) -> Path:
    wav = video.with_name("audio.wav")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )
    return wav


def loudness_per_second(wav: Path) -> np.ndarray:
    """RMS loudness for each second of audio (used to find energetic moments)."""
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    secs = len(data) // sr
    if secs == 0:
        return np.zeros(1)
    data = data[: secs * sr].reshape(secs, sr)
    return np.sqrt((data ** 2).mean(axis=1))


def energy_fine(wav: Path, hop=0.02):
    """Loudness in dB every 20 ms - used to place cuts inside real pauses."""
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    n = int(sr * hop)
    frames = len(data) // n
    if frames == 0:
        return np.zeros(1), hop
    rms = np.sqrt((data[: frames * n].reshape(frames, n) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1.0), hop


def transcribe(wav: Path, progress=None):
    """Returns list of segments: {start, end, text, words:[{start,end,word}]}"""
    global _model
    from faster_whisper import WhisperModel
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")
    # pass raw samples (not the file path) so faster-whisper doesn't need PyAV to decode
    with wave.open(str(wav)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    total = max(1.0, len(audio) / 16000)
    segments, _info = _model.transcribe(audio, word_timestamps=True, vad_filter=True)
    out, last = [], -5
    for s in segments:
        pct = min(99.0, s.end / total * 100)
        if progress and pct - last >= 2:
            progress(pct)
            last = pct
        out.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [{"start": w.start, "end": w.end, "word": w.word.strip()} for w in (s.words or [])],
        })
    return out
