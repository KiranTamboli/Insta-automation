"""Find the most 'viral' moments in a transcript.

Clips always start at the beginning of a sentence and end at the end of a sentence
(caption lines from YouTube are first re-joined into real sentences).

Two strategies:
  1. AI (Gemini free key or Claude key): reads every numbered sentence, picks hook-first, complete
     moments by sentence id, then a second pass re-checks each cut's first/last sentence in context.
  2. Heuristic fallback: scores sentence windows by loudness, speech pace and hook words.
"""
import json
import re
import numpy as np
from .config import DEFAULT_HASHTAGS
from . import config

MIN_CLIP, MAX_CLIP = config.MIN_CLIP, config.MAX_CLIP     # replaced per job by set_limits()
HARD_MAX = 90.0                                           # Instagram Reels tab limit


def set_limits(min_len, max_len):
    global MIN_CLIP, MAX_CLIP
    MIN_CLIP, MAX_CLIP = float(min_len), float(min(max_len, HARD_MAX))

TAIL = 0.6          # seconds kept after the last word (a breath before the cut)
LEAD = 0.12         # seconds kept before the first word

HOOK_WORDS = {
    "secret", "never", "always", "mistake", "truth", "crazy", "insane", "shocking", "best", "worst",
    "how to", "why", "stop", "nobody", "everyone", "money", "million", "free", "hack", "trick",
    "you need", "don't", "biggest", "first time", "finally", "actually", "wait", "imagine", "problem",
    "story", "learned", "lesson", "change", "success", "fail", "rich", "important", "listen",
    "confidence", "fear", "love", "life", "remember", "the reason", "here's",
}
WEAK_STARTS = ("and ", "but ", "so ", "because ", "of ", "or ", "which ", "that ", "to ", "then ", "it ", "they ")
END_PUNCT = re.compile(r"[.!?…।]['\")\]]*$")


# ---------------------------------------------------------------- sentences
def build_sentences(segments, max_len=20.0):
    """-> (sentences, words, has_punct). Each sentence knows its first/last word index (w0, w1)."""
    words = [w for s in segments for w in s.get("words", []) if w.get("word")]
    if not words:   # no word timing at all - treat each segment as a sentence
        return [dict(start=s["start"], end=s["end"], text=s["text"]) for s in segments if s["text"].strip()], [], True

    has_punct = sum(1 for w in words if END_PUNCT.search(w["word"])) > len(words) / 80
    if not has_punct:
        return _line_units(segments, words), words, False
    sents, first = [], 0
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 99
        dur = w["end"] - words[first]["start"]
        boundary = (
            nxt is None
            or (has_punct and END_PUNCT.search(w["word"]))
            or (not has_punct and gap > 0.7)
            or gap > 1.5                                   # long silence always ends a sentence
            or (dur > max_len and (w["word"].endswith(",") or gap > 0.35))
        )
        if boundary:
            sents.append({"start": words[first]["start"], "end": w["end"], "w0": first, "w1": i,
                          "text": " ".join(x["word"] for x in words[first:i + 1])})
            first = i + 1
    return sents, words, has_punct


def _line_units(segments, words, min_dur=2.0, max_dur=9.0):
    """No punctuation (typical for auto captions, Hindi etc.): caption word timings have no gaps, so pauses
    can't be seen in the text. Use YouTube's own caption lines as units (short ones are merged); the AI
    then picks the exact first/last word and the audio check places the cut in a real pause."""
    units, idx = [], 0
    for seg in segments:
        n = len([w for w in seg.get("words", []) if w.get("word")])
        if not n:
            continue
        w0, w1 = idx, idx + n - 1
        idx += n
        if units and (units[-1]["end"] - units[-1]["start"] < min_dur) \
                and words[w1]["end"] - units[-1]["start"] <= max_dur:
            u = units[-1]                                  # previous line is tiny: join them
            u["w1"], u["end"] = w1, words[w1]["end"]
        else:
            units.append({"start": words[w0]["start"], "end": words[w1]["end"], "w0": w0, "w1": w1})
    for u in units:
        u["text"] = " ".join(x["word"] for x in words[u["w0"]:u["w1"] + 1])
    return units


def to_sentences(segments, max_len=20.0):
    return build_sentences(segments, max_len)[0]


def _fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


# ---------------------------------------------------------------- YouTube "Most replayed" data
def heat_curve(heatmap, duration):
    """Per-second replay intensity (0..1) from YouTube's 'Most replayed' graph, or None.

    The first seconds of every video score high simply because everyone starts there,
    so the intro is flattened to the video's typical level."""
    if not heatmap or duration <= 0:
        return None
    n = int(duration) + 2
    curve = np.zeros(n)
    for h in heatmap:
        try:
            a, b, v = int(h["start_time"]), int(h["end_time"]) + 1, float(h["value"])
        except (KeyError, TypeError, ValueError):
            continue
        curve[max(0, a):min(n, b)] = v
    if curve.max() <= 0:
        return None
    intro = int(max(20, duration * 0.03))
    if n > intro:
        curve[:intro] = np.median(curve[intro:])
    return curve


def mark_hot(sents, curve):
    """Attach heat (mean replay value) and a HOT level to every sentence."""
    if curve is None:
        return False
    hi, top = np.percentile(curve, 80), np.percentile(curve, 95)
    for s in sents:
        a, b = int(s["start"]), max(int(s["start"]) + 1, int(s["end"]) + 1)
        s["heat"] = float(curve[a:b].mean()) if a < len(curve) else 0.0
        s["hot"] = 2 if s["heat"] >= top else 1 if s["heat"] >= hi else 0
    return True


# ---------------------------------------------------------------- AI picker (Gemini / Claude)
PICK_PROMPT = """You are an expert short-form video editor. You turn long YouTube videos into Instagram Reels
that people watch to the end and share.

Video title: {title}
{profile}

Below is the full transcript split into numbered sentences:  [S<id> @<start seconds> | <length>s | optional HOT mark] text
{transcript}

{audience}

TASK: choose the {n} best, non-overlapping moments for Reels. Each moment is a run of consecutive
sentences from S<start_id> to S<end_id> (inclusive).

Think like the viewer: a stranger scrolling Instagram who has never seen this video and gives you
2 seconds. Only pick a moment if it gives that stranger something REAL on its own:
  - a surprising truth or counter-intuitive idea,   - a short story with a turn,
  - a practical tip they can use today,             - an emotional or vulnerable confession,
  - a bold opinion people will argue about,         - a memorable line worth quoting.
Skip filler: greetings, sponsor reads, "in this video", vague motivation with no specific point,
anything that only makes sense after watching the rest.

A great Reel:
1. HOOK - its first sentence grabs attention on its own (bold claim, surprising fact, emotional line,
   strong question). It must NOT depend on earlier context: never start with words like "and", "so", "but",
   "because", "that's why", "this", "it", "he/she/they" that point back to something unseen.
2. ONE COMPLETE STORY - setup (who / what / why it matters) -> the turn or insight -> the payoff.
   Include every setup sentence the payoff needs. The viewer must never wonder "who is he talking
   about?" or "what happened before this?". If a person, company or event is referred to, the reel
   must contain the sentence that introduces it. Prefer a longer complete story over a short broken one.
3. PAYOFF ENDING - the last sentence lands the point (conclusion, punchline, lesson, emotional peak).
   NEVER end on a setup, a question that is answered later, "for example", a list that continues,
   or a sentence that leads into the next one.
4. LENGTH between {min_len} and {max_len} seconds (sum of sentence lengths).
5. TAKEAWAY TEST - you must be able to write, in one sentence, what the viewer learns or feels.
   If you cannot, the moment is not good enough: pick another.

Return ONLY JSON - a list of objects:
[{{"start_id": 12, "end_id": 25,
   "takeaway": "the one thing the viewer learns or feels",
   "hook_title": "on-screen hook, max 7 words, punchy, creates curiosity",
   "caption": "Instagram caption, 1-2 lines, may use 1-2 emoji, ends with a question to drive comments",
   "hashtags": "#5 #to #8 #relevant #hashtags",
   "score": 0-100 virality score,
   "reason": "why it works, one line"}}]
Order by score, best first."""

REVIEW_PROMPT = """You are checking Reel cuts made from a video transcript. For each reel you get its sentences,
plus a few sentences just BEFORE and just AFTER it for context.

Fix each reel if needed by moving start_id / end_id (only to sentence ids shown):
- The FIRST sentence must make sense to someone who never saw the video (no dangling "and/so/this/that/it",
  no reference to something said earlier). If it does not, move start_id to an earlier or later sentence.
- The LAST sentence must finish the thought (a conclusion or punchline). If the idea continues in the
  "after" sentences, extend end_id to where it really ends; if the reel trails off, cut back to the last
  strong sentence.
- Keep length between {min_len} and {max_len} seconds.

{reels}

Return ONLY JSON: [{{"reel": 1, "start_id": .., "end_id": .., "ok_before": true/false, "note": "what you changed"}}]"""


def _sent_lines(sents, a, b):
    hot = {1: " | HOT", 2: " | HOT+"}
    return "\n".join(
        f"[S{i} @{sents[i]['start']:.0f} | {sents[i]['end'] - sents[i]['start']:.1f}s{hot.get(sents[i].get('hot', 0), '')}] {sents[i]['text']}"
        for i in range(max(0, a), min(len(sents), b + 1)))


AUDIENCE_YES = """AUDIENCE DATA: sentences marked HOT (top 20%) or HOT+ (top 5%) are the parts real YouTube viewers
re-watched the most ("Most replayed"). This is strong evidence of what is engaging. Prefer moments built
around HOT/HOT+ sentences - but a moment still has to pass every rule below."""
AUDIENCE_NO = "(No audience replay data is available for this video - judge by content alone.)"


def _fix_length(a, b, sents):
    """Keep the clip inside MIN..MAX seconds while staying on sentence boundaries."""
    a, b = max(0, min(a, len(sents) - 1)), max(0, min(b, len(sents) - 1))
    if b < a:
        a, b = b, a
    # a complete ending matters more than the exact length: allow 20% over the target (never over 90s)
    limit = min(HARD_MAX, MAX_CLIP * 1.2)
    while b > a and sents[b]["end"] - sents[a]["start"] > limit:
        b -= 1
    while b + 1 < len(sents) and sents[b]["end"] - sents[a]["start"] < MIN_CLIP \
            and sents[b + 1]["end"] - sents[a]["start"] <= MAX_CLIP:
        b += 1
    return a, b


WORD_PROMPT = """This transcript has NO punctuation, so sentence boundaries were only guessed from pauses.
For each reel below you get the words around its START and around its END as  index:word .
Judge by meaning (any language) and return the exact boundaries:
- start_word = index of the FIRST word of the sentence that should open the reel (a clean beginning,
  not the tail of the previous sentence).
- end_word = index of the LAST word of the sentence that completes the reel's final thought
  (never stop in the middle of a sentence or just before its last words).
Only use indices that are shown.

{reels}

Return ONLY JSON: [{{"reel": 1, "start_word": 123, "end_word": 456}}]"""


def refine_words(picks, sents, words, status=None):
    """Unpunctuated transcripts: let the AI choose the exact first and last WORD of each reel."""
    from . import ai
    if status:
        status("AI is fixing exact start and end words…")
    blocks, wins = [], []
    for k, p in enumerate(picks, 1):
        s0, s1 = sents[max(0, p["a"] - 1)]["w0"], min(sents[p["a"]]["w1"], sents[p["a"]]["w0"] + 25)
        e0 = max(sents[p["b"]]["w0"], sents[p["b"]]["w1"] - 25)
        e1 = sents[min(len(sents) - 1, p["b"] + 1)]["w1"]
        s0, e1 = max(s0, s1 - 60), min(e1, e0 + 60)
        wins.append((s0, s1, e0, e1))
        line = lambda x, y: " ".join(f"{i}:{words[i]['word']}" for i in range(x, y + 1))
        blocks.append(f"=== Reel {k} ===\nSTART area: {line(s0, s1)}\nEND area: {line(e0, e1)}")
    try:
        res = ai.ask_json(WORD_PROMPT.format(reels="\n\n".join(blocks)))
    except Exception as e:
        print(f"[ai] word-level pass skipped: {e}")
        return
    for f in res if isinstance(res, list) else []:
        try:
            k, sw, ew = int(f["reel"]) - 1, int(f["start_word"]), int(f["end_word"])
        except (KeyError, ValueError, TypeError):
            continue
        if not 0 <= k < len(picks):
            continue
        s0, s1, e0, e1 = wins[k]
        if s0 <= sw <= s1 and e0 <= ew <= e1 and words[ew]["end"] - words[sw]["start"] <= HARD_MAX:
            picks[k]["t0"], picks[k]["t1"] = words[sw]["start"], words[ew]["end"]


def pick_with_ai(sents, title, n, status=None, words=None, has_punct=True, profile=""):
    from . import ai
    say = status or (lambda m: None)
    transcript = _sent_lines(sents, 0, len(sents) - 1)
    if len(transcript) > 400_000:                    # extremely long video: keep it within model limits
        transcript = transcript[:400_000]
    say("AI is reading the whole transcript…")
    if not has_punct:
        transcript = ("NOTE: this transcript has no punctuation. Each numbered line is a caption line, NOT a full "
                      "sentence - a real sentence may start or end in the middle of a line. Choose lines so the "
                      "whole thought is inside; exact words are fixed in a later step.\n") + transcript
    has_heat = any("hot" in x for x in sents)
    raw = ai.ask_json(PICK_PROMPT.format(title=title, transcript=transcript, n=n + 2, profile=profile,
                                         audience=AUDIENCE_YES if has_heat else AUDIENCE_NO,
                                         min_len=int(MIN_CLIP), max_len=int(MAX_CLIP)))
    picks = []
    for c in raw if isinstance(raw, list) else raw.get("reels", []):
        try:
            a, b = _fix_length(int(c["start_id"]), int(c["end_id"]), sents)
        except (KeyError, ValueError, TypeError):
            continue
        if any(not (b < p["a"] or a > p["b"]) for p in picks):   # overlap
            continue
        picks.append({"a": a, "b": b, **c})
        if len(picks) >= n:
            break
    if not picks:
        return []

    # second pass: the AI double-checks each cut's first and last sentence with context around it
    say("AI is double-checking where each reel starts and ends…")
    blocks = []
    for k, p in enumerate(picks, 1):
        blocks.append(f"=== Reel {k}: S{p['a']}..S{p['b']} ===\n"
                      f"-- before --\n{_sent_lines(sents, p['a'] - 4, p['a'] - 1)}\n"
                      f"-- reel --\n{_sent_lines(sents, p['a'], p['b'])}\n"
                      f"-- after --\n{_sent_lines(sents, p['b'] + 1, p['b'] + 5)}")
    try:
        fixes = ai.ask_json(REVIEW_PROMPT.format(reels="\n\n".join(blocks),
                                                 min_len=int(MIN_CLIP), max_len=int(MAX_CLIP + 5)))
        for f in fixes if isinstance(fixes, list) else []:
            k = int(f.get("reel", 0)) - 1
            if 0 <= k < len(picks):
                a, b = _fix_length(int(f["start_id"]), int(f["end_id"]), sents)
                others = [p for j, p in enumerate(picks) if j != k]
                if abs(a - picks[k]["a"]) <= 6 and abs(b - picks[k]["b"]) <= 8 and \
                        all(b < p["a"] or a > p["b"] for p in others):
                    if (a, b) != (picks[k]["a"], picks[k]["b"]):
                        print(f"[ai] reel {k + 1} adjusted S{picks[k]['a']}-S{picks[k]['b']} -> S{a}-S{b}: {f.get('note', '')}")
                    picks[k]["a"], picks[k]["b"] = a, b
    except Exception as e:
        print(f"[ai] review pass skipped: {e}")

    if words and not has_punct:
        refine_words(picks, sents, words, status)

    out = []
    for p in picks:
        hot = max((sents[k].get("hot", 0) for k in range(p["a"], p["b"] + 1)), default=0)
        why = p.get("takeaway") or p.get("reason") or "AI pick"
        out.append({
            "start": p.get("t0", sents[p["a"]]["start"]), "end": p.get("t1", sents[p["b"]]["end"]),
            "score": float(p.get("score", 50) or 50),
            "title": (p.get("hook_title") or sents[p["a"]]["text"])[:80],
            "caption": p.get("caption", ""), "hashtags": p.get("hashtags") or DEFAULT_HASHTAGS,
            "reason": ("🔥 Most replayed · " if hot else "") + why,
        })
    return out


# ---------------------------------------------------------------- heuristic picker
def pick_heuristic(sents, loudness, title, n, curve=None):
    if not sents:
        return []
    loud = np.asarray(loudness, dtype=float)
    loud_norm = (loud - loud.mean()) / (loud.std() + 1e-6) if len(loud) > 1 else loud * 0

    heat_mu, heat_sd, heat_top = ((curve.mean(), curve.std() + 1e-6, np.percentile(curve, 85))
                                  if curve is not None else (0, 1, 9))

    candidates = []
    for i, first in enumerate(sents):
        start = first["start"]
        first_low = first["text"].lower()
        texts = []
        for j in range(i, len(sents)):
            end = sents[j]["end"]
            if end - start > MAX_CLIP:
                break
            texts.append(sents[j]["text"])
            if end - start < MIN_CLIP:
                continue
            txt = " ".join(texts).lower()
            dur = end - start
            pace = len(txt.split()) / dur
            hooks = sum(txt.count(h) for h in HOOK_WORDS)
            first_hooks = sum(first_low.count(h) for h in HOOK_WORDS)
            a, b = int(start), max(int(start) + 1, int(end))
            energy = float(loud_norm[a:b].mean()) if b <= len(loud_norm) else 0.0
            last = sents[j]["text"].rstrip()
            heat = float(curve[a:b].mean()) if curve is not None and a < len(curve) else None
            score = (
                (0 if heat is None else (heat - heat_mu) / heat_sd * 3.5)   # what viewers actually re-watch
                + energy * (1.0 if heat is not None else 2.0)
                + min(pace, 4.0)
                + hooks * 0.5
                + first_hooks * 1.5                              # strong opening line matters most
                + ("?" in first["text"]) * 1.0                   # opening question = good hook
                - first_low.startswith(WEAK_STARTS) * 2.5        # don't open mid-thought
                - last.endswith("?") * 2.0                       # don't end on an unanswered question
                + last.endswith(("!", ".")) * 0.5
                - abs(dur - 38) * 0.03                           # sweet spot ~38s
            )
            candidates.append((score, start, end, i, heat))

    candidates.sort(reverse=True)
    chosen = []
    for spacing in (45, 0):                       # first try picks from different parts of the video
        for score, s, e, i, heat in candidates:
            if len(chosen) >= n:
                break
            if all(e <= c["start"] - spacing or s >= c["end"] + spacing for c in chosen):
                hook = sents[i]["text"].strip()
                hook = hook if len(hook) <= 70 else hook[:67].rsplit(" ", 1)[0] + "…"
                chosen.append({
                    "start": s, "end": e, "score": round(score * 10, 1),
                    "title": hook, "caption": f"{hook} 👀\n\nFrom: {title}",
                    "hashtags": DEFAULT_HASHTAGS,
                    "reason": ("🔥 Most replayed part of the video" if heat is not None and heat >= heat_top
                               else "Picked by simple rules - turn on the AI editor for smarter picks"),
                })
    return chosen


def _quiet_runs(db, thr, i0, i1, min_len):
    """Yield (start, end) frame index of runs below thr inside [i0, i1)."""
    run = None
    for i in range(max(0, i0), min(len(db), i1)):
        if db[i] < thr:
            run = i if run is None else run
        elif run is not None:
            if i - run >= min_len:
                yield run, i
            run = None
    if run is not None and min(len(db), i1) - run >= min_len:
        yield run, min(len(db), i1)


def snap_to_audio(clip, fine, next_word_start=None):
    """Move the cut points into real pauses in the audio (caption timings can be off by a few tenths)."""
    db, hop = fine
    if len(db) < 50:
        return False
    lo, hi = np.percentile(db, 10), np.percentile(db, 90)
    thr = lo + (hi - lo) * 0.3                       # 'quiet' = clearly below normal speech level
    f = lambda t: int(t / hop)

    # ---- end: the first real pause after the last word (>= 140 ms)
    se = clip["speech_end"]
    limit = se + 1.2 if next_word_start is None else min(se + 1.2, max(se + 0.25, next_word_start + 0.5))
    runs = list(_quiet_runs(db, thr, f(se - 0.25), f(limit), 7))
    if runs:
        a, b = runs[0]
        clip["speech_end"] = max(clip["start"] + 1, a * hop + 0.04)
        clip["end"] = min(b * hop - 0.03, a * hop + TAIL)
    else:                                            # no pause (music / fast talker): cut at the quietest instant
        i0, i1 = f(se - 0.12), f(se + 0.3)
        if i1 > i0 + 1 and i1 <= len(db):
            clip["end"] = (i0 + int(np.argmin(db[i0:i1]))) * hop + 0.02
            clip["speech_end"] = min(clip["speech_end"], clip["end"])
    # ---- start: begin just after the pause that precedes the first word
    st = clip["start"] + LEAD                        # the first word's own start time
    runs = list(_quiet_runs(db, thr, f(st - 0.7), f(st + 0.2), 4))
    if runs:
        clip["start"] = max(0.0, runs[-1][1] * hop - 0.06)
    else:
        i0, i1 = f(max(0, st - 0.3)), f(st + 0.06)
        if i1 > i0 + 1:
            clip["start"] = max(0.0, (i0 + int(np.argmin(db[i0:i1]))) * hop)
    return True


def find_viral_moments(segments, loudness, title, n=3, status=None, heatmap=None, fine=None, limits=None,
                       profile=""):
    from . import ai
    if limits:
        set_limits(*limits)
    sents, words, has_punct = build_sentences(segments)
    curve = heat_curve(heatmap, sents[-1]["end"] if sents else 0)
    if mark_hot(sents, curve) and status:
        status("Using YouTube's 'Most replayed' data to find what viewers love…")
    method, clips = "heuristic", []
    prov = ai.provider()
    if prov:
        try:
            clips = pick_with_ai(sents, title, n, status, words, has_punct, profile)
            method = prov
        except Exception as e:
            print(f"[viral] AI picker failed, using auto picker: {e}")
            if status:
                status(f"AI unavailable ({str(e)[:80]}) - using auto picker…")
    if not clips:
        clips = pick_heuristic(sents, loudness, title, n, curve)
    wstarts = [w["start"] for w in words] or [x["start"] for x in sents]
    for c in clips:
        c["speech_end"] = c["end"]                   # captions stop here
        nxt = next((t for t in wstarts if t >= c["end"] - 0.01), None)
        gap = (nxt - c["end"]) if nxt is not None else TAIL
        # cut before the next word begins (never include the first syllable of the next line)
        c["end"] = c["end"] + max(0.04, min(TAIL, gap - 0.08))
        c["start"] = max(0.0, c["start"] - LEAD)
        if fine is not None:
            try:
                snap_to_audio(c, fine, nxt)
            except Exception as e:
                print(f"[viral] audio snap skipped: {e}")
    clips.sort(key=lambda c: c["start"])
    return clips, method


# ---------------------------------------------------------------- understanding the video + summary reel
UNDERSTAND_PROMPT = """You are an expert video editor. First UNDERSTAND this video, then build ONE short "summary reel"
from the speaker's own words that gives a stranger the whole idea in under a minute.

Video title: {title}

Transcript as numbered lines:  [S<id> @<start seconds> | <length>s] text
{transcript}

STEP 1 - PROFILE. Work out: who is speaking (name/role if it can be told, and their style), who they are
talking to, the ONE main message of the video, and the 3-6 key points that support it.

STEP 2 - SUMMARY REEL. Choose 4 to 6 short pieces from ANYWHERE in the video and put them in the order
that tells the story best (this may differ from the order they were spoken):
   hook (a line that stops the scroll) -> the problem -> the key insight -> proof / example -> closing line.
Rules for every piece:
- it is 1 to 3 consecutive lines, S<start_id>..S<end_id>, and lasts 4 to 18 seconds;
- it is a COMPLETE sentence or thought by itself - it must make sense when it follows the previous piece
  directly, with nothing in between (no dangling "and", "so", "that's why", "it", "they");
- no two pieces say the same thing; no greetings, sponsor reads or filler.
The whole reel must be {min_total}-{max_total} seconds and feel like one continuous argument.

Return ONLY JSON:
{{"profile": {{"speaker": "...", "audience": "...", "main_message": "one sentence",
              "key_points": ["...", "..."], "tone": "..."}},
  "summary": {{"hook_title": "on-screen title, max 7 words",
              "caption": "Instagram caption, 1-2 lines, ends with a question",
              "hashtags": "#5 #to #8 #hashtags",
              "pieces": [{{"start_id": 3, "end_id": 4, "role": "hook"}}, {{"start_id": 88, "end_id": 89, "role": "insight"}}]}}}}"""


def profile_text(profile):
    if not isinstance(profile, dict) or not profile:
        return ""
    pts = profile.get("key_points") or []
    return ("\nWHAT THIS VIDEO IS ABOUT (use it to judge what matters and who is being talked about):\n"
            f"- Speaker: {profile.get('speaker', '?')} | Tone: {profile.get('tone', '?')}\n"
            f"- Audience: {profile.get('audience', '?')}\n"
            f"- Main message: {profile.get('main_message', '?')}\n"
            + "".join(f"- Key point: {x}\n" for x in pts[:6]))


def understand_video(segments, title, fine=None, status=None, min_total=30, max_total=60):
    """AI reads the whole video. -> (profile text for later prompts, summary-reel plan or None)."""
    from . import ai
    if not ai.provider():
        return "", None
    sents, words, has_punct = build_sentences(segments)
    if len(sents) < 12:
        return "", None
    if status:
        status("AI is understanding the whole video…")
    try:
        res = ai.ask_json(UNDERSTAND_PROMPT.format(title=title, transcript=_sent_lines(sents, 0, len(sents) - 1)[:400_000],
                                                   min_total=min_total, max_total=max_total))
    except Exception as e:
        print(f"[ai] understanding step skipped: {e}")
        return "", None
    if not isinstance(res, dict):
        return "", None
    profile = res.get("profile") or {}
    summ = res.get("summary") or {}

    pieces, used = [], set()
    for pc in summ.get("pieces") or []:
        try:
            a, b = int(pc["start_id"]), int(pc["end_id"])
        except (KeyError, ValueError, TypeError):
            continue
        a, b = max(0, min(a, len(sents) - 1)), max(0, min(b, len(sents) - 1))
        if b < a:
            a, b = b, a
        while b > a and sents[b]["end"] - sents[a]["start"] > 20:
            b -= 1
        if used & set(range(a, b + 1)):
            continue                                      # the same line twice
        used |= set(range(a, b + 1))
        pieces.append((a, b))
    total = lambda ps: sum(sents[b]["end"] - sents[a]["start"] for a, b in ps)
    while len(pieces) > 2 and total(pieces) > max_total * 1.25:
        pieces.pop(len(pieces) // 2)                      # too long: drop a middle piece, keep hook + ending
    if len(pieces) < 2 or total(pieces) < 12:
        return profile_text(profile), None

    wstarts = [w["start"] for w in words] or [x["start"] for x in sents]
    parts = []
    for k, (a, b) in enumerate(pieces):
        c = {"start": max(0.0, sents[a]["start"] - LEAD), "end": sents[b]["end"], "speech_end": sents[b]["end"]}
        nxt = next((t for t in wstarts if t >= c["end"] - 0.01), None)
        gap = (nxt - c["end"]) if nxt is not None else TAIL
        c["end"] = c["end"] + max(0.04, min(TAIL, gap - 0.08))
        if fine is not None:
            try:
                snap_to_audio(c, fine, nxt)
            except Exception as e:
                print(f"[viral] audio snap skipped: {e}")
        if k < len(pieces) - 1:                           # jump cuts stay tight
            c["end"] = min(c["end"], c["speech_end"] + 0.2)
        if c["end"] - c["start"] >= 1.5:
            parts.append(c)
    if len(parts) < 2:
        return profile_text(profile), None
    msg = profile.get("main_message") or "the main idea of the video"
    return profile_text(profile), {
        "parts": parts, "title": (summ.get("hook_title") or title)[:80],
        "caption": summ.get("caption") or msg, "hashtags": summ.get("hashtags") or DEFAULT_HASHTAGS,
        "reason": f"🎬 Summary reel ({len(parts)} moments) · {msg}", "score": 100.0,
    }
