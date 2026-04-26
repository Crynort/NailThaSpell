"""Parse a user's spoken spelling into a sequence of letters.

The STT will hand us a transcript like one of these:

    "A P P L E"
    "a-p-p-l-e"
    "ay pee pee el ee"
    "a, p, p, l, e"
    "apple"                 # user just said the word
    "I think it's a-p-p-l-e"

We need to extract the letters the user actually intended to spell. The
strategy is a small pipeline:

1. Lowercase, strip punctuation.
2. Tokenize on whitespace.
3. For each token, try (in order):
   - Single letter? Take it.
   - Phonetic letter name (e.g. "ay", "bee", "see")? Map it.
   - Otherwise: if it's the only token, treat it as the user saying the
     whole word and explode it into letters. Otherwise drop it as filler.
"""

from __future__ import annotations

import re
from typing import List

# NATO + common spoken letter pronunciations. Many STTs surface "B" as
# "bee", "C" as "see", etc. We intentionally accept both spellings of a
# few ambiguous ones (e.g. "are"/"r", "you"/"u").
LETTER_NAMES = {
    "ay": "a", "a": "a",
    "bee": "b", "be": "b", "b": "b",
    "see": "c", "sea": "c", "cee": "c", "c": "c",
    "dee": "d", "d": "d",
    "ee": "e", "e": "e",
    "ef": "f", "eff": "f", "f": "f",
    "gee": "g", "g": "g",
    "aitch": "h", "h": "h", "haitch": "h",
    "eye": "i", "i": "i",
    "jay": "j", "j": "j",
    "kay": "k", "k": "k",
    "el": "l", "ell": "l", "l": "l",
    "em": "m", "m": "m",
    "en": "n", "n": "n",
    "oh": "o", "o": "o",
    "pee": "p", "p": "p",
    "queue": "q", "cue": "q", "q": "q",
    "ar": "r", "are": "r", "r": "r",
    "es": "s", "ess": "s", "s": "s",
    "tee": "t", "tea": "t", "t": "t",
    "you": "u", "yoo": "u", "u": "u",
    "vee": "v", "v": "v",
    "double-u": "w", "double u": "w", "doubleyou": "w", "w": "w",
    "ex": "x", "x": "x",
    "why": "y", "wye": "y", "y": "y",
    "zee": "z", "zed": "z", "z": "z",
}

_FILLER = {"the", "letter", "letters", "spelling", "spell", "is", "it",
           "its", "thats", "think", "um", "uh", "okay", "ok",
           "so", "and", "then", "well", "like", "maybe"}


def parse_spelling(transcript: str) -> str:
    """Return the user's intended spelling as a lowercase letter string.

    Returns an empty string if nothing letter-like was found.
    """
    if not transcript:
        return ""

    # Normalize: lowercase, drop "'s"/"'re"/"'m" contractions entirely so
    # they don't leak stray letters, then replace separators with spaces
    # and keep only letters and whitespace.
    text = transcript.lower()
    text = re.sub(r"\b(it|that|here|there|what)'s\b", r"\1", text)
    text = re.sub(r"'(s|re|m|ll|ve|d)\b", "", text)
    text = re.sub(r"[-_,./]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    tokens = text.split()

    # Special case: the user uttered exactly one token of length >= 2 and it
    # isn't a known letter name. They probably said the whole word.
    if len(tokens) == 1 and len(tokens[0]) >= 2 and tokens[0] not in LETTER_NAMES:
        return tokens[0]

    letters: List[str] = []
    for tok in tokens:
        if tok in _FILLER:
            continue
        if tok in LETTER_NAMES:
            letters.append(LETTER_NAMES[tok])
        elif len(tok) == 1 and tok.isalpha():
            letters.append(tok)
        # If a token is multi-character and not a letter name, it's noise —
        # e.g. "I think a p p l e". We ignore it rather than guessing.

    return "".join(letters)


def is_correct(target: str, attempt: str) -> bool:
    """Compare the parsed attempt against the target word, case-insensitively."""
    return attempt.strip().lower() == target.strip().lower()
