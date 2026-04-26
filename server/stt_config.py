"""STT configuration — edit this file to tune speech recognition.

Swap model, language, keyterms, endpointing, etc. here without touching
the pipeline in bot.py.
"""

from __future__ import annotations

import os

from pipecat.services.deepgram.stt import DeepgramSTTService

# Letters + phonetic names to bias the model toward isolated letters.
LETTER_KEYTERMS: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "ay", "bee", "cee", "dee", "ee", "eff", "gee", "aitch",
    "eye", "jay", "kay", "ell", "em", "en", "oh", "pee",
    "cue", "ar", "ess", "tee", "you", "vee",
    "double-you", "ex", "why", "zee",
]


def build_stt() -> DeepgramSTTService:
    """Return a configured DeepgramSTTService ready to drop into the pipeline."""
    return DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            # --- Model -------------------------------------------------
            # nova-3-general: best accuracy, supports language="multi"
            # nova-2-general: slightly lower accuracy, supports en-IN etc.
            model="nova-3-general",

            # --- Language ----------------------------------------------
            # "multi"  → auto-detect; handles Indian/British/etc. English
            # "en-IN"  → force Indian English (nova-2 only)
            # "en"     → force English exclusively
            language="en",

            # --- Letter boosting ---------------------------------------
            # keyterm is nova-3's token-boost API (nova-2 uses keywords).
            keyterm=LETTER_KEYTERMS,

            # --- Timing ------------------------------------------------
            # endpointing: ms of silence before Deepgram finalises a segment
            endpointing=500,
            # utterance_end_ms: group fragments into one utterance
            utterance_end_ms=1500,

            # --- Misc --------------------------------------------------
            interim_results=True,
            punctuate=False,
            profanity_filter=False,
        ),
    )
