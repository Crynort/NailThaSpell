"""LLM helper for varied feedback phrasing.

The deterministic SpellBeeProcessor knows *what* to tell the user
(correct / incorrect / score). An LLM is overkill for that decision.
But canned phrases like "Correct! You spelled apple perfectly." get
tedious after a few rounds.

This module wraps a Groq chat call so the processor can ask:

    "Generate a one-sentence celebratory reaction to a correct spelling
     of 'apple', with a friendly spell-bee-host tone."

…and get back a fresh, natural sentence each round. The LLM never
decides correctness — that's the processor's job — it only paraphrases.

Why Groq specifically: free tier, OpenAI-compatible, very low TTFT
(typically 200-300ms for llama-3.3-70b), which keeps the conversation
snappy. If the LLM call fails or times out, we fall back to the canned
phrase so the game keeps playing — never block the pipeline on this.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI


# Single shared client. Groq exposes an OpenAI-compatible endpoint, so we
# use the OpenAI SDK pointed at Groq's base URL — same approach the
# Pipecat GroqLLMService uses internally.
_client: Optional[AsyncOpenAI] = None


def _get_client() -> Optional[AsyncOpenAI]:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — LLM feedback will use fallback phrases")
        return None
    _client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    return _client


_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_TIMEOUT_SECS = 2.0  # If the LLM is slow, fall back instantly.


_SYSTEM_PROMPT = """You are an upbeat, encouraging spell bee host. \
Generate ONE short sentence (max 15 words) reacting to the user's \
spelling attempt. Do not include the word's letters. Do not ask \
questions. Do not add follow-up instructions — those are handled \
elsewhere. Just the reaction, one sentence."""


_SANITIZE_SYSTEM_PROMPT = """You are a precise letter extraction algorithm.
Your only job is to extract the exact sequence of letters a user spoke from a raw, noisy speech transcript.

RULES:
1. Extract ALL spoken letters exactly in the order they appear.
2. Convert phonetic letter representations (e.g., "bee" -> "b", "ay" -> "a", "double l" -> "ll") into their letter equivalents.
3. Completely ignore all conversational filler, hesitations, or surrounding words (e.g., "I think it is", "um", "the spelling is").
4. MAINTAIN THE EXACT LETTER SEQUENCE SPOKEN. You are strictly forbidden from autocorrecting, adding, or modifying letters to form a valid dictionary word.
5. Output the result combined into a single lowercase string with NO spaces, punctuation, or explanations.
6. If the transcript contains no recognizable spelling attempt at all, output exactly the word "UNCLEAR"."""


async def sanitize_transcript(transcript: str, fallback: str) -> str:
    """Ask the LLM to sanitize the raw speech transcript into just the intended spelling."""
    client = _get_client()
    if client is None:
        return fallback

    user_msg = f"Raw transcript: '{transcript}'\n\nSanitized intended spelling:"

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SANITIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=20,
                temperature=0.1,  # Low temperature for predictability
            ),
            timeout=_TIMEOUT_SECS,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return fallback
        return text.strip().strip('"').strip("'").strip()
    except asyncio.TimeoutError:
        logger.warning("Groq sanitize timed out after %.1fs — using fallback", _TIMEOUT_SECS)
        return fallback
    except Exception as e:
        logger.warning("Groq sanitize failed (%s) — using fallback", e)
        return fallback


async def rephrase_feedback(
    *,
    word: str,
    correct: bool,
    score: int,
    attempted: int,
    fallback: str,
) -> str:
    """Ask the LLM for a fresh feedback sentence.

    Returns the LLM output on success, or `fallback` on any failure
    (no key, timeout, network error, empty response). The caller is
    responsible for the *informational* part of the reply (e.g. spelling
    out the correct letters, announcing the score) — this only handles
    the colorful opener.
    """
    client = _get_client()
    if client is None:
        return fallback

    if correct:
        user_msg = (
            f"The user just correctly spelled '{word}'. "
            f"Their score is now {score} out of {attempted}. "
            "Give a short, energetic reaction."
        )
    else:
        user_msg = (
            f"The user just attempted to spell '{word}' but got it wrong. "
            f"Their score is {score} out of {attempted}. "
            "Give a short, kind, encouraging reaction — no shaming."
        )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=60,
                temperature=0.8,  # A bit of variety, but not unhinged.
            ),
            timeout=_TIMEOUT_SECS,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return fallback
        # Strip trailing whitespace and quotes the model sometimes adds.
        return text.strip().strip('"').strip("'").strip()
    except asyncio.TimeoutError:
        logger.warning("Groq feedback timed out after %.1fs — using fallback", _TIMEOUT_SECS)
        return fallback
    except Exception as e:  # noqa: BLE001 — we genuinely want to swallow any failure here
        logger.warning("Groq feedback failed (%s) — using fallback", e)
        return fallback
