"""LLM helpers — intent classification and feedback generation for spell bee."""

from __future__ import annotations

import asyncio
import json
import os
from typing import List, Optional

from loguru import logger
from openai import AsyncOpenAI


_client: Optional[AsyncOpenAI] = None

_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_TIMEOUT_SECS = 2.5


def _get_client() -> Optional[AsyncOpenAI]:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — LLM calls will use fallbacks")
        return None
    _client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


# ---------------------------------------------------------------------------
# process_turn
# ---------------------------------------------------------------------------

_GAME_HOST_SYSTEM = """\
You are a spell bee game host AI. You track game context through conversation history.

Given the conversation history and the user's latest utterance, determine the user's \
intent and generate an appropriate bot reply.

Return valid JSON ONLY — no markdown, no explanation:
{
  "action": "<one of the actions below>",
  "parsed_spelling": "<lowercase letters only, no spaces>" or null,
  "reply": "<what the bot should say next>"
}

Actions:
- "spelling_attempt": user is spelling out letters (e.g. "A P P L E", "ay pee el ee")
- "confirm_yes": bot asked "Is that correct?" and user confirmed (yes/yeah/right/correct/yep/ok/sure)
- "confirm_no": bot asked "Is that correct?" and user rejected (no/nope/wrong/wait/not)
- "repeat_word": user wants the word repeated or asked what the word is
- "ignore": ambient noise, silence, gibberish, or completely unclear intent

Rules:
- Infer context from history. If the last bot message asked for a spelling, expect "spelling_attempt".
  If the last bot message asked for confirmation, expect "confirm_yes" or "confirm_no".
- For "spelling_attempt": extract ONLY the letters the user explicitly spelled out one by one.
  If the user says the full word as a word (e.g. "happy spelled as h a p p y"), extract only
  the individual letters "happy" — never the spoken word itself. Do NOT autocorrect or fix typos.
- For "confirm_yes"/"confirm_no": reply field can be left empty (""), the game handles it.
- For "repeat_word": include the actual word in the reply.
- For "ignore": set reply to empty string "".
"""


async def process_turn(
    *,
    messages: List[dict],
    current_word: str,
    proposed_attempt: Optional[str],
    score: int,
    attempted: int,
    round_index: int,
    max_rounds: int,
    transcript: str,
) -> dict:
    """Classify user intent from conversation history.

    Returns: {action, parsed_spelling, reply}. Falls back to action=ignore on failure.
    """
    client = _get_client()
    if client is None:
        return {"action": "ignore", "parsed_spelling": None, "reply": ""}

    state_line = (
        f"Game state — word: '{current_word}', "
        f"round {round_index + 1}/{max_rounds}, score {score}/{attempted}."
    )
    if proposed_attempt:
        state_line += f" Awaiting confirmation for attempt: '{proposed_attempt}'."

    call_messages = [{"role": "system", "content": state_line + "\n\n" + _GAME_HOST_SYSTEM}]
    call_messages.extend(messages[-20:])
    call_messages.append({"role": "user", "content": transcript})

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=call_messages,
                max_tokens=150,
                temperature=0.2,
                response_format={"type": "json_object"},
            ),
            timeout=_TIMEOUT_SECS,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        action = result.get("action", "ignore")
        if action not in ("spelling_attempt", "confirm_yes", "confirm_no", "repeat_word", "ignore"):
            action = "ignore"
        return {
            "action": action,
            "parsed_spelling": result.get("parsed_spelling"),
            "reply": result.get("reply", ""),
        }
    except asyncio.TimeoutError:
        logger.warning("Groq process_turn timed out — falling back to ignore")
        return {"action": "ignore", "parsed_spelling": None, "reply": ""}
    except Exception as e:
        logger.warning("Groq process_turn failed (%s) — falling back to ignore", e)
        return {"action": "ignore", "parsed_spelling": None, "reply": ""}


# ---------------------------------------------------------------------------
# rephrase_feedback
# ---------------------------------------------------------------------------

_FEEDBACK_SYSTEM = """You are an upbeat, encouraging spell bee host. \
Generate ONE short sentence (max 15 words) reacting to the user's \
spelling attempt. Do not include the word's letters. Do not ask \
questions. Do not add follow-up instructions — those are handled \
elsewhere. Just the reaction, one sentence."""


async def rephrase_feedback(
    *,
    word: str,
    correct: bool,
    score: int,
    attempted: int,
    fallback: str,
) -> str:
    """Return a varied one-sentence opener for a correct/incorrect result.

    The caller appends the deterministic score/spelling info — this only
    generates the colorful reaction. Returns fallback on any failure.
    """
    client = _get_client()
    if client is None:
        return fallback

    user_msg = (
        f"The user just correctly spelled '{word}'. Score: {score}/{attempted}. Short energetic reaction."
        if correct else
        f"The user spelled '{word}' wrong. Score: {score}/{attempted}. Short kind reaction, no shaming."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _FEEDBACK_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=60,
                temperature=0.8,
            ),
            timeout=_TIMEOUT_SECS,
        )
        text = (response.choices[0].message.content or "").strip().strip('"').strip("'").strip()
        return text or fallback
    except asyncio.TimeoutError:
        logger.warning("Groq rephrase_feedback timed out — using fallback")
        return fallback
    except Exception as e:
        logger.warning("Groq rephrase_feedback failed (%s) — using fallback", e)
        return fallback
