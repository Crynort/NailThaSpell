from __future__ import annotations

import asyncio
import webbrowser
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from spell_bee.llm_feedback import process_turn, rephrase_feedback
from spell_bee.words import Word, get_word


def _is_correct(target: str, attempt: str) -> bool:
    return attempt.strip().lower() == target.strip().lower()


MAX_ROUNDS = 5


@dataclass
class GameState:
    round_index: int = 0
    score: int = 0
    attempted: int = 0
    streak: int = 0
    current_word: Optional[Word] = None
    last_result: Optional[str] = None
    last_attempt: str = ""
    proposed_attempt: str = ""           # spelling buffered during confirm step
    history: List[dict] = field(default_factory=list)
    messages: List[dict] = field(default_factory=list)  # LLM conversation memory

    def to_dict(self) -> dict:
        return {
            "round_index": self.round_index,
            "score": self.score,
            "attempted": self.attempted,
            "streak": self.streak,
            "current_word": self.current_word.text if self.current_word else None,
            "last_result": self.last_result,
            "last_attempt": self.last_attempt,
            "history": self.history,
        }


class SpellBeeProcessor(FrameProcessor):
    """
    Drives the spell bee game.

    Turn flow is tracked through LLM conversation history (self._state.messages)
    rather than an explicit phase enum. The LLM reads prior turns to decide
    whether the user is spelling a word or confirming a previous attempt.
    Score and round tracking are always deterministic — the LLM only handles
    intent classification and natural language generation.
    """

    def __init__(self, max_rounds: int = MAX_ROUNDS):
        super().__init__()
        self._max_rounds = max_rounds
        self._state = GameState()
        self._lock = asyncio.Lock()
        self._pending_transcript: List[str] = []
        self._user_speaking = False
        self._turn_ready = False

    @property
    def state(self) -> GameState:
        return self._state

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def begin_round(self):
        """Announce the next word, or end the game if all rounds are done."""
        async with self._lock:
            if self._state.round_index >= self._max_rounds:
                await self._end_game()
                return

            word = get_word(self._state.round_index)
            self._state.current_word = word
            self._state.last_result = None
            self._state.last_attempt = ""
            self._state.proposed_attempt = ""
            self._pending_transcript = []

        prompt = (
            f"Round {self._state.round_index + 1}. "
            f"Your word is: {word.text}. "
            f"{word.text}. "
            "Please spell the word letter by letter."
        )
        await self._bot_say(prompt)

    # ------------------------------------------------------------------
    # Pipecat frame handler
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._user_speaking = True
            self._turn_ready = False
            self._pending_transcript = []
            logger.info("[MIC] User started speaking")
            await self.push_frame(InterruptionFrame())

        elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._user_speaking = False
            self._turn_ready = True
            logger.info("[MIC] User stopped speaking; pending tokens={}", len(self._pending_transcript))
            asyncio.create_task(self._eval_after_grace())

        elif isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            logger.info("[STT] Transcript: {!r}", text)
            if text:
                self._pending_transcript.append(text)

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Turn evaluation
    # ------------------------------------------------------------------

    async def _eval_after_grace(self):
        """Wait for any late STT finals after VAD stop, then evaluate."""
        await asyncio.sleep(1.5)
        if not self._turn_ready or not self._pending_transcript:
            if not self._pending_transcript:
                logger.info("[MIC] VAD triggered but STT returned nothing — ignoring noise")
            return
        transcript = " ".join(self._pending_transcript)
        self._pending_transcript = []
        self._turn_ready = False
        await self._evaluate(transcript)

    async def _evaluate(self, transcript: str):
        if self._state.current_word is None:
            return

        target = self._state.current_word.text

        # Catch "user said the word instead of spelling it" before the LLM call.
        if transcript.strip().lower() == target.lower():
            await self._bot_say(f"You said the word! Please spell {target} out letter by letter.")
            return

        result = await process_turn(
            messages=list(self._state.messages),
            current_word=target,
            proposed_attempt=self._state.proposed_attempt or None,
            score=self._state.score,
            attempted=self._state.attempted,
            round_index=self._state.round_index,
            max_rounds=self._max_rounds,
            transcript=transcript,
        )

        action = result["action"]
        logger.info("[EVAL] action={} transcript={!r}", action, transcript)

        self._state.messages.append({"role": "user", "content": transcript})

        if action == "ignore":
            return

        if action == "repeat_word":
            reply = result.get("reply") or f"Your word is {target}. Please spell it letter by letter."
            await self._bot_say(reply)

        elif action == "spelling_attempt":
            await self._handle_spelling_attempt(result, target)

        elif action == "confirm_no":
            self._state.proposed_attempt = ""
            await self._bot_say(f"No problem. Please spell {target} again, letter by letter.")

        elif action == "confirm_yes":
            await self._handle_confirmed_attempt(target)

    async def _handle_spelling_attempt(self, result: dict, target: str):
        raw = (result.get("parsed_spelling") or "").strip().lower()
        attempt = "".join(c for c in raw if c.isalpha())

        if not attempt or len(attempt) < max(2, len(target) // 2):
            await self._bot_say(f"I couldn't catch a clear spelling. Your word is {target}. Please spell it letter by letter.")
            return

        self._state.proposed_attempt = attempt
        spoken_letters = ", ".join(attempt.upper())
        await self._bot_say(f"I heard you spell: {spoken_letters}. Is that correct?")

    async def _handle_confirmed_attempt(self, target: str):
        attempt = self._state.proposed_attempt
        if not attempt:
            await self._bot_say(f"Let me give you the word again. Your word is {target}. Please spell it.")
            return

        correct = _is_correct(target, attempt)

        async with self._lock:
            self._state.last_attempt = attempt
            self._state.attempted += 1
            if correct:
                self._state.score += 1
                self._state.streak += 1
                self._state.last_result = "correct"
            else:
                self._state.streak = 0
                self._state.last_result = "incorrect"
            self._state.history.append({"word": target, "attempt": attempt, "correct": correct})
            self._state.round_index += 1

        await self._publish_state()

        if correct:
            fallback = f"Correct! You spelled {target} perfectly."
            tail = f" Your score is {self._state.score} out of {self._state.attempted}."
        else:
            fallback = "Not quite."
            tail = (
                f" The correct spelling is: {', '.join(target.upper())}. "
                f"That spells {target}. "
                f"Your score is {self._state.score} out of {self._state.attempted}."
            )

        opener = await rephrase_feedback(
            word=target,
            correct=correct,
            score=self._state.score,
            attempted=self._state.attempted,
            fallback=fallback,
        )
        await self._bot_say(opener.rstrip(".!?") + "." + tail)
        await asyncio.sleep(0.2)

        if not correct:
            await self._end_game()
            return

        await self.begin_round()

    async def _end_game(self):
        summary = (
            f"Game over. You spelled {self._state.score} "
            f"out of {self._state.attempted} words correctly. "
            "Thanks for playing!"
        )
        self._state.current_word = None
        # webbrowser.open("https://youtube.com/shorts/PV3ezuSb3DQ?si=s4IsxszdP6JWHxku")
        await self._publish_state()
        await self._bot_say(summary)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _bot_say(self, text: str):
        print(f"\n[BOT SPEAKS] {text}\n")
        self._state.messages.append({"role": "assistant", "content": text})
        await self._publish_state()
        await self.push_frame(TTSSpeakFrame(text))

    async def _publish_state(self):
        await self.push_frame(RTVIServerMessageFrame(data={
            "type": "game_state",
            "state": self._state.to_dict(),
        }))
