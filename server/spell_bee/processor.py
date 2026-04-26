"""SpellBeeProcessor — the custom FrameProcessor that runs the game.

This is the heart of the assignment. It implements the assignment's
"custom frame processors where needed (e.g., for spelling validation
logic)" requirement.

# Why a FrameProcessor?

A Pipecat pipeline is a chain of FrameProcessors. Each one can read frames
flowing past it and emit new frames. We could have wired the game logic
into the LLM with a system prompt — but a system prompt can't reliably
*compare strings*. The LLM might hallucinate that "appel" is correct, or
forget how many points the user has. A deterministic processor solves
both problems and gives us full control over turn flow.

# Where it sits in the pipeline

    transport.input → STT → spell_bee_processor → TTS → transport.output
                                  │
                                  └── also emits app messages to the UI

The processor sits *between* STT and TTS. It:

  - Watches for `TranscriptionFrame` (final user transcript from STT).
  - Maintains game state (round, score, current word, phase).
  - Decides what to say next and pushes a `TTSSpeakFrame` downstream so
    the TTS service speaks it.
  - Pushes `RTVIServerMessageFrame`s with state updates so the frontend
    can show the current score / word count / last result.

# Turn-taking and interruptions

The transport's VAD (Silero) detects when the user starts speaking and
emits a `UserStartedSpeakingFrame`. Pipecat's pipeline machinery
automatically converts this into an `InterruptionFrame` that flushes
pending TTS audio. We additionally listen for `UserStartedSpeakingFrame`
ourselves so that if the user starts spelling *while the bot is still
saying the word*, we don't get confused — we just discard the
in-progress prompt and wait for the user's transcription.

We commit a turn (i.e. evaluate the spelling) when we see
`UserStoppedSpeakingFrame` followed by a final `TranscriptionFrame`.
That gives the user time to spell out a long word like "entrepreneur"
without us jumping the gun after the first letter.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from spell_bee.llm_feedback import rephrase_feedback, sanitize_transcript
from spell_bee.spelling_parser import is_correct, parse_spelling
from spell_bee.words import Word, get_word


# --- Game state ------------------------------------------------------------

class Phase(str, Enum):
    """Where in the round are we?"""
    IDLE = "idle"             # Waiting for the session to start
    PROMPTING = "prompting"   # Bot is announcing the word
    LISTENING = "listening"   # Waiting for the user to spell
    CONFIRMING = "confirming" # Waiting for user to confirm spelling
    EVALUATING = "evaluating" # Comparing the attempt
    GAME_OVER = "game_over"   # Out of words / user said "stop"


@dataclass
class GameState:
    round_index: int = 0
    score: int = 0
    attempted: int = 0
    streak: int = 0
    current_word: Optional[Word] = None
    phase: Phase = Phase.IDLE
    last_result: Optional[str] = None  # "correct" / "incorrect" / None
    last_attempt: str = ""
    proposed_attempt: str = ""
    history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["current_word"] = self.current_word.text if self.current_word else None
        d["phase"] = self.phase.value
        return d


# --- Processor -------------------------------------------------------------

# The maximum number of words in a single session before the bot wraps up.
MAX_ROUNDS = 5


class SpellBeeProcessor(FrameProcessor):
    """Drives the spell bee game.

    The processor is *stateful* but uses an asyncio.Lock around state
    transitions because Pipecat dispatches system frames (interruptions)
    on a separate task from data frames. Without the lock we could race
    between an interruption resetting state and an evaluation reading it.
    """

    def __init__(self, max_rounds: int = MAX_ROUNDS):
        super().__init__()
        self._max_rounds = max_rounds
        self._state = GameState()
        self._lock = asyncio.Lock()
        # Buffer for transcripts that arrive while the user is still
        # speaking. We concatenate them and only evaluate on stop.
        self._pending_transcript: List[str] = []
        self._user_speaking = False
        self._turn_ready = False  # True after UserStoppedSpeakingFrame

    # ----- public hooks the runner uses -----

    @property
    def state(self) -> GameState:
        return self._state

    async def begin_round(self):
        """Kick off the next round. Called from the bot runner once the
        client connects, and again after each evaluation."""
        async with self._lock:
            if self._state.round_index >= self._max_rounds:
                self._state.phase = Phase.GAME_OVER
                await self._publish_state()
                summary = (
                    f"That's the end of the game. "
                    f"You spelled {self._state.score} out of "
                    f"{self._state.attempted} words correctly. "
                    "Thanks for playing!"
                )
                await self._speak(summary)
                return

            word = get_word(self._state.round_index)
            self._state.current_word = word
            self._state.phase = Phase.PROMPTING
            self._state.last_result = None
            self._state.last_attempt = ""
            self._pending_transcript = []

        prompt = (
            f"Round {self._state.round_index + 1}. "
            f"Your word is: {word.text}. "
            f"{word.text}. "
            "Please spell the word."
        )
        await self._publish_state()
        await self._speak(prompt)

    # ----- FrameProcessor contract -----

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # ALWAYS call super first. This handles StartFrame, EndFrame,
        # InterruptionFrame, etc. for us.
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Pipeline is starting. We don't kick off the first round
            # here — the bot runner does it on `on_client_connected`
            # so the user's browser has finished negotiating audio.
            pass

        elif isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            # User is speaking. Note that Pipecat will already have
            # generated an InterruptionFrame for us if the bot was
            # talking — we just need to track game state.
            self._user_speaking = True
            self._turn_ready = False
            if self._state.phase in (Phase.PROMPTING, Phase.CONFIRMING, Phase.EVALUATING):
                print("\n[BOT] Stopped in between sentence due to user interruption!\n")
            if self._state.phase != Phase.LISTENING:
                self._pending_transcript = []
            logger.info("[MIC] User started speaking")
            await self.push_frame(InterruptionFrame())

        elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._user_speaking = False
            self._turn_ready = True
            logger.info("[MIC] User stopped speaking; pending tokens={}",
                        len(self._pending_transcript))
            asyncio.create_task(self._eval_after_grace())
            # Evaluation happens on the *next* TranscriptionFrame, since
            # final transcripts often arrive a beat after stop. If the
            # user stopped without producing a final transcript, the
            # next start_speaking will simply reset the buffer.

        elif isinstance(frame, BotStoppedSpeakingFrame):
            async with self._lock:
                if self._state.phase == Phase.PROMPTING:
                    self._state.phase = Phase.LISTENING
                    await self._publish_state()

        elif isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            logger.info("[STT] Transcript: {!r}", text)
            if text:
                self._pending_transcript.append(text)
            # Eval is fired by _eval_after_grace() scheduled at VAD stop;
            # late transcripts just append to pending and get picked up.

        # Forward every frame so downstream processors keep working. Without
        # this call, audio would never reach the TTS or the transport.
        await self.push_frame(frame, direction)

    # ----- internal -----

    async def _speak(self, text: str):
        print(f"\n[BOT SPEAKS] {text}\n")
        await self.push_frame(TTSSpeakFrame(text))

    async def _eval_after_grace(self):
        """Wait for any late Deepgram final transcripts after VAD stop,
        then evaluate the buffered transcript."""
        await asyncio.sleep(1.5)
        if not self._turn_ready:
            return  # already evaluated or interrupted
        if not self._pending_transcript:
            logger.info("[MIC] VAD triggered but STT returned no transcription. Ignoring noise...")
            return
        joined = " ".join(self._pending_transcript)
        self._pending_transcript = []
        self._turn_ready = False
        await self._evaluate(joined)

    async def _evaluate(self, transcript: str):
        """Validate the user's spelling and respond."""
        if self._state.phase not in (Phase.LISTENING, Phase.CONFIRMING):
            logger.debug("Ignoring transcript in phase %s: %r", self._state.phase, transcript)
            return

        if self._state.phase == Phase.CONFIRMING:
            logger.info("[CONFIRM] raw_transcript={!r}", transcript)
            t_lower = transcript.lower().strip()
            words = t_lower.split()
            if any(w in words for w in ["no", "nope", "wrong", "incorrect", "nah", "wait", "not"]):
                async with self._lock:
                    self._state.phase = Phase.PROMPTING
                    self._state.proposed_attempt = ""
                    self._pending_transcript = []
                await self._publish_state()
                await self._speak("Okay, let's try again. Please spell the word.")
                return
            elif any(w in words for w in ["yes", "yeah", "yep", "correct", "right", "sure", "yup", "ok", "true", "ya"]):
                attempt = self._state.proposed_attempt
            else:
                await self._speak("I didn't catch that. Please say yes if the spelling was correct, or no to try again.")
                return
        else:
            sanitized_transcript = await sanitize_transcript(transcript, fallback=transcript)
            logger.info("[EVAL] raw_transcript={!r} sanitized_transcript={!r}", transcript, sanitized_transcript)

            async with self._lock:
                if self._state.phase != Phase.LISTENING:
                    logger.debug("Ignoring transcript after sanitize in phase %s: %r",
                                 self._state.phase, sanitized_transcript)
                    return
                if self._state.current_word is None:
                    return
                
                target = self._state.current_word.text
                if sanitized_transcript == "UNCLEAR":
                    attempt = ""
                else:
                    attempt = parse_spelling(sanitized_transcript)
                
                unclear_check = (not attempt) or (len(attempt) < max(2, len(target) // 2))
                if unclear_check:
                    self._state.phase = Phase.PROMPTING
                else:
                    self._state.phase = Phase.CONFIRMING
                    self._state.proposed_attempt = attempt
            
            if unclear_check:
                await self._publish_state()
                await self._speak(f"Sorry, I didn't catch a valid spelling. Your word is {target}. Please spell it.")
                return
                
            await self._publish_state()
            spoken_letters = ", ".join(attempt.upper())
            await self._speak(f"I heard you spell: {spoken_letters}. Is that correct?")
            return

        # --- Evaluate the confirmed attempt ---
        async with self._lock:
            if self._state.phase != Phase.CONFIRMING:
                return
            if self._state.current_word is None:
                return
            self._state.phase = Phase.EVALUATING
            target = self._state.current_word.text
            self._state.last_attempt = attempt
            correct = is_correct(target, attempt) if attempt else False
            logger.info("[EVAL] target={!r} attempt={!r} correct={}",
                        target, attempt, correct)

            self._state.attempted += 1
            if correct:
                self._state.score += 1
                self._state.streak += 1
                self._state.last_result = "correct"
            else:
                self._state.streak = 0
                self._state.last_result = "incorrect"

            self._state.history.append({
                "word": target,
                "attempt": attempt,
                "correct": correct,
                "raw": self._state.proposed_attempt,
            })
            self._state.round_index += 1

        await self._publish_state()

        # Deterministic facts the bot must say (correct letters, score).
        if correct:
            fallback_opener = f"Correct! You spelled {target} perfectly."
            tail = (
                f" Your score is {self._state.score} out of "
                f"{self._state.attempted}."
            )
        else:
            fallback_opener = "Not quite."
            tail = (
                f" The correct spelling is: "
                f"{', '.join(target.upper())}. "
                f"That spells {target}. "
                f"Your score is {self._state.score} out of "
                f"{self._state.attempted}."
            )

        # Ask the LLM for a varied opener. Falls back to the canned
        # phrase if Groq is slow / unavailable / unconfigured — the
        # game keeps playing either way.
        opener = await rephrase_feedback(
            word=target,
            correct=correct,
            score=self._state.score,
            attempted=self._state.attempted,
            fallback=fallback_opener,
        )
        feedback = opener.rstrip(".!?") + "." + tail
        await self._speak(feedback)

        # Small delay so the feedback finishes before the next prompt.
        await asyncio.sleep(0.2)

        if not correct:
            # Wrong answer ends the game.
            async with self._lock:
                self._state.phase = Phase.GAME_OVER
            await self._publish_state()
            summary = (
                f"Game over. You spelled {self._state.score} "
                f"out of {self._state.attempted} words correctly. "
                "Thanks for playing!"
            )
            await self._speak(summary)
            return

        await self.begin_round()

    async def _publish_state(self):
        """Push a state update to the frontend via the RTVI data channel."""
        msg = RTVIServerMessageFrame(data={
            "type": "game_state",
            "state": self._state.to_dict(),
        })
        await self.push_frame(msg)
