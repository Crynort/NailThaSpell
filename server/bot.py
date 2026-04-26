"""Spell Bee bot — the Pipecat pipeline.

Run this through the FastAPI server in `server.py`. When a browser opens
the page and grants microphone access, `server.py` creates a fresh
WebRTC peer connection, hands it to `run_bot()`, and we build a pipeline
that:

    [browser mic]
        │ audio
        ▼
    transport.input()
        ▼
    stt          (Deepgram speech-to-text, streaming)
        ▼
    spell_bee    (custom processor — game logic, evaluation;
                  asynchronously calls Groq LLM for varied feedback)
        ▼
    tts          (Deepgram text-to-speech, streaming)
        ▼
    transport.output()
        │ audio
        ▼
    [browser speakers]

Plus an RTVI processor on the side that lets the frontend receive
game-state updates over the WebRTC data channel.

Pipecat handles everything around it: VAD detects speech, interruptions
flush in-flight TTS, and frame ordering is preserved automatically.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from spell_bee.processor import SpellBeeProcessor
from stt_config import build_stt

load_dotenv(override=True)


async def run_bot(webrtc_connection):
    """Run a single Spell Bee session for one connected browser."""
    logger.info("Starting spell bee bot for connection %s", webrtc_connection.pc_id)

    # --- Transport: WebRTC to the browser -------------------------------
    # Silero VAD gives us reliable speaking-start / speaking-stop events
    # which Pipecat translates into interruption frames automatically.
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                start_secs=0.2,      # increased from 0.05 to resist quick noises
                stop_secs=0.9,       # slightly longer to prevent chopping
                confidence=0.75,     # require higher confidence it is a human
                min_volume=0.6,      # ignore quiet background noise
            )
        )
    )

    # --- STT: Deepgram streaming ----------------------------------------
    stt = build_stt()

    # --- TTS: Deepgram streaming ----------------------------------------
    # We use Deepgram for both STT and TTS so the project only needs
    # one API key. aura-2-helena-en is a clear, friendly default voice
    # that works well for the spell-bee host persona.
    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        voice=os.getenv("DEEPGRAM_TTS_VOICE", "aura-2-helena-en"),
    )

    # --- Custom game logic ---------------------------------------------
    spell_bee = SpellBeeProcessor()

    # --- RTVI: lets the frontend receive game-state messages -----------
    rtvi = RTVIProcessor()

    pipeline = Pipeline([
        transport.input(),
        vad,              # emits VADUserStarted/StoppedSpeakingFrame
        stt,
        rtvi,             # so RTVIServerMessageFrames from spell_bee reach the client
        spell_bee,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            allow_interruptions=True,
            enable_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected, starting first round")
        # Greet the user, then begin the first round.
        await spell_bee.begin_round()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected, cancelling task")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
