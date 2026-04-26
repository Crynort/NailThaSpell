# 🐝 Spell Bee — A Voice Bot Built on Pipecat

A voice-based spell bee game. The bot speaks a word, you spell it letter-by-letter
into your mic, and the bot evaluates your answer in real time over a WebRTC voice
call. Built with [Pipecat](https://docs.pipecat.ai), Deepgram (STT + TTS), and
Groq (for varied feedback phrasing).

```
┌─────────┐   audio    ┌──────────────┐   text   ┌─────────────────┐   text   ┌──────────────┐   audio   ┌─────────┐
│ Browser │ ─────────► │   Deepgram   │ ───────► │ SpellBeeProcessor│ ───────► │   Deepgram   │ ────────► │ Browser │
│   mic   │   WebRTC   │     STT      │          │ (game logic)    │          │     TTS      │   WebRTC  │ speaker │
└─────────┘            └──────────────┘          └────┬────┬───────┘          └──────────────┘           └─────────┘
                                                      │    │
                                                      │    └── Groq LLM (async, for varied
                                                      │        feedback phrasing — never
                                                      │        decides correctness)
                                                      │
                                                      │ RTVI data channel
                                                      ▼
                                              ┌──────────────┐
                                              │  UI: score,  │
                                              │  streak,     │
                                              │  history     │
                                              └──────────────┘
```

---

## Quick start

### 1. Prerequisites

- **Python 3.10+** (3.11 recommended)
- A modern browser (Chrome, Edge, or Firefox — Safari works but is fussier with WebRTC)
- API keys for two services (both have generous free tiers):
  - **Deepgram** — streaming STT *and* TTS → https://console.deepgram.com/ ($200 free credits)
  - **Groq** — fast LLM for varied feedback phrasing → https://console.groq.com/ (free tier)

### 2. Install

```bash
git clone <this-repo>
cd spell-bee-bot

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# then open .env and paste your two API keys
```

### 4. Run

```bash
cd server
python server.py
```

Open **http://localhost:7860** in your browser, click **Start Session**, allow microphone
access, and start playing. The bot will announce the first word — spell it out loud,
letter by letter ("A — P — P — L — E"), and the bot will tell you whether you got it right.

---

## Project structure

```
spell-bee-bot/
├── server/
│   ├── bot.py                    # Pipeline: transport → STT → spell_bee → TTS
│   ├── server.py                 # FastAPI + WebRTC signaling endpoint
│   └── spell_bee/
│       ├── __init__.py
│       ├── words.py              # Hardcoded word list (easy/medium/hard tiers)
│       ├── spelling_parser.py    # Parses "ay pee pee el ee" → "apple"
│       ├── llm_feedback.py       # Async Groq call for varied reactions
│       └── processor.py          # ★ Custom FrameProcessor (game logic)
├── client/
│   └── index.html                # Single-page UI with live score & history
├── requirements.txt
├── .env.example
└── README.md
```

---

## How it works

### The pipeline (server/bot.py)

A Pipecat **Pipeline** is an ordered chain of **FrameProcessors** that pass **Frames**
to each other. Each frame is a typed message — audio, transcribed text, an
interruption signal, etc. The pipeline for this bot is:

```python
pipeline = Pipeline([
    transport.input(),   # WebRTC audio in from the browser
    stt,                 # Deepgram: audio → TranscriptionFrame
    rtvi,                # Forwards game-state messages to the frontend
    spell_bee,           # ★ Our custom processor — the game itself
    tts,                 # Deepgram: TTSSpeakFrame → audio
    transport.output(),  # WebRTC audio out to the browser
])
```

The order matters: audio has to be transcribed *before* the game logic can read it,
and text has to be synthesized *before* it can be played back. Each stage runs in
its own asyncio task, so everything streams concurrently.

### The custom processor (server/spell_bee/processor.py)

`SpellBeeProcessor` is the heart of the assignment. It's a `FrameProcessor` subclass
that:

1. **Watches the frame stream** for `TranscriptionFrame`s (final transcripts from STT)
   and `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame` (from VAD).
2. **Holds game state** — current round, score, streak, current word, phase.
3. **Drives the conversation** by pushing `TTSSpeakFrame`s downstream, which the
   TTS service then speaks.
4. **Pushes state updates to the UI** via `RTVIServerMessageFrame`s on the data channel.

A simplified view of the per-turn flow:

```
                     ┌─────────────────┐
                     │   begin_round() │
                     └────────┬────────┘
                              │ push TTSSpeakFrame("Round 1. Your word is apple...")
                              ▼
                     ┌─────────────────┐
                     │  phase=LISTENING │ ◄──── publish state to UI
                     └────────┬────────┘
                              │
                              ▼
              ┌──── UserStartedSpeakingFrame
              │     (clears pending buffer; auto-interrupts TTS via Pipecat)
              │
              ▼
       ┌─────────────────┐
       │ TranscriptionFrame  ──► append to pending_transcript
       └─────────────────┘
              │
              ▼
       ┌─────────────────┐
       │ UserStoppedSpeakingFrame
       └────────┬────────┘
                │ on next TranscriptionFrame, evaluate
                ▼
       ┌─────────────────┐
       │   _evaluate()   │
       │   - parse       │
       │   - compare     │
       │   - update state│
       │   - push feedback TTSSpeakFrame
       └────────┬────────┘
                │
                └──► begin_round()  (loop until MAX_ROUNDS)
```

### Hybrid design: deterministic logic + LLM polish

This bot uses both a custom `FrameProcessor` *and* an LLM, each for what
it's good at:

|                                  | Custom processor                        | LLM (Groq)                                |
|----------------------------------|-----------------------------------------|-------------------------------------------|
| Decides if a spelling is correct | ✅ deterministic, never wrong            | ❌ would hallucinate                       |
| Tracks score / round / streak    | ✅ exact integer state                   | ❌ would drift over many turns             |
| Reads out the correct letters    | ✅ from the source-of-truth word list    | ❌ would mis-spell hard words              |
| Generates varied openers         | ❌ only canned phrases                   | ✅ "Nice work!" / "You nailed that one!"   |

The flow per turn:

1. STT produces a transcript → `SpellBeeProcessor` parses it deterministically.
2. The processor decides correctness *itself* and updates state.
3. **Then** it asks Groq for a varied opener like "Brilliant!" via
   `spell_bee/llm_feedback.py` — a fire-and-fail-gracefully helper with a
   2-second timeout. If Groq is slow, unavailable, or unconfigured, the
   processor falls back to canned phrases and the game keeps playing.
4. The processor concatenates the LLM opener with the deterministic facts
   ("The correct spelling is: A, P, P, L, E. Your score is 2 out of 3.")
   and pushes it as a `TTSSpeakFrame`.

**Why isn't the LLM in the main pipeline?** A standard pipeline-resident
LLM would try to respond to *every* user transcript. That fights with our
spelling validation — the LLM would generate its own commentary on partial
spellings, the score would drift, the game logic would race. Calling the
LLM *from* the processor keeps it as a tool the processor uses, not a
co-driver. This pattern is worth highlighting in your code walkthrough.

### Spelling parser (server/spell_bee/spelling_parser.py)

The hardest practical problem in this project: the user's spoken spelling
arrives from STT in many shapes. All of these mean "apple":

```
"A P P L E"
"a-p-p-l-e"
"ay pee pee el ee"
"a, p, p, l, e"
"apple"
"I think it's a-p-p-l-e"
```

The parser handles them with a small pipeline:

1. Lowercase, drop apostrophe-s contractions (`it's` → `it`), strip punctuation.
2. Tokenize.
3. For each token: single letter? phonetic letter name like "ay"/"bee"/"see"? Take it.
   Filler word ("um", "the", "i think")? Drop it. Multi-character non-letter-name
   token in a single-token utterance? Treat it as the user saying the whole word.

Verified against 12 representative inputs (see the function's docstring for the
full set). Wrong spellings like `"appel"` correctly fail the comparison.

### Turn-taking & interruptions

These two are explicit requirements in the assignment, so worth calling out:

- **Turn-taking.** The processor commits a turn (evaluates the spelling) only after
  it sees `UserStoppedSpeakingFrame` *plus* a final `TranscriptionFrame`. This
  prevents premature evaluation on long words like "entrepreneur" where the user
  pauses between letters. Silero VAD's silence threshold gives the user about
  600ms of pause before the turn ends.

- **Interruptions.** When the user starts speaking, Silero VAD emits
  `UserStartedSpeakingFrame`. Pipecat's machinery automatically propagates an
  `InterruptionFrame` upstream that:
  1. Cancels the current TTS audio in flight (so the bot stops mid-sentence)
  2. Clears queued data frames in every processor (no stale audio after the
     interruption)
  3. Resets each processor's interruption hooks
  
  Our processor additionally clears its own `pending_transcript` buffer on
  `UserStartedSpeakingFrame` so an interruption mid-prompt doesn't pollute the
  next evaluation.

  Because system frames (interruptions) run on a higher-priority task than data
  frames (transcripts), we wrap state transitions in an `asyncio.Lock` to avoid
  races between an interruption resetting state and an evaluation reading it.

### The frontend (client/index.html)

A single static HTML page. No build step. It pulls Pipecat's official client
library (`@pipecat-ai/client-js`) and the `@pipecat-ai/small-webrtc-transport`
plugin from esm.sh.

The page:
1. POSTs an SDP offer to `/api/offer` to negotiate the WebRTC connection
2. Pipes the bot's audio track into a hidden `<audio>` element
3. Listens on the RTVI data channel for `{type: "game_state", state: {...}}`
   messages and re-renders the score / round / history live

---

## Configuration

Environment variables (set in `.env`):

| Variable | Purpose | Required |
|---|---|---|
| `DEEPGRAM_API_KEY` | Streaming STT and TTS (one key, both services) | ✅ |
| `GROQ_API_KEY` | LLM for varied feedback phrasing | ✅* |
| `DEEPGRAM_TTS_VOICE` | Override the default Deepgram voice | optional |
| `GROQ_MODEL` | Override the Groq model (default `llama-3.3-70b-versatile`) | optional |
| `HOST` | Server bind address (default `0.0.0.0`) | optional |
| `PORT` | Server port (default `7860`) | optional |

\* If `GROQ_API_KEY` is missing, the bot still works — it just falls back
to canned feedback phrases. Setting it makes the bot feel noticeably more alive.

Game tuning lives in code:

- **Word list** — `server/spell_bee/words.py` (3 difficulty tiers, 5 words each)
- **Max rounds per session** — `MAX_ROUNDS` in `server/spell_bee/processor.py` (default 5)

---

## Troubleshooting

**The bot can hear me but doesn't respond.** Make sure both API keys are valid.
Check the server logs — Deepgram and Groq will surface auth errors clearly.

**The bot keeps thinking my spelling is wrong.** Speak the letters with a brief
pause between each one ("A. P. P. L. E."). Deepgram tends to merge fast-spoken
letters into a single word. The parser is forgiving but not magical.

**Microphone permission denied.** WebRTC requires HTTPS in production, but
`localhost` is exempt — so use `http://localhost:7860`, not your machine's LAN IP.

**Audio is choppy.** WebRTC is sensitive to network conditions. Try a wired
connection or a different network. The bot also runs entirely locally apart
from the STT/TTS calls, so most of the latency is the cloud round-trip.

---

## Extending the bot

A few obvious next steps if you want to keep building:

- **Hints.** Add a `give_hint()` path triggered by the user saying "hint" — the
  parser already filters that word; just add a check before the spelling parse.
- **Adaptive difficulty.** Use the `streak` field to jump tiers faster or back
  off after a wrong answer.
- **Multi-user sessions.** `SmallWebRTCConnection` supports one peer; switch to
  `DailyTransport` for rooms with multiple players.
- **Persistence.** Log each session's history to SQLite so users can resume.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Voice orchestration | Pipecat | Required by the assignment; best-in-class for streaming voice pipelines |
| Transport | SmallWebRTCTransport (aiortc) | No external service required; runs entirely on localhost |
| STT | Deepgram (nova-3) | Low-latency streaming, good with letter-by-letter speech |
| TTS | Deepgram (aura-2) | Streaming, fast TTFT, natural voices, same key as STT |
| LLM | Groq (llama-3.3-70b-versatile) | Free tier, OpenAI-compatible, ~250ms TTFT keeps reactions snappy |
| VAD | Silero (bundled with Pipecat) | Reliable speech start/stop detection |
| Web server | FastAPI + uvicorn | Standard async Python web stack |
| Frontend | Vanilla HTML + Pipecat client-js (via esm.sh) | No build step |
