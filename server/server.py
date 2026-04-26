"""FastAPI server: serves the frontend and handles WebRTC signaling.

We use Pipecat's SmallWebRTCConnection (a thin aiortc wrapper) so the
browser can establish a peer-to-peer WebRTC link with the bot without
needing Daily, LiveKit, or any external service. Everything runs locally.

Endpoints
---------
GET  /                         → the spell bee frontend (static HTML)
POST /api/offer                → browser sends WebRTC SDP offer; we
                                 spin up a bot for that connection and
                                 return the SDP answer
"""

from __future__ import annotations

import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from bot import run_bot

load_dotenv(override=True)

# Track active connections so we can clean them up on shutdown.
_active_connections: dict[str, SmallWebRTCConnection] = {}

CLIENT_DIR = Path(__file__).parent.parent / "client"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # On shutdown, gracefully tear down every WebRTC connection.
    logger.info("Shutting down %d active connections", len(_active_connections))
    coros = [pc.disconnect() for pc in _active_connections.values()]
    await asyncio.gather(*coros, return_exceptions=True)
    _active_connections.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    """Negotiate a WebRTC connection and start a bot for it."""
    pc_id = request.get("pc_id")

    if pc_id and pc_id in _active_connections:
        # The client is renegotiating an existing connection (e.g. after a
        # network blip). Just feed it the new offer.
        pc = _active_connections[pc_id]
        logger.info("Renegotiating existing connection %s", pc_id)
        await pc.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        # Fresh connection — build a new peer + bot.
        pc = SmallWebRTCConnection()
        await pc.initialize(sdp=request["sdp"], type=request["type"])

        @pc.event_handler("closed")
        async def on_closed(pc: SmallWebRTCConnection):
            logger.info("Connection %s closed", pc.pc_id)
            _active_connections.pop(pc.pc_id, None)

        background_tasks.add_task(run_bot, pc)

    answer = pc.get_answer()
    _active_connections[answer["pc_id"]] = pc
    return answer


# Mount the static frontend last so /api/* routes win.
app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(CLIENT_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    logger.info("Starting Spell Bee server on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
