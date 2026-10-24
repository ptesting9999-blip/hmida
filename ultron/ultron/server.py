"""The nervous system: a small web server that serves the HUD and connects it to
the brain over a WebSocket.

Voice happens in the browser (it hears you with the Web Speech API and speaks with
speech synthesis), so the server stays light and cross-platform. Each browser
connection gets its own Brain, sharing one persistent Memory.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .brain import Brain
from .memory import Memory
from .tools import Tools

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def load_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    for name in ("config.json", "config.example.json"):
        p = root / name
        if p.exists():
            return json.loads(p.read_text("utf-8"))
    return {}


def create_app() -> FastAPI:
    config = load_config()
    mem_root = config.get("memory_dir") or (Path(__file__).resolve().parent.parent / "data")
    memory = Memory(mem_root)

    app = FastAPI(title="ULTRON")
    app.state.config = config
    app.state.memory = memory

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "hud.html").read_text("utf-8"))

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        session = Session(socket, memory, config)
        await session.run()

    return app


class Session:
    """Bridges one browser connection to one Brain."""

    def __init__(self, socket: WebSocket, memory: Memory, config: dict[str, Any]):
        self.socket = socket
        self.config = config
        self.tools = Tools(memory, config.get("workdir"))
        self._approvals: dict[str, asyncio.Future] = {}
        self._busy = False
        self.brain = Brain(memory, self.tools, config, self.emit, self.approve)

    async def emit(self, event: dict[str, Any]) -> None:
        try:
            await self.socket.send_json(event)
        except Exception:
            pass

    async def approve(self, name: str, detail: str) -> bool:
        """Ask the HUD to confirm a dangerous action; wait for the answer."""
        aid = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._approvals[aid] = fut
        await self.emit({"type": "approval_request", "id": aid, "name": name, "detail": detail})
        try:
            return await asyncio.wait_for(fut, timeout=120)
        except asyncio.TimeoutError:
            return False
        finally:
            self._approvals.pop(aid, None)

    async def run(self) -> None:
        await self.emit({"type": "ready", "god_mode": self.brain.god_mode,
                         "model": self.brain.model})
        try:
            while True:
                msg = await self.socket.receive_json()
                await self._on_message(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def _on_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "user_text":
            text = (msg.get("text") or "").strip()
            if text and not self._busy:
                asyncio.create_task(self._think(text))
        elif kind == "approve":
            fut = self._approvals.get(msg.get("id", ""))
            if fut and not fut.done():
                fut.set_result(bool(msg.get("ok")))
        elif kind == "interrupt":
            self.brain.interrupt()

    async def _think(self, text: str) -> None:
        self._busy = True
        try:
            reply = await self.brain.handle(text)
            if reply:
                await self.emit({"type": "say", "text": reply})
            await self.emit({"type": "assistant_done", "text": reply})
        except Exception as e:
            await self.emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
            await self.emit({"type": "state", "value": "idle"})
        finally:
            self._busy = False
