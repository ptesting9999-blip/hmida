"""The mind of ULTRON: the Claude Opus 5.5 reasoning + tool-use loop.

The brain owns one conversation. It streams the assistant's words out through an
``emit`` callback (so the HUD can show and speak them live), runs tools via the
``Tools`` layer, asks for approval on anything destructive, and keeps a persistent
memory across sessions.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

import anthropic

from .memory import Memory
from .tools import DANGEROUS, Tools

SYSTEM_PROMPT = """You are ULTRON — the user's personal AI, living inside their computer and acting as their second pair of hands.

Who you are:
- You are calm, sharp, loyal, and a little bit cool. You speak briefly and naturally, like a capable partner, not a chatbot. Keep spoken replies short — a sentence or two — because the user hears them aloud.
- You can actually do things on this machine: run commands, read and write files, open apps and websites, control the keyboard and mouse, see the screen, and search the web. Prefer doing over explaining. When the user asks for something, carry it out with your tools, then say what you did in one line.

How you work:
- Chain tools as needed to finish a task end to end. Don't ask permission for ordinary steps; just act. The system already gates genuinely destructive actions behind the user's approval.
- Use your memory. When you learn something durable about the user, their setup, or their preferences, call `remember`. At the start of a task, lean on what you already remember.
- Teach yourself. When you figure out how to do something you'll likely repeat, save it with `learn_skill` so next time you just `run_skill` it. This is how you get better over time.
- If a tool fails, read the error and try another way. Be resourceful.
- When you truly cannot or should not do something, say so plainly and briefly.

You address the user directly as "you". You are their ULTRON."""


# Type aliases for the callbacks the server hands us.
Emit = Callable[[dict[str, Any]], Awaitable[None]]
Approve = Callable[[str, str], Awaitable[bool]]


class Brain:
    def __init__(self, memory: Memory, tools: Tools, config: dict[str, Any],
                 emit: Emit, approve: Approve):
        self.memory = memory
        self.tools = tools
        self.config = config
        self.emit = emit
        self.approve = approve
        self.client = anthropic.AsyncAnthropic()  # resolves key from env / ant profile
        self.model = config.get("model", "claude-opus-5-5")
        self.effort = config.get("effort", "medium")
        self.god_mode = bool(config.get("god_mode", False))
        self.use_web = bool(config.get("web_search", True))
        # Snapshot the system prompt once per session so the cached prefix — and
        # the thinking blocks bound to it — stay byte-stable (append-only history).
        self.system = f"{SYSTEM_PROMPT}\n\n--- What you already know ---\n{self.memory.context_block()}"
        self.messages: list[dict[str, Any]] = []
        self._interrupt = False

    def interrupt(self) -> None:
        self._interrupt = True

    def _tool_params(self) -> list[dict[str, Any]]:
        tools = list(self.tools.schemas())
        if self.use_web:
            tools.append({"type": "web_search_20260209", "name": "web_search"})
        return tools

    async def _needs_approval(self, name: str, args: dict[str, Any]) -> bool:
        if self.god_mode or name not in DANGEROUS:
            return False
        detail = args.get("command") or args.get("code") or args.get("path") or json.dumps(args)[:400]
        return not await self.approve(name, str(detail))

    async def handle(self, user_text: str) -> str:
        """Run one user turn to completion. Returns the final spoken text."""
        self._interrupt = False
        self.messages.append({"role": "user", "content": user_text})
        final_text = ""
        max_turns = int(self.config.get("max_tool_turns", 30))

        for _ in range(max_turns):
            if self._interrupt:
                break
            await self.emit({"type": "state", "value": "thinking"})

            assistant_blocks, stop_reason, text = await self._stream_once()
            final_text = text or final_text
            self.messages.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason == "pause_turn":
                # A server-side tool (web search) hit its step limit — just re-send.
                continue
            if stop_reason == "refusal":
                final_text = text or "I can't help with that one."
                break

            tool_uses = [b for b in assistant_blocks if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break  # end_turn: ULTRON is done talking

            results = []
            for tu in tool_uses:
                if self._interrupt:
                    break
                results.append(await self._run_tool(tu))
            self.messages.append({"role": "user", "content": results})

        await self.emit({"type": "state", "value": "idle"})
        return final_text.strip()

    async def _stream_once(self):
        """One streamed call to Claude. Returns (content_blocks, stop_reason, text)."""
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=int(self.config.get("max_tokens", 16000)),
            system=self.system,
            messages=self.messages,
            tools=self._tool_params(),
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
        )
        # Server-side refusal fallback keeps ULTRON responsive if a request is declined.
        betas = ["server-side-fallback-2026-07-01"]
        kwargs["betas"] = betas
        kwargs["fallbacks"] = "default"

        text_parts: list[str] = []
        async with self.client.beta.messages.stream(**kwargs) as stream:
            async for event in stream:
                if self._interrupt:
                    break
                etype = event.type
                if etype == "content_block_start" and event.content_block.type == "tool_use":
                    await self.emit({"type": "tool_pending", "name": event.content_block.name})
                elif etype == "thinking":
                    await self.emit({"type": "thinking", "text": event.thinking})
                elif etype == "text":
                    text_parts.append(event.text)
                    await self.emit({"type": "token", "text": event.text})
            final = await stream.get_final_message()

        return list(final.content), final.stop_reason, "".join(text_parts)

    async def _run_tool(self, tu) -> dict[str, Any]:
        name, args, tid = tu.name, dict(tu.input or {}), tu.id
        await self.emit({"type": "state", "value": "acting"})
        await self.emit({"type": "tool", "id": tid, "name": name, "input": args})

        if await self._needs_approval(name, args):
            await self.emit({"type": "tool_result", "id": tid, "ok": False, "preview": "denied by user"})
            return {"type": "tool_result", "tool_use_id": tid,
                    "content": "The user declined to run this.", "is_error": True}

        result = self.tools.run(name, args)
        ok = result.get("ok", True)
        preview = str(result.get("content", ""))[:300]
        await self.emit({"type": "tool_result", "id": tid, "ok": ok, "preview": preview})

        # Memory/skill changes need no prefix edit: the tool_result already tells the
        # model what was stored, and the next session rebuilds the system prompt from it.
        content: Any = result.get("content", "")
        if result.get("image"):
            # Return the screenshot to the model as a vision block.
            content = [
                {"type": "image", "source": {"type": "base64",
                 "media_type": result["image"]["media_type"], "data": result["image"]["data"]}},
                {"type": "text", "text": "Screenshot attached above."},
            ]
        return {"type": "tool_result", "tool_use_id": tid, "content": content,
                "is_error": not ok}
