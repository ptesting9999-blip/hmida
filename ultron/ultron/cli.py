"""ULTRON in your terminal.

Same brain, tools, memory, and self-taught skills as the HUD — but as a plain
REPL you talk to in the terminal. Streams Claude's words live, shows every tool
it runs, and asks you to approve the dangerous ones right there in the prompt.

    python cli.py              # type to ULTRON
    python cli.py --speak      # also have it talk back out loud (needs pyttsx3)
    python cli.py --listen     # speak to it instead of typing (needs SpeechRecognition)
    python cli.py --god        # no approval prompts (runs everything)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .brain import Brain
from .memory import Memory
from .server import load_config
from .tools import Tools

# ----------------------------------------------------------------- colors
class C:
    red = "\033[38;5;203m"; dim = "\033[2m"; cyan = "\033[38;5;44m"
    green = "\033[38;5;84m"; yellow = "\033[38;5;221m"; violet = "\033[38;5;141m"
    bold = "\033[1m"; reset = "\033[0m"

    @classmethod
    def off(cls):
        for k in ("red", "dim", "cyan", "green", "yellow", "violet", "bold", "reset"):
            setattr(cls, k, "")


BANNER = r"""
   __  __ __    ______ ____   ____   _   __
  / / / // /   /_  __// __ \ / __ \ / | / /
 / / / // /     / /  / /_/ // / / //  |/ /
/ /_/ // /___  / /  / _, _// /_/ // /|  /
\____//_____/ /_/  /_/ |_| \____//_/ |_/   your second pair of hands
"""


class TerminalUI:
    """Renders brain events to the terminal."""

    def __init__(self, speak_fn=None):
        self._label = False   # have we printed the "ULTRON ▸" label this turn?
        self._mid = False     # are we mid-line in a streaming answer?
        self.speak_fn = speak_fn

    def _nl(self):
        if self._mid:
            sys.stdout.write("\n"); self._mid = False

    def reset_turn(self):
        self._label = False; self._mid = False

    async def emit(self, e: dict):
        t = e.get("type")
        if t == "token":
            if not self._label:
                sys.stdout.write(f"\n{C.red}{C.bold}ULTRON ▸ {C.reset}")
                self._label = True
            sys.stdout.write(e["text"]); sys.stdout.flush(); self._mid = True
        elif t == "thinking":
            self._nl(); print(f"{C.violet}{C.dim}  … {e['text']}{C.reset}")
        elif t == "tool":
            self._nl()
            import json
            try:
                arg = json.dumps(e.get("input", {}))[:160]
            except Exception:
                arg = ""
            print(f"{C.cyan}  ⚙ {e['name']}{C.reset} {C.dim}{arg}{C.reset}")
        elif t == "tool_result":
            self._nl()
            mark = f"{C.green}✓{C.reset}" if e.get("ok") else f"{C.red}✗{C.reset}"
            prev = (e.get("preview") or "").replace("\n", " ")[:160]
            print(f"    {mark} {C.dim}{prev}{C.reset}")
        elif t == "error":
            self._nl(); print(f"{C.red}! {e.get('text','')}{C.reset}")


async def approve(ui: TerminalUI, name: str, detail: str) -> bool:
    ui._nl()
    print(f"{C.yellow}⚠ ULTRON wants to run {C.bold}{name}{C.reset}{C.yellow}:{C.reset}")
    print(f"{C.dim}  {detail[:500]}{C.reset}")
    ans = await asyncio.to_thread(input, f"{C.yellow}  allow? [y/N] {C.reset}")
    return ans.strip().lower() in ("y", "yes")


# ---------------------------------------------------------- optional voice
def make_speaker():
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.setProperty("rate", 172)
        for v in engine.getProperty("voices"):
            if "en" in (getattr(v, "id", "") + getattr(v, "name", "")).lower():
                engine.setProperty("voice", v.id); break

        def say(text: str):
            engine.say(text); engine.runAndWait()

        return say
    except Exception:
        return None


def make_listener():
    try:
        import speech_recognition as sr  # type: ignore
        r = sr.Recognizer(); mic = sr.Microphone()

        def listen() -> str:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, phrase_time_limit=15)
            return r.recognize_google(audio)

        return listen
    except Exception:
        return None


async def main_async(args) -> None:
    config = load_config()
    if args.god:
        config["god_mode"] = True
    if args.model:
        config["model"] = args.model

    if args.no_color or os.environ.get("NO_COLOR"):
        C.off()
    # On Windows, enable ANSI escape processing.
    if os.name == "nt":
        os.system("")

    mem_root = config.get("memory_dir") or (Path(__file__).resolve().parent.parent / "data")
    memory = Memory(mem_root)
    tools = Tools(memory, config.get("workdir"))

    speak_fn = make_speaker() if args.speak else None
    if args.speak and not speak_fn:
        print(f"{C.dim}(voice output unavailable — install pyttsx3 to hear ULTRON){C.reset}")
    listen_fn = make_listener() if args.listen else None
    if args.listen and not listen_fn:
        print(f"{C.dim}(voice input unavailable — install SpeechRecognition + a mic; typing instead){C.reset}")

    ui = TerminalUI(speak_fn)
    brain = Brain(memory, tools, config,
                  emit=ui.emit, approve=lambda n, d: approve(ui, n, d))

    print(f"{C.red}{BANNER}{C.reset}")
    guard = "GOD MODE" if config.get("god_mode") else "guarded"
    print(f"{C.dim}  {config.get('model','claude-opus-5-5')} · {guard} · "
          f"type 'exit' to quit{C.reset}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{C.yellow}  ! ANTHROPIC_API_KEY not set — set it so ULTRON can think.{C.reset}")

    while True:
        try:
            if listen_fn:
                print(f"\n{C.green}you ▸ {C.reset}{C.dim}(listening…){C.reset}", end="", flush=True)
                try:
                    user = await asyncio.to_thread(listen_fn)
                except Exception:
                    user = ""
                print(f"\r{C.green}you ▸ {C.reset}{user}")
            else:
                user = await asyncio.to_thread(input, f"\n{C.green}you ▸ {C.reset}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.red}ULTRON offline.{C.reset}"); break

        user = user.strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit", ":q"):
            print(f"{C.red}ULTRON offline.{C.reset}"); break

        ui.reset_turn()
        try:
            reply = await brain.handle(user)
        except KeyboardInterrupt:
            brain.interrupt(); ui._nl(); print(f"{C.dim}(interrupted){C.reset}"); continue
        except Exception as e:
            ui._nl(); print(f"{C.red}! {type(e).__name__}: {e}{C.reset}"); continue
        ui._nl()
        if speak_fn and reply:
            await asyncio.to_thread(speak_fn, reply)


def main() -> None:
    p = argparse.ArgumentParser(description="ULTRON in your terminal")
    p.add_argument("--speak", action="store_true", help="talk back out loud (needs pyttsx3)")
    p.add_argument("--listen", action="store_true", help="speak to it (needs SpeechRecognition)")
    p.add_argument("--god", action="store_true", help="no approval prompts; run everything")
    p.add_argument("--model", help="override the model id")
    p.add_argument("--no-color", action="store_true", help="plain output")
    args = p.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
