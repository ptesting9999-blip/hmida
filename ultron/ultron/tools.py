"""The hands of ULTRON: everything it can actually do on the machine.

Each tool is a plain function returning a short string (what Claude reads back).
GUI control (keyboard/mouse/screenshot) is optional and degrades gracefully when
no desktop is available, so the backend still runs on a headless server.
"""

from __future__ import annotations

import base64
import os
import platform
import shlex
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

from .memory import Memory

# Tools whose side effects can be destructive. The server asks the user to
# approve these before they run, unless god-mode is enabled in the config.
DANGEROUS = {"run_shell", "run_python", "write_file", "delete_file", "run_skill"}


def _clip(text: str, limit: int = 6000) -> str:
    text = text if isinstance(text, str) else str(text)
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
    return text


class Tools:
    def __init__(self, memory: Memory, workdir: str | Path | None = None):
        self.memory = memory
        self.workdir = Path(workdir or Path.home()).expanduser()
        self._gui = None  # lazily imported pyautogui, or False if unavailable

    # ------------------------------------------------------------------ GUI
    def _get_gui(self):
        if self._gui is None:
            try:
                import pyautogui  # type: ignore

                pyautogui.FAILSAFE = False
                self._gui = pyautogui
            except Exception:
                self._gui = False
        return self._gui

    # ------------------------------------------------------- tool schemas
    def schemas(self) -> list[dict[str, Any]]:
        """JSON-schema definitions handed to the Claude API."""
        s = lambda **p: {"type": "object", "properties": p, "additionalProperties": False}
        return [
            {
                "name": "run_shell",
                "description": (
                    "Run a shell command on the user's computer and return its output. "
                    "Use this to launch programs, inspect the system, install things, or "
                    "run any CLI tool. This is your most general capability."
                ),
                "input_schema": {**s(command={"type": "string", "description": "The command line to run."},
                                     timeout={"type": "integer", "description": "Seconds before giving up (default 60)."}),
                                  "required": ["command"]},
            },
            {
                "name": "run_python",
                "description": "Execute a snippet of Python on the user's machine and return stdout/stderr. Good for quick computations, file wrangling, or using Python libraries.",
                "input_schema": {**s(code={"type": "string"}), "required": ["code"]},
            },
            {
                "name": "read_file",
                "description": "Read a text file and return its contents.",
                "input_schema": {**s(path={"type": "string"}), "required": ["path"]},
            },
            {
                "name": "write_file",
                "description": "Create or overwrite a text file with the given contents.",
                "input_schema": {**s(path={"type": "string"}, content={"type": "string"}), "required": ["path", "content"]},
            },
            {
                "name": "delete_file",
                "description": "Delete a file.",
                "input_schema": {**s(path={"type": "string"}), "required": ["path"]},
            },
            {
                "name": "list_dir",
                "description": "List the entries in a directory (defaults to the working directory).",
                "input_schema": s(path={"type": "string"}),
            },
            {
                "name": "open_url",
                "description": "Open a URL in the user's default web browser.",
                "input_schema": {**s(url={"type": "string"}), "required": ["url"]},
            },
            {
                "name": "open_app",
                "description": "Open an application or file by name using the OS launcher.",
                "input_schema": {**s(name={"type": "string"}), "required": ["name"]},
            },
            {
                "name": "type_text",
                "description": "Type text on the physical keyboard, as if the user typed it, into whatever window is focused. Requires a desktop session.",
                "input_schema": {**s(text={"type": "string"}), "required": ["text"]},
            },
            {
                "name": "press_keys",
                "description": "Press a keyboard shortcut, e.g. ['ctrl','s'] or ['enter']. Requires a desktop session.",
                "input_schema": {**s(keys={"type": "array", "items": {"type": "string"}}), "required": ["keys"]},
            },
            {
                "name": "mouse_click",
                "description": "Move the mouse to (x, y) and click. Requires a desktop session.",
                "input_schema": {**s(x={"type": "integer"}, y={"type": "integer"},
                                     button={"type": "string", "enum": ["left", "right", "double"]}),
                                  "required": ["x", "y"]},
            },
            {
                "name": "screenshot",
                "description": "Capture the screen so you can see what is on it. Returns the image to you directly. Requires a desktop session.",
                "input_schema": s(),
            },
            {
                "name": "remember",
                "description": "Store a durable fact about the user or the world so you recall it in future sessions (e.g. their name, preferences, project paths).",
                "input_schema": {**s(key={"type": "string"}, value={"type": "string"}), "required": ["key", "value"]},
            },
            {
                "name": "recall",
                "description": "Search your long-term memory. Omit the query to get everything you remember.",
                "input_schema": s(query={"type": "string"}),
            },
            {
                "name": "forget",
                "description": "Delete a stored fact by its key.",
                "input_schema": {**s(key={"type": "string"}), "required": ["key"]},
            },
            {
                "name": "learn_skill",
                "description": (
                    "Teach yourself a new reusable skill by saving a Python script. The script "
                    "should define a function run(**kwargs) that performs the skill. Use this when "
                    "you work out how to do something you expect to repeat, so next time you just "
                    "call run_skill instead of figuring it out again."
                ),
                "input_schema": {**s(name={"type": "string"}, description={"type": "string"}, code={"type": "string"}),
                                  "required": ["name", "description", "code"]},
            },
            {
                "name": "run_skill",
                "description": "Run one of the skills you previously taught yourself. Pass arguments as a JSON object in 'args'.",
                "input_schema": {**s(name={"type": "string"}, args={"type": "object"}), "required": ["name"]},
            },
            {
                "name": "list_skills",
                "description": "List the skills you have already taught yourself.",
                "input_schema": s(),
            },
        ]

    # ---------------------------------------------------------- dispatch
    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool. Returns {'content': <for Claude>, 'ok': bool,
        and optionally 'image': {media_type, data} for vision results}."""
        try:
            fn = getattr(self, f"_t_{name}", None)
            if fn is None:
                return {"ok": False, "content": f"Unknown tool: {name}"}
            return fn(**args)
        except TypeError as e:
            return {"ok": False, "content": f"Bad arguments for {name}: {e}"}
        except Exception as e:  # never let a tool crash the whole agent
            return {"ok": False, "content": f"{name} failed: {type(e).__name__}: {e}"}

    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.workdir / p)

    # ---------------------------------------------------------- tools
    def _t_run_shell(self, command: str, timeout: int = 60) -> dict[str, Any]:
        proc = subprocess.run(command, shell=True, cwd=str(self.workdir), capture_output=True,
                              text=True, timeout=timeout)
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        out = out.strip() or f"(no output, exit code {proc.returncode})"
        return {"ok": proc.returncode == 0, "content": _clip(out)}

    def _t_run_python(self, code: str) -> dict[str, Any]:
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(self.workdir),
                              capture_output=True, text=True, timeout=120)
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return {"ok": proc.returncode == 0, "content": _clip(out.strip() or "(no output)")}

    def _t_read_file(self, path: str) -> dict[str, Any]:
        p = self._resolve(path)
        return {"ok": True, "content": _clip(p.read_text("utf-8", errors="replace"))}

    def _t_write_file(self, path: str, content: str) -> dict[str, Any]:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, "utf-8")
        return {"ok": True, "content": f"Wrote {len(content)} chars to {p}"}

    def _t_delete_file(self, path: str) -> dict[str, Any]:
        p = self._resolve(path)
        p.unlink()
        return {"ok": True, "content": f"Deleted {p}"}

    def _t_list_dir(self, path: str = "") -> dict[str, Any]:
        p = self._resolve(path) if path else self.workdir
        entries = []
        for e in sorted(p.iterdir()):
            entries.append(("[dir] " if e.is_dir() else "      ") + e.name)
        return {"ok": True, "content": _clip(f"{p}:\n" + "\n".join(entries) or "(empty)")}

    def _t_open_url(self, url: str) -> dict[str, Any]:
        webbrowser.open(url)
        return {"ok": True, "content": f"Opened {url}"}

    def _t_open_app(self, name: str) -> dict[str, Any]:
        system = platform.system()
        if system == "Windows":
            os.startfile(name)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", name])
        else:
            subprocess.Popen(shlex.split(name))
        return {"ok": True, "content": f"Launched {name}"}

    def _t_type_text(self, text: str) -> dict[str, Any]:
        gui = self._get_gui()
        if not gui:
            return {"ok": False, "content": "No desktop session available for typing."}
        gui.typewrite(text, interval=0.01)
        return {"ok": True, "content": f"Typed {len(text)} characters."}

    def _t_press_keys(self, keys: list[str]) -> dict[str, Any]:
        gui = self._get_gui()
        if not gui:
            return {"ok": False, "content": "No desktop session available."}
        gui.hotkey(*keys)
        return {"ok": True, "content": f"Pressed {'+'.join(keys)}"}

    def _t_mouse_click(self, x: int, y: int, button: str = "left") -> dict[str, Any]:
        gui = self._get_gui()
        if not gui:
            return {"ok": False, "content": "No desktop session available."}
        gui.moveTo(x, y)
        if button == "double":
            gui.doubleClick()
        else:
            gui.click(button=button)
        return {"ok": True, "content": f"{button} click at ({x}, {y})"}

    def _t_screenshot(self) -> dict[str, Any]:
        gui = self._get_gui()
        if not gui:
            return {"ok": False, "content": "No desktop session available for screenshots."}
        import io

        img = gui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue()).decode()
        return {"ok": True, "content": "Screen captured.",
                "image": {"media_type": "image/png", "data": data}}

    def _t_remember(self, key: str, value: str) -> dict[str, Any]:
        return {"ok": True, "content": self.memory.remember(key, value), "memory_changed": True}

    def _t_recall(self, query: str = "") -> dict[str, Any]:
        facts = self.memory.recall(query)
        if not facts:
            return {"ok": True, "content": "Nothing matching in memory."}
        return {"ok": True, "content": "\n".join(f"{k}: {v}" for k, v in facts.items())}

    def _t_forget(self, key: str) -> dict[str, Any]:
        return {"ok": True, "content": self.memory.forget(key), "memory_changed": True}

    def _t_learn_skill(self, name: str, description: str, code: str) -> dict[str, Any]:
        return {"ok": True, "content": self.memory.learn_skill(name, description, code),
                "memory_changed": True}

    def _t_list_skills(self) -> dict[str, Any]:
        skills = self.memory.list_skills()
        if not skills:
            return {"ok": True, "content": "No skills learned yet."}
        return {"ok": True, "content": "\n".join(f"{s['id']}: {s['description']}" for s in skills)}

    def _t_run_skill(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self.memory.skill_path(name)
        if not path or not path.exists():
            return {"ok": False, "content": f"No skill named '{name}'."}
        # run the skill in a subprocess so a bad skill can't take down the agent
        args = args or {}
        runner = (
            "import json, runpy, sys\n"
            f"ns = runpy.run_path({str(path)!r})\n"
            "fn = ns.get('run')\n"
            "args = json.loads(sys.argv[1])\n"
            "print(fn(**args) if fn else 'skill executed (no run() function)')\n"
        )
        import json as _json

        proc = subprocess.run([sys.executable, "-c", runner, _json.dumps(args)],
                              cwd=str(self.workdir), capture_output=True, text=True, timeout=180)
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return {"ok": proc.returncode == 0, "content": _clip(out.strip() or "(no output)")}
