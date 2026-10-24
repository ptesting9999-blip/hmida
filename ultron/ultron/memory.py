"""Persistent memory and self-taught skills for ULTRON.

Memory is a plain JSON file so it is easy to inspect and edit by hand.
Skills are small Python scripts ULTRON writes for itself; each one lives in
its own file under ``skills/`` and is described in ``skills.json`` so ULTRON
can recall what it already knows how to do.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


class Memory:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.skills_dir = self.root / "skills"
        self.skills_dir.mkdir(exist_ok=True)
        self.mem_file = self.root / "memory.json"
        self.skills_file = self.root / "skills.json"
        self._mem = self._load(self.mem_file, {"facts": {}, "log": []})
        self._skills = self._load(self.skills_file, {})

    # ----- low level -------------------------------------------------------
    @staticmethod
    def _load(path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text("utf-8"))
            except Exception:
                pass
        return default

    def _save(self, path: Path, data: Any) -> None:
        # write to a temp file then replace, so a crash never corrupts memory
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(path)

    # ----- facts -----------------------------------------------------------
    def remember(self, key: str, value: str) -> str:
        key = key.strip()
        self._mem["facts"][key] = value
        self._mem["log"].append({"t": time.time(), "key": key, "value": value})
        self._mem["log"] = self._mem["log"][-500:]  # keep the log bounded
        self._save(self.mem_file, self._mem)
        return f"Remembered: {key} = {value}"

    def forget(self, key: str) -> str:
        if key in self._mem["facts"]:
            del self._mem["facts"][key]
            self._save(self.mem_file, self._mem)
            return f"Forgot: {key}"
        return f"Nothing stored under '{key}'."

    def recall(self, query: str = "") -> dict[str, str]:
        facts = self._mem["facts"]
        if not query:
            return dict(facts)
        q = query.lower()
        return {k: v for k, v in facts.items() if q in k.lower() or q in str(v).lower()}

    def all_facts(self) -> dict[str, str]:
        return dict(self._mem["facts"])

    # ----- skills ----------------------------------------------------------
    @staticmethod
    def _safe_name(name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
        return name or "skill"

    def learn_skill(self, name: str, description: str, code: str) -> str:
        """Save a Python skill. The code may define a ``run(**kwargs)`` function;
        if it does, ``run_skill`` will call it, otherwise the module is executed
        top to bottom."""
        safe = self._safe_name(name)
        path = self.skills_dir / f"{safe}.py"
        header = f'"""{description}\n\nLearned by ULTRON on {time.ctime()}.\n"""\n\n'
        path.write_text(header + code, "utf-8")
        self._skills[safe] = {
            "name": name,
            "description": description,
            "file": str(path),
            "learned_at": time.time(),
        }
        self._save(self.skills_file, self._skills)
        return f"Learned skill '{safe}'. Call it with run_skill('{safe}', ...)."

    def skill_path(self, name: str) -> Path | None:
        safe = self._safe_name(name)
        if safe in self._skills:
            return Path(self._skills[safe]["file"])
        return None

    def list_skills(self) -> list[dict[str, str]]:
        return [
            {"name": s["name"], "id": sid, "description": s["description"]}
            for sid, s in self._skills.items()
        ]

    def forget_skill(self, name: str) -> str:
        safe = self._safe_name(name)
        if safe in self._skills:
            p = Path(self._skills[safe]["file"])
            p.unlink(missing_ok=True)
            del self._skills[safe]
            self._save(self.skills_file, self._skills)
            return f"Removed skill '{safe}'."
        return f"No skill named '{safe}'."

    # ----- prompt context --------------------------------------------------
    def context_block(self) -> str:
        """A compact summary of what ULTRON knows, to inject into the system prompt."""
        facts = self.all_facts()
        skills = self.list_skills()
        parts = []
        if facts:
            lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
            parts.append(f"Things you remember about the user and their world:\n{lines}")
        if skills:
            lines = "\n".join(f"- {s['id']}: {s['description']}" for s in skills)
            parts.append(f"Skills you have already taught yourself (call with run_skill):\n{lines}")
        return "\n\n".join(parts) if parts else "You have no stored memories or skills yet."
