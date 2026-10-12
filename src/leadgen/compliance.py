"""Suppression list: people and companies who must never be contacted (opt-outs, clients, competitors)."""

from __future__ import annotations

from pathlib import Path


class Suppression:
    """Lines are emails (jane@acme.com) or whole domains (acme.com or @acme.com); # for comments."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.emails: set[str] = set()
        self.domains: set[str] = set()
        if self.path and self.path.is_file():
            for line in self.path.read_text().splitlines():
                self._add(line)

    def _add(self, entry: str) -> None:
        entry = entry.split("#", 1)[0].strip().lower()
        if not entry:
            return
        if entry.startswith("@"):
            self.domains.add(entry[1:])
        elif "@" in entry:
            self.emails.add(entry)
        else:
            self.domains.add(entry)

    def add(self, entry: str) -> None:
        """Add an entry and persist it to the suppression file."""
        self._add(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(entry.strip() + "\n")

    def blocks(self, email_or_domain: str) -> bool:
        value = email_or_domain.strip().lower()
        if value in self.emails:
            return True
        domain = value.rpartition("@")[2]
        parts = domain.split(".")
        return any(".".join(parts[i:]) in self.domains for i in range(len(parts) - 1))
