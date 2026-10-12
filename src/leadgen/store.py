"""SQLite state: posts already processed, cached verifications, delivered leads, niches."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from leadgen.dates import ensure_utc, utcnow
from leadgen.models import Lead, VerificationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_posts (
    key TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    outcome TEXT NOT NULL,
    seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
    email TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    checks TEXT NOT NULL,
    checked_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leads (
    email TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    intent_url TEXT NOT NULL,
    row TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS leads_domain ON leads(domain);
CREATE TABLE IF NOT EXISTS niches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            self._conn.commit()
            return rows

    # -- posts -------------------------------------------------------------------------

    def is_seen(self, key: str) -> bool:
        return bool(self._exec("SELECT 1 FROM seen_posts WHERE key = ?", (key,)))

    def mark_seen(self, key: str, url: str, outcome: str) -> None:
        self._exec(
            "INSERT OR REPLACE INTO seen_posts(key, url, outcome, seen_at) VALUES (?, ?, ?, ?)",
            (key, url, outcome, utcnow().isoformat()),
        )

    # -- verifications -----------------------------------------------------------------

    def cached_verification(self, email: str, max_age_days: int) -> Optional[VerificationResult]:
        if max_age_days <= 0:
            return None
        rows = self._exec(
            "SELECT status, reason, checks, checked_at FROM verifications WHERE email = ?", (email.lower(),)
        )
        if not rows:
            return None
        status, reason, checks, checked_at = rows[0]
        when = ensure_utc(datetime.fromisoformat(checked_at))
        if utcnow() - when > timedelta(days=max_age_days):
            return None
        return VerificationResult(email=email, status=status, reason=reason, checks=json.loads(checks), checked_at=when)

    def save_verification(self, result: VerificationResult) -> None:
        self._exec(
            "INSERT OR REPLACE INTO verifications(email, status, reason, checks, checked_at) VALUES (?, ?, ?, ?, ?)",
            (
                result.email.lower(),
                result.status,
                result.reason,
                json.dumps(result.checks, sort_keys=True),
                (result.checked_at or utcnow()).isoformat(),
            ),
        )

    # -- leads -------------------------------------------------------------------------

    def lead_exists(self, email: str) -> bool:
        return bool(self._exec("SELECT 1 FROM leads WHERE email = ?", (email.lower(),)))

    def domain_delivered_since(self, domain: str, days: int) -> bool:
        since = (utcnow() - timedelta(days=days)).isoformat()
        return bool(self._exec("SELECT 1 FROM leads WHERE domain = ? AND created_at >= ?", (domain.lower(), since)))

    def save_lead(self, lead: Lead, domain: str) -> None:
        self._exec(
            "INSERT OR IGNORE INTO leads(email, domain, intent_url, row, created_at) VALUES (?, ?, ?, ?, ?)",
            (lead.email.lower(), domain.lower(), lead.intent_url, json.dumps(lead.as_row()), utcnow().isoformat()),
        )

    def all_leads(self) -> list[dict[str, Any]]:
        return [json.loads(r[0]) for r in self._exec("SELECT row FROM leads ORDER BY created_at")]

    # -- niches ------------------------------------------------------------------------

    def latest_niches(self, max_age_days: int) -> Optional[list[dict[str, Any]]]:
        rows = self._exec("SELECT payload, created_at FROM niches ORDER BY id DESC LIMIT 1")
        if not rows:
            return None
        payload, created_at = rows[0]
        if utcnow() - ensure_utc(datetime.fromisoformat(created_at)) > timedelta(days=max_age_days):
            return None
        return json.loads(payload)

    def save_niches(self, niches: list[dict[str, Any]]) -> None:
        self._exec(
            "INSERT INTO niches(payload, created_at) VALUES (?, ?)", (json.dumps(niches), utcnow().isoformat())
        )
