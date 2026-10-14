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
    seen_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0
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
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    verified_at TEXT,
    exported_at TEXT
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
        self._migrate()
        self._lock = threading.Lock()

    def _migrate(self) -> None:
        """Bring databases created by older versions up to the current schema."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(leads)")}
        if "status" not in cols:
            self._conn.execute("ALTER TABLE leads ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "verified_at" not in cols:
            self._conn.execute("ALTER TABLE leads ADD COLUMN verified_at TEXT")
            self._conn.execute("UPDATE leads SET verified_at = created_at")
        if "exported_at" not in cols:
            # Leads from before the outbox existed were already written out by their run.
            self._conn.execute("ALTER TABLE leads ADD COLUMN exported_at TEXT")
            self._conn.execute("UPDATE leads SET exported_at = created_at")
        seen_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(seen_posts)")}
        if "attempts" not in seen_cols:
            self._conn.execute("ALTER TABLE seen_posts ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            self._conn.commit()
            return rows

    # -- posts -------------------------------------------------------------------------

    def is_seen(self, key: str, max_attempts: int = 3) -> bool:
        """Processed for good. Posts parked for retry count as seen once they used up their attempts."""
        rows = self._exec("SELECT outcome, attempts FROM seen_posts WHERE key = ?", (key,))
        if not rows:
            return False
        outcome, attempts = rows[0]
        return outcome != "retry" or attempts >= max_attempts

    def mark_seen(self, key: str, url: str, outcome: str) -> None:
        self._exec(
            "INSERT OR REPLACE INTO seen_posts(key, url, outcome, seen_at, attempts) VALUES (?, ?, ?, ?, 0)",
            (key, url, outcome, utcnow().isoformat()),
        )

    def mark_retry(self, key: str, url: str) -> int:
        """A temporary failure on our side: try this post again on a later run. Returns attempts so far."""
        self._exec(
            "INSERT INTO seen_posts(key, url, outcome, seen_at, attempts) VALUES (?, ?, 'retry', ?, 1) "
            "ON CONFLICT(key) DO UPDATE SET attempts = attempts + 1, seen_at = excluded.seen_at, outcome = 'retry'",
            (key, url, utcnow().isoformat()),
        )
        return int(self._exec("SELECT attempts FROM seen_posts WHERE key = ?", (key,))[0][0])

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
        now = utcnow().isoformat()
        self._exec(
            "INSERT OR IGNORE INTO leads(email, domain, intent_url, row, created_at, status, verified_at) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?)",
            (lead.email.lower(), domain.lower(), lead.intent_url, json.dumps(lead.as_row()), now, now),
        )

    def unexported_leads(self) -> list[dict[str, Any]]:
        """The outbox: saved leads not yet written to a CSV/JSON/webhook (e.g. after a crash)."""
        rows = self._exec("SELECT row FROM leads WHERE exported_at IS NULL AND status = 'active' ORDER BY created_at")
        return [json.loads(r[0]) for r in rows]

    def mark_exported(self, emails: list[str]) -> None:
        now = utcnow().isoformat()
        for email in emails:
            self._exec("UPDATE leads SET exported_at = ? WHERE email = ?", (now, email.lower()))

    def all_leads(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        """Delivered leads, oldest first. By default only those whose email is still verified."""
        where = "" if include_inactive else "WHERE status = 'active'"
        rows = self._exec(f"SELECT row, status, verified_at FROM leads {where} ORDER BY created_at")
        return [{**json.loads(r[0]), "status": r[1], "last_verified_at": r[2] or ""} for r in rows]

    def leads_due_for_reverification(self, older_than_days: int) -> list[str]:
        """Emails of active leads last verified before the cutoff, plus every unconfirmed one."""
        cutoff = (utcnow() - timedelta(days=older_than_days)).isoformat()
        rows = self._exec(
            "SELECT email FROM leads WHERE status = 'unconfirmed' "
            "OR (status = 'active' AND (verified_at IS NULL OR verified_at <= ?)) ORDER BY created_at",
            (cutoff,),
        )
        return [r[0] for r in rows]

    def set_lead_status(self, email: str, status: str, verification: str, verified: bool) -> None:
        """status: active | unconfirmed | invalid. verified=True stamps verified_at with now."""
        rows = self._exec("SELECT row FROM leads WHERE email = ?", (email.lower(),))
        if not rows:
            return
        row = json.loads(rows[0][0])
        row["verification"] = verification
        if verified:
            self._exec(
                "UPDATE leads SET status = ?, row = ?, verified_at = ? WHERE email = ?",
                (status, json.dumps(row), utcnow().isoformat(), email.lower()),
            )
        else:
            self._exec("UPDATE leads SET status = ?, row = ? WHERE email = ?", (status, json.dumps(row), email.lower()))

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
