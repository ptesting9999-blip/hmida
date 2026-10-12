"""CSV / JSON exports of verified leads (and, for auditing, the reasons posts were rejected)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

from leadgen.config import OutputConfig
from leadgen.models import Lead, Rejection

log = logging.getLogger(__name__)

LEAD_COLUMNS = [
    "verified_email",
    "contact_name",
    "contact_role",
    "business_name",
    "business_website",
    "platform",
    "intent_type",
    "posted_date",
    "posted_at",
    "date_source",
    "intent_url",
    "services_needed",
    "niche",
    "location",
    "intent_score",
    "summary",
    "post_excerpt",
    "email_source",
    "verification",
    "found_at",
]
# Exports of stored leads also say whether the email is still verified.
EXPORT_COLUMNS = LEAD_COLUMNS + ["status", "last_verified_at"]


def _safe_cell(value: Any) -> Any:
    """Neutralise spreadsheet formula injection: post text comes from strangers on the internet."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str], append: bool = False) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not (append and exists):
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_cell(v) for k, v in row.items()})


SLACK_LEADS_PER_MESSAGE = 25


def slack_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for i in range(0, len(rows), SLACK_LEADS_PER_MESSAGE):
        chunk = rows[i : i + SLACK_LEADS_PER_MESSAGE]
        lines = [
            f"• *{r['business_name'] or r['business_website']}*: {r['contact_name']} ({r['contact_role'] or 'decision-maker'}) "
            f"<mailto:{r['verified_email']}|{r['verified_email']}>, {r['platform']}, posted {r['posted_date']}, "
            f"<{r['intent_url']}|see post>\n    _{r['summary']}_"
            for r in chunk
        ]
        header = f"*{len(rows)} new verified lead{'s' if len(rows) != 1 else ''}*" if i == 0 else "(continued)"
        messages.append({"text": header + "\n" + "\n".join(lines)})
    return messages


def post_webhook(url: str, fmt: str, rows: list[dict[str, Any]], session: Optional[requests.Session] = None) -> bool:
    """Deliver new leads; failures are logged, never fatal (the CSV is the record)."""
    http = session or requests.Session()
    payloads = slack_messages(rows) if fmt == "slack" else [{"source": "leadgen", "count": len(rows), "leads": rows}]
    try:
        for payload in payloads:
            http.post(url, json=payload, timeout=20).raise_for_status()
    except requests.RequestException as exc:
        log.warning("webhook delivery failed: %s", exc)
        return False
    return True


class OutputWriter:
    def __init__(self, cfg: OutputConfig, webhook_url: Optional[str] = None) -> None:
        self.cfg = cfg
        self.dir = Path(cfg.dir)
        self.webhook_url = webhook_url

    def write(self, leads: list[Lead], rejections: list[Rejection], started_at: datetime) -> list[str]:
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = started_at.strftime("%Y%m%d-%H%M%S")
        rows = [lead.as_row() for lead in leads]
        files: list[str] = []
        if "csv" in self.cfg.formats:
            path = self.dir / f"leads_{stamp}.csv"
            write_csv(path, rows, LEAD_COLUMNS)
            write_csv(self.dir / "leads_all.csv", rows, LEAD_COLUMNS, append=True)
            files.append(str(path))
        if "json" in self.cfg.formats:
            path = self.dir / f"leads_{stamp}.json"
            path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
            files.append(str(path))
        if self.cfg.write_rejected and rejections:
            path = self.dir / f"rejected_{stamp}.csv"
            write_csv(path, (asdict(r) for r in rejections), ["url", "platform", "stage", "reason", "business"])
            files.append(str(path))
        if self.webhook_url and rows and post_webhook(self.webhook_url, self.cfg.webhook_format, rows):
            files.append(f"webhook ({self.cfg.webhook_format}): {len(rows)} leads delivered")
        return files
