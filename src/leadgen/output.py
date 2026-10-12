"""CSV / JSON exports of verified leads (and, for auditing, the reasons posts were rejected)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from leadgen.config import OutputConfig
from leadgen.models import Lead, Rejection

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


class OutputWriter:
    def __init__(self, cfg: OutputConfig) -> None:
        self.cfg = cfg
        self.dir = Path(cfg.dir)

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
        return files
