"""Import posts exported from other tools (CSV or JSONL).

Columns / keys: url (required), posted_at (required - posts without a date are dropped),
platform, title, text, author, business_name, website.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from leadgen.dates import Window, date_bounds, datetime_from_post_url, parse_date
from leadgen.models import RawPost
from leadgen.sources.base import NicheTarget, clean_text, platform_for_url

log = logging.getLogger(__name__)


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    # utf-8-sig: Excel's "CSV UTF-8" export starts with a byte-order mark.
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        yield from (data if isinstance(data, list) else [data])
    else:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            yield from csv.DictReader(fh)


class ImportSource:
    name = "import"

    def __init__(self, paths: list[str]) -> None:
        self.paths = [Path(p) for p in paths]

    def collect(self, window: Window, niches: list[NicheTarget]) -> Iterable[RawPost]:
        for path in self.paths:
            if not path.is_file():
                log.warning("import file not found: %s", path)
                continue
            count = 0
            for row in _rows(path):
                url = (row.get("url") or "").strip()
                if not url:
                    continue
                count += 1
                date_raw = str(row.get("posted_at") or "")
                exact = datetime_from_post_url(url)  # beats whatever date the export claims
                earliest, latest = (None, None) if exact else (date_bounds(date_raw, window.now) or (None, None))
                yield RawPost(
                    source=self.name,
                    platform=(row.get("platform") or platform_for_url(url)).lower(),
                    url=url,
                    title=clean_text(row.get("title") or "", 500),
                    text=clean_text(row.get("text") or ""),
                    author=row.get("author") or None,
                    posted_at=exact or parse_date(date_raw, window.now),
                    date_source="post_id" if exact else "import",
                    posted_earliest=earliest,
                    posted_latest=latest,
                    query=str(path),
                    extra={k: row[k] for k in ("business_name", "website") if row.get(k)},
                )
            log.info("import: %d rows from %s", count, path)
