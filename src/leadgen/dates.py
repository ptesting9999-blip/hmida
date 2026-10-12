"""Date handling: the recency window, parsing loose date strings, and decoding post IDs.

LinkedIn and X post IDs embed their creation time, which gives an exact posted date even
when the only thing we have is a search-engine result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import unquote

UTC = timezone.utc

# X/Twitter snowflake epoch (2010-11-04T01:42:54.657Z), in milliseconds.
X_EPOCH_MS = 1288834974657
# Earliest plausible timestamp for a decoded LinkedIn ID; anything older is a misparse.
_LINKEDIN_MIN = datetime(2014, 1, 1, tzinfo=UTC)

_LINKEDIN_ID_RE = re.compile(r"(?:activity|ugcPost|share)[-:](\d{18,20})", re.IGNORECASE)
_X_ID_RE = re.compile(r"(?:twitter\.com|x\.com)/[^/]+/status(?:es)?/(\d{10,20})", re.IGNORECASE)


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class Window:
    start: datetime  # oldest allowed post time
    end: datetime  # newest allowed post time
    now: datetime

    @classmethod
    def from_days(cls, max_age_days: int, min_age_days: int = 0, now: Optional[datetime] = None) -> "Window":
        now = ensure_utc(now or utcnow())
        return cls(start=now - timedelta(days=max_age_days), end=now - timedelta(days=min_age_days), now=now)

    def contains(self, dt: Optional[datetime]) -> bool:
        if dt is None:
            return False
        dt = ensure_utc(dt)
        return self.start <= dt <= self.end

    def describe(self) -> str:
        return f"{self.start.date().isoformat()} to {self.end.date().isoformat()}"


def linkedin_post_datetime(url: str) -> Optional[datetime]:
    """Decode the creation time from a LinkedIn activity/ugcPost/share ID.

    The first 41 bits of the 64-bit ID are a Unix timestamp in milliseconds.
    """
    m = _LINKEDIN_ID_RE.search(unquote(url))
    if not m:
        return None
    post_id = int(m.group(1))
    ms = post_id >> 22
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if dt < _LINKEDIN_MIN or dt > utcnow() + timedelta(days=1):
        return None
    return dt


def x_post_datetime(url: str) -> Optional[datetime]:
    """Decode the creation time from an X/Twitter status ID (snowflake)."""
    m = _X_ID_RE.search(url)
    if not m:
        return None
    status_id = int(m.group(1))
    if status_id < 2**32:  # pre-snowflake IDs (2010 and earlier) carry no timestamp
        return None
    ms = (status_id >> 22) + X_EPOCH_MS
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if dt > utcnow() + timedelta(days=1):
        return None
    return dt


def datetime_from_post_url(url: str) -> Optional[datetime]:
    return linkedin_post_datetime(url) or x_post_datetime(url)


_UNIT_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 7 * 86400,
    "month": 30 * 86400,
    "year": 365 * 86400,
}
_UNIT_ALIASES = {
    "s": "second", "sec": "second", "secs": "second", "second": "second", "seconds": "second",
    "m": "minute", "min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute",
    "h": "hour", "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
    "d": "day", "day": "day", "days": "day",
    "w": "week", "wk": "week", "wks": "week", "week": "week", "weeks": "week",
    "mo": "month", "mos": "month", "month": "month", "months": "month",
    "y": "year", "yr": "year", "yrs": "year", "year": "year", "years": "year",
}
_RELATIVE_RE = re.compile(
    r"^\s*(\d+|an?|one)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?|[smhdwy])\s*(?:ago)?\.?\s*$",
    re.IGNORECASE,
)
_ABSOLUTE_FORMATS = (
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%Y/%m/%d",
)


def parse_relative(text: str, now: datetime) -> Optional[datetime]:
    t = text.strip().lower()
    if t in ("just now", "now", "today"):
        return now
    if t == "yesterday":
        return now - timedelta(days=1)
    m = _RELATIVE_RE.match(t)
    if not m:
        return None
    qty_raw, unit_raw = m.group(1), m.group(2).lower()
    qty = 1 if qty_raw in ("a", "an", "one") else int(qty_raw)
    unit = _UNIT_ALIASES.get(unit_raw)
    if unit is None:
        return None
    return now - timedelta(seconds=qty * _UNIT_SECONDS[unit])


def parse_absolute(text: str) -> Optional[datetime]:
    t = text.strip()
    if not t:
        return None
    try:
        return ensure_utc(datetime.fromisoformat(t.replace("Z", "+00:00")))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(t)
        if dt is not None:
            return ensure_utc(dt)
    except (TypeError, ValueError, IndexError):
        pass
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t).replace("Sept ", "Sep ")
    for fmt in _ABSOLUTE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def parse_date(text: Optional[str], now: Optional[datetime] = None) -> Optional[datetime]:
    """Parse ISO, RFC 2822, "Oct 5, 2026", "3 days ago", "2d", "yesterday" and similar."""
    if not text:
        return None
    now = ensure_utc(now or utcnow())
    return parse_relative(text, now) or parse_absolute(text)
