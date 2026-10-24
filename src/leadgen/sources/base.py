"""Common pieces for intent sources."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol
from urllib.parse import urlsplit

from leadgen.dates import Window
from leadgen.models import RawPost

_PLATFORM_HOSTS = {
    "linkedin.com": "linkedin",
    "instagram.com": "instagram",
    "facebook.com": "facebook",
    "fb.com": "facebook",
    "x.com": "x",
    "twitter.com": "x",
    "reddit.com": "reddit",
    "craigslist.org": "craigslist",
    "tiktok.com": "tiktok",
    "threads.net": "threads",
    "threads.com": "threads",
    "upwork.com": "upwork",
    "nextdoor.com": "nextdoor",
}


def host_of(url: str) -> str:
    host = urlsplit(url if "://" in url else f"https://{url}").hostname or ""
    return host.lower().removeprefix("www.")


def platform_for_url(url: str) -> str:
    host = host_of(url)
    for suffix, platform in _PLATFORM_HOSTS.items():
        if host == suffix or host.endswith("." + suffix):
            return platform
    return "web"


@dataclass
class NicheTarget:
    """A niche the agent should watch for growth signals (see leadgen.niches)."""

    name: str
    rationale: str = ""
    search_queries: list[str] = field(default_factory=list)
    subreddits: list[str] = field(default_factory=list)


class Source(Protocol):
    name: str

    def collect(self, window: Window, niches: list[NicheTarget]) -> Iterable[RawPost]: ...


# "Jane Doe on LinkedIn: Looking for ..." / "Acme Gym on Instagram: "..."" / "Jane (@jane) on X"
_TITLE_AUTHOR_RE = re.compile(
    r"^(?P<author>[^:|]{2,80}?)\s+on\s+(?:LinkedIn|Instagram|X|Twitter|Facebook|Threads|TikTok)\b",
    re.IGNORECASE,
)
_HANDLE_RE = re.compile(r"\s*\(@[\w.]+\)\s*$")


def author_from_title(title: str) -> Optional[str]:
    m = _TITLE_AUTHOR_RE.match(title.strip())
    if not m:
        return None
    return _HANDLE_RE.sub("", m.group("author")).strip() or None


def author_from_linkedin_url(url: str) -> Optional[str]:
    """Guess the poster's display name from a linkedin.com/posts/<slug>_... URL.

    The slug is a person's vanity name ("jane-doe-4a1b2c") or a company's ("acme-fitness").
    """
    m = re.search(r"linkedin\.com/posts/([^/_?#]+)_", url)
    if not m:
        return None
    tokens = [t for t in m.group(1).split("-") if t and not any(c.isdigit() for c in t)]
    return " ".join(t.capitalize() for t in tokens) or None


def clean_text(text: str, limit: int = 4000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]
