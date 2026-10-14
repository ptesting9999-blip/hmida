"""Records passed between pipeline stages."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

_HONORIFICS = {"dr", "mr", "mrs", "ms", "miss", "mx", "prof", "coach", "sir", "dame", "chef", "rev"}
_SUFFIXES = {
    "jr", "sr", "ii", "iii", "iv", "mba", "md", "dds", "dmd", "dpt", "phd", "cpa", "rn", "np", "pa-c", "esq",
    "dc", "do", "od", "lmt", "cpt", "pmp", "msc", "bsc", "ma", "ba",
}


def _ascii_letters(word: str) -> str:
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode().lower())


DateSource = Literal["platform_api", "post_id", "search_index", "feed", "import"]
VerificationStatus = Literal["deliverable", "undeliverable", "risky", "unknown"]


@dataclass
class RawPost:
    """A post / listing that may express intent, as collected from a source."""

    source: str  # e.g. "search:brave", "reddit", "rss", "import"
    platform: str  # linkedin, instagram, facebook, x, reddit, craigslist, web, ...
    url: str
    title: str = ""
    text: str = ""
    author: Optional[str] = None
    author_url: Optional[str] = None
    posted_at: Optional[datetime] = None
    date_source: Optional[DateSource] = None
    # For loose dates ("2w", "Oct 5"): the earliest/latest instant the post could be from.
    posted_earliest: Optional[datetime] = None
    posted_latest: Optional[datetime] = None
    query: Optional[str] = None
    # Set for posts found through a niche growth-signal query rather than an intent query.
    niche: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return hashlib.sha256(normalize_url(self.url).encode()).hexdigest()

    def content(self) -> str:
        parts = [self.title.strip(), self.text.strip()]
        return "\n".join(p for p in parts if p)


def normalize_url(url: str) -> str:
    url = url.strip()
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix) :]
    url = url.removeprefix("www.").removeprefix("m.")
    url = url.split("#", 1)[0]
    # Tracking parameters make the same post look new; drop the whole query string except
    # for hosts where it identifies the content.
    if "?" in url and not url.startswith(("facebook.com", "youtube.com")):
        url = url.split("?", 1)[0]
    return url.rstrip("/").lower()


@dataclass
class Person:
    name: Optional[str]
    role: Optional[str]
    email: Optional[str] = None
    source: str = ""  # post, website, jsonld, hunter, llm
    evidence: str = ""

    @property
    def name_parts(self) -> list[str]:
        """Name words without honorifics, credentials, suffixes, pronouns or emoji:
        'Dr. Amy Chen, DDS' -> ['Amy', 'Chen'], 'Mark Smith Jr.' -> ['Mark', 'Smith']."""
        if not self.name:
            return []
        name = re.sub(r"\([^)]*\)", " ", self.name).split(",", 1)[0]
        words = [w.strip(".'’") for w in name.split()]
        words = [w for w in words if w and _ascii_letters(w)]
        while words and words[0].lower().rstrip(".") in _HONORIFICS:
            words.pop(0)
        while words and words[-1].lower().replace(".", "") in _SUFFIXES:
            words.pop()
        return words

    @property
    def first_last(self) -> tuple[str, str] | None:
        parts = self.name_parts
        if len(parts) < 2:
            return None
        return parts[0], parts[-1]


@dataclass
class EmailCandidate:
    email: str
    person: Optional[Person]
    source: str  # post, website, jsonld, hunter, pattern
    priority: float  # higher is tried first


@dataclass
class VerificationResult:
    email: str
    status: VerificationStatus
    reason: str
    checks: dict[str, str] = field(default_factory=dict)
    checked_at: Optional[datetime] = None
    # True when the answer was "try again later" (greylisting); worth re-checking in a few minutes.
    retryable: bool = False
    # True when no answer came back because of OUR side (DNS down, port 25 unreachable, verifier
    # API failing): the address was not judged, so the post should be tried again on a later run.
    transient: bool = False

    @property
    def deliverable(self) -> bool:
        return self.status == "deliverable"


@dataclass
class Lead:
    email: str
    contact_name: str
    contact_role: str
    business_name: str
    business_website: str
    platform: str
    intent_type: str
    intent_url: str
    posted_at: datetime
    date_source: str
    services_needed: str
    niche: str
    location: str
    intent_score: int
    summary: str
    post_excerpt: str
    email_source: str
    verification: str
    found_at: datetime

    def as_row(self) -> dict[str, Any]:
        return {
            "verified_email": self.email,
            "contact_name": self.contact_name,
            "contact_role": self.contact_role,
            "business_name": self.business_name,
            "business_website": self.business_website,
            "platform": self.platform,
            "intent_type": self.intent_type,
            "posted_date": self.posted_at.date().isoformat(),
            "posted_at": self.posted_at.isoformat(),
            "date_source": self.date_source,
            "intent_url": self.intent_url,
            "services_needed": self.services_needed,
            "niche": self.niche,
            "location": self.location,
            "intent_score": self.intent_score,
            "summary": self.summary,
            "post_excerpt": self.post_excerpt,
            "email_source": self.email_source,
            "verification": self.verification,
            "found_at": self.found_at.isoformat(),
        }


@dataclass
class Rejection:
    url: str
    platform: str
    stage: str
    reason: str
    business: str = ""
