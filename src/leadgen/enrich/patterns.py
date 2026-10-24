"""Company email address patterns ({first}.{last}@domain and friends)."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Ordered by how common they are at small and mid-sized businesses.
PATTERNS = (
    "{first}",
    "{first}.{last}",
    "{first}{last}",
    "{f}{last}",
    "{f}.{last}",
    "{first}_{last}",
    "{first}{l}",
    "{last}",
    "{first}-{last}",
    "{last}.{first}",
    "{last}{f}",
)


def ascii_name(part: str) -> str:
    text = unicodedata.normalize("NFKD", part).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", text.lower())


def render(pattern: str, first: str, last: str) -> Optional[str]:
    first, last = ascii_name(first), ascii_name(last)
    needs_last = "{last}" in pattern or "{l}" in pattern
    if not first or (needs_last and not last):
        return None
    return pattern.format(first=first, last=last, f=first[:1], l=last[:1])


def infer_pattern(email: str, first: str, last: str) -> Optional[str]:
    local = email.split("@", 1)[0].lower()
    for pattern in PATTERNS:
        if render(pattern, first, last) == local:
            return pattern
    return None


def permutations(first: str, last: str, domain: str, preferred: Optional[str] = None, limit: int = 8) -> list[str]:
    order = list(PATTERNS)
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    out: list[str] = []
    for pattern in order:
        local = render(pattern, first, last)
        if local and f"{local}@{domain}" not in out:
            out.append(f"{local}@{domain}")
        if len(out) >= limit:
            break
    return out


def email_matches_name(email: str, first: Optional[str], last: Optional[str]) -> bool:
    """True when an address clearly belongs to this person (sarah@, sarah.jones@, sjones@ ...)."""
    if not first:
        return False
    local = re.sub(r"[^a-z]", "", email.split("@", 1)[0].lower())
    first_n, last_n = ascii_name(first), ascii_name(last or "")
    if not first_n:
        return False
    if last_n and infer_pattern(email, first, last or "") is not None:
        return True
    return local == first_n or (bool(last_n) and local.startswith(first_n) and last_n in local)
