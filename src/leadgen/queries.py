"""Search queries that surface people asking to hire social media / ads help."""

from __future__ import annotations

# Buyer-side phrasing only. Sellers say "I help brands..." - buyers say "looking for...".
INTENT_PHRASES: list[str] = [
    "looking for a social media manager",
    "need a social media manager",
    "hiring a social media manager",
    "recommend a social media manager",
    "looking for a social media agency",
    "recommendations for a social media agency",
    "looking for someone to manage our social media",
    "need help with our social media",
    "looking for an ads manager",
    "looking for a facebook ads expert",
    "need someone to run facebook ads",
    "looking for a meta ads specialist",
    "looking for a google ads specialist",
    "looking for a ppc agency",
    "need help with instagram marketing",
    "looking for a tiktok ads",
    "looking for a digital marketing agency",
    "can anyone recommend a marketing agency",
]

PLATFORM_FILTERS: dict[str, str] = {
    "linkedin": "site:linkedin.com/posts",
    "instagram": "site:instagram.com",
    "facebook": "site:facebook.com",
    "x": "(site:x.com OR site:twitter.com)",
    "threads": "site:threads.net",
    "reddit": "site:reddit.com",
    "craigslist": "site:craigslist.org",
    "web": "",
}

PHRASES_PER_QUERY = 3


def intent_queries(platforms: list[str], phrases: list[str] | None = None) -> list[tuple[str, str]]:
    """Return (query, platform) pairs, OR-ing a few quoted phrases per query to stay short."""
    phrases = phrases or INTENT_PHRASES
    out: list[tuple[str, str]] = []
    for platform in platforms:
        if platform not in PLATFORM_FILTERS:
            raise ValueError(f"unknown platform {platform!r}; choose from {sorted(PLATFORM_FILTERS)}")
        site = PLATFORM_FILTERS[platform]
        for i in range(0, len(phrases), PHRASES_PER_QUERY):
            chunk = phrases[i : i + PHRASES_PER_QUERY]
            ored = " OR ".join(f'"{p}"' for p in chunk)
            out.append(((f"{site} ({ored})" if site else f"({ored})").strip(), platform))
    return out
