"""Reddit: exact post timestamps from the API. Uses OAuth when app credentials are configured."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

import requests

from leadgen.config import RedditSourceConfig
from leadgen.dates import UTC, Window
from leadgen.http import HttpClient
from leadgen.models import RawPost
from leadgen.sources.base import NicheTarget, clean_text

log = logging.getLogger(__name__)

# Subreddits where posts are tagged [Hiring] / [For Hire]; only [Hiring] posts are buyers.
TAGGED_SUBS = {"forhire", "jobbit", "freelance_forhire", "hiring", "slavelabour", "designjobs"}
_HIRING_RE = re.compile(r"\[\s*hiring\s*\]|\bhiring\b", re.IGNORECASE)
_FOR_HIRE_RE = re.compile(r"\[\s*for\s*hire\s*\]|\bfor hire\b", re.IGNORECASE)
# Unauthenticated Reddit allows roughly 10 requests per minute.
_ANON_DELAY_SECONDS = 6.5
# Pages of up to 100 results fetched per subreddit while looking back through the window.
MAX_PAGES = 10


def is_hiring_post(title: str, flair: Optional[str]) -> bool:
    label = f"{flair or ''} {title}"
    return bool(_HIRING_RE.search(label)) and not _FOR_HIRE_RE.search(label)


class RedditSource:
    name = "reddit"

    def __init__(
        self,
        cfg: RedditSourceConfig,
        http: HttpClient,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        anon_delay: float = _ANON_DELAY_SECONDS,
    ) -> None:
        self.cfg = cfg
        self.anon_delay = anon_delay
        self.http = http
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: Optional[str] = None

    def _auth_headers(self) -> dict[str, str]:
        if not (self.client_id and self.client_secret):
            return {}
        if self._token is None:
            resp = self.http.session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                timeout=self.http.timeout,
            )
            resp.raise_for_status()
            self._token = resp.json()["access_token"]
        return {"Authorization": f"bearer {self._token}"}

    def _search(self, subreddit: str, query: str, window: Window) -> list[dict[str, Any]]:
        """Newest-first search results, paging back until posts are older than the window."""
        headers = self._auth_headers()
        base = "https://oauth.reddit.com" if headers else "https://www.reddit.com"
        suffix = "" if headers else ".json"
        span = window.now - window.start
        t = "week" if span <= timedelta(days=7) else "month" if span <= timedelta(days=31) else "year"
        items: list[dict[str, Any]] = []
        after: Optional[str] = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "q": query,
                "restrict_sr": 1,
                "sort": "new",
                "t": t,
                "limit": self.cfg.max_results_per_subreddit,
                "raw_json": 1,
            }
            if after:
                params["after"] = after
            data = self.http.get_json(f"{base}/r/{subreddit}/search{suffix}", params=params, headers=headers)
            if not headers:
                time.sleep(self.anon_delay)
            listing = data.get("data") or {}
            page = [c.get("data", {}) for c in listing.get("children", [])]
            items += page
            after = listing.get("after")
            oldest = min((float(d.get("created_utc") or 0) for d in page), default=0.0)
            if not after or not page or oldest < window.start.timestamp():
                break
        return items

    def collect(self, window: Window, niches: list[NicheTarget]) -> Iterable[RawPost]:
        jobs: list[tuple[str, Optional[str]]] = [(s, None) for s in self.cfg.subreddits]
        for niche in niches:
            jobs += [(s, niche.name) for s in niche.subreddits]
        seen_subs: set[str] = set()
        for subreddit, niche in jobs:
            subreddit = subreddit.removeprefix("r/").strip()
            if not subreddit or subreddit.lower() in seen_subs:
                continue
            seen_subs.add(subreddit.lower())
            try:
                items = self._search(subreddit, self.cfg.query, window)
            except (requests.RequestException, ValueError, KeyError) as exc:
                log.warning("reddit r/%s failed: %s", subreddit, exc)
                continue
            log.info("reddit: %d results in r/%s", len(items), subreddit)
            for d in items:
                post = self.to_post(d, subreddit, niche)
                if post is not None and window.contains(post.posted_at):
                    yield post

    def to_post(self, d: dict[str, Any], subreddit: str, niche: Optional[str]) -> Optional[RawPost]:
        created = d.get("created_utc")
        permalink = d.get("permalink")
        if created is None or not permalink:
            return None
        title = d.get("title", "")
        flair = d.get("link_flair_text")
        if subreddit.lower() in TAGGED_SUBS and not is_hiring_post(title, flair):
            return None
        author = d.get("author")
        return RawPost(
            source=self.name,
            platform="reddit",
            url=f"https://www.reddit.com{permalink}",
            title=clean_text(title, 500),
            text=clean_text(d.get("selftext", "")),
            author=f"u/{author}" if author and author != "[deleted]" else None,
            author_url=f"https://www.reddit.com/user/{author}" if author and author != "[deleted]" else None,
            posted_at=datetime.fromtimestamp(float(created), tz=UTC),
            date_source="platform_api",
            query=self.cfg.query,
            niche=niche,
            extra={"subreddit": subreddit, "flair": flair},
        )
