"""Search-engine discovery of public posts on LinkedIn, Instagram, Facebook, X and the open web.

The agent never logs in to a social network. It finds public posts that search engines have
indexed, restricted to the recency window, and reads the indexed title/snippet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable, Optional, Protocol

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from leadgen.config import SearchSourceConfig
from leadgen.dates import Window, date_bounds, datetime_from_post_url, parse_date
from leadgen.http import HttpClient
from leadgen.llm import LLM, LLMTransient, LLMUnavailable
from leadgen.models import RawPost, normalize_url
from leadgen.queries import intent_queries
from leadgen.sources.base import NicheTarget, author_from_linkedin_url, author_from_title, clean_text, platform_for_url

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    date_text: Optional[str] = None
    extra: dict[str, str] = field(default_factory=dict)


class SearchBackend(Protocol):
    name: str
    # How many intent phrases to OR into one query (see leadgen.queries).
    phrases_per_query: int

    def search(self, query: str, window: Optional[Window], limit: int, niche: Optional[str] = None) -> list[SearchHit]: ...


def _strip_html(text: str) -> str:
    if "<" in text:
        text = BeautifulSoup(text, "html.parser").get_text()
    return " ".join(text.split())


class BraveSearch:
    """Brave Search API. Supports an exact date range through ``freshness=YYYY-MM-DDtoYYYY-MM-DD``."""

    name = "brave"
    phrases_per_query = 3
    URL = "https://api.search.brave.com/res/v1/web/search"
    PAGE_SIZE = 20

    def __init__(self, api_key: str, http: HttpClient) -> None:
        self.api_key = api_key
        self.http = http

    def search(self, query: str, window: Optional[Window], limit: int, niche: Optional[str] = None) -> list[SearchHit]:
        hits: list[SearchHit] = []
        offset = 0  # Brave's offset is a page index, so the page size must stay fixed
        while len(hits) < limit and offset < 10:
            params = {"q": query, "count": self.PAGE_SIZE, "offset": offset}
            if window is not None:
                params["freshness"] = f"{window.start.date().isoformat()}to{window.end.date().isoformat()}"
            data = self.http.get_json(
                self.URL,
                params=params,
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            )
            results = (data.get("web") or {}).get("results") or []
            for r in results:
                snippet = " ".join([r.get("description", "")] + list(r.get("extra_snippets") or []))
                hits.append(
                    SearchHit(
                        url=r.get("url", ""),
                        title=_strip_html(r.get("title", "")),
                        snippet=_strip_html(snippet),
                        date_text=r.get("page_age") or r.get("age"),
                    )
                )
            more = ((data.get("query") or {}).get("more_results_available"))
            if len(results) < self.PAGE_SIZE or more is False:
                break
            offset += 1
        return hits[:limit]


class SerperSearch:
    """Serper.dev (Google results). Date range via Google's ``tbs=cdr:1,cd_min:..,cd_max:..``."""

    name = "serper"
    phrases_per_query = 3
    URL = "https://google.serper.dev/search"
    PAGE_SIZE = 10

    def __init__(self, api_key: str, http: HttpClient) -> None:
        self.api_key = api_key
        self.http = http

    def search(self, query: str, window: Optional[Window], limit: int, niche: Optional[str] = None) -> list[SearchHit]:
        hits: list[SearchHit] = []
        page = 1
        max_pages = min(10, -(-limit // self.PAGE_SIZE))
        while len(hits) < limit and page <= max_pages:
            body: dict[str, object] = {"q": query, "num": self.PAGE_SIZE, "page": page}
            if window is not None:
                s, e = window.start, window.end
                body["tbs"] = f"cdr:1,cd_min:{s.month}/{s.day}/{s.year},cd_max:{e.month}/{e.day}/{e.year}"
            data = self.http.post_json(self.URL, json=body, headers={"X-API-KEY": self.api_key})
            organic = data.get("organic") or []
            for r in organic:
                hits.append(
                    SearchHit(
                        url=r.get("link", ""),
                        title=r.get("title", ""),
                        snippet=r.get("snippet", ""),
                        date_text=r.get("date"),
                    )
                )
            if len(organic) < self.PAGE_SIZE:
                break
            page += 1
        return hits[:limit]


class SearxngSearch:
    """Your own SearXNG instance (https://docs.searxng.org): free metasearch with a JSON API.

    The instance must have `json` in search.formats (docker-compose.free.yml sets that up).
    """

    name = "searxng"
    phrases_per_query = 3
    MAX_PAGES = 3

    def __init__(self, base_url: str, http: HttpClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http

    def search(self, query: str, window: Optional[Window], limit: int, niche: Optional[str] = None) -> list[SearchHit]:
        hits: list[SearchHit] = []
        page = 1
        while len(hits) < limit and page <= self.MAX_PAGES:
            params: dict[str, object] = {"q": query, "format": "json", "pageno": page}
            if window is not None:
                span = window.now - window.start
                params["time_range"] = "day" if span <= timedelta(days=1) else "month" if span <= timedelta(days=31) else "year"
            data = self.http.get_json(f"{self.base_url}/search", params=params)
            results = data.get("results") or []
            for r in results:
                hits.append(
                    SearchHit(
                        url=r.get("url", ""),
                        title=_strip_html(r.get("title", "")),
                        snippet=_strip_html(r.get("content", "") or ""),
                        date_text=r.get("publishedDate") or None,
                    )
                )
            if not results:
                break
            page += 1
        return hits[:limit]


class _FoundPost(BaseModel):
    url: str = Field(description="Exact URL of the post, copied from the search results")
    author: Optional[str] = Field(description="Name of the person or business that wrote the post, if shown")
    quote: str = Field(description="Verbatim text of the post as shown in the search results")


class _FoundPosts(BaseModel):
    posts: list[_FoundPost]


class ClaudeWebSearch:
    """Uses Claude's server-side web search tool. Needs only an Anthropic API key.

    Claude can read result snippets we cannot, so it reports each post's text; the URL and
    date come from the raw search results, never from the model's own claims.
    """

    name = "claude"
    # Claude plans its own searches, so hand it every phrase for a platform in one research
    # call instead of one call per 3 phrases: about 6x fewer calls per run.
    phrases_per_query = 100

    def __init__(self, llm: LLM, max_searches: int = 5) -> None:
        self.llm = llm
        self.max_searches = max_searches

    def search(self, query: str, window: Optional[Window], limit: int, niche: Optional[str] = None) -> list[SearchHit]:
        if window is None:
            user = f"Search the web for: {query}\nThen list the most relevant result URLs."
        elif niche:
            user = (
                f"Search the web for this (split it into several shorter searches as you see fit): {query}\n\n"
                f"Find individual public posts published between {window.describe()} by a specific business in "
                f"the niche '{niche}' announcing an opening, launch, expansion, rebrand or funding round. For each "
                "post report its exact URL, the business or author name, and a verbatim quote of the post text. "
                f"Report at most {limit} posts. Ignore articles, listicles and agencies advertising themselves."
            )
        else:
            user = (
                f"Search the web for this (split it into several shorter searches as you see fit): {query}\n\n"
                f"Find individual public posts published between {window.describe()} in which the author "
                "is asking to hire, or asking for recommendations for, help with social media or paid ads. "
                "For each relevant post report its exact URL, the author's name, and a verbatim quote of the "
                f"post text. Report at most {limit} posts. Ignore posts by agencies or freelancers advertising "
                "their own services."
            )
        system = "You are a research assistant that finds and reports web search results accurately."
        result = self.llm.research(system=system, user=user, effort="low", max_searches=self.max_searches)
        if result is None:
            return []
        by_url = {normalize_url(h.url): h for h in result.hits}
        if window is None:
            return [SearchHit(url=h.url, title=h.title, snippet="", date_text=h.page_age) for h in result.hits][:limit]
        try:
            found = self.llm.parse(
                _FoundPosts,
                system="Extract the posts listed in this research report. Copy URLs exactly.",
                user=result.text,
                effort="low",
            )
        except LLMTransient as exc:
            log.warning("could not structure Claude search results right now: %s", exc)
            return []
        hits: list[SearchHit] = []
        for p in found.posts if found else []:
            hit = by_url.get(normalize_url(p.url))
            if hit is None:  # the model reported a URL that no search returned - discard
                continue
            # Post text = what the search server itself quoted from the page. The model's own
            # quote/author are kept only as unverified hints, so nothing it wrote (e.g. an email
            # address) can pass as part of the real post.
            hits.append(
                SearchHit(
                    url=hit.url,
                    title=hit.title,
                    snippet=" ".join(hit.cited),
                    date_text=hit.page_age,
                    extra={"reported_quote": p.quote, "reported_author": p.author or ""},
                )
            )
        return hits[:limit]


# Only URLs that point at a single post / listing, not profiles, hashtags or search pages.
_POST_URL_RES = {
    "linkedin": re.compile(r"linkedin\.com/(posts/|feed/update/)", re.I),
    "instagram": re.compile(r"instagram\.com/(?:[A-Za-z0-9._]+/)?(?:p|reels?|tv)/(?!audio/)[A-Za-z0-9_-]+", re.I),
    "facebook": re.compile(
        r"(?:facebook|fb)\.com/(?:[^/?]+/(?:posts|videos|photos)/[^/?]+|groups/[^/]+/(?:posts|permalink)/\d+|"
        r"share/[pvr]/|reel/\d+|photo(?:\.php)?/?\?fbid=|permalink\.php|story\.php)",
        re.I,
    ),
    "x": re.compile(r"(x|twitter)\.com/(?:i/web|[^/]+)/status/", re.I),
    "threads": re.compile(r"threads\.(net|com)/@[^/]+/post/", re.I),
    "reddit": re.compile(r"reddit\.com/r/[^/]+/comments/", re.I),
}


def is_post_url(platform: str, url: str) -> bool:
    pattern = _POST_URL_RES.get(platform)
    return True if pattern is None else bool(pattern.search(url))


class SearchSource:
    def __init__(self, cfg: SearchSourceConfig, backend: SearchBackend) -> None:
        self.cfg = cfg
        self.backend = backend
        self.name = f"search:{backend.name}"

    def queries(self, niches: list[NicheTarget]) -> list[tuple[str, Optional[str]]]:
        """(query, niche name or None for explicit-intent queries)."""
        per_query = getattr(self.backend, "phrases_per_query", 3)
        out: list[tuple[str, Optional[str]]] = [(q, None) for q, _ in intent_queries(self.cfg.platforms, per_query=per_query)]
        out += [(q, None) for q in self.cfg.extra_queries]
        for niche in niches:
            out += [(q, niche.name) for q in niche.search_queries]
        return out

    def collect(self, window: Window, niches: list[NicheTarget]) -> Iterable[RawPost]:
        for query, niche in self.queries(niches):
            try:
                hits = self.backend.search(query, window, self.cfg.results_per_query, niche=niche)
            except (requests.RequestException, ValueError) as exc:
                log.warning("%s query failed (%s): %s", self.name, exc, query)
                continue
            except LLMUnavailable:
                raise
            log.info("%s: %d results for %s", self.name, len(hits), query)
            for hit in hits:
                post = self.to_post(hit, query, niche, window)
                if post is not None:
                    yield post

    def to_post(self, hit: SearchHit, query: str, niche: Optional[str], window: Window) -> Optional[RawPost]:
        if not hit.url:
            return None
        platform = platform_for_url(hit.url)
        if not is_post_url(platform, hit.url):
            return None
        posted_at = datetime_from_post_url(hit.url)
        date_source = "post_id" if posted_at else None
        earliest = latest = None
        if posted_at is None:
            posted_at = parse_date(hit.date_text, window.now)
            date_source = "search_index" if posted_at else None
            earliest, latest = date_bounds(hit.date_text, window.now) or (None, None)
        author = author_from_title(hit.title)
        if author is None and platform == "linkedin":
            author = author_from_linkedin_url(hit.url)
        return RawPost(
            source=self.name,
            platform=platform,
            url=hit.url,
            title=clean_text(hit.title, 500),
            text=clean_text(hit.snippet),
            author=author,
            posted_at=posted_at,
            date_source=date_source,
            posted_earliest=earliest,
            posted_latest=latest,
            query=query,
            niche=niche,
            extra=dict(hit.extra),
        )
