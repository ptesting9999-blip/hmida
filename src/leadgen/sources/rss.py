"""Generic RSS / Atom feeds (job boards, community forums, saved searches)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Iterable, Optional

from bs4 import BeautifulSoup

from leadgen.config import FeedConfig
from leadgen.dates import Window, parse_date
from leadgen.http import HttpClient
from leadgen.models import RawPost
from leadgen.sources.base import NicheTarget, clean_text

log = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_DC = "{http://purl.org/dc/elements/1.1/}"


def _text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


def _html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ") if "<" in html else html


def parse_feed(xml_text: str, platform: str, feed_url: str, window: Window) -> list[RawPost]:
    root = ET.fromstring(xml_text)
    posts: list[RawPost] = []
    for item in root.iter("item"):  # RSS 2.0
        date_raw = _text(item.find("pubDate")) or _text(item.find(f"{_DC}date"))
        posts.append(
            RawPost(
                source="rss",
                platform=platform,
                url=_text(item.find("link")),
                title=clean_text(_text(item.find("title")), 500),
                text=clean_text(_html_to_text(_text(item.find("description")))),
                author=_text(item.find(f"{_DC}creator")) or _text(item.find("author")) or None,
                posted_at=parse_date(date_raw, window.now),
                date_source="feed",
                query=feed_url,
            )
        )
    for entry in root.iter(f"{_ATOM}entry"):  # Atom
        link_el = entry.find(f"{_ATOM}link[@rel='alternate']")
        if link_el is None:
            link_el = entry.find(f"{_ATOM}link")
        body = _text(entry.find(f"{_ATOM}content")) or _text(entry.find(f"{_ATOM}summary"))
        date_raw = _text(entry.find(f"{_ATOM}published")) or _text(entry.find(f"{_ATOM}updated"))
        posts.append(
            RawPost(
                source="rss",
                platform=platform,
                url=link_el.get("href", "") if link_el is not None else "",
                title=clean_text(_text(entry.find(f"{_ATOM}title")), 500),
                text=clean_text(_html_to_text(body)),
                author=_text(entry.find(f"{_ATOM}author/{_ATOM}name")) or None,
                posted_at=parse_date(date_raw, window.now),
                date_source="feed",
                query=feed_url,
            )
        )
    return [p for p in posts if p.url]


class RssSource:
    name = "rss"

    def __init__(self, feeds: list[FeedConfig], http: HttpClient) -> None:
        self.feeds = feeds
        self.http = http

    def collect(self, window: Window, niches: list[NicheTarget]) -> Iterable[RawPost]:
        for feed in self.feeds:
            page = self.http.fetch_page(feed.url)
            if page is None or page.status >= 400 or not page.text:
                log.warning("feed unavailable: %s", feed.url)
                continue
            try:
                posts = parse_feed(page.text, feed.platform, feed.url, window)
            except ET.ParseError as exc:
                log.warning("feed %s is not valid XML: %s", feed.url, exc)
                continue
            log.info("rss: %d items from %s", len(posts), feed.url)
            yield from posts
