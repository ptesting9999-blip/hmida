"""Generic RSS / Atom feeds (job boards, community forums, saved searches)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Iterable, Optional

from bs4 import BeautifulSoup

from leadgen.config import FeedConfig
from leadgen.dates import Window, date_bounds, datetime_from_post_url, parse_date
from leadgen.http import HttpClient
from leadgen.models import RawPost
from leadgen.sources.base import NicheTarget, clean_text

log = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_DC = "{http://purl.org/dc/elements/1.1/}"
_RSS1 = "{http://purl.org/rss/1.0/}"
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def _text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


def _html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ") if "<" in html else html


def _post(platform: str, url: str, title: str, body: str, author: str, date_raw: str, feed_url: str, window: Window) -> RawPost:
    # The timestamp inside a LinkedIn/X post ID beats any date the feed claims.
    exact = datetime_from_post_url(url)
    earliest, latest = (None, None) if exact else (date_bounds(date_raw, window.now) or (None, None))
    return RawPost(
        source="rss",
        platform=platform,
        url=url,
        title=clean_text(title, 500),
        text=clean_text(_html_to_text(body)),
        author=author or None,
        posted_at=exact or parse_date(date_raw, window.now),
        date_source="post_id" if exact else "feed",
        posted_earliest=earliest,
        posted_latest=latest,
        query=feed_url,
    )


def parse_feed(xml_text: str, platform: str, feed_url: str, window: Window) -> list[RawPost]:
    root = ET.fromstring(xml_text)
    posts: list[RawPost] = []
    for item in root.iter("item"):  # RSS 2.0
        link = _text(item.find("link"))
        guid = item.find("guid")
        if not link and guid is not None and guid.get("isPermaLink", "true").lower() == "true":
            link = _text(guid)
        date_raw = _text(item.find("pubDate")) or _text(item.find(f"{_DC}date"))
        author = _text(item.find(f"{_DC}creator")) or _text(item.find("author"))
        posts.append(_post(platform, link, _text(item.find("title")), _text(item.find("description")), author, date_raw, feed_url, window))
    for item in root.iter(f"{_RSS1}item"):  # RSS 1.0 / RDF
        link = _text(item.find(f"{_RSS1}link")) or item.get(f"{_RDF}about", "")
        author = _text(item.find(f"{_DC}creator"))
        posts.append(
            _post(platform, link, _text(item.find(f"{_RSS1}title")), _text(item.find(f"{_RSS1}description")),
                  author, _text(item.find(f"{_DC}date")), feed_url, window)
        )
    for entry in root.iter(f"{_ATOM}entry"):  # Atom
        link_el = entry.find(f"{_ATOM}link[@rel='alternate']")
        if link_el is None:
            link_el = entry.find(f"{_ATOM}link")
        body = _text(entry.find(f"{_ATOM}content")) or _text(entry.find(f"{_ATOM}summary"))
        date_raw = _text(entry.find(f"{_ATOM}published")) or _text(entry.find(f"{_ATOM}updated"))
        url = link_el.get("href", "") if link_el is not None else ""
        author = _text(entry.find(f"{_ATOM}author/{_ATOM}name"))
        posts.append(_post(platform, url, _text(entry.find(f"{_ATOM}title")), body, author, date_raw, feed_url, window))
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
