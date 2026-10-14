"""Polite HTTP client: one session, identifying user agent, robots.txt, per-host throttling."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib import robotparser
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)

# Business pages are small; anything bigger is a file or a trap. Read at most this much.
MAX_PAGE_BYTES = 2_000_000
_TEXTUAL = ("html", "xml", "text", "rss", "atom")


@dataclass
class Page:
    url: str  # final URL after redirects
    status: int
    text: str
    content_type: str


class HttpClient:
    def __init__(
        self,
        user_agent: str,
        timeout: float = 15.0,
        respect_robots_txt: bool = True,
        per_host_delay: float = 1.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.respect_robots_txt = respect_robots_txt
        self.per_host_delay = per_host_delay
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", user_agent)
        self._robots: dict[str, Optional[robotparser.RobotFileParser]] = {}
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()

    def _throttle(self, host: str) -> None:
        if self.per_host_delay <= 0:
            return
        with self._lock:
            last = self._last_hit.get(host, 0.0)
            wait = last + self.per_host_delay - time.monotonic()
            self._last_hit[host] = time.monotonic() + max(wait, 0.0)
        if wait > 0:
            time.sleep(wait)

    def allowed(self, url: str) -> bool:
        if not self.respect_robots_txt:
            return True
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        with self._lock:
            cached = self._robots.get(base, False)
        if cached is False:
            rp: Optional[robotparser.RobotFileParser] = robotparser.RobotFileParser()
            try:
                resp = self.session.get(f"{base}/robots.txt", timeout=self.timeout)
                if resp.status_code >= 500 or resp.status_code == 429:
                    rp.parse(["User-agent: *", "Disallow: /"])  # RFC 9309: unreachable means disallow
                elif resp.status_code >= 400:
                    rp = None  # no robots.txt: everything allowed
                else:
                    rp.parse(resp.text.splitlines())
            except requests.RequestException:
                rp.parse(["User-agent: *", "Disallow: /"])
            with self._lock:
                self._robots[base] = rp
            cached = rp
        return cached is None or cached.can_fetch(self.user_agent, url)

    def fetch_page(self, url: str) -> Optional[Page]:
        """GET an HTML page for crawling. Returns None when disallowed or unreachable."""
        if not self.allowed(url):
            log.debug("robots.txt disallows %s", url)
            return None
        self._throttle(urlsplit(url).netloc)
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True, stream=True)
        except requests.RequestException as exc:
            log.debug("fetch failed %s: %s", url, exc)
            return None
        try:
            ctype = resp.headers.get("Content-Type", "")
            if not any(t in ctype.lower() for t in _TEXTUAL):
                return Page(url=resp.url, status=resp.status_code, text="", content_type=ctype)
            chunks: list[bytes] = []
            size = 0
            for chunk in resp.iter_content(64 * 1024):
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_PAGE_BYTES:
                    break
            raw = b"".join(chunks)[:MAX_PAGE_BYTES]
        except requests.RequestException as exc:
            log.debug("read failed %s: %s", url, exc)
            return None
        finally:
            resp.close()
        # requests guesses ISO-8859-1 for text/* without a charset; the web is UTF-8.
        encoding = resp.encoding if "charset=" in ctype.lower() and resp.encoding else "utf-8"
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:  # unknown charset label such as "utf8mb4"
            text = raw.decode("utf-8", errors="replace")
        return Page(url=resp.url, status=resp.status_code, text=text, content_type=ctype)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        """GET a JSON API. Raises requests.HTTPError on non-2xx."""
        self._throttle(urlsplit(url).netloc)
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def post_json(self, url: str, **kwargs: Any) -> Any:
        self._throttle(urlsplit(url).netloc)
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.post(url, **kwargs)
        resp.raise_for_status()
        return resp.json()
