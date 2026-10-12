"""Crawl a business website (homepage + about/team/contact pages) for people and email addresses."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

from leadgen.http import HttpClient
from leadgen.models import Person

log = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}(?![\w-])")
_OBFUSCATED_RE = re.compile(
    r"([A-Za-z0-9._%+\-]+)\s*(?:\[at\]|\(at\)|\{at\}|\s\[\s*@\s*\]\s)\s*"
    r"([A-Za-z0-9\-]+(?:\s*(?:\[dot\]|\(dot\)|\{dot\}|\.)\s*[A-Za-z0-9\-]+)+)",
    re.IGNORECASE,
)
_JUNK_EMAIL_RE = re.compile(
    r"\.(png|jpe?g|gif|webp|svg|css|js)$|@(example|domain|email|yourdomain|company|sentry)\.|wixpress\.com$|"
    r"sentry\.io$|^(your|name|first|firstname|user|username|email)@",
    re.IGNORECASE,
)
PAGE_KEYWORDS = (
    "about", "team", "founder", "leadership", "management", "people", "staff", "story", "who-we-are",
    "meet", "owner", "contact", "company", "our-",
)


@dataclass
class SitePage:
    url: str
    text: str


@dataclass
class SiteProfile:
    domain: str
    base_url: Optional[str] = None
    pages: list[SitePage] = field(default_factory=list)
    emails: set[str] = field(default_factory=set)
    people: list[Person] = field(default_factory=list)

    @property
    def reachable(self) -> bool:
        return self.base_url is not None


def decode_cfemail(encoded: str) -> Optional[str]:
    """Decode a Cloudflare-protected email (data-cfemail / #hex in /cdn-cgi/l/email-protection)."""
    try:
        key = int(encoded[:2], 16)
        return "".join(chr(int(encoded[i : i + 2], 16) ^ key) for i in range(2, len(encoded), 2))
    except ValueError:
        return None


def _clean_email(email: str) -> Optional[str]:
    email = unquote(email).strip().strip(".,;:()<>[]\"'").lower()
    email = email.split("?", 1)[0]
    if not EMAIL_RE.fullmatch(email) or _JUNK_EMAIL_RE.search(email):
        return None
    return email


def extract_emails(soup: BeautifulSoup, text: str) -> set[str]:
    found: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().startswith("mailto:"):
            for addr in href[7:].split(","):
                if (e := _clean_email(addr)) is not None:
                    found.add(e)
        elif "/cdn-cgi/l/email-protection#" in href:
            decoded = decode_cfemail(href.rsplit("#", 1)[1])
            if decoded and (e := _clean_email(decoded)) is not None:
                found.add(e)
    for el in soup.select("[data-cfemail]"):
        decoded = decode_cfemail(el.get("data-cfemail", ""))
        if decoded and (e := _clean_email(decoded)) is not None:
            found.add(e)
    for m in EMAIL_RE.finditer(text):
        if (e := _clean_email(m.group(0))) is not None:
            found.add(e)
    for m in _OBFUSCATED_RE.finditer(text):
        domain = re.sub(r"\s*(?:\[dot\]|\(dot\)|\{dot\}|\.)\s*", ".", m.group(2), flags=re.IGNORECASE)
        if (e := _clean_email(f"{m.group(1)}@{domain}")) is not None:
            found.add(e)
    return found


def _iter_jsonld(soup: BeautifulSoup) -> Iterable[dict[str, Any]]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                yield item
                stack.extend(v for v in item.values() if isinstance(v, (dict, list)))


def _types(item: dict[str, Any]) -> set[str]:
    t = item.get("@type", [])
    return {x.lower() for x in ([t] if isinstance(t, str) else t if isinstance(t, list) else [])}


def extract_jsonld_people(soup: BeautifulSoup) -> list[Person]:
    people: list[Person] = []
    for item in _iter_jsonld(soup):
        types = _types(item)
        if "person" in types and isinstance(item.get("name"), str):
            people.append(
                Person(
                    name=item["name"].strip(),
                    role=item.get("jobTitle") if isinstance(item.get("jobTitle"), str) else None,
                    email=_clean_email(str(item["email"]).removeprefix("mailto:")) if item.get("email") else None,
                    source="jsonld",
                )
            )
        founders = item.get("founder")
        for f in founders if isinstance(founders, list) else [founders] if founders else []:
            name = f.get("name") if isinstance(f, dict) else f if isinstance(f, str) else None
            if name:
                people.append(Person(name=name.strip(), role="Founder", source="jsonld"))
    # A founder object is also a Person node, so the same name can appear twice.
    merged: dict[str, Person] = {}
    for p in people:
        key = (p.name or "").lower()
        if key in merged:
            old = merged[key]
            merged[key] = Person(name=old.name, role=old.role or p.role, email=old.email or p.email, source="jsonld")
        else:
            merged[key] = p
    return list(merged.values())


def visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg", "template", "iframe"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def candidate_links(soup: BeautifulSoup, base_url: str, domain: str) -> list[str]:
    scored: dict[str, int] = {}
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"]).split("#", 1)[0]
        parts = urlsplit(url)
        host = (parts.hostname or "").removeprefix("www.")
        if parts.scheme not in ("http", "https") or not (host == domain or host.endswith("." + domain)):
            continue
        hay = f"{parts.path} {a.get_text(' ')}".lower()
        score = sum(1 for k in PAGE_KEYWORDS if k in hay)
        if score and re.search(r"\.(pdf|jpe?g|png|zip|mp4)$", parts.path, re.I) is None:
            scored[url] = max(scored.get(url, 0), score)
    return [u for u, _ in sorted(scored.items(), key=lambda kv: -kv[1])]


class WebsiteCrawler:
    def __init__(self, http: HttpClient, max_pages: int = 6) -> None:
        self.http = http
        self.max_pages = max_pages

    def crawl(self, domain: str) -> SiteProfile:
        profile = SiteProfile(domain=domain)
        home = None
        for base in (f"https://{domain}/", f"https://www.{domain}/", f"http://{domain}/"):
            page = self.http.fetch_page(base)
            if page is not None and page.status < 400 and page.text:
                home = page
                break
        if home is None:
            return profile
        profile.base_url = home.url
        queue = [home.url]
        visited: set[str] = set()
        first = True
        while queue and len(visited) < self.max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            page = home if first else self.http.fetch_page(url)
            first = False
            if page is None or page.status >= 400 or not page.text:
                continue
            soup = BeautifulSoup(page.text, "html.parser")
            profile.people += extract_jsonld_people(soup)
            links = candidate_links(soup, page.url, domain) if len(visited) == 1 else []
            raw_text = soup.get_text(" ")
            profile.emails |= extract_emails(soup, raw_text)
            profile.pages.append(SitePage(url=page.url, text=visible_text(soup)))
            queue += [u for u in links if u not in visited]
        log.debug("crawled %s: %d pages, %d emails", domain, len(profile.pages), len(profile.emails))
        return profile
