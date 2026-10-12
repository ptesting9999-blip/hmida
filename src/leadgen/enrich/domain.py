"""Finding the business's own website domain (where owner mailboxes live)."""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests

from leadgen.sources.base import host_of

log = logging.getLogger(__name__)

# Hosts that are never a business's own email domain.
NON_BUSINESS_DOMAINS = frozenset(
    """
    linkedin.com instagram.com facebook.com fb.com fb.me x.com twitter.com tiktok.com youtube.com youtu.be
    reddit.com craigslist.org yelp.com google.com goo.gl g.page linktr.ee linkin.bio beacons.ai bit.ly
    wa.me whatsapp.com threads.net threads.com pinterest.com tripadvisor.com yellowpages.com bbb.org
    glassdoor.com indeed.com upwork.com fiverr.com crunchbase.com wikipedia.org medium.com substack.com
    eventbrite.com mindbodyonline.com classpass.com booksy.com vagaro.com apple.com amazon.com etsy.com
    shopify.com wix.com squarespace.com godaddy.com wordpress.com blogspot.com tumblr.com zoominfo.com
    apollo.io rocketreach.co mapquest.com foursquare.com nextdoor.com patch.com prnewswire.com
    businesswire.com globenewswire.com
    """.split()
)
# Free site-builder subdomains (acme.wixsite.com): no mailboxes on these.
NON_BUSINESS_SUFFIXES = (
    ".wixsite.com", ".square.site", ".business.site", ".myshopify.com", ".godaddysites.com",
    ".wordpress.com", ".blogspot.com", ".carrd.co", ".webflow.io", ".netlify.app", ".vercel.app",
    ".github.io", ".weebly.com", ".jimdosite.com", ".site123.me", ".mystrikingly.com",
)
_DOMAIN_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$")
_STOPWORDS = {
    "the", "and", "of", "co", "llc", "inc", "ltd", "limited", "company", "corp", "group", "studio",
    "studios", "official", "shop", "store", "club", "llp", "plc", "gmbh", "&",
}


def is_business_domain(domain: str) -> bool:
    if domain in NON_BUSINESS_DOMAINS or any(domain.endswith("." + d) for d in NON_BUSINESS_DOMAINS):
        return False
    return not domain.endswith(NON_BUSINESS_SUFFIXES)


def normalize_domain(website: Optional[str]) -> Optional[str]:
    """'https://www.AcmeGym.com/about' -> 'acmegym.com'. None for social/directory/builder hosts."""
    if not website:
        return None
    website = website.strip().strip("<>()[]\"'").rstrip(".,;")
    if "@" in website and "/" not in website:
        website = website.rpartition("@")[2]
    domain = host_of(website)
    if not domain or not _DOMAIN_RE.match(domain):
        return None
    return domain if is_business_domain(domain) else None


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) >= 3]


def name_matches(business_name: str, domain: str, title: str = "") -> bool:
    """Does this domain plausibly belong to the named business?"""
    tokens = _tokens(business_name)
    if not tokens:
        return False
    label = domain.split(".")[0].replace("-", "")
    compact = "".join(tokens)
    if compact in label or (len(label) >= 5 and label in compact):
        return True
    in_label = sum(1 for t in tokens if t in label)
    if in_label >= max(1, (len(tokens) + 1) // 2):
        return True
    title_tokens = set(_tokens(title))
    return bool(title_tokens) and all(t in title_tokens for t in tokens) and in_label >= 1


def resolve_domain(business_name: str, location: Optional[str], backend: object) -> Optional[str]:
    """Search the web for the business and return its own domain, if one clearly matches."""
    query = f'"{business_name}"' + (f" {location}" if location else "")
    try:
        hits = backend.search(query, None, 8)  # type: ignore[attr-defined]
    except requests.RequestException as exc:
        log.debug("domain lookup failed for %s: %s", business_name, exc)
        return None
    for hit in hits:
        domain = normalize_domain(hit.url)
        if domain and name_matches(business_name, domain, hit.title):
            return domain
    return None
