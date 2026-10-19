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
# Industry words shared by thousands of businesses: "Acme Fitness" must not match anytimefitness.com.
_GENERIC_WORDS = {
    "fitness", "fit", "gym", "gyms", "yoga", "pilates", "crossfit", "training", "spa", "salon", "beauty",
    "nails", "hair", "barber", "barbershop", "cafe", "coffee", "bakery", "restaurant", "kitchen", "grill",
    "pizza", "bistro", "bar", "clinic", "dental", "dentistry", "med", "medical", "health", "wellness",
    "boutique", "center", "centre", "services", "solutions", "agency", "consulting", "marketing", "media",
    "design", "realty", "homes", "law", "legal", "auto", "motors", "cleaning", "construction", "room",
}
# Service subdomains that rarely carry mail: shop.acme.com -> acme.com.
_SERVICE_LABELS = {"shop", "store", "book", "booking", "bookings", "app", "my", "members", "portal", "blog", "go", "order", "menu"}


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
    labels = domain.split(".")
    if len(labels) >= 3 and labels[0] in _SERVICE_LABELS:
        domain = ".".join(labels[1:])
    return domain if is_business_domain(domain) else None


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) >= 3]


def name_matches(business_name: str, domain: str, title: str = "") -> bool:
    """Does this domain plausibly belong to the named business?

    A distinctive (non-generic) word must be in the domain: "Acme Fitness" matches
    acmefitness.com and acme-fit.co.uk, never anytimefitness.com.
    """
    tokens = _tokens(business_name)
    distinctive = [t for t in tokens if t not in _GENERIC_WORDS]
    if not distinctive:
        return False  # "Fit Club": nothing to tell this business apart from any other
    label = domain.split(".")[0].replace("-", "")
    if not any(t in label for t in distinctive):
        return False  # only generic words in common: a different business
    compact = "".join(tokens)
    if compact in label or (len(label) >= 5 and label in compact):
        return True
    in_label = sum(1 for t in distinctive if t in label)
    if in_label >= max(1, (len(distinctive) + 1) // 2):
        return True
    title_tokens = set(_tokens(title))
    return bool(title_tokens) and all(t in title_tokens for t in tokens)


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
