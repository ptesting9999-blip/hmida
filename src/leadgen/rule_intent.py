"""Free intent classification without any AI model: phrase rules over the post text.

Less nuanced than Claude or a local model, but costs nothing and runs anywhere. It produces
the same IntentAssessment, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import re
from typing import Optional

from leadgen.config import Config
from leadgen.intent import SELLER_PATTERNS, IntentAssessment, ServiceKind
from leadgen.models import RawPost

_SERVICE = (
    r"social\s*media(?:\s*(?:manager|management|marketing|marketer|agency|person|help|expert|specialist))?|smm|"
    r"(?:facebook|meta|instagram|tiktok|google|linkedin|youtube|paid)\s*ads?(?:\s*(?:manager|specialist|expert|agency|person|help))?|"
    r"ads?\s*manager|media\s*buyer|ppc(?:\s*agency)?|(?:digital\s*)?marketing\s*agency|digital\s*marketer|"
    r"content\s*creator|ugc\s*creator|instagram\s*(?:marketing|growth|manager)|growth\s*marketer"
)
_ASK = (
    r"looking\s+for|need(?:s|ed|ing)?|hiring|seeking|searching\s+for|in\s+search\s+of|want\s+to\s+hire|"
    r"(?:can\s+)?anyone\s+(?:recommend|know)|recommend(?:ations?)?\s+(?:for|on)|who\s+(?:can|could)\s+(?:help|run|manage)|"
    r"any\s+(?:recommendations|suggestions)|(?:would|could)\s+(?:love|use)\s+(?:some\s+)?help\s+with"
)
# A request: an asking phrase followed (within ~8 words) by a service.
_REQUEST_RE = re.compile(rf"\b(?:{_ASK})\b(?:\W+\w+){{0,8}}?\W+(?:an?\s+|some\s+|our\s+)?(?:{_SERVICE})\b", re.I)
_SERVICE_RE = re.compile(rf"\b(?:{_SERVICE})\b", re.I)
# Pitches. Buyers say "send your portfolio"; sellers say "my portfolio" or "do you need ...?".
_SELLER_EXTRA = re.compile(
    r"\b(i'?m|i am|we are|we're)\s+an?\s+(freelance\s+)?(social media|smm|ads|marketing|ppc|content)|"
    r"\bour agency\b|\bmy clients\b|\bwe offer\b|\bservices include\b|\b(my|our) (portfolio|services|rates)\b|"
    r"\bavailable for (work|hire|projects)\b|\b(do|does|are) (you|your (business|brand))\s+(need|looking|want|struggling)\b|"
    r"\bfree (audit|consultation|strategy call)\b|\bbook a (free )?call\b|\b(i|we) (can )?help (you|businesses|brands)\b",
    re.I,
)
_GROWTH_RE = re.compile(
    r"grand opening|just opened|now open|soft (opening|launch)|opening soon|we('| a)re (now )?open|newly opened|"
    r"ribbon cutting|opening (day|week)|first day|we('| a)re live|just launched|launching|officially open|"
    r"new location|now accepting|raised|seed round|pre-?seed|series a|funding",
    re.I,
)
_URGENT_RE = re.compile(r"\basap\b|urgent|immediately|right away|this week|start (now|asap|immediately)|as soon as", re.I)
_BUDGET_RE = re.compile(r"[$£€]\s?\d|budget|per month|/mo\b|retainer|paid (role|gig|position)|\brate\b", re.I)
# "my friend's gym", "my favorite cafe": someone else's business.
_NOT_MINE = (
    r"(?:friend|sister|brother|mom|mum|dad|mother|father|wife|husband|partner|client|boss|cousin|aunt|uncle|son|"
    r"daughter|neighbou?r|favou?rite|local|old|former)(?:'?s)?\s"
)
_OWNER_RE = re.compile(
    r"\b(?:i'?m|i am|as)\s+(?:the\s+)?(co-?founder|founder|owner|co-?owner|ceo|managing director)\b|"
    r"\bmy\s+(?:(?!" + _NOT_MINE + r")[a-z'-]+\s+){0,2}?(gym|studio|business|shop|store|company|startup|brand|salon|spa|"
    r"restaurant|cafe|clinic|practice|bakery|boutique|agency|firm)\b",  # "my new pilates studio", not "my friend's gym"
    re.I,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
_URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]{1,62}(?:\.[a-z0-9-]{1,62})*\.[a-z]{2,24})(?:/[^\s]*)?", re.I)
_SOCIAL = ("linkedin.com", "instagram.com", "facebook.com", "fb.com", "x.com", "twitter.com", "tiktok.com",
           "reddit.com", "youtube.com", "craigslist.org", "linktr.ee", "threads.net")
_PERSON_RE = re.compile(r"^[A-Z][a-zA-Z'’\-]+(?: [A-Z][a-zA-Z'’\-]+){1,2}$")
_BUSINESS_WORDS = re.compile(
    r"\b(gym|fitness|studio|spa|salon|cafe|coffee|bakery|clinic|dental|agency|shop|store|boutique|co|llc|inc|ltd|"
    r"group|labs?|club|kitchen|restaurant|bar|grill|yoga|pilates|crossfit|barbers?|beauty|wellness|realty|"
    r"homes|motors|media|digital|solutions|services|official|hq|team|company|brand|academy|school|center|centre)\b",
    re.I,
)
# "we just opened Acme Fitness", "owner of Glow Med Spa", "founder at Launchly"
_BUSINESS_RE = re.compile(
    r"\b(?:opened|launched|launching|owner of|founder of|co-?founder of|ceo of|founder at|run|own|at)\s+"
    r"((?:[A-Z][\w&'’]*)(?:\s+(?:[A-Z][\w&'’]*|&|of)){0,4})"
)
_NICHE_WORDS = {
    "gym": "fitness", "fitness": "fitness", "pilates": "fitness", "yoga": "fitness", "crossfit": "fitness",
    "salon": "beauty", "spa": "beauty", "med spa": "beauty", "barber": "beauty", "nail": "beauty",
    "restaurant": "food & drink", "cafe": "food & drink", "coffee": "food & drink", "bakery": "food & drink",
    "clinic": "healthcare", "dental": "healthcare", "dentist": "healthcare", "chiropract": "healthcare",
    "startup": "startup", "saas": "startup", "app": "startup", "ecommerce": "e-commerce", "shopify": "e-commerce",
    "real estate": "real estate", "realtor": "real estate", "law firm": "legal", "boutique": "retail",
}


def _services(text: str) -> list[ServiceKind]:
    t = text.lower()
    out: list[ServiceKind] = []
    if re.search(r"social\s*media|smm|instagram|tiktok", t):
        out.append("social_media_management")
    if re.search(r"\bads?\b|ppc|media\s*buyer|paid", t):
        out.append("paid_ads_management")
    if re.search(r"content|ugc|creator|video|reels", t):
        out.append("content_creation")
    return out or ["general_digital_marketing"]


def _website(text: str) -> Optional[str]:
    for m in _URL_RE.finditer(text):
        start = m.start()
        if start > 0 and text[start - 1] == "@":  # part of an email address
            continue
        host = m.group(1).lower()
        if not any(host == s or host.endswith("." + s) for s in _SOCIAL):
            return host
    return None


def _niche(text: str, post: RawPost) -> Optional[str]:
    if post.niche:
        return post.niche
    t = text.lower()
    return next((label for word, label in _NICHE_WORDS.items() if word in t), None)


def assess_without_ai(post: RawPost, cfg: Config) -> IntentAssessment:
    text = post.content()
    seller = bool(SELLER_PATTERNS.search(text) or _SELLER_EXTRA.search(text))
    request = _REQUEST_RE.search(text)
    growth = bool(post.niche and _GROWTH_RE.search(text))

    if request and not seller:
        intent_type = "explicit_request"
        score = 72 + 10 * bool(_URGENT_RE.search(text)) + 8 * bool(_BUDGET_RE.search(text)) + 5 * bool(_website(text))
    elif growth and not seller:
        intent_type, score = "growth_signal", 78
    else:
        intent_type, score = "not_relevant", 20 if _SERVICE_RE.search(text) else 5

    author = (post.author or "").strip()
    person = author if _PERSON_RE.match(author) and not _BUSINESS_WORDS.search(author) else None
    owner = _OWNER_RE.search(text)
    role = None
    if owner:
        title = owner.group(1) or "owner"
        role = "CEO" if title.lower() == "ceo" else title.title()
    named = _BUSINESS_RE.search(text)
    business = (
        post.extra.get("business_name")
        or (named.group(1).strip() if named else None)
        or (author if author and not person and not author.startswith("u/") else None)
    )
    emails = list(dict.fromkeys(e for e in _EMAIL_RE.findall(text)))
    return IntentAssessment(
        intent_type=intent_type,
        author_is_service_provider=seller,
        intent_score=min(score, 100),
        services_needed=_services(text),
        business_name=business,
        business_website=post.extra.get("website") or _website(text),
        business_location=None,
        niche=_niche(text, post),
        poster_name=person,
        poster_role=role,
        poster_is_decision_maker=bool(owner),
        emails_in_post=emails,
        summary=(request.group(0) if request else text[:160]).strip(),
    )
