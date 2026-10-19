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
_SELLER_STRONG = re.compile(
    r"\b(?:i'?m|i am|we are|we're)\s+an?\s+(?:freelance\s+|certified\s+|experienced\s+)?"
    r"(?:social media|smm|facebook ads?|meta ads?|google ads?|tiktok ads?|paid ads?|ads|ppc|digital marketing|marketing|content|ugc)\s+"
    r"(?:manager|marketer|agency|specialist|expert|consultant|strategist|creator|freelancer|buyer|pro)\b|"
    r"\bour agency\b|\b(?:my|our) (?:portfolio|rates)\b|\bavailable for (?:work|hire|projects)\b",
    re.I,
)
# Phrases sellers use, but buyers too ("we offer botox", "free consultation" at a clinic): they
# count only when a marketing service follows in the same sentence.
_SELLER_WEAK = re.compile(
    r"\b(?:we offer|(?:my|our) services|services include|(?:for|with|help) my clients|free (?:audit|consultation|strategy call)|"
    r"book a (?:free )?call|(?:i|we) (?:can )?help (?:you|businesses|brands|companies)|"
    r"(?:do|does|are) (?:you|your (?:business|brand))\s+(?:need|looking|want|struggling))\b"
    rf"[^.?!\n]{{0,80}}?\b(?:{_SERVICE}|marketing|followers|instagram|ads)\b",
    re.I,
)
# People looking for a marketing job or clients, not for help.
_JOB_SEEKER_RE = re.compile(
    r"\blooking for (?:a |some )?(?:new )?(?:work|job|jobs|employment|opportunit\w*|clients|gigs?)\b|"
    r"\b(?:any(?:one|body)|who(?:'s| is)) (?:here )?hiring\b|\bneed (?:a )?job\b|\bmy (?:resume|cv|portfolio)\b|"
    r"\bopen to (?:work|opportunities|new roles)\b|"
    rf"\b(?:work|job|career|role|position) as an?\s+(?:freelance\s+|remote\s+)?(?:{_SERVICE})\b|"
    rf"\b(?:seeking|looking for)\s+(?:a\s+|an\s+)?(?:remote\s+|part[- ]time\s+|full[- ]time\s+|entry[- ]level\s+|freelance\s+)?"
    rf"(?:{_SERVICE})\s+(?:position|role|job|internship)\b",
    re.I,
)
_GROWTH_RE = re.compile(
    r"\b(?:grand opening|just opened|now open|soft (?:opening|launch)|opening soon|we(?:'| a)re (?:now )?open|newly opened|"
    r"ribbon cutting|opening (?:day|week)|first day (?:open|of business)|we(?:'| a)re live|just launched|"
    r"launching (?:our|my|the|a) |officially open|new location|now accepting new|"
    r"raised (?:\$|£|€)?\d|(?:closed|raised) (?:our |a )?(?:seed|pre-?seed|series [abc])|seed round|pre-?seed round|"
    r"series [ab] (?:round|funding)|funding round)",
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
# Lookbehinds anchor each match at a token start, so long runs of dots/letters stay linear.
_EMAIL_RE = re.compile(r"(?<![\w.%+-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
_URL_RE = re.compile(
    r"(?<![\w.@/-])(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]{0,62}(?:\.[a-z0-9-]{1,62})*\.([a-z]{2,24}))(?![\w@-])(?:/[^\s]*)?",
    re.I,
)
# Real TLDs a small business might use. Two-letter country codes are accepted too, except the
# ones that are really file or language extensions in running text (next.js, setup.py, README.md).
_GTLDS = frozenset(
    """com net org info biz io ai app dev co shop store online site website tech studio fitness salon spa cafe
    restaurant coffee health clinic dental beauty agency marketing media design academy club live life world today
    space xyz pro me tv bar pizza kitchen boutique fashion style yoga gym coach consulting solutions services
    company business email global group digital social network agency events photography photo gallery art
    church school education law legal realty homes properties house rentals travel tours vegas nyc london""".split()
)
_NOT_TLDS = frozenset({"js", "py", "ts", "md", "rb", "sh", "db", "cs"})
_SOCIAL = ("linkedin.com", "instagram.com", "facebook.com", "fb.com", "x.com", "twitter.com", "tiktok.com",
           "reddit.com", "youtube.com", "craigslist.org", "linktr.ee", "threads.net")
_HONORIFIC_RE = re.compile(r"^(?:dr|mr|mrs|ms|miss|mx|prof)\.?\s+", re.I)
_PERSON_RE = re.compile(r"^[A-Z][a-zA-Z'’\-]+(?: [A-Z][a-zA-Z'’\-]+){1,2}$")
_BUSINESS_WORDS = re.compile(
    r"\b(gym|fitness|studio|spa|salon|cafe|coffee|bakery|clinic|dental|agency|shop|store|boutique|co|llc|inc|ltd|"
    r"group|labs?|club|kitchen|restaurant|bar|grill|yoga|pilates|crossfit|barbers?|beauty|wellness|realty|"
    r"homes|motors|media|digital|solutions|services|official|hq|team|company|brand|academy|school|center|centre)\b",
    re.I,
)
# "we just opened Acme Fitness", "owner of Glow Med Spa", "founder at Launchly", "manager needed at Glow Spa".
# [ \t] rather than \s: a name never continues onto the next line.
_BUSINESS_RE = re.compile(
    r"\b(?:opened|launched|launching|owner of|founder of|co-?founder of|ceo of|founder at|run|own|"
    r"(?:needed|wanted|hiring|position|role|job|manager|owner|founder|ceo) at)[ \t]+"
    r"((?:[A-Z][\w&'’]*)(?:[ \t]+(?:[A-Z][\w&'’]*|&|of)){0,4})"
)
# Capitalised words that start sentences or name platforms, not businesses.
_NOT_A_NAME = frozenset(
    """facebook instagram tiktok linkedin google meta youtube twitter x reddit pinterest snapchat shopify etsy amazon
    threads whatsapp monday tuesday wednesday thursday friday saturday sunday today tomorrow tonight january february
    march april may june july august september october november december the a an our my this that we i it its
    new last next""".split()
)
# (pattern, niche). Word starts only: "coworking space" is not a spa, "happy" is not an app.
_NICHE_WORDS = [
    (r"gyms?|fitness|pilates|yoga|crossfit", "fitness"),
    (r"med ?spas?|spas?|salons?|barbers?(?:hop)?|nails?|beauty", "beauty"),
    (r"restaurants?|cafes?|coffee|bakery|bakeries", "food & drink"),
    (r"clinics?|dental|dentists?|chiropract\w*", "healthcare"),
    (r"startups?|saas|apps?", "startup"),
    (r"e-?commerce|shopify", "e-commerce"),
    (r"real estate|realtors?", "real estate"),
    (r"law firms?", "legal"),
    (r"boutiques?", "retail"),
]
_NICHE_RES = [(re.compile(rf"\b(?:{pat})\b", re.I), label) for pat, label in _NICHE_WORDS]


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
        host, tld = m.group(1).lower(), m.group(2).lower()
        if m.end() < len(text) and text[m.end()] == "@":  # the name part of an email address
            continue
        if tld in _NOT_TLDS or (len(tld) > 2 and tld not in _GTLDS):
            continue  # next.js, resume.pdf, "opened.Need"
        if not any(host == s or host.endswith("." + s) for s in _SOCIAL):
            return host
    return None


def _niche(text: str, post: RawPost) -> Optional[str]:
    if post.niche:
        return post.niche
    return next((label for pattern, label in _NICHE_RES if pattern.search(text)), None)


def _business_name(text: str) -> Optional[str]:
    for m in _BUSINESS_RE.finditer(text):
        words = m.group(1).split()
        while words and words[0].lower().strip("'’") in _NOT_A_NAME:
            words.pop(0)
        while words and words[-1] in ("&", "of"):
            words.pop()
        if words:
            return " ".join(words)
    return None


def assess_without_ai(post: RawPost, cfg: Config) -> IntentAssessment:
    text = post.content()
    seller = bool(SELLER_PATTERNS.search(text) or _SELLER_STRONG.search(text) or _SELLER_WEAK.search(text))
    job_seeker = bool(_JOB_SEEKER_RE.search(text))
    request = None if job_seeker else _REQUEST_RE.search(text)
    growth = bool(post.niche and _GROWTH_RE.search(text))

    if request and not seller:
        intent_type = "explicit_request"
        score = 72 + 10 * bool(_URGENT_RE.search(text)) + 8 * bool(_BUDGET_RE.search(text)) + 5 * bool(_website(text))
    elif growth and not seller:
        intent_type, score = "growth_signal", 78
    else:
        intent_type, score = "not_relevant", 20 if _SERVICE_RE.search(text) else 5

    author = (post.author or "").strip()
    bare = _HONORIFIC_RE.sub("", author)
    person = bare if _PERSON_RE.match(bare) and not _BUSINESS_WORDS.search(bare) else None
    owner = _OWNER_RE.search(text)
    role = None
    if owner:
        title = owner.group(1) or "owner"
        role = "CEO" if title.lower() == "ceo" else title.title()
    business = (
        post.extra.get("business_name")
        or _business_name(text)
        or (author if not person and _BUSINESS_WORDS.search(author) and not author.startswith("u/") else None)
    )
    emails = list(dict.fromkeys(e for e in _EMAIL_RE.findall(text)))
    return IntentAssessment(
        intent_type=intent_type,
        author_is_service_provider=seller or job_seeker,
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
