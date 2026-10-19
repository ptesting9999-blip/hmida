"""From "a business wants help" to "these are the decision-makers and their likely addresses"."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from leadgen.config import Config
from leadgen.enrich.domain import name_matches, normalize_domain, resolve_domain
from leadgen.enrich.hunter import Hunter
from leadgen.enrich.patterns import ascii_name, email_matches_name, infer_pattern, permutations
from leadgen.enrich.people import (
    MIN_DECISION_WEIGHT,
    is_excluded_title,
    keyword_windows,
    llm_people,
    regex_people,
    title_weight,
)
from leadgen.enrich.website import SiteProfile, WebsiteCrawler
from leadgen.llm import LLM, LLMTransient
from leadgen.models import EmailCandidate, Person
from leadgen.verify.rules import is_free_provider, is_role_account, split_email

log = logging.getLogger(__name__)

# Weight given to a poster whom Claude judged a decision-maker but whose title we can't rank.
POSTER_DECISION_WEIGHT = 0.9


@dataclass
class BusinessContext:
    business_name: Optional[str]
    website: Optional[str]
    location: Optional[str]
    poster: Optional[Person]
    poster_is_decision_maker: bool
    emails_in_post: list[str] = field(default_factory=list)


@dataclass
class DecisionMaker:
    person: Person
    weight: float
    candidates: list[EmailCandidate] = field(default_factory=list)


@dataclass
class ContactPlan:
    domain: Optional[str]
    website_reachable: Optional[bool]
    decision_makers: list[DecisionMaker]
    notes: list[str] = field(default_factory=list)
    # True when a step failed because of an outage (e.g. Claude API error), not a real "no one found".
    transient: bool = False


def _name_key(name: Optional[str]) -> str:
    return " ".join((name or "").lower().split())


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _same_business(domain: str, final: str, business_name: Optional[str]) -> bool:
    """Is a redirect target the same business (acme.com -> acmefitness.com), not a parking page?"""
    if not normalize_domain(final):
        return False
    a, b = _letters(domain.split(".")[0]), _letters(final.split(".")[0])
    contained = (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a)
    return contained or bool(business_name and name_matches(business_name, final))


class ContactFinder:
    def __init__(
        self,
        cfg: Config,
        *,
        crawler: Optional[WebsiteCrawler],
        llm: Optional[LLM],
        hunter: Optional[Hunter],
        lookup_backend: Optional[object],
    ) -> None:
        self.cfg = cfg
        self.crawler = crawler
        self.llm = llm
        self.hunter = hunter
        self.lookup_backend = lookup_backend

    def plan(self, ctx: BusinessContext) -> ContactPlan:
        notes: list[str] = []
        domain = self._domain(ctx, notes)

        site: Optional[SiteProfile] = None
        if domain and self.crawler is not None and self.cfg.enrichment.crawl_website:
            site = self.crawler.crawl(domain)
            if not site.reachable:
                notes.append(f"website {domain} unreachable")
            elif site.final_domain and site.final_domain != domain:
                final = site.final_domain
                if final.endswith("." + domain) or domain.endswith("." + final):
                    pass  # the same site on a subdomain: keep the mail domain
                elif _same_business(domain, final, ctx.business_name):
                    # Mailboxes live on the domain the business actually uses now.
                    notes.append(f"{domain} redirects to {site.final_domain}")
                    domain = site.final_domain
                else:
                    notes.append(f"{domain} redirects to unrelated {site.final_domain}; site ignored")
                    site = None

        people, hunter_pattern, transient = self._gather_people(ctx, domain, site, notes)
        first_names = Counter(ascii_name(p.name_parts[0]) for p, _ in people if p.name_parts)
        shared_first = {n for n, count in first_names.items() if n and count > 1}
        pattern = hunter_pattern or self._pattern(people, site, domain, shared_first)
        site_emails = sorted(site.emails) if site else []

        makers: list[DecisionMaker] = []
        for person, weight in people:
            if weight < MIN_DECISION_WEIGHT:
                continue
            dm = DecisionMaker(person=person, weight=weight)
            dm.candidates = self._candidates(person, weight, domain, pattern, site_emails, ctx, shared_first)
            makers.append(dm)
        if ctx.poster is None and ctx.poster_is_decision_maker:
            # The owner wrote their email in the post but gave no usable name (common on Reddit
            # and forums): that address is still the warmest contact there is.
            emails = [e.strip().lower() for e in ctx.emails_in_post if not is_role_account(e)]
            if emails:
                anon = Person(name=None, role=None, source="post")
                cands = [EmailCandidate(e, anon, "post", 100 + POSTER_DECISION_WEIGHT) for e in dict.fromkeys(emails)]
                makers.append(DecisionMaker(person=anon, weight=POSTER_DECISION_WEIGHT, candidates=cands))
        makers.sort(key=lambda d: -d.weight)
        if not makers:
            notes.append("no owner / founder / decision-maker identified")
        return ContactPlan(
            domain=domain,
            website_reachable=site.reachable if site else None,
            decision_makers=makers[: max(self.cfg.enrichment.max_contacts_per_business * 3, 3)],
            notes=notes,
            transient=transient and not makers,
        )

    # -- steps -------------------------------------------------------------------------

    def _domain(self, ctx: BusinessContext, notes: list[str]) -> Optional[str]:
        domain = normalize_domain(ctx.website)
        if domain is None:
            for email in ctx.emails_in_post:
                d = split_email(email)[1]
                if d and not is_free_provider(d) and normalize_domain(d):
                    domain = d
                    break
        if domain is None and ctx.business_name and self.lookup_backend is not None:
            domain = resolve_domain(ctx.business_name, ctx.location, self.lookup_backend)
            if domain:
                notes.append(f"website found by search: {domain}")
        if domain is None:
            notes.append("no business website found")
        return domain

    def _gather_people(
        self, ctx: BusinessContext, domain: Optional[str], site: Optional[SiteProfile], notes: list[str]
    ) -> tuple[list[tuple[Person, float]], Optional[str], bool]:
        """(people with weights, Hunter's email pattern, whether a step failed temporarily)."""
        merged: dict[str, tuple[Person, float]] = {}
        hunter_pattern: Optional[str] = None
        transient = False

        def add(p: Person, weight: Optional[float] = None) -> None:
            key = _name_key(p.name)
            if not key:
                return
            w = title_weight(p.role) if weight is None else weight
            if key in merged:
                old, old_w = merged[key]
                p = Person(
                    name=old.name or p.name,
                    role=old.role if old_w >= w else p.role,
                    email=old.email or p.email,
                    # Keep the provenance of whichever record supplied the email.
                    source=old.source if (old.email or not p.email) else p.source,
                    evidence=old.evidence or p.evidence,
                )
                w = max(w, old_w)
            merged[key] = (p, w)

        if ctx.poster and ctx.poster.name:
            w = title_weight(ctx.poster.role)
            # Claude's "decision-maker" judgement fills in for titles we can't rank, but never
            # overrides an explicitly excluded one (assistant, intern, coordinator ...).
            if ctx.poster_is_decision_maker and not is_excluded_title(ctx.poster.role):
                w = max(w, POSTER_DECISION_WEIGHT)
            add(ctx.poster, w)

        if site is not None and site.pages:
            for p in site.people:
                add(p)
            text = " ".join(page.text for page in site.pages)
            window = keyword_windows(text, limit=self.cfg.enrichment.max_chars_per_page)
            if self.llm is not None:
                try:
                    found = llm_people(self.llm, ctx.business_name or domain or "", window, self.cfg.llm.extract_effort)
                except LLMTransient as exc:
                    notes.append(f"people extraction failed temporarily ({exc}); used pattern matching")
                    transient = True
                    found = regex_people(window)
            else:
                found = regex_people(window)
            for p in found:
                add(p)

        if domain and self.hunter is not None and self.cfg.enrichment.use_hunter != "never":
            result = self.hunter.domain_search(domain)
            for p in result.people:
                add(p)
            hunter_pattern = result.pattern
            notes.append(f"hunter: {len(result.people)} people")
        return list(merged.values()), hunter_pattern, transient

    def _pattern(
        self, people: list[tuple[Person, float]], site: Optional[SiteProfile], domain: Optional[str], shared_first: set[str]
    ) -> Optional[str]:
        if not domain:
            return None
        known = [(p.email, p) for p, _ in people if p.email]
        known += [(e, p) for e in (site.emails if site else ()) for p, _ in people]
        for email, person in known:
            fl = person.first_last
            if email and _letters(email.split("@")[0]) in shared_first:
                continue  # "dan@" could be either Dan; it proves no pattern
            if fl and email and split_email(email)[1] == domain:
                pattern = infer_pattern(email, *fl)
                if pattern:
                    return pattern
        return None

    def _candidates(
        self,
        person: Person,
        weight: float,
        domain: Optional[str],
        pattern: Optional[str],
        site_emails: list[str],
        ctx: BusinessContext,
        shared_first: set[str] = frozenset(),  # type: ignore[assignment]
    ) -> list[EmailCandidate]:
        out: dict[str, EmailCandidate] = {}

        def add(email: Optional[str], source: str, priority: float) -> None:
            if not email:
                return
            email = email.strip().lower()
            if is_role_account(email):
                return
            if source != "post" and _letters(email.split("@")[0]) in shared_first:
                return  # "dan@" can't be told apart when two people on the team are called Dan
            if email not in out or out[email].priority < priority:
                out[email] = EmailCandidate(email=email, person=person, source=source, priority=priority)

        fl = person.first_last
        parts = person.name_parts
        first = parts[0] if parts else ""
        last = parts[-1] if len(parts) > 1 else ""
        is_poster = ctx.poster is not None and _name_key(ctx.poster.name) == _name_key(person.name)
        if is_poster:
            for email in ctx.emails_in_post:
                add(email, "post", 100 + weight)
        add(person.email, person.source or "website", 90 + weight)
        first_ascii = ascii_name(first)
        for email in site_emails:
            email_domain = split_email(email)[1]
            # A footer address on another company's domain (web designer, parent brand) is not theirs.
            if email_domain != domain and not is_free_provider(email_domain):
                continue
            if not email_matches_name(email, first, last):
                continue
            add(email, "website", 85 + weight)
        if domain and fl:
            limit = self.cfg.enrichment.max_email_candidates
            for i, email in enumerate(permutations(fl[0], fl[1], domain, preferred=pattern, limit=limit)):
                add(email, "pattern" if not pattern else f"pattern {pattern}", 50 + weight - i * 0.5)
        elif domain and first_ascii and not last:
            add(f"{first_ascii}@{domain}", "pattern", 45 + weight)
        return sorted(out.values(), key=lambda c: -c.priority)[: self.cfg.enrichment.max_email_candidates + 4]
