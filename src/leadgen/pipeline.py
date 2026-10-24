"""The agent loop: collect -> date window -> prefilter -> Claude intent check -> find decision-makers
-> strict email verification -> save."""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional

from leadgen.config import Config
from leadgen.dates import Window, utcnow
from leadgen.enrich import BusinessContext, ContactFinder, ContactPlan
from leadgen.enrich.domain import normalize_domain
from leadgen.intent import IntentAssessment, IntentClassifier, build_user_message, prefilter
from leadgen.llm import LLMUnavailable
from leadgen.models import Lead, Person, RawPost, Rejection
from leadgen.niches import NicheResearcher
from leadgen.output import OutputWriter
from leadgen.sources.base import NicheTarget, Source
from leadgen.store import Store
from leadgen.verify import EmailVerifier

log = logging.getLogger(__name__)

PLATFORM_LABELS = {
    "linkedin": "LinkedIn",
    "instagram": "Instagram",
    "facebook": "Facebook",
    "x": "X (Twitter)",
    "threads": "Threads",
    "tiktok": "TikTok",
    "reddit": "Reddit",
    "craigslist": "Craigslist",
    "upwork": "Upwork",
    "nextdoor": "Nextdoor",
    "web": "Web",
}
# Don't hand over the same business twice within this many days.
BUSINESS_COOLDOWN_DAYS = 60
# A post that failed because of OUR side (Claude/DNS/SMTP/verifier outage) is tried again on later
# runs, at most this many times in total, so a long outage cannot re-bill it forever.
MAX_POST_ATTEMPTS = 3


@dataclass
class RunReport:
    started_at: datetime
    window: Window
    niches: list[NicheTarget] = field(default_factory=list)
    collected: int = 0
    already_seen: int = 0
    out_of_window: int = 0
    prefiltered: int = 0
    classified: int = 0
    qualified: int = 0
    leads: list[Lead] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        by_stage: dict[str, int] = {}
        for r in self.rejections:
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
        lines = [
            f"Window: posts from {self.window.describe()}",
            f"Niches monitored: {', '.join(n.name for n in self.niches) or 'none'}",
            f"Collected {self.collected} posts ({self.already_seen} already processed in earlier runs)",
            f"Outside the date window or undated: {self.out_of_window}",
            f"Dropped by keyword prefilter: {self.prefiltered}",
            f"Assessed for buying intent: {self.classified}; qualified as warm: {self.qualified}",
            f"Verified leads saved: {len(self.leads)}",
        ]
        if by_stage:
            lines.append("Rejections by stage: " + ", ".join(f"{k}={v}" for k, v in sorted(by_stage.items())))
        lines += [f"Wrote {f}" for f in self.files]
        return "\n".join(lines)


def _comparable(text: str) -> str:
    """Words without accents, case or punctuation, so "José O’Neil" and "jose o'neil" match.
    Works for any script (Иван Петров), not only Latin."""
    text = re.sub(r"['’‘`]", "", text)  # O'Neil / O’Neil -> ONeil
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return " " + " ".join(re.findall(r"[^\W_]+", plain)) + " "


def _real_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = name.strip()
    if not name or name.startswith(("u/", "/u/", "@")):
        return None
    return name


@dataclass
class PendingVerification:
    """A qualified business whose best address got "try again later" (greylisting)."""

    assessment: IntentAssessment
    ctx: BusinessContext
    plan: ContactPlan
    deferred_at: float  # time.monotonic()


@dataclass
class PostOutcome:
    post: RawPost
    leads: list[Lead] = field(default_factory=list)
    rejection: Optional[Rejection] = None
    assessment: Optional[IntentAssessment] = None
    retry: Optional[PendingVerification] = None


class LeadAgent:
    def __init__(
        self,
        cfg: Config,
        *,
        sources: list[Source],
        classifier: IntentClassifier,
        finder: ContactFinder,
        verifier: EmailVerifier,
        store: Store,
        writer: OutputWriter,
        niche_researcher: Optional[NicheResearcher] = None,
        preflight: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.preflight = preflight
        self.sources = sources
        self.classifier = classifier
        self.finder = finder
        self.verifier = verifier
        self.store = store
        self.writer = writer
        self.niche_researcher = niche_researcher
        self._save_lock = threading.Lock()
        self._sleep = time.sleep

    # -- run ---------------------------------------------------------------------------

    def run(self, limit: Optional[int] = None, now: Optional[datetime] = None) -> RunReport:
        # Fresh DNS/catch-all knowledge and the latest suppression list for every run.
        self.verifier.reset()
        if self.preflight is not None:
            self.preflight()  # a bad Claude key fails here, before any paid search query runs
        window = Window.from_days(self.cfg.window.max_age_days, self.cfg.window.min_age_days, now=now)
        report = RunReport(started_at=window.now, window=window)
        try:
            report.niches = self.niche_researcher.get() if self.niche_researcher else []
        except LLMUnavailable:
            raise
        except Exception:  # niche research is an extra; explicit-intent search still runs
            log.exception("niche research failed; continuing with explicit-intent queries only")

        candidates = self._screen(self._collect(window, report.niches), window, report, limit)
        log.info("%d posts to assess for buying intent", len(candidates))

        deferred: list[PostOutcome] = []
        pool = ThreadPoolExecutor(max_workers=self.cfg.workers)
        try:
            for outcome in pool.map(self._process_safely, candidates):
                if outcome.retry is not None:
                    deferred.append(outcome)
                else:
                    self._record(outcome, report)
            for outcome in self._retry_greylisted(deferred):
                self._record(outcome, report)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            # Outbox: hand over every saved lead not exported yet - this run's, and any left behind
            # by an earlier run that was killed before it could write its files.
            rows = self.store.unexported_leads()
            keep = self.verifier.suppression.filter_rows(rows)  # opted out since they were saved
            report.files = self.writer.write(keep, report.rejections, report.started_at)
            self.store.mark_exported([r["verified_email"] for r in rows])
            if len(keep) > len(report.leads):
                log.info("delivered %d leads saved by an earlier, interrupted run", len(keep) - len(report.leads))
        return report

    def _record(self, outcome: PostOutcome, report: RunReport) -> None:
        post, rejection = outcome.post, outcome.rejection
        if outcome.assessment is not None:
            report.classified += 1
            report.qualified += int(rejection is None or rejection.stage != "intent")
        if rejection is not None:
            report.rejections.append(rejection)
        report.leads += outcome.leads
        for lead in outcome.leads:
            log.info("LEAD %s <%s> posted %s on %s", lead.business_name, lead.email, lead.posted_at.date(), lead.platform)
        if rejection is not None and rejection.stage in ("error", "retry"):
            # Our side failed (outage, crash): try the post again later, a bounded number of times.
            attempts = self.store.mark_retry(post.key, post.url)
            if attempts >= MAX_POST_ATTEMPTS:
                log.warning("giving up on %s after %d attempts: %s", post.url, attempts, rejection.reason)
        else:
            stage = "lead" if outcome.leads else rejection.stage if rejection else "done"
            self.store.mark_seen(post.key, post.url, stage)

    def _retry_greylisted(self, deferred: list[PostOutcome]) -> Iterable[PostOutcome]:
        if not deferred:
            return
        delay = self.cfg.verification.greylist_retry_seconds
        ready_at = max(o.retry.deferred_at for o in deferred if o.retry) + delay
        log.info("%d businesses hit mail-server greylisting; re-checking in %.0fs", len(deferred), max(0.0, ready_at - time.monotonic()))
        wait = ready_at - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        for outcome in deferred:
            pending = outcome.retry
            assert pending is not None
            try:
                leads, rejection, _ = self._deliver(outcome.post, pending.assessment, pending.ctx, pending.plan, allow_retry=False)
            except LLMUnavailable:
                raise
            except Exception as exc:
                log.exception("greylist retry failed for %s", outcome.post.url)
                leads, rejection = [], Rejection(outcome.post.url, outcome.post.platform, "error", f"{exc.__class__.__name__}: {exc}")
            yield PostOutcome(outcome.post, leads, rejection, pending.assessment)

    def _collect(self, window: Window, niches: list[NicheTarget]) -> Iterable[RawPost]:
        for source in self.sources:
            log.info("collecting from %s", source.name)
            try:
                yield from source.collect(window, niches)
            except LLMUnavailable:
                raise
            except Exception:  # one broken source must not stop the others
                log.exception("source %s failed", source.name)

    def _screen(self, posts: Iterable[RawPost], window: Window, report: RunReport, limit: Optional[int]) -> list[RawPost]:
        keep: list[RawPost] = []
        keys: set[str] = set()
        for post in posts:
            report.collected += 1
            if post.key in keys:
                continue
            keys.add(post.key)
            if self.store.is_seen(post.key, MAX_POST_ATTEMPTS):
                report.already_seen += 1
                continue
            earliest = post.posted_earliest or post.posted_at
            latest = post.posted_latest or post.posted_at
            if post.posted_at is None or not window.contains_range(earliest, latest):
                report.out_of_window += 1
                reason = "no reliable post date" if post.posted_at is None else f"posted {post.posted_at.date()}"
                report.rejections.append(Rejection(post.url, post.platform, "date", reason))
                continue
            reason = prefilter(post)
            if reason:
                report.prefiltered += 1
                report.rejections.append(Rejection(post.url, post.platform, "prefilter", reason))
                self.store.mark_seen(post.key, post.url, "prefilter")
                continue
            keep.append(post)
            if limit is not None and len(keep) >= limit:
                break
        return keep

    # -- per post ----------------------------------------------------------------------

    def _process_safely(self, post: RawPost) -> PostOutcome:
        try:
            return self.process(post)
        except LLMUnavailable:
            raise
        except Exception as exc:
            log.exception("failed to process %s", post.url)
            return PostOutcome(post, rejection=Rejection(post.url, post.platform, "error", f"{exc.__class__.__name__}: {exc}"))

    def process(self, post: RawPost) -> PostOutcome:
        a = self.classifier.assess(post)
        if a is None:
            # Transient failure: leave the post unseen so the next run retries it.
            return PostOutcome(post, rejection=Rejection(post.url, post.platform, "error", "the post could not be assessed"))
        reason = self.classifier.accept(a)
        if reason:
            return PostOutcome(post, rejection=Rejection(post.url, post.platform, "intent", reason, a.business_name or ""), assessment=a)

        content_lower = post.content().lower()
        hint_text = _comparable(build_user_message(post, include_unverified=False))  # page-sourced text only
        poster_name = _real_name(a.poster_name)
        if poster_name and (not _comparable(poster_name).strip() or _comparable(poster_name) not in hint_text):
            poster_name = None  # not in the post or its metadata: Claude guessed it
        ctx = BusinessContext(
            business_name=a.business_name or post.extra.get("business_name"),
            website=self._grounded_website(post, a.business_website) or post.extra.get("website"),
            location=a.business_location,
            poster=Person(name=poster_name, role=a.poster_role, source="post") if poster_name else None,
            poster_is_decision_maker=a.poster_is_decision_maker,
            # Only addresses that really appear in the post, never model output alone.
            emails_in_post=[e for e in a.emails_in_post if e.lower() in content_lower],
        )
        plan = self.finder.plan(ctx)
        business = ctx.business_name or plan.domain or ""
        for domain in {plan.domain, normalize_domain(ctx.website)}:
            if domain and self.verifier.suppression.blocks(domain):
                # A suppressed client/competitor stays suppressed even if the owner wrote a gmail address.
                rejection = Rejection(post.url, post.platform, "suppressed", f"{domain} is on the suppression list", business)
                return PostOutcome(post, rejection=rejection, assessment=a)
        if not plan.decision_makers:
            stage = "retry" if plan.transient else "contacts"
            return PostOutcome(post, rejection=Rejection(post.url, post.platform, stage, "; ".join(plan.notes), business), assessment=a)
        if plan.domain and self.store.domain_delivered_since(plan.domain, BUSINESS_COOLDOWN_DAYS):
            rejection = Rejection(post.url, post.platform, "duplicate", f"lead for {plan.domain} already delivered", business)
            return PostOutcome(post, rejection=rejection, assessment=a)

        allow_retry = self.cfg.verification.greylist_retry_seconds > 0
        leads, rejection, greylisted = self._deliver(post, a, ctx, plan, allow_retry=allow_retry)
        if greylisted:
            return PostOutcome(post, assessment=a, retry=PendingVerification(a, ctx, plan, time.monotonic()))
        return PostOutcome(post, leads, rejection, a)

    def _deliver(
        self, post: RawPost, a: IntentAssessment, ctx: BusinessContext, plan: ContactPlan, allow_retry: bool
    ) -> tuple[list[Lead], Optional[Rejection], bool]:
        """Verify each decision-maker's candidates and save leads. Returns (leads, rejection, retry_later)."""
        business = ctx.business_name or plan.domain or ""
        leads: list[Lead] = []
        failures: list[str] = []
        greylisted = transient = False
        max_contacts = self.cfg.enrichment.max_contacts_per_business
        budget = self.cfg.verification.max_checks_per_business  # shared by every person at this business
        for dm in plan.decision_makers:
            if len(leads) >= max_contacts:
                break
            if not dm.candidates:
                failures.append(f"{dm.person.name}: no candidate addresses")
                continue
            if budget is not None and budget <= 0:
                failures.append(f"{dm.person.name}: not checked, verification budget for this business used up")
                break
            results = self.verifier.verify_many([c.email for c in dm.candidates], stop_at_first=True, max_checks=budget)
            if budget is not None:
                budget -= sum(1 for r in results.values() if r.checked_now)
            chosen = next((c for c in dm.candidates if results[c.email].deliverable), None)
            if chosen is None:
                greylisted = greylisted or any(results[c.email].retryable for c in dm.candidates)
                transient = transient or any(results[c.email].transient for c in dm.candidates)
                best = dm.candidates[0]
                failures.append(f"{dm.person.name}: {best.email} {results[best.email].status} ({results[best.email].reason})")
                continue
            lead = self._lead(post, a, ctx, plan.domain, dm.person, chosen.email, chosen.source, results[chosen.email].reason)
            # Without a business domain (e.g. an owner's gmail) the address itself is the dedupe key.
            business_key = plan.domain or chosen.email
            with self._save_lock:  # posts are processed in parallel; check-and-save atomically
                if self.store.lead_exists(chosen.email) or (
                    not leads and self.store.domain_delivered_since(business_key, BUSINESS_COOLDOWN_DAYS)
                ):
                    failures.append(f"{chosen.email} already delivered")
                    continue
                self.store.save_lead(lead, business_key)
            leads.append(lead)
        if leads:
            return leads, None, False
        if greylisted and allow_retry:
            return [], None, True
        reason = " | ".join(failures)[:500]
        if greylisted:
            reason = "still greylisted after retry: " + reason
        # An outage on our side (DNS, port 25, verifier API) decided nothing: try again on a later run.
        stage = "retry" if transient else "verify"
        return [], Rejection(post.url, post.platform, stage, reason[:500], business), False

    @staticmethod
    def _grounded_website(post: RawPost, website: Optional[str]) -> Optional[str]:
        """Claude's business_website, only if that domain really appears in the post (no guesses)."""
        domain = normalize_domain(website)
        if not domain:
            return None
        hints = " ".join([post.content(), post.url, str(post.extra.get("website") or "")]).lower()
        if domain in hints:
            return website
        log.debug("ignoring website %s: not in the post, will look the business up instead", website)
        return None

    def _lead(
        self,
        post: RawPost,
        a: IntentAssessment,
        ctx: BusinessContext,
        domain: Optional[str],
        person: Person,
        email: str,
        email_source: str,
        verification: str,
    ) -> Lead:
        assert post.posted_at is not None and post.date_source is not None
        return Lead(
            email=email,
            contact_name=person.name or "",
            contact_role=person.role or ("Decision-maker (per post)" if person.source == "post" else ""),
            business_name=ctx.business_name or domain or "",
            business_website=f"https://{domain}" if domain else "",
            platform=PLATFORM_LABELS.get(post.platform, post.platform.capitalize()),
            intent_type=a.intent_type,
            intent_url=post.url,
            posted_at=post.posted_at,
            date_source=post.date_source,
            services_needed=", ".join(a.services_needed),
            niche=a.niche or post.niche or "",
            location=a.business_location or "",
            intent_score=a.intent_score,
            summary=a.summary,
            post_excerpt=post.content()[:280],
            email_source=email_source,
            verification=verification,
            found_at=utcnow(),
        )
