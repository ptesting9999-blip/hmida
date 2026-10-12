"""Strict verification: an address passes only if it is positively confirmed deliverable.

Order of checks (cheapest first, stop at the first hard failure):
  syntax -> suppression -> role account -> disposable -> free-mail policy
  -> MX records -> SMTP mailbox + catch-all probe -> verification API (+ mailbox activity)

Anything catch-all, unknown, greylisted, full, inactive or contradictory is rejected.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable, Optional

from leadgen.compliance import Suppression
from leadgen.config import VerificationConfig
from leadgen.dates import utcnow
from leadgen.models import VerificationResult, VerificationStatus
from leadgen.store import Store
from leadgen.verify import rules
from leadgen.verify.dns_check import MxResult, lookup_mx
from leadgen.verify.providers import VerificationProvider
from leadgen.verify.smtp_check import SmtpProbe, SmtpProber

log = logging.getLogger(__name__)

# A domain that always has MX records; if it doesn't resolve, the resolver is the problem.
DNS_CANARY_DOMAIN = "gmail.com"


class EmailVerifier:
    def __init__(
        self,
        cfg: VerificationConfig,
        *,
        store: Optional[Store] = None,
        suppression: Optional[Suppression] = None,
        prober: Optional[SmtpProber] = None,
        provider: Optional[VerificationProvider] = None,
        mx_lookup: Callable[[str], MxResult] = lookup_mx,
        disposable: Optional[frozenset[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.suppression = suppression or Suppression()
        self.prober = prober
        self.provider = provider
        self.mx_lookup = mx_lookup
        self.disposable = disposable if disposable is not None else rules.load_disposable_domains(
            cfg.extra_disposable_domains_file
        )
        self._mx_cache: dict[str, MxResult] = {}
        self._resolver_healthy: Optional[bool] = None
        # One SMTP session per domain at a time; mail servers dislike parallel probes.
        self._domain_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    @property
    def methods(self) -> list[str]:
        out = []
        if self.prober is not None:
            out.append("smtp")
        if self.provider is not None:
            out.append(self.provider.name)
        return out

    def verify(self, email: str) -> VerificationResult:
        return self.verify_many([email])[email.strip().lower()]

    def verify_many(
        self, emails: list[str], stop_at_first: bool = False, use_cache: bool = True
    ) -> dict[str, VerificationResult]:
        """Verify addresses in the given (priority) order.

        With stop_at_first, checking stops at the first deliverable address so paid API credits
        are not spent on lower-priority guesses; skipped addresses come back as "unknown".
        All candidates on one domain still share a single SMTP session.
        """
        order: list[str] = []
        for raw in emails:
            email = raw.strip().lower()
            if email not in order:
                order.append(email)

        results: dict[str, VerificationResult] = {}
        pending_by_domain: dict[str, list[str]] = defaultdict(list)
        for email in order:
            problem = self._policy_checks(email)
            if problem is not None:
                results[email] = problem
                continue
            cached = self.store.cached_verification(email, self.cfg.cache_days) if self.store and use_cache else None
            if cached is not None:
                results[email] = cached
                continue
            pending_by_domain[rules.split_email(email)[1]].append(email)

        probes: dict[str, tuple[MxResult, Optional[SmtpProbe]]] = {}
        for email in order:
            if email not in results:
                domain = rules.split_email(email)[1]
                if domain not in probes:
                    with self._lock_for(domain):
                        probes[domain] = self._probe_domain(domain, pending_by_domain[domain])
                mx, probe = probes[domain]
                results[email] = self._decide(email, mx, probe)
                if self.store and results[email].status != "unknown":
                    self.store.save_verification(results[email])
            if stop_at_first and results[email].deliverable:
                break

        for email in order:
            results.setdefault(
                email, _result(email, "unknown", "not checked: a higher-priority address was confirmed", {})
            )
        return results

    # -- stages ------------------------------------------------------------------------

    def _policy_checks(self, email: str) -> Optional[VerificationResult]:
        problem = rules.check_syntax(email)
        if problem:
            return _result(email, "undeliverable", f"invalid syntax: {problem}", {"syntax": problem})
        checks = {"syntax": "ok"}
        if self.suppression.blocks(email):
            return _result(email, "risky", "on suppression list", {**checks, "suppression": "listed"})
        if rules.is_role_account(email):
            return _result(email, "risky", "role-based address, not a person", {**checks, "role": "role account"})
        checks["role"] = "personal"
        if rules.is_disposable(email, self.disposable):
            return _result(email, "undeliverable", "disposable email domain", {**checks, "disposable": "yes"})
        if not self.cfg.allow_free_email_providers and rules.is_free_provider(email):
            return _result(email, "risky", "free email provider not allowed by config", {**checks, "free": "yes"})
        return None

    def _mx(self, domain: str) -> MxResult:
        if domain not in self._mx_cache:
            result = self.mx_lookup(domain)
            if result.status in ("nxdomain", "no_mx") and not self._resolver_ok():
                # A broken or filtering resolver must not get real domains marked as dead.
                result = MxResult("error", detail="DNS resolver is not answering reliably (canary lookup failed)")
            self._mx_cache[domain] = result
        return self._mx_cache[domain]

    def _resolver_ok(self) -> bool:
        if self._resolver_healthy is None:
            self._resolver_healthy = self.mx_lookup(DNS_CANARY_DOMAIN).status == "ok"
            if not self._resolver_healthy:
                log.error("DNS resolver failed to find MX records for %s; MX checks are unreliable", DNS_CANARY_DOMAIN)
        return self._resolver_healthy

    def _lock_for(self, domain: str) -> threading.Lock:
        with self._locks_guard:
            return self._domain_locks[domain]

    def _probe_domain(self, domain: str, emails: list[str]) -> tuple[MxResult, Optional[SmtpProbe]]:
        mx = self._mx(domain)
        if mx.status != "ok" or self.prober is None:
            return mx, None
        return mx, self.prober.probe(domain, mx.hosts, emails)

    def _decide(self, email: str, mx: MxResult, probe: Optional[SmtpProbe]) -> VerificationResult:
        checks = {"syntax": "ok", "role": "personal", "disposable": "no", "mx": mx.status}
        if mx.status != "ok":
            status: VerificationStatus = "unknown" if mx.status == "error" else "undeliverable"
            return _result(email, status, mx.detail, checks)
        checks["mx"] = f"ok ({mx.hosts[0]})"
        verdicts: dict[str, VerificationStatus] = {}
        reasons: list[str] = []

        retryable = False
        if probe is not None:
            smtp_status, smtp_reason, retryable = _smtp_verdict(email, probe)
            accepted_on_catch_all = probe.catch_all is True and smtp_status == "risky" and "catch-all" in smtp_reason
            if accepted_on_catch_all and self.provider is not None and self.cfg.trust_provider_on_catch_all:
                # SMTP can't tell on a catch-all domain; let the verification API decide alone.
                smtp_status = "unknown"
            verdicts["smtp"] = smtp_status
            checks["smtp"] = smtp_reason
            if smtp_status in ("undeliverable", "risky"):
                # A definite negative needs no paid second opinion.
                return _result(email, smtp_status, smtp_reason, checks)
            reasons.append(f"smtp: {smtp_reason}")

        if self.provider is not None:
            v = self.provider.check(email)
            status = v.status
            detail = v.detail
            if (
                status == "deliverable"
                and v.inactive_days is not None
                and self.cfg.max_inactive_days is not None
                and v.inactive_days > self.cfg.max_inactive_days
            ):
                status = "risky"
                detail += f"; no activity seen in {v.inactive_days} days (dormant)"
            elif v.inactive_days is not None:
                detail += f"; active within {v.inactive_days} days"
            verdicts[self.provider.name] = status
            checks[self.provider.name] = detail
            reasons.append(f"{self.provider.name}: {detail}")
            if status in ("undeliverable", "risky"):
                return _result(email, status, detail, checks)

        if not verdicts:
            return _result(email, "unknown", "no mailbox verification method enabled", checks)
        confirmed = [k for k, s in verdicts.items() if s == "deliverable"]
        if self.cfg.confirmations_required == "all":
            ok = len(confirmed) == len(verdicts)
        else:
            ok = bool(confirmed)
        if ok:
            return _result(email, "deliverable", f"confirmed by {', '.join(confirmed)}", checks)
        result = _result(email, "unknown", "; ".join(reasons) or "could not confirm mailbox", checks)
        result.retryable = retryable
        return result


def _smtp_verdict(email: str, probe: SmtpProbe) -> tuple[VerificationStatus, str, bool]:
    """(status, reason, retryable) for one address from a domain's SMTP probe."""
    r = probe.results.get(email)
    if r is None:
        return "unknown", "not probed", False
    msg = " ".join(r.message.split())[:160]
    if r.outcome == "accepted":
        if probe.catch_all is True:
            return "risky", "catch-all domain: server accepts any address, mailbox cannot be confirmed", False
        if probe.catch_all is False:
            return "deliverable", f"mailbox accepted ({r.code}), random address rejected", False
        greylisted = probe.catch_all_outcome == "temp"
        return "unknown", f"mailbox accepted ({r.code}) but catch-all test was inconclusive", greylisted
    if r.outcome == "rejected":
        return "undeliverable", f"mailbox rejected: {r.code} {msg}", False
    if r.outcome == "full":
        return "risky", f"mailbox full / over quota (likely abandoned): {r.code} {msg}", False
    if r.outcome == "temp":
        return "unknown", f"temporary failure / greylisting: {r.code} {msg}", True
    if r.outcome == "blocked":
        return "unknown", f"probe blocked by server policy: {r.code} {msg}", False
    return "unknown", f"SMTP error: {msg}", False


def _result(email: str, status: VerificationStatus, reason: str, checks: dict[str, str]) -> VerificationResult:
    return VerificationResult(email=email, status=status, reason=reason, checks=checks, checked_at=utcnow())
