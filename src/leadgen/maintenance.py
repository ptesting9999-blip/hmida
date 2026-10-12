"""Keeping delivered leads trustworthy: re-verify their emails before they are used."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from leadgen.models import VerificationResult
from leadgen.store import Store
from leadgen.verify import EmailVerifier

log = logging.getLogger(__name__)


@dataclass
class ReverifyReport:
    checked: int = 0
    still_valid: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Re-verified {self.checked} leads: {len(self.still_valid)} still deliverable, "
            f"{len(self.invalid)} now invalid (removed from exports), "
            f"{len(self.unconfirmed)} could not be confirmed right now (excluded until confirmed)"
        )


def reverify_leads(
    store: Store,
    verifier: EmailVerifier,
    older_than_days: int,
    greylist_retry_seconds: int = 0,
    sleep: Callable[[float], None] = time.sleep,
) -> ReverifyReport:
    report = ReverifyReport()
    emails = store.leads_due_for_reverification(older_than_days)
    report.checked = len(emails)
    results: dict[str, VerificationResult] = {}
    for email in emails:
        results[email] = verifier.verify_many([email], use_cache=False)[email]

    retry = [e for e, r in results.items() if r.retryable]
    if retry and greylist_retry_seconds > 0:
        log.info("%d mailboxes greylisted; re-checking in %ds", len(retry), greylist_retry_seconds)
        sleep(greylist_retry_seconds)
        for email in retry:
            results[email] = verifier.verify_many([email], use_cache=False)[email]

    for email, r in results.items():
        if r.deliverable:
            store.set_lead_status(email, "active", r.reason, verified=True)
            report.still_valid.append(email)
        elif r.status in ("undeliverable", "risky"):
            store.set_lead_status(email, "invalid", f"{r.status}: {r.reason}", verified=True)
            report.invalid.append(email)
        else:
            store.set_lead_status(email, "unconfirmed", f"could not re-confirm: {r.reason}", verified=False)
            report.unconfirmed.append(email)
    return report
