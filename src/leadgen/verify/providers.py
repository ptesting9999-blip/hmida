"""Third-party verification APIs. Used alongside (or instead of) direct SMTP probing.

These services verify from warmed IP pools, handle greylisting, and some report mailbox
activity, which a single SMTP probe cannot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import requests

from leadgen.http import HttpClient
from leadgen.models import VerificationStatus

log = logging.getLogger(__name__)


@dataclass
class ProviderVerdict:
    status: VerificationStatus
    detail: str
    inactive_days: Optional[int] = None
    # The API call itself failed (network, auth, credits): no judgement was made.
    failed: bool = False
    # The provider says the whole domain is catch-all: every other guess there will be too.
    catch_all: bool = False


class VerificationProvider(Protocol):
    name: str

    def check(self, email: str) -> ProviderVerdict: ...


def _parse_activity_days(value: object) -> Optional[int]:
    """ZeroBounce reports activity buckets such as "30", "180", "365", "365+"."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.endswith("+"):
        try:
            return int(text[:-1]) + 1
        except ValueError:
            return None
    try:
        return int(text)
    except ValueError:
        return None


class ZeroBounce:
    name = "zerobounce"
    VALIDATE_URL = "https://api.zerobounce.net/v2/validate"
    ACTIVITY_URL = "https://api.zerobounce.net/v2/activity"
    _STATUS = {
        "valid": "deliverable",
        "invalid": "undeliverable",
        "catch-all": "risky",
        "unknown": "unknown",
        "spamtrap": "undeliverable",
        "abuse": "undeliverable",
        "do_not_mail": "undeliverable",
    }

    def __init__(self, api_key: str, http: HttpClient, check_activity: bool = True) -> None:
        self.api_key = api_key
        self.http = http
        self.check_activity = check_activity

    def check(self, email: str) -> ProviderVerdict:
        try:
            data = self.http.get_json(
                self.VALIDATE_URL, params={"api_key": self.api_key, "email": email, "ip_address": ""}
            )
        except (requests.RequestException, ValueError) as exc:
            return ProviderVerdict("unknown", f"zerobounce request failed: {exc}", failed=True)
        if data.get("error"):
            return ProviderVerdict("unknown", f"zerobounce error: {data['error']}", failed=True)
        raw = str(data.get("status", "unknown")).lower()
        sub = data.get("sub_status") or ""
        verdict = ProviderVerdict(self._STATUS.get(raw, "unknown"), f"{raw}{'/' + sub if sub else ''}", catch_all=raw == "catch-all")
        if verdict.status == "deliverable" and self.check_activity:
            try:
                activity = self.http.get_json(self.ACTIVITY_URL, params={"api_key": self.api_key, "email": email})
                if activity.get("found"):
                    verdict.inactive_days = _parse_activity_days(activity.get("active_in_days"))
            except (requests.RequestException, ValueError) as exc:
                log.debug("zerobounce activity lookup failed: %s", exc)
        return verdict


class NeverBounce:
    name = "neverbounce"
    URL = "https://api.neverbounce.com/v4/single/check"
    _RESULT = {
        "valid": "deliverable",
        "invalid": "undeliverable",
        "disposable": "undeliverable",
        "catchall": "risky",
        "unknown": "unknown",
    }

    def __init__(self, api_key: str, http: HttpClient) -> None:
        self.api_key = api_key
        self.http = http

    def check(self, email: str) -> ProviderVerdict:
        try:
            data = self.http.get_json(self.URL, params={"key": self.api_key, "email": email, "timeout": 10})
        except (requests.RequestException, ValueError) as exc:
            return ProviderVerdict("unknown", f"neverbounce request failed: {exc}", failed=True)
        if data.get("status") != "success":
            return ProviderVerdict("unknown", f"neverbounce error: {data.get('message') or data.get('status')}", failed=True)
        raw = str(data.get("result", "unknown")).lower()
        return ProviderVerdict(self._RESULT.get(raw, "unknown"), raw, catch_all=raw == "catchall")


class MillionVerifier:
    name = "millionverifier"
    URL = "https://api.millionverifier.com/api/v3/"
    _RESULT = {
        "ok": "deliverable",
        "catch_all": "risky",
        "unknown": "unknown",
        "error": "unknown",
        "disposable": "undeliverable",
        "invalid": "undeliverable",
    }

    def __init__(self, api_key: str, http: HttpClient) -> None:
        self.api_key = api_key
        self.http = http

    def check(self, email: str) -> ProviderVerdict:
        try:
            data = self.http.get_json(self.URL, params={"api": self.api_key, "email": email, "timeout": 10})
        except (requests.RequestException, ValueError) as exc:
            return ProviderVerdict("unknown", f"millionverifier request failed: {exc}", failed=True)
        if data.get("error"):
            return ProviderVerdict("unknown", f"millionverifier error: {data['error']}", failed=True)
        raw = str(data.get("result", "unknown")).lower()
        status = self._RESULT.get(raw, "unknown")
        if status == "deliverable" and str(data.get("quality", "")).lower() == "bad":
            status = "risky"
        return ProviderVerdict(status, f"{raw}/{data.get('subresult', '')}".rstrip("/"), catch_all=raw == "catch_all")
