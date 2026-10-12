"""Mailbox verification over SMTP, without sending mail.

Connects to the domain's MX, issues MAIL FROM and one RCPT TO per candidate, then RCPT TO a
random address that cannot exist. If the random address is accepted the domain is catch-all
and no address on it can be confirmed. The session ends with QUIT; no DATA is ever sent.

Outbound port 25 is blocked on most cloud providers and home ISPs. Run this from a host that
can reach port 25, with a HELO name and MAIL FROM domain you control (see README).
"""

from __future__ import annotations

import logging
import re
import secrets
import smtplib
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

log = logging.getLogger(__name__)

RcptOutcome = Literal["accepted", "rejected", "full", "temp", "blocked", "error"]

_ENHANCED_RE = re.compile(r"\b([245])\.(\d{1,3})\.(\d{1,3})\b")
_BLOCK_WORDS = re.compile(
    r"spamhaus|blocked|block list|blocklist|blacklist|black list|listed|reputation|policy|rbl|dnsbl|"
    r"not permitted|relay|spf|dmarc|authenticat|reverse dns|ptr record|helo|too many|rate limit|"
    r"try again later|banned|refused",
    re.IGNORECASE,
)
_UNKNOWN_USER_WORDS = re.compile(
    r"user unknown|unknown user|no such user|does not exist|doesn't exist|not exist|unknown recipient|"
    r"recipient unknown|invalid recipient|recipient not found|address rejected|mailbox unavailable|"
    r"mailbox not found|no mailbox|not a valid mailbox|undeliverable|disabled|deactivated|inactive|"
    r"suspended|no longer|unrouteable|unroutable",
    re.IGNORECASE,
)
_FULL_WORDS = re.compile(r"quota|mailbox full|mailbox is full|over quota|insufficient storage", re.IGNORECASE)


@dataclass
class RcptResult:
    outcome: RcptOutcome
    code: int
    message: str


@dataclass
class SmtpProbe:
    host: Optional[str]
    catch_all: Optional[bool]  # None: could not determine
    results: dict[str, RcptResult] = field(default_factory=dict)
    error: Optional[str] = None


def _decode(message: bytes | str) -> str:
    return message.decode("utf-8", "replace") if isinstance(message, bytes) else message


def classify_rcpt(code: int, message: bytes | str) -> RcptOutcome:
    text = " ".join(_decode(message).split())
    if code in (250, 251):
        return "accepted"
    if code == 252 or 400 <= code < 500:
        return "temp"  # 252: cannot verify; 4xx: greylisting / try later
    if code == 552:
        return "full"
    if not 500 <= code < 600:
        return "error"
    enhanced = _ENHANCED_RE.search(text)
    if enhanced:
        _, subject, detail = enhanced.groups()
        if subject == "2" and detail == "2":
            return "full"
        if subject == "1" or (subject == "2" and detail == "1"):
            return "rejected"  # 5.1.x bad mailbox / 5.2.1 mailbox disabled
        if subject == "4" and detail == "1" and "access denied" in text.lower():
            return "rejected"  # Microsoft 365 directory-based edge blocking: no such recipient
        if subject == "7":
            return "rejected" if _UNKNOWN_USER_WORDS.search(text) and not _BLOCK_WORDS.search(text) else "blocked"
    if _FULL_WORDS.search(text):
        return "full"
    if _BLOCK_WORDS.search(text):
        return "blocked"
    if _UNKNOWN_USER_WORDS.search(text):
        return "rejected"
    return "rejected" if code in (550, 551, 553) else "blocked"


SmtpFactory = Callable[..., smtplib.SMTP]


class SmtpProber:
    def __init__(
        self,
        helo_host: str,
        mail_from: str,
        timeout: float = 12.0,
        port: int = 25,
        smtp_factory: SmtpFactory = smtplib.SMTP,
        max_hosts: int = 2,
    ) -> None:
        self.helo_host = helo_host
        self.mail_from = mail_from
        self.timeout = timeout
        self.port = port
        self.smtp_factory = smtp_factory
        self.max_hosts = max_hosts

    def probe(self, domain: str, mx_hosts: list[str], emails: list[str]) -> SmtpProbe:
        last_error = "no MX hosts"
        for host in mx_hosts[: self.max_hosts]:
            try:
                return self._probe_host(host, domain, emails)
            except (OSError, smtplib.SMTPException) as exc:
                last_error = f"{host}: {exc.__class__.__name__}: {exc}"
                log.debug("SMTP probe failed on %s: %s", host, last_error)
        return SmtpProbe(
            host=None,
            catch_all=None,
            results={e: RcptResult("error", 0, last_error) for e in emails},
            error=last_error,
        )

    def _probe_host(self, host: str, domain: str, emails: list[str]) -> SmtpProbe:
        smtp = self.smtp_factory(timeout=self.timeout)
        try:
            smtp.connect(host, self.port)
            code, msg = smtp.ehlo(self.helo_host)
            if code != 250:
                code, msg = smtp.helo(self.helo_host)
            if code != 250:
                return self._all(host, emails, "blocked", code, f"HELO rejected: {_decode(msg)}")
            code, msg = smtp.mail(self.mail_from)
            if code != 250:
                return self._all(host, emails, "blocked", code, f"MAIL FROM rejected: {_decode(msg)}")

            results: dict[str, RcptResult] = {}
            for email in emails:
                code, msg = smtp.rcpt(email)
                results[email] = RcptResult(classify_rcpt(code, msg), code, _decode(msg))
                if code == 421:  # server is closing the connection
                    break
            for email in emails:
                results.setdefault(email, RcptResult("error", 421, "connection closed by server"))

            catch_all: Optional[bool] = None
            if results and any(r.code != 421 for r in results.values()):
                probe_addr = f"{secrets.token_hex(10)}@{domain}"
                code, msg = smtp.rcpt(probe_addr)
                outcome = classify_rcpt(code, msg)
                catch_all = True if outcome == "accepted" else False if outcome == "rejected" else None
            return SmtpProbe(host=host, catch_all=catch_all, results=results)
        finally:
            try:
                smtp.quit()
            except (OSError, smtplib.SMTPException):
                smtp.close()

    @staticmethod
    def _all(host: str, emails: list[str], outcome: RcptOutcome, code: int, message: str) -> SmtpProbe:
        return SmtpProbe(
            host=host,
            catch_all=None,
            results={e: RcptResult(outcome, code, message) for e in emails},
            error=message,
        )
