"""MX lookups: the system resolver, DNS-over-HTTPS, or the first falling back to the second."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import dns.exception
import dns.resolver
import requests

MxStatus = Literal["ok", "no_mx", "nxdomain", "null_mx", "error"]


@dataclass
class MxResult:
    status: MxStatus
    hosts: list[str] = field(default_factory=list)  # ordered by preference
    detail: str = ""


def lookup_mx(domain: str, timeout: float = 8.0, resolver: Optional[dns.resolver.Resolver] = None) -> MxResult:
    try:
        # Building a resolver fails on machines with no nameservers configured (NoResolverConfiguration).
        answer = (resolver or dns.resolver.Resolver()).resolve(domain, "MX", lifetime=timeout)
    except dns.resolver.NXDOMAIN:
        return MxResult("nxdomain", detail="domain does not exist")
    except dns.resolver.NoAnswer:
        return MxResult("no_mx", detail="domain has no MX record")
    except dns.exception.DNSException as exc:
        return MxResult("error", detail=f"DNS lookup failed: {exc.__class__.__name__}")
    records = sorted((r.preference, str(r.exchange).rstrip(".").lower()) for r in answer)
    hosts = [host for _, host in records if host]
    if not hosts:  # RFC 7505 null MX: "0 ." means the domain accepts no mail
        return MxResult("null_mx", detail="domain publishes a null MX (accepts no mail)")
    return MxResult("ok", hosts=hosts)


# DNS-over-HTTPS JSON API (RFC 8484 servers that also speak application/dns-json).
DEFAULT_DOH_URL = "https://cloudflare-dns.com/dns-query"
_MX_TYPE = 15


def lookup_mx_doh(
    domain: str, url: str = DEFAULT_DOH_URL, timeout: float = 8.0, session: Optional[requests.Session] = None
) -> MxResult:
    """MX lookup over HTTPS, for machines without working DNS (containers, locked-down networks)."""
    try:
        resp = (session or requests).get(
            url, params={"name": domain, "type": "MX"}, headers={"Accept": "application/dns-json"}, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        return MxResult("error", detail=f"DNS-over-HTTPS lookup failed: {exc.__class__.__name__}")
    rcode = data.get("Status")
    if rcode == 3:
        return MxResult("nxdomain", detail="domain does not exist")
    if rcode != 0:
        return MxResult("error", detail=f"DNS-over-HTTPS answered rcode {rcode}")
    records = []
    for answer in data.get("Answer") or []:
        if answer.get("type") != _MX_TYPE:
            continue  # CNAME hops etc.
        pref, _, host = str(answer.get("data", "")).strip().partition(" ")
        try:
            records.append((int(pref), host.strip().rstrip(".").lower()))
        except ValueError:
            continue
    if not records:
        return MxResult("no_mx", detail="domain has no MX record")
    hosts = [host for _, host in sorted(records) if host]
    if not hosts:
        return MxResult("null_mx", detail="domain publishes a null MX (accepts no mail)")
    return MxResult("ok", hosts=hosts)


class MxLookup:
    """mode system: this machine's resolver. doh: DNS-over-HTTPS only.
    auto: the resolver, and DNS-over-HTTPS whenever it fails or says a domain has no mail."""

    def __init__(self, mode: Literal["auto", "system", "doh"] = "auto", doh_url: str = DEFAULT_DOH_URL,
                 timeout: float = 8.0, session: Optional[requests.Session] = None) -> None:
        self.mode = mode
        self.doh_url = doh_url
        self.timeout = timeout
        self.session = session  # None: a fresh connection per lookup (lookups are cached; this is thread-safe)

    def __call__(self, domain: str) -> MxResult:
        if self.mode == "doh":
            return self._doh(domain)
        local = lookup_mx(domain, self.timeout)
        if self.mode == "system" or local.status in ("ok", "null_mx"):
            return local
        # Filtering or missing resolvers answer NXDOMAIN for everything: get a second opinion.
        remote = self._doh(domain)
        return local if remote.status == "error" else remote

    def _doh(self, domain: str) -> MxResult:
        return lookup_mx_doh(domain, self.doh_url, self.timeout, self.session)
