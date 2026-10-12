"""MX lookups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import dns.exception
import dns.resolver

MxStatus = Literal["ok", "no_mx", "nxdomain", "null_mx", "error"]


@dataclass
class MxResult:
    status: MxStatus
    hosts: list[str] = field(default_factory=list)  # ordered by preference
    detail: str = ""


def lookup_mx(domain: str, timeout: float = 8.0, resolver: Optional[dns.resolver.Resolver] = None) -> MxResult:
    resolver = resolver or dns.resolver.Resolver()
    try:
        answer = resolver.resolve(domain, "MX", lifetime=timeout)
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
