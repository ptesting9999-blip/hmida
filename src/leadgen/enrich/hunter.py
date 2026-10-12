"""Hunter.io: known people and the email pattern for a domain (optional, needs HUNTER_API_KEY)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

from leadgen.http import HttpClient
from leadgen.models import Person

log = logging.getLogger(__name__)


@dataclass
class HunterDomainResult:
    pattern: Optional[str] = None
    people: list[Person] = field(default_factory=list)


class Hunter:
    DOMAIN_SEARCH_URL = "https://api.hunter.io/v2/domain-search"

    def __init__(self, api_key: str, http: HttpClient) -> None:
        self.api_key = api_key
        self.http = http

    def domain_search(self, domain: str, limit: int = 10) -> HunterDomainResult:
        try:
            data = self.http.get_json(
                self.DOMAIN_SEARCH_URL,
                params={"domain": domain, "api_key": self.api_key, "type": "personal", "limit": limit},
            )
        except (requests.RequestException, ValueError) as exc:
            log.warning("hunter domain search failed for %s: %s", domain, exc)
            return HunterDomainResult()
        d = data.get("data") or {}
        people = []
        for e in d.get("emails") or []:
            name = " ".join(p for p in (e.get("first_name"), e.get("last_name")) if p) or None
            people.append(
                Person(
                    name=name,
                    role=e.get("position"),
                    email=(e.get("value") or "").lower() or None,
                    source="hunter",
                    evidence=f"hunter confidence {e.get('confidence')}",
                )
            )
        return HunterDomainResult(pattern=d.get("pattern"), people=people)
