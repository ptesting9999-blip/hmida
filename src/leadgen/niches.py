"""Autonomous audience research: Claude searches the web for niches that need SMM / ads help now,
and writes the search queries the agent then uses to find individual businesses in them."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from pydantic import BaseModel, Field

from leadgen.config import Config
from leadgen.dates import utcnow
from leadgen.llm import LLM
from leadgen.sources.base import NicheTarget
from leadgen.store import Store

log = logging.getLogger(__name__)

MAX_QUERIES_PER_NICHE = 5
MAX_QUERY_CHARS = 300


class NicheSuggestion(BaseModel):
    name: str = Field(description="Short niche label, e.g. 'newly opened boutique gyms in Texas'")
    rationale: str = Field(description="Why this niche needs social media / ads help right now, with evidence")
    search_queries: list[str] = Field(
        description="Web search queries that surface individual businesses in this niche showing the signal "
        "recently, e.g. '\"grand opening\" \"pilates studio\" site:instagram.com'"
    )
    subreddits: list[str] = Field(description="Subreddit names (without r/) where these owners post")


class NicheReport(BaseModel):
    niches: list[NicheSuggestion]


RESEARCH_SYSTEM = (
    "You are a market researcher for a social media marketing and paid ads agency. Use web search to "
    "ground every claim in current evidence. Be specific and practical."
)


def research_prompt(cfg: Config) -> str:
    locations = ", ".join(cfg.agency.target_locations) or "anywhere (online services)"
    seeds = "\n".join(f"- {s}" for s in cfg.niches.seed_niches) or "- (none)"
    return f"""Today is {utcnow().date().isoformat()}.

Our agency sells: {'; '.join(cfg.agency.services)}.
Ideal client: {cfg.agency.ideal_client}
Target locations: {locations}

Research which business niches show the strongest signals, right now, of needing outside help with \
social media management or paid ads. Look for evidence such as waves of new openings or launches, \
recent funding, seasonal demand that is about to peak, platform or regulatory changes, and niches where \
owners are publicly asking for marketing help.

Seed niches to evaluate (keep the ones that hold up, add better ones):
{seeds}

Recommend at most {cfg.niches.max_niches} niches. For each, give:
1. a short name,
2. why now, citing what you found,
3. {MAX_QUERIES_PER_NICHE} or fewer web search queries that would surface INDIVIDUAL businesses in that niche \
showing the signal in the last {cfg.window.max_age_days} days - e.g. public Instagram, LinkedIn or Facebook \
posts announcing an opening or launch (use operators such as site:instagram.com, site:linkedin.com/posts \
and quoted phrases),
4. subreddits where owners in that niche post."""


class NicheResearcher:
    def __init__(self, cfg: Config, llm: Optional[LLM], store: Store) -> None:
        self.cfg = cfg
        self.llm = llm
        self.store = store

    def get(self, refresh: bool = False) -> list[NicheTarget]:
        if not refresh:
            cached = self.store.latest_niches(self.cfg.niches.refresh_days)
            if cached is not None:
                return [NicheTarget(**n) for n in cached]
        if self.llm is None or not self.cfg.niches.auto_research:
            return []
        niches = self.research()
        if niches:
            self.store.save_niches([asdict(n) for n in niches])
        return niches

    def research(self) -> list[NicheTarget]:
        assert self.llm is not None
        log.info("researching niches with web search")
        result = self.llm.research(
            system=RESEARCH_SYSTEM,
            user=research_prompt(self.cfg),
            effort=self.cfg.llm.research_effort,
            max_searches=self.cfg.llm.max_web_searches,
        )
        if result is None or not result.text.strip():
            log.warning("niche research returned nothing; continuing with explicit-intent queries only")
            return []
        report = self.llm.parse(
            NicheReport,
            system="Convert this market research report into the requested structure. Keep every niche, "
            "query and subreddit exactly as written; do not add new ones.",
            user=result.text,
            effort=self.cfg.llm.extract_effort,
        )
        if report is None:
            return []
        return [to_target(s) for s in report.niches[: self.cfg.niches.max_niches]]


def to_target(s: NicheSuggestion) -> NicheTarget:
    queries = [q.strip()[:MAX_QUERY_CHARS] for q in s.search_queries if q.strip()][:MAX_QUERIES_PER_NICHE]
    subs = [r.strip().removeprefix("/").removeprefix("r/") for r in s.subreddits if r.strip()]
    return NicheTarget(name=s.name.strip(), rationale=s.rationale.strip(), search_queries=queries, subreddits=subs)
