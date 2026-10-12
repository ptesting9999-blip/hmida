"""Builds the agent's components from config + secrets, and checks the setup makes sense."""

from __future__ import annotations

import logging
from typing import Optional

import anthropic

from leadgen.compliance import Suppression
from leadgen.config import Config, Secrets
from leadgen.enrich import ContactFinder
from leadgen.enrich.hunter import Hunter
from leadgen.enrich.website import WebsiteCrawler
from leadgen.http import HttpClient
from leadgen.intent import IntentClassifier
from leadgen.llm import LLM
from leadgen.niches import NicheResearcher
from leadgen.output import OutputWriter
from leadgen.pipeline import LeadAgent
from leadgen.sources import (
    BraveSearch,
    ClaudeWebSearch,
    ImportSource,
    RedditSource,
    RssSource,
    SearchBackend,
    SearchSource,
    SerperSearch,
    Source,
)
from leadgen.store import Store
from leadgen.verify import EmailVerifier
from leadgen.verify.providers import MillionVerifier, NeverBounce, VerificationProvider, ZeroBounce
from leadgen.verify.smtp_check import SmtpProber

log = logging.getLogger(__name__)


class SetupError(RuntimeError):
    pass


def make_http(cfg: Config) -> tuple[HttpClient, HttpClient]:
    """(polite crawler client, API client). APIs have no robots.txt and their own rate limits."""
    crawler = HttpClient(cfg.http.user_agent, cfg.http.timeout, cfg.http.respect_robots_txt, cfg.http.per_host_delay)
    api = HttpClient(cfg.http.user_agent, cfg.http.timeout, respect_robots_txt=False, per_host_delay=0.25)
    return crawler, api


def make_llm(cfg: Config, secrets: Secrets) -> LLM:
    try:
        return LLM(cfg.llm, api_key=secrets.anthropic_api_key)
    except anthropic.AnthropicError as exc:
        raise SetupError(f"Claude is required: set ANTHROPIC_API_KEY ({exc})") from exc


def make_search_backend(cfg: Config, secrets: Secrets, api_http: HttpClient, llm: LLM) -> SearchBackend:
    choice = cfg.sources.search.provider
    if choice in ("auto", "brave") and secrets.brave_api_key:
        return BraveSearch(secrets.brave_api_key, api_http)
    if choice in ("auto", "serper") and secrets.serper_api_key:
        return SerperSearch(secrets.serper_api_key, api_http)
    if choice in ("brave", "serper"):
        raise SetupError(f"search provider {choice!r} selected but {choice.upper()}_API_KEY is not set")
    return ClaudeWebSearch(llm)


def make_verification_provider(cfg: Config, secrets: Secrets, api_http: HttpClient) -> Optional[VerificationProvider]:
    choice = cfg.verification.provider
    if choice == "none":
        return None
    options = {
        "zerobounce": (secrets.zerobounce_api_key, lambda k: ZeroBounce(k, api_http, cfg.verification.max_inactive_days is not None)),
        "neverbounce": (secrets.neverbounce_api_key, lambda k: NeverBounce(k, api_http)),
        "millionverifier": (secrets.millionverifier_api_key, lambda k: MillionVerifier(k, api_http)),
    }
    if choice != "auto":
        key, build = options[choice]
        if not key:
            raise SetupError(f"verification provider {choice!r} selected but {choice.upper()}_API_KEY is not set")
        return build(key)
    for key, build in options.values():
        if key:
            return build(key)
    return None


def make_verifier(cfg: Config, secrets: Secrets, store: Optional[Store], api_http: HttpClient) -> EmailVerifier:
    v = cfg.verification
    prober = SmtpProber(v.helo_host, v.smtp_from, v.smtp_timeout, v.smtp_port) if v.smtp else None
    provider = make_verification_provider(cfg, secrets, api_http)
    if prober is None and provider is None:
        raise SetupError(
            "no mailbox verification method: enable verification.smtp or set a verification API key "
            "(ZEROBOUNCE_API_KEY / NEVERBOUNCE_API_KEY / MILLIONVERIFIER_API_KEY)"
        )
    if prober is not None and v.smtp_from.endswith("@example.com"):
        log.warning(
            "verification.smtp_from is still %s - set it to an address on a domain you own, "
            "or many mail servers will refuse the probe",
            v.smtp_from,
        )
    return EmailVerifier(v, store=store, suppression=Suppression(cfg.suppression_file), prober=prober, provider=provider)


def build_agent(cfg: Config, secrets: Secrets) -> LeadAgent:
    crawler_http, api_http = make_http(cfg)
    llm = make_llm(cfg, secrets)
    store = Store(cfg.db_path)
    backend = make_search_backend(cfg, secrets, api_http, llm)

    sources: list[Source] = []
    if cfg.sources.search.enabled:
        sources.append(SearchSource(cfg.sources.search, backend))
    if cfg.sources.reddit.enabled:
        sources.append(RedditSource(cfg.sources.reddit, api_http, secrets.reddit_client_id, secrets.reddit_client_secret))
    if cfg.sources.rss:
        sources.append(RssSource(cfg.sources.rss, crawler_http))
    if cfg.sources.imports:
        sources.append(ImportSource(cfg.sources.imports))
    if not sources:
        raise SetupError("no sources enabled")

    hunter = None
    if cfg.enrichment.use_hunter != "never":
        if secrets.hunter_api_key:
            hunter = Hunter(secrets.hunter_api_key, api_http)
        elif cfg.enrichment.use_hunter == "always":
            raise SetupError("enrichment.use_hunter is 'always' but HUNTER_API_KEY is not set")

    finder = ContactFinder(
        cfg,
        crawler=WebsiteCrawler(crawler_http, cfg.enrichment.max_pages_per_site),
        llm=llm,
        hunter=hunter,
        lookup_backend=backend,
    )
    verifier = make_verifier(cfg, secrets, store, api_http)
    log.info(
        "sources: %s | search: %s | verification: %s",
        ", ".join(s.name for s in sources),
        backend.name,
        " + ".join(verifier.methods),
    )
    return LeadAgent(
        cfg,
        sources=sources,
        classifier=IntentClassifier(cfg, llm),
        finder=finder,
        verifier=verifier,
        store=store,
        writer=OutputWriter(cfg.output),
        niche_researcher=NicheResearcher(cfg, llm, store),
    )
