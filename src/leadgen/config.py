"""Configuration: a YAML file for behaviour, environment variables for secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgencyConfig(_Section):
    name: str = "Your Agency"
    services: list[str] = Field(
        default_factory=lambda: [
            "social media management",
            "paid ads management (Meta, Instagram, TikTok, Google)",
            "content creation for social media",
        ]
    )
    # Where your clients are. Steers niche research, and Claude rejects businesses whose stated
    # location is clearly outside these (unknown locations still pass). Empty = anywhere.
    target_locations: list[str] = Field(default_factory=list)
    # Short description of the ideal client, given to Claude when judging leads.
    ideal_client: str = (
        "Small and mid-sized businesses, local businesses, startups and founders who want to "
        "grow through social media and paid ads."
    )


class WindowConfig(_Section):
    """Only posts whose age is within [min_age_days, max_age_days] are kept (default: 2-14 days)."""

    max_age_days: int = Field(default=14, ge=1, le=60)
    min_age_days: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "WindowConfig":
        if self.min_age_days >= self.max_age_days:
            raise ValueError("window.min_age_days must be smaller than window.max_age_days")
        return self


class IntentConfig(_Section):
    # Minimum Claude intent score (0-100) for someone explicitly asking for the service.
    min_score: int = Field(default=70, ge=0, le=100)
    # "Growth signal" leads: businesses that did not ask explicitly but strongly need it
    # (e.g. a gym that opened last week with no marketing). Set false for explicit-only.
    include_growth_signals: bool = True
    growth_signal_min_score: int = Field(default=75, ge=0, le=100)


class SearchSourceConfig(_Section):
    enabled: bool = True
    # auto picks the first one available: brave, serper (paid keys), searxng (free, self-hosted),
    # then claude (only when llm.provider is claude).
    provider: Literal["auto", "brave", "serper", "searxng", "claude"] = "auto"
    # Your own SearXNG instance (free): e.g. http://localhost:8080 or http://searxng:8080 in Docker.
    searxng_url: Optional[str] = None
    platforms: list[str] = Field(
        default_factory=lambda: ["linkedin", "instagram", "facebook", "x", "craigslist", "web"]
    )
    results_per_query: int = Field(default=20, ge=1, le=100)
    # Extra raw search queries (site: operators allowed) run in addition to the built-ins.
    extra_queries: list[str] = Field(default_factory=list)

    @field_validator("platforms")
    @classmethod
    def _known_platforms(cls, value: list[str]) -> list[str]:
        from leadgen.queries import PLATFORM_FILTERS

        unknown = [p for p in value if p not in PLATFORM_FILTERS]
        if unknown:
            raise ValueError(f"unknown platform(s) {unknown}; choose from {sorted(PLATFORM_FILTERS)}")
        return value


class RedditSourceConfig(_Section):
    enabled: bool = True
    subreddits: list[str] = Field(
        default_factory=lambda: [
            "forhire",
            "hiring",
            "smallbusiness",
            "Entrepreneur",
            "startups",
            "marketing",
            "socialmedia",
            "PPC",
            "FacebookAds",
            "digital_marketing",
            "jobbit",
            "freelance_forhire",
        ]
    )
    query: str = (
        '"social media" OR "ads manager" OR "facebook ads" OR "meta ads" OR "instagram" '
        'OR "tiktok ads" OR "google ads" OR "ppc" OR "marketing agency"'
    )
    max_results_per_subreddit: int = Field(default=100, ge=1, le=100)


class FeedConfig(_Section):
    url: str
    platform: str = "web"


class SourcesConfig(_Section):
    search: SearchSourceConfig = Field(default_factory=SearchSourceConfig)
    reddit: RedditSourceConfig = Field(default_factory=RedditSourceConfig)
    rss: list[FeedConfig] = Field(default_factory=list)
    # CSV / JSONL files of posts exported from other tools (see README).
    imports: list[str] = Field(default_factory=list)


class NicheConfig(_Section):
    auto_research: bool = True
    refresh_days: int = Field(default=7, ge=1)
    max_niches: int = Field(default=6, ge=1, le=20)
    # Niches you already know you want; research adds to these.
    seed_niches: list[str] = Field(
        default_factory=lambda: [
            "newly opened gyms and fitness studios",
            "recently launched startups",
            "new restaurants, cafes and med spas",
        ]
    )


class EnrichmentConfig(_Section):
    crawl_website: bool = True
    max_pages_per_site: int = Field(default=6, ge=1, le=25)
    # Page text beyond this many characters is not sent to Claude for people extraction.
    max_chars_per_page: int = Field(default=15000, ge=1000)
    use_hunter: Literal["auto", "always", "never"] = "auto"
    max_email_candidates: int = Field(default=8, ge=1, le=20)
    max_contacts_per_business: int = Field(default=1, ge=1, le=5)


class VerificationConfig(_Section):
    smtp: bool = True
    # MAIL FROM used for SMTP probes. Use an address on a domain you own with valid SPF.
    smtp_from: str = "verify@example.com"
    # EHLO hostname. Should resolve to (and reverse-resolve from) the IP you probe from.
    helo_host: str = "localhost"
    smtp_timeout: float = Field(default=12.0, gt=0)
    smtp_port: int = 25
    # MX lookups. system: this machine's resolver. doh: DNS-over-HTTPS (doh_url) only.
    # auto: the resolver, with DNS-over-HTTPS as a second opinion when it fails.
    dns: Literal["auto", "system", "doh"] = "auto"
    doh_url: str = "https://cloudflare-dns.com/dns-query"
    # auto uses whichever provider has an API key: zerobounce, neverbounce, millionverifier.
    provider: Literal["auto", "zerobounce", "neverbounce", "millionverifier", "none"] = "auto"
    # any: at least one enabled check (smtp/provider) confirms and none objects.
    # all: every enabled check must confirm. Both always reject catch-all/unknown.
    confirmations_required: Literal["any", "all"] = "any"
    # Catch-all domains (and Yahoo/AOL, which accept every RCPT) can't be confirmed over SMTP.
    # true: accept such an address if the verification API positively confirms that mailbox.
    # false (strict default): every address on a catch-all domain is rejected.
    trust_provider_on_catch_all: bool = False
    allow_free_email_providers: bool = True
    # Reject mailboxes with no activity for longer than this (ZeroBounce activity data).
    max_inactive_days: Optional[int] = 365
    cache_days: int = Field(default=30, ge=0)
    # Greylisting servers answer "try again later" to first contact. Businesses whose best
    # address got that answer are re-checked once, this many seconds later (0 = never).
    greylist_retry_seconds: int = Field(default=360, ge=0, le=3600)
    extra_disposable_domains_file: Optional[str] = None


class LLMConfig(_Section):
    # claude: best judgement, paid API. ollama: free model on your own computer.
    # none: free built-in rules (no AI) - less nuanced, zero setup.
    provider: Literal["claude", "ollama", "none"] = "claude"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    model: str = "claude-opus-5-5"
    classify_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    extract_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    research_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    # Server-side refusal fallback (Claude API only). Ignored for Haiku models.
    server_side_fallback: bool = True
    # Web searches Claude may run per research call: niche research, and post discovery when
    # no Brave/Serper key is set. Lower it to cut cost.
    max_web_searches: int = Field(default=8, ge=1, le=30)


class OutputConfig(_Section):
    dir: str = "output"
    formats: list[Literal["csv", "json"]] = Field(default_factory=lambda: ["csv", "json"])
    write_rejected: bool = True
    # New leads are also POSTed to LEADGEN_WEBHOOK_URL (env) when it is set:
    # json = {"leads": [...]} for Zapier / Make / n8n / your CRM, slack = a Slack incoming webhook.
    webhook_format: Literal["json", "slack"] = "json"


class HttpConfig(_Section):
    user_agent: str = "LeadGenAgent/0.1 (+https://example.com/bot)"
    timeout: float = Field(default=15.0, gt=0)
    respect_robots_txt: bool = True
    per_host_delay: float = Field(default=1.0, ge=0)


class Config(_Section):
    agency: AgencyConfig = Field(default_factory=AgencyConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    niches: NicheConfig = Field(default_factory=NicheConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    db_path: str = "data/leadgen.db"
    suppression_file: str = "data/suppression.txt"
    workers: int = Field(default=4, ge=1, le=32)


class Secrets(BaseModel):
    anthropic_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    hunter_api_key: Optional[str] = None
    zerobounce_api_key: Optional[str] = None
    neverbounce_api_key: Optional[str] = None
    millionverifier_api_key: Optional[str] = None
    leadgen_webhook_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(**{name: os.environ.get(name.upper()) or None for name in cls.model_fields})


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    return Config.model_validate(data)
