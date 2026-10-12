"""Thin wrapper over the Anthropic SDK for the two kinds of calls the agent makes:

* ``parse``: one structured-output call (intent classification, people extraction).
* ``research``: an open-ended web-search session (niche research, Claude-backed search).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

import anthropic
from pydantic import BaseModel

from leadgen.config import LLMConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH_TOOL = "web_search_20260209"
MAX_PAUSE_CONTINUATIONS = 5


@dataclass
class WebHit:
    url: str
    title: str
    page_age: Optional[str]


@dataclass
class ResearchResult:
    text: str
    hits: list[WebHit] = field(default_factory=list)


class LLMUnavailable(RuntimeError):
    """Raised when there are no usable credentials; the run cannot continue."""


def has_credentials(client: anthropic.Anthropic) -> bool:
    """Without any of these the SDK only fails later, with a TypeError, on the first request."""
    return any(getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials"))


class LLM:
    def __init__(self, cfg: LLMConfig, client: Optional[anthropic.Anthropic] = None, api_key: Optional[str] = None):
        self.cfg = cfg
        if client is None:
            # With api_key=None the SDK resolves credentials itself (ANTHROPIC_API_KEY, auth token, profile).
            client = anthropic.Anthropic(api_key=api_key, max_retries=4)
            if not has_credentials(client):
                raise LLMUnavailable("no Anthropic credentials found: set ANTHROPIC_API_KEY in .env")
        self.client = client

    def _fallback_kwargs(self) -> dict[str, Any]:
        # Haiku has no server-side fallback; sending the parameter there is an error or a no-op.
        if self.cfg.server_side_fallback and not self.cfg.model.startswith("claude-haiku"):
            return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return {}

    def parse(self, schema: type[T], *, system: str, user: str, effort: str, max_tokens: int = 8000) -> Optional[T]:
        """Single structured-output request. Returns None if the model declined or ran out of tokens."""
        try:
            resp = self.client.beta.messages.parse(
                model=self.cfg.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": effort},
                **self._fallback_kwargs(),
            )
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable(f"Anthropic authentication failed: {exc.message}") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMUnavailable(f"Anthropic API key lacks permission: {exc.message}") from exc
        except anthropic.BadRequestError as exc:
            log.warning("Claude rejected the request (%s); skipping item", exc.message)
            return None
        except anthropic.APIStatusError as exc:
            log.warning("Claude API error %s; skipping item", exc.status_code)
            return None
        except anthropic.APIConnectionError:
            log.warning("could not reach the Claude API; skipping item")
            return None

        if resp.stop_reason == "refusal":
            category = resp.stop_details.category if resp.stop_details else None
            log.warning("Claude declined the request (category=%s); skipping item", category)
            return None
        if resp.stop_reason == "max_tokens":
            log.warning("Claude hit max_tokens before finishing structured output; skipping item")
            return None
        return resp.parsed_output

    def research(self, *, system: str, user: str, effort: str, max_searches: int) -> Optional[ResearchResult]:
        """Let Claude search the web on its own, then return its final report and every result it saw."""
        tools = [{"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": max_searches}]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        hits: dict[str, WebHit] = {}
        resp = None
        try:
            for _ in range(MAX_PAUSE_CONTINUATIONS + 1):
                with self.client.beta.messages.stream(
                    model=self.cfg.model,
                    max_tokens=64000,
                    system=system,
                    messages=messages,
                    tools=tools,
                    output_config={"effort": effort},
                    **self._fallback_kwargs(),
                ) as stream:
                    resp = stream.get_final_message()
                _collect_hits(resp.content, hits)
                if resp.stop_reason != "pause_turn":
                    break
                # A long server-tool turn was paused; send it back so Claude can resume.
                messages.append({"role": "assistant", "content": resp.content})
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable(f"Anthropic authentication failed: {exc.message}") from exc
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            log.warning("web research call failed: %s", exc)
            return None

        if resp is None or resp.stop_reason == "refusal":
            log.warning("Claude declined the research request")
            return None
        text = "\n".join(b.text for b in resp.content if b.type == "text")
        return ResearchResult(text=text, hits=list(hits.values()))


def _collect_hits(content: list[Any], hits: dict[str, WebHit]) -> None:
    for block in content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        results = block.content
        if not isinstance(results, list):  # error object, e.g. max_uses_exceeded
            log.debug("web search error: %s", getattr(results, "error_code", results))
            continue
        for r in results:
            if getattr(r, "type", None) == "web_search_result" and r.url not in hits:
                hits[r.url] = WebHit(url=r.url, title=r.title, page_age=r.page_age)
