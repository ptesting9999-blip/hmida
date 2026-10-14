"""Thin wrapper over the Anthropic SDK for the two kinds of calls the agent makes:

* ``parse``: one structured-output call (intent classification, people extraction).
* ``research``: an open-ended web-search session (niche research, Claude-backed search).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

import anthropic
from anthropic import transform_schema
from pydantic import BaseModel, TypeAdapter, ValidationError

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
    # Excerpts of this page quoted by the search server in Claude's citations (real page text,
    # unlike anything the model writes itself).
    cited: list[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    text: str
    hits: list[WebHit] = field(default_factory=list)


class LLMUnavailable(RuntimeError):
    """Raised when there are no usable credentials; the run cannot continue."""


class LLMTransient(RuntimeError):
    """The Claude API failed in a way that may pass (overload, 5xx, rate limit, network).

    Callers treat the item as "try again later" rather than as a real negative answer.
    """


@functools.lru_cache(maxsize=None)
def _output_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {"type": "json_schema", "schema": transform_schema(TypeAdapter(schema).json_schema())}


def _final_text(content: list[Any]) -> Optional[str]:
    """Text of the final answer: only blocks after the last server-side fallback switch point.

    With refusal fallbacks the response can hold a declined partial answer, a `fallback` block,
    then the fallback model's complete answer. Only the last one is the structured output.
    """
    start = 1 + max((i for i, b in enumerate(content) if getattr(b, "type", None) == "fallback"), default=-1)
    texts = [b.text for b in content[start:] if getattr(b, "type", None) == "text"]
    return "".join(texts) if texts else None


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

    def preflight(self) -> None:
        """Cheap check that the key works and the model is available (no tokens spent).

        Run before collecting posts, so a revoked key does not waste paid search queries.
        """
        try:
            self.client.models.retrieve(self.cfg.model)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMUnavailable(f"Anthropic API key rejected: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMUnavailable(f"model {self.cfg.model} is not available to this API key") from exc
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise LLMUnavailable(f"Claude API unreachable right now ({exc.__class__.__name__})") from exc

    def _fallback_kwargs(self) -> dict[str, Any]:
        # Haiku has no server-side fallback; sending the parameter there is an error or a no-op.
        if self.cfg.server_side_fallback and not self.cfg.model.startswith("claude-haiku"):
            return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return {}

    def parse(self, schema: type[T], *, system: str, user: str, effort: str, max_tokens: int = 8000) -> Optional[T]:
        """Single structured-output request.

        Returns None when the model declined, ran out of tokens or produced output that does not
        match the schema. Raises LLMTransient for failures worth retrying later, LLMUnavailable
        for bad credentials.
        """
        try:
            resp = self.client.beta.messages.create(
                model=self.cfg.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={"effort": effort, "format": _output_format(schema)},
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
            if exc.status_code == 429 or exc.status_code >= 500:
                raise LLMTransient(f"Claude API error {exc.status_code}") from exc
            log.warning("Claude API error %s; skipping item", exc.status_code)
            return None
        except anthropic.APIConnectionError as exc:
            raise LLMTransient("could not reach the Claude API") from exc

        if resp.stop_reason == "refusal":
            category = resp.stop_details.category if resp.stop_details else None
            log.warning("Claude declined the request (category=%s); skipping item", category)
            return None
        if resp.stop_reason == "max_tokens":
            log.warning("Claude hit max_tokens before finishing structured output; skipping item")
            return None
        text = _final_text(resp.content)
        if text is None:
            log.warning("Claude returned no structured output; skipping item")
            return None
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            log.warning("Claude output did not match %s (%d errors); skipping item", schema.__name__, exc.error_count())
            return None

    def research(self, *, system: str, user: str, effort: str, max_searches: int) -> Optional[ResearchResult]:
        """Let Claude search the web on its own, then return its final report and every result it saw."""
        tools = [{"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": max_searches}]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        hits: dict[str, WebHit] = {}
        texts: list[str] = []  # text written before a pause_turn is part of the report too
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
                texts += [b.text for b in resp.content if getattr(b, "type", None) == "text"]
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
        return ResearchResult(text="\n".join(texts), hits=list(hits.values()))


def _collect_hits(content: list[Any], hits: dict[str, WebHit]) -> None:
    for block in content:
        if getattr(block, "type", None) == "text":
            for c in getattr(block, "citations", None) or []:
                if getattr(c, "type", None) == "web_search_result_location" and c.url in hits and c.cited_text:
                    if c.cited_text not in hits[c.url].cited:
                        hits[c.url].cited.append(c.cited_text)
            continue
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        results = block.content
        if not isinstance(results, list):  # error object, e.g. max_uses_exceeded
            log.debug("web search error: %s", getattr(results, "error_code", results))
            continue
        for r in results:
            if getattr(r, "type", None) == "web_search_result" and r.url not in hits:
                hits[r.url] = WebHit(url=r.url, title=r.title, page_age=r.page_age)
