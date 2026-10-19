"""Free alternative to Claude: a model running locally under Ollama (https://ollama.com).

Same interface as leadgen.llm.LLM for the calls that do not need the web (structured
extraction). There is no web search, so research() returns None and niche research falls
back to query templates.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional, TypeVar

import requests
from pydantic import BaseModel, TypeAdapter, ValidationError

from leadgen.llm import LLMTransient, LLMUnavailable, ResearchResult

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace local $ref/$defs with their definitions; simple grammars handle that best."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return resolve(copy.deepcopy(defs[ref.split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class OllamaLLM:
    """Structured output from a local model. Costs nothing; quality depends on the model."""

    can_search_web = False

    def __init__(self, base_url: str, model: str, timeout: float = 300.0, session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()

    def preflight(self) -> None:
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=15)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except (requests.RequestException, ValueError) as exc:
            raise LLMUnavailable(f"Ollama is not reachable at {self.base_url} ({exc.__class__.__name__}); start it first") from exc
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            raise LLMUnavailable(f"Ollama model {self.model} is not installed: run `ollama pull {self.model}`")

    def parse(self, schema: type[T], *, system: str, user: str, effort: str = "low", max_tokens: int = 8000) -> Optional[T]:
        body = {
            "model": self.model,
            "stream": False,
            "format": inline_refs(TypeAdapter(schema).json_schema()),
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system + "\n\nAnswer with JSON that matches the requested schema."},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self.session.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
        except requests.ConnectionError as exc:
            raise LLMTransient(f"Ollama not reachable: {exc}") from exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise LLMTransient(f"Ollama error {exc.response.status_code}") from exc
            log.warning("Ollama rejected the request (%s); skipping item", exc)
            return None
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise LLMTransient(f"Ollama request failed: {exc}") from exc
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            log.warning("local model output did not match %s (%d errors); skipping item", schema.__name__, exc.error_count())
            return None

    def research(self, **_: Any) -> Optional[ResearchResult]:
        return None  # a local model cannot search the web
