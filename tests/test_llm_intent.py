import json
from contextlib import contextmanager
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from leadgen.config import Config, LLMConfig
from leadgen.intent import IntentAssessment, IntentClassifier, build_user_message, prefilter
from leadgen.llm import FALLBACK_BETA, LLM, LLMTransient, LLMUnavailable
from leadgen.models import RawPost
from leadgen.niches import NicheResearcher
from leadgen.store import Store

ASSESSMENT = {
    "intent_type": "explicit_request",
    "author_is_service_provider": False,
    "intent_score": 88,
    "services_needed": ["social_media_management", "paid_ads_management"],
    "business_name": "Acme Fitness",
    "business_website": None,
    "business_location": "Austin, TX",
    "niche": "boutique gym",
    "poster_name": "Sarah Lee",
    "poster_role": "Founder",
    "poster_is_decision_maker": True,
    "emails_in_post": [],
    "summary": "New gym owner wants someone to run Instagram and Meta ads.",
}


def mock_client(responder, captured):
    def handler(request):
        captured.append({"body": json.loads(request.content), "headers": dict(request.headers)})
        status, payload = responder(captured[-1]["body"])
        return httpx2.Response(status, json=payload)

    return anthropic.Anthropic(
        api_key="test-key",
        base_url="http://claude.test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def message(text, stop_reason="end_turn", **extra):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "content": [{"type": "text", "text": text}] if text is not None else [],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        **extra,
    }


def test_parse_sends_structured_output_effort_and_fallback():
    captured = []
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (200, message(json.dumps(ASSESSMENT))), captured))
    result = llm.parse(IntentAssessment, system="sys", user="post", effort="low")
    assert result.business_name == "Acme Fitness"
    body, headers = captured[0]["body"], captured[0]["headers"]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in headers["anthropic-beta"]
    assert "thinking" not in body  # Opus 5.5 runs adaptive thinking by default
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_haiku_does_not_send_fallbacks():
    captured = []
    llm = LLM(LLMConfig(model="claude-haiku-5-5"), client=mock_client(lambda b: (200, message(json.dumps(ASSESSMENT))), captured))
    llm.parse(IntentAssessment, system="s", user="u", effort="low")
    assert "fallbacks" not in captured[0]["body"]


def test_refusal_returns_none():
    captured = []
    payload = message(None, stop_reason="refusal", stop_details={"type": "refusal", "category": None, "explanation": None})
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (200, payload), captured))
    assert llm.parse(IntentAssessment, system="s", user="u", effort="low") is None


def test_auth_error_stops_the_run():
    err = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (401, err), []))
    with pytest.raises(LLMUnavailable):
        llm.parse(IntentAssessment, system="s", user="u", effort="low")


def test_server_error_is_transient():
    err = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (500, err), []))
    with pytest.raises(LLMTransient):
        llm.parse(IntentAssessment, system="s", user="u", effort="low")


def test_truncated_or_invalid_output_returns_none():
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (200, message('{"intent_type": "expl', stop_reason="max_tokens")), []))
    assert llm.parse(IntentAssessment, system="s", user="u", effort="low") is None
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (200, message('{"not": "the schema"}')), []))
    assert llm.parse(IntentAssessment, system="s", user="u", effort="low") is None


def test_fallback_response_uses_final_answer_block():
    content = [
        {"type": "text", "text": '{"intent_type": "expl'},
        {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-4-8"}, "trigger": {"type": "refusal"}},
        {"type": "text", "text": json.dumps(ASSESSMENT)},
    ]
    payload = {**message(None), "content": content}
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (200, payload), []))
    assert llm.parse(IntentAssessment, system="s", user="u", effort="low").business_name == "Acme Fitness"


class FakeStreamClient:
    """Returns a paused turn first, then a finished one - checks pause_turn resumption."""

    def __init__(self):
        self.requests = []
        hit = lambda url, age: SimpleNamespace(type="web_search_result", url=url, title="t", page_age=age)  # noqa: E731
        self.responses = [
            SimpleNamespace(
                stop_reason="pause_turn",
                content=[SimpleNamespace(type="web_search_tool_result", content=[hit("https://a.com/1", "2026-10-01")])],
            ),
            SimpleNamespace(
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(type="web_search_tool_result", content=SimpleNamespace(error_code="max_uses_exceeded")),
                    SimpleNamespace(type="text", text="Report: gyms are opening."),
                ],
            ),
        ]
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    @contextmanager
    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        resp = self.responses.pop(0)
        yield SimpleNamespace(get_final_message=lambda: resp)


def test_research_resumes_paused_turns_and_collects_hits():
    client = FakeStreamClient()
    llm = LLM(LLMConfig(), client=client)
    result = llm.research(system="s", user="u", effort="high", max_searches=3)
    assert result.text == "Report: gyms are opening."
    assert [h.url for h in result.hits] == ["https://a.com/1"]
    assert len(client.requests) == 2
    assert client.requests[1]["messages"][-1]["role"] == "assistant"
    assert client.requests[0]["tools"][0]["type"] == "web_search_20260209"


# -- intent -----------------------------------------------------------------------------


def post(text, **kw):
    return RawPost(source="test", platform="linkedin", url="https://www.linkedin.com/posts/x", text=text, **kw)


def test_prefilter():
    assert prefilter(post("We just opened our gym and are looking for a social media manager")) is None
    assert prefilter(post("[For Hire] Social media manager, DM me for rates")) is not None
    assert prefilter(post("I help brands grow on Instagram, book a free audit")) is not None
    assert prefilter(post("Looking for a plumber in Austin for our new office")) == "no social media / ads terms"
    assert prefilter(post("Grand opening of our pilates studio this weekend!", niche="new studios")) is None
    assert prefilter(post("hi")) == "too little text to judge"


def test_user_message_wraps_untrusted_post():
    msg = build_user_message(post("Need ads help", author="Jane Doe", extra={"subreddit": "smallbusiness"}))
    assert "Author (from page metadata): Jane Doe" in msg
    assert "Subreddit: smallbusiness" in msg
    assert msg.endswith("<post>\nNeed ads help\n</post>")


def test_accept_rules():
    cfg = Config()
    clf = IntentClassifier(cfg, llm=None)
    a = IntentAssessment(**ASSESSMENT)
    assert clf.accept(a) is None
    assert clf.accept(a.model_copy(update={"intent_score": 50})) is not None
    assert clf.accept(a.model_copy(update={"author_is_service_provider": True})) == "author is a service provider"
    growth = a.model_copy(update={"intent_type": "growth_signal", "intent_score": 80})
    assert clf.accept(growth) is None
    cfg.intent.include_growth_signals = False
    assert clf.accept(growth) == "growth signals disabled"
    assert clf.accept(a.model_copy(update={"intent_type": "not_relevant"})) == "not a buying signal"


# -- niches -----------------------------------------------------------------------------


class FakeNicheLLM:
    def __init__(self):
        self.research_calls = 0

    def research(self, **kw):
        self.research_calls += 1
        return SimpleNamespace(text="report", hits=[])

    def parse(self, schema, **kw):
        return schema.model_validate(
            {
                "niches": [
                    {
                        "name": "New pilates studios",
                        "rationale": "Openings up 30%",
                        "search_queries": ['"grand opening" pilates site:instagram.com', " ", "q2", "q3", "q4", "q5", "q6"],
                        "subreddits": ["r/pilates", "smallbusiness"],
                    }
                ]
            }
        )


def test_niche_research_is_cached(tmp_path):
    llm = FakeNicheLLM()
    store = Store(tmp_path / "db.sqlite")
    researcher = NicheResearcher(Config(), llm, store)
    niches = researcher.get()
    assert niches[0].name == "New pilates studios"
    assert len(niches[0].search_queries) == 5
    assert niches[0].subreddits == ["pilates", "smallbusiness"]
    assert researcher.get()[0].name == "New pilates studios"
    assert llm.research_calls == 1
