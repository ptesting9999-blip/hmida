"""Free mode: rule-based brain, local Ollama model, SearXNG search, DNS-over-HTTPS, setup checks."""

import json

import pytest
import requests
from test_doctor import FakeResp, FakeSession, by_name, good_config
from test_llm_intent import mock_client
from test_pipeline import NOW
from test_sources import WINDOW

from leadgen.config import Config, LLMConfig, Secrets
from leadgen.dates import Window
from leadgen.doctor import Doctor
from leadgen.factory import SetupError, make_llm, make_search_backend
from leadgen.intent import IntentAssessment, IntentClassifier
from leadgen.llm import LLM, LLMTransient, LLMUnavailable
from leadgen.local_llm import OllamaLLM, inline_refs
from leadgen.models import RawPost
from leadgen.niches import NicheReport, NicheResearcher, template_niche
from leadgen.rule_intent import assess_without_ai
from leadgen.sources.search import SearxngSearch
from leadgen.store import Store
from leadgen.verify.dns_check import MxLookup, MxResult, lookup_mx_doh


def post(text, niche=None, author=None, **extra):
    return RawPost("t", "linkedin", "https://www.linkedin.com/posts/x", text=text, niche=niche, author=author, extra=extra)


# -- rule-based brain ---------------------------------------------------------------------


def test_rules_accept_a_clear_buyer_and_score_specifics_higher():
    plain = assess_without_ai(post("Looking for a social media manager for my bakery."), Config())
    rich = assess_without_ai(post("I'm the founder of Glow Med Spa and we need someone to run our Facebook ads ASAP. "
                                  "Budget $1,500/mo. glowmedspa.com"), Config())
    assert plain.intent_type == rich.intent_type == "explicit_request"
    assert IntentClassifier(Config(), None).accept(plain) is None
    assert rich.intent_score > plain.intent_score
    assert (rich.business_name, rich.business_website, rich.poster_role) == ("Glow Med Spa", "glowmedspa.com", "Founder")
    assert rich.poster_is_decision_maker and "paid_ads_management" in rich.services_needed


def test_rules_reject_sellers_but_not_buyers_asking_for_portfolios():
    for pitch in ("Do you need a social media manager? I help brands grow on Instagram.",
                  "I'm a freelance social media manager, check out my portfolio",
                  "[For Hire] Meta ads specialist, book a free call"):
        a = assess_without_ai(post(pitch), Config())
        assert a.author_is_service_provider and IntentClassifier(Config(), None).accept(a), pitch
    buyer = assess_without_ai(post("Hiring a social media manager for our cafe, please send your portfolio"), Config())
    assert not buyer.author_is_service_provider and buyer.intent_type == "explicit_request"


def test_rules_growth_signals_only_inside_researched_niches():
    text = "Grand opening this Saturday! We just opened Iron Temple Gym downtown."
    assert assess_without_ai(post(text), Config()).intent_type == "not_relevant"
    a = assess_without_ai(post(text, niche="new gyms"), Config())
    assert a.intent_type == "growth_signal" and a.business_name == "Iron Temple Gym" and a.niche == "new gyms"


def test_rules_take_people_and_emails_only_from_the_post():
    a = assess_without_ai(post("Need an ads manager for my studio, email jane@bloomyoga.com", author="Jane Park"), Config())
    assert a.poster_name == "Jane Park" and a.emails_in_post == ["jane@bloomyoga.com"]
    assert a.business_website is None  # the email's domain is not a website the post mentioned
    biz = assess_without_ai(post("Need an ads manager for our studio", author="Bloom Yoga Studio"), Config())
    assert biz.poster_name is None and biz.business_name == "Bloom Yoga Studio"
    social = assess_without_ai(post("Looking for a social media manager, see instagram.com/bloomyoga"), Config())
    assert social.business_website is None


def test_rules_are_fast_on_huge_posts():
    import time

    start = time.monotonic()
    assess_without_ai(post("need " + "very " * 5000 + "social media " * 2000), Config())
    assert time.monotonic() - start < 2


def test_classifier_without_a_model_uses_rules():
    a = IntentClassifier(Config(), None).assess(post("Looking for a TikTok ads manager for our shop"))
    assert a is not None and a.intent_type == "explicit_request"


# -- local model (Ollama) -----------------------------------------------------------------


class OllamaSession:
    def __init__(self, content=None, status=200, exc=None, models=("qwen2.5:7b",)):
        self.content, self.status, self.exc, self.models, self.bodies = content, status, exc, models, []

    def post(self, url, json=None, timeout=None):
        if self.exc:
            raise self.exc
        self.bodies.append(json)
        return FakeResp(self.status, {"message": {"content": self.content}})

    def get(self, url, timeout=None):
        if self.exc:
            raise self.exc
        return FakeResp(200, {"models": [{"name": m} for m in self.models]})


VALID = {
    "intent_type": "explicit_request", "author_is_service_provider": False, "intent_score": 80,
    "services_needed": ["social_media_management"], "business_name": "Acme", "business_website": None,
    "business_location": None, "niche": None, "poster_name": None, "poster_role": None,
    "poster_is_decision_maker": False, "emails_in_post": [], "summary": "needs SMM",
}


def test_ollama_parses_structured_output_with_an_inlined_schema():
    session = OllamaSession(json.dumps(VALID))
    out = OllamaLLM("http://ollama:11434/", "qwen2.5:7b", session=session).parse(IntentAssessment, system="s", user="u")
    assert out is not None and out.business_name == "Acme"
    body = session.bodies[0]
    assert body["stream"] is False and body["options"]["temperature"] == 0
    assert "$ref" not in json.dumps(body["format"]) and "$defs" not in body["format"]


def test_inline_refs_resolves_nested_models():
    schema = inline_refs(NicheReport.model_json_schema())
    assert "$ref" not in json.dumps(schema)
    assert "search_queries" in schema["properties"]["niches"]["items"]["properties"]


def test_ollama_failures():
    def llm(**kw):
        return OllamaLLM("http://o", "qwen2.5:7b", session=OllamaSession(**kw))

    with pytest.raises(LLMTransient):
        llm(exc=requests.ConnectionError("down")).parse(IntentAssessment, system="s", user="u")
    with pytest.raises(LLMTransient):
        llm(status=500).parse(IntentAssessment, system="s", user="u")
    assert llm(status=400).parse(IntentAssessment, system="s", user="u") is None
    assert llm(content='{"intent_type": "maybe"}').parse(IntentAssessment, system="s", user="u") is None
    assert llm().research(system="s", user="u") is None


def test_ollama_preflight():
    OllamaLLM("http://o", "qwen2.5:7b", session=OllamaSession()).preflight()
    OllamaLLM("http://o", "llama3.2", session=OllamaSession(models=("llama3.2:latest",))).preflight()
    with pytest.raises(LLMUnavailable, match="ollama pull"):
        OllamaLLM("http://o", "qwen2.5:7b", session=OllamaSession(models=())).preflight()
    with pytest.raises(LLMUnavailable, match="not reachable"):
        OllamaLLM("http://o", "qwen2.5:7b", session=OllamaSession(exc=requests.ConnectionError())).preflight()


# -- niches without web research ----------------------------------------------------------


def test_template_niche_queries():
    t = template_niche("newly opened gyms and fitness studios")
    assert len(t.search_queries) == 3
    assert all('"grand opening"' in q and "(gym OR fitness OR studio)" in q for q in t.search_queries)
    assert any("site:instagram.com" in q for q in t.search_queries)


def test_free_mode_niches_come_from_seeds(tmp_path):
    cfg = Config()
    cfg.niches.seed_niches = ["new bakeries", "new med spas"]
    niches = NicheResearcher(cfg, None, Store(str(tmp_path / "db.sqlite"))).get()
    assert [n.name for n in niches] == ["new bakeries", "new med spas"]


def test_local_model_down_falls_back_to_templates(tmp_path):
    cfg = Config()
    cfg.niches.seed_niches = ["new bakeries"]
    down = OllamaLLM("http://o", "m", session=OllamaSession(exc=requests.ConnectionError()))
    niches = NicheResearcher(cfg, down, Store(str(tmp_path / "db.sqlite"))).get()
    assert niches and niches[0].name == "new bakeries" and niches[0].search_queries


# -- SearXNG ------------------------------------------------------------------------------


class SearxApi:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get_json(self, url, params=None, **kw):
        self.calls.append((url, dict(params)))
        return {"results": self.pages[len(self.calls) - 1] if len(self.calls) <= len(self.pages) else []}


def _results(n, start=0):
    return [{"url": f"https://www.instagram.com/p/{i}/", "title": f"<b>Post {i}</b>", "content": "We just opened!",
             "publishedDate": "2026-10-05T10:00:00"} for i in range(start, start + n)]


def test_searxng_pages_until_limit_and_restricts_time():
    api = SearxApi([_results(10), _results(10, 10), _results(10, 20)])
    hits = SearxngSearch("http://127.0.0.1:8888/", api).search('"now open" gym', WINDOW, 15)
    assert len(hits) == 15 and len(api.calls) == 2
    url, params = api.calls[0]
    assert url == "http://127.0.0.1:8888/search" and params["format"] == "json" and params["time_range"] == "month"
    assert hits[0].title == "Post 0" and hits[0].date_text == "2026-10-05T10:00:00"
    assert api.calls[1][1]["pageno"] == 2


def test_searxng_stops_on_empty_page_and_caps_pages():
    api = SearxApi([_results(3)])
    assert len(SearxngSearch("http://s", api).search("q", None, 50)) == 3 and len(api.calls) == 2
    api = SearxApi([_results(1, i) for i in range(10)])
    SearxngSearch("http://s", api).search("q", Window.from_days(1, 0, now=NOW), 50)
    assert len(api.calls) == SearxngSearch.MAX_PAGES and api.calls[0][1]["time_range"] == "day"


# -- choosing the brain and search backend ------------------------------------------------


def test_factory_picks_free_components():
    cfg = Config()
    cfg.llm.provider = "none"
    assert make_llm(cfg, Secrets()) is None
    cfg.llm.provider = "ollama"
    assert isinstance(make_llm(cfg, Secrets()), OllamaLLM)
    cfg.sources.search.searxng_url = "http://127.0.0.1:8888"
    assert make_search_backend(cfg, Secrets(), None, None).name == "searxng"
    assert make_search_backend(cfg, Secrets(brave_api_key="k"), None, None).name == "brave"  # auto prefers a key


def test_factory_without_search_or_with_bad_choice():
    cfg = Config()
    assert make_search_backend(cfg, Secrets(), None, None) is None  # Reddit/RSS only
    cfg.sources.search.provider = "searxng"
    with pytest.raises(SetupError, match="searxng_url"):
        make_search_backend(cfg, Secrets(), None, None)
    cfg.sources.search.provider = "claude"
    with pytest.raises(SetupError, match="llm.provider: claude"):
        make_search_backend(cfg, Secrets(), None, OllamaLLM("http://o", "m"))
    claude = LLM(LLMConfig(), client=mock_client(lambda b: (200, {}), []))
    assert make_search_backend(cfg, Secrets(), None, claude).name == "claude"


# -- DNS over HTTPS -----------------------------------------------------------------------


class DohSession:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc, self.calls = payload, exc, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(params["name"])
        if self.exc:
            raise self.exc
        return FakeResp(200, self.payload)


def test_doh_parses_answers():
    ok = DohSession({"Status": 0, "Answer": [
        {"type": 5, "data": "alias.example.com."},
        {"type": 15, "data": "20 alt.mx.example.com."},
        {"type": 15, "data": "10 MX.Example.com."},
    ]})
    assert lookup_mx_doh("example.com", session=ok).hosts == ["mx.example.com", "alt.mx.example.com"]
    assert lookup_mx_doh("x.com", session=DohSession({"Status": 3})).status == "nxdomain"
    assert lookup_mx_doh("x.com", session=DohSession({"Status": 0})).status == "no_mx"
    assert lookup_mx_doh("x.com", session=DohSession({"Status": 0, "Answer": [{"type": 15, "data": "0 ."}]})).status == "null_mx"
    assert lookup_mx_doh("x.com", session=DohSession({"Status": 2})).status == "error"
    assert lookup_mx_doh("x.com", session=DohSession(exc=requests.ConnectionError())).status == "error"


def test_mx_lookup_modes(monkeypatch):
    import leadgen.verify.dns_check as dns_check

    local = {"status": "nxdomain"}
    monkeypatch.setattr(dns_check, "lookup_mx", lambda d, t=8.0: MxResult(local["status"], ["mx.local"] if local["status"] == "ok" else []))
    doh = DohSession({"Status": 0, "Answer": [{"type": 15, "data": "10 mx.remote."}]})

    # A filtering resolver says NXDOMAIN for everything: DNS-over-HTTPS gets a second opinion.
    assert MxLookup("auto", session=doh)("acme.com").hosts == ["mx.remote"]
    assert MxLookup("system", session=doh)("acme.com").status == "nxdomain"
    assert MxLookup("doh", session=doh)("acme.com").hosts == ["mx.remote"]
    local["status"] = "ok"
    calls = len(doh.calls)
    assert MxLookup("auto", session=doh)("acme.com").hosts == ["mx.local"] and len(doh.calls) == calls
    # DoH unreachable: keep the local answer (the verifier's canary then decides whether to trust it).
    local["status"] = "error"
    assert MxLookup("auto", session=DohSession(exc=requests.ConnectionError()))("acme.com").status == "error"


# -- doctor in free mode ------------------------------------------------------------------


def free_doctor(tmp_path, routes, mx="ok", **cfg_over):
    cfg = good_config(tmp_path)
    cfg.llm.provider = cfg_over.get("llm", "none")
    cfg.sources.search.searxng_url = cfg_over.get("searxng_url")
    cfg.verification.smtp = cfg_over.get("smtp", True)
    return Doctor(cfg, Secrets(**cfg_over.get("secrets", {})), session=FakeSession(routes),
                  mx_lookup=lambda d: MxResult(mx, ["gmail-smtp-in.l.google.com"] if mx == "ok" else [], "no DNS"),
                  connect=lambda h, p, t: b"220 ok", resolve_host=lambda h: True)


def test_doctor_free_setup_reports_no_cost(tmp_path):
    searx = {"http://s/search": FakeResp(200, {"results": [{"url": "u"}]})}
    c = by_name(free_doctor(tmp_path, searx, searxng_url="http://s").run())
    assert c["Brain"].status == "ok" and "rules" in c["Brain"].detail
    assert c["Post search (SearXNG)"].status == "ok"
    assert c["Running cost"].detail.startswith("free")
    assert not any(ch.status == "fail" for ch in c.values())


def test_doctor_searxng_problems(tmp_path):
    def check(resp):
        return by_name(free_doctor(tmp_path, {"http://s/search": resp}, searxng_url="http://s").run())["Post search (SearXNG)"]

    assert check(FakeResp(403)).status == "fail" and "json" in check(FakeResp(403)).fix
    blocked = check(FakeResp(200, {"results": [], "unresponsive_engines": [["google", "HTTP connection error"]]}))
    assert blocked.status == "warn" and "google" in blocked.detail
    assert by_name(free_doctor(tmp_path, {}, searxng_url="http://s").run())["Post search (SearXNG)"].status == "fail"


def test_doctor_ollama(tmp_path):
    tags = {"http://localhost:11434/api/tags": FakeResp(200, {"models": [{"name": "qwen2.5:7b"}]})}
    assert by_name(free_doctor(tmp_path, tags, llm="ollama").run())["Local model (Ollama)"].status == "ok"
    missing = {"http://localhost:11434/api/tags": FakeResp(200, {"models": []})}
    c = by_name(free_doctor(tmp_path, missing, llm="ollama").run())["Local model (Ollama)"]
    assert c.status == "fail" and "ollama pull" in c.fix
    assert by_name(free_doctor(tmp_path, {}, llm="ollama").run())["Local model (Ollama)"].status == "fail"


def test_doctor_dns_failure_is_only_a_warning_when_the_api_checks_domains(tmp_path):
    zb = {"https://api.zerobounce.net/v2/getcredits": FakeResp(200, {"Credits": "100"})}
    c = by_name(free_doctor(tmp_path, zb, mx="error", smtp=False, secrets={"zerobounce_api_key": "k"}).run())
    assert c["DNS resolver"].status == "warn" and c["Email verification"].status == "ok"
    assert "ZeroBounce" in c["Running cost"].detail
    c = by_name(free_doctor(tmp_path, {}, mx="error").run())
    assert c["DNS resolver"].status == "fail"


def test_cloud_and_free_configs_load():
    from pathlib import Path

    from leadgen.config import load_config

    root = Path(__file__).resolve().parent.parent
    cloud = load_config(root / "config.cloud.yaml")
    assert (cloud.llm.provider, cloud.verification.smtp, cloud.verification.dns) == ("none", False, "doh")
    free = load_config(root / "config.free.yaml")
    assert free.llm.provider == "ollama" and free.sources.search.provider == "searxng"
    assert (free.window.min_age_days, free.window.max_age_days) == (2, 14)


def test_rules_owner_phrases():
    def owner(text):
        return assess_without_ai(post("Looking for a social media manager for " + text), Config()).poster_is_decision_maker

    assert owner("my new pilates studio") and owner("my gym") and owner("my own business")
    assert not owner("my friend's gym") and not owner("my favorite coffee shop") and not owner("my sister's new salon")
