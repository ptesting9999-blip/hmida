from types import SimpleNamespace

import anthropic
import httpx2
import requests

from leadgen.config import Config, Secrets
from leadgen.doctor import Doctor, format_checks
from leadgen.verify.dns_check import MxResult


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self.payload = status, payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def _hit(self, url):
        self.calls.append(url)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        raise requests.ConnectionError(f"no route for {url}")

    def get(self, url, **kw):
        return self._hit(url)

    def post(self, url, **kw):
        return self._hit(url)


def fake_claude(error=None):
    def retrieve(model_id):
        if error:
            raise error
        return SimpleNamespace(id=model_id)

    return SimpleNamespace(models=SimpleNamespace(retrieve=retrieve))


def good_config(tmp_path):
    cfg = Config()
    cfg.output.dir = str(tmp_path / "out")
    cfg.db_path = str(tmp_path / "data" / "db.sqlite")
    cfg.verification.smtp_from = "verify@agency.com"
    cfg.verification.helo_host = "mail.agency.com"
    return cfg


def make(tmp_path, routes, secrets, claude=None, banner=b"220 mx.google.com ESMTP", mx="ok"):
    def connect(host, port, timeout):
        if isinstance(banner, Exception):
            raise banner
        return banner

    return Doctor(
        good_config(tmp_path),
        secrets,
        session=FakeSession(routes),
        claude=claude or fake_claude(),
        mx_lookup=lambda d: MxResult(mx, ["gmail-smtp-in.l.google.com"] if mx == "ok" else [], "boom"),
        connect=connect,
        resolve_host=lambda h: True,
    )


def by_name(checks):
    return {c.name: c for c in checks}


def test_everything_configured(tmp_path):
    secrets = Secrets(brave_api_key="b", reddit_client_id="i", reddit_client_secret="s", hunter_api_key="h", zerobounce_api_key="z")
    routes = {
        "https://api.search.brave.com": FakeResp(200, {"web": {"results": []}}),
        "https://www.reddit.com/api/v1/access_token": FakeResp(200, {"access_token": "t"}),
        "https://api.hunter.io/v2/account": FakeResp(200, {"data": {"plan_name": "Starter"}}),
        "https://api.zerobounce.net/v2/getcredits": FakeResp(200, {"Credits": "2500"}),
    }
    checks = make(tmp_path, routes, secrets).run()
    assert all(c.status == "ok" for c in checks), format_checks(checks)
    c = by_name(checks)
    assert "SMTP + ZeroBounce" in c["Email verification"].detail
    assert "2500 credits" in c["ZeroBounce"].detail
    assert "Ready to run." in format_checks(checks)


def test_port_25_blocked_without_provider_fails(tmp_path):
    checks = make(tmp_path, {}, Secrets(), banner=TimeoutError()).run()
    c = by_name(checks)
    assert c["SMTP port 25"].status == "fail"
    assert c["Email verification"].status == "fail"
    assert "must be fixed" in format_checks(checks)


def test_port_25_blocked_but_provider_works(tmp_path):
    routes = {"https://api.neverbounce.com/v4/account/info": FakeResp(200, {"status": "success", "credits_info": {"paid_credits_remaining": 10}})}
    checks = make(tmp_path, routes, Secrets(neverbounce_api_key="n"), banner=ConnectionRefusedError()).run()
    c = by_name(checks)
    assert c["SMTP port 25"].status == "fail"
    assert c["Email verification"].status == "ok" and "NeverBounce" in c["Email verification"].detail


def test_bad_keys_are_reported(tmp_path):
    request = httpx2.Request("GET", "https://api.anthropic.com/v1/models/x")
    auth_error = anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=request), body=None)
    routes = {"https://api.zerobounce.net/v2/getcredits": FakeResp(200, {"Credits": -1})}
    checks = make(tmp_path, routes, Secrets(zerobounce_api_key="z"), claude=fake_claude(auth_error)).run()
    c = by_name(checks)
    assert c["Claude API"].status == "fail"
    assert c["ZeroBounce"].status == "fail" and c["ZeroBounce"].detail == "key rejected"


def test_broken_dns_and_placeholder_smtp_identity(tmp_path):
    d = make(tmp_path, {}, Secrets(), mx="error")
    d.cfg.verification.smtp_from = "verify@example.com"
    c = by_name(d.run())
    assert c["DNS resolver"].status == "fail"
    assert c["SMTP sender"].status == "warn"
    assert c["Post search"].status == "warn"  # Claude web search fallback
    assert c["Reddit"].status == "fail" and "anonymous" in c["Reddit"].detail  # no route: Reddit unreachable
    assert c["Post sources"].status == "fail"  # neither search nor Reddit can collect anything


def test_anonymous_reddit_that_answers_is_a_working_source(tmp_path):
    routes = {"https://www.reddit.com/r/forhire/new.json": FakeResp(200, {"data": {"children": []}})}
    c = by_name(make(tmp_path, routes, Secrets()).run())
    assert c["Reddit"].status == "ok" and c["Post sources"].status == "ok" and "Reddit" in c["Post sources"].detail


def test_missing_anthropic_credentials(tmp_path, monkeypatch):
    import pytest

    from leadgen.llm import LLM, LLMUnavailable

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile either
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(LLMUnavailable):
        LLM(Config().llm)
    d = Doctor(good_config(tmp_path), Secrets(), session=FakeSession({}), mx_lookup=lambda x: MxResult("ok", ["mx"]),
               connect=lambda h, p, t: b"220 ok", resolve_host=lambda h: True)
    claude_check = d.claude()
    assert claude_check.status == "fail" and "no Anthropic credentials" in claude_check.detail
