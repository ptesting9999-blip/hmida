"""Regression tests for the issues found by the completeness audit."""

import smtplib
from datetime import timedelta

import pytest
from test_pipeline import NOW, FakeProber, assessment, li_post, make_agent
from test_verifier import FakeProvider
from test_verifier import FakeProber as VFakeProber
from test_verifier import make as make_verifier

from leadgen.compliance import Suppression
from leadgen.config import Config, SearchSourceConfig, VerificationConfig
from leadgen.llm import LLMUnavailable
from leadgen.models import Lead
from leadgen.verify import rules
from leadgen.verify.dns_check import MxResult
from leadgen.verify.providers import ProviderVerdict
from leadgen.verify.smtp_check import RcptResult, SmtpProbe, SmtpProber, classify_rcpt

# -- suppression ------------------------------------------------------------------------


def _suppressed_agent(tmp_path, posts, by_url, mailboxes=("sarah@acmefit.com",)):
    agent, llm = make_agent(tmp_path, posts, by_url, mailboxes=mailboxes)
    agent.verifier.suppression = Suppression(tmp_path / "suppression.txt")
    return agent, llm


def test_opt_out_added_while_agent_is_running_is_honoured(tmp_path):
    post = li_post("https://www.linkedin.com/posts/s_a-activity-41", 3)
    agent, _ = _suppressed_agent(tmp_path, [post], {post.url: assessment()})
    Suppression(tmp_path / "suppression.txt").add("sarah@acmefit.com")  # what `leadgen suppress` does
    report = agent.run(now=NOW)
    assert report.leads == []


def test_suppressed_business_domain_blocks_owner_personal_email(tmp_path):
    post = li_post("https://www.linkedin.com/posts/s_a-activity-42", 3,
                   text="Acme Fitness (acmefit.com) needs a social media manager! Email me: sarah.lee@gmail.com")
    a = assessment(emails_in_post=["sarah.lee@gmail.com"])
    agent, _ = _suppressed_agent(tmp_path, [post], {post.url: a}, mailboxes=("sarah.lee@gmail.com",))
    (tmp_path / "suppression.txt").write_text("acmefit.com\n")
    report = agent.run(now=NOW)
    assert report.leads == []
    assert report.rejections[-1].stage == "suppressed"


def test_export_rows_filter_suppressed_emails_and_domains(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("jane@acme.com\nblocked.com\n")
    rows = [
        {"verified_email": "jane@acme.com", "business_website": "https://acme.com"},
        {"verified_email": "owner@gmail.com", "business_website": "https://www.blocked.com"},
        {"verified_email": "ok@fine.com", "business_website": "https://fine.com"},
    ]
    assert [r["verified_email"] for r in Suppression(f).filter_rows(rows)] == ["ok@fine.com"]


# -- window / config --------------------------------------------------------------------


def test_default_window_is_2_to_14_days(tmp_path):
    assert Config().window.min_age_days == 2 and Config().window.max_age_days == 14
    fresh = li_post("https://www.linkedin.com/posts/s_a-activity-43", 1)
    agent, llm = make_agent(tmp_path, [fresh], {fresh.url: assessment()})
    report = agent.run(now=NOW)
    assert report.out_of_window == 1 and llm.calls == 0


def test_unknown_platform_fails_at_config_load():
    with pytest.raises(ValueError, match="unknown platform"):
        SearchSourceConfig(platforms=["linkedin", "tiktok"])


# -- SMTP classification and session handling -------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        b"Verification failed for <verify@agency.com>\nNo Such User Here\nSender verify failed",
        b"5.1.8 <verify@agency.com>: Sender address rejected: Domain not found",
        b"5.7.27 <verify@agency.com>: Sender address rejected: Domain has null MX",
        b"Service unavailable; Client host [5.1.22.33] blocked using zen.spamhaus.org",
        b"rejected because 185.5.1.20 is in a black list at zen.spamhaus.org",
        b"Access denied - Invalid HELO name (See RFC2821 4.1.1.1)",
    ],
)
def test_sender_and_host_blocks_are_not_mailbox_rejections(msg):
    assert classify_rcpt(550, msg) == "blocked"


def test_real_mailbox_rejection_still_detected():
    assert classify_rcpt(550, b"5.1.1 <x@acme.com>: Recipient address rejected: User unknown") == "rejected"
    assert classify_rcpt(550, b"5.1.1 The email account that you tried to reach does not exist.\n5.1.1 https://support.google.com") == "rejected"


class ScriptedSMTP:
    def __init__(self, rcpt_replies, mail_reply=(250, b"ok"), ehlo_reply=(250, b"hi")):
        self.rcpt_replies = list(rcpt_replies)
        self.mail_reply, self.ehlo_reply = mail_reply, ehlo_reply
        self.rcpts = []

    def __call__(self, timeout=None):
        return self

    def connect(self, host, port):
        return 220, b"ready"

    def ehlo(self, name):
        return self.ehlo_reply

    def helo(self, name):
        return self.ehlo_reply

    def mail(self, sender):
        return self.mail_reply

    def rcpt(self, addr):
        self.rcpts.append(addr)
        reply = self.rcpt_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def quit(self):
        return 221, b"bye"

    def close(self):
        pass


def test_421_mid_session_keeps_answers_and_marks_rest_retryable():
    fake = ScriptedSMTP([(550, b"5.1.1 no such user"), (421, b"4.7.0 too many errors, closing")])
    probe = SmtpProber("h", "v@agency.com", smtp_factory=fake).probe("acme.com", ["mx"], ["a@acme.com", "b@acme.com", "c@acme.com"])
    assert probe.results["a@acme.com"].outcome == "rejected"
    assert probe.results["b@acme.com"].outcome == "temp"
    assert probe.results["c@acme.com"].outcome == "temp"
    assert len(fake.rcpts) == 2  # no catch-all probe on a closing connection


def test_disconnect_mid_session_keeps_answers():
    fake = ScriptedSMTP([(250, b"ok"), smtplib.SMTPServerDisconnected("gone")])
    probe = SmtpProber("h", "v@agency.com", smtp_factory=fake).probe("acme.com", ["mx"], ["a@acme.com", "b@acme.com"])
    assert probe.results["a@acme.com"].outcome == "accepted"
    assert probe.results["b@acme.com"].outcome == "temp"
    assert probe.catch_all is None


def test_temporary_mail_from_reply_is_retryable():
    fake = ScriptedSMTP([], mail_reply=(451, b"4.7.1 Greylisting in action"))
    probe = SmtpProber("h", "v@agency.com", smtp_factory=fake).probe("acme.com", ["mx"], ["a@acme.com"])
    assert probe.results["a@acme.com"].outcome == "temp"


# -- verifier ---------------------------------------------------------------------------


def test_dns_errors_are_not_cached_and_reset_clears_state():
    calls = {"n": 0}

    def flaky(domain):
        calls["n"] += 1
        if domain == "acme.com" and calls["n"] == 1:
            return MxResult("error", detail="timeout")
        return MxResult("ok", [f"mx.{domain}"])

    v = make_verifier(prober=VFakeProber(accepted={"jane@acme.com"}), mx_lookup=flaky)
    first = v.verify("jane@acme.com")
    assert first.status == "unknown" and first.transient
    assert v.verify("jane@acme.com").status == "deliverable"


def test_broken_resolver_recovers_after_reset():
    state = {"broken": True}

    def lookup(domain):
        if state["broken"]:
            return MxResult("nxdomain", detail="nope")
        return MxResult("ok", [f"mx.{domain}"])

    v = make_verifier(prober=VFakeProber(accepted={"jane@acme.com"}), mx_lookup=lookup)
    assert v.verify("jane@acme.com").status == "unknown"
    state["broken"] = False
    v.reset()
    assert v.verify("jane@acme.com").status == "deliverable"


def test_provider_catch_all_stops_paying_for_more_guesses():
    provider = FakeProvider({"a@acme.com": ProviderVerdict("risky", "catch-all", catch_all=True)})
    v = make_verifier(VerificationConfig(smtp=False), provider=provider)
    v.verify_many(["a@acme.com", "b@acme.com", "c@acme.com"], stop_at_first=True)
    assert provider.calls == ["a@acme.com"]


def test_activity_is_reported_even_when_unknown():
    v = make_verifier(prober=VFakeProber(accepted={"jane@acme.com"}))
    r = v.verify("jane@acme.com")
    assert r.deliverable and r.checks["activity"].startswith("unknown") and "activity" in r.reason
    zb = FakeProvider({"jane@acme.com": ProviderVerdict("deliverable", "valid", inactive_days=30)})
    r2 = make_verifier(VerificationConfig(smtp=False), provider=zb).verify("jane@acme.com")
    assert "active within 30 days" in r2.reason


def test_provider_decides_when_local_dns_is_unavailable():
    provider = FakeProvider({"jane@acme.com": ProviderVerdict("deliverable", "valid")})
    v = make_verifier(VerificationConfig(smtp=False), provider=provider, mx_lookup=lambda d: MxResult("error", detail="no DNS"))
    assert v.verify("jane@acme.com").status == "deliverable"


def test_provider_request_failure_is_transient():
    provider = FakeProvider({"jane@acme.com": ProviderVerdict("unknown", "request failed", failed=True)})
    v = make_verifier(VerificationConfig(smtp=False), provider=provider)
    r = v.verify("jane@acme.com")
    assert r.status == "unknown" and r.transient


@pytest.mark.parametrize(
    "email",
    ["marketingteam@a.com", "salesteam@a.com", "infodesk@a.com", "info.nyc@a.com", "hello-london@a.com",
     "bookings.miami@a.com", "contactme@a.com", "hola@a.com", "acme.info@acme.com", "mailer-daemon@a.com"],
)
def test_more_generic_inboxes_are_role_accounts(email):
    assert rules.is_role_account(email)


@pytest.mark.parametrize("email", ["dev@a.com", "dev.patel@a.com", "jane.sales@a.com", "mike.info@acme.com"])
def test_people_are_not_role_accounts(email):
    assert not rules.is_role_account(email)


# -- pipeline durability ----------------------------------------------------------------


class TransientProber(FakeProber):
    def probe(self, domain, hosts, emails):
        return SmtpProbe(host=None, catch_all=None, results={e: RcptResult("error", 0, "connection refused") for e in emails})


def test_outage_does_not_discard_post_but_retries_are_capped(tmp_path):
    post = li_post("https://www.linkedin.com/posts/s_a-activity-44", 3)
    agent, llm = make_agent(tmp_path, [post], {post.url: assessment()})
    agent.verifier.prober = TransientProber()
    for expected_calls in (1, 2, 3):
        report = agent.run(now=NOW)
        assert report.rejections[-1].stage == "retry"
        assert llm.calls == expected_calls
    report = agent.run(now=NOW)  # gave up after 3 attempts
    assert report.already_seen == 1 and llm.calls == 3


def test_leads_saved_by_an_interrupted_run_are_delivered_next_run(tmp_path):
    agent, _ = make_agent(tmp_path, [], {})
    stranded = Lead(
        email="owner@stranded.com", contact_name="Owner", contact_role="Founder", business_name="Stranded",
        business_website="https://stranded.com", platform="LinkedIn", intent_type="explicit_request",
        intent_url="https://www.linkedin.com/posts/x", posted_at=NOW - timedelta(days=3), date_source="post_id",
        services_needed="paid_ads_management", niche="", location="", intent_score=90, summary="s",
        post_excerpt="p", email_source="website", verification="confirmed by smtp", found_at=NOW,
    )
    agent.store.save_lead(stranded, "stranded.com")  # saved, then the process was killed
    agent.run(now=NOW)
    files = list((tmp_path / "out").glob("leads_2*.csv"))
    assert files and "owner@stranded.com" in files[0].read_text()
    assert agent.store.unexported_leads() == []


def test_bad_key_fails_before_any_source_is_queried(tmp_path):
    class Source:
        name = "s"
        called = False

        def collect(self, window, niches):
            Source.called = True
            return iter([])

    agent, _ = make_agent(tmp_path, [], {})
    agent.sources = [Source()]

    def preflight():
        raise LLMUnavailable("key rejected")

    agent.preflight = preflight
    with pytest.raises(LLMUnavailable):
        agent.run(now=NOW)
    assert Source.called is False


def test_guessed_website_is_ignored(tmp_path):
    post = li_post("https://www.linkedin.com/posts/s_a-activity-45", 3,
                   text="We just opened our gym and need a social media manager for Instagram ads.")
    agent, _ = make_agent(tmp_path, [post], {post.url: assessment(business_website="acmefit.com")})
    report = agent.run(now=NOW)
    assert report.leads == []  # acmefit.com was a guess: not in the post, and no search backend here
    assert "no business website" in report.rejections[-1].reason or report.rejections[-1].stage == "verify"


# -- CLI --------------------------------------------------------------------------------


def test_every_mode_survives_bad_key(monkeypatch, tmp_path):
    from leadgen import cli

    class Agent:
        runs = 0

        def run(self, limit=None):
            Agent.runs += 1
            raise LLMUnavailable("key rejected")

    def stop(_seconds):
        raise KeyboardInterrupt

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_agent", lambda cfg, secrets: Agent())
    monkeypatch.setattr(cli.time, "sleep", stop)
    assert cli.main(["run", "--every", "6"]) == 130  # stopped by us, not crashed by the bad key
    assert Agent.runs == 1


def test_config_yaml_folder_gives_clear_error(monkeypatch, tmp_path, capsys):
    from leadgen import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").mkdir()
    assert cli.main(["export"]) == 2
    assert "is a folder" in capsys.readouterr().err
