from leadgen.compliance import Suppression
from leadgen.config import VerificationConfig
from leadgen.store import Store
from leadgen.verify.dns_check import MxResult
from leadgen.verify.providers import ProviderVerdict
from leadgen.verify.smtp_check import RcptResult, SmtpProbe
from leadgen.verify.verifier import EmailVerifier


class FakeProber:
    def __init__(self, accepted=(), catch_all=False, outcome_for_others="rejected"):
        self.accepted = set(accepted)
        self.catch_all = catch_all
        self.other = outcome_for_others
        self.calls = []

    def probe(self, domain, hosts, emails):
        self.calls.append((domain, list(emails)))
        results = {
            e: RcptResult("accepted", 250, "ok") if e in self.accepted else RcptResult(self.other, 550, "5.1.1 unknown")
            for e in emails
        }
        return SmtpProbe(host=hosts[0], catch_all=self.catch_all, results=results)


class FakeProvider:
    name = "fakeverifier"

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []

    def check(self, email):
        self.calls.append(email)
        return self.verdicts.get(email, ProviderVerdict("unknown", "no data"))


def mx_ok(domain):
    return MxResult("ok", hosts=[f"mx.{domain}"])


def make(cfg=None, **kw):
    kw.setdefault("mx_lookup", mx_ok)
    kw.setdefault("disposable", frozenset({"mailinator.com"}))
    return EmailVerifier(cfg or VerificationConfig(), **kw)


def test_smtp_confirmed_mailbox_is_deliverable():
    v = make(prober=FakeProber(accepted={"jane@acme.com"}))
    r = v.verify("jane@acme.com")
    assert r.status == "deliverable"
    assert "smtp" in r.reason


def test_catch_all_is_rejected_even_when_accepted():
    v = make(prober=FakeProber(accepted={"jane@acme.com"}, catch_all=True))
    r = v.verify("jane@acme.com")
    assert r.status == "risky"
    assert "catch-all" in r.reason


def test_inconclusive_catch_all_is_not_deliverable():
    v = make(prober=FakeProber(accepted={"jane@acme.com"}, catch_all=None))
    assert v.verify("jane@acme.com").status == "unknown"


def test_smtp_rejection_skips_paid_provider():
    provider = FakeProvider({})
    v = make(prober=FakeProber(), provider=provider)
    assert v.verify("ghost@acme.com").status == "undeliverable"
    assert provider.calls == []


def test_provider_confirms_when_smtp_blocked():
    provider = FakeProvider({"jane@acme.com": ProviderVerdict("deliverable", "valid")})
    v = make(prober=FakeProber(outcome_for_others="blocked"), provider=provider)
    assert v.verify("jane@acme.com").status == "deliverable"


def test_all_mode_requires_every_method():
    provider = FakeProvider({"jane@acme.com": ProviderVerdict("unknown", "timeout")})
    cfg = VerificationConfig(confirmations_required="all")
    v = make(cfg, prober=FakeProber(accepted={"jane@acme.com"}), provider=provider)
    assert v.verify("jane@acme.com").status == "unknown"


def test_provider_negative_overrides_smtp_positive():
    provider = FakeProvider({"jane@acme.com": ProviderVerdict("risky", "catch-all")})
    v = make(prober=FakeProber(accepted={"jane@acme.com"}), provider=provider)
    assert v.verify("jane@acme.com").status == "risky"


def test_dormant_mailbox_rejected():
    provider = FakeProvider({"jane@acme.com": ProviderVerdict("deliverable", "valid", inactive_days=366)})
    v = make(VerificationConfig(smtp=False, max_inactive_days=365), provider=provider)
    r = v.verify("jane@acme.com")
    assert r.status == "risky"
    assert "dormant" in r.reason


def test_policy_rejections_never_hit_network():
    prober = FakeProber(accepted={"info@acme.com", "x@mailinator.com"})
    v = make(prober=prober, suppression=Suppression())
    assert v.verify("info@acme.com").status == "risky"
    assert v.verify("x@mailinator.com").status == "undeliverable"
    assert v.verify("not-an-email").status == "undeliverable"
    assert prober.calls == []


def test_suppression_list(tmp_path):
    f = tmp_path / "suppress.txt"
    f.write_text("blocked.com\njane@acme.com\n")
    v = make(prober=FakeProber(accepted={"jane@acme.com", "bob@blocked.com"}), suppression=Suppression(f))
    assert v.verify("jane@acme.com").reason == "on suppression list"
    assert v.verify("bob@sub.blocked.com").reason == "on suppression list"


def test_free_mail_policy():
    v = make(VerificationConfig(allow_free_email_providers=False), prober=FakeProber(accepted={"jane@gmail.com"}))
    assert v.verify("jane@gmail.com").status == "risky"


def test_no_mx_is_undeliverable():
    v = make(prober=FakeProber(), mx_lookup=lambda d: MxResult("ok", ["mx.gmail.com"]) if d == "gmail.com" else MxResult("nxdomain", detail="domain does not exist"))
    assert v.verify("jane@nope-not-real.com").status == "undeliverable"


def test_broken_resolver_is_unknown_not_undeliverable():
    v = make(prober=FakeProber(), mx_lookup=lambda d: MxResult("nxdomain", detail="domain does not exist"))
    r = v.verify("jane@acme.com")
    assert r.status == "unknown"
    assert "resolver" in r.reason


def test_stop_at_first_shares_one_smtp_session_and_saves_credits():
    prober = FakeProber(accepted={"jane@acme.com", "jane.doe@acme.com"}, outcome_for_others="blocked")
    provider = FakeProvider(
        {"jane@acme.com": ProviderVerdict("deliverable", "valid"), "jane.doe@acme.com": ProviderVerdict("deliverable", "valid")}
    )
    v = make(VerificationConfig(confirmations_required="all"), prober=prober, provider=provider)
    results = v.verify_many(["jane@acme.com", "jane.doe@acme.com", "jdoe@acme.com"], stop_at_first=True)
    assert results["jane@acme.com"].status == "deliverable"
    assert results["jdoe@acme.com"].status == "unknown"
    assert len(prober.calls) == 1 and len(prober.calls[0][1]) == 3
    assert provider.calls == ["jane@acme.com"]


def test_results_are_cached(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    prober = FakeProber(accepted={"jane@acme.com"})
    make(prober=prober, store=store).verify("jane@acme.com")
    second = FakeProber()
    r = make(prober=second, store=store).verify("jane@acme.com")
    assert r.status == "deliverable"
    assert second.calls == []


def test_requires_some_method():
    v = make(prober=None, provider=None)
    assert v.verify("jane@acme.com").status == "unknown"


def test_catch_all_can_defer_to_provider_when_opted_in():
    provider = FakeProvider({"jane@yahoo.com": ProviderVerdict("deliverable", "valid")})
    prober = FakeProber(accepted={"jane@yahoo.com"}, catch_all=True)
    strict = make(prober=prober, provider=provider)
    assert strict.verify("jane@yahoo.com").status == "risky"
    lenient = make(VerificationConfig(trust_provider_on_catch_all=True), prober=prober, provider=provider)
    assert lenient.verify("jane@yahoo.com").status == "deliverable"
    both = make(VerificationConfig(trust_provider_on_catch_all=True, confirmations_required="all"), prober=prober, provider=provider)
    assert both.verify("jane@yahoo.com").status == "unknown"


def test_greylisting_is_marked_retryable():
    class GreyProber(FakeProber):
        def probe(self, domain, hosts, emails):
            return SmtpProbe(host="mx", catch_all=None, results={e: RcptResult("temp", 451, "4.7.1 greylisted") for e in emails}, catch_all_outcome="temp")

    r = make(prober=GreyProber()).verify("jane@acme.com")
    assert r.status == "unknown" and r.retryable


def test_blocked_probe_is_not_retryable():
    r = make(prober=FakeProber(outcome_for_others="blocked")).verify("jane@acme.com")
    assert r.status == "unknown" and not r.retryable
