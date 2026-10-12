import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from leadgen.config import Config, load_config
from leadgen.dates import UTC
from leadgen.enrich import ContactFinder
from leadgen.enrich.website import SitePage, SiteProfile
from leadgen.intent import IntentAssessment, IntentClassifier
from leadgen.models import Person, RawPost
from leadgen.output import OutputWriter, _safe_cell
from leadgen.pipeline import LeadAgent
from leadgen.store import Store
from leadgen.verify.dns_check import MxResult
from leadgen.verify.smtp_check import RcptResult, SmtpProbe
from leadgen.verify.verifier import EmailVerifier

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)


def assessment(**over):
    base = dict(
        intent_type="explicit_request",
        author_is_service_provider=False,
        intent_score=90,
        services_needed=["social_media_management", "paid_ads_management"],
        business_name="Acme Fitness",
        business_website="acmefit.com",
        business_location="Austin, TX",
        niche="boutique gym",
        poster_name="Sarah Lee",
        poster_role="Founder",
        poster_is_decision_maker=True,
        emails_in_post=[],
        summary="New gym needs Instagram management and Meta ads.",
    )
    base.update(over)
    return IntentAssessment(**base)


class FakeSource:
    name = "fake"

    def __init__(self, posts):
        self.posts = posts

    def collect(self, window, niches):
        return iter(self.posts)


class FakeLLM:
    """Stands in for Claude: one canned assessment per post URL."""

    def __init__(self, by_url):
        self.by_url = by_url
        self.calls = 0

    def parse(self, schema, *, system, user, effort, max_tokens=8000):
        self.calls += 1
        for url, a in self.by_url.items():
            if url in user:
                return a
        return None


class FakeCrawler:
    def crawl(self, domain):
        if domain == "acmefit.com":
            return SiteProfile(
                domain=domain,
                base_url="https://acmefit.com/",
                pages=[SitePage("https://acmefit.com/about", "Sarah Lee, Founder & Head Coach.")],
                emails={"info@acmefit.com"},
                people=[Person("Sarah Lee", "Founder", source="jsonld")],
            )
        return SiteProfile(domain=domain)


class FakeProber:
    def __init__(self, mailboxes=("sarah@acmefit.com",)):
        self.mailboxes = set(mailboxes)

    def probe(self, domain, hosts, emails):
        results = {
            e: RcptResult("accepted", 250, "ok") if e in self.mailboxes else RcptResult("rejected", 550, "5.1.1 no such user")
            for e in emails
        }
        return SmtpProbe(host=hosts[0], catch_all=domain == "catchall.com", results=results)


def make_agent(tmp_path, posts, by_url, mailboxes=("sarah@acmefit.com",)):
    cfg = Config()
    cfg.output.dir = str(tmp_path / "out")
    cfg.workers = 2
    store = Store(tmp_path / "db.sqlite")
    llm = FakeLLM(by_url)
    verifier = EmailVerifier(
        cfg.verification,
        store=store,
        prober=FakeProber(mailboxes),
        mx_lookup=lambda d: MxResult("ok", [f"mx.{d}"]),
        disposable=frozenset(),
    )
    agent = LeadAgent(
        cfg,
        sources=[FakeSource(posts)],
        classifier=IntentClassifier(cfg, llm),
        finder=ContactFinder(cfg, crawler=FakeCrawler(), llm=None, hunter=None, lookup_backend=None),
        verifier=verifier,
        store=store,
        writer=OutputWriter(cfg.output),
    )
    return agent, llm


def li_post(url, days_ago, text="We just opened Acme Fitness and need a social media manager to run our Instagram and Meta ads."):
    return RawPost(
        source="fake",
        platform="linkedin",
        url=url,
        title="Sarah Lee on LinkedIn",
        text=text,
        posted_at=NOW - timedelta(days=days_ago),
        date_source="post_id",
    )


def test_end_to_end(tmp_path):
    good = li_post("https://www.linkedin.com/posts/sarah-lee_good-activity-1", 3)
    stale = li_post("https://www.linkedin.com/posts/sarah-lee_old-activity-2", 20)
    undated = li_post("https://www.linkedin.com/posts/x_undated-activity-3", 3)
    undated.posted_at = None
    seller = li_post("https://www.linkedin.com/posts/y_seller-activity-4", 2, text="I help brands grow on Instagram. DM me for rates on social media management.")
    agency = li_post("https://www.linkedin.com/posts/z_agency-activity-5", 2)
    catch_all = li_post("https://www.linkedin.com/posts/w_catch-activity-6", 4)

    agent, llm = make_agent(
        tmp_path,
        [good, stale, undated, seller, agency, catch_all],
        {
            good.url: assessment(),
            agency.url: assessment(author_is_service_provider=True),
            catch_all.url: assessment(business_name="Catch All Co", business_website="catchall.com", poster_name="Tom Fox"),
        },
    )
    report = agent.run(now=NOW)

    assert [lead.email for lead in report.leads] == ["sarah@acmefit.com"]
    lead = report.leads[0]
    assert lead.contact_name == "Sarah Lee" and lead.platform == "LinkedIn"
    assert lead.posted_at == NOW - timedelta(days=3) and lead.intent_url == good.url
    assert report.out_of_window == 2 and report.prefiltered == 1
    assert llm.calls == 3  # only posts that survive the cheap filters reach Claude

    stages = {r.url: r.stage for r in report.rejections}
    assert stages[agency.url] == "intent"
    assert stages[catch_all.url] == "verify"  # catch-all domain: nothing confirmed

    out = Path(tmp_path / "out")
    rows = list(csv.DictReader(next(out.glob("leads_2*.csv")).open()))
    assert rows[0]["verified_email"] == "sarah@acmefit.com"
    assert rows[0]["posted_date"] == "2026-10-09"
    assert rows[0]["platform"] == "LinkedIn"
    assert json.loads(next(out.glob("leads_*.json")).read_text())[0]["business_name"] == "Acme Fitness"
    assert next(out.glob("rejected_*.csv")).exists()

    # Second run: nothing is re-processed or re-delivered.
    report2 = agent.run(now=NOW)
    assert report2.leads == []
    assert report2.already_seen >= 4
    assert llm.calls == 3


def test_safe_cell_blocks_formula_injection():
    assert _safe_cell("=HYPERLINK(\"x\")") == "'=HYPERLINK(\"x\")"
    assert _safe_cell("normal") == "normal"
    assert _safe_cell(5) == 5


def test_example_config_is_valid():
    cfg = load_config(Path(__file__).parent.parent / "config.example.yaml")
    assert cfg.window.max_age_days == 14
    assert cfg.llm.model == "claude-opus-5-5"


def test_free_mail_owners_are_not_deduplicated_by_provider(tmp_path):
    a = li_post("https://www.linkedin.com/posts/a_x-activity-11", 2, text="Opening my bakery, need a social media manager! Email dana.reyes@gmail.com")
    b = li_post("https://www.linkedin.com/posts/b_x-activity-12", 2, text="New salon here, need someone for Instagram ads. Contact: lee.park@gmail.com")
    agent, _ = make_agent(
        tmp_path,
        [a, b],
        {
            a.url: assessment(business_name="Dana's Bakery", business_website=None, poster_name="Dana Reyes", emails_in_post=["dana.reyes@gmail.com"]),
            b.url: assessment(business_name="Lee Salon", business_website=None, poster_name="Lee Park", emails_in_post=["lee.park@gmail.com"]),
        },
        mailboxes={"dana.reyes@gmail.com", "lee.park@gmail.com"},
    )
    report = agent.run(now=NOW)
    assert sorted(lead.email for lead in report.leads) == ["dana.reyes@gmail.com", "lee.park@gmail.com"]
