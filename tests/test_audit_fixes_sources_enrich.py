"""Regression tests for the sources / enrichment / Claude-call issues found by the audit."""

import json
from datetime import timedelta
from types import SimpleNamespace

import requests
from bs4 import BeautifulSoup
from test_pipeline import NOW, assessment, li_post, make_agent
from test_sources import WINDOW, FakeApi

from leadgen.config import Config, RedditSourceConfig, SearchSourceConfig
from leadgen.dates import Window, date_bounds
from leadgen.enrich import BusinessContext, ContactFinder
from leadgen.enrich.domain import name_matches, normalize_domain
from leadgen.enrich.people import llm_people, title_weight
from leadgen.enrich.website import SitePage, SiteProfile, WebsiteCrawler, extract_jsonld_people
from leadgen.http import HttpClient, Page
from leadgen.intent import IntentClassifier, build_system_prompt, build_user_message, prefilter
from leadgen.llm import LLM, LLMTransient, WebHit, _collect_hits
from leadgen.models import Person, RawPost
from leadgen.sources.importer import ImportSource
from leadgen.sources.reddit import RedditSource
from leadgen.sources.rss import parse_feed
from leadgen.sources.search import BraveSearch, ClaudeWebSearch, SearchHit, SearchSource, is_post_url

# -- dates ------------------------------------------------------------------------------


def test_coarse_relative_dates_are_ranges():
    lo, hi = date_bounds("2 weeks ago", NOW)
    assert hi == NOW - timedelta(weeks=2) and lo < NOW - timedelta(days=20)
    w = Window.from_days(14, 0, now=NOW)
    assert not w.contains_range(lo, hi)  # could be 20 days old: rejected
    assert w.contains_range(*date_bounds("1w", NOW))  # 7-13.99 days: fine
    assert date_bounds("3 hours ago", NOW) == (NOW - timedelta(hours=3),) * 2


def test_date_only_value_respects_min_age():
    w = Window.from_days(14, 2, now=NOW)  # NOW is Oct 12 12:00 -> end is Oct 10 12:00
    assert not w.contains_range(*date_bounds("Oct 10, 2026", NOW))  # could be Oct 10 23:00
    assert w.contains_range(*date_bounds("Oct 9, 2026", NOW))


def test_search_post_with_loose_date_is_screened_by_range(tmp_path):
    hit = SearchHit("https://www.instagram.com/p/OLD1/", "Glow Spa on Instagram", "Looking for an ads manager", "2 weeks ago")
    post = SearchSource(SearchSourceConfig(), SimpleNamespace(name="fake")).to_post(hit, "q", None, Window.from_days(14, 0, now=NOW))
    assert post.posted_earliest is not None and post.posted_latest is not None
    agent, llm = make_agent(tmp_path, [post], {})
    agent.cfg.window.min_age_days = 0
    report = agent.run(now=NOW)
    assert report.out_of_window == 1 and llm.calls == 0


# -- URL filters / search backends ------------------------------------------------------


def test_post_url_forms():
    assert is_post_url("instagram", "https://www.instagram.com/acmefit/p/C9xyz12/")
    assert is_post_url("instagram", "https://www.instagram.com/acmefit/reel/C9xyz12/")
    assert is_post_url("instagram", "https://www.instagram.com/reels/C9xyz12/")
    assert not is_post_url("instagram", "https://www.instagram.com/reels/audio/123/")
    assert is_post_url("facebook", "https://www.facebook.com/photo/?fbid=123")
    assert is_post_url("facebook", "https://www.facebook.com/acmegym/posts/pfbid0abc")
    assert not is_post_url("facebook", "https://www.facebook.com/acmegym/posts/")
    assert is_post_url("x", "https://x.com/i/web/status/1844000000000000000")


def test_brave_keeps_page_size_fixed():
    api = FakeApi({"web": {"results": [{"url": f"https://x.com/a/status/{i}", "title": "t", "description": "d"} for i in range(20)]}})
    BraveSearch("k", api).search("q", WINDOW, 30)
    assert [c[1]["params"]["count"] for c in api.calls] == [20, 20]
    assert [c[1]["params"]["offset"] for c in api.calls] == [0, 1]


class FakeResearchLLM:
    def __init__(self, hits, posts):
        self.hits, self.posts, self.prompts = hits, posts, []

    def research(self, *, system, user, effort, max_searches):
        self.prompts.append(user)
        return SimpleNamespace(text="report", hits=self.hits)

    def parse(self, schema, **kw):
        return schema.model_validate({"posts": self.posts})


def test_claude_search_uses_server_quoted_text_only():
    hit = WebHit("https://www.instagram.com/p/ABC/", "Glow Spa on Instagram", "2026-10-09", cited=["We need an ads manager for our spa!"])
    llm = FakeResearchLLM([hit], [{"url": "https://instagram.com/p/ABC", "author": "Dana", "quote": "email me: dana@gmail.com"}])
    hits = ClaudeWebSearch(llm).search("q", WINDOW, 5)
    assert hits[0].snippet == "We need an ads manager for our spa!"
    assert hits[0].title == "Glow Spa on Instagram"
    assert hits[0].extra["reported_quote"] == "email me: dana@gmail.com"
    post = SearchSource(SearchSourceConfig(), ClaudeWebSearch(llm)).to_post(hits[0], "q", None, WINDOW)
    assert "dana@gmail.com" not in post.content()  # model text can't become "post text"
    assert "unverified" in build_user_message(post) and "dana@gmail.com" not in build_user_message(post, include_unverified=False)


def test_claude_search_niche_prompt_asks_for_growth_signals():
    llm = FakeResearchLLM([], [])
    ClaudeWebSearch(llm).search('"grand opening" pilates', WINDOW, 5, niche="new pilates studios")
    assert "announcing an opening" in llm.prompts[0] and "asking to hire" not in llm.prompts[0]


def test_citations_are_collected():
    hits = {"https://a.com/1": WebHit("https://a.com/1", "t", None)}
    cite = SimpleNamespace(type="web_search_result_location", url="https://a.com/1", cited_text="We need SMM help")
    _collect_hits([SimpleNamespace(type="text", text="x", citations=[cite])], hits)
    assert hits["https://a.com/1"].cited == ["We need SMM help"]


# -- reddit / rss / import / http -------------------------------------------------------


class PagedApi:
    def __init__(self, pages):
        self.pages, self.calls, self.timeout = pages, [], 5

    def get_json(self, url, **kw):
        self.calls.append(kw["params"])
        return self.pages[len(self.calls) - 1]


def _listing(created, after, n=100):
    return {"data": {"after": after, "children": [
        {"data": {"title": "[Hiring] social media manager", "selftext": "Need IG help", "author": "a", "created_utc": created,
                  "permalink": f"/r/forhire/comments/{created}{i}/x/"}} for i in range(n)]}}


def test_reddit_pages_back_to_the_window():
    fresh = (NOW - timedelta(hours=10)).timestamp()
    older = (NOW - timedelta(days=5)).timestamp()
    too_old = (NOW - timedelta(days=30)).timestamp()
    api = PagedApi([_listing(fresh, "t3_a"), _listing(older, "t3_b", n=1), _listing(too_old, "t3_c", n=1)])
    window = Window.from_days(14, 2, now=NOW)
    posts = list(RedditSource(RedditSourceConfig(subreddits=["forhire"]), api, anon_delay=0).collect(window, []))
    assert len(api.calls) == 3 and api.calls[1]["after"] == "t3_a"
    assert len(posts) == 1  # only the 5-day-old post is inside 2-14 days


def test_rss_1_0_and_guid_links():
    rdf = """<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"><item rdf:about="https://b.example/1"><title>Need ads help</title>
      <link>https://b.example/1</link><description>Our cafe needs Instagram ads</description>
      <dc:date>2026-10-07T09:00:00Z</dc:date></item></rdf:RDF>"""
    guid = """<rss><channel><item><title>Hiring SMM</title><guid isPermaLink="true">https://b.example/2</guid>
      <pubDate>Mon, 05 Oct 2026 10:00:00 +0000</pubDate></item></channel></rss>"""
    assert parse_feed(rdf, "web", "f", WINDOW)[0].url == "https://b.example/1"
    assert parse_feed(guid, "web", "f", WINDOW)[0].url == "https://b.example/2"


def test_import_handles_bom_and_trusts_post_id_dates(tmp_path):
    posted = NOW - timedelta(days=40)
    li_id = int(posted.timestamp() * 1000) << 22
    f = tmp_path / "export.csv"
    f.write_text(f"url,text,posted_at\nhttps://www.linkedin.com/posts/x_y-activity-{li_id}-Ab,need SMM,2026-10-10\n", encoding="utf-8-sig")
    posts = list(ImportSource([str(f)]).collect(WINDOW, []))
    assert len(posts) == 1
    assert posts[0].date_source == "post_id" and posts[0].posted_at == posted  # not the export's claim


class _Resp:
    def __init__(self, status, text="", ctype="text/plain", body=b""):
        self.status_code, self.text, self.headers, self.body = status, text, {"Content-Type": ctype}, body
        self.url, self.encoding = "https://acme.test/", "utf8mb4"

    def iter_content(self, n):
        yield self.body

    def close(self):
        pass


class _Session:
    def __init__(self, robots_status=200, page=None):
        self.robots_status, self.page, self.headers = robots_status, page, {}

    def get(self, url, **kw):
        if url.endswith("/robots.txt"):
            if self.robots_status is None:
                raise requests.ConnectionError("down")
            return _Resp(self.robots_status, "User-agent: *\nAllow: /")
        return self.page


def test_robots_outage_means_disallow():
    assert not HttpClient("ua", per_host_delay=0, session=_Session(503)).allowed("https://acme.test/team")
    assert not HttpClient("ua", per_host_delay=0, session=_Session(None)).allowed("https://acme.test/team")
    assert HttpClient("ua", per_host_delay=0, session=_Session(404)).allowed("https://acme.test/team")


def test_unknown_charset_falls_back_to_utf8():
    page = _Resp(200, ctype="text/html; charset=utf8mb4", body="Café owner".encode())
    client = HttpClient("ua", per_host_delay=0, session=_Session(404, page))
    assert client.fetch_page("https://acme.test/").text == "Café owner"


# -- intent -----------------------------------------------------------------------------


def test_buyers_saying_dm_me_are_not_prefiltered():
    for text in ("Looking for a social media manager for our new gym... DM me for details!",
                 "Need someone to run our Facebook and Instagram ads. Budget $1.5k/mo. DM me to discuss.",
                 "Anyone offering a free audit? We need Instagram marketing help for our bakery."):
        assert prefilter(RawPost("t", "linkedin", "u", text=text)) is None


def test_target_locations_reach_the_classifier():
    cfg = Config()
    cfg.agency.target_locations = ["London, UK"]
    assert "London, UK" in build_system_prompt(cfg)
    assert "only serves" not in build_system_prompt(Config())


def test_transient_claude_failure_leaves_post_for_retry():
    class Down:
        def parse(self, *a, **k):
            raise LLMTransient("overloaded")

    assert IntentClassifier(Config(), Down()).assess(RawPost("t", "x", "u", text="need smm")) is None


# -- enrichment -------------------------------------------------------------------------


def test_generic_words_do_not_match_other_businesses():
    assert not name_matches("Acme Fitness", "anytimefitness.com")
    assert not name_matches("Bloom Yoga Studio", "yogaworks.com")
    assert name_matches("Bloom Yoga Studio", "bloomyoga.com")
    assert normalize_domain("https://shop.acme.com/products") == "acme.com"


def test_name_cleanup_and_titles():
    assert Person("Dr. Amy Chen, DDS", None).first_last == ("Amy", "Chen")
    assert Person("Mark Smith Jr.", None).first_last == ("Mark", "Smith")
    assert Person("Jane Doe (She/Her)", None).first_last == ("Jane", "Doe")
    assert title_weight("Product Owner") == 0.0
    assert title_weight("Vice President of Engineering") == 0.0
    assert title_weight("Vice President of Marketing") == 0.9
    assert title_weight("Principal Engineer") == 0.0


def test_jsonld_odd_founder_values_do_not_crash():
    html = """<script type="application/ld+json">{"@type":"Organization","founder":[{"@type":"Person","name":[{"@value":"Jane Doe"}]},{"name":42}]}</script>"""
    people = extract_jsonld_people(BeautifulSoup(html, "html.parser"))
    assert [p.name for p in people] == ["Jane Doe"]


class _Crawler:
    def __init__(self, profile):
        self.profile = profile

    def crawl(self, domain):
        return self.profile


def _finder(profile=None, llm=None):
    return ContactFinder(Config(), crawler=_Crawler(profile) if profile else None, llm=llm, hunter=None, lookup_backend=None)


def test_foreign_domain_and_ambiguous_first_name_emails_are_not_assigned():
    profile = SiteProfile(
        domain="launchly.io", base_url="https://launchly.io/",
        pages=[SitePage("u", "Jane Smith, Founder & CEO. Dan Smith, Owner. Dan Brown, Front Desk Coordinator.")],
        emails={"jane@pixelforge.studio", "dan@launchly.io"},
        people=[Person("Jane Smith", "Founder & CEO", source="jsonld"), Person("Dan Smith", "Owner", source="jsonld"),
                Person("Dan Brown", "Front Desk Coordinator", source="jsonld")],
    )
    plan = _finder(profile).plan(BusinessContext("Launchly", "launchly.io", None, None, False))
    emails = {c.email for dm in plan.decision_makers for c in dm.candidates}
    assert "jane@pixelforge.studio" not in emails
    dan = next(dm for dm in plan.decision_makers if dm.person.name == "Dan Smith")
    assert all(c.source != "website" for c in dan.candidates)  # "dan@" could be Dan Brown's


def test_anonymous_owner_with_email_in_post_is_kept():
    plan = _finder().plan(BusinessContext(None, None, None, None, True, emails_in_post=["coach.mike@ironhousegym.com"]))
    assert plan.decision_makers and plan.decision_makers[0].candidates[0].email == "coach.mike@ironhousegym.com"


def test_intern_poster_is_not_promoted_to_decision_maker():
    ctx = BusinessContext("Acme", None, None, Person("Kim Park", "Social Media Intern", source="post"), True, ["kim@acme.com"])
    assert _finder().plan(ctx).decision_makers == []


def test_unrelated_redirect_is_ignored_and_related_one_followed():
    parked = SiteProfile(domain="acmefit.com", base_url="https://hugedomains.com/", final_domain="hugedomains.com",
                         pages=[SitePage("u", "Buy this domain. John Owner, Founder.")], people=[Person("John Owner", "Founder", source="jsonld")])
    plan = _finder(parked).plan(BusinessContext("Acme Fitness", "acmefit.com", None, None, False))
    assert plan.domain == "acmefit.com" and plan.decision_makers == []
    moved = SiteProfile(domain="acme.com", base_url="https://acmefitness.com/", final_domain="acmefitness.com",
                        pages=[SitePage("u", "Jane Doe, Founder")], people=[Person("Jane Doe", "Founder", source="jsonld")])
    plan = _finder(moved).plan(BusinessContext("Acme Fitness", "acme.com", None, None, False))
    assert plan.domain == "acmefitness.com"
    assert plan.decision_makers[0].candidates[-1].email.endswith("@acmefitness.com")


def test_crawler_follows_links_on_redirect_target():
    class Http:
        def __init__(self):
            self.fetched = []

        def fetch_page(self, url):
            self.fetched.append(url)
            if url == "https://acme.com/":
                return Page("https://acmefitness.com/", 200, '<a href="https://acmefitness.com/team">Team</a>', "text/html")
            if url == "https://acmefitness.com/team":
                return Page(url, 200, "<p>Jane Doe, Founder</p>", "text/html")
            return None

    http = Http()
    profile = WebsiteCrawler(http, 4).crawl("acme.com")
    assert profile.final_domain == "acmefitness.com" and "https://acmefitness.com/team" in http.fetched


def test_llm_people_drops_emails_not_in_text():
    class FakeLLM:
        def parse(self, schema, **kw):
            return schema.model_validate({"people": [{"name": "Jane Doe", "title": "Founder", "email": "jane.doe@gmail.com"}]})

    people = llm_people(FakeLLM(), "Acme", "Jane Doe, Founder of Acme", "low")
    assert people[0].name == "Jane Doe" and people[0].email is None


def test_claude_outage_during_people_extraction_retries_post(tmp_path):
    class DownLLM:
        def parse(self, *a, **k):
            raise LLMTransient("529 overloaded")

    profile = SiteProfile(domain="acmefit.com", base_url="https://acmefit.com/", pages=[SitePage("u", "Our founder story")])
    plan = _finder(profile, llm=DownLLM()).plan(BusinessContext("Acme Fitness", "acmefit.com", None, None, False))
    assert plan.decision_makers == [] and plan.transient


def test_guessed_poster_name_is_dropped(tmp_path):
    post = li_post("https://www.linkedin.com/posts/s_a-activity-51", 3,
                   text="Our gym Acme Fitness (acmefit.com) needs a social media manager for Instagram ads.")
    post.title = "LinkedIn post"
    agent, _ = make_agent(tmp_path, [post], {post.url: assessment(poster_name="Sarah Lee")}, mailboxes=("sarah@acmefit.com",))
    agent.finder.crawler = None  # only the poster could supply a decision-maker
    report = agent.run(now=NOW)
    assert report.leads == []  # "Sarah Lee" appears nowhere in the post


def test_research_keeps_text_written_before_a_pause():
    from contextlib import contextmanager

    seg1 = SimpleNamespace(stop_reason="pause_turn", content=[SimpleNamespace(type="text", text="Post one: gym opening.")])
    seg2 = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Post two: spa launch.")])
    responses = [seg1, seg2]

    @contextmanager
    def stream(**kw):
        r = responses.pop(0)
        yield SimpleNamespace(get_final_message=lambda: r)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    result = LLM(Config().llm, client=client).research(system="s", user="u", effort="low", max_searches=2)
    assert "Post one" in result.text and "Post two" in result.text


def test_validation_error_in_niche_research_does_not_abort_run(tmp_path):
    agent, _ = make_agent(tmp_path, [], {})

    class Broken:
        def get(self):
            raise json.JSONDecodeError("bad", "", 0)

    agent.niche_researcher = Broken()
    report = agent.run(now=NOW)
    assert report.niches == []


def test_poster_name_grounding_ignores_accents_and_punctuation():
    from leadgen.pipeline import _comparable

    assert _comparable("José O’Neil") in _comparable("Hi, I'm Jose O'Neil, owner of Acme")
    assert _comparable("Ann Lee") not in _comparable("Joanne Leeds here")  # whole words only
