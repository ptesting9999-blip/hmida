import json
from types import SimpleNamespace
from datetime import datetime, timedelta

from leadgen.config import RedditSourceConfig, SearchSourceConfig
from leadgen.dates import UTC, Window
from leadgen.queries import intent_queries
from leadgen.sources.base import NicheTarget, author_from_linkedin_url, author_from_title, platform_for_url
from leadgen.sources.importer import ImportSource
from leadgen.sources.reddit import RedditSource, is_hiring_post
from leadgen.sources.rss import parse_feed
from leadgen.sources.search import BraveSearch, SearchHit, SearchSource, SerperSearch, is_post_url

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
WINDOW = Window.from_days(14, now=NOW)


class FakeApi:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.timeout = 5

    def get_json(self, url, **kw):
        self.calls.append((url, kw))
        return self.payload

    def post_json(self, url, **kw):
        self.calls.append((url, kw))
        return self.payload


def test_platform_and_author_helpers():
    assert platform_for_url("https://www.linkedin.com/posts/x") == "linkedin"
    assert platform_for_url("https://m.facebook.com/groups/1/posts/2") == "facebook"
    assert platform_for_url("https://twitter.com/a/status/1") == "x"
    assert platform_for_url("https://austin.craigslist.org/cpg/d/123.html") == "craigslist"
    assert platform_for_url("https://acmefit.com") == "web"
    assert author_from_title("Jane Doe on LinkedIn: Looking for a social media manager") == "Jane Doe"
    assert author_from_title('Acme Fitness (@acmefit) on Instagram: "Grand opening!"') == "Acme Fitness"
    assert author_from_title("10 tips for hiring a social media manager") is None
    assert author_from_linkedin_url("https://www.linkedin.com/posts/jane-doe-4a1b2c_hiring-activity-1-x") == "Jane Doe"


def test_intent_queries_cover_each_platform():
    queries = intent_queries(["linkedin", "web"])
    assert queries[0][0].startswith('site:linkedin.com/posts ("looking for a social media manager" OR ')
    assert {p for _, p in queries} == {"linkedin", "web"}
    assert not queries[-1][0].startswith("site:")


def test_post_url_filter():
    assert is_post_url("instagram", "https://www.instagram.com/p/C9xyz/")
    assert not is_post_url("instagram", "https://www.instagram.com/acmefit/")
    assert is_post_url("x", "https://x.com/acme/status/123")
    assert not is_post_url("linkedin", "https://www.linkedin.com/in/jane-doe")
    assert is_post_url("web", "https://forum.example.com/t/need-help")


def test_brave_search_sends_exact_date_range():
    api = FakeApi(
        {
            "web": {
                "results": [
                    {
                        "url": "https://www.instagram.com/p/ABC123/",
                        "title": "Acme Fitness on Instagram: <strong>Looking for a social media manager</strong>",
                        "description": "We just opened and need help!",
                        "page_age": "2026-10-08T10:00:00",
                    }
                ]
            }
        }
    )
    hits = BraveSearch("key", api).search("q", WINDOW, 5)
    params = api.calls[0][1]["params"]
    assert params["freshness"] == "2026-09-28to2026-10-12"
    assert api.calls[0][1]["headers"]["X-Subscription-Token"] == "key"
    assert hits[0].title == "Acme Fitness on Instagram: Looking for a social media manager"
    assert hits[0].date_text == "2026-10-08T10:00:00"


def test_serper_search_uses_google_custom_range():
    api = FakeApi({"organic": [{"link": "https://x.com/a/status/1", "title": "t", "snippet": "s", "date": "3 days ago"}]})
    hits = SerperSearch("key", api).search("q", WINDOW, 5)
    assert api.calls[0][1]["json"]["tbs"] == "cdr:1,cd_min:9/28/2026,cd_max:10/12/2026"
    assert hits[0].date_text == "3 days ago"


class FakeBackend:
    name = "fake"

    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, window, limit, niche=None):
        self.queries.append(query)
        return self.hits


def test_search_source_dates_and_filters():
    posted = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
    li_id = int(posted.timestamp() * 1000) << 22
    hits = [
        SearchHit(
            f"https://www.linkedin.com/posts/jane-doe-1a2b_hiring-activity-{li_id}-AbC",
            "Jane Doe on LinkedIn: We need a social media manager for our gym",
            "We need a social media manager for our new gym in Austin...",
            date_text="1 day ago",  # the ID beats the indexed date
        ),
        SearchHit("https://www.instagram.com/p/XYZ/", "Glow Spa on Instagram", "Looking for an ads manager", "3 days ago"),
        SearchHit("https://www.instagram.com/glowspa/", "Glow Spa profile", "bio", "3 days ago"),
    ]
    cfg = SearchSourceConfig(platforms=["linkedin"])
    src = SearchSource(cfg, FakeBackend(hits))
    niche = NicheTarget(name="new gyms", search_queries=['"grand opening" gym site:instagram.com'])
    posts = list(src.collect(WINDOW, [niche]))
    li = [p for p in posts if p.platform == "linkedin"][0]
    assert li.posted_at == posted and li.date_source == "post_id" and li.author == "Jane Doe"
    ig = [p for p in posts if p.platform == "instagram"][0]
    assert ig.posted_at == NOW - timedelta(days=3) and ig.date_source == "search_index"
    assert not any(p.url.endswith("/glowspa/") for p in posts)
    assert any(p.niche == "new gyms" for p in posts)


def test_reddit_listing_and_hiring_filter():
    created = (NOW - timedelta(days=2)).timestamp()
    old = (NOW - timedelta(days=30)).timestamp()
    listing = {
        "data": {
            "children": [
                {"data": {"title": "[Hiring] Social media manager for my bakery", "selftext": "Need IG help", "author": "bakerjo", "created_utc": created, "permalink": "/r/forhire/comments/a1/x/", "link_flair_text": "Hiring"}},
                {"data": {"title": "[For Hire] SMM with 5 years exp", "selftext": "", "author": "smm", "created_utc": created, "permalink": "/r/forhire/comments/a2/y/", "link_flair_text": "For Hire"}},
                {"data": {"title": "[Hiring] old post", "selftext": "", "author": "o", "created_utc": old, "permalink": "/r/forhire/comments/a3/z/"}},
            ]
        }
    }
    cfg = RedditSourceConfig(subreddits=["forhire"])
    posts = list(RedditSource(cfg, FakeApi(listing), anon_delay=0).collect(WINDOW, []))
    assert [p.url for p in posts] == ["https://www.reddit.com/r/forhire/comments/a1/x/"]
    assert posts[0].date_source == "platform_api" and posts[0].author == "u/bakerjo"
    assert is_hiring_post("[HIRING] need ads help", None)
    assert not is_hiring_post("[For Hire] ads expert", "For Hire")


def test_rss_and_atom_parsing():
    rss = """<rss><channel><item><title>Need a Facebook ads freelancer</title><link>https://jobs.example.com/1</link>
    <description>&lt;p&gt;Our cafe needs &lt;b&gt;ads&lt;/b&gt;&lt;/p&gt;</description><pubDate>Mon, 05 Oct 2026 10:00:00 +0000</pubDate></item></channel></rss>"""
    atom = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Hiring SMM</title>
    <link rel="self" href="https://jobs.example.com/self"/><link rel="alternate" href="https://jobs.example.com/2"/>
    <summary>Social media help wanted</summary><published>2026-10-07T09:00:00Z</published></entry></feed>"""
    a = parse_feed(rss, "web", "feed", WINDOW)[0]
    assert a.text == "Our cafe needs ads" and a.posted_at.day == 5
    b = parse_feed(atom, "web", "feed", WINDOW)[0]
    assert b.url == "https://jobs.example.com/2" and b.posted_at.day == 7


def test_import_source(tmp_path):
    f = tmp_path / "posts.jsonl"
    f.write_text(json.dumps({"url": "https://www.linkedin.com/posts/x", "text": "need SMM", "posted_at": "2026-10-09", "website": "acme.com"}) + "\n")
    posts = list(ImportSource([str(f)]).collect(WINDOW, []))
    assert posts[0].platform == "linkedin" and posts[0].extra["website"] == "acme.com"
    assert posts[0].posted_at.date().isoformat() == "2026-10-09"


def test_claude_backend_gets_one_query_per_platform():
    class ClaudeLike(FakeBackend):
        phrases_per_query = 100

    backend = ClaudeLike([])
    list(SearchSource(SearchSourceConfig(platforms=["linkedin", "instagram"]), backend).collect(WINDOW, []))
    assert len(backend.queries) == 2
    assert backend.queries[0].count(" OR ") == 17


class FakeResponse:
    def __init__(self, body: bytes, ctype: str, encoding=None):
        self.body, self.headers, self.encoding = body, {"Content-Type": ctype}, encoding
        self.url, self.status_code, self.closed = "https://acme.test/", 200, False

    def iter_content(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i : i + size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response, self.headers = response, {}

    def get(self, url, **kw):
        if url.endswith("/robots.txt"):
            return SimpleNamespace(status_code=404, text="")
        return self.response


def test_fetch_page_caps_size_and_decodes_utf8():
    from leadgen import http as http_mod

    body = "Café owner: Zoë".encode() + b"x" * (http_mod.MAX_PAGE_BYTES + 500_000)
    resp = FakeResponse(body, "text/html", encoding="ISO-8859-1")  # what requests guesses without a charset
    client = http_mod.HttpClient("ua", per_host_delay=0, session=FakeSession(resp))
    page = client.fetch_page("https://acme.test/")
    assert page.text.startswith("Café owner: Zoë")
    assert len(page.text.encode()) <= http_mod.MAX_PAGE_BYTES
    assert resp.closed


def test_fetch_page_skips_binary_content():
    from leadgen import http as http_mod

    resp = FakeResponse(b"%PDF-1.7", "application/pdf")
    page = http_mod.HttpClient("ua", per_host_delay=0, session=FakeSession(resp)).fetch_page("https://acme.test/x.pdf")
    assert page.text == ""
