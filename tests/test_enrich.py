from bs4 import BeautifulSoup

from leadgen.config import Config
from leadgen.enrich import BusinessContext, ContactFinder
from leadgen.enrich.domain import name_matches, normalize_domain, resolve_domain
from leadgen.enrich.hunter import HunterDomainResult
from leadgen.enrich.patterns import email_matches_name, infer_pattern, permutations
from leadgen.enrich.people import keyword_windows, regex_people, title_weight
from leadgen.enrich.website import (
    SitePage,
    SiteProfile,
    WebsiteCrawler,
    candidate_links,
    decode_cfemail,
    extract_emails,
    extract_jsonld_people,
)
from leadgen.http import Page
from leadgen.models import Person
from leadgen.sources.search import SearchHit

# -- patterns ---------------------------------------------------------------------------


def test_permutations_prefer_known_pattern():
    out = permutations("Sarah", "O'Connor", "acme.com", preferred="{f}{last}", limit=4)
    assert out == ["soconnor@acme.com", "sarah@acme.com", "sarah.oconnor@acme.com", "sarahoconnor@acme.com"]


def test_permutations_strip_accents():
    assert permutations("José", "Núñez", "acme.com", limit=2) == ["jose@acme.com", "jose.nunez@acme.com"]


def test_infer_pattern_and_name_match():
    assert infer_pattern("j.smith@acme.com", "John", "Smith") == "{f}.{last}"
    assert infer_pattern("random@acme.com", "John", "Smith") is None
    assert email_matches_name("sarah@acme.com", "Sarah", "Lee")
    assert email_matches_name("slee@acme.com", "Sarah", "Lee")
    assert not email_matches_name("mike@acme.com", "Sarah", "Lee")


# -- people -----------------------------------------------------------------------------


def test_title_weights():
    assert title_weight("Founder & CEO") == 1.0
    assert title_weight("Owner / Head Coach") == 1.0
    assert title_weight("Head of Marketing") == 0.9
    assert title_weight("Marketing Manager") == 0.65
    assert title_weight("Marketing Assistant") == 0.0
    assert title_weight("Personal Trainer") == 0.0
    assert title_weight(None) == 0.0


def test_regex_people():
    text = "Meet the team. Sarah Lee, Founder & Head Coach, opened the studio in 2026. Owner: Mark Diaz handles ops."
    people = {p.name: p.role for p in regex_people(text)}
    assert people["Sarah Lee"].startswith("Founder")
    assert people["Mark Diaz"] == "Owner"


def test_keyword_windows_only_keeps_relevant_text():
    text = "x" * 5000 + " Our founder Sarah Lee " + "y" * 5000
    window = keyword_windows(text, radius=50)
    assert "Sarah Lee" in window
    assert len(window) < 200
    assert keyword_windows("nothing relevant here") == ""


# -- website ----------------------------------------------------------------------------

SITE_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"HealthClub","name":"Acme Fitness",
 "founder":{"@type":"Person","name":"Sarah Lee"},"email":"info@acmefit.com"}
</script></head>
<body>
<a href="/about-us">About us</a> <a href="/team">Our Team</a> <a href="https://instagram.com/acme">IG</a>
<a href="/blog/post-1">Blog</a>
<a href="mailto:sarah@acmefit.com?subject=Hi">Email Sarah</a>
<a href="/cdn-cgi/l/email-protection#{cf}">[email protected]</a>
<p>Bookings: mark [at] acmefit [dot] com</p>
<img src="logo@2x.png">
</body></html>
"""


def _cf(email: str, key: int = 0x42) -> str:
    return f"{key:02x}" + "".join(f"{ord(c) ^ key:02x}" for c in email)


def test_decode_cfemail():
    assert decode_cfemail(_cf("owner@acmefit.com")) == "owner@acmefit.com"


def test_extract_emails_and_people():
    html = SITE_HTML.replace("{cf}", _cf("owner@acmefit.com"))
    soup = BeautifulSoup(html, "html.parser")
    emails = extract_emails(soup, soup.get_text(" "))
    assert {"sarah@acmefit.com", "owner@acmefit.com", "mark@acmefit.com"} <= emails
    assert not any("png" in e for e in emails)
    people = extract_jsonld_people(soup)
    assert [(p.name, p.role) for p in people] == [("Sarah Lee", "Founder")]
    links = candidate_links(soup, "https://acmefit.com/", "acmefit.com")
    assert set(links) == {"https://acmefit.com/about-us", "https://acmefit.com/team"}


class FakeHttp:
    def __init__(self, pages):
        self.pages = pages
        self.fetched = []

    def fetch_page(self, url):
        self.fetched.append(url)
        html = self.pages.get(url)
        return Page(url=url, status=200, text=html, content_type="text/html") if html else None


def test_crawler_follows_team_pages():
    http = FakeHttp(
        {
            "https://acmefit.com/": SITE_HTML.replace("{cf}", _cf("owner@acmefit.com")),
            "https://acmefit.com/team": "<p>Sarah Lee, Founder. Mark Diaz - Owner.</p>",
        }
    )
    profile = WebsiteCrawler(http, max_pages=4).crawl("acmefit.com")
    assert profile.reachable
    assert "https://acmefit.com/team" in http.fetched
    assert "https://acmefit.com/blog/post-1" not in http.fetched
    assert any("Mark Diaz" in p.text for p in profile.pages)


# -- domain -----------------------------------------------------------------------------


def test_normalize_domain():
    assert normalize_domain("https://www.AcmeFit.com/about?x=1") == "acmefit.com"
    assert normalize_domain("acmefit.com") == "acmefit.com"
    assert normalize_domain("https://instagram.com/acmefit") is None
    assert normalize_domain("https://acme.wixsite.com/gym") is None
    assert normalize_domain("linktr.ee/acme") is None
    assert normalize_domain(None) is None


def test_name_matches():
    assert name_matches("Acme Fitness Studio", "acmefitness.com")
    assert name_matches("Acme Fitness", "acme-fit.co.uk")
    assert not name_matches("Acme Fitness", "bestgyms.com")


class FakeBackend:
    name = "fake"

    def __init__(self, hits):
        self.hits = hits

    def search(self, query, window, limit):
        return self.hits


def test_resolve_domain_skips_directories():
    backend = FakeBackend(
        [
            SearchHit("https://www.yelp.com/biz/acme-fitness-austin", "Acme Fitness - Yelp", ""),
            SearchHit("https://www.facebook.com/acmefitness", "Acme Fitness | Facebook", ""),
            SearchHit("https://acmefitness.com/", "Acme Fitness | Boutique gym in Austin", ""),
        ]
    )
    assert resolve_domain("Acme Fitness", "Austin, TX", backend) == "acmefitness.com"


# -- contact finder ---------------------------------------------------------------------


class FakeCrawler:
    def __init__(self, profile):
        self.profile = profile

    def crawl(self, domain):
        return self.profile


class FakeHunter:
    def domain_search(self, domain):
        return HunterDomainResult(
            pattern="{first}.{last}",
            people=[
                Person("Mark Diaz", "Co-owner", "mark.diaz@acmefit.com", "hunter"),
                Person("Amy Cho", "Front Desk Assistant", "amy.cho@acmefit.com", "hunter"),
            ],
        )


def test_contact_finder_ranks_decision_makers_and_candidates():
    cfg = Config()
    profile = SiteProfile(
        domain="acmefit.com",
        base_url="https://acmefit.com/",
        pages=[SitePage("https://acmefit.com/team", "Sarah Lee, Founder & Head Coach. Amy Cho, Front Desk Assistant.")],
        emails={"info@acmefit.com", "sarah@acmefit.com"},
        people=[Person("Sarah Lee", "Founder", source="jsonld")],
    )
    finder = ContactFinder(cfg, crawler=FakeCrawler(profile), llm=None, hunter=FakeHunter(), lookup_backend=None)
    plan = finder.plan(
        BusinessContext(
            business_name="Acme Fitness",
            website="https://acmefit.com",
            location="Austin",
            poster=None,
            poster_is_decision_maker=False,
        )
    )
    names = [dm.person.name for dm in plan.decision_makers]
    assert names[0] == "Sarah Lee" and "Mark Diaz" in names and "Amy Cho" not in names
    sarah = plan.decision_makers[0]
    assert sarah.candidates[0].email == "sarah@acmefit.com"  # found on the site beats guesses
    assert "sarah.lee@acmefit.com" in [c.email for c in sarah.candidates]  # hunter pattern first among guesses
    assert all(not c.email.startswith("info@") for dm in plan.decision_makers for c in dm.candidates)


def test_contact_finder_uses_poster_email_and_domain_from_post():
    cfg = Config()
    finder = ContactFinder(cfg, crawler=None, llm=None, hunter=None, lookup_backend=None)
    plan = finder.plan(
        BusinessContext(
            business_name="Glow Med Spa",
            website=None,
            location=None,
            poster=Person("Dana Reyes", None, source="post"),
            poster_is_decision_maker=True,
            emails_in_post=["dana@glowmedspa.com"],
        )
    )
    assert plan.domain == "glowmedspa.com"
    assert plan.decision_makers[0].candidates[0].email == "dana@glowmedspa.com"
    assert plan.decision_makers[0].candidates[0].source == "post"


def test_contact_finder_reports_missing_decision_maker():
    finder = ContactFinder(Config(), crawler=None, llm=None, hunter=None, lookup_backend=None)
    plan = finder.plan(BusinessContext("Mystery Co", None, None, None, False))
    assert plan.decision_makers == []
    assert any("decision-maker" in n for n in plan.notes)
