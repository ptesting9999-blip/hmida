"""Rule-mode judgement on the post shapes seen in real search results (paraphrased, names invented).

Every case here was wrong in an earlier version: a seller, job board or third party accepted as a
lead, or a real buyer / opening rejected.
"""

import pytest

from leadgen.config import Config
from leadgen.enrich import BusinessContext, _owns_email
from leadgen.enrich.domain import normalize_domain
from leadgen.enrich.people import regex_people
from leadgen.intent import IntentClassifier, prefilter
from leadgen.models import Person, RawPost
from leadgen.rule_intent import assess_without_ai


def accepted(text, url="https://www.facebook.com/groups/1/posts/2/", platform="facebook", title="", author=None, niche=None):
    post = RawPost("t", platform, url, title=title, text=text, author=author, niche=niche)
    if prefilter(post) is not None:
        return False
    return IntentClassifier(Config(), None).accept(assess_without_ai(post, Config())) is None


BUYERS = [
    dict(text="WE'RE HIRING! Bloom Boutique is looking for a SOCIAL MEDIA MANAGER to join our team!", title="WE'RE HIRING! Bloom Boutique"),
    dict(text="Need a meta ad expert, appreciate any assistance. Please DM me.", title="Need a meta ad expert - Facebook"),
    dict(text="Looking for a social media agency / person for a fitness related brand. Please dm or comment.",
         title="Looking for recommendations for a social media marketing agency"),
    dict(text="HIRING: Meta Ads Expert Freelance | Remote | 2-5 yrs Can you turn ad spend into sales? FB & Insta Ads",
         title="Kanpo | HIRING: Meta Ads Expert", platform="instagram", url="https://www.instagram.com/p/abc/"),
    dict(text="I need someone to run a Facebook ad for my real estate listings.", platform="web",
         url="https://www.upwork.com/freelance-jobs/apply/run-facebook-ads"),
    dict(text="Glenhaven Bartending School is looking for a Google Ads Specialist to join our team!", platform="web",
         url="https://glenhaven.jobsoid.com/j/1/google-ads-specialist", title="Google Ads Specialist - Glenhaven - Jobsoid"),
    dict(text="Corvo is hiring a Social Media Manager who will own our brand presence across social channels.", platform="web",
         url="https://apply.workable.com/corvo/j/1/", title="Social Media Manager - Corvo - Workable Jobs"),
    dict(text="Any social media agencies you'd recommend for a med spa?"),
    dict(text="Just fired our Facebook ads agency. Need a replacement ASAP for our med spa"),
    dict(text="Social media manager wanted for a busy cafe in Leeds. £15/hr, 10 hrs a week"),
    dict(text="I’m a life coach (I help coaches and moms find balance) and I need a social media manager for my business."),
    dict(text="I’m looking for a Google Ads specialist who can help promote my newly built media app."),
]


@pytest.mark.parametrize("case", BUYERS, ids=lambda c: c["text"][:40])
def test_real_buyers_are_kept(case):
    assert accepted(**case)


NOT_LEADS = [
    # job aggregators and agencies hiring staff for their clients
    dict(text="We are looking for a Meta Ads Specialist with 2 years of experience.", platform="web",
         url="https://pk.indeed.com/q-facebook-ads-jobs.html", title="Facebook ads jobs in Sadhoke - Indeed"),
    dict(text="They're looking for a social media manager to own strategy across Instagram.", platform="web",
         url="https://www.example-jobs.co.uk/social", title="Social Media Manager Jobs in Greater Manchester - Updated Daily"),
    dict(text="Socialander is looking for a Social Media Manager with experience managing sports content.",
         title="WE'RE HIRING! Socialander is looking for a Social Media Manager"),
    dict(text="We're looking for a Google Ads Specialist to build and manage PPC campaigns across our client accounts."),
    dict(text="Advora Digital Agency | We're hiring! We are looking for a Meta Ads Specialist to join our team.",
         title="Advora Digital Agency | We're hiring!", platform="instagram", url="https://www.instagram.com/p/xyz/"),
    # agency websites using buyer phrases as copy, listicles, content hooks
    dict(text="Skyhook Media helps businesses looking for a social media agency build social programs.", platform="web",
         url="https://skyhookmedia.com/", title="Social Media Agency in St. Petersburg, FL"),
    dict(text="Looking for a digital marketing agency in Toronto that blends strategy and execution? You're in the right place.",
         platform="web", url="https://arnoldlane.agency/contact", title="Contact Us - Arnold Lane"),
    dict(text="I am looking for a social media agency to help launch our new product.", platform="web",
         url="https://lists.example.com/dubai", title="20+ Social Media Agencies in Dubai (2026)"),
    dict(text="Hiring a Social Media Manager... But your job description says: create content, write copy, run ads."),
    dict(text="What a social media manager costs as an employee or an agency, and when the hire pays off.",
         title="Hiring a Social Media Manager: Cost and When It Pays Off"),
    dict(text="I’m a Meta Ads Specialist with skills in Facebook and Instagram advertising.", title="I need meta experts"),
    # job seekers, recruiters, enterprises, learners and tool shoppers
    dict(text="Looking for an internship in social media marketing for summer 2027. Please DM"),
    dict(text="My client, a Fortune 500 retailer, is looking for a Paid Social Media Manager. $120k-$140k"),
    dict(text="Amazon is hiring a Senior Social Media Manager - Seattle, WA. Apply on our careers page."),
    dict(text="Looking for a good course on Facebook ads. Any recommendations?"),
    dict(text="Can anyone recommend a social media management tool? Hootsuite vs Buffer?"),
]


@pytest.mark.parametrize("case", NOT_LEADS, ids=lambda c: c["text"][:40])
def test_sellers_job_boards_and_others_are_rejected(case):
    assert not accepted(**case)


OPENINGS = [
    "7 DAYS TO GO! GRAND OPENING – ZEST FITNESS STUDIO 2.0 Get ready for a new fitness experience!",
    "MAPLE HEALTH & FITNESS IS NOW OPEN! We're excited to welcome you.",
    "Join us for the Grand Opening of Erasmo Med Spa – Wesley Chapel on November 5th!",
    "We're celebrating the GRAND OPENING of The Cedar Tub Store's Macomb location, and you're invited!",
    "TODAY IS THE DAY! Come celebrate the Grand Opening of our new studio downtown.",
]


@pytest.mark.parametrize("text", OPENINGS)
def test_own_openings_are_growth_signals(text):
    assert accepted(text, niche="new gyms and med spas")


NOT_OPENINGS = [
    "Huge congratulations to my friend on the grand opening of AK Health & Fitness! Go check them out",
    "One of my favorite French bakeries just opened a new location in Cambridge.",
    "We're so excited to celebrate the grand opening of Montecito Med Spa! - Santa Ynez Chamber of Commerce",
    "Grand Opening Spotlight: Casa Oro Med Spa, a new destination for wellness, is opening its doors.",
    "Stacked restaurant in Dothan now open. Summarized by local news. The restaurant held its grand opening today.",
    "Toast just launched Toast IQ Team, a new labor management solution for restaurant operators.",
    "IT'S FINALLY HERE Our VIO Med Spa Brier Creek Grand Opening is this week!",
    "We're officially less than one week away from the Grand Opening! Join us as we celebrate our newest location.",
    "The Lux MedSpa is celebrating 3 years + the grand opening of their new space downtown.",
    "Voting is now open until October 22! Our gym is proud to be nominated.",
    "Gyms that open strong start marketing 90 days before the doors open: a grand-opening playbook.",
]


@pytest.mark.parametrize("text", NOT_OPENINGS)
def test_other_peoples_openings_chains_news_and_vendors_are_not(text):
    assert not accepted(text, niche="new gyms and med spas")


def test_title_is_the_author_only_in_explicit_forms():
    # A headline full of provider words is not the poster's name ...
    assert accepted("Looking for a Meta Ads Specialist to manage our Facebook campaigns.",
                    title="Looking for a Meta Ads Specialist to manage and scale Facebook campaigns")
    # ... but "X's Post" and handles are.
    assert not accepted("We're hiring a social media manager!", title="Bright Pixel Marketing's Post - LinkedIn")
    assert not accepted("Hiring a social media manager now", platform="linkedin",
                        url="https://www.linkedin.com/posts/jane-doe-social-media-manager_hiring-activity-1")


def test_target_locations_reject_clearly_elsewhere():
    cfg = Config()
    cfg.agency.target_locations = ["United States"]
    post = RawPost("t", "linkedin", "https://ng.linkedin.com/posts/x", text="Looking for a social media manager for our cafe in Lagos")
    assert IntentClassifier(cfg, None).accept(assess_without_ai(post, cfg)) is not None
    home = RawPost("t", "linkedin", "https://www.linkedin.com/posts/x", text="Looking for a social media manager for our cafe in Austin, TX")
    assert IntentClassifier(cfg, None).accept(assess_without_ai(home, cfg)) is None
    unknown = RawPost("t", "linkedin", "https://www.linkedin.com/posts/x", text="Looking for a social media manager for our cafe")
    assert IntentClassifier(cfg, None).accept(assess_without_ai(unknown, cfg)) is None


def test_owner_phrases_and_extraction():
    def a(text, author=None):
        return assess_without_ai(RawPost("t", "linkedin", "u", text=text, author=author), Config())

    for text in ("I own Glow Studio and need a social media manager", "Owner of Glow Studio here! Need SMM help",
                 "I'm a co-founder of Glow Studio, looking for a social media manager", "I'm a small business owner, need IG help"):
        assert a(text).poster_is_decision_maker, text
    assert not a("My company is hiring a social media manager ASAP").poster_is_decision_maker
    assert not a("I'm the owner's assistant at Glow Studio, we need a social media manager").poster_is_decision_maker
    assert a("Need SMM help", author="José Núñez, MBA").poster_name == "José Núñez"
    assert a("As Founder of Glow Studio, I need a social media manager").business_name == "Glow Studio"
    assert a("We launched Black Friday sales at my boutique, need ads help").business_name is None
    assert a("Book here: calendly.com/janedoe or glowstudio.com", author="Jane Doe").business_website == "glowstudio.com"
    assert a("We're a salon in Austin.We need a social media manager").business_website is None


def test_booking_and_shortener_hosts_are_never_the_business_domain():
    for host in ("calendly.com", "toasttab.com", "fresha.com", "bit.ly", "gofundme.com", "linktr.ee"):
        assert normalize_domain(f"https://{host}/x") is None, host


def test_post_emails_count_only_if_they_belong_to_the_poster_or_business():
    jane = Person("Jane Doe", None, source="post")
    ctx = BusinessContext("Glow Studio", None, None, jane, True, ["jane@glowstudio.com", "mike@sparkmedia.com", "amy@glowstudio.com"])
    assert _owns_email("jane@glowstudio.com", ctx)
    assert not _owns_email("mike@sparkmedia.com", ctx)  # her old agency
    assert not _owns_email("amy@glowstudio.com", ctx)  # her assistant
    assert _owns_email("glowstudio@gmail.com", ctx)  # the business's own inbox


def test_page_owner_phrasings():
    names = {p.name: p.role for p in regex_people("Meet Sarah Lee, our founder. Glow was founded in 2021 by Tom Park.")}
    assert names == {"Sarah Lee": "founder", "Tom Park": "Founder"}


# -- verification budget ------------------------------------------------------------------------


class CountingProvider:
    """A verification API that charges one credit per address and knows no mailbox."""

    name = "zerobounce"

    def __init__(self):
        self.checked = []

    def check(self, email):
        from leadgen.verify.providers import ProviderVerdict

        self.checked.append(email)
        return ProviderVerdict(status="undeliverable", detail="mailbox not found")


def test_verify_many_spends_at_most_max_checks():
    from leadgen.config import VerificationConfig
    from leadgen.verify import EmailVerifier
    from leadgen.verify.dns_check import MxResult

    provider = CountingProvider()
    v = EmailVerifier(VerificationConfig(smtp=False), provider=provider, mx_lookup=lambda d: MxResult("ok", ["mx"]),
                      disposable=frozenset())
    emails = ["jane@glow.com", "jane.doe@glow.com", "jdoe@glow.com", "info@glow.com"]
    results = v.verify_many(emails, stop_at_first=True, max_checks=2)
    assert provider.checked == ["jane@glow.com", "jane.doe@glow.com"]
    assert sum(r.checked_now for r in results.values()) == 2
    assert "budget" in results["jdoe@glow.com"].reason


def test_one_business_never_costs_more_than_its_budget(tmp_path):
    from test_pipeline import NOW, assessment, li_post, make_agent

    from leadgen.enrich import ContactPlan, DecisionMaker
    from leadgen.models import EmailCandidate

    agent, _ = make_agent(tmp_path, [], {})
    provider = CountingProvider()
    agent.verifier.prober, agent.verifier.provider = None, provider
    agent.cfg.verification.max_checks_per_business = 3
    people = [Person(n, "Founder") for n in ("Sarah Lee", "Tom Park", "Ben Cole")]
    plan = ContactPlan(domain="glow.com", website_reachable=True, decision_makers=[
        DecisionMaker(p, 1.0, [EmailCandidate(f"{p.name_parts[0].lower()}{i}@glow.com", p, "pattern", 50 - i) for i in range(3)])
        for p in people
    ])
    ctx = BusinessContext("Glow", "glow.com", None, None, False)
    leads, rejection, _ = agent._deliver(li_post("https://www.linkedin.com/posts/x", 5), assessment(), ctx, plan, False)
    assert not leads and len(provider.checked) == 3  # 9 guesses across 3 people, 3 credits
    assert "budget" in rejection.reason
    assert NOW  # keep the fixture import used


# -- business names, languages, guessed websites -------------------------------------------------


def _assess(text, author=None, niche=None, url="https://www.linkedin.com/posts/x"):
    return assess_without_ai(RawPost("t", "linkedin", url, text=text, author=author, niche=niche), Config())


def test_business_names_from_hiring_phrases():
    assert _assess("Jive is hiring a social media manager! If you love content, apply.").business_name == "Jive"
    assert _assess("I'm hiring a social media manager for Luckin Coffee").business_name == "Luckin Coffee"
    agency = _assess("We're hiring a social media manager at RMD Advertising")
    assert agency.author_is_service_provider  # an agency hiring staff for its clients
    assert _assess("Looking for a freelance social media manager for our roastery", author="Downeastcoffee").business_name == "Downeastcoffee"


def test_posts_in_other_languages_are_not_judged():
    assert not accepted("I går var det grand opening hos Kitch'n på Bryne! Før: 4 000 ordre", niche="cafes")
    assert not accepted("Grand Opening af vores nye hovedkontor på Marmormolen", niche="cafes")
    assert accepted("HIRING: Meta Ads Expert Freelance | Remote | 2-5 yrs FB & Insta Ads, Pixel, CAPI, GA4")


def test_guessed_handle_domain_is_kept_only_if_the_site_names_the_business():
    from leadgen.enrich import ContactFinder
    from leadgen.enrich.website import SitePage, SiteProfile

    class Crawler:
        def __init__(self, text):
            self.text, self.crawled = text, []

        def crawl(self, domain):
            self.crawled.append(domain)
            return SiteProfile(domain=domain, base_url=f"https://{domain}/", pages=[SitePage("u", self.text)],
                               people=[Person("Mara Quinn", "Owner", source="jsonld")])

    ctx = BusinessContext("Downeastcoffee", None, None, None, False)
    right = ContactFinder(Config(), crawler=Crawler("Down East Coffee roasters. Mara Quinn, Owner"), llm=None, hunter=None, lookup_backend=None)
    plan = right.plan(ctx)
    assert plan.domain == "downeastcoffee.com" and plan.decision_makers
    wrong = ContactFinder(Config(), crawler=Crawler("This domain is for sale"), llm=None, hunter=None, lookup_backend=None)
    plan = wrong.plan(ctx)
    assert plan.domain is None and not plan.decision_makers


def test_posts_opening_with_marketing_hashtags_are_marketers():
    assert not accepted("Looking for a digital marketing agency? Struggling with reach?", platform="linkedin",
                        url="https://www.linkedin.com/posts/someone_metaads-facebookads-digitalmarketing-activity-1")
    assert not accepted("Looking for a Social Media Agency to drive Instagram growth", platform="linkedin",
                        title="#socialmedia #contentcreation #digitalmarketing | Zed Ali",
                        url="https://www.linkedin.com/posts/zedali_socialmedia-contentcreation-activity-2")
    assert accepted("We're looking for an experienced freelance social media manager for our roastery", platform="linkedin",
                    url="https://www.linkedin.com/posts/downeastroast_were-looking-for-an-experienced-freelance-activity-3")
