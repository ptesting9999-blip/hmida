"""Regression tests for the second review round (dates, URLs, titles, robots, Reddit, Claude errors)."""

from datetime import datetime, timedelta

import pytest
import requests
from test_audit_fixes_sources_enrich import PagedApi, _listing, _Session
from test_llm_intent import mock_client
from test_pipeline import NOW

from leadgen import http as http_mod
from leadgen.config import LLMConfig, RedditSourceConfig
from leadgen.dates import UTC, Window, date_bounds
from leadgen.enrich.domain import name_matches
from leadgen.enrich.people import title_weight
from leadgen.http import HttpClient
from leadgen.intent import IntentAssessment
from leadgen.llm import LLM, LLMTransient
from leadgen.models import Person, normalize_url
from leadgen.sources.reddit import RedditSource


def test_midnight_timestamps_mean_the_whole_day():
    day = datetime(2026, 10, 9, tzinfo=UTC)
    for text in ("2026-10-09T00:00:00Z", "2026-10-09 00:00:00+00:00", "2026-10-09T00:00:00.000Z", "2026-10-09"):
        assert date_bounds(text, NOW) == (day, day + timedelta(days=1, seconds=-1)), text
    exact = datetime(2026, 10, 9, 9, 30, tzinfo=UTC)
    assert date_bounds("2026-10-09T09:30:00Z", NOW) == (exact, exact)
    # Today's date: the latest possible moment is now, not tonight.
    assert date_bounds("2026-10-12", NOW)[1] == NOW


def test_facebook_hosts_share_one_key():
    keys = {normalize_url(u) for u in (
        "https://www.facebook.com/groups/1/posts/2/",
        "https://m.facebook.com/groups/1/posts/2/",
        "https://web.facebook.com/groups/1/posts/2/",
        "https://fb.com/groups/1/posts/2/",
    )}
    assert len(keys) == 1
    assert normalize_url("https://facebook.com/permalink.php?story_fbid=1&id=2") != normalize_url(
        "https://facebook.com/permalink.php?story_fbid=3&id=2")


@pytest.mark.parametrize("title, weight", [
    ("Founder & Lead Designer", 1.0),       # founders stay decision-makers whatever else they do
    ("Owner", 1.0),
    ("Executive Assistant to the CEO", 0.0),
    ("Product Owner", 0.0),
    ("VP Engineering", 0.0),
    ("VP of Marketing", 0.9),
    ("CEO", 0.95),
])
def test_title_weights(title, weight):
    assert title_weight(title) == weight


def test_short_surnames_are_not_mistaken_for_suffixes():
    assert Person("Linh Do", None).first_last == ("Linh", "Do")
    assert Person("Mark Smith Jr.", None).first_last == ("Mark", "Smith")


def test_domain_needs_a_distinctive_word_of_the_business():
    assert not name_matches("Bright Smiles Dental Clinic", "dentalclinic.com")
    assert name_matches("Bright Smiles Dental Clinic", "brightsmiles.com")


def test_robots_answers_expire(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(http_mod.time, "monotonic", lambda: clock["t"])
    session = _Session(503)
    client = HttpClient("ua", per_host_delay=0, session=session)
    assert not client.allowed("https://acme.test/team")
    session.robots_status = 200
    clock["t"] += http_mod.ROBOTS_FAILURE_TTL - 1
    assert not client.allowed("https://acme.test/team")  # still within the failure TTL
    clock["t"] += 2
    assert client.allowed("https://acme.test/team")  # asked again, now allowed


class FailingPages(PagedApi):
    def __init__(self, pages, fail_at):
        super().__init__(pages)
        self.fail_at = fail_at

    def get_json(self, url, **kw):
        if len(self.calls) + 1 == self.fail_at:
            self.calls.append(kw["params"])
            raise requests.ConnectionError("reset")
        return super().get_json(url, **kw)


def test_reddit_keeps_earlier_pages_when_a_later_one_fails():
    older = (NOW - timedelta(days=5)).timestamp()
    window = Window.from_days(14, 2, now=NOW)
    api = FailingPages([_listing(older, "t3_a", n=3)], fail_at=2)
    posts = list(RedditSource(RedditSourceConfig(subreddits=["forhire"]), api, anon_delay=0).collect(window, []))
    assert len(posts) == 3
    # A subreddit that fails outright is logged and skipped; the run goes on.
    first_fails = FailingPages([], fail_at=1)
    assert list(RedditSource(RedditSourceConfig(subreddits=["forhire"]), first_fails, anon_delay=0).collect(window, [])) == []


@pytest.mark.parametrize("status", [408, 409])
def test_timeouts_and_conflicts_are_transient(status):
    err = {"type": "error", "error": {"type": "api_error", "message": "try again"}}
    llm = LLM(LLMConfig(), client=mock_client(lambda b: (status, err), []))
    with pytest.raises(LLMTransient):
        llm.parse(IntentAssessment, system="s", user="u", effort="low")
