from datetime import datetime, timedelta

from leadgen.dates import (
    UTC,
    X_EPOCH_MS,
    Window,
    datetime_from_post_url,
    linkedin_post_datetime,
    parse_date,
    x_post_datetime,
)

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def test_linkedin_activity_id_decodes_to_post_time():
    posted = datetime(2026, 10, 5, 14, 30, 15, 123000, tzinfo=UTC)
    post_id = (_ms(posted) << 22) | 0b1011011
    url = f"https://www.linkedin.com/posts/jane-doe-4a1b2c_hiring-smm-activity-{post_id}-AbCd"
    assert linkedin_post_datetime(url) == posted


def test_linkedin_feed_update_urn_and_url_encoded_forms():
    posted = datetime(2026, 9, 30, 8, 0, tzinfo=UTC)
    post_id = _ms(posted) << 22
    assert linkedin_post_datetime(f"https://www.linkedin.com/feed/update/urn:li:activity:{post_id}/") == posted
    assert linkedin_post_datetime(f"https://www.linkedin.com/feed/update/urn%3Ali%3AugcPost%3A{post_id}") == posted


def test_linkedin_without_id_returns_none():
    assert linkedin_post_datetime("https://www.linkedin.com/pulse/some-article-jane-doe") is None


def test_x_snowflake_decodes():
    posted = datetime(2026, 10, 1, 9, 15, tzinfo=UTC)
    status_id = ((_ms(posted) - X_EPOCH_MS) << 22) | 4095
    assert x_post_datetime(f"https://x.com/acmegym/status/{status_id}") == posted
    assert datetime_from_post_url(f"https://twitter.com/acmegym/status/{status_id}?s=20") == posted


def test_parse_relative_and_compact_forms():
    assert parse_date("3 days ago", NOW) == NOW - timedelta(days=3)
    assert parse_date("2d", NOW) == NOW - timedelta(days=2)
    assert parse_date("1w", NOW) == NOW - timedelta(weeks=1)
    assert parse_date("a week ago", NOW) == NOW - timedelta(weeks=1)
    assert parse_date("5 hours ago", NOW) == NOW - timedelta(hours=5)
    assert parse_date("2mo", NOW) == NOW - timedelta(days=60)
    assert parse_date("yesterday", NOW) == NOW - timedelta(days=1)


def test_parse_absolute_forms():
    assert parse_date("Oct 5, 2026", NOW) == datetime(2026, 10, 5, tzinfo=UTC)
    assert parse_date("October 5th, 2026", NOW) == datetime(2026, 10, 5, tzinfo=UTC)
    assert parse_date("2026-10-05T10:00:00Z", NOW) == datetime(2026, 10, 5, 10, tzinfo=UTC)
    assert parse_date("Mon, 05 Oct 2026 10:00:00 +0000", NOW) == datetime(2026, 10, 5, 10, tzinfo=UTC)
    assert parse_date("not a date", NOW) is None
    assert parse_date(None, NOW) is None


def test_window_boundaries():
    w = Window.from_days(14, 2, now=NOW)
    assert w.contains(NOW - timedelta(days=5))
    assert w.contains(NOW - timedelta(days=14))
    assert not w.contains(NOW - timedelta(days=14, seconds=1))
    assert not w.contains(NOW - timedelta(days=1))  # newer than min age
    assert not w.contains(None)
    assert w.describe() == "2026-09-28 to 2026-10-10"
