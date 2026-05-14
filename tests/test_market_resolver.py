from datetime import datetime, timedelta, timezone

from market_resolver import _candidate_btc_5m_slugs, _infer_times_from_slug


def test_btc_5m_slug_timestamp_is_interval_start() -> None:
    start, end = _infer_times_from_slug("btc-updown-5m-1778690700")

    assert start == datetime(2026, 5, 13, 16, 45, tzinfo=timezone.utc)
    assert end == start + timedelta(minutes=5)


def test_candidate_btc_5m_slugs_include_current_interval_start() -> None:
    now = datetime(2026, 5, 14, 2, 2, 57, tzinfo=timezone.utc)
    slugs = _candidate_btc_5m_slugs(now)

    assert "btc-updown-5m-1778724000" in slugs
    assert "btc-updown-5m-1778724300" in slugs
