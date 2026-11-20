import json

import pytest

from src.dashboard.leaderboards import (
    LeaderboardFetchError,
    _parse_arena_html,
    _parse_benchlm_json,
    _parse_epoch_csv,
)


def test_arena_html_parser_extracts_ranked_rows_and_date():
    html = b"""
    <p>Nov 20, 2026</p>
    <ul>
      <li aria-label="Model A (High): 1,527, 95% CI 1,517 to 1,537"></li>
      <li aria-label="Model B: 1,520, 95% CI 1,515 to 1,525"></li>
      <li aria-label="Model A (High): 1,527, 95% CI 1,517 to 1,537"></li>
    </ul>
    """

    rows, updated_at = _parse_arena_html(html)

    assert updated_at == "Nov 20, 2026"
    assert rows == [
        {
            "rank": 1,
            "model": "Model A (High)",
            "score": "1,527 (95% CI 1,517 to 1,537)",
            "updated_at": "Nov 20, 2026",
        },
        {
            "rank": 2,
            "model": "Model B",
            "score": "1,520 (95% CI 1,515 to 1,525)",
            "updated_at": "Nov 20, 2026",
        },
    ]


def test_arena_html_parser_reports_changed_page_shape():
    with pytest.raises(LeaderboardFetchError, match="nem találtam rangsorsorokat"):
        _parse_arena_html(b"<html><body>No leaderboard rows</body></html>")


def test_benchlm_parser_returns_eligible_rows_with_attribution_fields():
    payload = json.dumps({
        "license": "CC BY-NC 4.0",
        "sourceLastUpdated": "November 19, 2026",
        "items": [
            {"rank": 2, "model": "Model B", "score": 40, "rankingEligible": True},
            {
                "rank": 1,
                "model": "Model A",
                "score": 42.5,
                "rankingEligible": True,
                "scoreInterval90": {"lower": 40, "upper": 44},
                "evidenceStatus": "supported",
            },
            {"rank": 3, "model": "Provisional", "score": 39, "rankingEligible": False},
        ],
    }).encode("utf-8")

    rows, updated_at = _parse_benchlm_json(payload)

    assert updated_at == "November 19, 2026"
    assert rows == [
        {
            "rank": 1,
            "model": "Model A",
            "score": "42.5 (90% CI 40-44) [supported]",
            "updated_at": "November 19, 2026",
        },
        {
            "rank": 2,
            "model": "Model B",
            "score": "40",
            "updated_at": "November 19, 2026",
        },
    ]


def test_benchlm_parser_rejects_unexpected_license():
    payload = json.dumps({"license": "CC BY 4.0", "items": []}).encode("utf-8")

    with pytest.raises(LeaderboardFetchError, match="adatlicence megváltozott"):
        _parse_benchlm_json(payload)


def test_epoch_parser_sorts_by_eci_and_keeps_model_dates_and_intervals():
    payload = (
        "Model,Display name,eci,eci_ci_low,eci_ci_high,date\n"
        "Model B,Model B,160.0,158.0,162.0,2026-10-10\n"
        "Model A,Model A,174.17,170.29,178.58,2026-11-17\n"
    ).encode("utf-8")

    rows, latest = _parse_epoch_csv(payload)

    assert latest == "2026-11-17"
    assert rows == [
        {
            "rank": 1,
            "model": "Model A",
            "score": "174.17 (CI 170.29-178.58)",
            "updated_at": "2026-11-17",
        },
        {
            "rank": 2,
            "model": "Model B",
            "score": "160.00 (CI 158.0-162.0)",
            "updated_at": "2026-10-10",
        },
    ]


def test_epoch_parser_rejects_unexpected_schema():
    with pytest.raises(LeaderboardFetchError, match="oszlopai megváltoztak"):
        _parse_epoch_csv(b"Name,score\nModel A,1\n")
