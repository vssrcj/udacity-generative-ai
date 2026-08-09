"""Tests for backend moderation analytics and dashboard rendering."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from multimodal_moderation.analytics import ModerationAnalytics, analytics_store
from multimodal_moderation.env import USER_API_KEY
from multimodal_moderation.fastapi_app import app
from multimodal_moderation.gradio_app import ANALYTICS_ENDPOINT, load_analytics_dashboard
from multimodal_moderation.types.moderation_result import ImageModerationResult, TextModerationResult


def _safe_text_result() -> TextModerationResult:
    return TextModerationResult(
        rationale="Professional content",
        contains_pii=False,
        is_unfriendly=False,
        is_unprofessional=False,
    )


def _spam_text_result() -> TextModerationResult:
    return TextModerationResult(
        rationale="Unsolicited promotion",
        contains_pii=False,
        is_unfriendly=False,
        is_unprofessional=True,
        is_spam=True,
    )


def test_analytics_summary_counts_outcomes_without_content():
    store = ModerationAnalytics()
    store.record("text", _safe_text_result())
    store.record("text", _spam_text_result())
    store.record(
        "image",
        ImageModerationResult(
            rationale="A person is visible",
            contains_pii=True,
            is_disturbing=False,
            is_low_quality=False,
        ),
    )

    summary = store.summary()

    assert summary.total_requests == 3
    assert summary.flagged_requests == 2
    assert summary.safe_requests == 1
    assert summary.flag_rate == 2 / 3
    assert summary.by_content_type["text"].model_dump() == {"total": 2, "flagged": 1, "safe": 1}
    assert summary.by_content_type["image"].model_dump() == {"total": 1, "flagged": 1, "safe": 0}
    assert summary.by_flag["is_spam"] == 1
    assert summary.by_flag["contains_pii"] == 1
    assert len(summary.recent_flagged) == 2
    assert "rationale" not in summary.recent_flagged[0].model_dump()


def test_backend_endpoint_records_and_returns_analytics():
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {USER_API_KEY}"}
    analytics_store.clear()

    try:
        with patch(
            "multimodal_moderation.fastapi_app.moderate_text",
            new=AsyncMock(return_value=_spam_text_result()),
        ):
            moderation_response = client.post(
                "/api/v1/moderate_text",
                headers=headers,
                json={"text": "promotional message"},
            )

        assert moderation_response.status_code == 200

        analytics_response = client.get("/api/v1/analytics/summary", headers=headers)
        assert analytics_response.status_code == 200

        summary = analytics_response.json()
        assert summary["total_requests"] == 1
        assert summary["flagged_requests"] == 1
        assert summary["by_content_type"]["text"] == {"total": 1, "flagged": 1, "safe": 0}
        assert summary["by_flag"]["is_spam"] == 1
    finally:
        analytics_store.clear()


def test_dashboard_fetches_and_renders_backend_summary():
    response = MagicMock(ok=True, status_code=200)
    response.json.return_value = {
        "generated_at": "2026-08-09T10:00:00+00:00",
        "total_requests": 4,
        "flagged_requests": 2,
        "safe_requests": 2,
        "flag_rate": 0.5,
        "by_content_type": {
            "text": {"total": 3, "flagged": 2, "safe": 1},
            "image": {"total": 1, "flagged": 0, "safe": 1},
        },
        "by_flag": {"is_spam": 2, "contains_pii": 1},
        "recent_flagged": [
            {
                "timestamp": "2026-08-09T09:59:00+00:00",
                "content_type": "text",
                "flags": ["is_spam"],
            }
        ],
    }

    with patch("multimodal_moderation.gradio_app.requests.get", return_value=response) as mock_get:
        total, flagged, safe, rate, dashboard, status = load_analytics_dashboard()

    assert (total, flagged, safe, rate) == (4, 2, 2, "50.0%")
    assert "Flagged by modality" in dashboard
    assert "Spam" in dashboard
    assert "2 / 3 flagged" in dashboard
    assert "Last refreshed" in status
    mock_get.assert_called_once_with(
        ANALYTICS_ENDPOINT,
        headers={"Authorization": f"Bearer {USER_API_KEY}"},
        timeout=10,
    )
