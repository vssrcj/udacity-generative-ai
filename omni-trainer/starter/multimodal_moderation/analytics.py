"""In-memory, privacy-conscious moderation analytics."""

from collections import Counter, deque
from datetime import datetime, timezone
from threading import Lock

from pydantic import BaseModel, Field

from multimodal_moderation.types.moderation_result import (
    AudioModerationResult,
    ImageModerationResult,
    ModerationResult,
    TextModerationResult,
    VideoModerationResult,
)


CONTENT_TYPES = ("text", "image", "video", "audio")
RESULT_TYPES = (
    TextModerationResult,
    ImageModerationResult,
    VideoModerationResult,
    AudioModerationResult,
)
MODERATION_FLAGS = tuple(
    sorted(
        {
            field_name
            for result_type in RESULT_TYPES
            for field_name, field in result_type.model_fields.items()
            if field.annotation is bool
        }
    )
)


class ContentTypeStats(BaseModel):
    total: int = 0
    flagged: int = 0
    safe: int = 0


class FlaggedEvent(BaseModel):
    timestamp: datetime
    content_type: str
    flags: list[str]


class AnalyticsSummary(BaseModel):
    generated_at: datetime
    total_requests: int
    flagged_requests: int
    safe_requests: int
    flag_rate: float = Field(ge=0, le=1)
    by_content_type: dict[str, ContentTypeStats]
    by_flag: dict[str, int]
    recent_flagged: list[FlaggedEvent]


class ModerationAnalytics:
    """Collect aggregate moderation outcomes without retaining user content."""

    def __init__(self, recent_limit: int = 20):
        self._lock = Lock()
        self._recent_limit = recent_limit
        self.clear()

    def clear(self) -> None:
        """Reset counters. Intended for process startup and isolated tests."""
        with self._lock:
            self._total_requests = 0
            self._flagged_requests = 0
            self._by_content_type = {
                content_type: Counter(total=0, flagged=0, safe=0) for content_type in CONTENT_TYPES
            }
            self._by_flag = Counter({flag: 0 for flag in MODERATION_FLAGS})
            self._recent_flagged: deque[FlaggedEvent] = deque(maxlen=self._recent_limit)

    def record(self, content_type: str, result: ModerationResult) -> None:
        if content_type not in CONTENT_TYPES:
            raise ValueError(f"Unsupported analytics content type: {content_type}")

        result_data = result.model_dump()
        triggered_flags = sorted(
            field_name for field_name, value in result_data.items() if field_name in MODERATION_FLAGS and value is True
        )
        is_flagged = bool(triggered_flags)

        with self._lock:
            self._total_requests += 1
            self._by_content_type[content_type]["total"] += 1

            if is_flagged:
                self._flagged_requests += 1
                self._by_content_type[content_type]["flagged"] += 1
                self._by_flag.update(triggered_flags)
                self._recent_flagged.appendleft(
                    FlaggedEvent(
                        timestamp=datetime.now(timezone.utc),
                        content_type=content_type,
                        flags=triggered_flags,
                    )
                )
            else:
                self._by_content_type[content_type]["safe"] += 1

    def summary(self) -> AnalyticsSummary:
        with self._lock:
            safe_requests = self._total_requests - self._flagged_requests
            flag_rate = self._flagged_requests / self._total_requests if self._total_requests else 0

            return AnalyticsSummary(
                generated_at=datetime.now(timezone.utc),
                total_requests=self._total_requests,
                flagged_requests=self._flagged_requests,
                safe_requests=safe_requests,
                flag_rate=flag_rate,
                by_content_type={
                    content_type: ContentTypeStats(**counts) for content_type, counts in self._by_content_type.items()
                },
                by_flag={flag: self._by_flag[flag] for flag in MODERATION_FLAGS},
                recent_flagged=list(self._recent_flagged),
            )


analytics_store = ModerationAnalytics()
