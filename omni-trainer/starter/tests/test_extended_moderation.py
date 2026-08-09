"""Regression tests for the extended moderation policy."""

import pytest

from multimodal_moderation.agents.audio_agent import MODERATION_INSTRUCTIONS as AUDIO_INSTRUCTIONS
from multimodal_moderation.agents.image_agent import MODERATION_INSTRUCTIONS as IMAGE_INSTRUCTIONS
from multimodal_moderation.agents.text_agent import MODERATION_INSTRUCTIONS as TEXT_INSTRUCTIONS
from multimodal_moderation.agents.video_agent import MODERATION_INSTRUCTIONS as VIDEO_INSTRUCTIONS
from multimodal_moderation.gradio_app import EXTENDED_SAFETY_FLAGS, MODERATION_CONFIG
from multimodal_moderation.types.moderation_result import (
    AudioModerationResult,
    ImageModerationResult,
    TextModerationResult,
    VideoModerationResult,
)


RESULT_CASES = [
    (
        TextModerationResult,
        {
            "rationale": "Safe text",
            "contains_pii": False,
            "is_unfriendly": False,
            "is_unprofessional": False,
        },
    ),
    (
        ImageModerationResult,
        {
            "rationale": "Safe image",
            "contains_pii": False,
            "is_disturbing": False,
            "is_low_quality": False,
        },
    ),
    (
        VideoModerationResult,
        {
            "rationale": "Safe video",
            "contains_pii": False,
            "is_disturbing": False,
            "is_low_quality": False,
        },
    ),
    (
        AudioModerationResult,
        {
            "rationale": "Safe audio",
            "transcription": "Hello",
            "contains_pii": False,
            "is_unfriendly": False,
            "is_unprofessional": False,
        },
    ),
]


@pytest.mark.parametrize(("result_type", "required_fields"), RESULT_CASES)
def test_extended_flags_are_backward_compatible(result_type, required_fields):
    result = result_type(**required_fields)

    for flag in EXTENDED_SAFETY_FLAGS:
        assert getattr(result, flag) is False
        assert flag in result_type.model_json_schema()["properties"]


@pytest.mark.parametrize(("result_type", "required_fields"), RESULT_CASES)
def test_extended_flags_accept_positive_results(result_type, required_fields):
    result = result_type(
        **required_fields,
        contains_hate_speech=True,
        is_spam=True,
        contains_misinformation=True,
    )

    assert result.contains_hate_speech is True
    assert result.is_spam is True
    assert result.contains_misinformation is True


def test_gradio_blocks_extended_flags_for_every_modality():
    for config in MODERATION_CONFIG.values():
        assert set(EXTENDED_SAFETY_FLAGS).issubset(config["unsafe_flags"])


@pytest.mark.parametrize(
    "instructions",
    [TEXT_INSTRUCTIONS, IMAGE_INSTRUCTIONS, AUDIO_INSTRUCTIONS, VIDEO_INSTRUCTIONS],
)
def test_every_agent_is_instructed_about_extended_flags(instructions):
    normalized_instructions = instructions.lower()

    assert "hate speech" in normalized_instructions
    assert "spam" in normalized_instructions
    assert "misinformation" in normalized_instructions
