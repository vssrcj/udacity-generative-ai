"""Tests for persona-driven customer simulation."""

from unittest.mock import AsyncMock, MagicMock, patch

import gradio as gr
import pytest

from multimodal_moderation.gradio_app import (
    ChatSessionWithTracing,
    create_chat_interface,
    describe_customer_persona,
)
from multimodal_moderation.types.customer_persona import (
    CUSTOMER_PERSONAS,
    DEFAULT_CUSTOMER_PERSONA_ID,
    get_customer_persona,
)


def test_customer_persona_catalog_has_varied_scenarios_and_tones():
    assert len(CUSTOMER_PERSONAS) >= 4
    assert len({persona.id for persona in CUSTOMER_PERSONAS}) == len(CUSTOMER_PERSONAS)
    assert len({persona.emotional_tone for persona in CUSTOMER_PERSONAS}) == len(CUSTOMER_PERSONAS)
    assert len({persona.scenario for persona in CUSTOMER_PERSONAS}) == len(CUSTOMER_PERSONAS)


@pytest.mark.parametrize("persona", CUSTOMER_PERSONAS)
def test_persona_prompt_contains_its_tone_scenario_and_goal(persona):
    assert persona.emotional_tone in persona.system_prompt
    assert persona.scenario in persona.system_prompt
    assert persona.goal in persona.system_prompt
    assert persona.emotional_tone in describe_customer_persona(persona.id)


def test_unknown_persona_falls_back_to_original_refund_scenario():
    persona = get_customer_persona("not-a-real-persona")

    assert persona.id == DEFAULT_CUSTOMER_PERSONA_ID
    assert "refund" in persona.goal.lower()


def test_gradio_interface_exposes_all_customer_personas():
    demo = create_chat_interface()
    persona_dropdown = next(
        (
            block
            for block in demo.blocks.values()
            if isinstance(block, gr.Dropdown) and block.label == "🎭 Customer Persona"
        ),
        None,
    )

    assert persona_dropdown is not None
    choice_values = {choice[1] if isinstance(choice, (list, tuple)) else choice for choice in persona_dropdown.choices}
    assert {persona.id for persona in CUSTOMER_PERSONAS} <= choice_values
    assert persona_dropdown.value == DEFAULT_CUSTOMER_PERSONA_ID


async def test_chat_passes_selected_persona_to_customer_agent():
    mock_result = MagicMock()
    mock_result.output = "I need this fixed before the presentation."
    mock_result.all_messages.return_value = []

    with (
        patch(
            "multimodal_moderation.gradio_app.check_content_safety",
            return_value=(True, "Content passed moderation", "text/plain"),
        ),
        patch(
            "multimodal_moderation.gradio_app.customer_agent.run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_agent,
    ):
        session = ChatSessionWithTracing()
        response, messages, feedback = await session.chat_with_gemini(
            {"text": "How can I help?"},
            [],
            [],
            "business_urgent",
        )

    assert response == mock_result.output
    assert messages == []
    assert feedback == "Content passed moderation"
    assert mock_agent.await_args.kwargs["deps"].id == "business_urgent"
    assert mock_agent.await_args.kwargs["message_history"] == []
