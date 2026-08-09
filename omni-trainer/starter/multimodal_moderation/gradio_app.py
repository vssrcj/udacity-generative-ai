"""
Gradio Chat Interface for ACME Customer Service Training

This module provides an interactive chat interface where users practice being
customer service agents. The AI plays the role of a customer with complaints,
and all service agent messages/media are moderated before being sent to the AI.

DATA FLOW:
1. Service agent (user) sends message/media → Gradio interface
2. Content sent to FastAPI moderation service (via HTTP)
3. If safe → Content sent to Gemini AI (which plays customer role)
4. AI customer response returned to user

KEY COMPONENTS:
- check_content_safety(): Calls FastAPI backend to moderate content
- ChatSessionWithTracing: Manages conversation state and tracing
- create_chat_interface(): Builds the Gradio UI
"""

import os
import requests
import gradio as gr
import uuid
from html import escape
from pathlib import Path
from typing import List, Tuple, Any
from pydantic_ai.messages import BinaryContent
import logging

from multimodal_moderation.env import USER_API_KEY, API_BASE_URL
from multimodal_moderation.tracing import setup_tracing, get_tracer, add_media_to_span
from multimodal_moderation.agents.customer_agent import customer_agent
from multimodal_moderation.utils import detect_file_type
from opentelemetry import trace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up tracing for observability with Phoenix
setup_tracing()
tracer = get_tracer(__name__)

# Constants
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Moderation Configuration
# Maps content types to their FastAPI backend endpoints and safety flags.
# This configuration determines which API endpoint to call for each content type
# and which flags indicate unsafe content.
EXTENDED_SAFETY_FLAGS = ["contains_hate_speech", "is_spam", "contains_misinformation"]
ANALYTICS_ENDPOINT = f"{API_BASE_URL}/api/v1/analytics/summary"

MODERATION_CONFIG = {
    "text": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_text",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii", *EXTENDED_SAFETY_FLAGS],
    },
    "image": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_image_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality", *EXTENDED_SAFETY_FLAGS],
    },
    "video": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_video_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality", *EXTENDED_SAFETY_FLAGS],
    },
    "audio": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_audio_file",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii", *EXTENDED_SAFETY_FLAGS],
    },
}


def _display_label(value: str) -> str:
    return value.replace("contains_", "").replace("is_", "").replace("_", " ").title()


def _render_analytics_dashboard(summary: dict[str, Any]) -> str:
    modality_rows = []
    for content_type in ("text", "image", "video", "audio"):
        stats = summary.get("by_content_type", {}).get(content_type, {})
        total = int(stats.get("total", 0))
        flagged = int(stats.get("flagged", 0))
        modality_rows.append(
            f"""
            <div class="analytics-row">
                <div class="analytics-row-label">
                    <span>{escape(content_type.title())}</span>
                    <strong>{flagged} / {total} flagged</strong>
                </div>
                <meter min="0" max="{max(total, 1)}" value="{flagged}">{flagged} of {total}</meter>
            </div>
            """
        )

    positive_flags = [(flag, int(count)) for flag, count in summary.get("by_flag", {}).items() if int(count) > 0]
    max_flag_count = max((count for _, count in positive_flags), default=1)
    if positive_flags:
        flag_rows = [
            f"""
            <div class="analytics-row">
                <div class="analytics-row-label">
                    <span>{escape(_display_label(flag))}</span>
                    <strong>{count}</strong>
                </div>
                <progress max="{max_flag_count}" value="{count}">{count}</progress>
            </div>
            """
            for flag, count in sorted(positive_flags, key=lambda item: (-item[1], item[0]))
        ]
    else:
        flag_rows = ['<p class="analytics-empty">No moderation flags have been triggered yet.</p>']

    recent_events = summary.get("recent_flagged", [])
    if recent_events:
        recent_rows = [
            f"""
            <tr>
                <td>{escape(str(event.get("timestamp", "")).replace("T", " ")[:19])} UTC</td>
                <td>{escape(str(event.get("content_type", "")).title())}</td>
                <td>{escape(", ".join(_display_label(flag) for flag in event.get("flags", [])))}</td>
            </tr>
            """
            for event in recent_events[:10]
        ]
        recent_content = f"""
            <div class="analytics-table-wrap">
                <table>
                    <thead><tr><th>Time</th><th>Modality</th><th>Triggered flags</th></tr></thead>
                    <tbody>{''.join(recent_rows)}</tbody>
                </table>
            </div>
        """
    else:
        recent_content = '<p class="analytics-empty">No flagged events have been recorded yet.</p>'

    return f"""
    <style>
        .analytics-dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 1rem; }}
        .analytics-panel {{ border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg); padding: 1rem; background: var(--block-background-fill); }}
        .analytics-panel h3 {{ margin: 0 0 0.75rem; }}
        .analytics-row {{ margin-bottom: 0.75rem; }}
        .analytics-row:last-child {{ margin-bottom: 0; }}
        .analytics-row-label {{ display: flex; justify-content: space-between; gap: 1rem; margin-bottom: 0.25rem; }}
        .analytics-row-label strong {{ font-weight: 500; }}
        .analytics-row meter, .analytics-row progress {{ width: 100%; accent-color: var(--primary-500); }}
        .analytics-recent {{ grid-column: 1 / -1; }}
        .analytics-table-wrap {{ overflow-x: auto; }}
        .analytics-table-wrap table {{ width: 100%; border-collapse: collapse; }}
        .analytics-table-wrap th, .analytics-table-wrap td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid var(--border-color-primary); }}
        .analytics-empty {{ color: var(--body-text-color-subdued); margin: 0; }}
    </style>
    <section class="analytics-dashboard" aria-label="Moderation analytics">
        <div class="analytics-panel">
            <h3>Flagged by modality</h3>
            {''.join(modality_rows)}
        </div>
        <div class="analytics-panel">
            <h3>Triggered flags</h3>
            {''.join(flag_rows)}
        </div>
        <div class="analytics-panel analytics-recent">
            <h3>Recent flagged events</h3>
            {recent_content}
        </div>
    </section>
    """


def load_analytics_dashboard() -> Tuple[int, int, int, str, str, str]:
    """Fetch the latest aggregate moderation analytics from the FastAPI backend."""
    try:
        response = requests.get(
            ANALYTICS_ENDPOINT,
            headers={"Authorization": f"Bearer {USER_API_KEY}"},
            timeout=10,
        )
        if not response.ok:
            raise RuntimeError(f"Analytics backend returned HTTP {response.status_code}")

        summary = response.json()
        generated_at = str(summary.get("generated_at", "")).replace("T", " ")[:19]
        status = f"Last refreshed: {generated_at} UTC. Counts reset when the backend restarts."
        return (
            int(summary["total_requests"]),
            int(summary["flagged_requests"]),
            int(summary["safe_requests"]),
            f"{float(summary['flag_rate']):.1%}",
            _render_analytics_dashboard(summary),
            status,
        )
    except (requests.RequestException, RuntimeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Unable to load moderation analytics: %s", exc)
        empty_summary = {"by_content_type": {}, "by_flag": {}, "recent_flagged": []}
        return (
            0,
            0,
            0,
            "0.0%",
            _render_analytics_dashboard(empty_summary),
            "Analytics are temporarily unavailable. Start the FastAPI backend and refresh.",
        )


def _call_text_moderation(text: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    """
    Call the FastAPI backend to moderate text content.

    Makes an HTTP POST request to the text moderation endpoint.

    Returns: (result_dict, feedback, content_type, mime_type)
    Raises: RuntimeError if the moderation service is unavailable
    """
    content_type = "text"
    mime_type = "text/plain"

    config = MODERATION_CONFIG[content_type]

    # BACKEND CALL: HTTP POST to FastAPI moderation service
    response = requests.post(
        config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, json={"text": text}
    )

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    # Add input to tracing span for observability
    span.set_attributes(
        {
            "input.text.content": text,
            "input.text.length": len(text),
        }
    )

    # Parse the moderation result from the backend
    result = response.json()
    feedback = result["rationale"]

    return result, feedback, content_type, mime_type


def _call_media_moderation(media: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    """
    Call the FastAPI backend to moderate media files (image/video/audio).

    Detects file type, validates size, and sends to appropriate endpoint.

    Returns: (result_dict, feedback, content_type, mime_type)
    Raises: ValueError if file is too large, RuntimeError if service unavailable
    """
    # Detect MIME type (e.g., "image/png", "video/mp4", "audio/mpeg")
    mime_type = detect_file_type(media, context=media)

    # Validate file size to prevent large uploads
    file_size = os.path.getsize(media)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(f"File too large: {size_mb:.1f}MB. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    # Extract content type from MIME type (e.g., "image" from "image/png")
    content_type = mime_type.split("/")[0]
    config = MODERATION_CONFIG[content_type]

    # BACKEND CALL: HTTP POST with file upload to FastAPI moderation service
    with open(media, "rb") as f:
        response = requests.post(
            config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, files={"file": f}
        )

    # Add media metadata to tracing span for Phoenix visualization
    add_media_to_span(span, media, f"{content_type}_moderation", 0)

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    # Parse the moderation result from the backend
    result = response.json()
    feedback = result["rationale"]

    # Special case for audio: include transcription in feedback
    if content_type == "audio" and "transcription" in result:
        feedback = f"Transcription: \"{result['transcription']}\"\n\n{feedback}"

    return result, feedback, content_type, mime_type


def check_content_safety(*, text: str | None = None, media: str | None = None) -> Tuple[bool, str, str]:
    """
    Check if content is safe by calling the moderation backend.

    This is the main entry point for all content moderation. It routes to
    either text or media moderation, then checks if any safety flags are set.

    Args:
        text: Text content to moderate (mutually exclusive with media)
        media: Path to media file to moderate (mutually exclusive with text)

    Returns:
        Tuple of (is_safe, feedback_message, mime_type)
    """
    # Start with a text span name, then update it after detecting the content type.
    with tracer.start_as_current_span("moderate_text") as span:

        # Route to the appropriate moderation function
        if text is not None:
            result, feedback, content_type, mime_type = _call_text_moderation(text, span)

        elif media is not None:
            result, feedback, content_type, mime_type = _call_media_moderation(media, span)

        else:
            raise ValueError("Must provide exactly one of text or media")

        # Add moderation results to tracing span
        span.set_attributes({f"output.{k}": v for k, v in result.items()})

        # Update span name now that we know the content type
        span.update_name(f"moderate_{content_type}")

    # Check if any unsafe flags were set by the moderation service
    config = MODERATION_CONFIG[content_type]
    for flag in config["unsafe_flags"]:
        if result[flag]:
            # Content is unsafe - return False with feedback
            return False, f"Content flagged: {feedback}", mime_type

    # Content is safe - return True with feedback
    return True, feedback, mime_type


class ChatSessionWithTracing:
    """
    Manages a chat session with tracing support.

    Each session has a unique ID and a root tracing span that encompasses
    all chat turns. This allows Phoenix to group all related interactions.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        # Keep the root span open across turns and close it when the session ends.
        self.conversation_span = tracer.start_span(
            "conversation",
            attributes={"session.id": self.session_id},
        )

    async def chat_with_gemini(self, message: dict, history: List, past_messages: List) -> Tuple[str, List, str]:
        """
        Process a chat turn: moderate content, then send to AI customer.

        DATA FLOW:
        1. Receive customer service agent message/files from Gradio
        2. Moderate each piece of content (text and/or media files) via FastAPI backend
        3. If any content is flagged, block and return error message
        4. If all content is safe, send to Gemini AI (which plays the role of a customer)
        5. Return AI customer's response

        Args:
            message: dict with 'text' and optional 'files' keys from Gradio
            history: Gradio's chat history (for display only, not used)
            past_messages: Pydantic AI's message history (used for agent context)

        Returns:
            Tuple of (response_text, updated_messages, feedback_text)
        """
        conversation_context = trace.set_span_in_context(self.conversation_span)
        with tracer.start_as_current_span("chat_turn", context=conversation_context) as span:

            logger.info(f"New turn - Text: '{message.get('text', '')[:50]}...', Files: {len(message.get('files', []))}")

            # Build prompt for the AI customer (includes text and media)
            prompt_parts: List[str | BinaryContent] = [
                "This is the next message from the support agent:",
            ]

            # Initialize safety message to empty string, will be set if content is flagged
            safety_message = ""

            # Process each part of the message
            for key, value in message.items():

                # MODERATION STEP 1: Check text content
                if key == "text" and value:

                    # Call moderation backend (FastAPI service)
                    is_safe, safety_message, mime_type = check_content_safety(text=value)

                    if not is_safe:
                        # Content flagged - block and return error
                        feedback = f"⚠️ Content flagged: {safety_message}"
                        response = "[This content was flagged by moderation and not sent to the AI. Please try again.]"

                        span.set_attribute("feedback", feedback)

                        return response, past_messages, feedback

                    # Content safe - add to prompt
                    prompt_parts.append(value)

                # MODERATION STEP 2: Check media files
                elif key == "files" and value:

                    for file_path in value:

                        try:

                            # Call moderation backend (FastAPI service)
                            is_safe, safety_message, mime_type = check_content_safety(media=file_path)

                            if not is_safe:
                                # Content flagged - block and return error
                                feedback = f"⚠️ Content flagged: {safety_message}"
                                response = (
                                    "[This content was flagged by moderation and not sent to the AI. Please try again.]"
                                )

                                return response, past_messages, feedback

                            # Content safe - read file and add to prompt
                            with open(file_path, "rb") as f:
                                file_bytes = f.read()

                            prompt_parts.append(BinaryContent(data=file_bytes, media_type=mime_type))

                        except ValueError as e:
                            raise gr.Error(str(e))

            if not prompt_parts:
                raise gr.Error("Please provide a message or at least one file.")

            # All content passed moderation - send to AI customer
            try:
                with tracer.start_as_current_span("llm_customer"):

                    # GEMINI CALL: Send the prompt and conversation history to the AI customer.
                    result = await customer_agent.run(
                        prompt_parts,
                        message_history=past_messages,
                    )

                logger.info(f"Response generated ({len(result.all_messages())} messages in history)")
                return result.output, result.all_messages(), safety_message

            except Exception as e:
                logger.error(f"Error in chat_with_gemini: {str(e)}")
                raise gr.Error(
                    f"I'm sorry, but I encountered an error while processing your request. "
                    f"Please try again or contact ACME support if the issue persists."
                )

    def end_conversation(self):
        """
        End the conversation and close the tracing span.

        This should be called when the user ends the chat session to properly
        close the conversation span in Phoenix.
        """
        if self.conversation_span:
            self.conversation_span.end()
            logger.info(f"Conversation {self.session_id} ended")
        return "Conversation ended. Refresh the page to start a new session."


def create_chat_interface() -> gr.Blocks:
    """
    Create the Gradio chat interface with moderation feedback.

    LAYOUT:
    - Left: Chat interface (ChatInterface with MultimodalTextbox)
    - Right: Moderation feedback and guidelines sidebar

    STATE MANAGEMENT:
    - past_messages_state: Holds Pydantic AI message history across turns
    - feedback_display: Shows moderation results from the backend
    """
    # Create a chat session that tracks conversation across multiple turns
    chat_session = ChatSessionWithTracing()

    with gr.Blocks(title="ACME Customer Service Training Agent", fill_height=True) as demo:
        # State to hold Pydantic AI's message history (preserves context across turns)
        past_messages_state = gr.State([])

        # Create feedback_display first (with render=False) so we can reference it
        # in ChatInterface's additional_outputs below, then render it in the sidebar later
        feedback_display = gr.Textbox(
            label="💬 Moderation Agent Feedback",
            placeholder="No feedback yet",
            interactive=False,
            visible=True,
            lines=10,
            render=False,  # Don't render yet - will render in sidebar
        )

        # UI Layout
        gr.Markdown("# 🤖 ACME Customer Service Training Agent")
        gr.Markdown("Welcome to ACME Corporation's customer service training!")

        with gr.Row():
            # Left column: Chat interface (75% width)
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_session.chat_with_gemini,
                    type="messages",  # Use newer messages format (supports multimodal)
                    multimodal=True,
                    editable=False,  # Don't allow editing past messages
                    textbox=gr.MultimodalTextbox(
                        file_count="multiple",  # Allow multiple files
                        file_types=["image", "video", "audio"],
                        sources=["upload", "microphone"],  # Allow file upload and recording
                        placeholder="Type a message, upload files, or record audio...",
                    ),
                    chatbot=gr.Chatbot(
                        show_copy_button=True,
                        type="messages",  # Use messages format for multimodal support
                        placeholder="👋 Start by greeting the customer or introducing yourself. The AI customer will respond with their complaint.",
                        height="75vh",
                    ),
                    additional_inputs=[past_messages_state],
                    additional_outputs=[past_messages_state, feedback_display],
                )

            # Right column: Feedback and guidelines (25% width)
            with gr.Column(scale=1):
                # Render the feedback display at the top of the sidebar
                feedback_display.render()

                # End conversation button - closes the tracing span
                end_button = gr.Button("📞 End Conversation", variant="secondary")
                end_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                )

                gr.Markdown("### 📋 Chat Guidelines")
                gr.Markdown(
                    """
                The AI acts as a customer complaining about an ACME product. Try to resolve the customer's issue.
                You can type messages, upload images/videos, or record audio.
                """
                )

                gr.Markdown("### 🔒 Content Moderation")
                gr.Markdown(
                    """
                All messages and media are automatically checked for:
                - Inappropriate content
                - Personally identifiable information
                - Unprofessional language
                - Hate speech
                - Spam or scams
                - Misinformation
                """
                )

        with gr.Accordion("📊 Moderation Analytics", open=False):
            gr.Markdown(
                "Aggregate outcomes from the moderation backend. Raw messages, media, and rationales are not stored."
            )
            refresh_analytics = gr.Button("Refresh analytics", variant="secondary")

            with gr.Row():
                total_requests = gr.Number(label="Total moderated", value=0, precision=0, interactive=False)
                flagged_requests = gr.Number(label="Flagged", value=0, precision=0, interactive=False)
                safe_requests = gr.Number(label="Passed", value=0, precision=0, interactive=False)
                flag_rate = gr.Textbox(label="Flag rate", value="0.0%", interactive=False)

            analytics_dashboard = gr.HTML(
                value=_render_analytics_dashboard({"by_content_type": {}, "by_flag": {}, "recent_flagged": []})
            )
            analytics_status = gr.Markdown("Analytics will load from the backend when the page opens.")

        analytics_outputs = [
            total_requests,
            flagged_requests,
            safe_requests,
            flag_rate,
            analytics_dashboard,
            analytics_status,
        ]

        refresh_analytics.click(fn=load_analytics_dashboard, outputs=analytics_outputs)
        demo.load(fn=load_analytics_dashboard, outputs=analytics_outputs)

        # Wire up the end conversation button
        end_button.click(fn=chat_session.end_conversation, outputs=end_status).then(
            lambda: gr.Textbox(visible=True), outputs=end_status
        )

    return demo


def main():
    """Main function to run the Gradio app"""
    demo = create_chat_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
