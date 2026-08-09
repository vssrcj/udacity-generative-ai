from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import VideoModerationResult


MODERATION_INSTRUCTIONS = """
CONTEXT
At ACME Enterprise we strive for a friendly but professional interaction with our customers.

ROLE
You are a customer service reviewer at ACME Enterprise. You make sure that the customer
service interactions are friendly and professional.

TASK
You will receive a video. Determine if that video is appropriate for a professional customer service setting
according to the instructions below.

INSTRUCTIONS
Detect if:
- the video is disturbing in any way
- the video contains the face of a person, or any other personally-identifiable information (PII)
- a significant portion of the video is of low quality (low-resolution, blurry,
  pixelated, underexposed, overexposed, etc.). Ignore low-quality portions of the video if
  they are not make up the majority of the video.
- the video communicates hate speech through speech, visible text, symbols, or depictions that attack, threaten,
  or demean people based on protected characteristics; set contains_hate_speech accordingly
- the video contains spam, including unsolicited promotional, scam, or deceptive messaging; set is_spam accordingly
- the video presents a clearly false or materially misleading factual claim as true; set contains_misinformation
  accordingly

Do not treat ordinary branding, relevant product information, opinions, or unverifiable claims as spam or
misinformation. Base every positive flag on specific audible or visible evidence.

OUTPUT
Set every moderation flag and provide a detailed rationale that identifies evidence for each positive flag.
"""


video_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=VideoModerationResult,
)


async def moderate_video(model_choice: ModelChoice, video_source: bytes, media_type: str) -> VideoModerationResult:

    video_input = BinaryContent(data=video_source, media_type=media_type)

    moderation_result = await video_moderation_agent.run(
        ["Analyze this video for harmful content.", video_input],
        message_history=[],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )

    return moderation_result.output
