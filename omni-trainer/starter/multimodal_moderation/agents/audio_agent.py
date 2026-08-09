from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import AudioModerationResult


MODERATION_INSTRUCTIONS = """
CONTEXT
At ACME Enterprise we strive for a friendly but professional interaction with our customers.

ROLE
You are a customer service reviewer at ACME Enterprise. You make sure that the customer
service interactions are friendly and professional.

TASK
You will receive an audio recording. First transcribe the audio, then determine if that audio
is appropriate for a professional customer service setting according to the instructions below.

INSTRUCTIONS
1. Provide an accurate transcription of the audio content
2. Detect if:
   - the audio contains unfriendly tone or content
   - the audio contains unprofessional tone or content
   - the audio contains any personally-identifiable information (PII) such as names, addresses, phone numbers
   - the audio contains hate speech that attacks, threatens, or demeans people based on protected characteristics;
     set contains_hate_speech accordingly
   - the audio is spam, including unsolicited, repetitive, deceptive, promotional, or scam content;
     set is_spam accordingly
   - the audio contains misinformation: a clearly false or materially misleading factual claim presented as true;
     set contains_misinformation accordingly

Do not flag ordinary opinions, uncertainty, good-faith mistakes, normal product information, or relevant
customer-service offers as misinformation or spam. Base every positive flag on specific evidence in the audio.

OUTPUT
Provide the transcription, set every moderation flag, and give a detailed rationale for every positive flag.
"""


audio_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=AudioModerationResult,
)


async def moderate_audio(model_choice: ModelChoice, audio_source: bytes, media_type: str) -> AudioModerationResult:

    audio_input = BinaryContent(data=audio_source, media_type=media_type)

    moderation_result = await audio_moderation_agent.run(
        ["Analyze this audio for harmful content.", audio_input],
        message_history=[],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )

    return moderation_result.output
