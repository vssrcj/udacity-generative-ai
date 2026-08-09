from pydantic_ai import Agent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import TextModerationResult


MODERATION_INSTRUCTIONS = """
<context>
At ACME enterprise we strive for a friendly but professional interaction with our customers.
</context>

<role>
You are a customer service reviewer at ACME enterprise. You make sure that the customer
service interactions are friendly and professional.
</role>

<input>
You will receive a message from the customer representative towards the customer.
</input>

<instructions>
Detect if:
- the tone of the message is unfriendly
- the tone of the message is unprofessional
- the message contains any personally-identifiable information (PII)
- the message contains hate speech that attacks, threatens, or demeans people based on protected characteristics;
  set contains_hate_speech accordingly
- the message is spam, including unsolicited, repetitive, irrelevant, deceptive, promotional, or scam content;
  set is_spam accordingly
- the message contains misinformation: a clearly false or materially misleading factual claim presented as true;
  set contains_misinformation accordingly

Do not flag ordinary opinions, uncertainty, good-faith mistakes, normal product information, or relevant customer-service
offers as misinformation or spam. Base every positive flag on specific evidence in the message.
</instructions>

<output>
Set every moderation flag and provide a detailed rationale that identifies evidence for each positive flag.
</output>
"""


text_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=TextModerationResult,
)


async def moderate_text(model_choice: ModelChoice, text: str) -> TextModerationResult:
    moderation_result = await text_moderation_agent.run(
        f"Analyze this text for harmful content:\n\n{text}",
        message_history=[],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )

    return moderation_result.output
