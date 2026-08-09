from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from multimodal_moderation.types.moderation_result import TextModerationResult


@dataclass
class TextModerationCheck(Evaluator):
    expected_pii: bool
    expected_unfriendly: bool
    expected_unprofessional: bool
    # Extended safety flags default to False so pre-existing cases keep their labels
    expected_hate_speech: bool = False
    expected_spam: bool = False
    expected_misinformation: bool = False

    async def evaluate(self, ctx: EvaluatorContext[str, TextModerationResult]) -> bool:
        return (
            ctx.output.contains_pii == self.expected_pii
            and ctx.output.is_unfriendly == self.expected_unfriendly
            and ctx.output.is_unprofessional == self.expected_unprofessional
            and ctx.output.contains_hate_speech == self.expected_hate_speech
            and ctx.output.is_spam == self.expected_spam
            and ctx.output.contains_misinformation == self.expected_misinformation
        )
