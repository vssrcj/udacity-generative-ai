from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from multimodal_moderation.types.moderation_result import ImageModerationResult


@dataclass
class ImageModerationCheck(Evaluator):
    expected_pii: bool
    expected_disturbing: bool
    expected_low_quality: bool
    # Extended safety flags default to False so pre-existing cases keep their labels
    expected_hate_speech: bool = False
    expected_spam: bool = False
    expected_misinformation: bool = False

    async def evaluate(self, ctx: EvaluatorContext[str, ImageModerationResult]) -> bool:
        return (
            ctx.output.contains_pii == self.expected_pii
            and ctx.output.is_disturbing == self.expected_disturbing
            and ctx.output.is_low_quality == self.expected_low_quality
            and ctx.output.contains_hate_speech == self.expected_hate_speech
            and ctx.output.is_spam == self.expected_spam
            and ctx.output.contains_misinformation == self.expected_misinformation
        )
