from pydantic import BaseModel, Field


class ModerationResult(BaseModel):

    rationale: str = Field(description="Explanation of what was harmful and why")


class TextModerationResult(ModerationResult):

    contains_pii: bool = Field(description="Whether the message contains any personally-identifiable information (PII)")
    is_unfriendly: bool = Field(description="Whether unfriendly tone or content was detected")
    is_unprofessional: bool = Field(description="Whether unprofessional tone or content was detected")
    contains_hate_speech: bool = Field(
        default=False,
        description="Whether the message attacks or demeans people based on protected characteristics",
    )
    is_spam: bool = Field(
        default=False,
        description="Whether the message is unsolicited, repetitive, deceptive, or irrelevant promotional content",
    )
    contains_misinformation: bool = Field(
        default=False,
        description="Whether the message presents clearly false or misleading factual claims as true",
    )


class ImageModerationResult(ModerationResult):

    contains_pii: bool = Field(
        description="Whether the image contains any person, part of a person, or personally-identifiable information (PII)"
    )
    is_disturbing: bool = Field(description="Whether the image is disturbing")
    is_low_quality: bool = Field(description="Whether the image is low quality")
    contains_hate_speech: bool = Field(
        default=False,
        description="Whether the image communicates hate through visible text, symbols, or depictions",
    )
    is_spam: bool = Field(
        default=False,
        description="Whether the image contains unsolicited promotional, scam, or spam messaging",
    )
    contains_misinformation: bool = Field(
        default=False,
        description="Whether the image communicates clearly false or misleading factual claims",
    )


class VideoModerationResult(ModerationResult):

    contains_pii: bool = Field(
        description="Whether the video contains any person or personally-identifiable information (PII)"
    )
    is_disturbing: bool = Field(description="Whether the video is disturbing")
    is_low_quality: bool = Field(description="Whether the video is low quality")
    contains_hate_speech: bool = Field(
        default=False,
        description="Whether the video communicates hate through speech, text, symbols, or depictions",
    )
    is_spam: bool = Field(
        default=False,
        description="Whether the video contains unsolicited promotional, scam, or spam messaging",
    )
    contains_misinformation: bool = Field(
        default=False,
        description="Whether the video communicates clearly false or misleading factual claims",
    )


class AudioModerationResult(ModerationResult):

    transcription: str = Field(description="Transcription of the audio content")
    contains_pii: bool = Field(description="Whether the audio contains any personally-identifiable information (PII)")
    is_unfriendly: bool = Field(description="Whether unfriendly tone or content was detected")
    is_unprofessional: bool = Field(description="Whether unprofessional tone or content was detected")
    contains_hate_speech: bool = Field(
        default=False,
        description="Whether the audio attacks or demeans people based on protected characteristics",
    )
    is_spam: bool = Field(
        default=False,
        description="Whether the audio contains unsolicited promotional, scam, or spam messaging",
    )
    contains_misinformation: bool = Field(
        default=False,
        description="Whether the audio presents clearly false or misleading factual claims as true",
    )
