from pydantic import BaseModel, Field
from typing import Literal

class GroundingResult(BaseModel):
    answerable: bool = Field(
        description="Whether the context supports a faithful and complete answer"
    )
    answer_type: Literal["direct", "negative", "partial", "not_answerable"] = Field(
        description="Type of answerability decision"
    )
    supported_answer: str = Field(
        description="The answer supported by the context. Empty if not answerable"
    )
    reason: str = Field(
        description="Brief explanation for the decision"
    )
    evidence: str = Field(
        description="Exact quote from context if answerable, otherwise empty string"
    )