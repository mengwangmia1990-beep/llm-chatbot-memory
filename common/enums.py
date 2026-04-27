from enum import Enum


class FailureType(Enum):
    NONE = "none"

    GATING_FALSE_POSITIVE = "gating_false_positive"
    GATING_FALSE_NEGATIVE = "gating_false_negative"

    TOPK_RECALL_FAILED = "topk_recall_failed"
    TOP1_RANKING_FAILED = "top1_ranking_failed"

    SHOULD_ANSWER_BUT_ABSTAINED = "should_answer_but_abstained"
    SHOULD_ABSTAIN_BUT_ANSWERED = "should_abstain_but_answered" # hallucination
    GROUNDEDNESS_FAILURE = "groundedness_failure"