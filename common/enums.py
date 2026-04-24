from enum import Enum


class FailureType(Enum):
    TOP1_RETRIEVAL_INCORRECT = "top1_retrieval_incorrect"
    TOPK_RECALL_FAILED = "topk_reccall_failed"
    SHOULD_ABSTAIN_BUT_ANSWERED = "should_abstain_but_answered"
    SHOULD_ANSWERED_BUT_ABSTAIN = "should_answered_but_abstain"
    ANSWER_INCORRECT = "answer_incorrect"
    TYPO_QUERY = "typo_query"
    GATING_FALSE_POSITIVE = "gating_false_positive"
    GATING_FALSE_NEGATIVE = "gating_false_negative"