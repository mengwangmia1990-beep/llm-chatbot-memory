import json
import config
import os
import rag
from main import generate_reply, system_message, set_trace
from common import enums

EVAL_DIR = config.EVAL_DIR
EVAL_DATA_FILE = os.path.join(EVAL_DIR, config.EVAL_DATA_FILE)

# load knowledge base
knowledge = rag.load_knowledge()
messages = [system_message]

def main():
    with open(EVAL_DATA_FILE) as f:
        for line in f:
            # load query
            data = json.loads(line)
            query = data["query"]
            print(query)

            # call generate_reply
            reply, rag_result = generate_reply(query, knowledge, messages)

            print(reply)
            print()
            
            # failure_type
            # failure_type = get_failure_type()
            # ft_1: top1_retrieval_incorrect (top1 retrieval precision)
            # ft_2: topk_retrieval_missing (topk retrieval failed to recall)
            # ft_3: should_abstain_but_answered
            # ft_4: should_answered_but_abstain
            # ft_5: answer_incorrect
            # ft_6: typo_query
            # ft_7: gating_false_positive (system use_rag but are not expected)
            # ft_8: gating_false_negative (system does not use rag but are expected to use rag)
            # failure_type = get_failure_type(top_chunks_contain_answer, answerable_from_kb, actual_use_rag, expected_use_rag, reply)
            # print(failure_type)

            # calculate metrics

            # output trace (runtime logging)
            set_trace(query, rag_result, reply)

            # output evaluation report
            set_eval_report(query, data, rag_result, reply)


def set_eval_report(query, data, rag_result, reply):
    EVAL_DIR = config.EVAL_DIR
    LOG_FILE = os.path.join(EVAL_DIR, config.EVAL_REPORT_FILE)
    os.makedirs(EVAL_DIR, exist_ok=True)

    answerable_from_kb = data["answerable_from_kb"]

    expect_use_rag = data["use_rag"]
    actual_use_rag = rag_result["use_rag"]
    
    abstained = reply and "i don't know" in reply.lower()

    gold_chunks = data["gold_chunks"]
    gold_chunk = gold_chunks[0] if gold_chunks else None
    actual_top_chunks = rag_result["top_chunks"]
    actual_top1_chunk = actual_top_chunks[0] if actual_top_chunks else None

    # default: not applicable
    top1_correct = None
    topk_recall = None

    if gold_chunks:
        if gold_chunk:
            top1_correct = actual_top1_chunk == gold_chunk
            topk_recall = gold_chunk in actual_top_chunks

    should_answer_from_kb = expect_use_rag and answerable_from_kb
    should_abstain = expect_use_rag and not answerable_from_kb

    gating_correct = expect_use_rag == actual_use_rag
    should_abstain_but_answered = should_abstain and not abstained
    should_answer_but_abstained = should_answer_from_kb and abstained

    eval_report = {
        "query": query,

        "expected": {
            "answerable_from_kb": answerable_from_kb,
            "expected_use_rag": expect_use_rag,
            "should_answer_from_kb": should_answer_from_kb,
            "should_abstain": should_abstain,
        },

        "actual": {
            "actual_use_rag": actual_use_rag,
            "abstained": abstained,
        },

        "result": {
            "gating_correct": gating_correct,
            "top1_correct": top1_correct,
            "topk_recall": topk_recall,
            "should_abstain_but_answered": should_abstain_but_answered,
            "should_answer_but_abstained": should_answer_but_abstained,
        },

        "generation": {
            "answer_correct": None, # needs human judge in current iteration
        },

        "failure_type": get_failure_type(expect_use_rag, actual_use_rag, answerable_from_kb, top1_correct, topk_recall,
                                         should_abstain_but_answered, should_answer_but_abstained)
    }
    
    # output metrics
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(eval_report, ensure_ascii=False) + "\n")

def get_failure_type(
    expect_use_rag,
    actual_use_rag,
    answerable_from_kb,
    top1_correct,
    topk_recall,
    should_abstain_but_answered,
    should_answer_but_abstained,
    answer_correct=None,):

    # gating
    if not expect_use_rag and actual_use_rag:
        return enums.FailureType.GATING_FALSE_POSITIVE.value
    if expect_use_rag and not actual_use_rag:
        return enums.FailureType.GATING_FALSE_NEGATIVE.value
    
    # topk recall
    if expect_use_rag and answerable_from_kb and not topk_recall:
        return enums.FailureType.TOPK_RECALL_FAILED.value
    
    if should_abstain_but_answered:
        return enums.FailureType.SHOULD_ABSTAIN_BUT_ANSWERED.value
    
    if should_answer_but_abstained:
        return enums.FailureType.SHOULD_ANSWER_BUT_ABSTAINED.value
    
    if top1_correct is not None and not top1_correct:
        return enums.FailureType.TOP1_RANKING_FAILED.value
    
    return enums.FailureType.NONE.value


if __name__ == "__main__":
    main()

