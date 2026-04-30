import json
import config
import os
import rag
from main import generate_reply, system_message, set_trace
from common import enums

EVAL_DIR = config.EVAL_DIR
EVAL_DATA_FILE = os.path.join(EVAL_DIR, config.EVAL_DATA_FILE)
EVAL_REPORT_FILE = os.path.join(EVAL_DIR, config.EVAL_REPORT_FILE)
OUTPUT_SUMMARY_FILE = os.path.join(EVAL_DIR, config.EVAL_SUMMARY_FILE)

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

            # call generate_reply for keyword retrieve mode
            reply, rag_result = generate_reply(query, knowledge, messages, config.RETRIEVE_MODE_KEYWORD)

            # TODO: call generate_reply for embedding retrieve mode
            # reply, rag_result = generate_reply(query, knowledge, messages, config.RETRIEVE_MODE_EMBEDDING)

            print(reply)
            print()
            
            # output trace (runtime logging)
            set_trace(query, rag_result, reply)

            # output evaluation report
            set_eval_report(query, data, rag_result, reply)

    
    # aggregate metrics
    aggregate_metrics()



def load_eval_report(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def safe_rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None

def aggregate_metrics():
    total_cases = 0

    top1_valid_count = 0
    top1_correct_count = 0

    topk_valid_count = 0
    topk_recall_count = 0

    gating_correct_count = 0

    should_abstain_count = 0
    correct_abstain_count = 0

    should_answer_count = 0
    false_abstain_count = 0

    hallucination_count = 0
    failure_counts = {}

    for record in load_eval_report(EVAL_REPORT_FILE):
        total_cases += 1

        expected = record["expected"]
        actual = record["actual"]
        result = record["result"]

        # gating
        if result["gating_correct"] is True:
            gating_correct_count += 1

        # top1 accuracy: only count applicable cases
        if result["top1_correct"] is not None:
            top1_valid_count += 1
            if result["top1_correct"] is True:
                top1_correct_count += 1

        # topk recall: only count applicable cases
        if result["topk_recall"] is not None:
            topk_valid_count += 1
            if result["topk_recall"] is True:
                topk_recall_count += 1

        # abstain metrics
        if expected["should_abstain"] is True:
            should_abstain_count += 1
            if actual["abstained"] is True:
                correct_abstain_count += 1

        if expected["should_answer_from_kb"] is True:
            should_answer_count += 1
            if result["should_answer_but_abstained"] is True:
                false_abstain_count += 1

        # hallucination / unsupported answer
        if result["should_abstain_but_answered"] is True:
            hallucination_count += 1

        # failure counts
        for ft in record["failure_type"]:
            if ft == enums.FailureType.NONE.value:
                continue
            failure_counts[ft] = failure_counts.get(ft, 0) + 1

    summary = {
        "total_cases": total_cases,

        "counts": {
            "gating_correct": gating_correct_count,
            "top1_correct": top1_correct_count,
            "top1_valid": top1_valid_count,
            "topk_recall": topk_recall_count,
            "topk_valid": topk_valid_count,
            "should_abstain": should_abstain_count,
            "correct_abstain": correct_abstain_count,
            "should_answer": should_answer_count,
            "false_abstain": false_abstain_count,
            "hallucination": hallucination_count,
        },

        "rates": {
            "gating_accuracy": safe_rate(gating_correct_count, total_cases),
            "top1_accuracy": safe_rate(top1_correct_count, top1_valid_count),
            "topk_recall_rate": safe_rate(topk_recall_count, topk_valid_count),
            "correct_abstain_rate": safe_rate(correct_abstain_count, should_abstain_count),
            "false_abstain_rate": safe_rate(false_abstain_count, should_answer_count),
            "hallucination_rate": safe_rate(hallucination_count, should_abstain_count),
        },

        "failure_counts": failure_counts,
    }

    with open(OUTPUT_SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        

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

    failures = []

    # gating
    if not expect_use_rag and actual_use_rag:
        failures.append(enums.FailureType.GATING_FALSE_POSITIVE.value)
    if expect_use_rag and not actual_use_rag:
        failures.append(enums.FailureType.GATING_FALSE_NEGATIVE.value)
    
    # topk recall
    if expect_use_rag and answerable_from_kb and not topk_recall:
        failures.append(enums.FailureType.TOPK_RECALL_FAILED.value)
    
    if should_abstain_but_answered:
        failures.append(enums.FailureType.SHOULD_ABSTAIN_BUT_ANSWERED.value)
    
    if should_answer_but_abstained:
        failures.append(enums.FailureType.SHOULD_ANSWER_BUT_ABSTAINED.value)
    
    if top1_correct is not None and not top1_correct:
        failures.append(enums.FailureType.TOP1_RANKING_FAILED.value)
    
    return failures or [enums.FailureType.NONE.value]


if __name__ == "__main__":
    main()

