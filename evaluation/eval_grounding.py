# TODO:
# will add evaluation between with answer grounding and without answer grounding
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
import config
import rag
from main import generate_reply, system_message, set_trace
from common import enums
from collections import defaultdict

EVAL_DIR = os.path.join(PROJECT_ROOT, config.EVAL_DIR) # evaluation/
EVAL_DATA_FILE = os.path.join(EVAL_DIR, config.EVAL_DATA_FILE) # evaluation/eval_data.jsonl

EVAL_REPORT_FILE = os.path.join(EVAL_DIR, "eval_grounding_report.jsonl")
OUTPUT_SUMMARY_FILE = os.path.join(EVAL_DIR, "eval_grounding_summary.jsonl")

# load knowledge base
knowledge_path = os.path.join(PROJECT_ROOT, "knowledge.txt")
knowledge = rag.load_knowledge(knowledge_path)

# pre-embedd knowledge
embedded_knowledge = rag.embed_knowledge(knowledge)

messages = [system_message]
retrieve_mode = config.RETRIEVE_MODE_EMBEDDING

def main():
    # clean previous evaluation round
    open(EVAL_REPORT_FILE, "w", encoding="utf-8").close()
    open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8").close()

    # grounding eval
    with open(EVAL_DATA_FILE) as f: # per-case evaluation
        for line in f:
            # load query
            data = json.loads(line)
            query = data["query"]
            print(query)

            reply_grounding, rag_result_grounding, grounding_response = generate_reply(
                query, 
                knowledge, 
                embedded_knowledge, 
                messages.copy(), 
                retrieve_mode,
                config.GENERATION_MODE_GROUNDING
            )
            
            reply_no_grounding, rag_result_no_grounding, _ = generate_reply(
                query, 
                knowledge, 
                embedded_knowledge, 
                messages.copy(), 
                retrieve_mode,
                config.GENERATION_MODE_RELEVANCE
            )

            # output evaluation report
            set_eval_report(
                query,
                data,
                rag_result_grounding,
                reply_grounding,
                grounding_response,
                rag_result_no_grounding,
                reply_no_grounding,
                retrieve_mode
            )

    # output summary
    set_summary()

def set_summary():
    total_cases = 0
    without_unsupported = 0
    without_false_abstain = 0
    with_unsupported = 0
    with_false_abstain = 0
    grounding_correct = 0

    with open(EVAL_REPORT_FILE) as f:
        for line in f:
            data = json.loads(line)

            total_cases += 1
            
            if data["with_grounding"]["should_abstain_but_answered"]:
                with_unsupported += 1
            if data["with_grounding"]["should_answer_but_abstained"]:
                with_false_abstain += 1
            if data["without_grounding"]["should_abstain_but_answered"]:
                without_unsupported += 1
            if data["without_grounding"]["should_answer_but_abstained"]:
                without_false_abstain += 1
            if data["comparison"]["grounding_correct"]:
                grounding_correct += 1

    summary = {
        "total_cases": total_cases,
        "without_grounding": {
            "should_abstain_but_answered_rate": without_unsupported / total_cases,
            "false_abstain_rate": without_false_abstain / total_cases,
        },
        "with_grounding": {
            "should_abstain_but_answered_rate": with_unsupported / total_cases,
            "false_abstain_rate": with_false_abstain / total_cases,
        },
        "comparison": {
            "should_abstain_but_answered_delta": with_unsupported - without_unsupported,
            "false_abstain_delta": with_false_abstain - without_false_abstain,
            "grounding_correct_rate": grounding_correct / total_cases
        }
    }

    with open(OUTPUT_SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))


def set_eval_report(query, data, rag_result_grounding, reply_grounding, grounding_response, rag_result_no_grounding, reply_no_grounding, retrieve_mode):
    LOG_FILE = EVAL_REPORT_FILE
    os.makedirs(EVAL_DIR, exist_ok=True)

    # gold metrics
    gold_answer = data["gold_answer"]
    expect_use_rag = data["use_rag"] # kb is relevant
    answerable_from_kb = data["answerable_from_kb"]
    should_abstain = expect_use_rag and not answerable_from_kb 


    answerable_grounding = grounding_response.answerable if grounding_response else False

    abstained_grounding = reply_grounding and "i don't know" in reply_grounding.lower()
    abstained_no_grounding = reply_no_grounding and "i don't know" in reply_no_grounding.lower()

    supported_answer = grounding_response.supported_answer if grounding_response else None
    kb_relevant_no_grounding = rag_result_no_grounding["use_rag"] if rag_result_no_grounding else False
    kb_relevant_grounding = rag_result_grounding["use_rag"] if rag_result_grounding else False

    should_abstain_but_answered_grounding = should_abstain and not abstained_grounding
    should_abstain_but_answered_no_grounding = should_abstain and not abstained_no_grounding

    should_answer_but_abstained_grounding = expect_use_rag and answerable_from_kb and abstained_grounding
    should_answer_but_abstained_no_grounding = expect_use_rag and answerable_from_kb and abstained_no_grounding

    eval_report = {
        "query": query,
        "retrieve_mode": retrieve_mode,
        "expected": {
            "relevant": expect_use_rag,
            "answerable": answerable_from_kb,
            "gold_answer": gold_answer,
        },
        "with_grounding": {
            #"relevant": kb_relevant_grounding,
            "answerable": answerable_grounding,
            "reply": reply_grounding,
            "abstained": abstained_grounding,
            "should_abstain_but_answered": should_abstain_but_answered_grounding,
            "should_answer_but_abstained": should_answer_but_abstained_grounding,
        },
        "without_grounding": {
            #"relevant": kb_relevant_no_grounding,
            "reply": reply_no_grounding,
            "abstained": abstained_no_grounding,
            "should_abstain_but_answered": should_abstain_but_answered_no_grounding,
            "should_answer_but_abstained": should_answer_but_abstained_no_grounding,
        },
        "comparison": {
            "grounding_correct": answerable_grounding == answerable_from_kb,
            "grounding_reduced_unsupported_answer": should_abstain_but_answered_no_grounding and not should_abstain_but_answered_grounding,
            "grounding_caused_false_abstain": not should_answer_but_abstained_no_grounding and should_answer_but_abstained_grounding,
            "grounding_reduced_false_abstain": should_answer_but_abstained_no_grounding and not should_answer_but_abstained_grounding
        }
    }

    # output metrics
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(eval_report, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    main()
