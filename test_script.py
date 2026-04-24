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


def get_failure_type(top_chunks_contain_answer, answerable_from_kb, actual_use_rag, expected_use_rag, reply):
    # routing/gating
    if actual_use_rag and not expected_use_rag:
        return enums.FailureType.GATING_FALSE_POSITIVE
    if not actual_use_rag and expected_use_rag:
        return enums.FailureType.GATING_FALSE_NEGATIVE
    
    # topk retrieval recall issue
    if answerable_from_kb and actual_use_rag and not top_chunks_contain_answer:
        return enums.FailureType.TOPK_RECALL_FAILED
    
    # abstain issue
    if actual_use_rag and not answerable_from_kb and reply and "I don't know" not in reply.lower():
        return enums.FailureType.SHOULD_ABSTAIN_BUT_ANSWERED
    if actual_use_rag and answerable_from_kb and reply and "I don't know" in reply.lower():
        return enums.FailureType.SHOULD_ANSWERED_BUT_ABSTAIN
    
    # generation problem
    
    return None

def main():
    with open(EVAL_DATA_FILE) as f:
        for line in f:
            # load query
            data = json.loads(line)
            query = data["query"]
            print(query)

            # call generate_reply
            reply, rag_result = generate_reply(query, knowledge, messages)

            # compare retrieval result with gold answer
            # top1_correct
            rag_result_top1_chunk = rag_result["top_chunks"][0] if rag_result["top_chunks"] else None
            gold_top1_chunk = data["gold_chunks"][0] if data["gold_chunks"] else None
            top1_correct = rag_result_top1_chunk == gold_top1_chunk

            # top_chunks_contain_answer
            rag_result_top_chunks = rag_result["top_chunks"]
            top_chunks_contain_answer = gold_top1_chunk in rag_result_top_chunks

            # answerable_from_kb
            answerable_from_kb = data["answerable_from_kb"]

            # answer_correct
            rag_result_answer = reply
            gold_answer = data["gold_answer"]
            # TODO: LLM_as_judge

            # actual_use_rag
            actual_use_rag = rag_result["use_rag"]

            # expected_use_rag
            expected_use_rag = data["use_rag"]

            # 

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
            failure_type = get_failure_type(top_chunks_contain_answer, answerable_from_kb, actual_use_rag, expected_use_rag, reply)
            print(failure_type)

            # TODO: output failure_type into trace_log

            # calculate metrics

            # output trace
            set_trace(query, rag_result, reply)


if __name__ == "__main__":
    main()

