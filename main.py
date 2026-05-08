import llm
import safety
import memory
import config
import rag
import json
import os

# global system prompt
system_message = {
    "role": "system",
    "content": """
        You are a helpful and reliable assistant.

        When answering:
        - Be clear, concise, and direct
        - If relevant knowledge is provided, use it as the primary source
        - Prefer knowledge-based answers over general knowledge when available
        - Do not invent or hallucinate any information

        Maintain a natural and conversational tone.
        """
    }

def main():
    messages = [
        system_message
    ]

    retrieve_mode = config.RETRIEVE_MODE_EMBEDDING
    generation_mode = config.GENERATION_MODE_GROUNDING

    # pre-load knowledge
    knowledge = rag.load_knowledge()

    # pre-embedd knowledge
    embedded_knowledge = rag.embed_knowledge(knowledge)

    print("Put your questions below:")
    while True:
         # collect user input from terminal
        user_input = input("your question: ").strip()
        
        if not user_input:
            print("Please enter a question:")
            continue
        
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        is_flagged, category, action = safety.is_unsafe(user_input)
        if is_flagged:
            if action == "support":
                print(f"AI: I understand you. What support do you need from me?")
                continue
            if action == "block":
                print(f"AI: This issue involves {category}, I cannot answer it.")
                continue

        reply, rag_result, grounding_response = generate_reply(user_input, knowledge, embedded_knowledge, messages, retrieve_mode, generation_mode)

        if reply is None:
            # rollback没成功的用户问题
            set_trace(user_input, rag_result, retrieve_mode, reply, "llm_failed")
            continue

        # attach user history
        messages.append(
            {"role": "user", "content": user_input}
        
        )
        # attach assistant history
        messages.append(
            {"role": "assistant", "content": reply}
        )
        
        # context window management
        messages = context_management(messages)

        # set trace and output to json file for observation
        set_trace(user_input, rag_result, retrieve_mode, reply, grounding_response)

        print("AI: ", reply)
        print("==========================================================================")


def set_trace(user_input, result, retrieve_mode, reply, grounding=None, gating=None, error=None):
    LOG_DIR = config.LOG_DIR
    LOG_FILE = os.path.join(LOG_DIR, config.TRACE_FILE_NAME)

    gating_threshold = config.RAG_RELEVANCE_THRESHOLD
    if retrieve_mode == config.RETRIEVE_MODE_EMBEDDING:
        if gating != None:
            gating_threshold = gating # embedding gating threshold tuning treatments
        else:
            gating_threshold = config.RAG_RELEVANCE_EMBEDDING_THRESHOLD

    trace = {
        "query": user_input,
        "retrieval": {
            "mode": retrieve_mode,
            "threshold": gating_threshold,
            "use_rag": result["use_rag"],

            "top-k": len(result["top_chunks"]) if result["top_chunks"] else 0,
            "top_chunks": result["top_chunks"],
            "top_scores": result["top_scores"],

            "top1_chunk": result["top1_chunk"],
            "top1_score": result["top1_score"],

            "top2_chunk": result["top2_chunk"],
            "top2_score": result["top2_score"],

            "top1_top2_gap": result["top1_score"] - result["top2_score"]
        },
        "response": {
            "reply": reply,
            "mode": "rag" if result["use_rag"] else "llm"
        },
        "grounding": grounding.model_dump() if grounding else None,
        "error": error
    }
    
    # output metrics
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def context_management(messages):
    if len(messages) > config.MAX_MESSAGES + 1:
        try:
            if memory.has_summary(messages):
                # system
                # old summary
                # chats need to be summarized
                # recent chats need to be saved

                old_summary = messages[1].get("content")
                chats_to_be_summarized = messages[2:-config.SHORT_TERM_MESSAGES]
                summary = memory.summarize(old_summary, chats_to_be_summarized)
            else:
                # system
                # chats need to be summarized
                # chats need to be saved

                summary = memory.summarize(None, messages[1:-config.SHORT_TERM_MESSAGES])
        except Exception as e:
            print("AI: Failed to generate summary")
            print("Error", e)
            summary = None

        # Construct message with summary
        if summary is not None:
            summary_message = {
                "role": "system",
                "type": "summary",
                "content": f"This is summarized history chats: {summary}"
                }
            
            short_term_message = messages[-config.SHORT_TERM_MESSAGES:]

            messages = [
                system_message,
                summary_message
            ]

            messages.extend(short_term_message)
        else:
            print("AI: History summary generation failed. Degrading to maintain only the most recent conversation rounds for continued operation.")
            # 这里，如果只打印不做处理，message会持续增长，最终导致latency增加，甚至超过context limit.
            # 需要fallback plan degredation 到维持最近几轮对话
            if memory.has_summary(messages):
                messages = [system_message, messages[1]] + messages[-config.SHORT_TERM_MESSAGES:]
            else:
                messages = [system_message] + messages[-config.SHORT_TERM_MESSAGES:]

    return messages


def generate_reply(user_input, knowledge, embedded_knowledge, messages, retrieve_mode, generation_mode, gating=0.0):
    if retrieve_mode == config.RETRIEVE_MODE_KEYWORD:
        result = rag.retrieve_keyword(user_input, knowledge)
    elif retrieve_mode == config.RETRIEVE_MODE_EMBEDDING:
        result = rag.retrieve_embedding(user_input, embedded_knowledge, config.TOPK, gating)
    
    grounding_response = None

    # Case 1: retrieval says no relevant context from knowledge base
    if not result["use_rag"]:
        user_messages = [
            {"role": "user", "content": user_input}
        ]
        response = llm.call_llm(messages + user_messages)
        return response, result, grounding_response
    
    # Case 2: relevance mode without grounding
    if generation_mode == config.GENERATION_MODE_RELEVANCE:
        rag_prompt = build_rag_messages(result, user_input)
        response = llm.call_llm(messages + rag_prompt)

    # Case 3: grounding mode
    if generation_mode == config.GENERATION_MODE_GROUNDING:
        grounding_prompt = build_grounding_prompt(result, user_input)
        grounding_response = llm.call_llm_structured_output(messages + grounding_prompt)

        if grounding_response is None:
            return "I don't know", result, grounding_response
    
        if not grounding_response.answerable:
            return "I don't know", result, grounding_response

        rag_prompt = build_rag_messages_grounding(result, user_input, grounding_response)
        response = llm.call_llm(messages + rag_prompt)

    return response, result, grounding_response


def build_grounding_prompt(result, user_input):
    context = "\n\n".join(result["top_chunks"])

    return [
        {
            "role": "system",
            "content": """
            You are a grounding checker for a RAG system.

            Your job is to determine whether the context supports a faithful answer to the user's question.

            You must first identify the answer that is explicitly supported by the context, then decide whether it answers the user's question.

            The answer is answerable ONLY if the context explicitly states:
            - the requested value
            OR
            - an explicit negation of the user's premise.

            Do NOT infer exclusivity, default business logic, or unstated policy implications.
            """
        },
        {
            "role": "user",
            "content": f"""
            Question:
            {user_input}

            Context:
            {context}

            Decide whether the context supports a faithful answer.

            Return:
            - answerable=true if the context supports a complete answer.
            - answer_type="direct" if the context provides the requested value directly.
            - answer_type="negative" if the context negates the premise of the question.
            - answer_type="partial" if the context is related but incomplete.
            - answer_type="not_answerable" if the context is unrelated or insufficient.
            - supported_answer should be the answer that can be given using only the context.
            - If answerable=false, supported_answer and evidence must be empty.
            - Evidence must be an exact quote from the context.

            Examples:
            Question: What time does the store open on Sunday?
            Context: Business Hours: Monday to Friday open at 9am and close at 8pm. Saturday open at 8am and close at 6pm. Sunday closed.
            Output:
            {{
            "answerable": true,
            "answer_type": "negative",
            "supported_answer": "The store is closed on Sunday, so it does not open that day.",
            "reason": "The context negates the premise that the store opens on Sunday.",
            "evidence": "Business Hours: Monday to Friday open at 9am and close at 8pm. Saturday open at 8am and close at 6pm. Sunday closed."
            }}

            Question: What time does the store open on Monday?
            Context: Business Hours: Monday to Friday open at 9am and close at 8pm. Saturday open at 8am and close at 6pm. Sunday closed.
            Output:
            {{
            "answerable": true,
            "answer_type": "direct",
            "supported_answer": "The store opens at 9am on Monday.",
            "reason": "The context provides Monday business hours.",
            "evidence": "Business Hours: Monday to Friday open at 9am and close at 8pm. Saturday open at 8am and close at 6pm. Sunday closed."
            }}

            Question: Do members get free shipping?
            Context: Membership Policy: Members get a 10% discount.
            Output:
            {{
            "answerable": false,
            "answer_type": "not_answerable",
            "supported_answer": "",
            "reason": "The context does not provide shipping options for members.",
            "evidence": ""
            }}

            Question: Do non-members get 10% discount?
            Context: Membership Policy: Members get a 10% discount.
            Output:
            {{
            "answerable": false,
            "answer_type": "not_answerable",
            "supported_answer": "",
            "reason": "The policy says members get a 10% discount, but it does not explicitly state whether non-members receive any discount.",
            "evidence": ""
            }}
            """
        }]

def build_rag_messages_grounding(result, user_input, grounding_response):
    context = "\n\n".join(result["top_chunks"])

    prompt = [{
        "role": "user",
        "content": f"""
        Use the provided knowledge to answer user's question.

        A grounding checker has already identified a supported answer.

        Supported answer:
        {grounding_response.supported_answer}

        Rules:
        - Use the provided knowledge as the primary source.
        - Use the supported answer as the basis of your response.
        - Do not contradict the supported answer.
        - If the supported_answer is empty or not supported by the knowledge, say "I don't know".
        - Do not hallucinate or invent any information.
        - Answer naturally and concisely.

        Knowledge:
        {context}

        Question:
        {user_input}
        """
    }]

    return prompt

def build_rag_messages(result, user_input):
    prompt = [{
        "role": "user",
        "content": f"""
            Use the following knowledge to answer the question.

            Rules:
            - Use the provided knowledge as the primary source
            - If the answer is not in the knowledge, say "I don't know"
            - Do not invent or hallucinate any information

            Knowledge: 
            {"\n\n".join(result["top_chunks"])}

            Question:
            {user_input}
        """
    }]
    return prompt

if __name__ == "__main__":
    main()