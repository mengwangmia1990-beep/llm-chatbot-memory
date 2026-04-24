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

    # load knowledge
    knowledge = rag.load_knowledge()

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

        reply, rag_result = generate_reply(user_input, knowledge, messages)

        if reply is None:
            # rollback没成功的用户问题
            set_trace(user_input, rag_result, reply, "llm_failed")
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
        set_trace(user_input, rag_result, reply)

        # TODO
        # top1_correct, topk_contains_answer, answerable_from_kb, answer_correct, failure_type

        print("AI: ", reply)
        print("==========================================================================")

def set_trace(user_input, result, reply, error=None):
    LOG_DIR = config.LOG_DIR
    LOG_FILE = os.path.join(LOG_DIR, config.TRACE_FILE_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)

    trace = {
        "query": user_input,
        "retrieval": {
            "use_rag": result["use_rag"],
            "top_chunks": result["top_chunks"],
            "top_chunk": result["top_chunks"][0] if result["top_chunks"] else None,
            "top_scores": result["top_scores"],
            "top_score": result["top_score"],
            "threshold": config.RAG_RELEVANCE_THRESHOLD
        },
        "response": {
            "reply": reply,
            "mode": "rag" if result["use_rag"] else "llm"
        },
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

def generate_reply(user_input, knowledge, messages):
    result = rag.retrieve(user_input, knowledge)
    
    if result["use_rag"]: # routing
        rag_prompt = build_rag_messages(result, user_input)
        response = llm.call_llm(messages + rag_prompt)
    else:
        # Call LLM model and collect response
        user_messages = [
            {"role": "user", "content": user_input}
        ]
        response = llm.call_llm(messages + user_messages)

    return response, result

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