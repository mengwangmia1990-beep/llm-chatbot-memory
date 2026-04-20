import llm
import safety
import memory
import config
import rag

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

    knowledge = rag.load_knowledge()

    print("Put your questions below:") # chatbot title
    while True:
        user_input = input("your question: ").strip() # collect user input from terminal
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

        messages.append(
            {"role": "user", "content": user_input}
        )

        # Before calling LLM, search knowledge first
        top_chunks = rag.retrieve(user_input, knowledge)

        # if found relevant knowledge
        if top_chunks:
            rag_message = [{
                "role": "user",
                "content": f"""
                    Use the following knowledge to answer the question.

                    Rules:
                    - Use the provided knowledge as the primary source
                    - If the answer is not in the knowledge, say "I don't know"
                    - Do not invent or hallucinate any information

                    Knowledge: 
                    {chr(10).join(top_chunks)}

                    Question:
                    {user_input}
                """
            }]
            # 临时加一轮, 不永久写入message对话
            reply = llm.call_llm(messages + rag_message)
        else:
            # Call LLM model and collect response
            reply = llm.call_llm(messages)

        if reply is None:
            messages.pop() # rollback没成功的用户问题
            continue

        # attach assistant history
        messages.append(
            {"role": "assistant", "content": reply}
        )
        
        # Context window management
        print(len(messages))
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

        print("AI: ", reply)
        print(messages)
        print("==========================================================================")

if __name__ == "__main__":
    main()