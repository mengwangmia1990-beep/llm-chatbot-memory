import llm
import safety
import memory
import config

system_message = {
    "role": "system", 
    "content":"""
        你是一个生活的chatbot. 
        只回答:
        - 是
        - 不是
        
        严禁输出任何其他内容，包括解释，换行，空格和标点符号。
        """
    }

def main():
    messages = [
        system_message
    ]

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
                print(f"AI: 我理解你，你需要我什么支持？")
                continue
            if action == "block":
                print(f"AI: 这个问题涉及{category}, 我不能回答")
                continue

        messages.append(
            {"role": "user", "content": user_input}
        )

        reply = llm.call_llm(messages)
        if reply is None:
            messages.pop() # rollback没成功的用户问题
            continue

        # attach assistant history
        messages.append(
            {"role": "assistant", "content": reply}
        )
        
        # Context window management
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
                print("AI: 历史摘要生成失败")
                print("Error", e)
                summary = None

            # Construct message with summary
            if summary is not None:
                summary_message = {
                    "role": "system",
                    "type": "summary",
                    "content": f"以下是历史对话摘要: {summary}"
                    }
                
                short_term_message = messages[-config.SHORT_TERM_MESSAGES:]

                messages = [
                    system_message,
                    summary_message
                ]

                messages.extend(short_term_message)
            else:
                print("AI: 历史摘要生成失败，降级为只保留最近几轮对话继续运行")
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