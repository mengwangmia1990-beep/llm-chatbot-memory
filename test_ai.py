from openai import OpenAI #从openai这个包里只导入OpenAI这个类
import os # 直接导入整个模块

MODEL_NAME="gpt-4o-mini"

#====================================================================================================================
# LLM API Call
def call_llm(messages, model):
    try:
        # raise Exception("mock API failure")
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print("AI: 模型调用失败, 请稍后重试")
        print("Error", e)
        return None

#====================================================================================================================
#RULE-BASED POLICY
unsafe_keywords = {
    "self-harm": {
        "keyword": ["自杀", "想死", "自我了断", "轻生"],
        "action": "support"
    },
    "violence": {
        "keyword": ["杀", "杀人", "伤害"],
        "action": "block"
    },
    "sexual": {
        "keyword": ["强奸", "色情"],
        "action": "block"
    }
}

def is_unsafe(user_input):
    text = user_input.lower()
    for category, value in unsafe_keywords.items():
        keywords = value["keyword"]
        action = value["action"]

        for keyword in keywords:
            if keyword in text:
                return True, category, action
    return False, None, None

#====================================================================================================================
# SUMMARIZE MESSAGE
def summarize(old_summary, new_messages):
    if old_summary is not None:
        summary_prompt = [
            {
                "role": "system",
                "content": "你是一个对话摘要助手。请根据旧摘要和新增对话，生成更新后的中文摘要。保留稳定的重要信息，用户需求和上下文。合并新增信息，避免重复，不要捏造。"
            },
            {
                "role": "user",
                "content": f"""
                这是旧摘要:
                {old_summary}

                这是新增对话:
                {new_messages}

                请输出更新后的摘要。
                """
            }
        ]
    else:
        summary_prompt = [
            {
                "role": "system",
                "content": "请以简洁中文总结以下对话历史。保留用户关键信息，用户需求和上下文。不可捏造信息。"
            },
            {
                "role": "user",
                "content": str(new_messages)
            }
        ]
    return call_llm(summary_prompt, MODEL_NAME)

#====================================================================================================================
def has_summary(messages):
    if messages is not None and len(messages) > 1 and messages[1].get("type") == "summary":
        return True
    return False

#====================================================================================================================
# System setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_TURN = 5
MAX_MESSAGES = MAX_TURN * 2
SHORT_TERM_TURN = 2
SHORT_TERM_MESSAGES = SHORT_TERM_TURN * 2

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

messages = [
    system_message
]

#====================================================================================================================
# Chatbot body
print("Put your questions below:") # title
while True:
    # construct message
    user_input = input("your question: ").strip() # collect user input from terminal
    if not user_input:
        print("Please enter a question:")
        continue
    
    if user_input.lower() in {"quit", "exit"}:
        print("Bye!")
        break

    is_flagged, category, action = is_unsafe(user_input)
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

    reply = call_llm(messages, MODEL_NAME)
    if reply is None:
        messages.pop() # rollback没成功的用户问题
        continue

    # attach assistant history
    messages.append(
        {"role": "assistant", "content": reply}
    )

    
    # Context window management
    if len(messages) > MAX_MESSAGES + 1:
        try:
            if has_summary(messages):
                # system
                # old summary
                # chats need to be summarized
                # recent chats need to be saved

                old_summary = messages[1].get("content")
                chats_to_be_summarized = messages[2:-SHORT_TERM_MESSAGES]
                summary = summarize(old_summary, chats_to_be_summarized)
            else:
                # system
                # chats need to be summarized
                # chats need to be saved

                summary = summarize(None, messages[1:-SHORT_TERM_MESSAGES])
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
            
            short_term_message = messages[-SHORT_TERM_MESSAGES:]

            messages = [
                system_message,
                summary_message
            ]

            messages.extend(short_term_message)
        else:
            print("AI: 历史摘要生成失败，降级为只保留最近几轮对话继续运行")
            # 这里，如果只打印不做处理，message会持续增长，最终导致latency增加，甚至超过context limit.
            # 需要fallback plan degredation 到维持最近几轮对话
            if has_summary(messages):
                messages = [system_message, messages[1]] + messages[-SHORT_TERM_MESSAGES:]
            else:
                messages = [system_message] + messages[-SHORT_TERM_MESSAGES:]


    print("AI: ", reply)
    print(messages)
    print()
    print("==========================================================================")
