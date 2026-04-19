import llm

# Generate summary
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
    return llm.call_llm(summary_prompt)

# Check if messages have summary
def has_summary(messages):
    return (
        messages is not None
        and len(messages) > 1
        and isinstance(messages[1], dict)
        and messages[1].get("type") == "summary"
    )