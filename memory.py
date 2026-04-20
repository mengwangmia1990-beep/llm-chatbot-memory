import llm

# Generate summary
def summarize(old_summary, new_messages):
    if old_summary is not None:
        summary_prompt = [
            {
                "role": "system",
                "content": """
                You are a conversation summarization assistant.
                
                Task:
                Given a previous summary and new conversation messages, generate an updated summary.

                Requirements:
                - Preserve important long-term information (e.g., user preferences, needs, background)
                - Incorporate key information from the new conversation
                - Avoid redundancy by merging overlapping information
                - Do not invent or hallucinate any information
                - Keep the summary clear and concise

                Output:
                Return only the final updated summary.
                """
            },
            {
                "role": "user",
                "content": f"""
                Old summary:
                {old_summary}

                New conversations:
                {new_messages}

                Please output the final updated summary.
                """
            }
        ]
    else:
        summary_prompt = [
            {
                "role": "system",
                "content": """
                You are a conversation summarization assistant.

                Task:
                Summarize the following conversation.

                Requirements:
                - Keep important user information, needs, and context
                - Do not invent or hallucinate any information
                - Keep the summary concise and clear

                Output:
                Return only the summary.
                """
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