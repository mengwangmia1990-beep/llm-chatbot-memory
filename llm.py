from openai import OpenAI
import os
import config

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(messages, model=config.MODEL_NAME):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print("AI: Failed to call LLM. Please retry later.")
        print("Error", e)
        return None