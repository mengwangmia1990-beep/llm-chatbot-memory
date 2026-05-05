from openai import OpenAI
import os
import config
from models.grounding import GroundingResult

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
    
def call_llm_structured_output(messages, model=config.MODEL_NAME):
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=GroundingResult
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print("AI: Failed to call LLM with structured output. Please retry later.")
        print("Error", e)
        return None