from openai import OpenAI
import os
import config

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(chunk: str, model=config.EMBEDDING_MODEL_NAME) -> list[float] | None:
    try:
        response = client.embeddings.create(
            model=model,
            input=chunk
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Failed to call embedding model {config.EMBEDDING_MODEL_NAME}. Please try later.")
        print("Error: ", e)
        return None
