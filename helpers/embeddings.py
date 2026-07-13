from typing import List

from openai import OpenAI


def embed_text(client: OpenAI, model: str, text: str) -> List[float]:
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
