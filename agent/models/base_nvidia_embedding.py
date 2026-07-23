import os
import time
from openai import OpenAI

from agent.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, EMBEDDING_INPUT_TYPE


def get_embedding_client() -> OpenAI:
    api_key = os.environ["NVIDIA_API_KEY"]
    return OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")


def _embed_batch(client, texts: list[str], model: str, input_type: str, retries: int = 2) -> list[list[float]]:
    for attempt in range(retries + 1):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "NONE"},
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            if attempt == retries:
                print(f"    [embedding] failed on batch of {len(texts)} after {retries + 1} attempt(s): {e}")
                print(f"    [embedding] texts in this batch: {texts}")
                raise
            time.sleep(1.5 * (attempt + 1))


def embed(
    texts: list[str],
    model: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    input_type: str = EMBEDDING_INPUT_TYPE,
) -> list[list[float]]:
    client = get_embedding_client()
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        results.extend(_embed_batch(client, chunk, model, input_type))
    return results


if __name__ == "__main__":
    vecs = embed(["What is the capital of France?"])
    print(len(vecs[0]), vecs[0][:5])