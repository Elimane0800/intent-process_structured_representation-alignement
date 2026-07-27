import os
import time
from openai import OpenAI

from agent.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, EMBEDDING_INPUT_TYPE

# Meme gap que base_llm.py avant patch (cf. commit correspondant) : get_embedding_client()
# ne passait aucun timeout explicite -- le SDK openai retombe alors sur son propre defaut
# (couramment plusieurs minutes), ce qui rend "Request timed out." apres 3 tentatives
# potentiellement tres long avant meme d'echouer, plutot qu'un echec rapide et diagnosticable.
# 30s choisi plus court que le timeout LLM (base_llm.py, 120s) car un appel d'embedding est
# structurellement plus rapide (pas de generation token par token) -- un embedding qui prend
# plus de 30s est deja anormal, pas juste lent sur un texte long.
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 30


def get_embedding_client(timeout: float | None = None) -> OpenAI:
    api_key = os.environ["NVIDIA_API_KEY"]
    return OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        timeout=timeout if timeout is not None else DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    )


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
    timeout: float | None = None,
) -> list[list[float]]:
    client = get_embedding_client(timeout=timeout)
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        results.extend(_embed_batch(client, chunk, model, input_type))
    return results


if __name__ == "__main__":
    vecs = embed(["What is the capital of France?"])
    print(len(vecs[0]), vecs[0][:5])