import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL_NAME = "openai/gpt-oss-120b"

def get_llm(temperature: float = 0.2, model: str = MODEL_NAME, max_tokens: int = 4096) -> ChatOpenAI:
    api_key = os.environ["NVIDIA_API_KEY"]
    return ChatOpenAI(
                    model=model, 
                    openai_api_key=api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                    temperature=temperature,
                    max_tokens=max_tokens
                    )


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Réponds uniquement par le mot: OK")
    print(response.content)