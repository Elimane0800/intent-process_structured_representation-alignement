from agent.prompt.reformulation_prompt import REFORMULATION_PROMPT_V2


def reformulate_text(text: str, llm, prompt_template: str = REFORMULATION_PROMPT_V2) -> str:
    """Reformulate a process description using one LLM call."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    prompt = prompt_template.format(text=text.strip())
    result = llm.invoke(prompt).content.strip()

    if not result:
        raise ValueError("reformulation returned an empty response")

    return result
