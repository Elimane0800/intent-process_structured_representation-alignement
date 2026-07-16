EVIDENCE_PROMPT = """You previously extracted the following state space from a text.

Text:
{text}

Extracted state space:
{state_space}

For each entity in the extraction, quote the exact sentence(s) from the text above that justify including it as an entity with those states. Copy the sentence(s) verbatim, character for character, from the text. Do not paraphrase or summarize.

If multiple sentences are needed for one entity, separate them with " | ".

After listing all quotes, output the line "FINAL_ANSWER:" followed immediately by strict JSON only: {{"entity_name": "verbatim quote(s)", ...}}
Nothing must follow the JSON. No prose, no explanation, no markdown fences.
"""