# Protocol A, Stage 1 prompt: the LLM receives, for one target state, the full list of
# mechanically generated candidate preconditions (Stage 0, pure Python, no LLM), and must
# classify EACH candidate independently as valid or invalid -- not select a single "best" one.
# Multiple candidates may be classified valid for the same target; this is observed as data,
# not prevented architecturally.

_CLASSIFICATION_DEFINITION = """Definitions:
- A precondition candidate is a conjunction of entity.state pairs (joined by " AND ") that was mechanically generated as a plausible precondition for a target state.
- A candidate is VALID if the text genuinely and specifically supports it as a precondition of the target state -- not just plausible in general, but stated or clearly and unambiguously implied by a specific part of the text.
- A candidate is INVALID if the text does not support it, even if it seems logically reasonable in the abstract. Plausibility alone is not sufficient; the text must actually say so."""

_CLASSIFICATION_RULES = """Rules:
- Classify every single candidate in the list independently. Do not skip any, and do not select only one -- more than one candidate may be VALID for the same target state if the text supports several.
- Never invent a new candidate, never modify a candidate's wording. Only judge the candidates exactly as given.
- Every VALID classification must include a verbatim quote from the text, copied exactly, character for character, that justifies it.
- If you cannot find a real quote justifying a candidate, classify it INVALID, even if it seems reasonable.
- Output strict JSON only: {{"<candidate string>": {{"valid": true, "quote": "<verbatim quote>"}} | {{"valid": false, "quote": null}}, ...}} for every candidate in the list.
- No prose, no explanation, no markdown fences."""

CLASSIFICATION_PROMPT_A = """You are classifying candidate preconditions for a target state, based on a source text.

{definition}

{rules}

Text:
{{text}}

Target state: {{target_state}}

Candidates:
{{candidates}}
""".format(definition=_CLASSIFICATION_DEFINITION, rules=_CLASSIFICATION_RULES)