# Protocol B, Stage 0 prompt: classify which states of each entity are START states (need no
# precondition) and/or END states (terminal for their own entity) -- a bounded classification
# task, not open generation.

_START_END_DEFINITION = """Definitions:
- A START state is a state that needs no precondition: it is the natural first condition of an entity's lifecycle in this text, reached without any other state having to hold first.
- An END state is a state that is terminal for its own entity: no later state of the SAME entity follows it in this text. An entity can have at most one END state in most cases, but mark none if genuinely unclear.
- A state can be both START and END only if the entity has a single state in its whole lifecycle."""

_START_END_RULES = """Rules:
- Only select states that literally appear in the provided state space, for their own entity.
- Mark a state as START only if the text supports it being the natural entry point -- if unsure, do not mark it as START (it will simply be treated as needing a precondition, which is the safer default).
- Mark a state as END only if the text clearly does not describe any further state of the same entity following it.
- Every entity must appear in the output, even if its start/end lists are empty.
- Output strict JSON only: {{"entity_name": {{"start": ["state1", ...], "end": ["state1", ...]}}, ...}}
- No prose, no explanation, no markdown fences."""

START_END_PROMPT = """You are identifying, for each entity in a state space, which of its states are the natural starting point(s) and which are terminal (ending) point(s), based on a source text.

{definition}

{rules}

Text:
{{text}}

State space:
{{state_space}}
""".format(definition=_START_END_DEFINITION, rules=_START_END_RULES)


# Protocol B, Stage 1 prompt: same per-candidate independent classification as Protocol A, but
# with two layers of incremental context:
# - "established facts": states already fully resolved earlier in the sequence (including START
#   states, resolved instantly without an LLM call), shown as known context -- informational only,
#   never a shortcut to skip grounding for the candidates currently being judged.
# - "already validated for this state": candidates already found valid in earlier batches of the
#   SAME target state, shown to keep classification internally consistent across batches.

_CLASSIFICATION_DEFINITION_B = """Definitions:
- A precondition candidate is a conjunction of entity.state pairs (joined by " AND ") that was mechanically generated as a plausible precondition for a target state.
- A candidate is VALID if the text genuinely and specifically supports it -- not just plausible, but stated or clearly and unambiguously implied by a specific part of the text.
- Established facts are preconditions already resolved earlier in this process, for OTHER states. They are given as known context to help you understand the process so far -- they are not evidence for the candidates you are judging now, and do not exempt you from finding a real textual justification for each candidate in front of you.
- Already-validated candidates (if any) are candidates already found valid, in an earlier batch, for this SAME target state. They are shown so your judgment on the remaining candidates stays consistent with what was already established -- do not contradict them without strong textual reason, and do not simply repeat them."""

_CLASSIFICATION_RULES_B = """Rules:
- Classify every single candidate in the current batch independently. Do not skip any.
- More than one candidate may be VALID for the same target state.
- Never invent a new candidate, never modify a candidate's wording.
- Every VALID classification must include a verbatim quote from the text, copied exactly, character for character.
- If you cannot find a real quote justifying a candidate, classify it INVALID, even if it seems reasonable or matches an established fact.
- Output strict JSON only, for every candidate in the CURRENT batch only: {{"<candidate string>": {{"valid": true, "quote": "<verbatim quote>"}} | {{"valid": false, "quote": null}}, ...}}
- No prose, no explanation, no markdown fences."""

CLASSIFICATION_PROMPT_B = """You are classifying candidate preconditions for a target state, based on a source text.

{definition}

{rules}

Text:
{{text}}

Established facts (already resolved for other states, context only):
{{established_facts}}

Already validated for this same target state (earlier batches, for consistency only):
{{already_validated}}

Target state: {{target_state}}

Candidates to classify now:
{{candidates}}
""".format(definition=_CLASSIFICATION_DEFINITION_B, rules=_CLASSIFICATION_RULES_B)