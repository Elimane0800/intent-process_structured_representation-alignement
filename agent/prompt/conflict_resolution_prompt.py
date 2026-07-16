CONFLICT_RESOLUTION_PROMPT = """You previously extracted the following state space from a text.

Text:
{text}

Extracted state space:
{state_space}

Review your own extraction against the definition below.

An entity is an identifiable object or case that persists through the process and progresses through a sequence of distinct conditions over time (its lifecycle).
An entity is NOT:
- an attribute or data value carried by an entity
- a boolean flag
- a role or actor, unless the text itself treats that actor as the subject undergoing the lifecycle

For each entity in the extraction, check:
1. Is it really an entity, or is it an actor that only triggers transitions for other entities?
2. Are its states mutually exclusive, or are some of them events/conditions that could hold at the same time?

Revise the extraction accordingly: remove or merge entities that fail check 1, and merge or discard states that fail check 2. Do not add new entities that were not in the original extraction.

After your review, output the line "FINAL_ANSWER:" followed immediately by strict JSON only, in the same format as the original extraction: {{"entity_name": ["state1", "state2", ...], ...}}
Nothing must follow the JSON. No prose, no explanation, no markdown fences.
"""