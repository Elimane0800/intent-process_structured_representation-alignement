# Single, narrow retry for a precondition that was mechanically rejected. Context is kept
# deliberately tight: only the target state, its rejected previous attempt, the exact rejection
# reason, and the list of valid states -- not the full graph, not other states' results. This is
# a single attempt, never a loop: the result goes back through the exact same mechanical
# validation as the original answer, so it can never regress below the current UNRESOLVED status.

_RETRY_DEFINITION = """Definitions:
- This is a single retry for one state whose previously proposed precondition was mechanically rejected.
- A precondition is the state (or combination of states) that must already hold before the target state can be reached, expressed only as entity.state pairs from the list of valid states given below.
- INITIAL means no precondition is needed. UNRESOLVED means the text does not support any precondition expressible this way -- always a legitimate answer, never forced.
- Preconditions are additive only (joined by AND). Disjunction (OR) is not supported."""

_RETRY_RULES = """Rules:
- Your previous answer was rejected for the specific reason given below. Produce a new answer that avoids that exact problem -- do not repeat it.
- Only reference entity.state pairs from the list of valid states given below. Never invent one, never reference the target state itself.
- Two states of the same entity can never appear together in one precondition.
- If you give a precondition (not INITIAL or UNRESOLVED), you must include a verbatim quote from the text, copied exactly, character for character, that justifies it. If you cannot find one, answer UNRESOLVED instead -- this is always acceptable.
- Output strict JSON only: {{"precondition": "INITIAL" | "UNRESOLVED" | "entity.state AND entity.state ...", "quote": "<verbatim quote>" | null}}
- No prose, no explanation, no markdown fences."""

RETRY_PROMPT = """You are retrying a single precondition that was previously rejected by a mechanical check.

{definition}

{rules}

Text:
{{text}}

Target state: {{target_state}}

Previous rejected answer: {{previous_precondition}}
Rejection reason: {{reason}}

Valid states you may reference (excluding the target itself):
{{valid_states}}
""".format(definition=_RETRY_DEFINITION, rules=_RETRY_RULES)