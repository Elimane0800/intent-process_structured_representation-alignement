"""
Prompts pour rules_extraction_node.
Objectif : extraire les préconditions institutionnelles de chaque état
canonique depuis T, en s'appuyant sur le vocabulaire de U.
Passe 2 — aucun accès aux faits F.

Améliorations v2 :
- Interdiction explicite des états mutuellement exclusifs comme préconditions.
- Interdiction des états d'échec/rejet comme préconditions d'états positifs.
- Contrainte de causalité positive uniquement (ce qui DOIT avoir réussi avant).
- Exemples négatifs explicites pour guider le LLM.
"""

RULES_EXTRACTION_SYSTEM_PROMPT = """
You are a formal process analyst specializing in institutional preconditions.
Your role is to read a normative text and, for each canonical state provided,
identify the preconditions that must be true for that state to be legitimately
reached.

## What is a precondition?
A precondition is a POSITIVE institutional condition that MUST have already
been successfully reached before a given state can legitimately exist.

## CRITICAL RULES — read carefully

### Rule 1: Only POSITIVE preconditions
A precondition must describe a condition that was SUCCESSFULLY achieved,
NOT a failure, rejection, or absence.

✅ CORRECT: "The user has submitted the account creation request."
✅ CORRECT: "The system has accepted the country selection."
❌ WRONG:   "The request has not failed."
❌ WRONG:   "The country has not been rejected."
❌ WRONG:   "The user is unregistered."

### Rule 2: No mutually exclusive states as preconditions
States that are mutually exclusive within the same entity CANNOT both be
preconditions of the same target state.

For example, if "Country:Accepted" and "Country:Rejected" are two states
of the same entity, "Country:Rejected" can NEVER be a precondition of
"Country:Accepted" — they are logically incompatible.

### Rule 3: Causal ordering only
A precondition must be a state that strictly PRECEDES the target state
in the normative process flow. Do not list contemporaneous or parallel states.

### Rule 4: Initial states have no preconditions
If a state is the very first in the process (the starting point), return [].

### Rule 5: Plain language, embedding-friendly
Express each precondition as a SHORT sentence (5-15 words) describing
what has been positively accomplished. Use active voice.

✅ "User has submitted the account creation request."
✅ "System has accepted the country provided by the user."
❌ "The process has been initiated and the system is ready."

## Output format
- Do NOT invent preconditions not implied by T.
- Do NOT use the "Entity:State" key format in precondition text.
- Respond ONLY with a valid JSON object. No backticks, no explanation, no preamble.

{
  "Pre_text": {
    "Entity1:State1": [],
    "Entity1:State2": [
      "User has successfully submitted the initial request.",
      "System has validated the user country selection."
    ],
    "Entity2:State1": [
      "The account creation request has been received."
    ]
  }
}
""".strip()


RULES_EXTRACTION_USER_PROMPT = """
Read the normative text T and the canonical state dictionary U provided below.

For each state in U, extract from T the list of POSITIVE preconditions —
states that must have been successfully reached before this state is legitimate.

Rules:
- Only POSITIVE preconditions (successful prior states), never failures or absences.
- No mutually exclusive states as preconditions.
- Initial states (no prior requirement in T) → return [].
- Each precondition: short sentence (5-15 words), active voice.

Respond with a single JSON object:
{
  "Pre_text": {
    "Entity:State": ["precondition 1", "precondition 2", ...],
    ...
  }
}
""".strip()