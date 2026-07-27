# Single, narrow retry for a precondition that was mechanically rejected. Context is kept
# deliberately tight: only the target state, its rejected previous attempt, the exact rejection
# reason, and the list of valid states -- not the full graph, not other states' results. This is
# a single attempt, never a loop: the result goes back through the exact same mechanical
# validation as the original answer, so it can never regress below the current UNRESOLVED status.
#
# Bug found while extending the main grammar to DNF (mixed AND/OR, precondition_prompt.py v3):
# this retry prompt still described the PRE-v2 grammar ("additive only, OR is not supported"),
# never updated when OR was added, let alone for the new mixed-AND/OR case -- retries were
# therefore silently held to a STRICTER, stale grammar than the main extraction call, which
# could only ever regress a retry's expressive power, never help it recover a case the main call
# could have expressed. Fixed then: mirrors the DNF grammar (precondition_prompt.py v3).
#
# v4: mirrors precondition_prompt.py v4 -- NOT (unary negation) added, for the exact same reason
# this file exists in the first place: if the main grammar supports NOT and this retry prompt
# does not, a state downgraded specifically because it needed negation would be retried under a
# STRICTER grammar than the one that could have solved it -- the same category of bug as above,
# for a different operator. Updated in the same commit as precondition_prompt.py's v4 change.
#
# v5: mirrors precondition_prompt.py v5 -- "quote" may now be a single verbatim string OR an
# ordered list of verbatim segments, for rules whose justification is genuinely built across
# separate, non-adjacent sentences rather than one continuous span. Same discipline as v3/v4
# above: a retry left describing only the single-quote form would silently regress the retry's
# expressive power below the main call's, exactly the failure mode this file's own history
# warns against. The two conditions that keep the multi-segment form anti-hallucination (order,
# and no disjunctive marker between segments backing an AND) are restated here too, condensed
# for the narrow retry context -- omitting them would let a retry "recover" a rule the main call
# correctly refused as fabricated.

_RETRY_DEFINITION = """Definitions:
- This is a single retry for one state whose previously proposed precondition was mechanically rejected.
- A precondition is the state (or combination of states) that must already hold before the target state can be reached, expressed only as entity.state pairs from the list of valid states given below.
- INITIAL means no precondition is needed. UNRESOLVED means the text does not support any precondition expressible this way -- always a legitimate answer, never forced.
- Preconditions can be a pure AND ("A AND B"), a pure OR ("A OR B"), a negated term (prefix a single term with "NOT ", e.g. "NOT A"), or a mix of these -- but AND/OR mixed together is ONLY valid if every AND-group of 2+ terms (negated terms included) is wrapped in parentheses before being joined with OR, e.g. "(A AND NOT B) OR C". A bare AND next to an OR with no parentheses is never valid. A pure AND or a pure OR needs no parentheses, and NOT never needs its own parentheses -- only attach it directly to one term, never to a parenthesized group.
- Only negate a state that is already in the list of valid states below. A term and its own exact negation (e.g. "A" and "NOT A") can never appear together in the same AND-group -- that is a direct contradiction, not a real rule."""

_RETRY_RULES = """Rules:
- Your previous answer was rejected for the specific reason given below. Produce a new answer that avoids that exact problem -- do not repeat it.
- Only reference entity.state pairs from the list of valid states given below. Never invent one, never reference the target state itself (negated or not), in any AND-group or OR-alternative.
- Two DIFFERENT states of the same entity can never appear together inside the same AND-group (whether or not that group is one of several OR-alternatives). Two states of the same entity as two different OR-alternatives is fine.
- If you give a precondition (not INITIAL or UNRESOLVED), you must ground it with a "quote", in one of two forms: (a) the default -- a single verbatim quote, copied exactly, character for character, that justifies the WHOLE rule in one continuous span of the text; or (b) only if no such single span exists but the rule is genuinely built from facts in separate, non-adjacent sentences -- an ORDERED LIST of two or more verbatim segments, each copied exactly from the text, appearing in the SAME ORDER as listed, with NO disjunctive language ("or", "unless", "either", "otherwise") in the text between segments backing an AND. If you cannot ground the rule either way, answer UNRESOLVED instead -- this is always acceptable, and strictly better than a partial, invented, or wrongly-combined quote.
- Output strict JSON only: {{"precondition": "INITIAL" | "UNRESOLVED" | "entity.state AND entity.state ..." | "entity.state OR entity.state ..." | "entity.state AND NOT entity.state ..." | "(entity.state AND entity.state) OR entity.state ...", "quote": "<verbatim quote>" | ["<verbatim segment 1>", "<verbatim segment 2>", "..."] | null}}
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