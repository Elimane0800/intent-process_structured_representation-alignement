# Naive baseline for Pre: the LLM receives the source text and the already-validated state space
# U (from Protocol 3), and must propose, for every state, the precondition that legitimizes
# reaching it -- expressed only in terms of other states already present in U.
#
# Each non-INITIAL/UNRESOLVED precondition must now be accompanied by a verbatim quote from the
# text -- the LLM still does all the semantic work in a single call (empirically the best-
# performing approach among everything tested so far), but a downstream mechanical layer verifies
# the quote actually exists in the text and that referenced states actually exist in U, before
# accepting the precondition. This mirrors the same grounding mechanism already used and proven
# on U itself (Protocol 1) and on the combinatorial protocols (A/B) -- generation stays a single
# LLM call, verification is deterministic and separate.
#
# v2: OR is now supported, as a direct response to empirical observation -- across multiple
# models and runs, the text genuinely expresses disjunctive triggers (e.g. "any work accident
# that leads to a fatality OR serious injury"), and forcing UNRESOLVED on every one of them was
# discarding grounded, correct extractions rather than a model error. Scope stays narrow and
# defendable: a precondition is either a pure conjunction (AND) or a pure disjunction (OR),
# never a mix of both at the same level -- mixed logic (e.g. "A AND (B OR C)") remains UNRESOLVED,
# exactly like disjunction did in v1. This mirrors the additive-only decision already made for
# AND: keep the expressible grammar small and unambiguous rather than reaching for a general
# boolean parser this task does not need yet.

_PRECONDITION_DEFINITION = """Definitions:
- A precondition is the state (or combination of states) that must already hold before an entity can reach a given target state. A precondition is always expressed using only entity.state pairs that are explicitly listed in the provided state space -- never a sentence, never a newly invented state.
- An INITIAL state is a state that requires no precondition: it is the natural starting point of an entity's lifecycle, reached without any other state needing to hold first.
- Preconditions can be additive (AND) or alternative (OR), but never both mixed in the same precondition:
  - Use AND when the text requires several states to hold at once (e.g. "job_application.rated AND probation_phase.completed").
  - Use OR when the text expresses that any one of several states is independently sufficient to trigger the target (e.g. "work_accident.fatal OR work_accident.serious_injury").
  - If the text requires a mix of both at once (e.g. "A, and either B or C"), do not try to express it -- mark the precondition UNRESOLVED instead. This task only supports a single pure operator per precondition.
- Mutual exclusivity applies only within an AND: two states of the SAME entity can never both appear in an AND-precondition, because an entity can only be in one of its states at a time -- if the text seems to require this, the precondition is contradictory and must be UNRESOLVED. This restriction does NOT apply to OR: two states of the same entity appearing in an OR is expected and valid (e.g. "fatal OR serious_injury" are two different, mutually exclusive outcomes of the same entity, and either one alone can trigger the target -- that is exactly what OR is for).
- A state can never be part of its own precondition, directly or as part of an AND or OR group (e.g. "self_employed_person.reported" cannot be a precondition of itself). A state cannot legitimize reaching itself -- if this seems to be the case, mark it UNRESOLVED instead.
- UNRESOLVED is the correct answer whenever the text does not support a precondition expressible this way, including: purely temporal transitions with no triggering state ("after some time"), conditions that are not states of any entity in the state space, negated conditions, and any logic that mixes AND and OR together."""

_PRECONDITION_RULES = """Rules:
- Only reference entity.state pairs that literally appear in the provided state space. Never invent a new entity or a new state.
- Base every precondition on something stated or clearly implied in the text -- do not guess a plausible business rule that is not supported by the text.
- Every non-INITIAL, non-UNRESOLVED precondition must be accompanied by a verbatim quote from the text, copied exactly, character for character, that justifies it. If you cannot find a real quote, use UNRESOLVED instead, even if the precondition seems reasonable.
- INITIAL and UNRESOLVED never need a quote -- use null for the quote in those cases.
- Join terms with " AND " for a conjunction, or " OR " for a disjunction -- never both operators in the same precondition string.
- Every state in the state space must appear exactly once in your output.
- Do not explain your reasoning outside the output. Output strict JSON only: {{"entity.state": {{"precondition": "INITIAL" | "UNRESOLVED" | "entity.state AND entity.state ..." | "entity.state OR entity.state ...", "quote": "<verbatim quote>" | null}}, ...}}
- No prose, no explanation, no markdown fences."""


PRECONDITION_PROMPT = """You are constructing the precondition for every state of a previously extracted state space, based on a source text.

{definition}

{rules}

Text:
{{text}}

State space:
{{state_space}}
""".format(definition=_PRECONDITION_DEFINITION, rules=_PRECONDITION_RULES)


_CLINIC_EXAMPLE = """Example 1:
Text: "A clinic processes patient appointments. When a patient books an appointment, it becomes scheduled. A nurse confirms the appointment, making it confirmed. Once the appointment is confirmed and the patient's file is complete, the appointment becomes ready. After some time, a ready appointment becomes completed. If the patient's vitals become abnormal or critical during the visit, the appointment is escalated."

State space: {{{{"appointment": ["scheduled", "confirmed", "ready", "completed", "escalated"], "patient_file": ["incomplete", "complete"], "vitals": ["abnormal", "critical"]}}}}

Output: {{{{
  "appointment.scheduled": {{{{"precondition": "INITIAL", "quote": null}}}},
  "appointment.confirmed": {{{{"precondition": "appointment.scheduled", "quote": "A nurse confirms the appointment, making it confirmed"}}}},
  "appointment.ready": {{{{"precondition": "appointment.confirmed AND patient_file.complete", "quote": "Once the appointment is confirmed and the patient's file is complete, the appointment becomes ready"}}}},
  "appointment.completed": {{{{"precondition": "UNRESOLVED", "quote": null}}}},
  "appointment.escalated": {{{{"precondition": "vitals.abnormal OR vitals.critical", "quote": "If the patient's vitals become abnormal or critical during the visit, the appointment is escalated"}}}},
  "patient_file.incomplete": {{{{"precondition": "INITIAL", "quote": null}}}},
  "patient_file.complete": {{{{"precondition": "UNRESOLVED", "quote": null}}}},
  "vitals.abnormal": {{{{"precondition": "INITIAL", "quote": null}}}},
  "vitals.critical": {{{{"precondition": "INITIAL", "quote": null}}}}
}}}}

Note: "appointment.ready" combines two different entities with AND, backed by a single quote covering both conditions. "appointment.completed" is UNRESOLVED because "after some time" gives no triggering state, only an implicit temporal gap -- no quote is given. "patient_file.complete" is UNRESOLVED because the text never states what causes the file to become complete. "appointment.scheduled", "patient_file.incomplete", "vitals.abnormal" and "vitals.critical" are INITIAL because nothing needs to hold before them -- no quote needed. "appointment.escalated" uses OR because either vitals state alone is enough to trigger it -- note both terms belong to the SAME entity ("vitals"), which is allowed in an OR (unlike in an AND)."""


_CLAIM_EXAMPLE = """Example 2:
Text: "An insurance claim process. A claim starts as filed. If the claim is either approved or auto-approved, it becomes payable. If the claim is approved and the claimant's bank details are confirmed, or if the claim is auto-approved, it becomes paid immediately. If the claim is rejected, the process ends there as rejected. After some time without complaint, a paid claim becomes closed."

State space: {{{{"claim": ["filed", "approved", "auto_approved", "payable", "paid", "rejected", "closed"], "bank_details": ["unconfirmed", "confirmed"]}}}}

Output: {{{{
  "claim.filed": {{{{"precondition": "INITIAL", "quote": null}}}},
  "claim.approved": {{{{"precondition": "UNRESOLVED", "quote": null}}}},
  "claim.auto_approved": {{{{"precondition": "UNRESOLVED", "quote": null}}}},
  "claim.payable": {{{{"precondition": "claim.approved OR claim.auto_approved", "quote": "If the claim is either approved or auto-approved, it becomes payable"}}}},
  "claim.paid": {{{{"precondition": "UNRESOLVED", "quote": null}}}},
  "claim.rejected": {{{{"precondition": "UNRESOLVED", "quote": null}}}},
  "claim.closed": {{{{"precondition": "UNRESOLVED", "quote": null}}}},
  "bank_details.unconfirmed": {{{{"precondition": "INITIAL", "quote": null}}}},
  "bank_details.confirmed": {{{{"precondition": "UNRESOLVED", "quote": null}}}}
}}}}

Note: "claim.payable" is a clean OR, same pattern as example 1. "claim.paid" is UNRESOLVED even though the text gives a clear rule for it -- the rule is "(approved AND bank_details.confirmed) OR auto_approved", a MIX of AND and OR, which this task does not support; do not flatten it into a single AND or a single OR, mark it UNRESOLVED instead. "claim.approved" and "claim.auto_approved" are UNRESOLVED because the text never states what causes approval. "claim.closed" is UNRESOLVED for the same temporal reason as "appointment.completed" in example 1, even though "paid" is clearly implied before it."""


PRECONDITION_PROMPT_FEWSHOT = """You are constructing the precondition for every state of a previously extracted state space, based on a source text.

{definition}

{rules}

{example1}

Now construct the preconditions for the following text and state space, following the same approach.

Text:
{{text}}

State space:
{{state_space}}
""".format(definition=_PRECONDITION_DEFINITION, rules=_PRECONDITION_RULES, example1=_CLINIC_EXAMPLE)


PRECONDITION_PROMPT_TWOSHOT = """You are constructing the precondition for every state of a previously extracted state space, based on a source text.

{definition}

{rules}

{example1}

{example2}

Now construct the preconditions for the following text and state space, following the same approach.

Text:
{{text}}

State space:
{{state_space}}
""".format(definition=_PRECONDITION_DEFINITION, rules=_PRECONDITION_RULES, example1=_CLINIC_EXAMPLE, example2=_CLAIM_EXAMPLE)


PRECONDITION_PROMPT_COT = """You are constructing the precondition for every state of a previously extracted state space, based on a source text.

{definition}

{rules}

Work through the following steps explicitly before answering:

Step 1 -- For every entity.state in the state space, scan the text for any clause that describes what must already hold, or what event/condition triggers reaching it.
Step 2 -- For each state, classify what you found: a single triggering condition, several conditions that must ALL hold at once (AND), several conditions where ANY ONE alone is sufficient (OR), a mix of both (not expressible here), a purely temporal trigger with no state involved, or nothing at all.
Step 3 -- For AND and OR cases only, locate the exact verbatim span in the text that justifies it, copied character for character. If no real verbatim span exists, downgrade this state to UNRESOLVED rather than inventing one.
Step 4 -- Re-check every AND group: reject (mark UNRESOLVED) any AND that combines two states of the same entity, since an entity cannot hold two of its own states at once. This check does not apply to OR groups.
Step 5 -- Re-check every precondition for self-reference: if the target state itself appears in its own AND or OR group, mark it UNRESOLVED instead.
Step 6 -- Re-check every mixed AND/OR case identified in Step 2: confirm it is marked UNRESOLVED, never forced into a single AND or a single OR.

Do not skip a state because it seems to have no clear answer -- INITIAL and UNRESOLVED are always legitimate, explicit answers, not omissions.

After Step 6, output the line "FINAL_ANSWER:" followed immediately by strict JSON only, no prose, no markdown fences:
{{{{"entity.state": {{{{"precondition": "INITIAL" | "UNRESOLVED" | "entity.state AND entity.state ..." | "entity.state OR entity.state ...", "quote": "<verbatim quote>" | null}}}}, ...}}}}

Text:
{{text}}

State space:
{{state_space}}
""".format(definition=_PRECONDITION_DEFINITION, rules=_PRECONDITION_RULES)