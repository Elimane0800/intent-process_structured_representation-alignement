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
# discarding grounded, correct extractions rather than a model error.
#
# v3: mixed AND/OR is now supported too, via a one-level DNF grammar (a disjunction of
# conjunctions) -- v2 still forced UNRESOLVED on any genuinely mixed rule (e.g. "(A and B) or
# C"), which turned out to be an unjustified restriction, not a real grammar limit: the "claim
# is approved and bank details are confirmed, or the claim is auto-approved" case (see Example
# 2 below) was a real, textually-grounded rule that v2 discarded purely because the expressible
# grammar was too narrow, not because the text was unclear. Parentheses are the only new syntax,
# and only REQUIRED when AND and OR are mixed in the same precondition -- a pure AND or a pure OR
# precondition is written exactly as before, no parentheses needed. This keeps the grammar as
# narrow as the data actually requires rather than reaching for a general boolean parser.

# v4: unary negation (NOT) is now supported, as a direct response to a gap confirmed across
# every test run so far -- text that negates a state directly ("unless X", "if it has not been
# reported") was systematically forced to UNRESOLVED, discarding grounded, textually-explicit
# rules for no reason other than the grammar lacking the operator. NOT applies to a single
# entity.state term only -- never to a parenthesized group -- and only ever references a state
# ALREADY present in the state space; it is not a way to invent a new state, only to reference
# the absence of one that was already extracted. This is deliberately narrower than a general
# negation operator: the text does not need a separate "expired"/"inactive"/"rejected" state to
# already exist for its negation to be expressible -- NOT lets the rule reference the absence of
# whichever positive state the text DID produce, instead of forcing an artificial complementary
# state into existence just to avoid negation.

_PRECONDITION_DEFINITION = """Definitions:
- A precondition is the state (or combination of states) that must already hold before an entity can reach a given target state. A precondition is always expressed using only entity.state pairs that are explicitly listed in the provided state space -- never a sentence, never a newly invented state.
- An INITIAL state is a state that requires no precondition: it is the natural starting point of an entity's lifecycle, reached without any other state needing to hold first.
- Preconditions can be additive (AND), alternative (OR), negated (NOT), or a combination of these:
  - Use AND when the text requires several states to hold at once (e.g. "job_application.rated AND probation_phase.completed").
  - Use OR when the text expresses that any one of several states is independently sufficient to trigger the target (e.g. "work_accident.fatal OR work_accident.serious_injury").
  - Use NOT immediately before a single entity.state term (e.g. "NOT warranty.active") when the text requires that an already-listed state does NOT hold. NOT attaches to exactly one term, never to a parenthesized group ("NOT (A AND B)" is not valid). Only negate a state that is already in the state space -- if the text negates something that was never extracted as a state, that is a gap in an earlier step, not something to invent here; use UNRESOLVED instead.
  - If the text genuinely requires a mix of AND and OR (e.g. "A, and either B or C" -- meaning "(A and B) or C"), you may express it, but ONLY by wrapping every AND-group in parentheses: "(A AND B) OR C". A bare AND next to an OR with no parentheses is never valid, because it does not say which operator binds first.
  - A pure AND (no OR anywhere) or a pure OR (no AND anywhere) never needs parentheses -- write it exactly as before: "A AND B", or "A OR B OR C". NOT does not require parentheses either: "A AND NOT B" and "NOT A OR B" are both written directly, with no extra wrapping needed around the negated term itself.
  - Parentheses may wrap only a group of 2 or more terms joined by AND. Never wrap a single term (negated or not) in parentheses, and never nest parentheses inside a group.
- Mutual exclusivity applies only within a single AND-group (whether or not it is one of several OR-alternatives): two DIFFERENT states of the SAME entity can never both appear inside the same AND-group, because an entity can only be in one of its states at a time -- if the text seems to require this, that specific AND-group is contradictory and the whole precondition must be UNRESOLVED. This restriction does NOT apply across OR-alternatives: two states of the same entity appearing as two different OR-branches is expected and valid (e.g. "fatal OR serious_injury" are two different, mutually exclusive outcomes of the same entity, and either one alone can trigger the target -- that is exactly what OR is for).
- A term and its exact negation (e.g. "warranty.active" and "NOT warranty.active") can never appear together in the same AND-group -- that asserts a state both holds and does not hold at once, a direct contradiction, not a real rule. If the text seems to require this, mark the whole precondition UNRESOLVED instead.
- A state can never be part of its own precondition, directly, negated, or nested inside any AND-group or OR-alternative (e.g. neither "self_employed_person.reported" nor "NOT self_employed_person.reported" can be a precondition of "self_employed_person.reported"). A state cannot legitimize reaching itself, positively or negatively -- if this seems to be the case, mark it UNRESOLVED instead.
- UNRESOLVED is the correct answer whenever the text does not support a precondition expressible this way, including: purely temporal transitions with no triggering state ("after some time"), conditions that are not states of any entity in the state space (including a negation of something that was never extracted as a state), and logic that this grammar cannot express even with parentheses and NOT (e.g. nested groups, or an AND/OR mix with no verbatim quote covering the whole rule)."""

_PRECONDITION_RULES = """Rules:
- Only reference entity.state pairs that literally appear in the provided state space. Never invent a new entity or a new state.
- Base every precondition on something stated or clearly implied in the text -- do not guess a plausible business rule that is not supported by the text.
- Every non-INITIAL, non-UNRESOLVED precondition must be grounded by a "quote" in one of two forms:
  - The default: a single verbatim quote (a string), copied exactly, character for character, that justifies the WHOLE rule in one continuous span of the text. Use this whenever a single such span exists.
  - Only when the rule's justification is genuinely built across separate, non-adjacent sentences rather than any single continuous span: an ORDERED LIST of two or more verbatim segments (a JSON array of strings). Each segment must independently be copied exactly, character for character, from the text, and the segments must appear in the SAME ORDER in the text as you list them. Do not use this form to combine facts the text does not actually present together -- if the text presents the facts as alternatives (using language like "or", "either", "unless", "otherwise") rather than as a joint requirement, that is grounds for OR or UNRESOLVED, never a fabricated AND grounded across disjunctive language.
- If you cannot find a real quote (single or ordered multi-segment) for a rule, use UNRESOLVED instead, even if the rule seems reasonable.
- INITIAL and UNRESOLVED never need a quote -- use null for the quote in those cases.
- Join terms with " AND " for a conjunction, " OR " for a disjunction. Prefix a single term with "NOT " (note the space) to negate it, e.g. "entity.state AND NOT entity.state" or "NOT entity.state OR entity.state" -- NOT never needs its own parentheses. If you need both AND and OR in the same precondition, wrap every AND-group of 2+ terms (negated terms included) in parentheses before joining with " OR " -- e.g. "(entity.state AND NOT entity.state) OR entity.state". Never write a bare AND next to an OR without parentheses.
- Every state in the state space must appear exactly once in your output.
- Do not explain your reasoning outside the output. Output strict JSON only: {{"entity.state": {{"precondition": "INITIAL" | "UNRESOLVED" | "entity.state AND entity.state ..." | "entity.state OR entity.state ..." | "entity.state AND NOT entity.state ..." | "(entity.state AND entity.state) OR entity.state ...", "quote": "<verbatim quote>" | ["<verbatim segment 1>", "<verbatim segment 2>", "..."] | null}}, ...}}
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

State space: {{"appointment": ["scheduled", "confirmed", "ready", "completed", "escalated"], "patient_file": ["incomplete", "complete"], "vitals": ["abnormal", "critical"]}}

Output: {{
  "appointment.scheduled": {{"precondition": "INITIAL", "quote": null}},
  "appointment.confirmed": {{"precondition": "appointment.scheduled", "quote": "A nurse confirms the appointment, making it confirmed"}},
  "appointment.ready": {{"precondition": "appointment.confirmed AND patient_file.complete", "quote": "Once the appointment is confirmed and the patient's file is complete, the appointment becomes ready"}},
  "appointment.completed": {{"precondition": "UNRESOLVED", "quote": null}},
  "appointment.escalated": {{"precondition": "vitals.abnormal OR vitals.critical", "quote": "If the patient's vitals become abnormal or critical during the visit, the appointment is escalated"}},
  "patient_file.incomplete": {{"precondition": "INITIAL", "quote": null}},
  "patient_file.complete": {{"precondition": "UNRESOLVED", "quote": null}},
  "vitals.abnormal": {{"precondition": "INITIAL", "quote": null}},
  "vitals.critical": {{"precondition": "INITIAL", "quote": null}}
}}

Note: "appointment.ready" combines two different entities with AND, backed by a single quote covering both conditions. "appointment.completed" is UNRESOLVED because "after some time" gives no triggering state, only an implicit temporal gap -- no quote is given. "patient_file.complete" is UNRESOLVED because the text never states what causes the file to become complete. "appointment.scheduled", "patient_file.incomplete", "vitals.abnormal" and "vitals.critical" are INITIAL because nothing needs to hold before them -- no quote needed. "appointment.escalated" uses OR because either vitals state alone is enough to trigger it -- note both terms belong to the SAME entity ("vitals"), which is allowed in an OR (unlike inside a single AND-group). No parentheses are needed anywhere in this example: every precondition here is a pure AND or a pure OR, never a mix."""


_CLAIM_EXAMPLE = """Example 2:
Text: "An insurance claim process. A claim starts as filed. If the claim is either approved or auto-approved, it becomes payable. If the claim is approved and the claimant's bank details are confirmed, or if the claim is auto-approved, it becomes paid immediately. If the claim is rejected, the process ends there as rejected. After some time without complaint, a paid claim becomes closed."

State space: {{"claim": ["filed", "approved", "auto_approved", "payable", "paid", "rejected", "closed"], "bank_details": ["unconfirmed", "confirmed"]}}

Output: {{
  "claim.filed": {{"precondition": "INITIAL", "quote": null}},
  "claim.approved": {{"precondition": "UNRESOLVED", "quote": null}},
  "claim.auto_approved": {{"precondition": "UNRESOLVED", "quote": null}},
  "claim.payable": {{"precondition": "claim.approved OR claim.auto_approved", "quote": "If the claim is either approved or auto-approved, it becomes payable"}},
  "claim.paid": {{"precondition": "(claim.approved AND bank_details.confirmed) OR claim.auto_approved", "quote": "If the claim is approved and the claimant's bank details are confirmed, or if the claim is auto-approved, it becomes paid immediately"}},
  "claim.rejected": {{"precondition": "UNRESOLVED", "quote": null}},
  "claim.closed": {{"precondition": "UNRESOLVED", "quote": null}},
  "bank_details.unconfirmed": {{"precondition": "INITIAL", "quote": null}},
  "bank_details.confirmed": {{"precondition": "UNRESOLVED", "quote": null}}
}}

Note: "claim.payable" is a clean OR, same pattern as example 1. "claim.paid" is now expressible: the rule is "(approved AND bank_details.confirmed) OR auto_approved" -- a genuine mix of AND and OR, written with the AND-group in parentheses so it is unambiguous which terms are joined by AND before the OR applies; do NOT write "claim.approved AND bank_details.confirmed OR claim.auto_approved" without parentheses, that is invalid. "claim.approved" and "claim.auto_approved" are UNRESOLVED because the text never states what causes approval. "claim.closed" is UNRESOLVED for the same temporal reason as "appointment.completed" in example 1, even though "paid" is clearly implied before it."""


_NOT_EXAMPLE = """Example 3:
Text: "A customer submits a device for repair, and the device becomes submitted. The manufacturer checks whether the device is still under warranty; if it is, the warranty becomes active. If a submitted device's warranty is active, the repair becomes free. If a submitted device's warranty is not active, the repair becomes charged instead."

State space: {{"device": ["submitted"], "warranty": ["active"], "repair": ["free", "charged"]}}

Output: {{
  "device.submitted": {{"precondition": "INITIAL", "quote": null}},
  "warranty.active": {{"precondition": "UNRESOLVED", "quote": null}},
  "repair.free": {{"precondition": "device.submitted AND warranty.active", "quote": "If a submitted device's warranty is active, the repair becomes free"}},
  "repair.charged": {{"precondition": "device.submitted AND NOT warranty.active", "quote": "If a submitted device's warranty is not active, the repair becomes charged instead"}}
}}

Note: "warranty.active" is UNRESOLVED because the text only states that warranty status is checked, never what causes the check itself to become active. "repair.charged" uses NOT because the state space has no separate state for warranty being inactive or expired -- the text negates the existing "active" state directly, and NOT exists exactly for this: referencing the absence of an already-listed state, never a reason to invent a new positive state (like "warranty.expired") just to avoid negation."""


_MULTISEGMENT_EXAMPLE = """Example 4:
Text: "A shipment is checked in by the warehouse team, becoming received. It is separately inspected and marked as passed by quality control. Once both of those have happened, the shipment is released to sellable stock, becoming released."

State space: {{"shipment": ["received", "passed", "released"]}}

Output: {{
  "shipment.received": {{"precondition": "INITIAL", "quote": null}},
  "shipment.passed": {{"precondition": "INITIAL", "quote": null}},
  "shipment.released": {{"precondition": "shipment.received AND shipment.passed", "quote": ["is checked in by the warehouse team", "inspected and marked as passed by quality control"]}}
}}

Note: no single continuous span in the text states both "received" and "passed" together -- the third sentence only refers back with "once both of those have happened", it does not itself name the two facts. A single quote could not justify "shipment.released" here, so an ORDERED LIST of two segments is used instead: one from the first sentence, one from the second, in the order they appear in the text. This is only valid because there is no disjunctive language ("or", "unless", etc.) between the two segments -- if the text had instead said the shipment was released once it was EITHER received OR passed, the same two segments would NOT justify an AND, and the correct answer would use OR instead."""


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

{example3}

{example4}

Now construct the preconditions for the following text and state space, following the same approach.

Text:
{{text}}

State space:
{{state_space}}
""".format(definition=_PRECONDITION_DEFINITION, rules=_PRECONDITION_RULES, example1=_CLINIC_EXAMPLE, example2=_CLAIM_EXAMPLE, example3=_NOT_EXAMPLE, example4=_MULTISEGMENT_EXAMPLE)


PRECONDITION_PROMPT_COT = """You are constructing the precondition for every state of a previously extracted state space, based on a source text.

{definition}

{rules}

Work through the following steps explicitly before answering:

Step 1 -- For every entity.state in the state space, scan the text for any clause that describes what must already hold, or what event/condition triggers reaching it.
Step 2 -- For each state, classify what you found: a single triggering condition, several conditions that must ALL hold at once (AND), several conditions where ANY ONE alone is sufficient (OR), a condition that requires an already-listed state to explicitly NOT hold, a genuine mix of AND/OR ("(A and B) or C"-shaped, possibly with a negated term inside), a purely temporal trigger with no state involved, or nothing at all.
Step 3 -- For AND, OR, NOT, and mixed cases, locate the exact verbatim span in the text that justifies the WHOLE rule, copied character for character. If no single continuous span covers the whole rule but the rule is genuinely built from facts stated in separate, non-adjacent sentences, use an ORDERED LIST of verbatim segments instead (one per sentence, in the order they appear in the text) -- but only if there is no disjunctive language ("or", "unless", "either", "otherwise") between those sentences, since that would mean the text presents the facts as alternatives, not a joint requirement. If neither a single span nor a valid ordered list of segments exists, downgrade this state to UNRESOLVED rather than inventing a quote or quoting only part of it.
Step 4 -- Re-check every AND-group (whether it is the whole precondition or one OR-alternative inside a mixed rule) for two kinds of conflict: (a) two DIFFERENT states of the same entity combined together, since an entity cannot hold two of its own states at once; (b) a term and its own exact negation both present (e.g. "warranty.active" and "NOT warranty.active"), a direct contradiction. Either conflict forces UNRESOLVED for the whole precondition. Neither check applies across different OR-alternatives.
Step 5 -- Re-check every precondition for self-reference: if the target state itself appears anywhere in it, negated or not, in any AND-group or OR-alternative, mark it UNRESOLVED instead.
Step 6 -- Re-check every mixed AND/OR case identified in Step 2: confirm every AND-group of 2+ terms (negated terms included) is wrapped in parentheses before being joined with " OR " (e.g. "(A AND NOT B) OR C"), never a bare AND next to an OR. If you cannot express the mix this way (e.g. it would require nested parentheses, or NOT applied to a whole parenthesized group), mark it UNRESOLVED instead of guessing.

Do not skip a state because it seems to have no clear answer -- INITIAL and UNRESOLVED are always legitimate, explicit answers, not omissions.

After Step 6, output the line "FINAL_ANSWER:" followed immediately by strict JSON only, no prose, no markdown fences:
{{{{"entity.state": {{{{"precondition": "INITIAL" | "UNRESOLVED" | "entity.state AND entity.state ..." | "entity.state OR entity.state ..." | "entity.state AND NOT entity.state ..." | "(entity.state AND entity.state) OR entity.state ...", "quote": "<verbatim quote>" | null}}}}, ...}}}}

Text:
{{text}}

State space:
{{state_space}}
""".format(definition=_PRECONDITION_DEFINITION, rules=_PRECONDITION_RULES)