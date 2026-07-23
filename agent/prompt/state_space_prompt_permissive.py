# Step A of the extraction protocol: permissive, over-inclusive candidate generation.
# The LLM no longer judges whether a candidate is a "true" entity or an actor -- that decision is
# delegated to a downstream deterministic layer (dependency parsing / WordNet lexnames). This step
# has one job only: do not miss anything. Precision on WHICH candidates to include is not this
# step's responsibility; recall is.
#
# However, formatting discipline IS this step's responsibility, independently of that judgment.
# Observed failure mode this version corrects: candidate names and states degenerating into full
# clauses copied from the source text (e.g. a "state" of {"description of the condition under
# which X is considered part of an event"} instead of a short label) -- this breaks every
# downstream step (symbolic validation, evidence citation, graph construction), regardless of
# whether the candidate itself is entity-worthy. Format correctness and inclusion judgment are
# two separate concerns, and this version enforces the first strictly while staying permissive on
# the second.

_CANDIDATE_DEFINITION = """A candidate is a short noun phrase (1 to 4 words) that names something in the process which could plausibly be described as passing through different conditions over time.

Do not decide whether a candidate is a "real" entity, an actor, or a role -- that judgment is made separately, after this step. Your only responsibility here is completeness: include every candidate that is associated with a condition, an action taken on it, or an outcome in the text, even if you are unsure it will ultimately qualify.

When in doubt about whether to INCLUDE a candidate, include it. Formatting correctness, below, is never optional -- it applies even to candidates you are unsure about."""

_FORMAT_RULES = """Formatting rules for every candidate name and every state -- these apply without exception:
- A candidate name is a short noun phrase, 1 to 4 words, never a sentence or a clause. "childcare facility" is a valid candidate name. "taking a child to or collecting them from a childcare facility" is NOT a valid candidate name -- that is a clause describing an action, not a thing.
- A state is a short condition label, 1 to 3 words, written as a past participle or an adjective (e.g. "submitted", "confirmed", "active", "under_review"). A state is NEVER a full sentence, a clause, a paraphrase of a rule, or a copy of a condition from the text.
- WRONG example: {{"childcare_facility": ["taking a child to or collecting them from is considered part of the work accident conditions"]}}
- RIGHT example: {{"childcare_facility": ["visited"]}}
- If you cannot express a condition as a short label without copying a clause or inventing unstated content, do NOT include it as a state -- omit it rather than write a sentence.
- Never start a candidate name or a state with a verb in -ing form, with "the fact that", "it is considered", or any other clause opener.
- Use lowercase. Use snake_case for multi-word labels (e.g. "information_collected", "childcare_facility")."""


STATE_SPACE_PROMPT_PERMISSIVE = """You are performing the first step of a two-step extraction pipeline: generous candidate generation. A separate, later step will filter your candidates -- do not filter them yourself.

{definition}

{format_rules}

Rules:
- Be over-inclusive on WHICH candidates you list. Do not omit a candidate because you are unsure it qualifies.
- Base every candidate and every state on something stated or clearly implied in the text -- do not invent content absent from the text.
- Output strict JSON only: {{{{"candidate_name": ["state1", "state2", ...], ...}}}}
- No prose, no explanation, no markdown fences.

Text:
{{text}}
""".format(definition=_CANDIDATE_DEFINITION, format_rules=_FORMAT_RULES)


STATE_SPACE_PROMPT_PERMISSIVE_FEWSHOT = """You are performing the first step of a two-step extraction pipeline: generous candidate generation. A separate, later step will filter your candidates -- do not filter them yourself.

{definition}

{format_rules}

Rules:
- Be over-inclusive on WHICH candidates you list. Do not omit a candidate because you are unsure it qualifies.
- Base every candidate and every state on something stated or clearly implied in the text -- do not invent content absent from the text.
- Output strict JSON only: {{{{"candidate_name": ["state1", "state2", ...], ...}}}}
- No prose, no explanation, no markdown fences.

Example:
Text: "A library manages book loans. When a member borrows a book, it becomes checked out. If it is returned on time, it becomes available again. If returned late, it is marked overdue before returning to available. If a book is lost, it becomes lost and is removed from the catalog. Members can reserve a book that is checked out; once returned, a reserved book becomes held for the requesting member before being checked out again."

Output: {{{{"book": ["available", "checked_out", "overdue", "held", "lost"]}}}}

Note: only "book" is described with explicit condition changes in this text. "member" is not included -- the text never describes a condition change happening to a member. Every state above is a single short label, never a clause copied from the text -- for example the state is "checked_out", not "a member borrows the book and it becomes checked out".

Now extract candidates for the following text, following the same approach: generous inclusion, short labels only, grounded only in what the text states or clearly implies.

Text:
{{text}}
""".format(definition=_CANDIDATE_DEFINITION, format_rules=_FORMAT_RULES)


STATE_SPACE_PROMPT_PERMISSIVE_TWOSHOT = """You are performing the first step of a two-step extraction pipeline: generous candidate generation. A separate, later step will filter your candidates -- do not filter them yourself.

{definition}

{format_rules}

Rules:
- Be over-inclusive on WHICH candidates you list. Do not omit a candidate because you are unsure it qualifies.
- Base every candidate and every state on something stated or clearly implied in the text -- do not invent content absent from the text.
- A text may require one candidate or several distinct candidates. Do not force everything into one, and do not split one lifecycle into unrelated fragments.
- Output strict JSON only: {{{{"candidate_name": ["state1", "state2", ...], ...}}}}
- No prose, no explanation, no markdown fences.

Example 1 (single candidate):
Text: "A library manages book loans. When a member borrows a book, it becomes checked out. If it is returned on time, it becomes available again. If returned late, it is marked overdue before returning to available. If a book is lost, it becomes lost and is removed from the catalog. Members can reserve a book that is checked out; once returned, a reserved book becomes held for the requesting member before being checked out again."

Output: {{{{"book": ["available", "checked_out", "overdue", "held", "lost"]}}}}

Note: "member" is not included -- the text never describes a condition change happening to a member. Every state is a single short label, never a clause.

Example 2 (two distinct candidates):
Text: "An online store processes customer orders. When a customer places an order, it becomes pending. The store confirms the order, making it confirmed. Once confirmed, a shipment is created for the order and starts as preparing. The shipment becomes shipped once it leaves the warehouse, and delivered once the customer receives it. If a shipment is lost in transit, it becomes lost. If the customer cancels before shipping, the order becomes cancelled and no shipment is created."

Output: {{{{"order": ["pending", "confirmed", "cancelled"], "shipment": ["preparing", "shipped", "delivered", "lost"]}}}}

Note: "order" and "shipment" both have explicit condition changes described in the text, so both are included as separate candidates. "customer" and "store" are not. Again, every state is a short label, not a clause.

Now extract candidates for the following text, following the same approach.

Text:
{{text}}
""".format(definition=_CANDIDATE_DEFINITION, format_rules=_FORMAT_RULES)


STATE_SPACE_PROMPT_PERMISSIVE_COT = """You are performing the first step of a two-step extraction pipeline: generous candidate generation. A separate, later step will filter your candidates -- do not filter them yourself.

{definition}

{format_rules}

Work through the following steps explicitly before answering:

Step 1 -- List every noun phrase in the text, without exception.
Step 2 -- For each one, note whether the text describes any change, status, action taken on it, or outcome associated with it. If yes, or if you are unsure, keep it as a candidate.
Step 3 -- For each candidate kept, list the conditions it is described as passing through, or could plausibly pass through, based only on the text. Write each condition as a short label following the formatting rules above -- never as a clause or sentence. If you cannot compress a condition into a short label without inventing content, drop it rather than writing a sentence.
Step 4 -- Re-check every candidate name and every state against the formatting rules above. If any of them is a clause, a sentence, or starts with a gerund, rewrite it as a short label or remove it.

Do not attempt to judge whether a candidate is a "real" entity or an actor at this step -- that is done later, outside this task.

After Step 4, output the line "FINAL_ANSWER:" followed immediately by strict JSON only, no prose, no markdown fences:
{{{{"candidate_name": ["state1", "state2", ...], ...}}}}

Text:
{{text}}
""".format(definition=_CANDIDATE_DEFINITION, format_rules=_FORMAT_RULES)