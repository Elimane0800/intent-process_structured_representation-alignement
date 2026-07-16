STATE_SPACE_PROMPT = """You are extracting the finite state space of a process description.

The state space is defined over entities. An entity is an identifiable object or case that persists through the process and progresses through a sequence of distinct conditions over time (its lifecycle). An entity is something that can be said to reach, or fail to reach, a state.

An entity is NOT:
- an attribute or data value carried by an entity (e.g. a score, a date, a count)
- a boolean flag
- a role or actor, unless the text itself treats that actor as the subject undergoing the lifecycle

For each entity, list its finite, mutually exclusive lifecycle states, in the order they typically occur.

Rules:
- Only include entities explicitly or implicitly required by the text.
- Output strict JSON only: {{"entity_name": ["state1", "state2", ...], ...}}
- No prose, no explanation, no markdown fences.

Text:
{text}
"""

STATE_SPACE_PROMPT_FEWSHOT = """You are extracting the finite state space of a process description.

The state space is defined over entities. An entity is an identifiable object or case that persists through the process and progresses through a sequence of distinct conditions over time (its lifecycle). An entity is something that can be said to reach, or fail to reach, a state.

An entity is NOT:
- an attribute or data value carried by an entity (e.g. a score, a date, a count)
- a boolean flag
- a role or actor, unless the text itself treats that actor as the subject undergoing the lifecycle

The states of one entity must be mutually exclusive: at any point in time, the entity is in exactly one of them. If two conditions can hold at the same time, they are not two states of the same entity.

Rules:
- Only include entities explicitly or implicitly required by the text.
- Output strict JSON only: {{"entity_name": ["state1", "state2", ...], ...}}
- No prose, no explanation, no markdown fences.

Example:
Text: "A library manages book loans. When a member borrows a book, it becomes checked out. If it is returned on time, it becomes available again. If returned late, it is marked overdue before returning to available. If a book is lost, it becomes lost and is removed from the catalog. Members can reserve a book that is checked out; once returned, a reserved book becomes held for the requesting member before being checked out again."

Output: {{"book": ["available", "checked_out", "overdue", "held", "lost"]}}

Note: "member" is not an entity here — it is an actor that triggers transitions but does not itself progress through a lifecycle in this text. The five states of "book" are mutually exclusive: a book cannot be both checked_out and available at once.

Now extract the state space for the following text.

Text:
{text}
"""

STATE_SPACE_PROMPT_TWOSHOT = """You are extracting the finite state space of a process description.

The state space is defined over entities. An entity is an identifiable object or case that persists through the process and progresses through a sequence of distinct conditions over time (its lifecycle). An entity is something that can be said to reach, or fail to reach, a state.

An entity is NOT:
- an attribute or data value carried by an entity (e.g. a score, a date, a count)
- a boolean flag
- a role or actor, unless the text itself treats that actor as the subject undergoing the lifecycle

The states of one entity must be mutually exclusive: at any point in time, the entity is in exactly one of them. If two conditions can hold at the same time, they are not two states of the same entity.

A text may require one entity or several distinct entities, each with its own lifecycle. Do not force everything into a single entity, and do not split a single lifecycle into unrelated fragments.

Rules:
- Only include entities explicitly or implicitly required by the text.
- Output strict JSON only: {{"entity_name": ["state1", "state2", ...], ...}}
- No prose, no explanation, no markdown fences.

Example 1 (single entity):
Text: "A library manages book loans. When a member borrows a book, it becomes checked out. If it is returned on time, it becomes available again. If returned late, it is marked overdue before returning to available. If a book is lost, it becomes lost and is removed from the catalog. Members can reserve a book that is checked out; once returned, a reserved book becomes held for the requesting member before being checked out again."

Output: {{"book": ["available", "checked_out", "overdue", "held", "lost"]}}

Note: "member" is not an entity here — it is an actor that triggers transitions but does not itself progress through a lifecycle in this text.

Example 2 (two distinct entities):
Text: "An online store processes customer orders. When a customer places an order, it becomes pending. The store confirms the order, making it confirmed. Once confirmed, a shipment is created for the order and starts as preparing. The shipment becomes shipped once it leaves the warehouse, and delivered once the customer receives it. If a shipment is lost in transit, it becomes lost. If the customer cancels before shipping, the order becomes cancelled and no shipment is created."

Output: {{"order": ["pending", "confirmed", "cancelled"], "shipment": ["preparing", "shipped", "delivered", "lost"]}}

Note: "customer" and "store" are actors, not entities. "order" and "shipment" are two separate entities because they have distinct, independently-progressing lifecycles — a shipment does not exist until the order reaches "confirmed", and an order can end in "cancelled" without ever producing a shipment.

Now extract the state space for the following text.

Text:
{text}
"""

STATE_SPACE_PROMPT_COT = """You are extracting the finite state space of a process description.

The state space is defined over entities. An entity is an identifiable object or case that persists through the process and progresses through a sequence of distinct conditions over time (its lifecycle). An entity is something that can be said to reach, or fail to reach, a state.

An entity is NOT:
- an attribute or data value carried by an entity (e.g. a score, a date, a count)
- a boolean flag
- a role or actor, unless the text itself treats that actor as the subject undergoing the lifecycle

The states of one entity must be mutually exclusive: at any point in time, the entity is in exactly one of them. If two conditions can hold at the same time, they are not two states of the same entity.

Work through the following steps explicitly before answering:

Step 1 — List every candidate subject or object mentioned in the text (including actors).
Step 2 — For each candidate, decide: entity or actor? An actor triggers transitions but does not itself progress through a lifecycle in this text. Briefly justify each decision.
Step 3 — For each entity kept, list the candidate conditions it can be in.
Step 4 — For each pair of conditions within one entity, check: could both hold at the same time? If yes, they are not separate states — merge or discard until the list is mutually exclusive.
Step 5 — Write the final state space.

After Step 5, output the line "FINAL_ANSWER:" followed immediately by strict JSON only, no prose, no markdown fences:
{{"entity_name": ["state1", "state2", ...], ...}}

Text:
{text}
"""


# --- V2 variants: reinforced, non-circular definition of "entity" vs "actor" ---
# The V1 definition above defines an entity by the very property being tested for
# (progresses through a lifecycle => is an entity), which is circular. These V2
# variants replace the definition block with positive, independent criteria
# (object-hood, grammatical role, a concrete flip test), while keeping every other
# element (examples, CoT steps, output rules) identical to the V1 counterpart, to
# isolate the effect of the definition itself from the other levers already tested.

_ENTITY_DEFINITION_V2 = """An entity is an object: something that is created, submitted, processed, transformed, approved, or otherwise acted upon in the course of the process. Concretely, an object is what a system would store as a record with its own identity (e.g. one row in a table) — a specific application, a specific shipment, a specific leave request — distinct from every other instance of the same kind. An entity progresses through a lifecycle: a sequence of distinct, mutually exclusive conditions that describe what has happened to that object so far.

An actor is NOT an entity. An actor is who (human) or what (system or organization or groups or animals) performs actions on objects, or who decides, approves, sends, or receives them. Grammatically, an actor is typically the recurring subject of action verbs (it confirms, it rates, it sends); an object is typically what those verbs act on (the application is confirmed, the shipment is sent). A useful test: if the text only ever says "[X] does something", and never says "[X]'s condition becomes...", then [X] is an actor, not an entity — even if [X] is mentioned often and seems important.

An entity is also NOT:
- an attribute or data value carried by an object (e.g. a score, a date, a count)
- a boolean flag
- a category, role, or organization type, unless the text itself tracks a specific instance of it through a lifecycle (rare — usually roles are actors, not entities)"""


STATE_SPACE_PROMPT_V2 = """You are extracting the finite state space of a process description.

The state space is defined over entities.

{definition}

For each entity, list its finite, mutually exclusive lifecycle states, in the order they typically occur.

Rules:
- Only include entities explicitly or implicitly required by the text.
- Output strict JSON only: {{{{"entity_name": ["state1", "state2", ...], ...}}}}
- No prose, no explanation, no markdown fences.

Text:
{{text}}
""".format(definition=_ENTITY_DEFINITION_V2)

STATE_SPACE_PROMPT_FEWSHOT_V2 = """You are extracting the finite state space of a process description.

The state space is defined over entities.

{definition}

The states of one entity must be mutually exclusive: at any point in time, the entity is in exactly one of them. If two conditions can hold at the same time, they are not two states of the same entity.

Rules:
- Only include entities explicitly or implicitly required by the text.
- Output strict JSON only: {{{{"entity_name": ["state1", "state2", ...], ...}}}}
- No prose, no explanation, no markdown fences.

Example:
Text: "A library manages book loans. When a member borrows a book, it becomes checked out. If it is returned on time, it becomes available again. If returned late, it is marked overdue before returning to available. If a book is lost, it becomes lost and is removed from the catalog. Members can reserve a book that is checked out; once returned, a reserved book becomes held for the requesting member before being checked out again."

Output: {{{{"book": ["available", "checked_out", "overdue", "held", "lost"]}}}}

Note: "member" is not an entity here — it is an actor that triggers transitions but does not itself progress through a lifecycle in this text. The five states of "book" are mutually exclusive: a book cannot be both checked_out and available at once.

Now extract the state space for the following text.

Text:
{{text}}
""".format(definition=_ENTITY_DEFINITION_V2)

STATE_SPACE_PROMPT_TWOSHOT_V2 = """You are extracting the finite state space of a process description.

The state space is defined over entities.

{definition}

The states of one entity must be mutually exclusive: at any point in time, the entity is in exactly one of them. If two conditions can hold at the same time, they are not two states of the same entity.

A text may require one entity or several distinct entities, each with its own lifecycle. Do not force everything into a single entity, and do not split a single lifecycle into unrelated fragments.

Rules:
- Only include entities explicitly or implicitly required by the text.
- Output strict JSON only: {{{{"entity_name": ["state1", "state2", ...], ...}}}}
- No prose, no explanation, no markdown fences.

Example 1 (single entity):
Text: "A library manages book loans. When a member borrows a book, it becomes checked out. If it is returned on time, it becomes available again. If returned late, it is marked overdue before returning to available. If a book is lost, it becomes lost and is removed from the catalog. Members can reserve a book that is checked out; once returned, a reserved book becomes held for the requesting member before being checked out again."

Output: {{{{"book": ["available", "checked_out", "overdue", "held", "lost"]}}}}

Note: "member" is not an entity here — it is an actor that triggers transitions but does not itself progress through a lifecycle in this text.

Example 2 (two distinct entities):
Text: "An online store processes customer orders. When a customer places an order, it becomes pending. The store confirms the order, making it confirmed. Once confirmed, a shipment is created for the order and starts as preparing. The shipment becomes shipped once it leaves the warehouse, and delivered once the customer receives it. If a shipment is lost in transit, it becomes lost. If the customer cancels before shipping, the order becomes cancelled and no shipment is created."

Output: {{{{"order": ["pending", "confirmed", "cancelled"], "shipment": ["preparing", "shipped", "delivered", "lost"]}}}}

Note: "customer" and "store" are actors, not entities. "order" and "shipment" are two separate entities because they have distinct, independently-progressing lifecycles — a shipment does not exist until the order reaches "confirmed", and an order can end in "cancelled" without ever producing a shipment.

Now extract the state space for the following text.

Text:
{{text}}
""".format(definition=_ENTITY_DEFINITION_V2)

STATE_SPACE_PROMPT_COT_V2 = """You are extracting the finite state space of a process description.

The state space is defined over entities.

{definition}

The states of one entity must be mutually exclusive: at any point in time, the entity is in exactly one of them. If two conditions can hold at the same time, they are not two states of the same entity.

Work through the following steps explicitly before answering:

Step 1 — List every candidate subject or object mentioned in the text (including actors).
Step 2 — For each candidate, decide: entity or actor? Apply the flip test above. Briefly justify each decision.
Step 3 — For each entity kept, list the candidate conditions it can be in.
Step 4 — For each pair of conditions within one entity, check: could both hold at the same time? If yes, they are not separate states — merge or discard until the list is mutually exclusive.
Step 5 — Write the final state space.

After Step 5, output the line "FINAL_ANSWER:" followed immediately by strict JSON only, no prose, no markdown fences:
{{{{"entity_name": ["state1", "state2", ...], ...}}}}

Text:
{{text}}
""".format(definition=_ENTITY_DEFINITION_V2)