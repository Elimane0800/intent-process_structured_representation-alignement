REFORMULATION_PROMPT = """You are reformulating a process description to make it clearer, without changing its content.

Rules:
- Do not invent steps, rules, actors, or entities that are not stated or clearly implied in the original text.
- Do not resolve genuine ambiguities by guessing — keep them as stated.
- Make explicit which objects or cases persist and progress through distinct conditions over time, and what causes a transition from one condition to another.
- Preserve every constraint, exception, and condition from the original text.
- Output plain text only. No JSON, no markdown, no headings.

Text:
{text}
"""

REFORMULATION_PROMPT_ONESHOT = """You are reformulating a process description to make it clearer, without changing its content.

Rules:
- Do not invent steps, rules, actors, or entities that are not stated or clearly implied in the original text.
- Do not resolve genuine ambiguities by guessing. If the text leaves something unspecified, say so explicitly (e.g. "the text does not specify whether...") instead of silently picking an interpretation.
- Do not add numeric thresholds, timeframes, or conditions that are not in the original text.
- Make explicit which objects or cases persist and progress through distinct conditions over time, and what causes a transition from one condition to another.
- Preserve every constraint, exception, and condition from the original text.
- Keep the output comparable in length and level of detail to the original — do not compress it into a short summary, and do not pad it with restated content.
- Output plain text only. No JSON, no markdown, no headings.

Example:
Text: "A car rental company lets customers book a car online. Once the customer picks up the car, it needs to be returned by the agreed date. If the car isn't returned in time, a late fee applies. Sometimes a car needs to go to maintenance if there's damage, and it can't be booked again until that's resolved."

Reformulation: "The process involves two entities: the booking and the car. A booking is created when a customer books a car online. When the customer picks up the car, the car becomes rented. The car is expected to be returned by an agreed date. If it is returned after that date, a late fee applies; the text does not specify whether this changes the booking's condition or is only a separate charge. If damage is found, the car enters maintenance, and it cannot be booked again until the maintenance is resolved, at which point it becomes available again. The text does not specify what happens to the booking while the car is in maintenance, nor whether returning before the agreed date is handled differently from returning exactly on it."

Now reformulate the following text, following the same approach: explicit entities and transitions, ambiguities flagged rather than resolved, nothing invented.

Text:
{text}
"""


REFORMULATION_PROMPT_V2 = """You are reformulating a process description to make every activity explicit and easy to understand.

This step performs reformulation only. Do not extract a state space and do not create a state graph.

Definitions:
- An activity is a unit of work in the process. Write it as a clear verb phrase, such as "submit the application".
- A state is a short past-participle label that describes the condition of the process or an object, such as "submitted", "reviewed", or "accepted". A state is not an activity and must not be written as a sentence.
- An explicit state is directly stated in the source text.
- An implicit state is not directly named, but is logically required before an activity can start or logically reached after it completes. Implicit states must be made explicit in the reformulation.
- A precondition is a source state that must be reached before an activity can start. An activity can have multiple preconditions.
- A milestone is a target state reached when an activity is completed successfully. An activity can produce multiple milestones.

Reformulation rules:
- Preserve all activities, conditions, branches, exceptions, deadlines, loops, and ordering constraints from the original text.
- Include every explicit state and every logically necessary implicit state associated with the activities.
- Infer necessary implicit preconditions and milestones, but do not add states that are only plausible business assumptions.
- Do not invent activities or business rules.
- Do not convert examples, definitions, or background information into activities.
- Do not infer an execution order only because items appear in a list.
- Identify each activity once and number the activities A1, A2, A3, and so on, following their order in the source text.
- Every activity must have at least one precondition and at least one milestone.
- For every activity, explicitly provide all its explicit and implicit preconditions and milestones.
- When the source does not directly state a precondition, infer the minimal state logically required immediately before the activity. For the first activity, use "initialized" only when no more meaningful initial state can be inferred.
- When the source does not directly state a milestone, derive the direct completion state from the activity, such as "submitted" after "submit application" or "reviewed" after "review application".
- Write every precondition and milestone as a short lowercase past-participle state label, not as a sentence. Examples: "submitted", "reviewed", "accepted", "selected", and "information_collected".
- Prefer a single-word state label. When more than one word is necessary, use lowercase snake_case, such as "information_collected" or "notification_sent".
- List every source-supported precondition and milestone; do not force an activity to have only one of either.
- When several preconditions are all required, list all of them. When they are alternatives, explicitly separate them with "OR".
- Every milestone must be a direct result of completing its activity. If no detailed result is given, use a faithful completion state such as "collected" for a collection activity.
- Reuse exactly the same state label when a milestone of one activity is a precondition of another activity.
- When the source says that one activity occurs after another, use the preceding activity's milestone as a precondition of the following activity.
- Treat preconditions as the source states, activities as the transitions, and milestones as the target states for the later state graph.
- Preserve alternatives and possible parallel activities without forcing them into a sequence.
- If the source is ambiguous, preserve the ambiguity and state it clearly.
- Write in the same language as the source text.
- Do not output JSON, markdown fences, a state inventory, graph nodes, graph edges, or transitions.

Return only the reformulated process in this format:

Activity A1: <clear activity>
Preconditions: [<one or more explicit or implicit state labels>]
Milestones: [<one or more resulting state labels>]

Activity A2: <clear activity>
Preconditions: [<one or more explicit or implicit state labels>]
Milestones: [<one or more resulting state labels>]

Continue the same structure for every activity. Add an "Uncertainties" section at the end only when the source contains relevant ambiguities.

Text:
{text}
"""
