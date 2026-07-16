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