import json
import os
import re
from agent.models.base_llm import get_llm
from agent.prompt.precondition_prompt import PRECONDITION_PROMPT, PRECONDITION_PROMPT_FEWSHOT
from agent.nodes.precondition_node_protocolA import _validate_quote, build_graph, detect_cycles

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "precondition_extraction")
PROTOCOL3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_space_protocol3")


def load_state_spaces(prompt_name: str = "one_shot", run_filename: str | None = None) -> dict:
    """Loads the 'kept' state space per case from a Protocol 3 run. Pass run_filename explicitly
    (e.g. 'run_17.json') to pin a fixed reference U; otherwise falls back to the latest run
    available, which can vary across executions -- pin it for any real comparison across models."""
    if run_filename is None:
        existing = [f for f in os.listdir(PROTOCOL3_DIR) if re.match(r"run_\d+\.json$", f)]
        run_filename = max(existing, key=lambda f: int(re.match(r"run_(\d+)\.json$", f).group(1)))
    with open(os.path.join(PROTOCOL3_DIR, run_filename)) as f:
        data = json.load(f)
    model_name = next(iter(data[prompt_name]))
    print(f"Loading state spaces from {run_filename} [{prompt_name}][{model_name}]")
    return {
        case_name: entry["state_space"]
        for case_name, entry in data[prompt_name][model_name].items()
    }


def extract_preconditions(text: str, state_space: dict, llm, prompt_template: str = PRECONDITION_PROMPT) -> dict:
    """Single LLM call: returns {entity.state: {"precondition": ..., "quote": ...}} as proposed
    by the model, unvalidated. Validation happens separately in validate_preconditions()."""
    prompt = prompt_template.format(text=text, state_space=json.dumps(state_space))
    response = llm.invoke(prompt).content.strip()
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e


# --- Mechanical validation layer, applied after the single LLM call ---
# Same principle already used and proven on U (Protocol 1) and on the combinatorial protocols
# (A/B): the LLM proposes, a deterministic layer verifies -- reference existence, quote grounding,
# structural rules (self-reference, mutual exclusivity), and cycle detection over the result.

def _all_valid_targets(state_space: dict) -> set:
    return {f"{entity}.{state}" for entity, states in state_space.items() for state in states}


def validate_preconditions(text: str, state_space: dict, raw_preconditions: dict) -> dict:
    """Returns {entity.state: {"precondition": ..., "quote": ..., "status": "OK"|"DOWNGRADED",
    "reason": str|None}}. Invalid preconditions are downgraded to UNRESOLVED with a reason,
    never silently dropped."""
    valid_targets = _all_valid_targets(state_space)
    validated = {}

    for target, entry in raw_preconditions.items():
        precondition = entry.get("precondition") if isinstance(entry, dict) else entry
        quote = entry.get("quote") if isinstance(entry, dict) else None

        if precondition in ("INITIAL", "UNRESOLVED"):
            validated[target] = {"precondition": precondition, "quote": None, "status": "OK", "reason": None}
            continue

        terms = precondition.split(" AND ")
        reason = None

        # reference existence
        unknown_terms = [t for t in terms if t not in valid_targets]
        if unknown_terms:
            reason = f"references unknown state(s): {unknown_terms}"

        # self-reference
        elif target in terms:
            reason = "self-reference"

        # mutual exclusivity: no two terms from the same entity
        else:
            entities_seen = [t.split(".")[0] for t in terms]
            if len(entities_seen) != len(set(entities_seen)):
                reason = "mutual exclusivity violation: two states of the same entity in one conjunction"

        # grounding: quote must exist verbatim (or near-verbatim) in the text
        if reason is None and not _validate_quote(quote, text):
            reason = "quote not found verbatim in text"

        if reason:
            validated[target] = {"precondition": "UNRESOLVED", "quote": None, "status": "DOWNGRADED", "reason": reason}
        else:
            validated[target] = {"precondition": precondition, "quote": quote, "status": "OK", "reason": None}

    return validated


def run_naive_with_validation(text: str, state_space: dict, llm, prompt_template: str = PRECONDITION_PROMPT) -> dict:
    raw = extract_preconditions(text, state_space, llm, prompt_template)
    validated = validate_preconditions(text, state_space, raw)

    accepted = {t: e["precondition"] for t, e in validated.items() if e["precondition"] not in ("INITIAL", "UNRESOLVED")}
    cycles = detect_cycles(build_graph({t: [p] for t, p in accepted.items()}))

    return {"raw": raw, "validated": validated, "cycles": cycles}


if __name__ == "__main__":
    MODELS = [
        #"meta/llama-3.1-8b-instruct",
        #"meta/llama-3.3-70b-instruct",
        #"mistralai/mistral-medium-3.5-128b",
        "mistralai/mistral-nemotron",
        #"openai/gpt-oss-20b",
        #"openai/gpt-oss-120b",
        #"mistralai/mistral-large-3-675b-instruct-2512",
        #"moonshotai/kimi-k2.6",
        #"thinkingmachines/inkling"
    ]

    CASES = {
        "job_application": (
            "You have to regularly report, to which companies you wrote job applications. "
            "Based on your job applications, new potential job offers are sent to you. "
            "Companies have to confirm that they received job applications and rate the application. "
            "A job interview can be negotiated. When a company wants you to work for them, you enter "
            "the probation phase. After probation phase, you can rate the company and the company can "
            "rate you. Reviews for a company can only be seen (by job applicants) after 1 year. If a "
            "job becomes permanent, the process ends, unless you rated the company C or less, then you "
            "continue to receive job offers, but no longer have to report."
        ),
        "maternity_leave": (
            "Create a process that support in planning, taking and extending a maternity leave. "
            "* Fetch information about potential models (months duration, split between parents) "
            "* Let parent select * Collect relevant information * Notify Social Security, Company in "
            "time * Gather information from companies * At the end of the period let parent decide "
            "about extension."
        ),
        "work_accident": (
            "Create a process that helps in gathering information about work accidents (and almost work accidents):\n\n"
            "For insured gainful employment, a work accident is considered to be an accident that occurs in the "
            "following circumstances, by way of example:\n"
            "- the accident occurs in a location, at a time and with a cause that correlates to the insured employment\n"
            "- when working from home\n"
            "- on the direct route from the permanent place of residence to work, to lunch or on the way home, "
            "whereby carpools are also protected\n"
            "- for training sessions that serve to provide specific professional knowledge, whereby an accident on "
            "the way to or from the training centre is considered a work accident\n"
            "- on the direct route from home, or from the workplace or training centre to a doctor and back, if "
            "this interrupts the direct route from the permanent place of residence to work or the route home "
            "(the doctor's visit must be notified to the employer in advance)\n"
            "- on the way to or from the workplace or training centre with the purpose of taking a child to or "
            "collecting them from a childcare facility, a daycare facility, external care or a school, insofar as "
            "they have a supervisory responsibility for the child\n"
            "- when making use of advocacy groups or professional associations (e.g. Chamber of Labour, trade "
            "union federation, guild etc.)\n\n"
            "For kindergarten children, schoolchildren and students, an accident is considered to be a work "
            "accident if, amongst other things:\n"
            "- the accident occurs in a location, at a time and with a cause that correlates to the school or "
            "university education or the compulsory kindergarten year that forms the basis of the insurance\n"
            "- when taking part in a school event or school-related event\n"
            "- on the direct route from the child's place of residence or permanent accommodation to kindergarten "
            "or a school visit or on the way home\n"
            "- when performing a practical task prescribed as part of the curriculum and/or the study regulations\n"
            "- in certain professional (training) orientations\n\n"
            "In an agricultural or forestry establishment, accidents can be considered work accidents if they do "
            "not occur directly while performing the insured employment, including working in the household of "
            "the owner, providing accommodation for guests, secondary occupations for selling agricultural "
            "products, building work for the establishment, and neighbourhood assistance for another "
            "establishment.\n\n"
            "Certain accidents are considered the same as work accidents outside of professional activity, "
            "including activities under unemployment/labour market support services, assisting with an accident "
            "or donating blood, and training or serving as a member of a volunteer aid organisation (fire "
            "brigade, water rescue, Red Cross, Alpine Rescue, air rescue, etc.).\n\n"
            "The employees must inform the employer immediately of: any work accident, any incident that would "
            "almost have led to an accident, any serious and immediate risk to safety and health that they "
            "discover, any defect discovered in protection systems.\n\n"
            "The employer must immediately report any work accident that leads to a fatality or serious injury "
            "to the Labour Inspectorate, if it has not been reported to the emergency services.\n\n"
            "Independent of this, every work accident in which a person with accident insurance has been killed, "
            "or injured in such a way that they are unable to work for three days, in full or in part, must be "
            "reported to the responsible accident insurance provider within five days.\n\n"
            "As a self-employed person, accidents involving people you employ and those that involve you must "
            "be reported to the accident insurance provider in good time.\n\n"
            "Students and schoolchildren should report accidents to the competent directorate. Schools, "
            "educational facilities and universities are obliged to report every work accident through which a "
            "person with accident insurance is physically injured or killed within a maximum of five days to the "
            "responsible accident insurance provider in triplicate.\n\n"
            "In the case of private insurance, the insured party must report an accident in writing immediately "
            "in line with the conditions and to avoid the provider from being released from the obligation to "
            "perform. If there are fatalities, this must be reported within three days, even if the accident has "
            "already been reported."
        ),
    }

    STATE_SPACES = load_state_spaces(run_filename="run_17.json")

    PROMPTS = {
        "zero_shot": PRECONDITION_PROMPT,
        "one_shot": PRECONDITION_PROMPT_FEWSHOT,
    }

    results = {}
    for prompt_name, prompt_template in PROMPTS.items():
        results[prompt_name] = {}
        for model_name in MODELS:
            llm = get_llm(temperature=0, model=model_name)
            results[prompt_name][model_name] = {}
            for case_name, text in CASES.items():
                state_space = STATE_SPACES.get(case_name, {})
                try:
                    output = run_naive_with_validation(text, state_space, llm, prompt_template)
                except Exception as e:
                    output = {"error": str(e)}
                results[prompt_name][model_name][case_name] = {
                    "state_space_used": state_space,
                    **output,
                }
                if "error" in output:
                    print(f"[{prompt_name}][{model_name}] {case_name}: ERROR - {output['error'][:150]}")
                else:
                    n_downgraded = sum(1 for e in output["validated"].values() if e["status"] == "DOWNGRADED")
                    print(f"[{prompt_name}][{model_name}] {case_name}: {n_downgraded} downgraded by validation, cycles={output['cycles']}")
                    for target, e in output["validated"].items():
                        tag = f" [DOWNGRADED: {e['reason']}]" if e["status"] == "DOWNGRADED" else ""
                        print(f"    {target:35s} <- {e['precondition']}{tag}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")