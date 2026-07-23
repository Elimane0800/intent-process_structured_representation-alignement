import difflib
import itertools
import json
import os
import re
from agent.models.base_llm import get_llm
from agent.prompt.precondition_prompt_protocolA import CLASSIFICATION_PROMPT_A

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "precondition_extraction_protocolA")
PROTOCOL3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_space_protocol3")


def load_state_spaces(prompt_name: str = "one_shot", run_filename: str | None = None) -> dict:
    """Loads the 'kept' state space per case from a Protocol 3 run. Pass run_filename explicitly
    to pin a fixed reference U; otherwise falls back to the latest run available."""
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


# --- Stage 0: mechanical combinatorial candidate generation for EVERY state (pure Python, no LLM) ---
# No start/end identification step in Protocol A: every state, including natural starting states,
# goes through the same candidate generation and classification process.

def generate_candidates(state_space: dict, target_entity: str, target_state: str) -> list[str]:
    """For a target state, generate candidate preconditions: for each entity, either 'none' or
    exactly one of its states (excluding the target state itself if same entity). This respects
    mutual exclusivity by construction -- no two states of the same entity ever co-occur in one
    candidate. No cap on conjunction size."""
    entity_choices = []
    for entity, states in state_space.items():
        options = [f"{entity}.{s}" for s in states if not (entity == target_entity and s == target_state)]
        entity_choices.append([None] + options)

    candidates = []
    for combo in itertools.product(*entity_choices):
        terms = [t for t in combo if t is not None]
        if terms:
            candidates.append(" AND ".join(terms))
    return candidates


def generate_all_candidates(state_space: dict) -> dict:
    """Generates candidates for every state of every entity, upfront, before any LLM call."""
    all_candidates = {}
    for entity, states in state_space.items():
        for state in states:
            target = f"{entity}.{state}"
            all_candidates[target] = generate_candidates(state_space, entity, state)
    return all_candidates


# --- Stage 1: LLM classifies EACH candidate independently, with grounding ---

def _validate_quote(quote: str, text: str, threshold: float = 0.9) -> bool:
    """Same grounding logic as validate_grounding() used for U: exact substring match first,
    else fuzzy longest-common-match ratio."""
    if not quote:
        return False
    text_norm = " ".join(text.split())
    quote_norm = " ".join(quote.split())
    if quote_norm in text_norm:
        return True
    ratio = difflib.SequenceMatcher(None, quote_norm, text_norm).find_longest_match().size / max(len(quote_norm), 1)
    return ratio >= threshold


def classify_candidates(text: str, target_state: str, candidates: list[str], llm) -> dict:
    """Classifies every candidate independently as valid/invalid. Returns
    {candidate: {"valid": bool, "quote": str|None}}. A candidate marked valid by the LLM but
    whose quote does not actually match the text is downgraded to invalid mechanically."""
    if not candidates:
        return {}
    prompt = CLASSIFICATION_PROMPT_A.format(
        text=text, target_state=target_state, candidates=json.dumps(candidates)
    )
    response = llm.invoke(prompt).content.strip()
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e

    validated = {}
    for candidate in candidates:
        verdict = result.get(candidate, {"valid": False, "quote": None})
        is_valid = bool(verdict.get("valid")) and _validate_quote(verdict.get("quote"), text)
        validated[candidate] = {"valid": is_valid, "quote": verdict.get("quote") if is_valid else None}
    return validated


# --- Stage 2: deterministic coherence solver (pure Python, no LLM) ---
# Protocol A has no start/end identification step, so the only structural check available at
# this stage is cycle detection over the accepted precondition graph.

def build_graph(valid_preconditions: dict) -> dict:
    """valid_preconditions: {target: [list of valid candidate conjunction strings]}.
    Returns adjacency: precondition term -> [targets]."""
    edges = {}
    for target, conjunctions in valid_preconditions.items():
        for conjunction in conjunctions:
            for term in conjunction.split(" AND "):
                edges.setdefault(term, []).append(target)
    return edges


def detect_cycles(edges: dict) -> list[list[str]]:
    """DFS-based cycle detection over the precondition graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    cycles = []

    def dfs(node, path):
        color[node] = GRAY
        path.append(node)
        for neighbor in edges.get(node, []):
            if color.get(neighbor, WHITE) == WHITE:
                dfs(neighbor, path)
            elif color.get(neighbor) == GRAY:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
        path.pop()
        color[node] = BLACK

    for node in list(edges.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node, [])
    return cycles


def run_protocol_a(text: str, state_space: dict, llm) -> dict:
    all_candidates = generate_all_candidates(state_space)

    classification = {}
    valid_preconditions = {}
    for target, candidates in all_candidates.items():
        try:
            result = classify_candidates(text, target, candidates, llm)
        except Exception as e:
            result = {"error": str(e)}
        classification[target] = result
        valid_preconditions[target] = [c for c, v in result.items() if isinstance(v, dict) and v.get("valid")]

    cycles = detect_cycles(build_graph(valid_preconditions))

    return {
        "candidates_generated": {t: len(c) for t, c in all_candidates.items()},
        "classification": classification,
        "valid_preconditions": valid_preconditions,
        "cycles": cycles,
    }


if __name__ == "__main__":
    MODELS = [
        "meta/llama-3.1-8b-instruct",
        #"meta/llama-3.3-70b-instruct",
        #"mistralai/mistral-nemotron",
        #"openai/gpt-oss-20b",
        #"openai/gpt-oss-120b",
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

    results = {}
    for model_name in MODELS:
        llm = get_llm(temperature=0, model=model_name)
        results[model_name] = {}
        for case_name, text in CASES.items():
            state_space = STATE_SPACES.get(case_name, {})
            output = run_protocol_a(text, state_space, llm)
            results[model_name][case_name] = output
            print(f"\n[{model_name}][{case_name}]")
            for target, valid_list in output["valid_preconditions"].items():
                n_candidates = output["candidates_generated"][target]
                print(f"  {target:35s} ({n_candidates} candidates) -> valid: {valid_list}")
            if output["cycles"]:
                print(f"  CYCLES: {output['cycles']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")