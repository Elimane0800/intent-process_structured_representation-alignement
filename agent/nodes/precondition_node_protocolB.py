import json
import os
import re
from agent.models.base_llm import get_llm
from agent.nodes.precondition_node_protocolA import (
    generate_candidates, _validate_quote, build_graph, detect_cycles, load_state_spaces,
)
from agent.prompt.precondition_prompt_protocolB import START_END_PROMPT, CLASSIFICATION_PROMPT_B

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "precondition_extraction_protocolB")


def _parse_json_response(response_text: str) -> dict:
    content = re.sub(r"^```(?:json)?|```$", "", response_text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response_text!r}") from e


# --- Stage 0: identify start/end states per entity (LLM, bounded classification) ---

def identify_start_end(text: str, state_space: dict, llm) -> dict:
    prompt = START_END_PROMPT.format(text=text, state_space=json.dumps(state_space))
    response = llm.invoke(prompt).content
    return _parse_json_response(response)


# --- Stage 1: batched classification, with incremental context (within-state and cross-state) ---

def classify_candidates_batched(
    text: str, target_state: str, candidates: list[str], llm,
    established_facts: dict, batch_size: int = 30,
) -> dict:
    """Splits candidates into batches. Each batch call sees: established facts from other,
    already-resolved states (cross-state context), and candidates already validated true in
    earlier batches of THIS state (within-state coherence). Returns {candidate: {"valid","quote"}}
    for all candidates across all batches."""
    validated = {}
    already_validated_this_state = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        prompt = CLASSIFICATION_PROMPT_B.format(
            text=text,
            target_state=target_state,
            candidates=json.dumps(batch),
            established_facts=json.dumps(established_facts),
            already_validated=json.dumps(already_validated_this_state),
        )
        try:
            response = llm.invoke(prompt).content
            result = _parse_json_response(response)
        except Exception as e:
            for candidate in batch:
                validated[candidate] = {"valid": False, "quote": None, "error": str(e)}
            continue

        for candidate in batch:
            verdict = result.get(candidate, {"valid": False, "quote": None})
            is_valid = bool(verdict.get("valid")) and _validate_quote(verdict.get("quote"), text)
            validated[candidate] = {"valid": is_valid, "quote": verdict.get("quote") if is_valid else None}
            if is_valid:
                already_validated_this_state.append(candidate)

    return validated


# --- Stage 2: coherence solver, including start/end consistency ---

def check_start_end_consistency(valid_preconditions: dict, start_end: dict) -> list[str]:
    issues = []
    for target, valid_list in valid_preconditions.items():
        entity, state = target.split(".", 1)
        if state in start_end.get(entity, {}).get("start", []) and valid_list:
            issues.append(f"CONTRADICTION: '{target}' declared START but has valid precondition(s) {valid_list}")

    for target, valid_list in valid_preconditions.items():
        target_entity = target.split(".")[0]
        for conjunction in valid_list:
            for term in conjunction.split(" AND "):
                term_entity, term_state = term.split(".", 1)
                if term_entity == target_entity and term_state in start_end.get(term_entity, {}).get("end", []):
                    issues.append(f"CONTRADICTION: '{term}' declared END but used as precondition for '{target}' (same entity)")
    return issues


def run_protocol_b(text: str, state_space: dict, llm, batch_size: int = 30) -> dict:
    start_end = identify_start_end(text, state_space, llm)

    established_facts = {}
    classification = {}
    valid_preconditions = {}
    candidates_generated = {}

    for entity, states in state_space.items():
        for state in states:
            target = f"{entity}.{state}"

            if state in start_end.get(entity, {}).get("start", []):
                valid_preconditions[target] = []
                established_facts[target] = "INITIAL"
                candidates_generated[target] = 0
                continue

            candidates = generate_candidates(state_space, entity, state)
            candidates_generated[target] = len(candidates)
            result = classify_candidates_batched(text, target, candidates, llm, established_facts, batch_size)
            classification[target] = result

            valid_list = [c for c, v in result.items() if v.get("valid")]
            valid_preconditions[target] = valid_list
            established_facts[target] = valid_list if valid_list else "UNRESOLVED"

    cycles = detect_cycles(build_graph(valid_preconditions))
    issues = check_start_end_consistency(valid_preconditions, start_end)

    return {
        "start_end": start_end,
        "candidates_generated": candidates_generated,
        "classification": classification,
        "valid_preconditions": valid_preconditions,
        "cycles": cycles,
        "consistency_issues": issues,
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
            output = run_protocol_b(text, state_space, llm, batch_size=30)
            results[model_name][case_name] = output
            print(f"\n[{model_name}][{case_name}]")
            for target, valid_list in output["valid_preconditions"].items():
                n = output["candidates_generated"][target]
                status = "INITIAL" if n == 0 else (valid_list if valid_list else "UNRESOLVED")
                print(f"  {target:35s} ({n} candidates) -> {status}")
            if output["cycles"]:
                print(f"  CYCLES: {output['cycles']}")
            if output["consistency_issues"]:
                print(f"  ISSUES: {output['consistency_issues']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")