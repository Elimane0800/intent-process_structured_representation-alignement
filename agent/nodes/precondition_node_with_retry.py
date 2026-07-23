import json
import os
import re
import difflib
from agent.models.base_llm import get_llm
from agent.prompt.precondition_prompt import (
    PRECONDITION_PROMPT,
    PRECONDITION_PROMPT_FEWSHOT,
    PRECONDITION_PROMPT_TWOSHOT,
    PRECONDITION_PROMPT_COT,
)
from agent.prompt.precondition_prompt_retry import RETRY_PROMPT

# v2 vs v1:
# - extract_preconditions now strips a "FINAL_ANSWER:" prefix before parsing, mirroring
#   generate_candidates() in state_space_node_v3.py -- required for PRECONDITION_PROMPT_COT,
#   which asks the model to reason step by step before answering. Without this, the COT variant
#   fails every parse, since the reasoning text precedes the JSON.
# - _validate_one now recognizes " OR " as a second valid operator alongside " AND ", per the
#   OR support added to precondition_prompt.py. A precondition is still required to be a PURE
#   AND or a PURE OR, never mixed -- a string containing both operators is mechanically rejected
#   (downgraded to UNRESOLVED) rather than guessed at. Mutual exclusivity (two states of the same
#   entity never together) is now scoped to AND only: it is expected and valid for an OR to
#   combine two states of the same entity (e.g. "work_accident.fatal OR work_accident.serious_injury"),
#   since that is exactly what a disjunction over one entity's own outcomes looks like.
# - build_graph() (Step "Cycle detection") now splits on either operator when building the
#   term -> target adjacency, since accepted preconditions can now be pure-OR strings too.

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "precondition_extraction_with_retry")
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


# --- Step 1: single LLM call, proposes a precondition + quote for every state ---

def extract_preconditions(text: str, state_space: dict, llm, prompt_template: str = PRECONDITION_PROMPT) -> dict:
    """Single LLM call: returns {entity.state: {"precondition": ..., "quote": ...}} as proposed
    by the model, unvalidated. Strips a "FINAL_ANSWER:" prefix if present, so COT-style prompts
    (which reason before answering) parse the same way as direct-answer prompts."""
    prompt = prompt_template.format(text=text, state_space=json.dumps(state_space))
    response = llm.invoke(prompt).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e


# --- Step 2: mechanical validation layer, no LLM ---

def _all_valid_targets(state_space: dict) -> set:
    return {f"{entity}.{state}" for entity, states in state_space.items() for state in states}


def _validate_quote(quote: str, text: str, threshold: float = 0.9) -> bool:
    """Grounding check: exact substring match first, else fuzzy longest-common-match ratio."""
    if not quote:
        return False
    text_norm = " ".join(text.split())
    quote_norm = " ".join(quote.split())
    if quote_norm in text_norm:
        return True
    ratio = difflib.SequenceMatcher(None, quote_norm, text_norm).find_longest_match().size / max(len(quote_norm), 1)
    return ratio >= threshold


def _split_precondition(precondition: str) -> tuple[str | None, list[str]]:
    """Splits a precondition into its operator ('AND' or 'OR') and its terms. A precondition
    must be a PURE conjunction or a PURE disjunction -- if both operators appear in the same
    string, this is mixed logic that this task does not support, and the operator returned is
    None as a signal to reject. A single term with no operator is treated as a trivial AND
    (an AND or OR of one term behaves identically for validation purposes)."""
    has_and = " AND " in precondition
    has_or = " OR " in precondition
    if has_and and has_or:
        return None, []
    if has_or:
        return "OR", precondition.split(" OR ")
    return "AND", precondition.split(" AND ")


_MALFORMED_TARGET_PREFIX = "malformed or unknown target key"
 
 
def _validate_one(target: str, precondition: str, quote: str, valid_targets: set, text: str) -> dict:
    """Validates a single (target, precondition, quote) triple against six mechanical rules:
    valid target key, pure operator (no AND/OR mix), reference existence, self-reference, mutual
    exclusivity (AND only), and textual grounding.
 
    Target-key check runs FIRST, before anything else -- a target that is not itself a real
    entity.state of the state space (bare entity name, hallucinated state, or an entire AND/OR
    expression used as a key) makes the rest of the entry meaningless regardless of how good the
    precondition content looks. Fix for a real, observed failure mode: llama-3.1-8b produced
    entries like {"probation_phase": {"precondition": "job_application.confirmed AND
    job_offer.sent", "status": "OK", ...}} -- content that passed every other check, silently
    discarded three nodes downstream by graph_node.py because the key itself was never a valid
    target. Catching it here, with an explicit reason, is strictly better than a silent drop
    later -- even though (see retry_downgraded) this particular failure can't be usefully
    retried, since there is no valid target to retry against."""
    if target not in valid_targets:
        return {
            "precondition": "UNRESOLVED", "quote": None, "status": "DOWNGRADED",
            "reason": f"malformed or unknown target key: {target!r} is not a valid entity.state of the state space",
        }
 
    if precondition in ("INITIAL", "UNRESOLVED"):
        return {"precondition": precondition, "quote": None, "status": "OK", "reason": None}
 
    operator, terms = _split_precondition(precondition)
    reason = None
 
    if operator is None:
        reason = "mixed AND/OR not supported in a single precondition"
    else:
        unknown_terms = [t for t in terms if t not in valid_targets]
        if unknown_terms:
            reason = f"references unknown state(s): {unknown_terms}"
        elif target in terms:
            reason = "self-reference"
        elif operator == "AND":
            entities_seen = [t.split(".")[0] for t in terms]
            if len(entities_seen) != len(set(entities_seen)):
                reason = "mutual exclusivity violation: two states of the same entity in one conjunction"
 
    if reason is None and not _validate_quote(quote, text):
        reason = "quote not found verbatim in text"
 
    if reason:
        return {"precondition": "UNRESOLVED", "quote": None, "status": "DOWNGRADED", "reason": reason}
    return {"precondition": precondition, "quote": quote, "status": "OK", "reason": None}
 


def validate_preconditions(text: str, state_space: dict, raw_preconditions: dict) -> dict:
    """Returns {entity.state: {"precondition": ..., "quote": ..., "status": "OK"|"DOWNGRADED",
    "reason": str|None}}. Invalid preconditions are downgraded to UNRESOLVED with a reason,
    never silently dropped."""
    valid_targets = _all_valid_targets(state_space)
    validated = {}
    for target, entry in raw_preconditions.items():
        precondition = entry.get("precondition") if isinstance(entry, dict) else entry
        quote = entry.get("quote") if isinstance(entry, dict) else None
        validated[target] = _validate_one(target, precondition, quote, valid_targets, text)
    return validated


# --- Step 3: single, narrow retry -- only for states downgraded by validation ---
# Context is kept tight: only the target state, its rejected previous attempt, the exact
# rejection reason, and the list of valid states -- not the full graph, not other states'
# results. One attempt only, never a loop: the result goes back through the exact same
# _validate_one() check as the original answer (now AND/OR-aware), so it can never regress
# below the current UNRESOLVED status.

def retry_downgraded(text: str, state_space: dict, validated: dict, raw: dict, llm) -> dict:
    valid_targets = _all_valid_targets(state_space)
    updated = dict(validated)
 
    for target, entry in validated.items():
        if entry["status"] != "DOWNGRADED":
            continue
 
        # Cas exclu du retry, pas un oubli : si target lui-meme n'est pas un entity.state valide,
        # il n'y a pas de "cette cible precise" a retenter -- le retry demanderait au LLM de
        # refaire une precondition POUR UNE CLE QUI N'A JAMAIS ETE UNE VRAIE CIBLE, ce qui ne
        # peut que reproduire la meme confusion de granularite, jamais la resoudre. Laisse
        # explicitement UNRESOLVED, raison deja posee par _validate_one, jamais silencieux.
        if entry["reason"] and entry["reason"].startswith(_MALFORMED_TARGET_PREFIX):
            continue
 
        previous_entry = raw.get(target, {})
        previous_precondition = previous_entry.get("precondition") if isinstance(previous_entry, dict) else previous_entry
        other_valid_states = sorted(t for t in valid_targets if t != target)
 
        prompt = RETRY_PROMPT.format(
            text=text,
            target_state=target,
            previous_precondition=previous_precondition,
            reason=entry["reason"],
            valid_states=json.dumps(other_valid_states),
        )
        try:
            response = llm.invoke(prompt).content.strip()
            content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
            result = json.loads(content)
        except Exception:
            continue  # single attempt only: keep the existing UNRESOLVED on any failure
 
        new_entry = _validate_one(target, result.get("precondition"), result.get("quote"), valid_targets, text)
        new_entry["retried"] = True
        updated[target] = new_entry
 
    return updated


# --- Cycle detection over the final accepted graph ---

def build_graph(accepted: dict) -> dict:
    """accepted: {target: 'A AND B' or 'A OR B' or similar}. Returns adjacency: precondition
    term -> [targets]. Splits on either operator -- accepted preconditions are always pure
    AND or pure OR by this point (mixed logic never survives validation), so a single regex
    split is safe regardless of which operator was used."""
    edges = {}
    for target, precondition in accepted.items():
        for term in re.split(r" AND | OR ", precondition):
            edges.setdefault(term, []).append(target)
    return edges


def detect_cycles(edges: dict) -> list[list[str]]:
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


def run_naive_with_retry(text: str, state_space: dict, llm, prompt_template: str = PRECONDITION_PROMPT) -> dict:
    raw = extract_preconditions(text, state_space, llm, prompt_template)
    validated = validate_preconditions(text, state_space, raw)
    validated = retry_downgraded(text, state_space, validated, raw, llm)

    accepted = {t: e["precondition"] for t, e in validated.items() if e["precondition"] not in ("INITIAL", "UNRESOLVED")}
    cycles = detect_cycles(build_graph(accepted))

    return {"raw": raw, "validated": validated, "cycles": cycles}


if __name__ == "__main__":
    MODELS = [
        "meta/llama-3.1-8b-instruct",
        #"meta/llama-3.3-70b-instruct",
        #"mistralai/mistral-medium-3.5-128b",
        #"mistralai/mistral-nemotron",
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
        "two_shot": PRECONDITION_PROMPT_TWOSHOT,
        "cot": PRECONDITION_PROMPT_COT,
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
                    output = run_naive_with_retry(text, state_space, llm, prompt_template)
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
                    n_retried = sum(1 for e in output["validated"].values() if e.get("retried"))
                    n_retried_ok = sum(1 for e in output["validated"].values() if e.get("retried") and e["status"] == "OK")
                    print(f"[{prompt_name}][{model_name}] {case_name}: still_downgraded={n_downgraded} retried={n_retried} retried_and_fixed={n_retried_ok} cycles={output['cycles']}")
                    for target, e in output["validated"].items():
                        tag = ""
                        if e.get("retried"):
                            tag = " [RETRIED -> OK]" if e["status"] == "OK" else " [RETRIED -> still UNRESOLVED]"
                        elif e["status"] == "DOWNGRADED":
                            tag = f" [DOWNGRADED: {e['reason']}]"
                        print(f"    {target:35s} <- {e['precondition']}{tag}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")