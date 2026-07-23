# Requires: pip install nltk --break-system-packages
#           python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
#
# Protocol 3 vs Protocol 2: the validation layer no longer relies on a hand-picked, open-ended
# set of hypernym synsets (ACTOR_HYPERNYMS/ENTITY_HYPERNYMS in v2), nor on a dependency-parsing
# ratio that degrades on low-frequency candidates (f_srl in v2). Instead it uses WordNet's own
# lexnames: a FIXED, CLOSED taxonomy of 25 noun categories that already covers all of WordNet.
# The actor/entity mapping is chosen once, over these 25 fixed categories, and never needs to
# grow as new domains/test cases are added -- unlike a hypernym synset list, which does.

import json
import os
import re
from nltk.corpus import wordnet as wn
from agent.models.base_llm import get_llm
from agent.prompt.state_space_prompt_permissive import (
    STATE_SPACE_PROMPT_PERMISSIVE,
    STATE_SPACE_PROMPT_PERMISSIVE_FEWSHOT,
    STATE_SPACE_PROMPT_PERMISSIVE_TWOSHOT,
    STATE_SPACE_PROMPT_PERMISSIVE_COT,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_space_protocol3")

# Closed set: chosen once over WordNet's fixed 25 noun.* lexnames, not over its open synset space.
ACTOR_LEXNAMES = {"noun.person", "noun.group"}
ENTITY_LEXNAMES = {"noun.artifact", "noun.communication", "noun.act", "noun.event", "noun.cognition"}

# Closed class in English (pronouns), independent of domain -- unlike a hypernym list, it does
# not grow when moving to a new test case or business domain.
STOP_CANDIDATES = {"you", "it", "they", "he", "she", "we", "i"}


# --- Step A: permissive candidate generation (LLM) -- unchanged from Protocol 2 ---

def generate_candidates(text: str, llm, prompt_template: str = STATE_SPACE_PROMPT_PERMISSIVE) -> dict:
    response = llm.invoke(prompt_template.format(text=text)).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e


# --- Step B: deterministic lexical validation (no LLM, no training, no growing list) ---

def _head_word(candidate_name: str) -> str:
    """Compound nouns are headed by their rightmost word in English (e.g. 'job_application' -> 'application').
    No separate lemmatization step needed: WordNet's own morphy backoff handles plural/inflected forms."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", candidate_name).replace("_", " ").split()
    return words[-1].lower() if words else candidate_name.lower()


def lexname_signal(head: str) -> str:
    """Classify the candidate's dominant noun sense by WordNet lexname (fixed 25-category taxonomy).
    Returns 'ACTOR', 'ENTITY', or 'INDETERMINATE' (lexname exists but maps to neither camp, or word
    not found in WordNet at all)."""
    synsets = wn.synsets(head, pos=wn.NOUN)
    if not synsets:
        return "INDETERMINATE"
    lexname = synsets[0].lexname()
    if lexname in ACTOR_LEXNAMES:
        return "ACTOR"
    if lexname in ENTITY_LEXNAMES:
        return "ENTITY"
    return "INDETERMINATE"


def validate_candidates(candidates: dict) -> dict:
    """Reject only if the candidate's dominant sense is lexically ACTOR-like. Default to keeping
    otherwise (ENTITY or INDETERMINATE) -- recall matters more than precision at this stage."""
    report = {}
    for name in candidates:
        head = _head_word(name)
        if head in STOP_CANDIDATES:
            report[name] = {"f_lex": None, "kept": False, "reason": "stopword"}
            continue
        f_lex = lexname_signal(head)
        report[name] = {"f_lex": f_lex, "kept": f_lex != "ACTOR"}
    return report


def filter_state_space(candidates: dict) -> tuple[dict, dict]:
    report = validate_candidates(candidates)
    filtered = {name: states for name, states in candidates.items() if report[name]["kept"]}
    return filtered, report


# --- Step C: deterministic referential deduplication (no LLM, no lists, no thresholds) ---
#
# Added after the qualitative inspection of the 61 VIOLATED cases from the dataset run
# (violated_cases_dump.md): U sometimes contains two entities that are two SURFACE FORMS of the
# same referent (e.g. 'job_applications'/'job_application', '3d_model'/'model',
# 'work_accident'/'accident'). The matching then anchors activities on one form while the
# alignment verifies the other -- producing mechanical VIOLATED/UNRESOLVABLE that measure a
# vocabulary split, not a process property. Scope, deliberately narrow: this step merges
# REFERENTIAL duplicates only (same thing, different surface forms), never semantically related
# but distinct entities ('car' vs 'car_service' have different head words and are NOT merged --
# a service is not a car; residual artifacts on such pairs are handled downstream by the
# alignment's confidence demotion, not here).
#
# Two rules, both purely lexical/structural, in the same spirit as the lexname layer (closed
# resources, no list that grows with new domains, no tuned threshold):
#
# Rule 1 -- MORPHOLOGICAL IDENTITY: two names whose word-by-word singularized forms are
# identical are the same referent ('job_applications' == 'job_application'). Uses WordNet's
# morphological analyzer; note that wn.morphy() alone is insufficient here because some plurals
# are themselves WordNet lemmas (e.g. 'parts' is a lemma meaning "region", so morphy returns it
# unchanged) -- we therefore take the SHORTEST analysis among all forms wn._morphy() returns,
# which yields the true singular deterministically.
#
# Rule 2 -- ANAPHORIC CONTAINMENT: a name whose (normalized) word set is a proper subset of
# exactly ONE other name's, with the SAME head word, is an anaphoric mention of it ('model' is
# "the model" referring back to '3d_model'; 'accident' back to 'work_accident'). The general
# form is absorbed into the specific one (which keeps the maximum information). Uniqueness
# guard: if the general form is head-shared-contained in SEVERAL specifics (e.g. 'account'
# under both 'bank_account' and 'battle_net_account'), the anaphora is ambiguous -- nothing is
# merged, the case is only reported. Different heads never merge ('car_service' head 'service'
# vs 'car' head 'car').
#
# Singletons whose surface form is merely plural ('parts' alone in U) are NOT renamed -- that
# would be vocabulary normalization, a different intervention than deduplication, out of scope
# here by decision.

def _singularize(word: str) -> str:
    forms = wn._morphy(word, wn.NOUN)
    return min(forms, key=len) if forms else word


def _normalized_words(candidate_name: str) -> list[str]:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", candidate_name).replace("_", " ").split()
    return [_singularize(w.lower()) for w in words]


def _merge_states(base: list[str], extra: list[str]) -> list[str]:
    """Union preserving order of first appearance, exact dedup -- consistent with mutual
    exclusivity: one entity, one list of states."""
    merged = list(base)
    for s in extra:
        if s not in merged:
            merged.append(s)
    return merged


def deduplicate_state_space(state_space: dict) -> tuple[dict, dict]:
    """Returns (deduplicated U, report). Report = {"merged": {kept_name: {"absorbed": [...],
    "rule": ..., "states": [...]}}, "ambiguous_not_merged": {general: [specifics]}} -- every
    decision is reported, nothing merged or skipped silently."""
    report = {"merged": {}, "ambiguous_not_merged": {}}

    # Rule 1 -- group by fully normalized name, first-seen order preserved.
    groups: dict[str, list[str]] = {}
    for name in state_space:
        groups.setdefault("_".join(_normalized_words(name)), []).append(name)

    deduped: dict[str, list[str]] = {}
    for normalized, members in groups.items():
        if len(members) == 1:
            deduped[members[0]] = list(state_space[members[0]])
            continue
        states: list[str] = []
        for m in members:
            states = _merge_states(states, state_space[m])
        deduped[normalized] = states
        report["merged"][normalized] = {
            "absorbed": [m for m in members if m != normalized],
            "rule": "morphological_identity",
            "states": states,
        }

    # Rule 2 -- absorb anaphoric generals into their unique specific, iterating until stable
    # (handles chains like model -> 3d_model -> printed_3d_model in successive passes; bounded
    # by the number of entities, no unbounded loop possible).
    changed = True
    while changed:
        changed = False
        for general in list(deduped):
            g_words = _normalized_words(general)
            g_set, g_head = set(g_words), g_words[-1]
            specifics = [
                other for other in deduped
                if other != general
                and g_set < set(_normalized_words(other))
                and _normalized_words(other)[-1] == g_head
            ]
            if len(specifics) == 1:
                specific = specifics[0]
                merged_states = _merge_states(deduped[specific], deduped.pop(general))
                deduped[specific] = merged_states
                entry = report["merged"].setdefault(
                    specific, {"absorbed": [], "rule": "anaphoric_containment", "states": []}
                )
                entry["absorbed"].append(general)
                # A rule-1 entry that then absorbs by rule 2 carries both rules, traced as such.
                if entry["rule"] != "anaphoric_containment":
                    entry["rule"] = f"{entry['rule']}+anaphoric_containment"
                entry["states"] = merged_states
                changed = True
                break  # dict mutated -- restart the pass
            if len(specifics) > 1:
                report["ambiguous_not_merged"][general] = sorted(specifics)

    return deduped, report


if __name__ == "__main__":
    MODELS = [
        #"meta/llama-3.1-8b-instruct",
        #"meta/llama-3.3-70b-instruct",
        #"mistralai/mistral-nemotron",
        #"openai/gpt-oss-20b",
        #"openai/gpt-oss-120b",
        #"mistralai/mistral-large-3-675b-instruct-2512",
        #"nvidia/nemotron-3-super-120b-a12b",
        #"nvidia/nemotron-3-nano-30b-a3b",
        #"nvidia/llama-3.3-nemotron-super-49b-v1.5",
        #"qwen/qwen3.6-27b",
        #"cohere/north-mini-code:free",
        "google/gemma-4-31b-it:free"
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

    PROMPTS = {
        "zero_shot": STATE_SPACE_PROMPT_PERMISSIVE,
        "one_shot": STATE_SPACE_PROMPT_PERMISSIVE_FEWSHOT,
        #"two_shot": STATE_SPACE_PROMPT_PERMISSIVE_TWOSHOT,
        #"cot": STATE_SPACE_PROMPT_PERMISSIVE_COT,
    }

    results = {}
    for prompt_name, prompt_template in PROMPTS.items():
        results[prompt_name] = {}
        for model_name in MODELS:
            llm = get_llm(temperature=0, model=model_name)
            results[prompt_name][model_name] = {}
            for case_name, text in CASES.items():
                try:
                    candidates = generate_candidates(text, llm, prompt_template)
                    filtered, report = filter_state_space(candidates)
                    deduped, dedup_report = deduplicate_state_space(filtered)
                except Exception as e:
                    candidates, filtered, report = {"error": str(e)}, {}, {}
                    deduped, dedup_report = {}, {"merged": {}, "ambiguous_not_merged": {}}
                results[prompt_name][model_name][case_name] = {
                    "candidates": candidates,
                    "validation_report": report,
                    "state_space": deduped,
                    "state_space_dedup": dedup_report,
                }
                for kept, entry in dedup_report["merged"].items():
                    print(f"    [dedup] {' + '.join(entry['absorbed'])} -> {kept} ({entry['rule']})")
                for general, specifics in dedup_report["ambiguous_not_merged"].items():
                    print(f"    [dedup] {general!r} ambiguous between {specifics} -- NOT merged")
                print(f"[{prompt_name}][{model_name}] {case_name}: kept={list(deduped.keys())}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")