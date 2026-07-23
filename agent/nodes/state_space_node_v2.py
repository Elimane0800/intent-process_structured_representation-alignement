# Requires: pip install spacy nltk --break-system-packages
#           python -m spacy download en_core_web_sm
#           python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

import json
import os
import re
import spacy
from nltk.corpus import wordnet as wn
from agent.models.base_llm import get_llm
from agent.prompt.state_space_prompt_permissive import (
    STATE_SPACE_PROMPT_PERMISSIVE,
    STATE_SPACE_PROMPT_PERMISSIVE_FEWSHOT,
    STATE_SPACE_PROMPT_PERMISSIVE_TWOSHOT,
    STATE_SPACE_PROMPT_PERMISSIVE_COT,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_space_protocol2")

_NLP = spacy.load("en_core_web_sm")

ACTOR_HYPERNYMS = {"organization.n.01", "social_group.n.01", "person.n.01", "worker.n.01"}
ENTITY_HYPERNYMS = {"artifact.n.01", "document.n.01", "communication.n.02", "message.n.02", "act.n.02", "event.n.01"}

SUBJ_DEPS = {"nsubj"}
OBJ_DEPS = {"nsubjpass", "dobj", "pobj", "obj", "attr"}


# --- Step A: permissive candidate generation (LLM) ---

def generate_candidates(text: str, llm, prompt_template: str = STATE_SPACE_PROMPT_PERMISSIVE) -> dict:
    response = llm.invoke(prompt_template.format(text=text)).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e


# --- Step B: deterministic symbolic validation (no LLM, no training) ---

STOP_CANDIDATES = {"you", "it", "they", "he", "she", "we", "i", "process", "time", "way", "thing", "something"}


def _head_word(candidate_name: str) -> str:
    """Compound nouns are headed by their rightmost word in English (e.g. 'job_application' -> 'application').
    Lemmatized via spaCy so it matches token.lemma_ comparisons in srl_signal (plural/composite candidate
    names would otherwise never match their singular occurrences in the text)."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", candidate_name).replace("_", " ").split()
    raw_head = words[-1].lower() if words else candidate_name.lower()
    return _NLP(raw_head)[0].lemma_.lower()


def srl_signal(head: str, text: str) -> float | None:
    """Ratio of object occurrences over (subject + object) occurrences via dependency parsing.
    Returns None if the head word never appears as subject or object (signal unavailable)."""
    doc = _NLP(text)
    subj = obj = 0
    for token in doc:
        if token.lemma_.lower() != head:
            continue
        if token.dep_ in SUBJ_DEPS:
            subj += 1
        elif token.dep_ in OBJ_DEPS:
            obj += 1
    total = subj + obj
    return (obj / total) if total > 0 else None


def wordnet_signal(head: str) -> float:
    """1.0 = artifact/document/event-like (entity-leaning), 0.0 = organization/person-like (actor-leaning),
    0.5 = indeterminate. Uses the most common noun sense only."""
    synsets = wn.synsets(head, pos=wn.NOUN)
    if not synsets:
        return 0.5
    ancestors = {s.name() for path in synsets[0].hypernym_paths() for s in path}
    if ancestors & ACTOR_HYPERNYMS:
        return 0.0
    if ancestors & ENTITY_HYPERNYMS:
        return 1.0
    return 0.5


def validate_candidates(candidates: dict, text: str) -> dict:
    """Vote rule, no calibrated weights: reject a candidate only if BOTH signals agree it is
    actor-like (srl <= 0.5 and wordnet == 0.0). Ties and missing signals default to keeping the
    candidate, since recall matters more than precision at this stage. Pronouns and generic
    filler words are auto-rejected before signal computation — no signal is informative for them."""
    report = {}
    for name in candidates:
        head = _head_word(name)
        if head in STOP_CANDIDATES:
            report[name] = {"f_srl": None, "f_wn": None, "kept": False, "reason": "stopword"}
            continue
        f_srl = srl_signal(head, text)
        f_wn = wordnet_signal(head)
        rejected = (f_srl is not None and f_srl <= 0.5) and (f_wn == 0.0)
        report[name] = {"f_srl": f_srl, "f_wn": f_wn, "kept": not rejected}
    return report


def filter_state_space(candidates: dict, text: str) -> tuple[dict, dict]:
    report = validate_candidates(candidates, text)
    filtered = {name: states for name, states in candidates.items() if report[name]["kept"]}
    return filtered, report


if __name__ == "__main__":
    MODELS = [
        #"meta/llama-3.1-8b-instruct",
        "meta/llama-3.3-70b-instruct",
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

    PROMPTS = {
        "zero_shot": STATE_SPACE_PROMPT_PERMISSIVE,
        "one_shot": STATE_SPACE_PROMPT_PERMISSIVE_FEWSHOT,
        "two_shot": STATE_SPACE_PROMPT_PERMISSIVE_TWOSHOT,
        "cot": STATE_SPACE_PROMPT_PERMISSIVE_COT,
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
                    filtered, report = filter_state_space(candidates, text)
                except Exception as e:
                    candidates, filtered, report = {"error": str(e)}, {}, {}
                results[prompt_name][model_name][case_name] = {
                    "candidates": candidates,
                    "validation_report": report,
                    "state_space": filtered,
                }
                print(f"[{prompt_name}][{model_name}] {case_name}: kept={list(filtered.keys())}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")