import json
import os
import re
import difflib
from agent.models.base_llm import get_llm
from agent.prompt.state_space_prompt import (
    STATE_SPACE_PROMPT, STATE_SPACE_PROMPT_FEWSHOT, STATE_SPACE_PROMPT_TWOSHOT, STATE_SPACE_PROMPT_COT,
    STATE_SPACE_PROMPT_V2, STATE_SPACE_PROMPT_FEWSHOT_V2, STATE_SPACE_PROMPT_TWOSHOT_V2, STATE_SPACE_PROMPT_COT_V2,
    STATE_SPACE_PROMPT_V3, STATE_SPACE_PROMPT_FEWSHOT_V3, STATE_SPACE_PROMPT_TWOSHOT_V3, STATE_SPACE_PROMPT_COT_V3,
    STATE_SPACE_PROMPT_V4, STATE_SPACE_PROMPT_FEWSHOT_V4, STATE_SPACE_PROMPT_TWOSHOT_V4, STATE_SPACE_PROMPT_COT_V4,
)
from agent.prompt.reformulation_prompt import REFORMULATION_PROMPT, REFORMULATION_PROMPT_ONESHOT
from agent.prompt.evidence_prompt import EVIDENCE_PROMPT

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_space")


def extract_evidence(text: str, state_space: dict, llm) -> dict:
    prompt = EVIDENCE_PROMPT.format(text=text, state_space=json.dumps(state_space))
    response = llm.invoke(prompt).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e


def validate_grounding(evidence: dict, text: str, threshold: float = 0.9) -> dict:
    """Check each quoted evidence is actually a verbatim (or near-verbatim) span of the source text.
    Pure text matching, no LLM call. Tests the model's claimed justification, not the entity name itself."""
    text_norm = " ".join(text.split())
    result = {}
    for entity_name, quote in evidence.items():
        if not isinstance(quote, str) or not quote.strip():
            result[entity_name] = False
            continue
        for part in quote.split(" | "):
            part_norm = " ".join(part.split())
            if part_norm and part_norm in text_norm:
                result[entity_name] = True
                break
        else:
            ratio = difflib.SequenceMatcher(None, quote, text_norm).find_longest_match().size / max(len(quote), 1)
            result[entity_name] = ratio >= threshold
    return result


def postprocess(text: str, state_space: dict, llm) -> dict:
    """Run evidence extraction + grounding check on top of an already-produced state space.
    Conflict resolution (self-refine) is intentionally excluded here: it showed unreliable
    behavior (explosion, degenerate collapse, silent no-op) in prior runs and is tested separately."""
    result = {}
    try:
        evidence = extract_evidence(text, state_space, llm)
        result["evidence"] = evidence
        result["grounding"] = validate_grounding(evidence, text)
    except Exception as e:
        result["evidence"] = {"error": str(e)}
    return result


def reformulate_text(text: str, llm, prompt_template: str = REFORMULATION_PROMPT) -> str:
    result = llm.invoke(prompt_template.format(text=text)).content.strip()
    if not result:
        raise ValueError("reformulation returned empty response")
    return result


def extract_state_space(text: str, llm, prompt_template: str = STATE_SPACE_PROMPT) -> dict:
    response = llm.invoke(prompt_template.format(text=text)).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e


if __name__ == "__main__":
    MODELS = [
        #"meta/llama-3.1-8b-instruct",
        #"meta/llama-3.3-70b-instruct",
        #"mistralai/mistral-medium-3.5-128b",
        #"mistralai/mistral-nemotron",
        "openai/gpt-oss-20b",
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
        "zero_shot": STATE_SPACE_PROMPT,
        #"one_shot": STATE_SPACE_PROMPT_FEWSHOT,
        #"two_shot": STATE_SPACE_PROMPT_TWOSHOT,
        #"cot": STATE_SPACE_PROMPT_COT,
        "zero_shot_v2": STATE_SPACE_PROMPT_V2,
        #"one_shot_v2": STATE_SPACE_PROMPT_FEWSHOT_V2,
        #"two_shot_v2": STATE_SPACE_PROMPT_TWOSHOT_V2,
        #"cot_v2": STATE_SPACE_PROMPT_COT_V2,
        "zero_shot_v3": STATE_SPACE_PROMPT_V3,
        #"one_shot_v3": STATE_SPACE_PROMPT_FEWSHOT_V3,
        #"two_shot_v3": STATE_SPACE_PROMPT_TWOSHOT_V3,
        #"cot_v3": STATE_SPACE_PROMPT_COT_V3,
        "zero_shot_v4": STATE_SPACE_PROMPT_V4,
        #"one_shot_v4": STATE_SPACE_PROMPT_FEWSHOT_V4,
        #"two_shot_v4": STATE_SPACE_PROMPT_TWOSHOT_V4,
        #"cot_v4": STATE_SPACE_PROMPT_COT_V4,
    }

    results = {}
    for prompt_name, prompt_template in PROMPTS.items():
        results[prompt_name] = {}
        for model_name in MODELS:
            llm = get_llm(temperature=0, model=model_name)
            results[prompt_name][model_name] = {}
            for case_name, text in CASES.items():
                try:
                    state_space = extract_state_space(text, llm, prompt_template)
                except Exception as e:
                    state_space = {"error": str(e)}
                entry = {"state_space": state_space}
                if "error" not in state_space:
                    entry.update(postprocess(text, state_space, llm))
                results[prompt_name][model_name][case_name] = entry
                print(f"[{prompt_name}][{model_name}] {case_name}: {state_space}")

    # reformulation step: reformulate text, then extract with zero-shot prompt.
    # both the reformulated text and the resulting state space are logged,
    # so a gain can be traced back to genuine clarification vs. hallucinated content.
    REFORMULATION_PROMPTS = {
        "reformulated_zero_shot": REFORMULATION_PROMPT,
        "reformulated_one_shot": REFORMULATION_PROMPT_ONESHOT,
    }
    for reformulation_name, reformulation_template in REFORMULATION_PROMPTS.items():
        results[reformulation_name] = {}
        for model_name in MODELS:
            llm = get_llm(temperature=0, model=model_name)
            results[reformulation_name][model_name] = {}
            for case_name, text in CASES.items():
                try:
                    reformulated = reformulate_text(text, llm, reformulation_template)
                    state_space = extract_state_space(reformulated, llm, STATE_SPACE_PROMPT)
                except Exception as e:
                    reformulated = None
                    state_space = {"error": str(e)}
                entry = {"reformulated_text": reformulated, "state_space": state_space}
                if "error" not in state_space:
                    entry.update(postprocess(reformulated, state_space, llm))
                results[reformulation_name][model_name][case_name] = entry
                print(f"[{reformulation_name}][{model_name}] {case_name}: {state_space}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")