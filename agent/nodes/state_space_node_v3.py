# Requires: pip install nltk --break-system-packages
#           python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
#
# Protocol 3 vs Protocol 2: the validation layer no longer relies on a hand-picked, open-ended
# set of hypernym synsets (ACTOR_HYPERNYMS/ENTITY_HYPERNYMS in v2), nor on a dependency-parsing
# ratio that degrades on low-frequency candidates (f_srl in v2). Instead it uses WordNet's own
# lexnames: a FIXED, CLOSED taxonomy of 25 noun categories that already covers all of WordNet.
# The actor/entity mapping is chosen once, over these 25 fixed categories, and never needs to
# grow as new domains/test cases are added -- unlike a hypernym synset list, which does.
#
# Protocol 3.1 -- four corrections, each additive, none breaking the existing report/traceability
# contract ("nothing silent, everything reported with a reason"):
#
#   A.1 -- lexname_signal used only synsets[0], a bet on general-English word-sense frequency
#          that ignores the text's own domain. Now checks the top-3 senses, but unanimity is
#          computed only over senses that map to ACTOR or ENTITY -- an unmapped ("OTHER") sense
#          is uninformative, not counter-evidence (this refinement was itself necessary: literal
#          unanimity across all 3 senses regressed on 'company', whose 3rd WordNet sense is an
#          unrelated, unmapped meaning, and would have flipped a correctly-excluded actor back to
#          kept).
#
#   A.2 -- the head word actually used for the lexical decision is now recorded per candidate
#          (validation_report[name]["head_used"]).
#
#   A.6 -- Rule 2 (anaphoric containment dedup) now additionally requires the word difference
#          between the general and the specific name to be drawn from a short closed list of
#          generic determiners (GENERIC_MODIFIERS), or the pair is surfaced in
#          ambiguous_not_merged instead of being silently auto-merged. This measurably changes
#          behavior on the two examples that originally motivated Rule 2 ("3d_model"/"model",
#          "work_accident"/"accident") -- both now land in ambiguous_not_merged rather than
#          auto-merging, since neither word difference ("3d", "work") is a generic determiner.
#          If this proves too conservative in practice, widen GENERIC_MODIFIERS deliberately,
#          case by case, rather than reverting to unconditional auto-merge.
#
#   A.8 -- validate_candidates returns an aggregate under the reserved key "__meta__" (never a
#          real candidate name), counting kept-but-INDETERMINATE candidates for later audit. Any
#          code iterating validation_report.items() assuming every key is a literal candidate
#          name must skip "__meta__" explicitly.
#
# Protocol 3.2 (this revision) -- mutual-exclusivity detection is now a SEPARATE, second LLM call
# (generate_exclusive_groups, below), not part of Step A's output anymore. See the Revision 3 note
# at the top of state_space_prompt_permissive.py for the full rationale; in short: two full test
# runs showed generous state enumeration and precise relational judgment competing for the same
# instruction budget within one call, and the discipline needed for one starved the other. This
# also closes, by construction, a bug the single-pass design produced directly: one backbone
# asserted an exclusive_groups pair referencing a state ("not_permanent") it had never added to
# its own "states" list. generate_exclusive_groups() only ever offers the model states that Step
# A/B already committed to, and rejects (with a reason, not silently) anything it invents anyway.
# The general-purpose structural sanitizer (_sanitize_exclusive_groups, still used inside
# filter_state_space) is kept as defense in depth regardless -- cheap, and it protects the schema
# invariant even if a future caller feeds this module state that never went through
# generate_exclusive_groups at all.

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
    EXCLUSIVE_GROUPS_PROMPT,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_space_protocol3")

# Closed set: chosen once over WordNet's fixed 25 noun.* lexnames, not over its open synset space.
ACTOR_LEXNAMES = {"noun.person", "noun.group"}
ENTITY_LEXNAMES = {"noun.artifact", "noun.communication", "noun.act", "noun.event", "noun.cognition"}

# Closed class in English (pronouns), independent of domain -- unlike a hypernym list, it does
# not grow when moving to a new test case or business domain.
STOP_CANDIDATES = {"you", "it", "they", "he", "she", "we", "i"}

# Closed, short, and deliberately conservative (A.6): only true generic determiners/qualifiers.
# See the Protocol 3.1 note above about the behavioral consequence of this list being this short.
GENERIC_MODIFIERS = {"the", "this", "that", "said"}


# --- Step A: permissive candidate generation (LLM) -- schema-normalized on the way out ---

def _normalize_entity_value(value) -> dict:
    """Accept either the legacy bare list-of-states shape or the new {"states": [...],
    "exclusive_groups": [...], "concurrent_with": [...]} shape, and always return the latter,
    with all three keys present (possibly empty). Nothing downstream needs to branch on shape
    again after this point."""
    if isinstance(value, list):
        return {"states": list(value), "exclusive_groups": [], "concurrent_with": []}
    if isinstance(value, dict):
        return {
            "states": list(value.get("states", [])),
            "exclusive_groups": [list(pair) for pair in value.get("exclusive_groups", [])],
            "concurrent_with": list(value.get("concurrent_with", [])),
        }
    raise ValueError(f"Unrecognized candidate value shape (expected list or dict): {value!r}")


def generate_candidates(text: str, llm, prompt_template: str = STATE_SPACE_PROMPT_PERMISSIVE) -> dict:
    response = llm.invoke(prompt_template.format(text=text)).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e
    return {name: _normalize_entity_value(value) for name, value in raw.items()}


# --- Step B: deterministic lexical validation (no LLM, no training, no growing list) ---

def _head_word(candidate_name: str) -> str:
    """Compound nouns are headed by their rightmost word in English (e.g. 'job_application' -> 'application').
    No separate lemmatization step needed: WordNet's own morphy backoff handles plural/inflected forms.
    (A.2 records this choice per-candidate downstream so its error rate can be measured, not just assumed.)"""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", candidate_name).replace("_", " ").split()
    return words[-1].lower() if words else candidate_name.lower()


def lexname_signal(head: str) -> str:
    """Classify the candidate's dominant noun sense by WordNet lexname (fixed 25-category taxonomy).
    A.1: looks at the top-3 synsets instead of only the first, to stop betting everything on
    sense #1's frequency ranking. But unanimity is computed only over the senses that actually
    map to ACTOR or ENTITY -- an unmapped ("OTHER") sense among the top 3 is not counter-evidence,
    it is simply uninformative, and must not by itself force INDETERMINATE.

    This distinction is not cosmetic: 'company' senses 1-2 are noun.group (ACTOR) and sense 3 is
    noun.state (unmapped/OTHER). Requiring literal unanimity across ALL three would make 'company'
    -- an actor this pipeline must exclude, and a case exercised throughout this project's own
    manual tests -- flip to INDETERMINATE and get wrongly kept. Ignoring OTHER senses when judging
    agreement avoids that regression while still catching the case this correction is actually
    meant to catch: a word whose informative (ACTOR- or ENTITY-mapped) senses genuinely disagree
    with each other."""
    synsets = wn.synsets(head, pos=wn.NOUN)
    if not synsets:
        return "INDETERMINATE"
    camps = set()
    for s in synsets[:3]:
        lexname = s.lexname()
        if lexname in ACTOR_LEXNAMES:
            camps.add("ACTOR")
        elif lexname in ENTITY_LEXNAMES:
            camps.add("ENTITY")
        # unmapped lexnames are ignored here on purpose -- see docstring.
    if camps == {"ACTOR"}:
        return "ACTOR"
    if camps == {"ENTITY"}:
        return "ENTITY"
    return "INDETERMINATE"  # covers: no informative sense at all, or genuine ACTOR/ENTITY split


def validate_candidates(candidates: dict) -> dict:
    """Reject only if the candidate's dominant sense is lexically ACTOR-like. Default to keeping
    otherwise (ENTITY or INDETERMINATE) -- recall matters more than precision at this stage.

    A.2: every entry now carries "head_used". A.8: an aggregate under the reserved "__meta__" key
    counts, among KEPT candidates, how many were kept only because they were INDETERMINATE rather
    than a confirmed ENTITY -- instrumentation for a later audit, not a behavior change.

    Also performs a light structural sanity check (not a semantic one -- Step B stays lexical
    only): if a candidate's exclusive_groups references a state that isn't in its own states
    list, that's flagged as exclusive_group_warnings rather than silently accepted or dropped.
    """
    report = {}
    total = 0
    indeterminate_kept = 0
    for name, value in candidates.items():
        total += 1
        head = _head_word(name)
        if head in STOP_CANDIDATES:
            report[name] = {"f_lex": None, "kept": False, "reason": "stopword", "head_used": head}
            continue

        f_lex = lexname_signal(head)
        kept = f_lex != "ACTOR"
        entry = {"f_lex": f_lex, "kept": kept, "head_used": head}
        if kept and f_lex == "INDETERMINATE":
            indeterminate_kept += 1

        states = set(value.get("states", [])) if isinstance(value, dict) else set(value)
        raw_groups = value.get("exclusive_groups", []) if isinstance(value, dict) else []
        bad_pairs = [pair for pair in raw_groups if not set(pair) <= states]
        if bad_pairs:
            entry["exclusive_group_warnings"] = bad_pairs

        report[name] = entry

    report["__meta__"] = {"total_candidates": total, "indeterminate_kept_count": indeterminate_kept}
    return report


def _sanitize_exclusive_groups(value: dict) -> dict:
    """Drop any exclusive_groups pair that references a state absent from this same candidate's
    own states list, rather than letting it travel downstream as a dangling reference. The
    validation_report warning (exclusive_group_warnings, computed in validate_candidates on the
    UNSANITIZED value) is the audit trail for this -- it already recorded that it happened before
    this function ever runs, so nothing is silently lost, only silently propagated is avoided.
    Never invents the missing state to "fix" the pair instead of dropping it: state A.5's own
    principle (Step A/B never invent content) applies here just as much as to a missing state
    label -- a hallucinated complementary state is not something Step B is entitled to manufacture."""
    states = set(value.get("states", []))
    clean_groups = [g for g in value.get("exclusive_groups", []) if set(g) <= states]
    if clean_groups == list(value.get("exclusive_groups", [])):
        return value
    sanitized = dict(value)
    sanitized["exclusive_groups"] = clean_groups
    return sanitized


def filter_state_space(candidates: dict) -> tuple[dict, dict]:
    report = validate_candidates(candidates)
    filtered = {
        name: _sanitize_exclusive_groups(value)
        for name, value in candidates.items()
        if report[name]["kept"]
    }
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
# Rule 2 -- ANAPHORIC CONTAINMENT (tightened by A.6, see the Protocol 3.1 note above): a name
# whose (normalized) word set is a proper subset of exactly ONE other name's, with the SAME head
# word, is a CANDIDATE for being an anaphoric mention of it. It is only auto-merged when, in
# addition, every word present in the specific name but absent from the general one belongs to
# GENERIC_MODIFIERS. Otherwise the pair is reported in ambiguous_not_merged, not merged and not
# silently ignored -- the uniqueness of the subset match is still worth surfacing even when the
# auto-merge itself is refused.
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
    exclusivity: one entity, one list of states. Also reused, unchanged, for merging
    "concurrent_with" lists (both are just deduplicated string lists)."""
    merged = list(base)
    for s in extra:
        if s not in merged:
            merged.append(s)
    return merged


def _merge_exclusive_groups(base: list[list[str]], extra: list[list[str]]) -> list[list[str]]:
    """Union of pairs/groups, deduping by member-set regardless of order or list vs. tuple."""
    merged: list[list[str]] = []
    seen: list[frozenset] = []
    for group in list(base) + list(extra):
        fs = frozenset(group)
        if fs not in seen:
            seen.append(fs)
            merged.append(sorted(group))
    return merged


def _merge_entity_values(base: dict, extra: dict) -> dict:
    """Merge two normalized entity values ({"states", "exclusive_groups", "concurrent_with"})
    into a new dict. Never mutates its inputs -- callers can keep using the originals safely."""
    return {
        "states": _merge_states(base.get("states", []), extra.get("states", [])),
        "exclusive_groups": _merge_exclusive_groups(
            base.get("exclusive_groups", []), extra.get("exclusive_groups", [])
        ),
        "concurrent_with": _merge_states(base.get("concurrent_with", []), extra.get("concurrent_with", [])),
    }


def deduplicate_state_space(state_space: dict) -> tuple[dict, dict]:
    """Returns (deduplicated U, report). Report = {"merged": {kept_name: {"absorbed": [...],
    "rule": ..., "states": [...], "exclusive_groups": [...], "concurrent_with": [...]}},
    "ambiguous_not_merged": {general: [specifics]}} -- every decision is reported, nothing merged
    or skipped silently.

    Does not mutate its input. Builds a rename map across both rules so that, at the end,
    "concurrent_with" references pointing at a name that got merged away are resolved to whatever
    name it was merged INTO, instead of dangling."""
    report = {"merged": {}, "ambiguous_not_merged": {}}
    rename_map: dict[str, str] = {}

    # Rule 1 -- group by fully normalized name, first-seen order preserved.
    groups: dict[str, list[str]] = {}
    for name in state_space:
        groups.setdefault("_".join(_normalized_words(name)), []).append(name)

    deduped: dict[str, dict] = {}
    for normalized, members in groups.items():
        if len(members) == 1:
            deduped[normalized] = dict(state_space[members[0]])
            if normalized != members[0]:
                rename_map[members[0]] = normalized
            continue

        value = {"states": [], "exclusive_groups": [], "concurrent_with": []}
        for m in members:
            value = _merge_entity_values(value, state_space[m])
            if m != normalized:
                rename_map[m] = normalized
        deduped[normalized] = value
        report["merged"][normalized] = {
            "absorbed": [m for m in members if m != normalized],
            "rule": "morphological_identity",
            "states": value["states"],
            "exclusive_groups": value["exclusive_groups"],
            "concurrent_with": value["concurrent_with"],
        }

    # Rule 2 -- absorb anaphoric generals into their unique, generically-qualified specific,
    # iterating until stable (handles chains in successive passes; bounded by entity count).
    changed = True
    while changed:
        changed = False
        for general in list(deduped):
            g_words = _normalized_words(general)
            g_set, g_head = set(g_words), g_words[-1]
            candidates_by_subset = [
                other for other in deduped
                if other != general
                and g_set < set(_normalized_words(other))
                and _normalized_words(other)[-1] == g_head
            ]

            if len(candidates_by_subset) == 0:
                continue

            if len(candidates_by_subset) > 1:
                report["ambiguous_not_merged"][general] = sorted(candidates_by_subset)
                continue

            specific = candidates_by_subset[0]
            diff = set(_normalized_words(specific)) - g_set
            if not diff <= GENERIC_MODIFIERS:
                # Unique subset match, but the extra words are not generic-only (A.6): surfaced,
                # not auto-merged, and not silently left out of the report either.
                report["ambiguous_not_merged"][general] = [specific]
                continue

            merged_value = _merge_entity_values(deduped[specific], deduped.pop(general))
            deduped[specific] = merged_value
            rename_map[general] = specific
            entry = report["merged"].setdefault(
                specific,
                {"absorbed": [], "rule": "anaphoric_containment", "states": [], "exclusive_groups": [], "concurrent_with": []},
            )
            entry["absorbed"].append(general)
            if entry["rule"] != "anaphoric_containment":
                entry["rule"] = f"{entry['rule']}+anaphoric_containment"
            entry["states"] = merged_value["states"]
            entry["exclusive_groups"] = merged_value["exclusive_groups"]
            entry["concurrent_with"] = merged_value["concurrent_with"]
            changed = True
            break  # dict mutated -- restart the pass

    # Resolve concurrent_with references through the rename chain built by both rules above,
    # so a reference to a name that got merged away points at wherever it actually ended up.
    def _resolve(n: str) -> str:
        seen = set()
        while n in rename_map and n not in seen:
            seen.add(n)
            n = rename_map[n]
        return n

    for entity, value in deduped.items():
        resolved = []
        for other in value.get("concurrent_with", []):
            r = _resolve(other)
            if r != entity and r in deduped and r not in resolved:
                resolved.append(r)
        value["concurrent_with"] = resolved

    return deduped, report


# --- Second, decoupled LLM call: mutual-exclusivity detection (Protocol 3.2) ---
#
# Runs AFTER Step B (lexical filtering) and BEFORE Step C (dedup): after B, so this call never
# has to reason about actors that are about to be discarded anyway (smaller prompt, less noise);
# before C, so groups get attached per pre-dedup entity name and are then merged along by Step
# C's existing _merge_entity_values logic exactly like "states" already are -- no special-casing
# needed in the dedup step itself.
#
# Structurally cannot reproduce the dangling-reference bug that motivated this split: every
# claimed pair is checked against that entity's OWN already-committed states list before being
# accepted, with the reason recorded (not silently dropped) when it is not.

def generate_exclusive_groups(
    text: str, state_space: dict, llm, prompt_template: str = EXCLUSIVE_GROUPS_PROMPT
) -> tuple[dict, dict]:
    """Returns (state_space_with_groups, report). Does not mutate its input. report =
    {"accepted": [...], "rejected": [...]}, each entry carrying entity/states/quote (accepted) or
    entity/states/quote/reason (rejected) -- mirrors the traceability convention already used by
    validate_candidates and deduplicate_state_space elsewhere in this file."""
    entity_list = "\n".join(
        f'- "{name}": {json.dumps(value.get("states", []))}' for name, value in state_space.items()
    )
    response = llm.invoke(prompt_template.format(text=text, entities=entity_list)).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e

    result = {name: dict(value) for name, value in state_space.items()}
    report = {"accepted": [], "rejected": []}

    for entry in raw.get("exclusive_groups", []):
        name = entry.get("entity")
        pair = entry.get("states")
        quote = entry.get("quote")

        if name not in result:
            report["rejected"].append({**entry, "reason": f"unknown entity {name!r}"})
            continue
        if not pair or len(pair) < 2:
            report["rejected"].append({**entry, "reason": "missing or too-short states list"})
            continue
        if not quote:
            report["rejected"].append({**entry, "reason": "missing quote"})
            continue
        if not set(pair) <= set(result[name].get("states", [])):
            report["rejected"].append(
                {**entry, "reason": "references a state outside this entity's own states list"}
            )
            continue

        existing = result[name].setdefault("exclusive_groups", [])
        if list(pair) not in existing:
            existing.append(list(pair))
            report["accepted"].append({"entity": name, "states": list(pair), "quote": quote})

    return result, report


if __name__ == "__main__":
    MODELS = [
        "meta/llama-3.1-8b-instruct",
        #"meta/llama-3.3-70b-instruct",
        "mistralai/mistral-nemotron",
        "openai/gpt-oss-20b",
        #"openai/gpt-oss-120b",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-30b-a3b",
        #"nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "qwen/qwen3.6-27b",
        "cohere/north-mini-code:free",
        #"google/gemma-4-31b-it:free",
        "gpt-5.5",
        # "gpt-5.5-pro",
        # "gpt-5.4-nano",
        # "gpt-5.6-sol",
        # "gpt-5.6-terra",
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
        )
    }

    PROMPTS = {
        #"zero_shot": STATE_SPACE_PROMPT_PERMISSIVE,
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
                    with_groups, exclusive_report = generate_exclusive_groups(text, filtered, llm)
                    deduped, dedup_report = deduplicate_state_space(with_groups)
                except Exception as e:
                    candidates, filtered, report = {"error": str(e)}, {}, {}
                    exclusive_report = {"accepted": [], "rejected": []}
                    deduped, dedup_report = {}, {"merged": {}, "ambiguous_not_merged": {}}
                results[prompt_name][model_name][case_name] = {
                    "candidates": candidates,
                    "validation_report": report,
                    "exclusive_groups_report": exclusive_report,
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