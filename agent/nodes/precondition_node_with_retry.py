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

# v4 vs v3:
# - Unary negation (NOT) is now supported, mirroring precondition_prompt.py v4. NOT attaches to
#   a single term ("NOT entity.state"), never to a parenthesized group -- deliberately narrower
#   than a general negation operator, in the same spirit as the rest of this grammar. parse_
#   precondition()'s STRUCTURAL parsing needed no change at all: NOT is entirely a term-content
#   concern (does this specific string start with "NOT "?), never a separator concern, since
#   " AND "/" OR " splitting is unaffected by what a term's own text looks like. The actual
#   logic changes are isolated to two places:
#     (a) _validate_one -- existence, self-reference, and mutual-exclusivity checks now strip a
#         leading "NOT " before comparing a term against valid_targets/target/other terms in its
#         clause (via _strip_not), and a new check rejects direct contradiction: a term and its
#         own exact negation both present in the same AND-clause (e.g. "A" and "NOT A").
#     (b) build_graph -- a negated term collapses to the SAME graph node as its positive form for
#         cycle-detection connectivity ("A requires NOT B, B requires A" is just as circular as
#         "A requires B, B requires A"; polarity does not remove a causal cycle, and treating
#         "NOT B" and "B" as two unrelated nodes would make such a cycle undetectable).
# - RETRY_PROMPT (precondition_prompt_retry.py) updated in the SAME change, not after, to avoid
#   reproducing the exact stale-grammar bug already documented above for the v2->v3 OR/AND-mix
#   transition: a retry prompt lagging the main grammar can only ever narrow what a retry can
#   recover, never help it.
#
# v3 vs v2:
# - _validate_one / build_graph now use parse_precondition(), a one-level DNF grammar (OR of
#   AND-clauses) instead of requiring a PURE AND or a PURE OR. The mixed case ("A, and either B
#   or C") was previously forced to UNRESOLVED even though the text expressed it clearly (cf.
#   the discarded "claim.paid" example in precondition_prompt.py) -- an unjustified restriction,
#   not a real grammar limit. Parentheses are only REQUIRED when a precondition mixes AND and OR
#   at the same level ("(A AND B) OR C") -- a pure AND or pure OR precondition is still written
#   exactly as before, no parentheses needed, 100% backward compatible with every precondition
#   ever accepted under v1/v2. Mutual exclusivity (two states of the same entity never together)
#   is now scoped to each AND-CLAUSE individually, not to the whole precondition -- the same
#   entity can appear in two different OR-branches (that is exactly what a disjunction over one
#   entity's own outcomes expresses), but never twice inside the same AND-clause.
#
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


def load_state_spaces(
    prompt_name: str = "one_shot", run_filename: str | None = None, model_name: str | None = None
) -> dict:
    """Loads the 'kept' state space per case from a Protocol 3 run. Pass run_filename explicitly
    (e.g. 'run_17.json') to pin a fixed reference U; otherwise falls back to the latest run
    available, which can vary across executions -- pin it for any real comparison across models.

    Pass model_name explicitly to pick a specific backbone's state space. Previously this
    function silently took `next(iter(data[prompt_name]))` -- the FIRST model in whatever order
    the run's JSON happened to store them, itself just the iteration order of the MODELS list in
    the script that produced the run. Two calls with the exact same run_filename could therefore
    load two different state spaces depending only on dict/list ordering, with nothing in the
    call site indicating which one was actually used unless you read the printed log line. If
    model_name is omitted and the run contains more than one model for prompt_name, this now
    raises rather than silently picking one -- the whole point of accepting run_filename to pin
    reproducibility is defeated if model selection is still implicit. Omitting model_name only
    works when there is exactly one model to begin with, in which case there is no ambiguity to
    silently resolve."""
    if run_filename is None:
        existing = [f for f in os.listdir(PROTOCOL3_DIR) if re.match(r"run_\d+\.json$", f)]
        run_filename = max(existing, key=lambda f: int(re.match(r"run_(\d+)\.json$", f).group(1)))
    with open(os.path.join(PROTOCOL3_DIR, run_filename)) as f:
        data = json.load(f)

    available_models = list(data[prompt_name])
    if model_name is None:
        if len(available_models) != 1:
            raise ValueError(
                f"{run_filename}[{prompt_name}] contains {len(available_models)} models "
                f"{available_models} -- pass model_name explicitly, do not rely on iteration order."
            )
        model_name = available_models[0]
    elif model_name not in available_models:
        raise ValueError(f"model_name {model_name!r} not found in {run_filename}[{prompt_name}]: {available_models}")

    print(f"Loading state spaces from {run_filename} [{prompt_name}][{model_name}]")
    return {
        case_name: entry["state_space"]
        for case_name, entry in data[prompt_name][model_name].items()
    }


# --- Step 1: single LLM call, proposes a precondition + quote for every state ---

def extract_preconditions(text: str, state_space: dict, llm, prompt_template: str = PRECONDITION_PROMPT) -> dict:
    """Single LLM call: returns {entity.state: {"precondition": ..., "quote": ...}} as proposed
    by the model, unvalidated. Strips a "FINAL_ANSWER:" prefix if present, so COT-style prompts
    (which reason before answering) parse the same way as direct-answer prompts.

    The state space shown to the LLM is flattened to {entity: [state, ...]} via _entity_states
    before being embedded in the prompt -- PRECONDITION_PROMPT's own worked examples all show
    that flat shape, never the nested {"states": [...], "exclusive_groups": [...],
    "concurrent_with": [...]} shape state_space_node_v3.py actually produces since Protocol 3.2.
    Showing the model a differently-shaped state space than its own examples train it on is its
    own source of confusion, separate from (but discovered alongside) the _all_valid_targets bug
    this same mismatch caused -- see _entity_states' docstring. exclusive_groups/concurrent_with
    are deliberately NOT shown here: this prompt's grammar has no use for them (per the earlier,
    explicit decision to keep exclusive_groups a non-blocking, later cross-check rather than
    feed it into precondition extraction itself)."""
    flat_state_space = {entity: _entity_states(value) for entity, value in state_space.items()}
    prompt = prompt_template.format(text=text, state_space=json.dumps(flat_state_space))
    response = llm.invoke(prompt).content.strip()
    if "FINAL_ANSWER:" in response:
        response = response.split("FINAL_ANSWER:")[-1]
    content = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"parse failed, raw response={response!r}") from e


# --- Step 2: mechanical validation layer, no LLM ---

def _entity_states(value) -> list[str]:
    """Extracts the plain list of state labels from a state-space entity's value. Accepts both
    the legacy bare-list shape and the shape state_space_node_v3.py has produced since Protocol
    3.2 ({"states": [...], "exclusive_groups": [...], "concurrent_with": [...]}).

    Fixes a real, previously undetected bug: _all_valid_targets used to do
    `for state in states` directly on the entity's raw value. Once that value became a dict
    (Protocol 3.2), iterating it yields its KEYS ("states", "exclusive_groups",
    "concurrent_with"), not the actual state labels -- silently producing bogus targets like
    "job.states" instead of "job.permanent", which made every real target look "unknown" and
    downgraded every single precondition in a run, regardless of how good the LLM's answer
    actually was. Never caught earlier because this file's own tests were written against
    simplified flat state spaces, never against the real nested output of
    state_space_node_v3.py -- an integration gap between two files edited separately, not a
    fault in either one considered in isolation. Centralizing extraction here, used by both
    _all_valid_targets (below) and extract_preconditions (so the LLM is shown the same flat
    shape its own prompt examples train it to expect, not the nested one), closes it in one
    place rather than needing two matching fixes kept in sync by hand."""
    if isinstance(value, dict):
        return list(value.get("states", []))
    return list(value)


def _all_valid_targets(state_space: dict) -> set:
    return {f"{entity}.{state}" for entity, value in state_space.items() for state in _entity_states(value)}


def _find_quote_position(quote: str, text: str, threshold: float = 0.9) -> tuple[int, int] | None:
    """Returns (start, end) of `quote` within the whitespace-normalized text if it grounds there
    (exact substring first, else fuzzy longest-common-match ratio), else None. Indices are in
    the normalized-text coordinate space -- fine for the ordering/between-span checks this is
    used for (_validate_quote_grounding), never used to slice back into the original text."""
    if not quote:
        return None
    text_norm = " ".join(text.split())
    quote_norm = " ".join(quote.split())
    idx = text_norm.find(quote_norm)
    if idx != -1:
        return idx, idx + len(quote_norm)
    match = difflib.SequenceMatcher(None, quote_norm, text_norm).find_longest_match()
    ratio = match.size / max(len(quote_norm), 1)
    if ratio >= threshold:
        return match.b, match.b + match.size
    return None


def _validate_quote(quote: str, text: str, threshold: float = 0.9) -> bool:
    """Grounding check: exact substring match first, else fuzzy longest-common-match ratio."""
    return _find_quote_position(quote, text, threshold) is not None


_DISJUNCTIVE_MARKERS = (" or ", " either ", " unless ", " otherwise ", " alternatively ")


def _has_disjunctive_marker(span: str) -> bool:
    padded = f" {' '.join(span.split()).lower()} "
    return any(marker in padded for marker in _DISJUNCTIVE_MARKERS)


def _validate_quote_grounding(quote, precondition: str, text: str) -> bool:
    """Accepts either a single verbatim quote (str) justifying the whole rule in one continuous
    span, OR an ordered list of verbatim segments (list[str]) for rules whose justification is
    built by accumulation across separate, non-adjacent sentences rather than one contiguous
    span -- a real pattern in narrative source text (a fact established in one sentence, a
    second fact established two sentences later, with no single span stating both together),
    not a hypothetical relaxation for its own sake.

    Two conditions, both required for the multi-segment form, keep this anti-hallucination
    rather than a loophole:
      1. ORDER: each segment must independently ground in the text (same check as a single
         quote), and segments must appear in the text in the SAME order as listed, without
         overlapping.
      2. NO DISJUNCTIVE MARKER, when grounding an AND: if the precondition combines terms with
         AND, the text SPAN BETWEEN two consecutive segments must not contain a disjunctive
         marker ("or", "either", "unless", "otherwise", "alternatively"). Order alone is NOT
         sufficient here -- two independently-true facts stated as ALTERNATIVES ("either the
         manager approves, or the system auto-approves") still appear in text order, so a model
         could otherwise ground a fabricated AND on genuinely disjunctive language. This check
         does not apply when the precondition itself is a pure OR (no AND present) -- a
         disjunctive marker between two OR-alternatives' segments is expected, not suspicious.

    This is a deliberately narrow, generic heuristic (a short closed list of English discourse
    markers, not tuned to any domain or backbone) -- it catches the specific adversarial pattern
    that motivated it, not every conceivable way a model could misrepresent a text's logic."""
    if isinstance(quote, str):
        return _validate_quote(quote, text)
    if isinstance(quote, list) and len(quote) == 1:
        # A single-element list is just a plain quote wrapped in array syntax -- a real,
        # observed formatting choice from at least one backbone (nemotron-3-super-120b),
        # not a genuine multi-segment claim. There is nothing to order and no "between two
        # segments" span to screen for a disjunctive marker, so it degrades to the plain-string
        # case rather than being rejected outright for failing the 2+-segment requirement below.
        single = quote[0]
        return isinstance(single, str) and bool(single) and _validate_quote(single, text)
    if not isinstance(quote, list) or len(quote) < 2 or any(not isinstance(q, str) or not q for q in quote):
        return False

    text_norm = " ".join(text.split())
    positions = []
    for segment in quote:
        span = _find_quote_position(segment, text)
        if span is None:
            return False
        positions.append(span)

    is_conjunctive = " AND " in precondition
    for (start_i, end_i), (start_next, _) in zip(positions, positions[1:]):
        if start_next < end_i:
            return False  # overlapping or out of order
        if is_conjunctive and _has_disjunctive_marker(text_norm[end_i:start_next]):
            return False  # AND grounded across a disjunctive marker in the source -- suspicious
    return True


def _split_top_level(s: str, sep: str) -> list[str]:
    """Coupe `s` sur `sep` uniquement au niveau 0 de parenthesage -- une parenthese ouverte
    fait monter la profondeur, jamais coupe a l'interieur. Utilitaire generique, pas specifique
    a OR/AND, reutilise pour les deux niveaux du parseur ci-dessous."""
    parts, depth, buf, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif depth == 0 and s[i:i + len(sep)] == sep:
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _is_negated(term: str) -> bool:
    return term.startswith("NOT ")


def _strip_not(term: str) -> str:
    """Strips exactly one leading 'NOT ' if present, else returns the term unchanged. A term
    that starts with 'NOT' but not 'NOT ' (e.g. a malformed trailing 'NOT' with nothing after
    it) is intentionally left untouched -- it will then fail the valid_targets existence check
    naturally, exactly like any other unknown term, with no special-casing required."""
    return term[len("NOT "):] if _is_negated(term) else term


def parse_precondition(precondition: str) -> tuple[list[tuple[str, ...]], str | None]:
    """Parse une precondition en DNF a UN SEUL niveau : une liste de clauses-ET, jointes par OU
    (`clauses`, chaque clause un tuple de termes). Retourne (clauses, raison_erreur) --
    raison_erreur est None si le parsing a reussi.

    Grammaire volontairement etroite (dans le meme esprit que le reste du projet : jamais une
    grammaire booleenne generale quand un fragment restreint suffit) :
      precondition := clause (" OR " clause)*
      clause       := "(" terme (" AND " terme)+ ")"  |  terme
      terme        := ["NOT "] entity.state
    Une parenthese n'est OBLIGATOIRE que pour un groupe ET combine avec un OU ailleurs dans la
    meme precondition -- retrocompatible a 100% : "A AND B" (aucun OR) reste valide SANS
    parentheses, exactement comme avant ; "A OR B" (aucun AND) aussi. Les parentheses ne
    deviennent necessaires que pour lever l'ambiguite d'un vrai melange, ex.
    "(A AND B) OR C" -- "A AND B OR C" sans parentheses est rejete comme ambigu plutot que
    de deviner une associativite.

    NOT (v4) ne change RIEN a cette fonction structurellement : c'est un prefixe sur le CONTENU
    d'un terme, jamais un separateur -- "NOT entity.state" traverse le split sur " AND "/" OR "
    exactement comme un terme non-negatif, et se retrouve intact comme un seul element de tuple.
    L'interpretation de ce prefixe (existence, auto-reference, contradiction) est geree en aval,
    dans _validate_one et build_graph, jamais ici.

    Precondition triviale ("INITIAL"/"UNRESOLVED") : appel non prevu ici, gere par l'appelant
    avant d'atteindre ce parseur."""
    raw_clauses = _split_top_level(precondition, " OR ")

    if len(raw_clauses) == 1:
        # Aucun OR au niveau superieur -- ancienne grammaire pure-AND (ou terme seul),
        # parentheses optionnelles (tolerees si presentes de facon coherente).
        raw = raw_clauses[0].strip()
        if raw.startswith("(") and raw.endswith(")"):
            raw = raw[1:-1].strip()
        elif raw.startswith("(") or raw.endswith(")"):
            return [], f"unbalanced parentheses: {raw!r}"
        terms = tuple(t.strip() for t in raw.split(" AND "))
        if any(not t for t in terms):
            return [], f"empty term in precondition: {precondition!r}"
        return [terms], None

    # Plus d'une clause -- un vrai OR est present. Chaque clause doit alors etre soit un terme
    # nu, soit un groupe ET explicitement parenthese (2+ termes) -- un AND nu a ce niveau est
    # ambigu (ne dit pas s'il se lie avant ou apres le OR voisin), donc rejete plutot que devine.
    clauses: list[tuple[str, ...]] = []
    for raw in raw_clauses:
        raw = raw.strip()
        if not raw:
            return [], "empty clause between OR"
        if raw.startswith("(") or raw.endswith(")"):
            if not (raw.startswith("(") and raw.endswith(")")):
                return [], f"unbalanced parentheses in clause: {raw!r}"
            inner = raw[1:-1].strip()
            terms = tuple(t.strip() for t in inner.split(" AND "))
            if len(terms) < 2:
                return [], f"parenthesized clause must contain at least two AND-terms: {raw!r}"
        else:
            if " AND " in raw:
                return [], (f"AND used without parentheses in a clause combined with OR "
                            f"(ambiguous): {raw!r} -- wrap AND-groups in parentheses")
            if "(" in raw or ")" in raw:
                return [], f"unbalanced parentheses: {raw!r}"
            terms = (raw,)
        clauses.append(terms)
    return clauses, None


_MALFORMED_TARGET_PREFIX = "malformed or unknown target key"


def _entity_exclusive_groups(value) -> list:
    """Extracts exclusive_groups from a state-space entity's value, accepting both the nested
    shape ({"states": [...], "exclusive_groups": [...], ...}) and the legacy bare-list shape
    (which has no exclusive_groups at all -- correctly returns [])."""
    if isinstance(value, dict):
        return value.get("exclusive_groups", [])
    return []


def _clause_conflict_reason(clause: tuple[str, ...], state_space: dict) -> str | None:
    """Retourne une raison de rejet pour cette clause ET, ou None si elle est coherente. Deux
    verifications distinctes, dans cet ordre :
      1. Contradiction directe : un terme et sa propre negation exacte presents ensemble
         (ex. "warranty.active" ET "NOT warranty.active") -- affirme qu'un etat est a la fois
         vrai et faux, jamais une regle reelle.
      2. Exclusivite mutuelle : deux etats DIFFERENTS de la meme entite dans la meme clause ET,
         MAIS SEULEMENT si cette paire precise est listee dans le exclusive_groups de cette
         entite dans le state space -- jamais une hypothese generale "une entite = un seul etat
         actif". Le prefixe NOT est retire avant d'extraire le nom d'entite, sinon
         "NOT entity.state" serait lu comme appartenant a une entite nommee "NOT entity", ce qui
         est faux.

         CORRECTION (decouverte par test reel, pas anticipee) : la version precedente de cette
         fonction traitait TOUTE paire d'etats differents de la meme entite comme exclusive par
         defaut -- une hypothese ecrite avant que exclusive_groups n'existe, jamais reconciliee
         avec lui depuis. Elle contredisait directement la semantique que le state space lui-meme
         etablit depuis Protocol 3.2 : la plupart des etats d'une entite forment une sequence
         cumulative (ex. "sent" puis "confirmed" puis "rated" peuvent tous tenir a la fois pour
         une instance qui a progresse a travers les trois), seules les paires EXPLICITEMENT
         marquees exclusive_groups sont de vraies alternatives. Rejetait a tort des regles
         parfaitement legitimes comme "shipment.received AND shipment.passed" -- deux faits
         independants sur la meme entite, censes coexister, pas s'exclure.
    Ne s'applique qu'a l'interieur d'UNE clause -- deux clauses OR differentes ne sont jamais
    comparees ici, exactement comme avant l'ajout de NOT."""
    if len(clause) < 2:
        return None

    positive = {t for t in clause if not _is_negated(t)}
    negated_targets = {_strip_not(t) for t in clause if _is_negated(t)}
    contradiction = positive & negated_targets
    if contradiction:
        return (f"direct contradiction: {sorted(contradiction)} both asserted and negated "
                f"in the same AND-clause: {clause!r}")

    by_entity: dict[str, list[str]] = {}
    for t in clause:
        entity, _, state = _strip_not(t).rpartition(".")
        by_entity.setdefault(entity, []).append(state)

    for entity, states in by_entity.items():
        if len(states) < 2:
            continue
        exclusive_pairs = {frozenset(g) for g in _entity_exclusive_groups(state_space.get(entity, {}))}
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                if frozenset([states[i], states[j]]) in exclusive_pairs:
                    return (f"mutual exclusivity violation: '{entity}.{states[i]}' and "
                            f"'{entity}.{states[j]}' are marked exclusive_groups for this "
                            f"entity but both appear in the same AND-clause: {clause!r}")

    return None


def _validate_one(target: str, precondition: str, quote, valid_targets: set, text: str, state_space: dict) -> dict:
    """Validates a single (target, precondition, quote) triple against six mechanical rules:
    valid target key, parseable grammar (DNF, mixed AND/OR now allowed via explicit
    parentheses, unary NOT on individual terms -- see parse_precondition), reference existence,
    self-reference, mutual exclusivity + direct contradiction (scoped to each AND-clause
    individually, not the whole precondition -- see _clause_conflict_reason), and textual
    grounding. Every check involving a term strips a leading "NOT " first (via _strip_not)
    before comparing it against valid_targets/target/other terms -- a negated term references
    the SAME underlying entity.state for existence and self-reference purposes, it is only its
    truth value that is inverted.

    `quote` may be a single verbatim string (the default, justifying the whole rule in one
    continuous span) or an ordered list of verbatim segments, for rules whose justification is
    genuinely built by accumulation across separate, non-adjacent sentences rather than any
    single continuous span. See _validate_quote_grounding for the two conditions (ordering, and
    no disjunctive marker between segments backing an AND) that keep the multi-segment form
    anti-hallucination rather than a loophole -- this was a deliberate late addition after
    identifying that requiring a single contiguous quote for the whole rule was discarding
    genuinely text-grounded rules whose causal logic simply spans a few adjacent sentences, a
    real narrative pattern, not a hypothetical.

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

    clauses, reason = parse_precondition(precondition)

    if reason is None:
        all_terms = [t for clause in clauses for t in clause]
        unknown_terms = [t for t in all_terms if _strip_not(t) not in valid_targets]
        if unknown_terms:
            reason = f"references unknown state(s): {unknown_terms}"
        elif any(_strip_not(t) == target for t in all_terms):
            reason = "self-reference"
        else:
            for clause in clauses:
                reason = _clause_conflict_reason(clause, state_space)
                if reason:
                    break

    if reason is None and not _validate_quote_grounding(quote, precondition, text):
        reason = "quote(s) not found verbatim in text, out of order, or spanning a disjunctive marker under an AND rule"

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
        validated[target] = _validate_one(target, precondition, quote, valid_targets, text, state_space)
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
 
        new_entry = _validate_one(target, result.get("precondition"), result.get("quote"), valid_targets, text, state_space)
        new_entry["retried"] = True
        updated[target] = new_entry
 
    return updated


# --- Cycle detection over the final accepted graph ---

def build_graph(accepted: dict) -> dict:
    """accepted: {target: 'A AND B' or 'A OR B' or '(A AND B) OR C' or 'A AND NOT B' or
    similar}. Returns adjacency: term (NOT-stripped, for connectivity) -> list of
    (target, is_negated) pairs, one per edge, recording the polarity that produced it. Uses
    parse_precondition rather than a flat regex split -- a flat split on ' AND '/' OR ' would
    mis-cut a parenthesized DNF string (e.g. '(A AND B) OR C' would wrongly split into '(A',
    'B)', 'C'). Every entry in `accepted` is already validated OK by this point, so
    parse_precondition is guaranteed to succeed (no defensive error handling needed).

    A negated term (e.g. "NOT B") still collapses to the SAME graph node as its positive form
    ("B") for connectivity -- "A requires NOT B" and "B requires A" are about the same
    underlying state B either way, just a different polarity of reference to it. What changed
    from the previous version is that the polarity is now carried alongside each edge instead of
    being discarded, so detect_cycles() can tell apart a cycle built entirely of mutual
    negations (see its docstring) from one containing at least one positive dependency."""
    edges: dict[str, list[tuple[str, bool]]] = {}
    for target, precondition in accepted.items():
        clauses, _ = parse_precondition(precondition)
        for clause in clauses:
            for term in clause:
                node = _strip_not(term)
                edges.setdefault(node, []).append((target, _is_negated(term)))
    return edges


def detect_cycles(edges: dict) -> list[list[str]]:
    """Only reports a cycle if at least one edge along it is a POSITIVE (non-negated)
    dependency. Rationale, purely structural -- applies uniformly to any text or backbone, never
    tuned to a specific case: a cycle made ENTIRELY of negated edges (e.g. "A requires NOT B"
    and "B requires NOT A") is exactly how this grammar expresses two mutually exclusive
    outcomes of the same branch point -- the single most common, intended use of NOT -- not a
    genuine circular causal chain. Flagging it would make that legitimate pattern look
    suspicious every time it is used. A cycle containing at least one positive edge is still
    flagged: swapping even one negated edge in the loop for a positive dependency reintroduces a
    real "X must already hold before Y, which must already hold before X" problem that negation
    does not explain away."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    cycles = []

    def dfs(node, path, path_edges):
        color[node] = GRAY
        path.append(node)
        for neighbor, is_negated in edges.get(node, []):
            if color.get(neighbor, WHITE) == WHITE:
                path_edges.append(is_negated)
                dfs(neighbor, path, path_edges)
                path_edges.pop()
            elif color.get(neighbor) == GRAY:
                cycle_start = path.index(neighbor)
                cycle_nodes = path[cycle_start:] + [neighbor]
                cycle_edges = path_edges[cycle_start:] + [is_negated]
                if not all(cycle_edges):  # at least one positive (non-negated) edge present
                    cycles.append(cycle_nodes)
        path.pop()
        color[node] = BLACK

    for node in list(edges.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node, [], [])
    return cycles


# --- Branch coherence: additive, NON-BLOCKING cross-check across DIFFERENT targets (B.10) ---
#
# Everything above validates ONE target in isolation. This is the first check that looks across
# targets, using the state space's exclusive_groups -- and per the explicit decision recorded in
# state_space_observations_2.md, exclusive_groups is a noisy signal, not ground truth (measured
# false-positive rate on real runs: roughly as many wrong pairs as right ones). This check is
# therefore deliberately narrow and never punitive: it never downgrades anything, never mutates
# `validated`, and only flags the ONE case with no ambiguity regardless of whether the specific
# exclusive_groups pair itself is trustworthy -- two states the state space marks as mutually
# exclusive both resolving to the EXACT SAME accepted precondition. If that precondition is ever
# satisfied, both branches fire at once, which contradicts their own stated exclusivity no matter
# what. It does not attempt any broader logical-overlap analysis (e.g. two preconditions that
# partially share terms but are not identical) -- that would need a general boolean
# satisfiability check, out of scope for the same reason the rest of this grammar stays
# deliberately narrow rather than reaching for a general solver. Purely structural and
# deterministic -- no dependency on which text or which backbone produced the inputs.
#
# Motivated by two independent, real occurrences in the same run: gpt-oss-20b and
# nemotron-3-super-120b each assigned job.permanent and job.not_permanent (an exclusive_groups
# pair) the exact same precondition, meaning both branches would fire together -- invisible to
# every check above, since each target was validated in isolation.

def _normalize_precondition_for_comparison(precondition: str):
    """Hashable, order-independent representation of an ALREADY-VALIDATED precondition string,
    for exact structural comparison only -- never a logical equivalence check. Term order within
    a clause and clause order across OR-alternatives are ignored (so "A AND B" and "B AND A"
    compare equal); the terms themselves are compared as exact strings, so "X" and "NOT X" are
    correctly treated as different, not merged."""
    if precondition in ("INITIAL", "UNRESOLVED"):
        return precondition
    clauses, _ = parse_precondition(precondition)
    return frozenset(frozenset(clause) for clause in clauses)


def check_branch_coherence(state_space: dict, validated: dict) -> list[dict]:
    """Returns a list of conflict reports (never raises, never mutates `validated`). An empty
    list means no conflict was found under this narrow check -- NOT a guarantee of full
    coherence, only that this one specific, unambiguous pattern was not detected."""
    conflicts = []
    for entity, value in state_space.items():
        groups = value.get("exclusive_groups", []) if isinstance(value, dict) else []
        for group in groups:
            real_by_key: dict = {}
            for state in group:
                target = f"{entity}.{state}"
                entry = validated.get(target)
                if entry is None or entry["status"] != "OK":
                    continue
                precondition = entry["precondition"]
                if precondition in ("INITIAL", "UNRESOLVED"):
                    continue
                key = _normalize_precondition_for_comparison(precondition)
                real_by_key.setdefault(key, []).append(target)
            for key, targets in real_by_key.items():
                if len(targets) > 1:
                    conflicts.append({
                        "entity": entity,
                        "exclusive_group": group,
                        "conflicting_targets": targets,
                        "shared_precondition": validated[targets[0]]["precondition"],
                        "reason": ("targets marked mutually exclusive in the state space share "
                                   "the exact same accepted precondition -- reaching it would "
                                   "trigger both branches at once"),
                    })
    return conflicts


def run_naive_with_retry(text: str, state_space: dict, llm, prompt_template: str = PRECONDITION_PROMPT) -> dict:
    raw = extract_preconditions(text, state_space, llm, prompt_template)
    validated = validate_preconditions(text, state_space, raw)
    validated = retry_downgraded(text, state_space, validated, raw, llm)

    accepted = {t: e["precondition"] for t, e in validated.items() if e["precondition"] not in ("INITIAL", "UNRESOLVED")}
    cycles = detect_cycles(build_graph(accepted))
    branch_conflicts = check_branch_coherence(state_space, validated)

    return {"raw": raw, "validated": validated, "cycles": cycles, "branch_conflicts": branch_conflicts}


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
        #"gpt-5.5",
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
        ),
    }

    STATE_SPACES = load_state_spaces(run_filename="run_32.json", model_name="meta/llama-3.1-8b-instruct")

    PROMPTS = {
        "zero_shot": PRECONDITION_PROMPT,
        #"one_shot": PRECONDITION_PROMPT_FEWSHOT,
        #"two_shot": PRECONDITION_PROMPT_TWOSHOT,
        #"cot": PRECONDITION_PROMPT_COT,
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
                    print(f"[{prompt_name}][{model_name}] {case_name}: still_downgraded={n_downgraded} retried={n_retried} retried_and_fixed={n_retried_ok} cycles={output['cycles']} branch_conflicts={len(output['branch_conflicts'])}")
                    for conflict in output["branch_conflicts"]:
                        print(f"    [BRANCH_CONFLICT] {conflict['conflicting_targets']} share precondition: {conflict['shared_precondition']}")
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