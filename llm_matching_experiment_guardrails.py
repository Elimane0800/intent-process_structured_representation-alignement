"""Version AVEC garde-fous de l'experience LLM (state_matching remplace par GPT-5.5) -- cible
directement les trois failles diagnostiquees sur la version sans garde-fou
(llm_matching_experiment.py, Tables V/W/X) :

  1. ASSIGNATION FORCEE (ex. process_instance.finished force sur "Ship bicycle to customer" par
     manque d'alternative) -> le prompt autorise et ENCOURAGE explicitement de laisser un
     milestone sans correspondance ; chaque assignation retenue doit porter une justification
     courte, VERIFIEE MECANIQUEMENT (grounding lexical minimal contre le label reel de
     l'activite -- jamais acceptee sur la seule confiance du LLM). Meme discipline que
     precondition_prompt.py cote texte (citation verifiable, jamais une affirmation nue).

  2. DUPLICATION (bicycle.ordered ET order.received tous deux forces sur la meme activite) ->
     le prompt demande explicitement de verifier que le milestone correspond a l'ACTION PROPRE
     de l'activite, pas seulement a son theme general.

  3. INSTABILITE (15.8% d'accord 3/3 sur BicycleManufacturing) -> VOTE MAJORITAIRE sur 3
     repetitions : un match n'est retenu dans le resultat final que si au moins 2 des 3
     repetitions independantes l'ont produit.

Compare quatre points sur les MEMES 2 cas que la version sans garde-fou, pour permettre une
comparaison directe : (a) chaque repetition individuelle AVEC garde-fous (grounding seul, pas
encore de vote), (b) le consensus par vote majoritaire, contre (c) les resultats sans garde-fou
deja mesures (rappeles en dur ci-dessous, pas re-executes -- ce sont des appels LLM couteux et
deja obtenus)."""

import csv
import json
import os
import re
from collections import Counter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.nodes.state_matching_node import _build_contextualized_activities, _entity_states, render_candidate
from agent.nodes.state_matching_node import build_process_state_graph
from agent.nodes.alignment_node import check_alignment, CONFIDENCE_THRESHOLD
from agent.nodes.report_node import collect_failures, cluster_by_root, status_for_root, compute_verdict
from agent.models.base_llm import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "llm_matching_guardrails")

RUN_ID = "run_2"
ALIGNMENT_TAU = CONFIDENCE_THRESHOLD
MODEL_NAME = "gpt-5.5"
N_REPEATS = 3

# Rappel des resultats SANS garde-fou (llm_matching_experiment.py, deja obtenus -- pas
# re-executes ici) pour comparaison directe dans Table Z, sans confondre les deux experiences.
NO_GUARDRAIL_BASELINE = {
    "OriginalDataset/ClaimsCreation": [
        {"repeat": 1, "SATISFIED": 4, "VIOLATED": 0, "UNRESOLVABLE": 9, "verdict": "FAITHFUL_WITH_RESERVATIONS"},
        {"repeat": 2, "SATISFIED": 4, "VIOLATED": 0, "UNRESOLVABLE": 9, "verdict": "FAITHFUL_WITH_RESERVATIONS"},
        {"repeat": 3, "SATISFIED": 4, "VIOLATED": 0, "UNRESOLVABLE": 9, "verdict": "FAITHFUL_WITH_RESERVATIONS"},
    ],
    "OriginalDataset/BicycleManufacturing": [
        {"repeat": 1, "SATISFIED": 9, "VIOLATED": 3, "UNRESOLVABLE": 8, "verdict": "NOT_FAITHFUL"},
        {"repeat": 2, "SATISFIED": 3, "VIOLATED": 3, "UNRESOLVABLE": 14, "verdict": "NOT_FAITHFUL"},
        {"repeat": 3, "SATISFIED": 5, "VIOLATED": 5, "UNRESOLVABLE": 10, "verdict": "NOT_FAITHFUL"},
    ],
}

CASES: list[tuple[str, str, str]] = [
    ("gpt-5.5", "OriginalDataset", "ClaimsCreation"),
    ("gpt-5.5", "OriginalDataset", "BicycleManufacturing"),
]


MATCHING_PROMPT_TEMPLATE = """You are given a business process described by a BPMN diagram, and a list of \
"milestones" (facts of the form entity.state) that were extracted independently from the \
textual description of this same process.

Your task: for each BPMN activity below, decide which milestone(s) from the list -- if any -- \
that activity realizes.

IMPORTANT rules, read carefully:
- Not every milestone needs to be assigned to an activity. Many milestones will correspond to \
NOTHING in this diagram (e.g. generic process lifecycle facts like "a process instance was \
created/finished", or a milestone that is really a duplicate/paraphrase of another one already \
assigned elsewhere). Leaving a milestone unassigned is the CORRECT and EXPECTED answer whenever \
you are not confident -- do not force a plausible-sounding but weak connection.
- A milestone should only be assigned to an activity if it describes that SPECIFIC activity's \
own action or its direct, immediate outcome -- not merely the same general topic or a nearby \
step in the process. If two candidate milestones both seem related to one activity, check \
carefully whether they describe the SAME fact (redundant, pick the better one, leave the other \
unassigned) or genuinely TWO DISTINCT facts that activity realizes (keep both).
- For every milestone you DO assign, you must give a short "reason" (max 15 words) that \
explicitly refers to words actually present in that activity's own label. A reason that could \
apply to almost any activity (too generic) is not acceptable -- be specific to this activity's \
actual label.

BPMN activities (id, label, and surrounding routing context in [] where available), in \
approximate process order:
{activities_block}

Milestones (entity.state, rendered as plain text for readability):
{milestones_block}

Respond with ONLY a JSON object, no prose, no markdown fences. Structure:
{{"activity_id_1": [{{"milestone": "entity.state1", "reason": "short reason referencing the activity's own label"}}], "activity_id_2": []}}
"""


def call_llm(prompt: str) -> str:
    llm = get_llm(model=MODEL_NAME)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def load_case_state(model: str, subset: str, case_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, model, f"{subset}__{case_name}__{RUN_ID}.json")
    if not os.path.exists(path):
        raise SystemExit(f"[guardrails] fichier introuvable : {path}")
    with open(path) as f:
        state = json.load(f)
    for required in ("dfg_edges", "state_space", "reference_graph"):
        if required not in state:
            raise SystemExit(f"[guardrails] {path} : champ '{required}' manquant")
    return state


def build_prompt(dfg_edges: list[dict], state_space: dict) -> tuple[str, dict]:
    contextualized = _build_contextualized_activities(dfg_edges)
    valid = {k: v for k, v in contextualized.items() if v["context_text"].strip()}

    activities_block = "\n".join(
        f"- id={aid} | \"{v['label']}\" | context: {v['context_text']}"
        for aid, v in valid.items()
    )
    candidates = [f"{e}.{s}" for e, value in state_space.items() for s in _entity_states(value)]
    milestones_block = "\n".join(
        f"- {c}  (\"{render_candidate(*c.split('.', 1))}\")" for c in candidates
    )

    prompt = MATCHING_PROMPT_TEMPLATE.format(activities_block=activities_block, milestones_block=milestones_block)
    return prompt, valid


def parse_llm_response(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


_WORD_RE = re.compile(r"[a-zA-Z]{3,}")  # mots de 3+ lettres, ignore ponctuation/mots triviaux


def is_grounded(reason: str, activity_label: str) -> bool:
    """Verification mecanique minimale -- pas un jugement de qualite, juste : la justification
    reference-t-elle au moins UN mot (3+ lettres) reellement present dans le label de
    l'activite ? Rejette les justifications vides ou totalement generiques qui ne mentionnent
    jamais le label lui-meme. Heuristique deliberement simple et STRICTE PAR DEFAUT (rejette si
    aucun chevauchement), documentee comme telle -- pas un NLI, un garde-fou de premier niveau."""
    if not reason or not reason.strip():
        return False
    label_words = {w.lower() for w in _WORD_RE.findall(activity_label)}
    reason_words = {w.lower() for w in _WORD_RE.findall(reason)}
    return bool(label_words & reason_words)


def build_matches_from_llm_guarded(parsed: dict, contextualized: dict, valid_activity_ids: set[str]) -> tuple[dict, list[str], list[dict]]:
    """Comme la version sans garde-fou, mais applique is_grounded() -- une assignation rejetee
    par le grounding est SIMPLEMENT ABSENTE du resultat (jamais forcee, jamais une exception),
    coherente avec la discipline 'UNRESOLVED est toujours une reponse legitime' deja actee
    ailleurs dans ce pipeline. Retourne aussi le detail des rejets (audit)."""
    matches = {aid: {"activity_label": contextualized[aid]["label"], "matches": []} for aid in valid_activity_ids}
    hallucinated_ids = []
    rejected = []

    for aid, entries in parsed.items():
        if not isinstance(entries, list):
            continue
        if aid not in valid_activity_ids:
            hallucinated_ids.append(aid)
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "milestone" not in entry:
                continue
            milestone, reason = entry.get("milestone"), entry.get("reason", "")
            if not isinstance(milestone, str):
                continue
            if is_grounded(reason, contextualized[aid]["label"]):
                matches[aid]["matches"].append({"match": milestone, "score": 1.0})
            else:
                rejected.append({"activity_id": aid, "milestone": milestone, "reason": reason})

    return matches, hallucinated_ids, rejected


def compute_verdict_only(alignment: dict) -> dict:
    failures = collect_failures(alignment)
    clusters = cluster_by_root(failures)
    n_violated_roots = n_unresolvable_roots = 0
    for root, items in clusters.items():
        status = status_for_root(root, failures, alignment)
        if status == "VIOLATED":
            n_violated_roots += 1
        else:
            n_unresolvable_roots += 1
    n_satisfied = sum(
        1 for entry in alignment.values() for t in entry["terms"] if t["status"] == "SATISFIED"
    )
    summary = {"ecarts": n_violated_roots, "non_verifiable": n_unresolvable_roots, "conforme": n_satisfied}
    return {"summary": summary, "verdict": compute_verdict(summary)["verdict"]}


def evaluate(matches: dict, dfg_edges, reference_graph, activity_guards) -> dict:
    process_graph = build_process_state_graph(dfg_edges, matches)
    alignment = check_alignment(
        reference_graph, process_graph, matches,
        confidence_threshold=ALIGNMENT_TAU, activity_guards=activity_guards,
    )
    n_sat = n_viol = n_unres = 0
    for entry in alignment.values():
        for t in entry.get("terms", []):
            if t["status"] == "SATISFIED":
                n_sat += 1
            elif t["status"] == "VIOLATED":
                n_viol += 1
            else:
                n_unres += 1
    result = compute_verdict_only(alignment)
    return {"SATISFIED": n_sat, "VIOLATED": n_viol, "UNRESOLVABLE": n_unres, "verdict": result["verdict"]}


def majority_vote_matches(repeats_matches: list[dict], contextualized: dict, min_votes: int = 2) -> dict:
    """Consensus : pour chaque paire (activity_id, milestone), retenue si presente dans au moins
    min_votes des len(repeats_matches) repetitions independantes."""
    counts: Counter = Counter()
    for m in repeats_matches:
        for aid, data in m.items():
            for entry in data["matches"]:
                counts[(aid, entry["match"])] += 1

    consensus = {aid: {"activity_label": contextualized[aid]["label"], "matches": []} for aid in contextualized}
    for (aid, milestone), n in counts.items():
        if n >= min_votes:
            consensus[aid]["matches"].append({"match": milestone, "score": 1.0})
    return consensus


def run_experiment() -> tuple[list[dict], list[dict], list[dict]]:
    """Retourne (repeat_rows, rejected_rows, consensus_rows)."""
    repeat_rows, rejected_rows, consensus_rows = [], [], []

    for model, subset, case_name in CASES:
        case_label = f"{subset}/{case_name}"
        state = load_case_state(model, subset, case_name)
        dfg_edges, state_space, reference_graph = state["dfg_edges"], state["state_space"], state["reference_graph"]
        activity_guards = state.get("activity_guards")

        prompt, contextualized = build_prompt(dfg_edges, state_space)
        valid_set = set(contextualized.keys())

        repeats_matches = []
        for repeat_idx in range(1, N_REPEATS + 1):
            print(f"[guardrails] {case_label} -- repeat {repeat_idx}/{N_REPEATS} -- appel LLM ({MODEL_NAME})...")
            raw = call_llm(prompt)
            parsed = parse_llm_response(raw)
            if parsed is None:
                print(f"  [guardrails] JSON invalide, repeat ignore")
                repeat_rows.append({
                    "case": case_label, "repeat": repeat_idx, "parse_ok": False,
                    "n_hallucinated_ids": None, "n_rejected_grounding": None,
                    "SATISFIED": None, "VIOLATED": None, "UNRESOLVABLE": None, "verdict": "PARSE_FAILED",
                })
                continue

            matches, hallucinated_ids, rejected = build_matches_from_llm_guarded(parsed, contextualized, valid_set)
            repeats_matches.append(matches)

            for r in rejected:
                rejected_rows.append({"case": case_label, "repeat": repeat_idx, **r})

            result = evaluate(matches, dfg_edges, reference_graph, activity_guards)
            repeat_rows.append({
                "case": case_label, "repeat": repeat_idx, "parse_ok": True,
                "n_hallucinated_ids": len(hallucinated_ids), "n_rejected_grounding": len(rejected),
                **result,
            })

        if len(repeats_matches) >= 2:
            consensus = majority_vote_matches(repeats_matches, contextualized, min_votes=2)
            consensus_result = evaluate(consensus, dfg_edges, reference_graph, activity_guards)
            consensus_rows.append({"case": case_label, **consensus_result})

    return repeat_rows, rejected_rows, consensus_rows


def print_table(title: str, table: list[dict]):
    print(f"\n=== {title} ===")
    if not table:
        print("  (aucune donnee)")
        return
    headers = list(table[0].keys())
    widths = [max(len(str(h)), max((len(str(r.get(h, ""))) for r in table), default=0)) for h in headers]
    print("  " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in table:
        print("  " + " | ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))


def build_table_z(repeat_rows: list[dict], consensus_rows: list[dict]) -> list[dict]:
    """Comparaison a 3 colonnes : sans garde-fou (baseline rappelee) / avec garde-fous
    (moyenne des 3 repetitions individuelles) / avec garde-fous + vote majoritaire (consensus)."""
    table = []
    for case in sorted({r["case"] for r in repeat_rows}):
        baseline = NO_GUARDRAIL_BASELINE.get(case, [])
        guarded = [r for r in repeat_rows if r["case"] == case and r["parse_ok"]]
        consensus = next((r for r in consensus_rows if r["case"] == case), None)

        def avg(rows, key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        table.append({
            "case": case,
            "no_guardrail_avg_VIOLATED": avg(baseline, "VIOLATED"),
            "guarded_avg_VIOLATED": avg(guarded, "VIOLATED"),
            "consensus_VIOLATED": consensus["VIOLATED"] if consensus else None,
            "no_guardrail_verdicts": [r["verdict"] for r in baseline],
            "guarded_verdicts": [r["verdict"] for r in guarded],
            "consensus_verdict": consensus["verdict"] if consensus else None,
        })
    return table


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    repeat_rows, rejected_rows, consensus_rows = run_experiment()

    with open(os.path.join(OUT_DIR, "repeat_rows_raw.csv"), "w", newline="") as f:
        if repeat_rows:
            writer = csv.DictWriter(f, fieldnames=list(repeat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(repeat_rows)
    with open(os.path.join(OUT_DIR, "rejected_rows_raw.csv"), "w", newline="") as f:
        if rejected_rows:
            writer = csv.DictWriter(f, fieldnames=list(rejected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rejected_rows)
    with open(os.path.join(OUT_DIR, "consensus_rows_raw.csv"), "w", newline="") as f:
        if consensus_rows:
            writer = csv.DictWriter(f, fieldnames=list(consensus_rows[0].keys()))
            writer.writeheader()
            writer.writerows(consensus_rows)

    print_table("Table Y1 -- Repetitions individuelles AVEC garde-fous", repeat_rows)
    print_table("Table Y2 -- Rejets par grounding (justification insuffisante)", rejected_rows)
    print_table("Table Y3 -- Consensus par vote majoritaire (>=2/3)", consensus_rows)

    table_z = build_table_z(repeat_rows, consensus_rows)
    print_table("Table Z -- Comparaison sans garde-fou / avec garde-fous / consensus", table_z)

    print(f"\nResultats dans {OUT_DIR}")