"""Test isole : remplacer state_matching (embedding + regle de retention) par un LLM (GPT-5.5),
SANS garde-fou dans cette premiere passe -- prompt permissif, aucune contrainte de citation
verbatim, aucun retry sur sortie malformee, aucune verification anti-hallucination CORRECTIVE
(seulement une MESURE du taux d'hallucination, cf. plus bas). Objectif : observer d'abord le
comportement brut sur 3 repetitions du MEME cas, avant de decider quels garde-fous ajouter.

Choix de conception, tranche explicitement (cf. discussion) : le LLM recoit l'espace d'etats U
DEJA EXTRAIT DU TEXTE (state_space, deja sur disque, run_2 -- aucun nouvel appel LLM pour cette
partie), pas seulement le BPMN seul. Deux raisons :
  1. Comparabilite : reference_graph/check_alignment reposent sur le vocabulaire de U. Si le LLM
     invente ses propres noms d'etats a partir du BPMN seul, aucune comparaison directe avec les
     4 regimes deja testes (current_margin / margin elargie / dissimilarite / assignation
     globale) n'est plus possible -- ce serait une tache differente pour le solveur.
  2. Portee : ceci teste UNIQUEMENT le remplacement de delta (la correspondance semantique,
     state_matching), pas de Rel (la structure, deja verifiee formellement par bpmn_to_spo +
     bpmn_guards_node) -- coherent avec la remarque faite avant de lancer cette piste : Rel
     reste une machinerie de graphe deterministe, seul delta est ici confie au LLM.

Ce que ce script mesure, sur 3 repetitions du MEME cas (temperature par defaut du modele, PAS
force a 0 -- observer la variance reelle, pas la masquer) :
  - Table V : accord milestone par milestone entre les 3 repetitions (coherence du LLM avec
    lui-meme).
  - Table W : taux d'hallucination -- part des activity_id retournes par le LLM qui n'existent
    PAS reellement dans le BPMN du cas (verifiable mecaniquement, sans jugement).
  - Table X : SATISFIED/VIOLATED/UNRESOLVABLE + verdict, une ligne par repetition, comparable
    directement aux Tables P/Q/R/T des scripts precedents (meme MATCHING regime alternatif,
    meme evaluation en aval).

ATTENTION -- interface LLM supposee, jamais verifiee sur agent/models/base_llm.py (fichier non
disponible) : `get_llm(temperature=..., model=...).invoke(prompt).content`, coherent avec
l'usage observe dans graph.py/state_space_node.py. A AJUSTER si l'interface reelle differe --
cf. la fonction call_llm() ci-dessous, isolee expres pour n'avoir qu'un seul endroit a corriger."""

import csv
import json
import os
import re

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
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "llm_matching_experiment")

RUN_ID = "run_2"
ALIGNMENT_TAU = CONFIDENCE_THRESHOLD
MODEL_NAME = "gpt-5.5"
N_REPEATS = 3

# Cas testes -- un sous-ensemble borne pour cette premiere passe (appels LLM reels, pas gratuits
# contrairement a tout ce qu'on a fait jusqu'ici). ClaimsCreation = le cas ou la contamination
# croisee a ete diagnostiquee (diagnose_claimscreation.py) -- le test le plus direct pour savoir
# si un LLM evite ce defaut precis. BicycleManufacturing = cas de reference deja bien compris
# (Cas 1, analyse a la main en tout debut de discussion), utile comme point de comparaison
# stable. AJUSTER cette liste si besoin d'un echantillon different.
CASES: list[tuple[str, str, str]] = [
    ("gpt-5.5", "OriginalDataset", "ClaimsCreation"),
    ("gpt-5.5", "OriginalDataset", "BicycleManufacturing"),
]


# --- Prompt, SANS garde-fou dans cette premiere passe (cf. docstring de tete) ---

MATCHING_PROMPT_TEMPLATE = """You are given a business process described by a BPMN diagram, and a list of \
"milestones" (facts of the form entity.state) that were extracted independently from the \
textual description of this same process.

Your task: for each BPMN activity below, decide which milestone(s) from the list -- if any -- \
that activity realizes. An activity can realize zero, one, or several milestones. A milestone \
can be realized by more than one activity if genuinely appropriate, but prefer assigning each \
milestone to its single best-matching activity when the choice is clear.

BPMN activities (id, label, and surrounding routing context in [] where available), in \
approximate process order:
{activities_block}

Milestones (entity.state, rendered as plain text for readability):
{milestones_block}

Respond with ONLY a JSON object, no prose, no markdown fences, mapping each activity id to the \
list of milestone strings (use the exact "entity.state" form, not the rendered text) it \
realizes. Omit activities with no matching milestone, or include them with an empty list --  \
either is fine. Example shape:
{{"activity_id_1": ["entity.state1", "entity.state2"], "activity_id_2": []}}
"""


def call_llm(prompt: str) -> str:
    """Isole -- SEUL endroit a corriger si l'interface reelle de get_llm() differe de
    l'hypothese documentee en tete de fichier."""
    llm = get_llm(model=MODEL_NAME)  # temperature PAS forcee a 0 -- on veut la variance reelle
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def parse_filename(filename: str):
    if not filename.endswith(".json"):
        return None
    stem = filename[:-5]
    if "__" not in stem:
        return None
    subset, rest = stem.split("__", 1)
    if "__" not in rest:
        return None
    case_name, run_id = rest.rsplit("__", 1)
    return subset, case_name, run_id


def load_case_state(model: str, subset: str, case_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, model, f"{subset}__{case_name}__{RUN_ID}.json")
    if not os.path.exists(path):
        raise SystemExit(f"[llm_matching] fichier introuvable : {path}")
    with open(path) as f:
        state = json.load(f)
    for required in ("dfg_edges", "state_space", "reference_graph"):
        if required not in state:
            raise SystemExit(f"[llm_matching] {path} : champ '{required}' manquant, choisis un autre cas")
    return state


def build_prompt(dfg_edges: list[dict], state_space: dict) -> tuple[str, list[str], dict]:
    """Retourne (prompt, all_valid_activity_ids, contextualized) -- all_valid_activity_ids sert
    a mesurer l'hallucination (Table W) sans aucun jugement, juste une appartenance a un
    ensemble connu."""
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

    prompt = MATCHING_PROMPT_TEMPLATE.format(
        activities_block=activities_block, milestones_block=milestones_block
    )
    return prompt, list(valid.keys()), valid


def parse_llm_response(raw: str) -> dict | None:
    """Aucune tentative de reparation/retry ici (sans garde-fou, cf. docstring) -- juste un
    nettoyage minimal des fences markdown eventuelles, pour ne pas faire echouer json.loads sur
    un artefact de formatage trivial plutot que sur une vraie erreur de contenu."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def build_matches_from_llm(parsed: dict, contextualized: dict, valid_activity_ids: set[str]) -> tuple[dict, list[str]]:
    """Convertit la sortie LLM {activity_id: [milestones]} au format 'matches' attendu par
    build_process_state_graph -- identique au format des autres scripts. Retourne aussi la
    liste des activity_id HALLUCINES (presents dans la reponse LLM mais absents du BPMN reel)."""
    matches = {aid: {"activity_label": contextualized[aid]["label"], "matches": []} for aid in valid_activity_ids}
    hallucinated_ids = []
    for aid, milestones in parsed.items():
        if not isinstance(milestones, list):
            continue
        if aid not in valid_activity_ids:
            hallucinated_ids.append(aid)
            continue
        for m in milestones:
            if isinstance(m, str):
                # score=1.0, jamais None : le LLM rend une decision BINAIRE (assigne ou non),
                # pas un score continu comme l'embedding -- confiance.get(s, 0.0) dans
                # tgms_solver.py suppose une valeur numerique des que la cle existe (None y
                # casse la comparaison "< tau"). 1.0 = jamais demote pour raison de confiance,
                # coherent avec le fait qu'on teste ici la justesse de la decision du LLM, pas
                # une notion de confiance graduee qu'on ne lui a pas demandee dans ce prompt.
                matches[aid]["matches"].append({"match": m, "score": 1.0})
    return matches, hallucinated_ids


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


def run_experiment() -> tuple[list[dict], list[dict]]:
    """Retourne (repeat_rows, milestone_assignments) -- repeat_rows pour Table W/X,
    milestone_assignments (une ligne par (cas, repeat, milestone, activity_id ou None)) pour
    Table V (accord inter-repetitions)."""
    repeat_rows = []
    milestone_assignments = []

    for model, subset, case_name in CASES:
        case_label = f"{subset}/{case_name}"
        state = load_case_state(model, subset, case_name)
        dfg_edges, state_space, reference_graph = state["dfg_edges"], state["state_space"], state["reference_graph"]
        activity_guards = state.get("activity_guards")

        prompt, valid_activity_ids, contextualized = build_prompt(dfg_edges, state_space)
        valid_set = set(valid_activity_ids)

        all_milestones = [f"{e}.{s}" for e, v in state_space.items() for s in _entity_states(v)]

        for repeat_idx in range(1, N_REPEATS + 1):
            print(f"[llm_matching] {case_label} -- repeat {repeat_idx}/{N_REPEATS} -- appel LLM ({MODEL_NAME})...")
            raw = call_llm(prompt)
            parsed = parse_llm_response(raw)

            if parsed is None:
                print(f"  [llm_matching] JSON invalide, repeat ignore (sans garde-fou = sans retry)")
                repeat_rows.append({
                    "case": case_label, "repeat": repeat_idx, "parse_ok": False,
                    "n_hallucinated_ids": None, "SATISFIED": None, "VIOLATED": None,
                    "UNRESOLVABLE": None, "verdict": "PARSE_FAILED",
                })
                continue

            matches, hallucinated_ids = build_matches_from_llm(parsed, contextualized, valid_set)

            for aid, data in matches.items():
                assigned = {m["match"] for m in data["matches"]}
                for milestone in all_milestones:
                    milestone_assignments.append({
                        "case": case_label, "repeat": repeat_idx, "milestone": milestone,
                        "assigned_activity": aid if milestone in assigned else None,
                    })

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

            repeat_rows.append({
                "case": case_label, "repeat": repeat_idx, "parse_ok": True,
                "n_hallucinated_ids": len(hallucinated_ids),
                "SATISFIED": n_sat, "VIOLATED": n_viol, "UNRESOLVABLE": n_unres,
                "verdict": result["verdict"],
            })
            if hallucinated_ids:
                print(f"  [llm_matching] {len(hallucinated_ids)} activity_id hallucine(s) : {hallucinated_ids}")

    return repeat_rows, milestone_assignments


def build_table_v(milestone_assignments: list[dict]) -> list[dict]:
    """Accord inter-repetitions : pour chaque (cas, milestone), les 3 repetitions donnent-elles
    la MEME assignation (meme activity_id, ou None les 3 fois) ?"""
    by_case_milestone: dict[tuple[str, str], set] = {}
    for row in milestone_assignments:
        key = (row["case"], row["milestone"])
        by_case_milestone.setdefault(key, set()).add(row["assigned_activity"])

    table = []
    by_case: dict[str, list[bool]] = {}
    for (case, milestone), assignments in by_case_milestone.items():
        agree = len(assignments) == 1  # les 3 repetitions ont donne exactement la meme reponse
        by_case.setdefault(case, []).append(agree)

    for case in sorted(by_case):
        flags = by_case[case]
        n = len(flags)
        table.append({
            "case": case, "n_milestones": n,
            "full_agreement_3_3_%": round(100 * sum(flags) / n, 1) if n else 0.0,
        })
    return table


def build_table_w(repeat_rows: list[dict]) -> list[dict]:
    ok_rows = [r for r in repeat_rows if r["parse_ok"]]
    table = []
    for case in sorted({r["case"] for r in repeat_rows}):
        case_rows = [r for r in ok_rows if r["case"] == case]
        n = len(case_rows)
        n_parse_fail = sum(1 for r in repeat_rows if r["case"] == case and not r["parse_ok"])
        avg_halluc = sum(r["n_hallucinated_ids"] for r in case_rows) / n if n else None
        table.append({
            "case": case, "n_repeats_parsed_ok": n, "n_repeats_parse_failed": n_parse_fail,
            "avg_hallucinated_ids_per_repeat": round(avg_halluc, 2) if avg_halluc is not None else None,
        })
    return table


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


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    repeat_rows, milestone_assignments = run_experiment()

    with open(os.path.join(OUT_DIR, "repeat_rows_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(repeat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(repeat_rows)
    with open(os.path.join(OUT_DIR, "milestone_assignments_raw.csv"), "w", newline="") as f:
        if milestone_assignments:
            writer = csv.DictWriter(f, fieldnames=list(milestone_assignments[0].keys()))
            writer.writeheader()
            writer.writerows(milestone_assignments)

    table_v = build_table_v(milestone_assignments)
    table_w = build_table_w(repeat_rows)

    print_table("Table V -- Accord inter-repetitions (3/3 identique) par cas", table_v)
    print_table("Table W -- Taux d'echec de parsing / hallucination d'activity_id par cas", table_w)
    print_table("Table X -- SATISFIED/VIOLATED/UNRESOLVABLE + verdict, une ligne par repetition", repeat_rows)

    with open(os.path.join(OUT_DIR, "table_v.csv"), "w", newline="") as f:
        if table_v:
            writer = csv.DictWriter(f, fieldnames=list(table_v[0].keys()))
            writer.writeheader()
            writer.writerows(table_v)
    with open(os.path.join(OUT_DIR, "table_w.csv"), "w", newline="") as f:
        if table_w:
            writer = csv.DictWriter(f, fieldnames=list(table_w[0].keys()))
            writer.writeheader()
            writer.writerows(table_w)

    print(f"\nResultats dans {OUT_DIR}")