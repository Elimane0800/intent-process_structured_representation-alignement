"""Table I + Table J -- decomposition fine d'UNRESOLVABLE, sur TOUT le dataset deja tourne, au
seuil MATCHING_THRESHOLD desormais fixe (0.25, cf. threshold_impact_at_scale.py). Repond aux
deux questions laissees ouvertes avant qu'on ne s'interrompe pour regler le seuil :

  Table I : quand un terme est UNRESOLVABLE, POURQUOI -- aucun ancrage du tout (missing),
    ancre mais sous le seuil de confiance (low_confidence), retrograde par l'analyse
    structurelle (activity_guards, marqueur deterministe), ou -- specifique aux termes NOT --
    aucun co-terme positif ancre dans la meme clause pour servir de point de depart au test
    d'atteignabilite (status_of_negated_term). Quatre causes distinctes, quatre remedes
    distincts -- les confondre dans un seul "non_verifiable" (comme le fait Table C
    d'analyze_lre_results.py) empeche de savoir ou agir.

  Table J : le taux SATISFIED/VIOLATED/UNRESOLVABLE d'une CLAUSE (pas d'un terme), croise avec
    son ARITE (nombre de termes AND qu'elle contient). Teste l'hypothese d'amplification par
    _combine(..., "AND") (tgms_solver.py) : UNRESOLVABLE domine des qu'UN SEUL terme de la
    clause l'est, meme si tous les autres sont SATISFIED -- si cette regle amplifie
    artificiellement le taux d'UNRESOLVABLE observe, le taux doit croitre avec l'arite.

Aucun appel LLM. Meme reutilisation qu'threshold_impact_at_scale.py (dfg_edges/state_space/
reference_graph/activity_guards deja sur disque, run_2 ; embeddings calcules une fois par cas) --
cf. sa docstring pour la justification complete. check_alignment() expose deja "clauses" par
cible (v3, additif, cf. AlignmentEntry dans state.py) -- Table J n'a besoin d'aucun recalcul
depuis reference_graph, seulement de lire ce champ."""

import csv
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.nodes.state_matching_node import (
    _build_contextualized_activities,
    _entity_states,
    _cluster_matches_by_similarity,
    render_candidate,
    cosine,
    build_process_state_graph,
    CLUSTER_SIMILARITY_THRESHOLD,
)
from agent.nodes.alignment_node import check_alignment, CONFIDENCE_THRESHOLD
from agent.models.base_nvidia_embedding import embed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "unresolvable_decomposition")

RUN_ID = "run_2"
MATCHING_THRESHOLD_FIXED = 0.25  # nouveau defaut, cf. threshold_impact_at_scale.py
ALIGNMENT_TAU = CONFIDENCE_THRESHOLD

# Meme marqueur litteral que analyze_lre_results.py -- genere par notre propre code
# (status_of_negated_term, tgms_solver.py), jamais du texte libre.
STRUCTURAL_DEMOTION_MARKER = "independent structural guard analysis"
# Genere par status_of_negated_term quand aucun co-terme positif ancre n'existe dans la clause --
# litteral egalement, meme fiabilite.
NO_ANCHORED_CO_TERM_MARKER = "no anchored positive co-term"


# --- Duplique depuis threshold_impact_at_scale.py -- meme discipline de decouplage assumee
# partout ailleurs dans ce projet (chaque script autonome). ---

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


def discover_run2_cases(results_dir: str, run_id: str) -> dict:
    results: dict[str, list] = {}
    for model in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for fname in sorted(os.listdir(model_dir)):
            parsed = parse_filename(fname)
            if parsed is None:
                continue
            subset, case_name, found_run_id = parsed
            if found_run_id != run_id:
                continue
            results.setdefault(model, []).append((subset, case_name, os.path.join(model_dir, fname)))
    return results


def precompute_embeddings(dfg_edges: list[dict], state_space: dict):
    contextualized = _build_contextualized_activities(dfg_edges)
    valid_contextualized = {k: v for k, v in contextualized.items() if v["context_text"].strip()}

    candidates = [f"{e}.{s}" for e, value in state_space.items() for s in _entity_states(value)]
    candidate_texts = [render_candidate(*c.split(".", 1)) for c in candidates]

    activity_ids = list(valid_contextualized.keys())
    activity_texts = [
        " ".join(valid_contextualized[aid]["context_text"].split()) for aid in activity_ids
    ]

    if not activity_texts or not candidate_texts:
        return activity_ids, valid_contextualized, [], {}, candidates, []

    import numpy as np
    activity_vecs = np.array(embed(activity_texts, input_type="query"))
    candidate_vecs = np.array(embed(candidate_texts, input_type="query"))
    vector_by_candidate = dict(zip(candidates, candidate_vecs))

    return activity_ids, valid_contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs


def matches_at_threshold(
    activity_ids, contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs,
    threshold: float, cluster_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
) -> dict:
    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        if not scores:
            matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": [], "clusters": []}
            continue

        best_score = max(scores)
        valid_matches = []
        for idx, score in enumerate(scores):
            if score >= threshold and score >= (best_score - 0.05):
                valid_matches.append({"match": candidates[idx], "score": round(float(score), 4)})
        valid_matches.sort(key=lambda x: x["score"], reverse=True)

        clusters = _cluster_matches_by_similarity(valid_matches, vector_by_candidate, cluster_threshold)
        clusters.sort(key=lambda cluster: max(m["score"] for m in cluster), reverse=True)

        matches[aid] = {
            "activity_label": contextualized[aid]["label"],
            "matches": valid_matches,
            "clusters": [[m["match"] for m in cluster] for cluster in clusters],
        }
    return matches


# --- Nouveau pour ce script : classification de cause + agregation par arite ---

def classify_unresolvable_term(term_result: dict) -> str:
    """Une des quatre causes distinctes -- cf. docstring de tete. Ordre de test volontaire :
    missing d'abord (la cause la plus en amont possible dans status_of_term/status_of_negated_
    term -- si elle est presente, aucune des trois autres n'a pu etre evaluee, cf. leur
    structure en cascade dans tgms_solver.py)."""
    if term_result["status"] != "UNRESOLVABLE":
        return "N/A"
    if term_result.get("missing"):
        return "missing"
    if term_result.get("low_confidence"):
        return "low_confidence"
    reason = term_result.get("reason") or ""
    if STRUCTURAL_DEMOTION_MARKER in reason:
        return "structural_marker"
    if NO_ANCHORED_CO_TERM_MARKER in reason:
        return "no_anchored_co_term"
    return "other_unclassified"  # ne devrait pas arriver -- signal a investiguer si non vide


def run_decomposition() -> tuple[list[dict], list[dict]]:
    """Retourne (term_rows, clause_rows) -- une ligne par terme UNRESOLVABLE pour Table I, une
    ligne par clause (toutes, pas seulement UNRESOLVABLE) pour Table J."""
    by_model = discover_run2_cases(RESULTS_DIR, RUN_ID)
    term_rows = []
    clause_rows = []

    for model in sorted(by_model):
        for subset, case_name, filepath in by_model[model]:
            try:
                with open(filepath) as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if "fatal_error" in state:
                continue
            if any(k not in state for k in ("dfg_edges", "state_space", "reference_graph")):
                continue

            dfg_edges = state["dfg_edges"]
            state_space = state["state_space"]
            reference_graph = state["reference_graph"]
            activity_guards = state.get("activity_guards")

            precomputed = precompute_embeddings(dfg_edges, state_space)
            activity_ids, contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs = precomputed
            if not activity_ids or len(candidate_vecs) == 0:
                continue

            matches = matches_at_threshold(
                activity_ids, contextualized, activity_vecs, vector_by_candidate,
                candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED,
            )
            process_graph = build_process_state_graph(dfg_edges, matches)
            alignment = check_alignment(
                reference_graph, process_graph, matches,
                confidence_threshold=ALIGNMENT_TAU, activity_guards=activity_guards,
            )

            case_label = f"{subset}/{case_name}"
            for target, entry in alignment.items():
                # Table I -- un terme a la fois, toutes cibles confondues
                for term_result in entry.get("terms", []):
                    if term_result["status"] != "UNRESOLVABLE":
                        continue
                    term_rows.append({
                        "model": model, "case": case_label, "target": target,
                        "term": term_result["term"],
                        "negated": term_result["term"].startswith("NOT "),
                        "cause": classify_unresolvable_term(term_result),
                    })

                # Table J -- une clause a la fois (structure DNF explicite, additive depuis v3)
                for clause in entry.get("clauses", []):
                    clause_rows.append({
                        "model": model, "case": case_label, "target": target,
                        "arity": len(clause["terms"]),
                        "status": clause["status"],
                    })

    return term_rows, clause_rows


def build_table_i(term_rows: list[dict]) -> list[dict]:
    """Global + par modele, part de chaque cause parmi les UNRESOLVABLE."""
    causes = ["missing", "low_confidence", "structural_marker", "no_anchored_co_term", "other_unclassified"]
    table = []
    for scope, rows in [("ALL MODELS", term_rows)] + [
        (m, [r for r in term_rows if r["model"] == m]) for m in sorted({r["model"] for r in term_rows})
    ]:
        n = len(rows)
        row = {"model": scope, "n_unresolvable_terms": n}
        for cause in causes:
            count = sum(1 for r in rows if r["cause"] == cause)
            row[f"{cause}_%"] = round(100 * count / n, 1) if n else 0.0
        table.append(row)
    return table


def build_table_j(clause_rows: list[dict]) -> list[dict]:
    """Taux SATISFIED/VIOLATED/UNRESOLVABLE par arite de clause, toutes cibles/modeles/cas
    confondus -- teste l'amplification par _combine(..., "AND")."""
    arities = sorted({r["arity"] for r in clause_rows})
    table = []
    for arity in arities:
        rows = [r for r in clause_rows if r["arity"] == arity]
        n = len(rows)
        table.append({
            "arity": arity, "n_clauses": n,
            "SATISFIED_%": round(100 * sum(1 for r in rows if r["status"] == "SATISFIED") / n, 1) if n else 0.0,
            "VIOLATED_%": round(100 * sum(1 for r in rows if r["status"] == "VIOLATED") / n, 1) if n else 0.0,
            "UNRESOLVABLE_%": round(100 * sum(1 for r in rows if r["status"] == "UNRESOLVABLE") / n, 1) if n else 0.0,
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

    term_rows, clause_rows = run_decomposition()
    if not term_rows and not clause_rows:
        raise SystemExit(f"[decomp] aucune donnee produite -- verifie {RESULTS_DIR} et RUN_ID={RUN_ID!r}")

    with open(os.path.join(OUT_DIR, "table_i_raw_terms.csv"), "w", newline="") as f:
        if term_rows:
            writer = csv.DictWriter(f, fieldnames=list(term_rows[0].keys()))
            writer.writeheader()
            writer.writerows(term_rows)

    with open(os.path.join(OUT_DIR, "table_j_raw_clauses.csv"), "w", newline="") as f:
        if clause_rows:
            writer = csv.DictWriter(f, fieldnames=list(clause_rows[0].keys()))
            writer.writeheader()
            writer.writerows(clause_rows)

    table_i = build_table_i(term_rows)
    table_j = build_table_j(clause_rows)

    print_table(f"Table I -- Decomposition des causes d'UNRESOLVABLE (threshold={MATCHING_THRESHOLD_FIXED})", table_i)
    print_table(f"Table J -- Taux de verdict par arite de clause AND (threshold={MATCHING_THRESHOLD_FIXED})", table_j)

    with open(os.path.join(OUT_DIR, "table_i_summary.csv"), "w", newline="") as f:
        if table_i:
            writer = csv.DictWriter(f, fieldnames=list(table_i[0].keys()))
            writer.writeheader()
            writer.writerows(table_i)
    with open(os.path.join(OUT_DIR, "table_j_summary.csv"), "w", newline="") as f:
        if table_j:
            writer = csv.DictWriter(f, fieldnames=list(table_j[0].keys()))
            writer.writeheader()
            writer.writerows(table_j)

    print(f"\nResultats dans {OUT_DIR}")