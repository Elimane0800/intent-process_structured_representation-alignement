"""Balayage de la MARGE top-k (actuellement 0.05, magic number code en dur dans
state_matching_node.py::match_activities_to_states -- 'score >= best_score - 0.05') --
MATCHING_THRESHOLD desormais FIXE a 0.25 (cf. threshold_impact_at_scale.py), seule la marge
varie ici.

Motive directement par node_type_audit.py (Table O) : parmi les milestones MISSING a
MATCHING_THRESHOLD=0.25, 63.7% (128/201) ne manquent pas de signal absolu (score moyen 0.38,
nettement au-dessus du seuil) -- ils perdent face a un CONCURRENT sur la MEME activite, dans la
marge de 0.05. Contrairement a MATCHING_THRESHOLD (seuil ABSOLU, deja calibre) et
CONFIDENCE_THRESHOLD/tau (deja calibre, effet marginal confirme), cette marge n'a jamais eu de
nom, jamais ete exposee comme constante nommee, jamais mesuree -- meme categorie de defaut que
les deux precedents avant leur calibration.

Risque a surveiller specifiquement ici, absent des deux balayages precedents : elargir la marge
ne peut QUE accepter plus de candidats par activite (jamais en retirer), donc ne peut jamais
degrader SATISFIED -- MAIS peut degrader la PRECISION du matching en acceptant des candidats
moins pertinents comme co-ancrages d'une meme activite (bruit dans build_process_state_graph,
qui cree une arete process pour CHAQUE paire (etat source, etat cible) ancree -- plus de matches
par activite = plus d'aretes, dont certaines potentiellement fausses). A verifier via VIOLATED_%
(devrait rester stable ou baisser, comme pour MATCHING_THRESHOLD) ET via le NOMBRE MOYEN de
matches par activite (devrait rester raisonnable, pas exploser).

Aucun appel LLM, meme reutilisation qu'ailleurs (embeddings calcules une fois par cas)."""

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
from agent.nodes.report_node import collect_failures, cluster_by_root, status_for_root, compute_verdict
from agent.models.base_nvidia_embedding import embed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "margin_sweep")

RUN_ID = "run_2"
MATCHING_THRESHOLD_FIXED = 0.25  # inchange -- seule la marge varie ici
ALIGNMENT_TAU = CONFIDENCE_THRESHOLD

# 0.05 = valeur actuelle du pipeline (magic number, jamais nomme ni calibre, cf. docstring).
# Testee jusqu'a 0.30 -- au-dela, le risque de bruit (co-ancrages non pertinents) devient
# probablement dominant, pas la peine d'aller plus loin sans d'abord voir la tendance.
MARGIN_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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


def matches_at_margin(
    activity_ids, contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs,
    threshold: float, margin: float, cluster_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
) -> dict:
    """Identique a matches_at_threshold() des autres scripts, sauf que la marge (0.05 en dur
    partout ailleurs) est ici un PARAMETRE -- seule difference avec le reste du code, ligne
    'score >= (best_score - margin)'."""
    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        if not scores:
            matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": [], "clusters": []}
            continue

        best_score = max(scores)
        valid_matches = []
        for idx, score in enumerate(scores):
            if score >= threshold and score >= (best_score - margin):
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


def run_sweep() -> list[dict]:
    by_model = discover_run2_cases(RESULTS_DIR, RUN_ID)
    rows = []

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
            case_label = f"{subset}/{case_name}"

            precomputed = precompute_embeddings(dfg_edges, state_space)
            activity_ids, contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs = precomputed
            if not activity_ids or len(candidate_vecs) == 0:
                continue

            for margin in MARGIN_VALUES:
                matches = matches_at_margin(
                    activity_ids, contextualized, activity_vecs, vector_by_candidate,
                    candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED, margin,
                )
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

                # Nombre moyen de matches retenus par activite -- signal de bruit potentiel
                # (cf. docstring de tete) : une marge trop large gonfle ce nombre sans que ce
                # soit necessairement du signal utile.
                n_activities = len(matches)
                avg_matches_per_activity = (
                    sum(len(m["matches"]) for m in matches.values()) / n_activities if n_activities else 0.0
                )

                rows.append({
                    "model": model, "case": case_label, "margin": margin,
                    "SATISFIED": n_sat, "VIOLATED": n_viol, "UNRESOLVABLE": n_unres,
                    "verdict": result["verdict"],
                    "avg_matches_per_activity": round(avg_matches_per_activity, 3),
                })

    return rows


def build_table_p(rows: list[dict]) -> list[dict]:
    """Comptes de termes par marge, agrege ALL MODELS -- meme forme que Table K (tau_sweep.py)."""
    table = []
    for margin in MARGIN_VALUES:
        subset = [r for r in rows if r["margin"] == margin]
        n = sum(r["SATISFIED"] + r["VIOLATED"] + r["UNRESOLVABLE"] for r in subset)
        n_sat = sum(r["SATISFIED"] for r in subset)
        n_viol = sum(r["VIOLATED"] for r in subset)
        n_unres = sum(r["UNRESOLVABLE"] for r in subset)
        avg_matches = sum(r["avg_matches_per_activity"] for r in subset) / len(subset) if subset else 0.0
        table.append({
            "margin": margin, "n_terms": n,
            "SATISFIED_%": round(100 * n_sat / n, 1) if n else 0.0,
            "VIOLATED_%": round(100 * n_viol / n, 1) if n else 0.0,
            "UNRESOLVABLE_%": round(100 * n_unres / n, 1) if n else 0.0,
            "avg_matches_per_activity": round(avg_matches, 3),
        })
    return table


def build_table_q(rows: list[dict]) -> list[dict]:
    """Taux de verdict par marge, agrege ALL MODELS -- meme forme que Table L (tau_sweep.py)."""
    table = []
    for margin in MARGIN_VALUES:
        verdicts = [r["verdict"] for r in rows if r["margin"] == margin]
        n = len(verdicts)
        table.append({
            "margin": margin, "n_cases": n,
            "FAITHFUL_%": round(100 * verdicts.count("FAITHFUL") / n, 1) if n else 0.0,
            "FAITHFUL_WITH_RESERVATIONS_%": round(100 * verdicts.count("FAITHFUL_WITH_RESERVATIONS") / n, 1) if n else 0.0,
            "NOT_FAITHFUL_%": round(100 * verdicts.count("NOT_FAITHFUL") / n, 1) if n else 0.0,
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

    rows = run_sweep()
    if not rows:
        raise SystemExit(f"[margin_sweep] aucune donnee produite -- verifie {RESULTS_DIR} et RUN_ID={RUN_ID!r}")

    with open(os.path.join(OUT_DIR, "margin_rows_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    table_p = build_table_p(rows)
    table_q = build_table_q(rows)

    print_table("Table P -- Comptes de termes par marge top-k (ALL MODELS)", table_p)
    print_table("Table Q -- Taux de verdict par marge top-k (ALL MODELS)", table_q)

    for name, table in [("table_p", table_p), ("table_q", table_q)]:
        with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="") as f:
            if table:
                writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
                writer.writeheader()
                writer.writerows(table)

    print(f"\nResultats dans {OUT_DIR}")