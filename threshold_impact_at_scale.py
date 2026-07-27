"""Mesure l'effet de MATCHING_THRESHOLD (state_matching_node.py) sur le TAUX DE VERDICTS
(FAITHFUL / FAITHFUL_WITH_RESERVATIONS / NOT_FAITHFUL), a l'echelle de TOUT le dataset deja
tourne -- pas seulement les 3 cas d'ablation de threshold_sweep.py. Repond directement a la
question laissee ouverte par cette ablation : le motif observe sur BicycleManufacturing (un
plateau atteint vers 0.25, aucun gain au-dela) se generalise-t-il, et de combien deplace-t-il le
taux de NOT_FAITHFUL/FAITHFUL_WITH_RESERVATIONS a l'echelle du dataset complet ?

Aucun appel LLM : le verdict (report_node.py::compute_verdict) est une fonction PURE de
{ecarts, non_verifiable, conforme} -- des comptes de CAUSES RACINES (collect_failures +
cluster_by_root + status_for_root), pas des comptes de termes bruts. Seule la generation de
l'explication en langage metier (generate_cluster_explanation) appelle un LLM, et elle n'est pas
necessaire pour mesurer un taux de verdict -- reproduite ici a l'identique (memes fonctions
importees depuis report_node.py, jamais reimplementees a cote, pour ne jamais diverger du
comportement reel du pipeline sur CETTE partie).

Meme principe de reutilisation qu'threshold_sweep.py sur le reste : dfg_edges/state_space/
reference_graph/activity_guards deja calcules et sauvegardes (run_2), embeddings calcules UNE
SEULE FOIS par cas puis reutilises pour chaque valeur de seuil testee -- cf. sa docstring de
tete pour la justification complete (evite de reintroduire un appel LLM generatif OU une
variation d'embedding comme facteur confondu avec l'effet du seuil)."""

import csv
import json
import os
import re

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
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "threshold_impact_at_scale")

RUN_ID = "run_2"  # source des dfg_edges/state_space/reference_graph/activity_guards deja calcules

# Valeurs de seuil comparees -- 0.35 = valeur actuelle du pipeline (MATCHING_THRESHOLD dans
# state_matching_node.py), 0.25 = valeur candidate identifiee par l'ablation sur
# BicycleManufacturing (plateau atteint, aucun gain constate au-dela sur les 3 modeles testes).
# Ajouter d'autres valeurs ici si besoin d'une comparaison plus fine.
THRESHOLDS_TO_COMPARE = [0.35, 0.25]

ALIGNMENT_TAU = CONFIDENCE_THRESHOLD  # fixe -- seul MATCHING_THRESHOLD varie, cf. discussion


def parse_filename(filename: str):
    """Copie de analyze_lre_results.py::parse_filename -- meme discipline de duplication que
    partout ailleurs dans ce projet (chaque script reste autonome, decouple des autres)."""
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
    """{model: [(subset, case_name, filepath), ...]} -- restreint au run_id demande, contrairement
    a analyze_lre_results.discover_results() qui garde tous les run_id (celui-ci n'a besoin que
    d'UN run source pour les etages amont, peu importe combien de run_id existent par ailleurs)."""
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
    """Identique a threshold_sweep.py::precompute_embeddings -- cf. sa docstring pour la
    justification (calcul unique, reutilise pour chaque seuil, jamais recalcule)."""
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
    """Identique a threshold_sweep.py::matches_at_threshold -- meme reimplementation fidele de
    la regle d'acceptation de match_activities_to_states(), cf. sa docstring."""
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


def compute_verdict_only(alignment: dict) -> dict:
    """Reproduit EXACTEMENT la portion de report_node.build_report() qui calcule le verdict --
    memes fonctions importees (collect_failures/cluster_by_root/status_for_root/compute_verdict),
    jamais reimplementees a cote. Saute uniquement generate_cluster_explanation (LLM, non
    necessaire pour un taux). 'conforme' comme dans build_report : un compte de TERMES
    individuels SATISFIED, jamais clusterise par racine (seuls les echecs le sont, cf.
    build_report -- 'satisfied' y est une liste plate, pas un dict de clusters)."""
    failures = collect_failures(alignment)
    clusters = cluster_by_root(failures)

    n_violated_roots = 0
    n_unresolvable_roots = 0
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


def run_at_scale() -> list[dict]:
    """Une ligne par (model, subset, case_name, threshold) -- verdict recalcule pour chaque
    seuil teste, a partir des MEMES etages amont (dfg_edges/state_space/reference_graph/
    activity_guards deja sur disque, run_2) et des MEMES embeddings (calcules une fois par
    cas)."""
    by_model = discover_run2_cases(RESULTS_DIR, RUN_ID)
    rows = []

    for model in sorted(by_model):
        for subset, case_name, filepath in by_model[model]:
            try:
                with open(filepath) as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [scale] {model}/{subset}/{case_name} -- fichier illisible ({e}), ignore")
                continue

            if "fatal_error" in state:
                continue  # echec definitif du run source, rien a reutiliser
            missing_fields = [k for k in ("dfg_edges", "state_space", "reference_graph") if k not in state]
            if missing_fields:
                print(f"  [scale] {model}/{subset}/{case_name} -- champs manquants "
                      f"{missing_fields} (run incomplet), ignore")
                continue

            dfg_edges = state["dfg_edges"]
            state_space = state["state_space"]
            reference_graph = state["reference_graph"]
            activity_guards = state.get("activity_guards")

            precomputed = precompute_embeddings(dfg_edges, state_space)
            activity_ids, contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs = precomputed
            if not activity_ids or len(candidate_vecs) == 0:
                print(f"  [scale] {model}/{subset}/{case_name} -- pas assez de donnees, ignore")
                continue

            for threshold in THRESHOLDS_TO_COMPARE:
                matches = matches_at_threshold(
                    activity_ids, contextualized, activity_vecs, vector_by_candidate,
                    candidates, candidate_vecs, threshold,
                )
                process_graph = build_process_state_graph(dfg_edges, matches)
                alignment = check_alignment(
                    reference_graph, process_graph, matches,
                    confidence_threshold=ALIGNMENT_TAU, activity_guards=activity_guards,
                )
                result = compute_verdict_only(alignment)

                rows.append({
                    "model": model, "case": f"{subset}/{case_name}", "threshold": threshold,
                    "verdict": result["verdict"],
                    "ecarts": result["summary"]["ecarts"],
                    "non_verifiable": result["summary"]["non_verifiable"],
                    "conforme": result["summary"]["conforme"],
                })

    return rows


def build_rate_table(rows: list[dict], threshold: float) -> list[dict]:
    """Table B-equivalent (analyze_lre_results.py) pour UNE valeur de seuil -- meme forme, pour
    comparaison directe avec les tables deja produites par ailleurs."""
    by_model: dict[str, list[str]] = {}
    for r in rows:
        if r["threshold"] != threshold:
            continue
        by_model.setdefault(r["model"], []).append(r["verdict"])

    table = []
    all_verdicts = []
    for model in sorted(by_model):
        verdicts = by_model[model]
        n = len(verdicts)
        all_verdicts.extend(verdicts)
        table.append({
            "model": model, "n": n,
            "FAITHFUL_%": round(100 * verdicts.count("FAITHFUL") / n, 1) if n else 0.0,
            "FAITHFUL_WITH_RESERVATIONS_%": round(100 * verdicts.count("FAITHFUL_WITH_RESERVATIONS") / n, 1) if n else 0.0,
            "NOT_FAITHFUL_%": round(100 * verdicts.count("NOT_FAITHFUL") / n, 1) if n else 0.0,
        })
    if all_verdicts:
        n = len(all_verdicts)
        table.append({
            "model": "ALL MODELS", "n": n,
            "FAITHFUL_%": round(100 * all_verdicts.count("FAITHFUL") / n, 1),
            "FAITHFUL_WITH_RESERVATIONS_%": round(100 * all_verdicts.count("FAITHFUL_WITH_RESERVATIONS") / n, 1),
            "NOT_FAITHFUL_%": round(100 * all_verdicts.count("NOT_FAITHFUL") / n, 1),
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

    rows = run_at_scale()
    if not rows:
        raise SystemExit(f"[scale] aucune ligne produite -- verifie {RESULTS_DIR} et RUN_ID={RUN_ID!r}")

    out_path = os.path.join(OUT_DIR, "threshold_impact_raw.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV brut ecrit : {out_path} ({len(rows)} lignes)")

    for threshold in THRESHOLDS_TO_COMPARE:
        table = build_rate_table(rows, threshold)
        print_table(f"Table B @ threshold={threshold}", table)
        write_path = os.path.join(OUT_DIR, f"table_b_threshold_{str(threshold).replace('.', '')}.csv")
        with open(write_path, "w", newline="") as f:
            if table:
                writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
                writer.writeheader()
                writer.writerows(table)

    # Delta direct, modele par modele, entre le premier et le dernier seuil de la liste --
    # le resume le plus lisible pour repondre a "le taux d'UNRESOLVABLE (FAITHFUL_WITH_
    # RESERVATIONS + NOT_FAITHFUL) change-t-il, oui ou non".
    t_before, t_after = THRESHOLDS_TO_COMPARE[0], THRESHOLDS_TO_COMPARE[-1]
    table_before = {r["model"]: r for r in build_rate_table(rows, t_before)}
    table_after = {r["model"]: r for r in build_rate_table(rows, t_after)}
    delta_rows = []
    for model in sorted(set(table_before) | set(table_after)):
        b = table_before.get(model, {})
        a = table_after.get(model, {})
        delta_rows.append({
            "model": model,
            f"NOT_FAITHFUL_%_@{t_before}": b.get("NOT_FAITHFUL_%"),
            f"NOT_FAITHFUL_%_@{t_after}": a.get("NOT_FAITHFUL_%"),
            f"FAITHFUL_WITH_RESERVATIONS_%_@{t_before}": b.get("FAITHFUL_WITH_RESERVATIONS_%"),
            f"FAITHFUL_WITH_RESERVATIONS_%_@{t_after}": a.get("FAITHFUL_WITH_RESERVATIONS_%"),
            f"FAITHFUL_%_@{t_before}": b.get("FAITHFUL_%"),
            f"FAITHFUL_%_@{t_after}": a.get("FAITHFUL_%"),
        })
    print_table(f"Table Delta -- {t_before} vs {t_after}", delta_rows)
    with open(os.path.join(OUT_DIR, "table_delta.csv"), "w", newline="") as f:
        if delta_rows:
            writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(delta_rows)

    print(f"\nResultats dans {OUT_DIR}")