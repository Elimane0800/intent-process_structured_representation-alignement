"""Balayage de CONFIDENCE_THRESHOLD (tau, alignment_node.py) -- MATCHING_THRESHOLD desormais
FIXE a 0.25 (cf. threshold_impact_at_scale.py), seul tau varie ici. Meme discipline
d'isolation d'une seule variable a la fois que pour MATCHING_THRESHOLD.

Difference de nature avec le balayage precedent, a lire avant d'interpreter les resultats :

  - tau n'agit JAMAIS sur SATISFIED (path_exists() est invariant a tau, cf. tgms_solver.py) --
    seulement sur la frontiere VIOLATED <-> UNRESOLVABLE, pour les candidats VIOLATED dont
    l'ancrage est faible. Le plafond de gain possible ici est donc borne par la part
    "low_confidence" de Table I (5.1% de la masse d'UNRESOLVABLE, tres inferieur aux 94.4% de
    "missing" resolus en grande partie par MATCHING_THRESHOLD) -- ne pas attendre un effet de
    la meme ampleur.

  - Le sens de la calibration est probablement INVERSE : le dataset LRE est suppose entierement
    fidele par construction (run.py, docstring de tete) -- tout VIOLATED observe est, par
    hypothese, une FAUSSE accusation du pipeline, pas un vrai ecart. Baisser tau ne peut donc
    QUE degrader (plus de VIOLATED confirmes a tort) ; monter tau devrait faire refluer une
    partie des VIOLATED actuels vers UNRESOLVABLE -- un verdict honnete plutot qu'un faux
    positif. On cherche donc ici la valeur qui fait tendre VIOLATED_% vers 0 sans faire
    exploser UNRESOLVABLE_% au-dela de ce que Table I/J indiquent deja comme structurel.

Comme les balayages precedents : aucun appel LLM, dfg_edges/state_space/reference_graph/
activity_guards deja sur disque (run_2), embeddings calcules UNE SEULE FOIS par cas. Ici,
MATCHING_THRESHOLD etant fixe, matches/process_graph sont eux-memes calcules UNE SEULE FOIS par
cas (independants de tau) -- seul check_alignment() est rappele pour chaque valeur de tau
testee, l'operation la moins couteuse de toute la chaine."""

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
from agent.nodes.alignment_node import check_alignment
from agent.nodes.tgms_solver import tgms_from_pipeline, _strip_not
from agent.nodes.report_node import collect_failures, cluster_by_root, status_for_root, compute_verdict
from agent.models.base_nvidia_embedding import embed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "tau_sweep")

RUN_ID = "run_2"
MATCHING_THRESHOLD_FIXED = 0.25  # nouveau defaut -- fixe pour ce balayage, cf. docstring de tete

# Grille testee, incluant 0.35 (valeur actuelle, jamais recalibree jusqu'ici) et des valeurs
# plus HAUTES (plus conservatrices) -- contrairement au balayage MATCHING_THRESHOLD qui
# explorait vers le bas, cf. justification du sens inverse dans la docstring de tete.
TAU_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# --- Duplique depuis les scripts precedents -- meme discipline de decouplage. ---

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


def compute_verdict_only(alignment: dict) -> dict:
    """Identique a threshold_impact_at_scale.py::compute_verdict_only."""
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


def run_sweep() -> tuple[list[dict], list[dict]]:
    """Retourne (term_rows, confidence_rows). term_rows : une ligne par (cas, tau) avec les
    comptes de termes/verdict. confidence_rows : une ligne par terme VIOLATED-candidat (a
    tau=0.0, aucune demotion possible -- la classification la plus permissive, donc la plus
    proche du statut "brut" avant tout seuillage), avec le score de confiance minimal entre le
    terme et sa cible -- pour Table M, la distribution independante de tout choix de tau."""
    by_model = discover_run2_cases(RESULTS_DIR, RUN_ID)
    term_rows = []
    confidence_rows = []

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

            # MATCHING_THRESHOLD_FIXED etant fixe, matches/process_graph sont calcules UNE
            # SEULE FOIS par cas ici -- seul tau varie dans la boucle ci-dessous.
            matches = matches_at_threshold(
                activity_ids, contextualized, activity_vecs, vector_by_candidate,
                candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED,
            )
            process_graph = build_process_state_graph(dfg_edges, matches)
            case_label = f"{subset}/{case_name}"

            for tau in TAU_VALUES:
                alignment = check_alignment(
                    reference_graph, process_graph, matches,
                    confidence_threshold=tau, activity_guards=activity_guards,
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
                term_rows.append({
                    "model": model, "case": case_label, "tau": tau,
                    "SATISFIED": n_sat, "VIOLATED": n_viol, "UNRESOLVABLE": n_unres,
                    "verdict": result["verdict"],
                })

            # Table M : classification a tau=0.0 (aucune demotion possible, cf. docstring),
            # confiance minimale (terme, cible) pour chaque VIOLATED-candidat.
            tgms, _ = tgms_from_pipeline(reference_graph, process_graph, matches, activity_guards)
            alignment_baseline = check_alignment(
                reference_graph, process_graph, matches,
                confidence_threshold=0.0, activity_guards=activity_guards,
            )
            for target, entry in alignment_baseline.items():
                for t in entry.get("terms", []):
                    if t["status"] != "VIOLATED":
                        continue
                    raw_term = _strip_not(t["term"])
                    score_term = tgms.confidence.get(raw_term, 0.0)
                    score_target = tgms.confidence.get(target, 0.0)
                    confidence_rows.append({
                        "model": model, "case": case_label, "target": target,
                        "term": t["term"], "min_confidence": round(min(score_term, score_target), 4),
                    })

    return term_rows, confidence_rows


def build_table_k(term_rows: list[dict]) -> list[dict]:
    """Comptes de termes par tau, agrege ALL MODELS."""
    table = []
    for tau in TAU_VALUES:
        rows = [r for r in term_rows if r["tau"] == tau]
        n = sum(r["SATISFIED"] + r["VIOLATED"] + r["UNRESOLVABLE"] for r in rows)
        n_sat = sum(r["SATISFIED"] for r in rows)
        n_viol = sum(r["VIOLATED"] for r in rows)
        n_unres = sum(r["UNRESOLVABLE"] for r in rows)
        table.append({
            "tau": tau, "n_terms": n,
            "SATISFIED_%": round(100 * n_sat / n, 1) if n else 0.0,
            "VIOLATED_%": round(100 * n_viol / n, 1) if n else 0.0,
            "UNRESOLVABLE_%": round(100 * n_unres / n, 1) if n else 0.0,
        })
    return table


def build_table_l(term_rows: list[dict]) -> list[dict]:
    """Taux de verdict (FAITHFUL/FAITHFUL_WITH_RESERVATIONS/NOT_FAITHFUL) par tau, agrege ALL
    MODELS -- une ligne par (case, tau) dans term_rows, mais chaque (model, case) apparait UNE
    fois par tau (pas par terme), donc pas besoin de deduplication supplementaire ici."""
    table = []
    for tau in TAU_VALUES:
        verdicts = [r["verdict"] for r in term_rows if r["tau"] == tau]
        # NB : term_rows contient une ligne par (model, case, tau), on ne veut qu'UNE ligne par
        # (model, case) pour ce tau -- deja le cas ici puisque la boucle source ecrit exactement
        # une entree par (model, case, tau).
        n = len(verdicts)
        table.append({
            "tau": tau, "n_cases": n,
            "FAITHFUL_%": round(100 * verdicts.count("FAITHFUL") / n, 1) if n else 0.0,
            "FAITHFUL_WITH_RESERVATIONS_%": round(100 * verdicts.count("FAITHFUL_WITH_RESERVATIONS") / n, 1) if n else 0.0,
            "NOT_FAITHFUL_%": round(100 * verdicts.count("NOT_FAITHFUL") / n, 1) if n else 0.0,
        })
    return table


def build_table_m(confidence_rows: list[dict], bin_width: float = 0.05) -> list[dict]:
    """Histogramme brut des scores de confiance minimaux parmi les VIOLATED-candidats (a
    tau=0.0) -- independant de toute valeur de tau testee dans TAU_VALUES, pour reperer un
    point de coupure naturel plutot que de deviner entre les valeurs de la grille."""
    if not confidence_rows:
        return []
    max_score = max(r["min_confidence"] for r in confidence_rows)
    n_bins = int(max_score / bin_width) + 1
    table = []
    for i in range(n_bins):
        low, high = i * bin_width, (i + 1) * bin_width
        count = sum(1 for r in confidence_rows if low <= r["min_confidence"] < high)
        table.append({"bin": f"[{low:.2f}, {high:.2f})", "count": count})
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

    term_rows, confidence_rows = run_sweep()
    if not term_rows:
        raise SystemExit(f"[tau_sweep] aucune donnee produite -- verifie {RESULTS_DIR} et RUN_ID={RUN_ID!r}")

    with open(os.path.join(OUT_DIR, "term_rows_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(term_rows[0].keys()))
        writer.writeheader()
        writer.writerows(term_rows)
    with open(os.path.join(OUT_DIR, "confidence_rows_raw.csv"), "w", newline="") as f:
        if confidence_rows:
            writer = csv.DictWriter(f, fieldnames=list(confidence_rows[0].keys()))
            writer.writeheader()
            writer.writerows(confidence_rows)

    table_k = build_table_k(term_rows)
    table_l = build_table_l(term_rows)
    table_m = build_table_m(confidence_rows)

    print_table("Table K -- Comptes de termes par tau (ALL MODELS)", table_k)
    print_table("Table L -- Taux de verdict par tau (ALL MODELS)", table_l)
    print_table("Table M -- Distribution des scores de confiance (VIOLATED-candidats, tau=0.0)", table_m)

    for name, table in [("table_k", table_k), ("table_l", table_l), ("table_m", table_m)]:
        with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="") as f:
            if table:
                writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
                writer.writeheader()
                writer.writerows(table)

    print(f"\nResultats dans {OUT_DIR}")