"""Compare la regle ACTUELLE (marge locale de 0.05, decision par activite) a une regle
d'ASSIGNATION GLOBALE : pour chaque milestone (candidat), on prend son ARGMAX sur TOUTES les
activites du cas (pas seulement une comparaison locale a l'interieur d'une activite), et on le
retient uniquement sur cette activite-la, si le score depasse MATCHING_THRESHOLD.

Ce n'est PAS un appariement biparti optimal au sens strict (pas de contrainte de capacite type
Hungarian) -- inutile ici, puisqu'une activite peut legitimement porter plusieurs faits distincts
(Categorie 2, deja etablie : storehouse.informed + engineering_department.informed sur la meme
tache). La regle est simplement : chaque CANDIDAT resout son propre argmax INDEPENDAMMENT des
autres candidats, sur l'ensemble des activites -- contrairement a la regle actuelle qui resout
un argmax PAR ACTIVITE (quels candidats gagnent la marge locale de CETTE activite). L'inversion
du sens de l'argmax (colonne plutot que ligne de la matrice score[activite][candidat]) elimine
par construction la contamination croisee diagnostiquee sur ClaimsCreation
(diagnose_claimscreation.py) : un milestone ne peut plus etre retenu sur une activite voisine ou
il a un score correct en absolu MAIS PAS le meilleur, puisqu'il ne considere qu'une seule
activite -- la sienne, la meilleure, jamais un choix local sous-optimal.

Risque a verifier explicitement (pas suppose regle) : cette regle peut-elle re-degrader la
Categorie 1 (paraphrases, ex. cart.readied/cart.ready) ? Non par construction -- les deux
candidats paraphrases trouvent chacun independamment la MEME activite comme meilleur score
(c'est justement pourquoi ils sont paraphrases), donc les deux sont retenus dessus -- exactement
le comportement souhaite (Categorie 1 correctement geree, comme la regle actuelle), sans le
defaut de contamination de la regle 'dissimilarite' testee precedemment.

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
    render_candidate,
    cosine,
    build_process_state_graph,
)
from agent.nodes.alignment_node import check_alignment, CONFIDENCE_THRESHOLD
from agent.nodes.report_node import collect_failures, cluster_by_root, status_for_root, compute_verdict
from agent.models.base_nvidia_embedding import embed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "global_assignment_comparison")

RUN_ID = "run_2"
MATCHING_THRESHOLD_FIXED = 0.25
ALIGNMENT_TAU = CONFIDENCE_THRESHOLD


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


def matches_current_rule(activity_ids, contextualized, activity_vecs, candidates, candidate_vecs, threshold: float) -> dict:
    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        if not scores:
            matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": []}
            continue
        best_score = max(scores)
        valid = [
            {"match": candidates[idx], "score": round(float(s), 4)}
            for idx, s in enumerate(scores) if s >= threshold and s >= (best_score - 0.05)
        ]
        valid.sort(key=lambda x: x["score"], reverse=True)
        matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": valid}
    return matches


def matches_global_assignment(
    activity_ids, contextualized, activity_vecs, candidates, candidate_vecs, threshold: float
) -> dict:
    """Chaque CANDIDAT (colonne) choisit sa meilleure activite (ligne), une seule fois, sur
    TOUTE la matrice -- inverse du sens de decision de la regle actuelle (qui raisonne par
    activite). Cf. docstring de tete."""
    matches = {aid: {"activity_label": contextualized[aid]["label"], "matches": []} for aid in activity_ids}
    if len(activity_ids) == 0:
        return matches

    for cidx, cname in enumerate(candidates):
        cvec = candidate_vecs[cidx]
        best_score, best_aid = -1.0, None
        for aid, avec in zip(activity_ids, activity_vecs):
            s = cosine(avec, cvec)
            if s > best_score:
                best_score, best_aid = s, aid
        if best_aid is not None and best_score >= threshold:
            matches[best_aid]["matches"].append({"match": cname, "score": round(float(best_score), 4)})

    for aid in matches:
        matches[aid]["matches"].sort(key=lambda x: x["score"], reverse=True)
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
    n_activities = len(matches)
    avg_matches = sum(len(m["matches"]) for m in matches.values()) / n_activities if n_activities else 0.0
    return {
        "SATISFIED": n_sat, "VIOLATED": n_viol, "UNRESOLVABLE": n_unres,
        "verdict": result["verdict"], "avg_matches_per_activity": round(avg_matches, 3),
    }


def run_comparison() -> list[dict]:
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

            matches_current = matches_current_rule(
                activity_ids, contextualized, activity_vecs, candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED
            )
            matches_global = matches_global_assignment(
                activity_ids, contextualized, activity_vecs, candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED
            )

            for regime, matches in [("current_margin", matches_current), ("global_assignment", matches_global)]:
                result = evaluate(matches, dfg_edges, reference_graph, activity_guards)
                rows.append({"model": model, "case": case_label, "regime": regime, **result})

    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    table = []
    for regime in ["current_margin", "global_assignment"]:
        subset = [r for r in rows if r["regime"] == regime]
        n = sum(r["SATISFIED"] + r["VIOLATED"] + r["UNRESOLVABLE"] for r in subset)
        n_sat = sum(r["SATISFIED"] for r in subset)
        n_viol = sum(r["VIOLATED"] for r in subset)
        n_unres = sum(r["UNRESOLVABLE"] for r in subset)
        verdicts = [r["verdict"] for r in subset]
        n_cases = len(verdicts)
        avg_matches = sum(r["avg_matches_per_activity"] for r in subset) / len(subset) if subset else 0.0
        table.append({
            "regime": regime, "n_terms": n,
            "SATISFIED_%": round(100 * n_sat / n, 1) if n else 0.0,
            "VIOLATED_%": round(100 * n_viol / n, 1) if n else 0.0,
            "UNRESOLVABLE_%": round(100 * n_unres / n, 1) if n else 0.0,
            "avg_matches_per_activity": round(avg_matches, 3),
            "n_cases": n_cases,
            "FAITHFUL_%": round(100 * verdicts.count("FAITHFUL") / n_cases, 1) if n_cases else 0.0,
            "FAITHFUL_WITH_RESERVATIONS_%": round(100 * verdicts.count("FAITHFUL_WITH_RESERVATIONS") / n_cases, 1) if n_cases else 0.0,
            "NOT_FAITHFUL_%": round(100 * verdicts.count("NOT_FAITHFUL") / n_cases, 1) if n_cases else 0.0,
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

    rows = run_comparison()
    if not rows:
        raise SystemExit(f"[global_assign] aucune donnee produite -- verifie {RESULTS_DIR} et RUN_ID={RUN_ID!r}")

    with open(os.path.join(OUT_DIR, "comparison_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows)
    print_table("Table T -- Regle actuelle (marge locale) vs assignation globale (argmax par candidat)", summary)

    by_case_regime: dict[tuple[str, str, str], str] = {
        (r["model"], r["case"], r["regime"]): r["verdict"] for r in rows
    }
    changed = []
    seen = set()
    for (model, case, regime) in list(by_case_regime):
        key = (model, case)
        if key in seen:
            continue
        seen.add(key)
        v_current = by_case_regime.get((model, case, "current_margin"))
        v_global = by_case_regime.get((model, case, "global_assignment"))
        if v_current != v_global:
            changed.append({"model": model, "case": case, "current_margin": v_current, "global_assignment": v_global})
    print_table("Table U -- Cas ou le verdict change entre les deux regimes", changed)

    with open(os.path.join(OUT_DIR, "table_t_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(OUT_DIR, "table_u_changed_cases.csv"), "w", newline="") as f:
        if changed:
            writer = csv.DictWriter(f, fieldnames=list(changed[0].keys()))
            writer.writeheader()
            writer.writerows(changed)

    print(f"\nResultats dans {OUT_DIR}")