"""Compare la regle de retention ACTUELLE (marge fixe de 0.05 au meilleur score de l'activite)
a une regle ALTERNATIVE motivee par l'echantillon lost_to_margin inspecte a la main :

  ACTUELLE : garder tout candidat >= MATCHING_THRESHOLD ET >= (best_score - 0.05).
    Corrige les paraphrases (Categorie 1, ex. cart.readied/cart.ready) mais rejette aussi les
    faits distincts co-portes par la meme activite (Categorie 2, ex. storehouse.informed +
    engineering_department.informed sur "Inform storehouse and engineering department") -- les
    deux sont traites pareil, seule la distance au meilleur score compte.

  ALTERNATIVE : garder le meilleur candidat >= MATCHING_THRESHOLD, puis garder tout AUTRE
    candidat >= MATCHING_THRESHOLD qui n'est PAS une paraphrase (similarite candidat-candidat
    < CLUSTER_SIMILARITY_THRESHOLD) d'un candidat DEJA retenu -- peu importe sa distance au
    meilleur score de l'activite. Distingue les deux categories a la racine (au moment de la
    retention), plutot que de les traiter identiquement comme le fait la marge fixe.

Pourquoi ce n'est PAS la meme chose qu'elargir la marge globalement (deja teste et rejete,
margin_sweep.py) : elargir la marge accepte TOUT candidat proche en score absolu, y compris du
bruit sans rapport avec l'activite (d'ou la degradation VIOLATED_%/NOT_FAITHFUL_% observee).
Ici, un candidat n'est retenu que s'il represente un FAIT DISTINCT (dissimilaire des candidats
deja retenus) -- un candidat de bruit, sans rapport thematique avec ce qui est deja ancre,
resterait exclu s'il est lui-meme dissemblable... ATTENTION, point a verifier explicitement par
la mesure ci-dessous : rien n'empeche structurellement un candidat de bruit (score juste au-dessus
de MATCHING_THRESHOLD, dissemblable du meilleur candidat par hasard) d'etre retenu a tort sous
cette regle -- elle n'est pas immune au meme risque que la marge globale, seulement moins
exposee (elle ne retient PAS aveuglement tout ce qui est proche en score, mais elle ne verifie
pas non plus la PERTINENCE du candidat dissemblable, seulement sa dissemblance). D'ou la mesure
comparative ci-dessous plutot qu'une adoption sans verification.

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
    CLUSTER_SIMILARITY_THRESHOLD,
)
from agent.nodes.alignment_node import check_alignment, CONFIDENCE_THRESHOLD
from agent.nodes.report_node import collect_failures, cluster_by_root, status_for_root, compute_verdict
from agent.models.base_nvidia_embedding import embed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "retention_rule_comparison")

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
    """Regle ACTUELLE (marge fixe de 0.05) -- reimplementation identique aux autres scripts."""
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


def matches_dissimilarity_rule(
    activity_ids, contextualized, activity_vecs, candidates, candidate_vecs,
    threshold: float, cluster_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
) -> dict:
    """Regle ALTERNATIVE -- cf. docstring de tete. Garde le meilleur candidat au-dessus du seuil
    absolu, puis tout candidat SUPPLEMENTAIRE au-dessus du seuil qui n'est PAS une paraphrase
    (similarite < cluster_threshold) d'un candidat deja retenu, quelle que soit sa distance au
    meilleur score."""
    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        if not scores:
            matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": []}
            continue

        above = sorted(
            [(idx, s) for idx, s in enumerate(scores) if s >= threshold],
            key=lambda x: -x[1],
        )
        if not above:
            matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": []}
            continue

        retained_idx = [above[0][0]]  # meilleur candidat toujours retenu
        for idx, score in above[1:]:
            cvec = candidate_vecs[idx]
            is_paraphrase = any(
                cosine(cvec, candidate_vecs[r]) >= cluster_threshold for r in retained_idx
            )
            if not is_paraphrase:
                retained_idx.append(idx)

        valid = [{"match": candidates[i], "score": round(float(scores[i]), 4)} for i in retained_idx]
        valid.sort(key=lambda x: x["score"], reverse=True)
        matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": valid}
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
            matches_alt = matches_dissimilarity_rule(
                activity_ids, contextualized, activity_vecs, candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED
            )

            for regime, matches in [("current_margin", matches_current), ("dissimilarity", matches_alt)]:
                result = evaluate(matches, dfg_edges, reference_graph, activity_guards)
                rows.append({"model": model, "case": case_label, "regime": regime, **result})

    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    table = []
    for regime in ["current_margin", "dissimilarity"]:
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
        raise SystemExit(f"[retention] aucune donnee produite -- verifie {RESULTS_DIR} et RUN_ID={RUN_ID!r}")

    with open(os.path.join(OUT_DIR, "comparison_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows)
    print_table("Table R -- Regle actuelle (marge 0.05) vs regle alternative (dissimilarite)", summary)

    # Cas ou le verdict CHANGE entre les deux regimes -- le signal le plus direct pour juger si
    # le changement va dans le bon sens (moins de NOT_FAITHFUL, sans que VIOLATED explose).
    by_case_regime: dict[tuple[str, str], str] = {
        (r["model"], r["case"], r["regime"]): r["verdict"] for r in rows
    }
    changed = []
    seen = set()
    for (model, case, regime), verdict in by_case_regime.items():
        key = (model, case)
        if key in seen:
            continue
        seen.add(key)
        v_current = by_case_regime.get((model, case, "current_margin"))
        v_alt = by_case_regime.get((model, case, "dissimilarity"))
        if v_current != v_alt:
            changed.append({"model": model, "case": case, "current_margin": v_current, "dissimilarity": v_alt})
    print_table("Table S -- Cas ou le verdict change entre les deux regimes", changed)

    with open(os.path.join(OUT_DIR, "table_r_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(OUT_DIR, "table_s_changed_cases.csv"), "w", newline="") as f:
        if changed:
            writer = csv.DictWriter(f, fieldnames=list(changed[0].keys()))
            writer.writeheader()
            writer.writerows(changed)

    print(f"\nResultats dans {OUT_DIR}")