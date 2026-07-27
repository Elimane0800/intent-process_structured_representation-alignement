"""Audit des types de noeuds BPMN reifies comme "activite" par resolve_dfg() (bpmn_to_spo_node.py)
-- question posee explicitement par ce fichier lui-meme, jamais tranchee sur donnees jusqu'ici :
"les evenements intermediaires (timers, messages) restent reifies [...] limite documentee, a
reexaminer sur donnees si les artefacts residuels le justifient". Meme classe de risque que les
deux bugs deja trouves et corriges dans ce fichier et dans bpmn_guards_node.py (association/
textAnnotation traites comme du flux ; messageFlow compte dans le DAG de controle) -- jamais
d'audit systematique equivalent sur resolve_dfg() lui-meme.

Deux questions distinctes, croisees avec le meme signal (type BPMN reel du noeud, obtenu par
type(node).__name__ -- jamais suppose via une liste de classes attendues, la meme methode que
bpmn_to_spo_node.py utilise deja pour gateway_context) :

  Table N : parmi tous les noeuds reifies comme activite (tous types confondus), quelle part
    n'est JAMAIS retenue par le matching (a MATCHING_THRESHOLD=0.25), par type BPMN -- un type
    surrepresente dans le "jamais matche" est un candidat "bruit" qui vole potentiellement des
    slots top-k a de vraies taches sans jamais apporter de signal utile.

  Table O : pour chaque milestone MISSING (aucun ancrage), quel est le type BPMN du MEILLEUR
    candidat trouve MEME SOUS le seuil -- un type surrepresente ici est un candidat "quasi
    manque" potentiellement recuperable (proche mais jamais choisi), a distinguer d'un milestone
    sans aucun signal proche nulle part dans le BPMN (vraie absence, pas un probleme de type de
    noeud).

Aucun appel LLM. Meme reutilisation qu'ailleurs (dfg_edges/state_space/reference_graph deja sur
disque, run_2, embeddings calcules une fois par cas) -- PLUS le fichier BPMN brut lui-meme
(state["bpmn_path"], deja sauvegarde dans l'etat initial et jamais ecrase par aucun node du
pipeline), reparse une fois par cas pour obtenir le type reel de chaque noeud."""

import csv
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.nodes.bpmn_to_spo_node import parse_bpmn
from agent.nodes.state_matching_node import (
    _build_contextualized_activities,
    _entity_states,
    render_candidate,
    cosine,
)
from agent.nodes.alignment_node import check_alignment
from agent.models.base_nvidia_embedding import embed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "node_type_audit")

RUN_ID = "run_2"
MATCHING_THRESHOLD_FIXED = 0.25
ALIGNMENT_TAU = 0.35  # inchange -- hors sujet de cet audit, cf. discussion tau_sweep.py


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


def matches_at_threshold(activity_ids, contextualized, activity_vecs, candidates, candidate_vecs, threshold: float) -> dict:
    """Version allegee de la fonction du meme nom ailleurs -- pas besoin du clustering ici,
    seulement de savoir si une activite a >=1 match retenu (Table N)."""
    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        if not scores:
            matches[aid] = []
            continue
        best_score = max(scores)
        valid = [candidates[idx] for idx, s in enumerate(scores) if s >= threshold and s >= (best_score - 0.05)]
        matches[aid] = valid
    return matches


def node_types_by_id(bpmn) -> dict[str, str]:
    """id -> type(node).__name__, pour TOUS les noeuds du BPMN, jamais restreint a une liste de
    classes attendues (cf. docstring de tete -- meme methode que bpmn_to_spo_node.py utilise
    deja pour gateway_context, la seule facon de ne jamais rater un type non anticipe)."""
    return {n.get_id(): type(n).__name__ for n in bpmn.get_nodes()}


def run_audit() -> tuple[list[dict], list[dict]]:
    by_model = discover_run2_cases(RESULTS_DIR, RUN_ID)
    node_rows = []       # Table N -- une ligne par noeud reifie
    missing_rows = []    # Table O -- une ligne par milestone missing

    for model in sorted(by_model):
        for subset, case_name, filepath in by_model[model]:
            try:
                with open(filepath) as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if "fatal_error" in state:
                continue
            if any(k not in state for k in ("dfg_edges", "state_space", "reference_graph", "bpmn_path")):
                continue

            bpmn_path = state["bpmn_path"]
            if not os.path.exists(bpmn_path):
                print(f"  [audit] {model}/{subset}/{case_name} -- bpmn_path introuvable "
                      f"({bpmn_path}), ignore")
                continue

            dfg_edges = state["dfg_edges"]
            state_space = state["state_space"]
            reference_graph = state["reference_graph"]
            activity_guards = state.get("activity_guards")
            case_label = f"{subset}/{case_name}"

            try:
                bpmn = parse_bpmn(bpmn_path)
            except Exception as e:
                print(f"  [audit] {model}/{subset}/{case_name} -- reparse BPMN echoue ({e}), ignore")
                continue
            types_by_id = node_types_by_id(bpmn)

            precomputed = precompute_embeddings(dfg_edges, state_space)
            activity_ids, contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs = precomputed
            if not activity_ids or len(candidate_vecs) == 0:
                continue

            # Table N -- un noeud a la fois
            matches = matches_at_threshold(
                activity_ids, contextualized, activity_vecs, candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED
            )
            for aid in activity_ids:
                node_rows.append({
                    "model": model, "case": case_label, "activity_id": aid,
                    "node_type": types_by_id.get(aid, "UNKNOWN_NOT_IN_BPMN_NODES"),
                    "matched_at_least_once": bool(matches.get(aid)),
                })

            # Table O -- milestones missing, identifies via une passe check_alignment reelle a
            # MATCHING_THRESHOLD_FIXED (matches complets, avec clustering -- recalcules
            # separement ici avec la fonction complete, pas la version allegee ci-dessus, pour
            # avoir un G_process fidele et donc une vraie liste `missing`).
            from agent.nodes.state_matching_node import (
                _cluster_matches_by_similarity, build_process_state_graph, CLUSTER_SIMILARITY_THRESHOLD,
            )
            full_matches = {}
            for aid, vec in zip(activity_ids, activity_vecs):
                scores = [cosine(vec, cvec) for cvec in candidate_vecs]
                if not scores:
                    full_matches[aid] = {"activity_label": contextualized[aid]["label"], "matches": [], "clusters": []}
                    continue
                best_score = max(scores)
                valid_matches = [
                    {"match": candidates[idx], "score": round(float(s), 4)}
                    for idx, s in enumerate(scores) if s >= MATCHING_THRESHOLD_FIXED and s >= (best_score - 0.05)
                ]
                valid_matches.sort(key=lambda x: x["score"], reverse=True)
                clusters = _cluster_matches_by_similarity(valid_matches, vector_by_candidate, CLUSTER_SIMILARITY_THRESHOLD)
                full_matches[aid] = {
                    "activity_label": contextualized[aid]["label"],
                    "matches": valid_matches,
                    "clusters": [[m["match"] for m in c] for c in clusters],
                }
            process_graph = build_process_state_graph(dfg_edges, full_matches)
            alignment = check_alignment(
                reference_graph, process_graph, full_matches,
                confidence_threshold=ALIGNMENT_TAU, activity_guards=activity_guards,
            )

            missing_milestones = set()
            for entry in alignment.values():
                for t in entry.get("terms", []):
                    for m in t.get("missing", []):
                        missing_milestones.add(m)

            # Meilleur score PROPRE de chaque activite (tous candidats confondus) -- necessaire
            # pour distinguer, pour un milestone missing donne, "aucun signal en absolu" de
            # "signal suffisant mais bat par un concurrent sur la meme activite (marge top-k
            # de 0.05)". Calcule une fois par cas, reutilise pour tous les milestones missing.
            activity_own_best: dict[str, float] = {}
            for aid, avec in zip(activity_ids, activity_vecs):
                scores = [cosine(avec, cvec) for cvec in candidate_vecs]
                activity_own_best[aid] = max(scores) if scores else -1.0

            for milestone in sorted(missing_milestones):
                cvec = vector_by_candidate.get(milestone)
                if cvec is None:
                    continue  # ne devrait pas arriver -- tous les milestones de guards viennent de state_space
                best_score, best_aid = -1.0, None
                for aid, avec in zip(activity_ids, activity_vecs):
                    s = cosine(avec, cvec)
                    if s > best_score:
                        best_score, best_aid = s, aid

                own_best = activity_own_best.get(best_aid, -1.0)
                if best_score < MATCHING_THRESHOLD_FIXED:
                    cause = "below_threshold"  # aucun signal en absolu, pas une question de concurrence
                elif best_score >= own_best - 0.05:
                    # Score suffisant ET dans la marge top-k de sa propre meilleure activite --
                    # DEVRAIT deja etre dans matches. Si ce milestone apparait quand meme comme
                    # missing, la cause reelle est ailleurs (ex. c'est le TERME COMPLEMENTAIRE
                    # de la meme cible qui manque, pas celui-ci) -- signal de bug potentiel dans
                    # CE script (extraction missing_milestones), pas dans le pipeline lui-meme.
                    cause = "SHOULD_HAVE_MATCHED_investigate"
                else:
                    cause = "lost_to_margin"  # signal suffisant, mais un concurrent l'emporte
                                                # sur CETTE MEME activite (marge de 0.05)

                winners = []
                if cause == "lost_to_margin":
                    winners = [
                        m["match"] for m in full_matches.get(best_aid, {}).get("matches", [])
                    ]

                missing_rows.append({
                    "model": model, "case": case_label, "milestone": milestone,
                    "best_score": round(float(best_score), 4),
                    "best_activity_label": contextualized.get(best_aid, {}).get("label", "?") if best_aid else "?",
                    "best_activity_type": types_by_id.get(best_aid, "UNKNOWN") if best_aid else "N/A",
                    "best_activity_own_best_score": round(float(own_best), 4),
                    "cause": cause,
                    "winning_candidates_on_same_activity": "; ".join(winners) if winners else "",
                })

    return node_rows, missing_rows


def build_table_n(node_rows: list[dict]) -> list[dict]:
    by_type: dict[str, list[bool]] = {}
    for r in node_rows:
        by_type.setdefault(r["node_type"], []).append(r["matched_at_least_once"])
    table = []
    for node_type in sorted(by_type):
        flags = by_type[node_type]
        n = len(flags)
        n_matched = sum(flags)
        table.append({
            "node_type": node_type, "n_nodes": n,
            "matched_at_least_once_%": round(100 * n_matched / n, 1) if n else 0.0,
            "never_matched_%": round(100 * (n - n_matched) / n, 1) if n else 0.0,
        })
    table.sort(key=lambda r: r["n_nodes"], reverse=True)
    return table


def build_table_o(missing_rows: list[dict]) -> list[dict]:
    """Une ligne par CAUSE (below_threshold / lost_to_margin / SHOULD_HAVE_MATCHED_investigate),
    jamais une moyenne unique qui les melangerait -- cf. le bug corrige (des scores > 0.7
    classes 'missing' a tort, faute de cette distinction dans la premiere version)."""
    causes = ["below_threshold", "lost_to_margin", "SHOULD_HAVE_MATCHED_investigate"]
    table = []
    for cause in causes:
        rows = [r for r in missing_rows if r["cause"] == cause]
        n = len(rows)
        if n == 0:
            table.append({"cause": cause, "n_missing_milestones": 0, "avg_best_score": None, "%_of_total": 0.0})
            continue
        scores = [r["best_score"] for r in rows]
        table.append({
            "cause": cause, "n_missing_milestones": n,
            "avg_best_score": round(sum(scores) / n, 4),
            "%_of_total": round(100 * n / len(missing_rows), 1) if missing_rows else 0.0,
        })
    return table


def build_table_o_by_type(missing_rows: list[dict]) -> list[dict]:
    """Croisement type d'activite x cause -- pour verifier si un type BPMN precis (Task vs
    autre) est surrepresente dans une cause donnee."""
    by_key: dict[tuple[str, str], int] = {}
    for r in missing_rows:
        key = (r["best_activity_type"], r["cause"])
        by_key[key] = by_key.get(key, 0) + 1
    table = [
        {"node_type": node_type, "cause": cause, "count": count}
        for (node_type, cause), count in sorted(by_key.items(), key=lambda kv: -kv[1])
    ]
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

    node_rows, missing_rows = run_audit()
    if not node_rows:
        raise SystemExit(f"[audit] aucune donnee produite -- verifie {RESULTS_DIR} et RUN_ID={RUN_ID!r}")

    with open(os.path.join(OUT_DIR, "node_rows_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()))
        writer.writeheader()
        writer.writerows(node_rows)
    with open(os.path.join(OUT_DIR, "missing_rows_raw.csv"), "w", newline="") as f:
        if missing_rows:
            writer = csv.DictWriter(f, fieldnames=list(missing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(missing_rows)

    table_n = build_table_n(node_rows)
    table_o = build_table_o(missing_rows)

    print_table("Table N -- Taux de match par type de noeud BPMN reifie", table_n)
    print_table("Table O -- Type du MEILLEUR candidat (sous le seuil) pour chaque milestone missing", table_o)

    for name, table in [("table_n", table_n), ("table_o", table_o)]:
        with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="") as f:
            if table:
                writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
                writer.writeheader()
                writer.writerows(table)

    print(f"\nResultats dans {OUT_DIR}")