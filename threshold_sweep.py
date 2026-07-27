"""Ablation isolee sur MATCHING_THRESHOLD (state_matching_node.py) : mesure son effet SEUL sur
les verdicts d'alignement, sans jamais retoucher aux etages en amont (state_space, Pre, G) ni
en aval a la marge (activity_guards).

Pourquoi pas simplement relancer run.py plusieurs fois avec des seuils differents : run.py
invoque agent.graph (le pipeline COMPLET), donc chaque relance regenererait state_space et les
preconditions par appel LLM -- reintroduisant exactement la variable confondante deja
diagnostiquee sur la comparaison run_1/run_2 (cf. Table H, analyze_lre_results.py) : un
changement de verdict deviendrait de nouveau illisible entre "effet du seuil" et "derive du LLM
d'un appel a l'autre".

Principe ici : reutiliser un run DEJA TERMINE (dfg_edges, state_space, reference_graph,
activity_guards deja calcules et sauvegardes dans results/lre_runs/{model}/{subset}__{case}__
{run_id}.json) et ne rejouer QUE state_matching + alignment, en isolant explicitement le seuil
comme unique variable. Aucun appel LLM dans la boucle de balayage.

Deuxieme precaution, du meme ordre : match_activities_to_states() de state_matching_node.py
calcule les embeddings ET applique le seuil dans le meme appel -- rappeler cette fonction une
fois par valeur de seuil recalculerait les embeddings a chaque fois, avec le meme risque (mineur
mais non nul, une API d'embedding n'est pas necessairement bit-a-bit deterministe) que celui
deja rencontre avec le LLM generatif. Les embeddings sont donc calcules ICI une seule fois par
cas (reutilisation directe des fonctions de contextualisation de state_matching_node.py), et
seule la logique de seuillage/clustering est rejouee pour chaque valeur de MATCHING_THRESHOLD --
une reimplementation locale et volontairement fidele de la meme regle d'acceptation
(score >= threshold ET score >= best_score - 0.05), pour ne jamais diverger du comportement reel
du pipeline.
"""

import csv
import json
import os

# Charge .env EXPLICITEMENT ici -- ce script est autonome (n'importe ni agent.config ni
# agent.graph, contrairement a run.py), donc ne beneficie jamais implicitement du chargement
# d'environnement fait ailleurs dans le pipeline. python-dotenv est deja une dependance du
# projet (utilisee par agent.config ou equivalent) -- reutilisee ici, jamais reimplementee.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # si python-dotenv n'est pas installe, on suppose que l'environnement est deja
          # renseigne autrement (export manuel, variable d'environnement systeme) -- jamais un
          # crash a l'import pour une dependance optionnelle de confort

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
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "threshold_sweep")

RUN_ID = "run_2"  # source des dfg_edges/state_space/reference_graph/activity_guards deja calcules

# Les 3 (modele, subset, case_name) sur lesquels tourner le balayage -- A REMPLIR : un cas par
# modele, parmi ceux deja presents dans results/lre_runs/{model}/. Chaque triplet doit avoir un
# fichier {subset}__{case_name}__{RUN_ID}.json deja sur disque (sinon SystemExit explicite,
# jamais une execution partielle silencieuse).
CASES: list[tuple[str, str, str]] = [
    # (model_key, subset, case_name) -- meme case_name pour les trois modeles, deliberement :
    # isole la variable modele (texte source et BPMN identiques), plutot que de melanger cas et
    # modele dans la meme lecture. BicycleManufacturing = le cas deja analyse a la main plus tot
    # dans la discussion (Cas 1), donc une intuition qualitative de reference existe deja dessus.
    ("gpt-5.5", "OriginalDataset", "BicycleManufacturing"),
    ("gpt-oss-20b", "OriginalDataset", "BicycleManufacturing"),
    ("llama-3.1-8b", "OriginalDataset", "BicycleManufacturing"),
]

# Valeurs de MATCHING_THRESHOLD testees, de la plus stricte (valeur actuelle du pipeline) a la
# plus permissive -- ordre decroissant pour lire la progression naturellement.
THRESHOLD_VALUES = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]

# CONFIDENCE_THRESHOLD (alignment_node) reste FIXE a sa valeur par defaut du pipeline reel tout
# au long du balayage -- seul MATCHING_THRESHOLD varie ici, cf. docstring de tete sur pourquoi
# les deux seuils ne doivent jamais etre confondus dans la lecture des resultats.
ALIGNMENT_TAU = CONFIDENCE_THRESHOLD


def load_run_state(model_key: str, subset: str, case_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, model_key, f"{subset}__{case_name}__{RUN_ID}.json")
    if not os.path.exists(path):
        raise SystemExit(f"[sweep] fichier introuvable : {path} -- verifie CASES/RUN_ID")
    with open(path) as f:
        state = json.load(f)
    for required in ("dfg_edges", "state_space", "reference_graph"):
        if required not in state:
            raise SystemExit(
                f"[sweep] {path} n'a pas de champ '{required}' exploitable (run incomplet ou "
                f"echoue pour ce cas) -- choisis un autre cas dans CASES"
            )
    return state


def precompute_embeddings(dfg_edges: list[dict], state_space: dict):
    """Calcule UNE SEULE FOIS ce qui ne depend pas du seuil : le texte contextualise de chaque
    activite, les vecteurs d'activite, les vecteurs de candidats (entity.state). Reutilise
    identiquement pour chaque valeur de MATCHING_THRESHOLD testee -- cf. docstring de tete."""
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
    """Reimplementation FIDELE de la boucle de seuillage de
    state_matching_node.match_activities_to_states() -- meme regle d'acceptation exacte
    (score >= threshold ET score >= best_score - 0.05), appliquee aux embeddings deja calcules
    par precompute_embeddings(). Volontairement dupliquee plutot qu'appelee : la fonction
    d'origine recalculerait les embeddings a chaque appel (cf. docstring de tete)."""
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


def summarize_alignment(alignment: dict) -> dict:
    """Compte les termes par statut, et decompose UNRESOLVABLE par cause exacte (missing vs
    low_confidence vs autre) -- la meme decomposition que Table I prevue sur l'analyse
    principale, appliquee ici a un seul balayage de seuil."""
    counts = {"SATISFIED": 0, "VIOLATED": 0, "UNRESOLVABLE": 0}
    unresolvable_missing = 0
    unresolvable_low_confidence = 0
    unresolvable_other = 0

    for target, entry in alignment.items():
        for t in entry.get("terms", []):
            status = t["status"]
            counts[status] = counts.get(status, 0) + 1
            if status == "UNRESOLVABLE":
                if t.get("missing"):
                    unresolvable_missing += 1
                elif t.get("low_confidence"):
                    unresolvable_low_confidence += 1
                else:
                    unresolvable_other += 1

    return {
        **counts,
        "unresolvable_missing": unresolvable_missing,
        "unresolvable_low_confidence": unresolvable_low_confidence,
        "unresolvable_other": unresolvable_other,
    }


def run_sweep() -> list[dict]:
    rows = []
    for model_key, subset, case_name in CASES:
        print(f"\n########## {model_key} / {subset}/{case_name} ##########")
        state = load_run_state(model_key, subset, case_name)
        dfg_edges = state["dfg_edges"]
        state_space = state["state_space"]
        reference_graph = state["reference_graph"]
        activity_guards = state.get("activity_guards")  # peut etre absent, comportement identique

        precomputed = precompute_embeddings(dfg_edges, state_space)
        activity_ids, contextualized, activity_vecs, vector_by_candidate, candidates, candidate_vecs = precomputed

        if not activity_ids or len(candidate_vecs) == 0:
            print(f"  [sweep] {model_key}/{subset}/{case_name} -- pas assez de donnees "
                  f"(activites contextualisees ou candidats vides), cas ignore")
            continue

        for threshold in THRESHOLD_VALUES:
            matches = matches_at_threshold(
                activity_ids, contextualized, activity_vecs, vector_by_candidate,
                candidates, candidate_vecs, threshold,
            )
            process_graph = build_process_state_graph(dfg_edges, matches)
            alignment = check_alignment(
                reference_graph, process_graph, matches,
                confidence_threshold=ALIGNMENT_TAU, activity_guards=activity_guards,
            )
            summary = summarize_alignment(alignment)

            row = {
                "model": model_key, "case": f"{subset}/{case_name}",
                "matching_threshold": threshold, "alignment_tau": ALIGNMENT_TAU,
                **summary,
            }
            rows.append(row)
            total = summary["SATISFIED"] + summary["VIOLATED"] + summary["UNRESOLVABLE"]
            print(f"  threshold={threshold:.2f} -- SATISFIED={summary['SATISFIED']} "
                  f"VIOLATED={summary['VIOLATED']} UNRESOLVABLE={summary['UNRESOLVABLE']} "
                  f"(missing={summary['unresolvable_missing']}, "
                  f"low_conf={summary['unresolvable_low_confidence']}) / total={total}")

    return rows


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    if any(case_name == "REMPLACER" for _, _, case_name in CASES):
        raise SystemExit(
            "[sweep] CASES contient encore des valeurs placeholder ('REMPLACER') -- "
            "renseigne 3 (model_key, subset, case_name) reels avant de lancer, "
            "ex. en piochant dans results/lre_runs/{model}/ pour des fichiers "
            f"'{{subset}}__{{case_name}}__{RUN_ID}.json' deja presents."
        )

    rows = run_sweep()

    out_path = os.path.join(OUT_DIR, "threshold_sweep.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV ecrit : {out_path}")
    else:
        print("\n[sweep] aucune ligne produite -- verifie CASES et RUN_ID")