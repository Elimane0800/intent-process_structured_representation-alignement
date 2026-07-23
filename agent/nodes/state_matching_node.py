"""Matching SPO -> U par similarite cosinus, et construction du graphe cote processus.
Pas de repli, pas de controle symbolique, pas de seuil de decision -- meilleur candidat retenu.

Deux entrees : une liste de triplets (subject, predicate, object) et un state_space (U) au
meme format que celui produit par state_space_node_v3.py."""

import numpy as np

from agent.models.base_nvidia_embedding import embed


def render_candidate(entity: str, state: str) -> str:
    return f"{entity.replace('_', ' ')} {state.replace('_', ' ')}"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_activities_to_states(activities: list[str], state_space: dict) -> dict:
    """Un seul appel d'embedding pour toutes les activites, un seul pour tous les candidats --
    pas un appel par paire. Cosinus, meilleur score retenu.

    nvidia/llama-nemotron-embed-1b-v2 est un modele query/passage (retrieval, entraine sur des
    paires question courte / paragraphe long). 'passage' a ete teste sur les candidats U et a
    degrade des matchs corrects en matchs faux (ex. 'Application process finished' ->
    process.ended devient job_application.confirmed) -- nos deux cotes sont courts et de meme
    nature, aucun n'est un vrai "document long", donc 'passage' n'apporte rien ici et deforme la
    representation. 'query' des deux cotes, empiriquement le seul reglage qui a produit des
    matchs majoritairement corrects -- ecart assume par rapport a l'usage documente par NVIDIA
    pour ce modele, a documenter comme tel dans le papier."""
    candidates = [f"{e}.{s}" for e, states in state_space.items() for s in states]
    candidate_texts = [render_candidate(*c.split(".", 1)) for c in candidates]

    # Labels BPMN observes tels quels (ex. "company rates\n application "), jamais nettoyes
    # jusqu'ici -- normalisation d'espacement uniquement avant l'appel embedding, le dict de
    # resultats reste indexe par le label brut pour que le lookup depuis les triplets SPO
    # continue de fonctionner.
    cleaned_activities = [" ".join(a.split()) for a in activities]

    activity_vecs = np.array(embed(cleaned_activities, input_type="query"))
    candidate_vecs = np.array(embed(candidate_texts, input_type="query"))

    matches = {}
    for activity, vec in zip(activities, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        best_idx = int(np.argmax(scores))
        matches[activity] = {"match": candidates[best_idx], "score": round(scores[best_idx], 4)}
    return matches


def build_process_state_graph(spo: list[tuple], matches: dict) -> list[dict]:
    """Reutilise directement l'ordre des activites porte par les triplets SPO (follows) -- chaque
    arete DFG devient une arete entre les deux etats matches, sans etape supplementaire.
    Une arete dont une extremite n'a pas ete matchee (ex. label vide, filtre en amont) est
    signalee explicitement et ecartee, jamais silencieusement."""
    graph = []
    for subj, _, obj in spo:
        if subj not in matches or obj not in matches:
            print(f"    [state_matching] skipping edge {subj!r} -> {obj!r}: unmatched activity")
            continue
        graph.append({
            "from": matches[subj]["match"],
            "to": matches[obj]["match"],
            "activity_from": subj,
            "activity_to": obj,
        })
    return graph


def match_spo_to_u(spo: list[tuple], state_space: dict) -> tuple[dict, list[dict]]:
    """Point d'entree du node : matches + graphe process, a partir d'un SPO et d'un U donnes."""
    all_activities = sorted({a for subj, _, obj in spo for a in (subj, obj)})
    activities = [a for a in all_activities if a.strip()]
    empty_count = len(all_activities) - len(activities)
    if empty_count:
        print(f"    [state_matching] skipping {empty_count} empty activity label(s) -- "
              f"cannot be embedded (e.g. an unnamed BPMN start/end event)")

    matches = match_activities_to_states(activities, state_space)
    graph = build_process_state_graph(spo, matches)
    return matches, graph


if __name__ == "__main__":
    import json
    import os
    import re

    from agent.nodes.precondition_node_with_retry import load_state_spaces

    TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests_bloc2")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_matching")

    # Prefixe du nom de fichier = index de description (use case), suffixe = index du modele
    # BPMN parmi les 8-11 disponibles pour ce use case (corpus Zenodo) -- pas un identifiant a
    # lister fichier par fichier. Seul l'index 1 (job_application) est confirme pour l'instant ;
    # completer au fur et a mesure que les autres correspondances index<->case_name sont connues.
    DESC_INDEX_TO_CASE = {
        1: "job_application",
    }

    def parse_spo_filename(filename: str) -> tuple[int, int] | None:
        m = re.match(r"(\d+)_(\d+)_spo\.json$", filename)
        return (int(m.group(1)), int(m.group(2))) if m else None

    spo_files = sorted(f for f in os.listdir(TESTS_DIR) if f.endswith("_spo.json"))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Un seul run_N.json par execution, structure imbriquee {prompt_name: {spo_filename: {...}}}
    # -- meme convention que state_space_node_v3.py / precondition_node_with_retry.py. Jamais
    # d'ecrasement d'un run precedent : chaque execution produit un nouveau numero.
    results = {}
    for prompt_name in ("zero_shot", "one_shot"):
        state_spaces = load_state_spaces(prompt_name=prompt_name, run_filename="run_17.json")
        results[prompt_name] = {}

        for spo_filename in spo_files:
            parsed = parse_spo_filename(spo_filename)
            if parsed is None:
                print(f"[state_matching] {spo_filename}: filename doesn't match '<desc>_<model>_spo.json', skipped")
                continue
            desc_index, model_index = parsed
            case_name = DESC_INDEX_TO_CASE.get(desc_index)
            if case_name is None:
                print(f"[state_matching] {spo_filename}: no case mapping for description index {desc_index}, skipped")
                continue
            state_space = state_spaces.get(case_name)
            if not state_space:
                print(f"[state_matching] {spo_filename}: no state_space for case {case_name!r}, skipped")
                continue

            with open(os.path.join(TESTS_DIR, spo_filename)) as f:
                spo_data = json.load(f)
            spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_data]

            matches, graph = match_spo_to_u(spo, state_space)

            print(f"\n########## {spo_filename} [{case_name}][{prompt_name}] ##########")
            for activity, m in matches.items():
                print(f"{activity!r:45} -> {m['match']:35} ({m['score']})")
            print(f"Graphe process ({len(graph)} aretes)")

            results[prompt_name][spo_filename] = {
                "case_name": case_name,
                "model_index": model_index,
                "matches": matches,
                "graph": graph,
            }

    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")