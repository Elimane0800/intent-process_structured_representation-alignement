"""
Matching DFG -> U par similarité cosinus avec contextualisation topologique et 
sélection Top-K (multi-états).

Contrairement à la version précédente qui utilisait un SPO aplati et un argmax strict,
ce module :
1. Exploite le DFG complet pour récupérer le `gateway_context` (le texte des flèches/losanges).
2. Construit une représentation textuelle riche : [Contexte In] -> Label -> [Contexte Out].
3. Autorise une activité à ancrer plusieurs états (Top-K thresholding) pour résoudre
   le problème de perte de granularité des "Service Tasks" englobantes.

Ajout (ce fichier) : clustering des matches retenus par similarité inter-candidats.
Le top-k ci-dessus resout "combien d'etats une activite peut ancrer", mais laisse ouverte
une question distincte : quand une activite ancre plusieurs etats a la fois, ces etats
sont-ils des FAITS DISTINCTS que l'activite encode simultanement (une AND-clause locale,
ex. une Service Task "verifier et approuver" qui recouvre reellement deux etats), ou des
CANDIDATS QUASI-SYNONYMES que le matcher n'arrive pas a departager (une OR-clause locale,
ex. "application.verified" / "application.reviewed") ? Le score cosinus activite<->candidat
deja utilise pour le seuil et la marge top-k ne peut structurellement pas repondre a cette
question : il mesure la ressemblance activite-candidat, pas la relation candidat-candidat.
Le signal qui la porte est la similarite cosinus ENTRE LES CANDIDATS RETENUS EUX-MEMES
(deja disponible sans cout supplementaire, cf. candidate_vecs) : des candidats mutuellement
proches sont un cluster de quasi-synonymes (OR interne, ambiguite du matcher) ; des candidats
mutuellement eloignes sont des faits distincts (AND entre clusters, ambiguite du texte/BPMN).

Ce champ ("clusters") est PUREMENT ADDITIF : "matches" (liste plate) reste identique et
continue d'alimenter build_process_state_graph et l'ancien chemin BFS de tgms_solver.py sans
aucune modification de comportement. "clusters" n'est pas encore consomme par
_project_process_clause (a venir) -- il est produit ici, teste isolement, avant d'etre
branche plus loin dans le pipeline.

Limite assumee, a ecrire noir sur blanc dans le papier : ce critere est un PROXY geometrique,
pas une verite -- il peut fusionner a tort deux etats du domaine authentiquement distincts
mais lexicalement proches en embedding, ou scinder a tort deux formulations eloignees d'un
meme fait. CLUSTER_SIMILARITY_THRESHOLD est une hypothese de travail, pas une valeur figee :
a calibrer empiriquement sur le run dataset, meme discipline que CONFIDENCE_THRESHOLD
(alignment_node.py) et MATCHING_THRESHOLD ci-dessous -- un seuil pose puis rejete par
inspection manuelle si non justifie, jamais garde par defaut sans verification.
"""

import numpy as np
from agent.models.base_nvidia_embedding import embed

# Seuil global d'acceptation sémantique (remplace la sélection forcée de l'argmax).
# Si aucun score ne dépasse ce seuil pour une activité, elle n'ancre aucun état.
MATCHING_THRESHOLD = 0.25 

# Seuil de similarite COSINUS ENTRE CANDIDATS retenus pour une meme activite, utilise pour
# decider s'ils forment un cluster de quasi-synonymes (OR interne) ou des faits distincts
# (AND entre clusters) -- cf. discussion dans la docstring de tete. Valeur de depart plausible
# (zone typique des quasi-synonymes en embedding de phrases courtes), NON calibree sur le
# corpus reel -- a verifier empiriquement avant d'etre citee comme un choix definitif.
CLUSTER_SIMILARITY_THRESHOLD = 0.80


def render_candidate(entity: str, state: str) -> str:
    return f"{entity.replace('_', ' ')} {state.replace('_', ' ')}"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _entity_states(value) -> list[str]:
    """Helper pour supporter le format imbriqué du Protocole 3.2"""
    if isinstance(value, dict):
        return list(value.get("states", []))
    return list(value)

def _build_contextualized_activities(dfg_edges: list[dict]) -> dict[str, dict]:
    """
    Extrait les activités uniques et construit une représentation textuelle riche
    incluant la topologie locale et le contexte de routage (gateway_context).
    Retourne : {activity_id: {"label": label_brut, "context_text": "texte enrichi"}}
    """
    nodes_labels = {}
    incoming_ctx = {}
    outgoing_ctx = {}

    for edge in dfg_edges:
        src_id, src_label = edge["source_id"], edge["source_label"]
        tgt_id, tgt_label = edge["target_id"], edge["target_label"]
        ctx = edge.get("gateway_context", [])

        if src_label:
            nodes_labels[src_id] = src_label
            outgoing_ctx.setdefault(src_id, []).extend(ctx)
        if tgt_label:
            nodes_labels[tgt_id] = tgt_label
            incoming_ctx.setdefault(tgt_id, []).extend(ctx)

    contextualized = {}
    for node_id, label in nodes_labels.items():
        # Déduplication tout en gardant un ordre stable
        in_c = " | ".join(sorted(set(incoming_ctx.get(node_id, []))))
        out_c = " | ".join(sorted(set(outgoing_ctx.get(node_id, []))))

        parts = []
        if in_c: 
            parts.append(f"[{in_c}]")
        parts.append(label)
        if out_c: 
            parts.append(f"[{out_c}]")

        contextualized[node_id] = {
            "label": label,
            "context_text": " -> ".join(parts).strip()
        }

    return contextualized


def _cluster_matches_by_similarity(
    valid_matches: list[dict],
    vector_by_candidate: dict[str, np.ndarray],
    cluster_threshold: float,
) -> list[list[dict]]:
    """Regroupe les matches retenus d'UNE activite en composantes connexes de similarite
    cosinus (>= cluster_threshold) ENTRE LES CANDIDATS eux-memes -- jamais entre l'activite et
    un candidat, cf. justification en tete de fichier. Union-find simple (le nombre de matches
    par activite est petit, O(n^2) sur les paires est largement suffisant).

    Retourne une liste de clusters (chaque cluster = liste de matches, ordre d'origine
    preserve a l'interieur). Un cluster singleton par match = comportement inchange pour tout
    cas non ambigu (non-regression). Fonction pure, testable sans dependance a l'embedder.
    """
    n = len(valid_matches)
    if n == 0:
        return []
    if n == 1:
        return [list(valid_matches)]

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    vecs = [vector_by_candidate[m["match"]] for m in valid_matches]
    for i in range(n):
        for j in range(i + 1, n):
            if cosine(vecs[i], vecs[j]) >= cluster_threshold:
                union(i, j)

    grouped: dict[int, list[dict]] = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(valid_matches[i])
    return list(grouped.values())


def match_activities_to_states(
    contextualized_activities: dict[str, dict],
    state_space: dict,
    threshold: float,
    cluster_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Matching N-to-M (Top-K relatif) : une activité peut ancrer plusieurs états de U si le score
    dépasse le seuil absolu ET se trouve dans une marge très proche (0.05) du meilleur score.
    L'embedding est calculé sur le texte contextualisé, améliorant la discrimination topologique.
    """
    # Utilisation de _entity_states pour gérer correctement le format imbriqué du Protocole 3.2
    candidates = [f"{e}.{s}" for e, value in state_space.items() for s in _entity_states(value)]
    candidate_texts = [render_candidate(*c.split(".", 1)) for c in candidates]

    activity_ids = list(contextualized_activities.keys())
    # Nettoyage strict des espaces pour l'embedding
    activity_texts = [" ".join(contextualized_activities[aid]["context_text"].split()) for aid in activity_ids]

    if not activity_texts or not candidate_texts:
        return {
            aid: {"activity_label": contextualized_activities.get(aid, {}).get("label", ""), "matches": [], "clusters": []}
            for aid in activity_ids
        }

    # "query" est maintenu pour la symétrie sémantique
    activity_vecs = np.array(embed(activity_texts, input_type="query"))
    candidate_vecs = np.array(embed(candidate_texts, input_type="query"))

    vector_by_candidate = dict(zip(candidates, candidate_vecs))

    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        
        if not scores:
            matches[aid] = {"activity_label": contextualized_activities[aid]["label"], "matches": [], "clusters": []}
            continue
            
        best_score = max(scores)
        valid_matches = []
        
        for idx, score in enumerate(scores):
            # Règle inattaquable : on garde le score s'il passe le seuil absolu
            # ET s'il est à moins de 0.05 du meilleur candidat pour limiter le bruit.
            if score >= threshold and score >= (best_score - 0.05):
                valid_matches.append({"match": candidates[idx], "score": round(score, 4)})
        
        # Tri décroissant : le meilleur match reste premier
        valid_matches.sort(key=lambda x: x["score"], reverse=True)

        # "clusters" : structure locale AND-of-OR de l'activite -- cf. docstring de tete et
        # _cluster_matches_by_similarity. Purement additif, "matches" ci-dessus est inchange.
        clusters = _cluster_matches_by_similarity(valid_matches, vector_by_candidate, cluster_threshold)
        clusters.sort(key=lambda cluster: max(m["score"] for m in cluster), reverse=True)
        
        matches[aid] = {
            "activity_label": contextualized_activities[aid]["label"],
            "matches": valid_matches,
            "clusters": [[m["match"] for m in cluster] for cluster in clusters],
        }

    return matches


def build_process_state_graph(dfg_edges: list[dict], matches: dict) -> list[dict]:
    """
    Construit le graphe d'accessibilité (G_process) en intégrant la multiplicité
    des ancrages. Si un noeud A (ancre S1, S2) est suivi de B (ancre S3),
    le graphe génère les arêtes S1->S3 et S2->S3 pour le solver BFS de repli.
    """
    graph = []
    for edge in dfg_edges:
        src_id = edge["source_id"]
        tgt_id = edge["target_id"]

        src_data = matches.get(src_id, {})
        tgt_data = matches.get(tgt_id, {})

        src_matches = src_data.get("matches", [])
        tgt_matches = tgt_data.get("matches", [])

        if not src_matches or not tgt_matches:
            continue

        for s_match in src_matches:
            for t_match in tgt_matches:
                graph.append({
                    "from": s_match["match"],
                    "to": t_match["match"],
                    "activity_from": src_data["activity_label"],
                    "activity_to": tgt_data["activity_label"],
                    "source_id": src_id,
                    "target_id": tgt_id
                })
    return graph


def match_dfg_to_u(
    dfg_edges: list[dict],
    state_space: dict,
    threshold: float = MATCHING_THRESHOLD,
    cluster_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
) -> tuple[dict, list[dict]]:
    """Point d'entrée du node : retourne le dictionnaire de traduction (N-to-M) et le graphe de repli."""
    contextualized = _build_contextualized_activities(dfg_edges)
    
    empty_count = sum(1 for data in contextualized.values() if not data["context_text"].strip())
    if empty_count:
        print(f"    [state_matching] skipping {empty_count} empty contextualized activity label(s)")

    valid_contextualized = {k: v for k, v in contextualized.items() if v["context_text"].strip()}
    
    matches = match_activities_to_states(valid_contextualized, state_space, threshold, cluster_threshold)
    graph = build_process_state_graph(dfg_edges, matches)
    
    return matches, graph


if __name__ == "__main__":
    import json
    import os
    import re
    from agent.nodes.precondition_node_with_retry import load_state_spaces

    TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests_bloc2")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_matching")

    # Mappage de test existant
    DESC_INDEX_TO_CASE = {
        1: "job_application",
    }

    def parse_dfg_filename(filename: str) -> tuple[int, int] | None:
        # Note: En production, le script consomme les outputs JSON contenant le DFG complet
        m = re.match(r"(\d+)_(\d+)_dfg\.json$", filename)
        return (int(m.group(1)), int(m.group(2))) if m else None

    # Chargement depuis des fichiers _dfg.json pour garantir la présence du gateway_context
    dfg_files = sorted(f for f in os.listdir(TESTS_DIR) if f.endswith("_dfg.json"))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    for prompt_name in ("one_shot",):
        # Chargement du vrai state space validé de Llama 8B
        state_spaces = load_state_spaces(
            prompt_name=prompt_name, 
            run_filename="run_32.json", 
            model_name="meta/llama-3.1-8b-instruct"
        )
        results[prompt_name] = {}

        for dfg_filename in dfg_files:
            parsed = parse_dfg_filename(dfg_filename)
            if parsed is None:
                continue
            desc_index, model_index = parsed
            case_name = DESC_INDEX_TO_CASE.get(desc_index)
            if not case_name or case_name not in state_spaces:
                continue

            with open(os.path.join(TESTS_DIR, dfg_filename)) as f:
                dfg_edges = json.load(f)

            matches, graph = match_dfg_to_u(dfg_edges, state_spaces[case_name])

            print(f"\n########## {dfg_filename} [{case_name}][{prompt_name}] ##########")
            for aid, m_data in matches.items():
                match_strings = [f"{m['match']} ({m['score']})" for m in m_data["matches"]]
                if match_strings:
                    print(f"{m_data['activity_label'][:45]:45} -> {', '.join(match_strings)}")
                    clusters = m_data.get("clusters", [])
                    if len(clusters) > 1:
                        clause_str = " AND ".join("(" + " OR ".join(c) + ")" for c in clusters)
                        print(f"{'':45}    local clause: {clause_str}")
                else:
                    print(f"{m_data['activity_label'][:45]:45} -> [NO MATCH >= {MATCHING_THRESHOLD}]")
            print(f"Graphe process de repli (BFS) : {len(graph)} arêtes")

            results[prompt_name][dfg_filename] = {
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