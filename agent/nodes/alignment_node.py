"""Alignement Micro (partiel) : verifie que chaque precondition Pre(u) de G (issue du texte,
Protocole 3 + Pre + graph_construction) est positionnee correctement dans G_process (issu du
BPMN via DFG -> SPO -> matching), sans jamais exiger que G_process soit reduit a G.

Principe retenu explicitement dans la conversation : une description textuelle peut
sous-specifier un processus -- G_process a le droit d'etre plus riche que G. L'alignement ne
verifie donc jamais l'egalite des deux graphes, seulement une observation positionnelle : pour
chaque precondition non triviale de G, les etats qu'elle requiert sont-ils atteignables, via un
CHEMIN (pas une arete directe -- le texte n'affirme jamais que rien d'autre ne peut se passer
entre les deux), avant l'etat cible, dans G_process ?

Source de G : graph_construction (pas precondition_extraction_with_retry directement) --
build_edges() y tague desormais chaque arete avec l'operateur ('AND'/'OR') qui l'a produite
(fix graph_node.py). La grammaire de Pre interdisant tout melange AND/OR pour une meme cible,
toutes les aretes pointant vers une meme cible partagent necessairement le meme operateur : un
seul groupe par cible, donc regrouper par 'to' suffit a reconstruire Pre(u) exactement, sans
retourner au validated Pre brut.

Trois issues par terme, jamais reduites a deux :
- SATISFIED    : chemin trouve dans G_process.
- VIOLATED     : les deux etats existent dans G_process, aucun chemin entre eux.
- UNRESOLVABLE : un des deux etats n'a aucun correspondant dans G_process -- ambigu entre deux
  causes qu'on a explicitement choisi de ne PAS demeler ici (sous-specification legitime du
  texte, vs. echec d'extraction en amont, Protocole 3) ; reporte tel quel, jamais force vers
  SATISFIED ou VIOLATED.

Demotion par confiance (ajoutee apres l'inspection qualitative des 61 VIOLATED du run dataset,
cf. violated_cases_dump.md) : un VIOLATED dont l'un des deux etats repose sur un match de score
< CONFIDENCE_THRESHOLD est retrograde en UNRESOLVABLE, avec la raison exacte et les scores
conserves. Justification, coherente avec la semantique des trois issues : VIOLATED est la seule
issue accusatoire (elle affirme un desordre reel du processus) -- l'affirmer sur un appariement
incertain produit exactement les artefacts observes (ex. 'Brag to friends' a 0.08 accuse un
desordre inexistant). Un match incertain signifie litteralement "on ne peut pas conclure", ce
qui est la definition d'UNRESOLVABLE. SATISFIED n'est volontairement PAS retrograde par ce
mecanisme : c'est un constat de non-contradiction, pas une accusation, et le retrograder
exploserait le taux d'UNRESOLVABLE sans corriger aucun artefact observe -- asymetrie assumee,
a documenter comme telle. Le seuil (0.35) est celui valide comme le plus prudent par le test
GMM + inspection manuelle (dataset_run_observations.md, sections 18-19) -- PAS le seuil GMM
(~0.48), invalide par cette meme inspection.
"""

import json
import os
import re

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "graph_construction")
MATCHING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_matching")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "alignment")

# Seuil sous lequel un match est juge trop incertain pour porter une accusation VIOLATED --
# calibre en deux temps sur le run dataset (215x3) : seuil initial pose sur le seul cas faux
# confirme (0.29), puis conserve apres que le seuil GMM (~0.48) a ete invalide par inspection
# manuelle (il flaggait majoritairement des paraphrases correctes). Cf. docstring de tete.
CONFIDENCE_THRESHOLD = 0.35


def build_state_scores(matches: dict) -> dict:
    """entity.state -> meilleur score de matching parmi toutes les activites qui y ont ete
    appariees. Le meilleur (pas la moyenne) : c'est le score de l'activite effectivement
    retenue par label_for_state() cote rapport, et un seul bon appariement suffit a rendre
    l'etat fiable comme point d'ancrage dans G_process."""
    scores = {}
    for m in matches.values():
        state = m["match"]
        scores[state] = max(scores.get(state, 0.0), m["score"])
    return scores


def group_edges_by_target(edges: list[dict]) -> dict:
    """Regroupe les aretes de graph_construction par cible -- un seul groupe par cible possible
    (grammaire AND/OR pure garantie par Pre), donc l'operateur et la citation du groupe sont
    simplement ceux de la premiere arete rencontree (identiques sur toutes les aretes d'un meme
    groupe, la citation justifie toute la precondition, pas un terme isole)."""
    groups = {}
    for edge in edges:
        target = edge["to"]
        groups.setdefault(target, {"operator": edge["operator"], "quote": edge.get("quote"), "terms": []})
        groups[target]["terms"].append(edge["from"])
    return groups


def build_process_adjacency(process_graph: list[dict]) -> dict:
    adj = {}
    for edge in process_graph:
        adj.setdefault(edge["from"], []).append(edge["to"])
    return adj


def path_exists(adjacency: dict, source: str, target: str) -> bool:
    """BFS. source == target compte comme un chemin trivial (le meme etat matche des deux
    cotes). Protection contre les cycles deja documentes comme possibles cote G_process."""
    if source == target:
        return True
    visited = {source}
    queue = [source]
    while queue:
        node = queue.pop(0)
        for neighbor in adjacency.get(node, []):
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def check_term(term: str, target: str, matched_states: set, adjacency: dict,
               state_scores: dict | None = None,
               confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """state_scores=None conserve exactement l'ancien comportement (aucune demotion) -- les
    appels existants sans scores restent valides et comparables aux anciens runs."""
    missing = [s for s in (term, target) if s not in matched_states]
    if missing:
        return {"term": term, "status": "UNRESOLVABLE", "reason": f"no match in process for: {missing}", "missing": missing}
    if path_exists(adjacency, term, target):
        return {"term": term, "status": "SATISFIED", "reason": None, "missing": []}
    # Candidat VIOLATED -- demotion si l'un des deux points d'ancrage repose sur un match trop
    # incertain : accuser un desordre sur un appariement douteux produit les artefacts
    # documentes (cf. docstring de tete). missing reste [] : les deux etats EXISTENT dans
    # G_process, c'est la conclusion qui est indecidable, pas la presence -- report_node
    # retombe alors sur le terme lui-meme pour la remontee de cause, comme pour un VIOLATED.
    if state_scores is not None:
        weak = [s for s in (term, target) if state_scores.get(s, 0.0) < confidence_threshold]
        if weak:
            detail = ", ".join(f"{s}={state_scores.get(s, 0.0):.3f}" for s in weak)
            return {
                "term": term,
                "status": "UNRESOLVABLE",
                "reason": f"would be VIOLATED, but match confidence below "
                          f"{confidence_threshold} for: {detail} -- cannot conclude on an "
                          f"uncertain match",
                "missing": [],
                "low_confidence": weak,
            }
    return {"term": term, "status": "VIOLATED", "reason": "both states matched, no path in process", "missing": []}


def aggregate(term_results: list[dict], operator: str) -> str:
    """AND : un seul UNRESOLVABLE rend l'ensemble UNRESOLVABLE (on ne peut rien conclure), sinon
    un seul VIOLATED suffit a rendre l'ensemble VIOLATED. OR : un seul SATISFIED suffit ; sinon
    UNRESOLVABLE si au moins un terme ne peut etre tranche, VIOLATED seulement si tous les termes
    sont resolus et tous violes."""
    statuses = [r["status"] for r in term_results]
    if operator == "AND":
        if "UNRESOLVABLE" in statuses:
            return "UNRESOLVABLE"
        if "VIOLATED" in statuses:
            return "VIOLATED"
        return "SATISFIED"
    else:  # OR
        if "SATISFIED" in statuses:
            return "SATISFIED"
        if "UNRESOLVABLE" in statuses:
            return "UNRESOLVABLE"
        return "VIOLATED"


def check_alignment(reference_graph: dict, process_graph: list[dict], matches: dict,
                    confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Point d'entree : une observation par cible non triviale de G (graph_construction).
    matched_states = tout entity.state ayant recu au moins une activite matchee (issu de
    state_matching_node). Les scores de matching servent uniquement a la demotion des
    VIOLATED incertains (cf. check_term) -- jamais a departager SATISFIED/UNRESOLVABLE."""
    matched_states = {m["match"] for m in matches.values()}
    state_scores = build_state_scores(matches)
    adjacency = build_process_adjacency(process_graph)
    groups = group_edges_by_target(reference_graph["edges"])

    results = {}
    for target, group in groups.items():
        term_results = [
            check_term(t, target, matched_states, adjacency, state_scores, confidence_threshold)
            for t in group["terms"]
        ]
        results[target] = {
            "operator": group["operator"],
            "quote": group["quote"],
            "terms": term_results,
            "status": aggregate(term_results, group["operator"]),
        }
    return results


if __name__ == "__main__":
    from agent.nodes.precondition_node_with_retry import load_state_spaces

    # graph_construction/run_N.json mirrors precondition_extraction_with_retry/run_N.json 1:1
    # (meme nom de fichier, cf. graph_node.py) -- qui lui-meme pin run_filename="run_17.json"
    # pour U dans son propre __main__. Donc N'IMPORTE QUEL run_N.json ici a ete construit avec
    # le U de run_17, par construction, pas par supposition. Le vrai choix ici, c'est quel run
    # (quel modele/prompt) aligner, pas s'il correspond a run_17 -- ca, c'est garanti.
    GRAPH_RUN = "run_1.json"
    CASE_NAME = "job_application"

    with open(os.path.join(GRAPH_DIR, GRAPH_RUN)) as f:
        graph_data = json.load(f)

    prompt_name_ref = next(iter(graph_data))
    model_name_ref = next(iter(graph_data[prompt_name_ref]))
    reference_graph = graph_data[prompt_name_ref][model_name_ref].get(CASE_NAME)
    if reference_graph is None or "error" in reference_graph:
        raise SystemExit(f"[alignment] no usable graph for {CASE_NAME!r} in {GRAPH_RUN}")
    print(f"Loaded G from {GRAPH_RUN} [{prompt_name_ref}][{model_name_ref}][{CASE_NAME}] "
          f"({len(reference_graph['edges'])} edges)")

    # Sanity check garde quand meme -- verifie que ce run precis a bien ete produit avec le
    # meme U que run_17, au cas ou le hardcode en amont aurait change depuis sa generation.
    reference_state_spaces = load_state_spaces(run_filename="run_17.json")
    reference_states = {
        f"{e}.{s}" for e, states in reference_state_spaces.get(CASE_NAME, {}).items() for s in states
    }
    graph_node_ids = {n["id"] for n in reference_graph["nodes"]}
    unknown_nodes = graph_node_ids - reference_states
    if unknown_nodes:
        print(f"    [alignment] WARNING: {len(unknown_nodes)} node(s) in {GRAPH_RUN} not in run_17's U "
              f"-- may not match run_17. First few: {sorted(unknown_nodes)[:5]}")

    # state_matching produit un run_N.json unique par execution (structure imbriquee
    # {prompt_name: {spo_filename: {...}}}) -- on prend le plus recent par defaut.
    existing_matching_runs = [f for f in os.listdir(MATCHING_DIR) if re.match(r"run_\d+\.json$", f)]
    if not existing_matching_runs:
        raise SystemExit(f"[alignment] no state_matching run found in {MATCHING_DIR}")
    latest_matching_run = max(existing_matching_runs, key=lambda f: int(re.match(r"run_(\d+)\.json$", f).group(1)))
    print(f"Loading state_matching results from {latest_matching_run}")
    with open(os.path.join(MATCHING_DIR, latest_matching_run)) as f:
        matching_data = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Un seul run_N.json par execution, structure imbriquee {prompt_name: {spo_filename: {...}}}
    # -- jamais d'ecrasement d'un run precedent.
    # BUG CORRIGE : precondition_node_with_retry.py charge STATE_SPACES = load_state_spaces(
    # run_filename="run_17.json") UNE SEULE FOIS, avec le prompt_name par defaut de la fonction
    # ("one_shot"). Les cles "zero_shot"/"one_shot"/"two_shot"/"cot" de PROMPTS designent le
    # prompt utilise pour GENERER LES PRECONDITIONS, pas celui utilise pour generer U -- U est
    # TOUJOURS celui de run_17 en one_shot (vocabulaire singulier "job_application"), quelle que
    # soit la cle du run_N.json de precondition_extraction_with_retry. state_matching_node.py,
    # lui, charge bien un U different par prompt (zero_shot -> vocabulaire pluriel
    # "job_applications", different de celui de G). Comparer G au bucket "zero_shot" du matching
    # revient donc a comparer deux vocabulaires qui ne se recouvrent jamais -- tout ressortait
    # UNRESOLVABLE, identique quel que soit le modele BPMN teste, sans rien mesurer. Seul le
    # bucket "one_shot" du matching partage le vocabulaire de G ; c'est le seul valide ici.
    if "one_shot" not in matching_data:
        raise SystemExit(f"[alignment] no 'one_shot' bucket in {latest_matching_run} -- "
                          f"G's U is always one_shot-sourced, nothing else is comparable")
    skipped = [k for k in matching_data if k != "one_shot"]
    if skipped:
        print(f"    [alignment] skipping bucket(s) {skipped} -- vocabulary mismatch with G "
              f"(see comment above), not a valid comparison")

    results = {}
    for matching_prompt_name in ("one_shot",):
        per_spo = matching_data[matching_prompt_name]
        results[matching_prompt_name] = {}
        for spo_filename, entry in per_spo.items():
            matches = entry["matches"]
            process_graph = entry["graph"]

            alignment = check_alignment(reference_graph, process_graph, matches)

            n_sat = sum(1 for r in alignment.values() if r["status"] == "SATISFIED")
            n_viol = sum(1 for r in alignment.values() if r["status"] == "VIOLATED")
            n_unres = sum(1 for r in alignment.values() if r["status"] == "UNRESOLVABLE")
            n_demoted = sum(
                1 for r in alignment.values() for t in r["terms"] if t.get("low_confidence")
            )

            print(f"\n########## {spo_filename} [{matching_prompt_name}] ##########")
            print(f"SATISFIED={n_sat} VIOLATED={n_viol} UNRESOLVABLE={n_unres} (total: {len(alignment)})")
            if n_demoted:
                print(f"  dont {n_demoted} terme(s) VIOLATED retrograde(s) en UNRESOLVABLE "
                      f"pour confiance de match < {CONFIDENCE_THRESHOLD}")
            for target, r in alignment.items():
                terms = " ".join(t["term"] for t in r["terms"])
                print(f"  [{r['status']:12}] {target:35} <- ({r['operator']}) {terms}")

            results[matching_prompt_name][spo_filename] = alignment

    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")