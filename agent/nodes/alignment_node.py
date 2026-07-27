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
build_edges() y tague desormais chaque arete avec un index de CLAUSE (fix graph_node.py, v3).

MISE A JOUR v3 : la grammaire de Pre n'interdit plus le melange AND/OR pour une meme cible --
restriction levee (cf. precondition_prompt.py, tgms_solver.py::Guard) car elle etait injustifiee,
pas une vraie limite du domaine (le texte exprime authentiquement des regles du type
"(A AND B) OR C"). Une cible peut donc desormais porter PLUSIEURS groupes d'aretes (une clause
= un groupe d'aretes partageant le meme index 'clause', plusieurs clauses pour une meme cible
= les OR-alternatives d'une DNF a un niveau). Regrouper par 'to' NE suffit PLUS a lui seul --
il faut regrouper par 'to' PUIS par 'clause' pour reconstruire Pre(u) exactement ; c'est ce que
fait tgms_solver.py::tgms_from_pipeline (avec repli explicite sur l'ancienne semantique a
groupe unique pour tout graphe deja genere avant ce changement, cf. sa docstring).

Trois issues par terme, jamais reduites a deux :
- SATISFIED    : chemin trouve dans G_process.
- VIOLATED     : les deux etats existent dans G_process, aucun chemin entre eux, ancrage
                  suffisamment confiant des deux cotes (cf. demotion ci-dessous).
- UNRESOLVABLE : soit un des deux etats n'a aucun correspondant dans G_process (ambigu entre
                  sous-specification legitime du texte et echec d'extraction en amont, jamais
                  demele automatiquement), soit un candidat VIOLATED a ete retrograde parce
                  qu'un de ses deux ancrages repose sur un appariement trop incertain (voir
                  CONFIDENCE_THRESHOLD ci-dessous). Jamais force vers SATISFIED ou VIOLATED.

=== Refactoring TGMS (cf. note de formalisme dediee "A Twin-Graph Milestone System") ===

La logique de verification (groupement des guards, atteignabilite, agregation AND/OR,
demotion par confiance) est desormais deleguee a tgms_solver.py, qui formalise ce mecanisme
comme instance d'un objet TGMS = (M, G, A, Gproc, phi, c) et prouve deux theoremes de
monotonie (en confiance, en enrichissement du graphe process) -- voir la note pour les
preuves completes. Ce fichier ne fait plus que l'adaptation format-dict <-> TGMS et
l'orchestration ; check_alignment() garde une signature ET un comportement de sortie
STRICTEMENT identiques a la version precedente (verifie par test d'equivalence stricte sur
donnees reelles v1/v2, cf. tgms_solver tests) -- aucun autre node du pipeline n'a besoin de
changer (graph.py importe toujours `check_alignment` seul).

Demotion par confiance (deja presente avant ce refactoring, desormais formalisee) : un
candidat VIOLATED dont un des deux etats repose sur un score de matching < CONFIDENCE_THRESHOLD
est retrograde en UNRESOLVABLE -- calibre empiriquement sur le run dataset (215x3, voir
dataset_run_observations.md), confirme par test de controle direct (le seuil GMM statistique
a ete teste et rejete par inspection manuelle, cf. meme document). SATISFIED n'est jamais
retrograde par ce mecanisme (asymetrie assumee, prouvee necessaire par le Theoreme de
monotonie en confiance : un SATISFIED est invariant a tau, un VIOLATED ne peut migrer que
vers UNRESOLVABLE quand tau monte -- gater SATISFIED n'ajouterait donc aucune protection que
le theoreme ne garantit deja)."""

import json
import os
import re

from agent.nodes.tgms_solver import solve, tgms_from_pipeline, verdicts_to_pipeline_alignment

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "graph_construction")
MATCHING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_matching")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "alignment")

# Seuil sous lequel un match est juge trop incertain pour porter une accusation VIOLATED --
# calibre en deux temps sur le run dataset (215x3) : seuil initial pose sur le seul cas faux
# confirme (0.29), conserve apres que le seuil GMM (~0.48) a ete invalide par inspection
# manuelle (il flaggait majoritairement des paraphrases correctes). Cf. docstring de tete.
CONFIDENCE_THRESHOLD = 0.35


def check_alignment(reference_graph: dict, process_graph: list[dict], matches: dict,
                    confidence_threshold: float = CONFIDENCE_THRESHOLD,
                    activity_guards: dict | None = None) -> dict:
    """Point d'entree : une observation par cible non triviale de G (graph_construction).
    Delegue au solveur TGMS (tgms_solver.solve) -- ce corps n'est plus que l'adaptation
    format-dict et le passage du seuil de confiance ; toute la logique de verification vit
    dans tgms_solver.py, formalisee et testee independamment.

    activity_guards (proposition 2, additif) : sortie de bpmn_guards_node.compute_activity_
    guards -- {"activities": {...}, "undeclared_splits": [...]}, calculee par analyse de
    dominance/post-dominance, jamais par un LLM. None si absente (bpmn_to_spo a echoue avant de
    l'ecrire) -- le solveur retombe alors sur l'atteignabilite seule, comportement identique a
    avant cet ajout. Quand fournie, elle ne fait jamais gagner un SATISFIED qui n'existait pas
    deja par atteignabilite -- elle peut seulement RETROGRADER un SATISFIED douteux vers
    UNRESOLVABLE quand l'analyse structurelle independante ne peut elle-meme pas confirmer
    l'exclusivite qu'un chemin qui evite un terme nie semble suggerer (cf. tgms_solver.
    status_of_negated_term) -- jamais invente vers VIOLATED."""
    tgms, quotes = tgms_from_pipeline(reference_graph, process_graph, matches, activity_guards)
    verdicts = solve(tgms, tau=confidence_threshold)
    return verdicts_to_pipeline_alignment(verdicts, quotes)


if __name__ == "__main__":
    from agent.nodes.precondition_node_with_retry import load_state_spaces

    # graph_construction/run_N.json mirrors precondition_extraction_with_retry/run_N.json 1:1
    # (meme nom de fichier, cf. graph_node.py) -- on aligne donc le run_22 (celui qu'on
    # travaille actuellement, cf. precondition_node_with_retry.py). Le vrai choix ici, c'est
    # quel run (quel modele/prompt) aligner, pas quel U il utilise -- ca, c'est garanti par
    # construction en amont, pas suppose ici.
    GRAPH_RUN = "run_22.json"
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
    # meme U que run_32/meta-llama-3.1-8b-instruct, au cas ou le hardcode en amont aurait
    # change depuis sa generation. model_name est desormais OBLIGATOIRE ici : run_32.json
    # contient plusieurs modeles pour prompt_name="one_shot" (cf. load_state_spaces, qui leve
    # une ValueError explicite plutot que de choisir silencieusement le premier du dict si
    # model_name est omis et que plusieurs modeles sont presents -- meme discipline que partout
    # ailleurs dans ce pipeline : jamais un choix implicite ou l'ordre d'un dict).
    reference_state_spaces = load_state_spaces(
        run_filename="run_32.json", model_name="meta/llama-3.1-8b-instruct"
    )
    reference_states = {
        f"{e}.{s}" for e, states in reference_state_spaces.get(CASE_NAME, {}).items() for s in states
    }
    graph_node_ids = {n["id"] for n in reference_graph["nodes"]}
    unknown_nodes = graph_node_ids - reference_states
    if unknown_nodes:
        print(f"    [alignment] WARNING: {len(unknown_nodes)} node(s) in {GRAPH_RUN} not in "
              f"run_32/meta-llama-3.1-8b-instruct's U -- may not match. "
              f"First few: {sorted(unknown_nodes)[:5]}")

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
    # PIN A JOUR (etait run_17/model implicite, desormais perime) : precondition_node_with_
    # retry.py charge STATE_SPACES = load_state_spaces(run_filename="run_32.json",
    # model_name="meta/llama-3.1-8b-instruct") UNE SEULE FOIS, avec le prompt_name par defaut
    # de la fonction ("one_shot"). Les cles "zero_shot"/"one_shot"/"two_shot"/"cot" de PROMPTS
    # designent le prompt utilise pour GENERER LES PRECONDITIONS, pas celui utilise pour
    # generer U -- U est TOUJOURS celui de run_32/meta-llama-3.1-8b-instruct en one_shot, quelle
    # que soit la cle du run_N.json de precondition_extraction_with_retry. state_matching_node.py,
    # lui, charge bien un U different par prompt s'il varie -- mais son propre __main__ est
    # desormais pin explicitement sur ce meme run_32/meme modele en one_shot (cf. son code),
    # donc le vocabulaire reste partage avec G tant que ce pin ne change pas des deux cotes en
    # meme temps. Comparer G a un autre bucket reviendrait a comparer deux vocabulaires qui ne
    # se recouvrent jamais -- tout ressortirait UNRESOLVABLE, sans rien mesurer. Seul le bucket
    # "one_shot" du matching partage le vocabulaire de G ; c'est le seul valide ici.
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