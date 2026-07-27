"""
Calcul déterministe des Guards (DNF) par analyse de la topologie BPMN.
Ce module n'utilise aucun LLM et repose sur l'analyse de flux de contrôle (dominance et 
post-dominance) pour extraire la structure causale exacte d'un processus.

Une région est qualifiée de saine (SESE) si et seulement si :
1. Le split domine le join.
2. Le join post-domine le split.
3. Leurs types BPMN (AND, XOR, OR) concordent.

Tout écart à ces règles (ex: graphe irréductible, chevauchement) marque la propagation
comme "STRUCTURE_UNRESOLVED", déclenchant un repli assumé en aval, sans jamais deviner.
"""

import json
import sys
import types

# Contournement : pm4py importe cvxopt.glpk au chargement (pour ses solveurs d'alignement/
# conformance-checking bases sur la programmation lineaire), meme si on ne s'en sert jamais ici
# -- on utilise uniquement read_bpmn() et le modele d'objets BPMN, jamais ces solveurs. Sur
# certains systemes (ex. macOS, cvxopt compile sans le support GLPK), "from cvxopt import glpk"
# echoue au chargement de pm4py et bloque meme le simple parsing BPMN qu'on veut faire.
# Enregistrer un module factice AVANT l'import de pm4py suffit a satisfaire l'import -- sans
# danger pour notre usage, puisqu'aucune fonction GLPK n'est jamais reellement appelee.
try:
    from cvxopt import glpk  # noqa: F401 -- si ca marche deja, rien a faire
except ImportError:
    import cvxopt
    _glpk_stub = types.ModuleType("cvxopt.glpk")
    sys.modules["cvxopt.glpk"] = _glpk_stub
    cvxopt.glpk = _glpk_stub

import pm4py

import networkx as nx
import itertools
import pm4py
from typing import Dict, List, Tuple, Set, Any, Optional

# On réutilise les types de passerelles définis dans bpmn_to_spo_node.py[cite: 7]
GATEWAY_TYPES = {
    pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway: "XOR",
    pm4py.objects.bpmn.obj.BPMN.ParallelGateway: "AND",
    pm4py.objects.bpmn.obj.BPMN.InclusiveGateway: "OR",
    pm4py.objects.bpmn.obj.BPMN.EventBasedGateway: "XOR",  # Comportement structurel XOR
}

ACTIVITY_TYPES = (
    pm4py.objects.bpmn.obj.BPMN.Task,
    pm4py.objects.bpmn.obj.BPMN.SubProcess,
)

def _get_node_type(node: Any) -> str:
    for gw_class, gw_type in GATEWAY_TYPES.items():
        if isinstance(node, gw_class):
            return gw_type
    if isinstance(node, ACTIVITY_TYPES):
        return "ACTIVITY"
    return "OTHER"


def _combine_and(guard1: List[List[str]], guard2: List[List[str]]) -> List[List[str]]:
    """Produit cartésien pour AND-join : (A ou B) ET (C ou D) -> AC ou AD ou BC ou BD."""
    if not guard1: return guard2
    if not guard2: return guard1
    return [list(set(c1 + c2)) for c1 in guard1 for c2 in guard2]


def _combine_or(guards: List[List[List[str]]]) -> List[List[str]]:
    """Union pour XOR-join : flatten de toutes les clauses entrantes."""
    result = []
    for g in guards:
        result.extend(g)
    # Déduplication rudimentaire des clauses (frozenset pour ignorer l'ordre interne)
    seen = set()
    deduped = []
    for clause in result:
        fs = frozenset(clause)
        if fs not in seen:
            seen.add(fs)
            deduped.append(clause)
    return deduped


def _combine_inclusive_or(guards: List[List[List[str]]]) -> List[List[str]]:
    """
    Combinatoire stricte pour OR-join sur région SESE valide.
    Union de tous les sous-ensembles non vides des branches entrantes.
    """
    if not guards:
        return []
    
    result = []
    for r in range(1, len(guards) + 1):
        for subset in itertools.combinations(guards, r):
            subset_guard = []
            for g in subset:
                subset_guard = _combine_and(subset_guard, g) if subset_guard else g
            result.extend(subset_guard)
            
    return _combine_or([result])  # Utilise _combine_or juste pour dédupliquer


def build_control_flow_dag(bpmn: Any) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Extrait le graphe complet (gateways inclus), détecte et retire les back-edges (cycles).
    Génère un DAG propre pour l'analyse de dominance.

    CORRECTIF (bug preexistant, decouvert en construisant find_undeclared_splits -- proposition
    3, mais affectant deja validate_sese_regions avant meme cet ajout) : bpmn.get_flows()
    retourne un melange de SequenceFlow ET MessageFlow, jamais distingues jusqu'ici. Un
    messageFlow relie deux ACTIVITES DE POOLS DIFFERENTS pour une communication asynchrone --
    ce n'est jamais un flux de controle a l'interieur de l'execution d'un pool, contrairement a
    un sequenceFlow. Les inclure comme aretes de controle donnait un out_degree artificiellement
    gonfle a toute tache d'envoi/reception (sendTask/receiveTask) ayant a la fois un
    sequenceFlow ET un messageFlow -- observe concretement sur un diagramme de collaboration a
    3 pools reel : 4 activites signalees a tort comme "fork non declare" (proposition 3) alors
    qu'elles n'avaient qu'un seul sequenceFlow sortant reel. Corrige a la racine ici, pas
    seulement contourne dans find_undeclared_splits -- meme fix beneficie a
    validate_sese_regions, deja expose au meme risque avant cet ajout."""
    nodes = {n.get_id(): n for n in bpmn.get_nodes()}
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    
    for flow in bpmn.get_flows():
        if not isinstance(flow, pm4py.objects.bpmn.obj.BPMN.SequenceFlow):
            continue
        G.add_edge(flow.get_source().get_id(), flow.get_target().get_id())

    # DFS pour isoler les back-edges
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in G.nodes}
    back_edges = set()

    def dfs(u):
        color[u] = GRAY
        for v in G.successors(u):
            if color.get(v, WHITE) == WHITE:
                dfs(v)
            elif color[v] == GRAY:
                back_edges.add((u, v))
        color[u] = BLACK

    for n in G.nodes:
        if color[n] == WHITE:
            dfs(n)

    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes)
    dag.add_edges_from(e for e in G.edges if e not in back_edges)
    
    return dag, nodes


def validate_sese_regions(dag: nx.DiGraph, nodes: Dict[str, Any]) -> Set[str]:
    """
    Calcule dominance et post-dominance pour certifier les pairings split/join.
    Retourne l'ensemble des IDs de gateways considérés comme invalides (non-SESE).
    """
    # Ajout de super-noeuds pour garantir l'unicité des racines/feuilles exigée par Lengauer-Tarjan
    aug_dag = dag.copy()
    aug_dag.add_node("__SUPER_START__")
    aug_dag.add_node("__SUPER_END__")
    
    sources = [n for n, d in dag.in_degree() if d == 0]
    sinks = [n for n, d in dag.out_degree() if d == 0]
    
    for s in sources: aug_dag.add_edge("__SUPER_START__", s)
    for s in sinks: aug_dag.add_edge(s, "__SUPER_END__")

    try:
        dom = nx.immediate_dominators(aug_dag, "__SUPER_START__")
        pdom = nx.immediate_dominators(aug_dag.reverse(copy=True), "__SUPER_END__")
    except nx.NetworkXError:
        # En cas de graphe déconnecté ou gravement malformé
        return set(dag.nodes())

    invalid_gateways = set()
    
    splits = [n for n in dag.nodes() if dag.out_degree(n) > 1 and _get_node_type(nodes[n]) in ("AND", "OR", "XOR")]
    
    for split in splits:
        join_candidate = pdom.get(split)
        
        # SESE check 1 & 2 : Unicité et existence du post-dominateur
        if not join_candidate or join_candidate == "__SUPER_END__":
            invalid_gateways.add(split)
            continue
            
        # SESE check 3 : Le candidat doit être un join (in_degree > 1)
        if dag.in_degree(join_candidate) <= 1:
            invalid_gateways.add(split)
            continue

        # SESE check 4 : Symétrie de la dominance (le split doit dominer le join)
        # On remonte l'arbre des dominateurs immédiats pour vérifier
        curr = join_candidate
        dominates = False
        while curr and curr != "__SUPER_START__":
            if curr == split:
                dominates = True
                break
            curr = dom.get(curr)
            
        if not dominates:
            invalid_gateways.add(split)
            invalid_gateways.add(join_candidate)
            continue
            
        # SESE check 5 : Concordance stricte des types BPMN
        if _get_node_type(nodes[split]) != _get_node_type(nodes[join_candidate]):
            invalid_gateways.add(split)
            invalid_gateways.add(join_candidate)

    return invalid_gateways


def find_undeclared_splits(dag: nx.DiGraph, nodes: Dict[str, Any]) -> Set[str]:
    """Detecte les noeuds NON-gateway (activites BPMN) avec plusieurs flux sortants directs --
    un fork qui n'est jamais reconnu comme "split" par validate_sese_regions(), dont l'analyse
    (a raison, pour sa propre fonction) ne porte QUE sur le pairage dominance/post-dominance
    des gateways AND/OR/XOR : `splits = [n for n in dag.nodes() if dag.out_degree(n) > 1 and
    _get_node_type(nodes[n]) in ("AND", "OR", "XOR")]` -- une activite (`_get_node_type` ==
    "ACTIVITY") n'entre jamais dans cette liste, quel que soit son out_degree.

    Consequence concrete, observee sur un cas reel (une tache BPMN nommee comme une action de
    notation, avec deux flux sortants directs vers deux issues censees etre mutuellement
    exclusives dans le texte source) : le jeton diverge sans jamais passer par une decision
    explicite -- ni AND (les deux branches s'executent forcement ensemble, sans jamais le
    declarer), ni XOR (un choix, sans jamais l'exprimer). Les deux successeurs calculent leur
    garde independamment, chacun avec un seul predecesseur de son propre point de vue, donc rien
    dans compute_activity_guards() ne signale jamais cette divergence -- elle reste absorbee en
    silence.

    Retourne l'ensemble des ids concernes, pour propagation explicite en aval (cf.
    compute_activity_guards) -- jamais silencieusement ignore comme un simple gateway non
    apparie le serait deja (invalid_gateways), parce qu'ici il n'y a meme pas de gateway a
    invalider : le noeud lui-meme n'est jamais entre dans l'analyse de depart."""
    return {
        n for n in dag.nodes()
        if dag.out_degree(n) > 1 and _get_node_type(nodes[n]) == "ACTIVITY"
    }


def compute_activity_guards(bpmn: Any) -> Dict[str, Any]:
    """
    Point d'entrée. Exécute le calcul bottom-up des guards pour chaque activité.

    Retourne {"activities": {nom_activite: {...}}, "undeclared_splits": [nom_activite, ...]} --
    changement de forme (avant : directement le dict d'activites) pour exposer separement les
    forks non declares repres par find_undeclared_splits(), sans reutiliser le canal
    STRUCTURE_UNRESOLVED pour DEUX raisons distinctes (pairage de gateway invalide vs fork
    jamais declare) sous un seul et meme statut opaque. Le noeud d'origine du fork garde son
    PROPRE guard normalement calcule (l'atteindre, lui, n'a rien d'ambigu) mais porte un flag
    "forks_undeclared" ; ce sont ses successeurs qui heritent de STRUCTURE_UNRESOLVED, via la
    meme logique de propagation deja en place pour un pairage de gateway invalide -- un seul
    mecanisme de propagation, jamais duplique.
    """
    dag, nodes = build_control_flow_dag(bpmn)
    invalid_gateways = validate_sese_regions(dag, nodes)
    undeclared_splits = find_undeclared_splits(dag, nodes)
    
    # State mapping: node_id -> Guard (List[List[str]]) ou "STRUCTURE_UNRESOLVED"
    guards_state: Dict[str, Any] = {}
    
    # Init : sources (vide = True logique)
    for n in nx.topological_sort(dag):
        preds = list(dag.predecessors(n))
        
        if not preds:
            guards_state[n] = [[]]
            continue
            
        # Collecte l'état des prédécesseurs
        incoming_guards = []
        is_unresolved = False
        
        for p in preds:
            state = guards_state.get(p)
            # CORRECTIF (proposition 3) : un predecesseur dans undeclared_splits poison la
            # garde exactement comme un STRUCTURE_UNRESOLVED deja etabli -- meme mecanisme de
            # propagation, une seule cause supplementaire de declenchement. p garde son PROPRE
            # guard correct dans guards_state[p] (calcule normalement ci-dessous quand p a ete
            # traite), seuls SES SUCCESSEURS heritent de l'indetermination -- cf. docstring de
            # find_undeclared_splits.
            if state == "STRUCTURE_UNRESOLVED" or p in undeclared_splits:
                is_unresolved = True
                break
            
            # Si le prédécesseur était une activité, il s'ajoute à l'historique
            if _get_node_type(nodes[p]) == "ACTIVITY":
                activity_name = nodes[p].get_name()
                if activity_name and activity_name.strip():
                    # Applique un AND entre la garde du prédécesseur et l'activité elle-même
                    state = _combine_and(state, [[activity_name.strip()]])
            
            incoming_guards.append(state)

        if is_unresolved or n in invalid_gateways:
            guards_state[n] = "STRUCTURE_UNRESOLVED"
            continue

        node_type = _get_node_type(nodes[n])
        
        # Logique de fusion si in_degree > 1 (Join)
        if len(incoming_guards) > 1:
            if node_type == "XOR":
                guards_state[n] = _combine_or(incoming_guards)
            elif node_type == "AND":
                # Réduction séquentielle des branches
                res = incoming_guards[0]
                for g in incoming_guards[1:]:
                    res = _combine_and(res, g)
                guards_state[n] = res
            elif node_type == "OR":
                guards_state[n] = _combine_inclusive_or(incoming_guards)
            else:
                # Merge par défaut sur des noeuds non-passerelles (rare si BPMN bien formé)
                guards_state[n] = _combine_or(incoming_guards)
        else:
            guards_state[n] = incoming_guards[0]

    # Formatage de la sortie (seulement pour les activités réelles)
    final_output = {}
    for n, state in guards_state.items():
        if _get_node_type(nodes[n]) == "ACTIVITY" and nodes[n].get_name():
            activity_name = nodes[n].get_name().strip()
            if not activity_name:
                continue
            
            if state == "STRUCTURE_UNRESOLVED":
                final_output[activity_name] = {"status": "STRUCTURE_UNRESOLVED"}
            else:
                # Filtre les clauses vides internes et formatte
                clean_clauses = [sorted(list(clause)) for clause in state if clause]
                if not clean_clauses:
                    # Condition INITIALE
                    final_output[activity_name] = {"status": "OK", "clauses": [[]]}
                else:
                    final_output[activity_name] = {"status": "OK", "clauses": clean_clauses}
                # Le noeud d'ORIGINE d'un fork non declare garde son propre guard correct
                # (l'atteindre n'a rien d'ambigu) mais est marque explicitement -- jamais
                # confondu avec un STRUCTURE_UNRESOLVED silencieux, cf. docstring de tete.
                if n in undeclared_splits:
                    final_output[activity_name]["forks_undeclared"] = True

    return {
        "activities": final_output,
        "undeclared_splits": sorted({
            nodes[n].get_name().strip() for n in undeclared_splits
            if nodes[n].get_name() and nodes[n].get_name().strip()
        }),
    }

if __name__ == "__main__":
    import os
    import json
    import re
    # Import uniquement du parseur depuis le node SPO validé
    from agent.nodes.bpmn_to_spo_node import parse_bpmn

    # Recopié ici pour ne pas avoir à modifier le fichier validé bpmn_to_spo_node.py
    EXCLUDED_MALFORMED = {
        "E_j01/10.bpmn2.xml",
        "E_j05/9.bpmn2.xml",
        "G_g03/0.bpmn2.xml",
        "M_g01/3.bpmn2.xml",
        "M_g02/5.bpmn2.xml",
        "M_j02/5.bpmn2.xml",
        "R_g01/0.bpmn2.xml",
        "R_j03/2.bpmn2.xml",
    }

    BPMN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "text_and_bpmn", "bpmn")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bpmn_guards")

    description_folders = sorted(
        d for d in os.listdir(BPMN_ROOT) if os.path.isdir(os.path.join(BPMN_ROOT, d))
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    
    global_stats = {
        "total_activities_processed": 0,
        "resolved_ok": 0,
        "structure_unresolved": 0,
        "files_processed": 0,
        "files_with_unresolved": 0,
        "undeclared_splits_found": 0,  # activites-origine d'un fork non declare (proposition 3)
        "files_with_undeclared_split": 0,
    }

    print("Début de la validation empirique sur le corpus BPMN...\n")

    for desc_folder in description_folders:
        desc_path = os.path.join(BPMN_ROOT, desc_folder)
        bpmn_files = sorted(f for f in os.listdir(desc_path) if f.endswith(".bpmn2.xml"))
        results[desc_folder] = {}

        for bpmn_filename in bpmn_files:
            key = f"{desc_folder}/{bpmn_filename}"
            if key in EXCLUDED_MALFORMED:
                results[desc_folder][bpmn_filename] = {
                    "excluded": "connectivite irrecuperable"
                }
                continue

            bpmn_path = os.path.join(desc_path, bpmn_filename)
            try:
                bpmn = parse_bpmn(bpmn_path)
                guards_result = compute_activity_guards(bpmn)
                
                results[desc_folder][bpmn_filename] = guards_result
                
                # Agrégation des métriques -- guards_result["activities"] depuis le changement
                # de forme (proposition 3) ; undeclared_splits desormais expose separement.
                guards = guards_result["activities"]
                file_has_unresolved = False
                for act, data in guards.items():
                    global_stats["total_activities_processed"] += 1
                    if data.get("status") == "STRUCTURE_UNRESOLVED":
                        global_stats["structure_unresolved"] += 1
                        file_has_unresolved = True
                    else:
                        global_stats["resolved_ok"] += 1

                n_splits = len(guards_result["undeclared_splits"])
                global_stats["undeclared_splits_found"] += n_splits
                if n_splits:
                    global_stats["files_with_undeclared_split"] += 1

                global_stats["files_processed"] += 1
                if file_has_unresolved:
                    global_stats["files_with_unresolved"] += 1

            except Exception as e:
                results[desc_folder][bpmn_filename] = {"error": str(e)}
                print(f"[{key}] ERROR: {e}")

    # Sauvegarde des résultats
    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    
    with open(out_path, "w") as f:
        json.dump({
            "global_stats": global_stats,
            "details": results
        }, f, indent=2, ensure_ascii=False)
        
    print("\n=== BILAN EMPIRIQUE ===")
    print(f"Fichiers traités (hors malformés exclus) : {global_stats['files_processed']}")
    print(f"Fichiers contenant au moins une asymétrie non-SESE : {global_stats['files_with_unresolved']}")
    print(f"Activités totales extraites : {global_stats['total_activities_processed']}")
    print(f"Activités avec Guard résolu (OK) : {global_stats['resolved_ok']}")
    print(f"Activités en repli (STRUCTURE_UNRESOLVED) : {global_stats['structure_unresolved']}")
    print(f"Forks non déclarés détectés (activité, pas gateway) : "
          f"{global_stats['undeclared_splits_found']} (dans {global_stats['files_with_undeclared_split']} fichier(s))")
    
    if global_stats["total_activities_processed"] > 0:
        taux_echec = (global_stats["structure_unresolved"] / global_stats["total_activities_processed"]) * 100
        print(f"Taux d'échec topologique : {taux_echec:.2f}%")
        
    print(f"\nRésultats complets sauvegardés dans : {out_path}")