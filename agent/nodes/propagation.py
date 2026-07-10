"""
Node : propagation_node
Côté Runtime (Post-Mortem).
Prend les violations primaires et propage l'infection via G_T.
Marque les états dépendants comme suspects en cascade.
"""

from __future__ import annotations
from agent.state import NeuroSymbolicState

def propagation_node(state: NeuroSymbolicState) -> dict:
    G_T = state.get("G_T", {})
    contaminated = set(state.get("contaminated_states", []))
    
    if not G_T or not contaminated:
        return {}

    # File d'attente pour le parcours en largeur (BFS)
    queue = list(contaminated)
    
    # Parcours du graphe pour la contagion
    while queue:
        current = queue.pop(0)
        
        # Récupération des états dépendants (Arête : current -> successeur)
        successors = G_T.get(current, [])
        for succ in successors:
            if succ not in contaminated:
                contaminated.add(succ)
                queue.append(succ)
                
    return {
        "contaminated_states": list(contaminated)
    }