"""
Node : aggregation_node
Côté Explicabilité.
Calcule le score final de conformité basé sur les poids structurels.
"""

from __future__ import annotations
from agent.state import NeuroSymbolicState

def aggregation_node(state: NeuroSymbolicState) -> dict:
    X = state.get("X", [])
    weights = state.get("weights", {})
    contaminated = set(state.get("contaminated_states", []))
    
    total_weight = 0
    compliant_weight = 0
    
    # Parcours de tous les états tentés dans la séquence
    for X_i in X:
        for U_k in X_i:
            # Récupération du poids (par défaut 1 si non trouvé pour éviter la division par 0)
            w = weights.get(U_k, 1) 
            total_weight += w
            
            # Si l'état n'a pas été contaminé, la transition est valide
            if U_k not in contaminated:
                compliant_weight += w
                
    # Calcul du score final [0, 1]
    if total_weight == 0:
        final_score = 1.0 # Aucun état évalué, on évite la division par zéro
    else:
        final_score = compliant_weight / total_weight
        
    return {
        "final_score": round(final_score, 4)
    }