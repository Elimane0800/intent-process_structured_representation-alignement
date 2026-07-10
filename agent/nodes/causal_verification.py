"""
Node : causal_verification_node
Côté Runtime.
Vérifie la conformité causale de l'étape courante (current_i).
Implémente l'équation : Pre(U_k) ⊆ A_i
"""

from __future__ import annotations
from typing import Dict, List, Any
from agent.state import NeuroSymbolicState

def causal_verification_node(state: NeuroSymbolicState) -> dict:
    X = state.get("X", [])
    current_i = state.get("current_i", 0)
    
    # Sécurité : fin de la trace atteinte
    if current_i >= len(X):
        return {}

    # Désérialisation en Set pour la vitesse O(1)
    A_i = set(state.get("A_i", []))
    contaminated = set(state.get("contaminated_states", []))
    violations = list(state.get("violations", []))
    
    Pre_mapped = state.get("Pre_mapped", {})
    current_activations = X[current_i]

    for U_k in current_activations:
        # Si l'état a déjà été signalé comme contaminé dans cette même étape
        if U_k in contaminated:
            continue
            
        preconditions = set(Pre_mapped.get(U_k, []))
        
        # LE CŒUR MATHÉMATIQUE : Ce qui manque
        missing = preconditions - A_i
        
        if not missing:
            # VALIDÉ : Toutes les préconditions sont présentes
            A_i.add(U_k)
        else:
            # VIOLATION DÉTECTÉE
            violations.append({
                "transition_index": current_i,
                "attempted_state": U_k,
                "missing_preconditions": list(missing)
            })
            # On plante la "graine" de l'infection
            contaminated.add(U_k)

    # Re-sérialisation en List pour LangGraph et incrémentation
    return {
        "A_i": list(A_i),
        "violations": violations,
        "contaminated_states": list(contaminated),
        "current_i": current_i + 1
    }