"""
Fichier : edges.py
Contient la logique de routage conditionnel (Conditional Edges) pour l'orchestrateur LangGraph.
"""

from __future__ import annotations
from agent.state import NeuroSymbolicState

def router_verification(state: NeuroSymbolicState) -> str:
    """
    Routage de la boucle de vérification causale.
    Évalue si l'index courant a atteint la fin de la séquence X.
    
    Retourne :
        - "continue" : la trace n'est pas terminée, on boucle.
        - "end_loop" : la trace est terminée, on passe à la contagion.
    """
    current_i = state.get("current_i", 0)
    X = state.get("X", [])
    
    # Si l'index est strictement inférieur à la taille de la séquence
    if current_i < len(X):
        return "continue"
    else:
        # Fin de la trace, on sort de la boucle
        return "end_loop"


def router_macro_conformance(state: NeuroSymbolicState) -> str:
    """
    (Optionnel mais recommandé) Routage post-inférence Zéro-Shot.
    Si les intentions de T et F sont radicalement opposées, 
    on peut court-circuiter le système.
    """
    macro_conformance = state.get("macro_conformance")
    
    # Si le champ est explicitement à False, on va direct à l'explication
    if macro_conformance is False:
        return "abort_to_explanation"
    
    # Sinon, on lance la machinerie lourde
    return "proceed_to_extraction"