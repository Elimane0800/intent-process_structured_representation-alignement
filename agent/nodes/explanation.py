"""
Node : explanation_node
Côté Explicabilité (Le Fan-In final).
Prend les résultats de l'inférence, de la vérification et de l'agrégation
pour générer un rapport d'audit humainement lisible.

Version corrigée : Utilise .invoke() et intègre des protections anti-NoneType.
"""

from __future__ import annotations
from agent.models.base_llm_nim import BaseLLMProvider
from agent.state import NeuroSymbolicState
from agent.prompts.explanation_prompt import (
    EXPLANATION_SYSTEM_PROMPT,
    EXPLANATION_USER_PROMPT,
)

def explanation_node(state: NeuroSymbolicState) -> dict:
    """
    Rédige l'explication finale de la conformité sous forme de rapport textuel.
    """
    intent_T = state.get("intent_T", "Unknown")
    intent_F = state.get("intent_F", "Unknown")
    final_score = state.get("final_score", 0.0)
    
    # Sécurité 1 : Si violations est None, on bascule sur une liste vide
    violations = state.get("violations") or []

    # Formatage des violations pour le LLM
    if not violations:
        violations_text = "No violations detected. The process is fully compliant."
    else:
        formatted_errs = []
        for v in violations:
            # Sécurité 2 : On s'assure que l'élément est bien un dictionnaire
            if not isinstance(v, dict):
                continue
                
            idx = v.get("transition_index", "?")
            state_attempted = v.get("attempted_state", "?")
            
            # Sécurité 3 : Si missing_preconditions est None, on bascule sur []
            missing_list = v.get("missing_preconditions") or []
            missing = ", ".join(missing_list)
            
            formatted_errs.append(
                f"- At index {idx}, tried to activate '{state_attempted}' but missing prerequisites: [{missing}]"
            )
        violations_text = "\n".join(formatted_errs)

    # Préparation du prompt dynamique
    # .replace() plutôt que .format() : immunisé contre les accolades {} dans les variables
    user_message = (
        EXPLANATION_USER_PROMPT
        .replace("{intent_T}", str(intent_T))
        .replace("{intent_F}", str(intent_F))
        .replace("{final_score}", str(final_score))
        .replace("{violations_text}", str(violations_text))
    )

    # Configuration du fournisseur LLM ARIA sur Llama 3.3 70B
    llm = BaseLLMProvider(
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=1500,
    )

    # Invocation avec protection totale
    try:
        raw_explanation = llm.invoke(user_message=user_message)
    except Exception as e:
        print(f"⚠️  LLM invoke failed in explanation_node: {e}")
        raw_explanation = None

    # Normalisation finale — on garantit un str non-vide quoi qu'il arrive
    if not raw_explanation or not isinstance(raw_explanation, str):
        raw_explanation = "Error: The model returned an empty or malformed audit report."

    return {"explanation": raw_explanation.strip()}