"""
Node : intent_inference_F_node
Lit F (triplets de faits), produit intent_F.
Zéro accès à T — isolation stricte.
"""

from __future__ import annotations

from agent.models.base_llm_nim import BaseLLMProvider
from agent.state import NeuroSymbolicState
from agent.prompts.intent_F_prompt import INTENT_F_SYSTEM_PROMPT, INTENT_F_USER_PROMPT


def intent_inference_F_node(state: NeuroSymbolicState) -> dict:
    """
    Passe 0 — côté F.
    Infère l'intention globale du processus observé depuis les faits F.
    Ne lit pas T.
    """
    llm = BaseLLMProvider(
        system_prompt=INTENT_F_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=512,
    )

    result = llm.invoke_for_json(
        user_message=INTENT_F_USER_PROMPT,
        input_data={"F": state["F"]},
    )

    if result is None:
        return {
            "intent_F": "EXTRACTION_FAILED",
        }

    return {
        "intent_F": result.get("intent", "UNKNOWN"),
    }