"""
Node : intent_inference_T_node
Lit T (texte normatif), produit intent_T.
Zéro accès à F — isolation stricte.
"""

from __future__ import annotations

from agent.models.base_llm_nim import BaseLLMProvider
from agent.state import NeuroSymbolicState
from agent.prompts.intent_T_prompt import INTENT_T_SYSTEM_PROMPT, INTENT_T_USER_PROMPT


def intent_inference_T_node(state: NeuroSymbolicState) -> dict:
    """
    Passe 0 — côté T.
    Infère l'intention globale et le résultat attendu du texte normatif T.
    Ne lit pas F.
    """
    llm = BaseLLMProvider(
        system_prompt=INTENT_T_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=512,
    )

    result = llm.invoke_for_json(
        user_message=INTENT_T_USER_PROMPT,
        input_data={"T": state["T"]},
    )

    if result is None:
        return {
            "intent_T": "EXTRACTION_FAILED",
        }

    return {
        "intent_T": result.get("intent", "UNKNOWN"),
    }