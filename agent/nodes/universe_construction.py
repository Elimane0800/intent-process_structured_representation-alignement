"""
Node : universe_construction_node
Passe 1 — côté T.
Lit T, produit U : le dictionnaire des états canoniques du processus.
Zéro accès à F — isolation stricte.
"""

from __future__ import annotations

from agent.models.base_llm_nim import BaseLLMProvider
from agent.state import NeuroSymbolicState
from agent.prompts.universe_construction_prompt import (
    UNIVERSE_CONSTRUCTION_SYSTEM_PROMPT,
    UNIVERSE_CONSTRUCTION_USER_PROMPT,
)


def universe_construction_node(state: NeuroSymbolicState) -> dict:
    """
    Passe 1 — côté T.
    Extrait l'espace d'états canoniques U depuis le texte normatif T.
    U est statique : aucune transition causale n'est définie ici.
    """
    llm = BaseLLMProvider(
        system_prompt=UNIVERSE_CONSTRUCTION_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=2048,
    )

    result = llm.invoke_for_json(
        user_message=UNIVERSE_CONSTRUCTION_USER_PROMPT,
        input_data={"T": state["T"]},
    )

    if result is None or "U" not in result:
        return {"U": {}}

    U = result["U"]

    if not isinstance(U, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in U.items()
    ):
        return {"U": {}}

    return {"U": U}