"""
Node : rules_extraction_node
Passe 2 — côté T.
Lit T et U, produit Pre_text :
les préconditions institutionnelles de chaque état, en langage naturel.
Zéro accès à F — isolation stricte.
"""

from __future__ import annotations

from agent.models.base_llm_nim import BaseLLMProvider
from agent.state import NeuroSymbolicState
from agent.prompts.rules_extraction_prompt import (
    RULES_EXTRACTION_SYSTEM_PROMPT,
    RULES_EXTRACTION_USER_PROMPT,
)


def rules_extraction_node(state: NeuroSymbolicState) -> dict:
    """
    Passe 2 — côté T.
    Pour chaque état Uk dans U, extrait depuis T les préconditions
    institutionnelles en langage naturel.
    Produit Pre_text : Dict[str, List[str]]
    clé   = identifiant canonique (ex: "Manuscript:Accepted")
    valeur = liste de préconditions textuelles
    """
    if not state.get("U"):
        return {"Pre_text": {}}

    llm = BaseLLMProvider(
        system_prompt=RULES_EXTRACTION_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=4096,
    )

    result = llm.invoke_for_json(
        user_message=RULES_EXTRACTION_USER_PROMPT,
        input_data={
            "T": state["T"],
            "U": state["U"],
        },
    )

    if result is None or "Pre_text" not in result:
        return {"Pre_text": {}}

    Pre_text = result["Pre_text"]

    # Validation : toutes les clés doivent être dans U
    # et les valeurs doivent être des listes de strings
    U_keys = set(state["U"].keys())
    validated = {}
    for state_id, preconditions in Pre_text.items():
        if state_id not in U_keys:
            continue
        if not isinstance(preconditions, list):
            continue
        cleaned = [p for p in preconditions if isinstance(p, str) and p.strip()]
        validated[state_id] = cleaned

    # Les états sans préconditions reçoivent une liste vide
    # (états initiaux — atteignables sans condition préalable)
    for state_id in U_keys:
        if state_id not in validated:
            validated[state_id] = []

    return {"Pre_text": validated}