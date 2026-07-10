from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict

class NeuroSymbolicState(TypedDict, total=False):
    # INPUTS
    T: str
    F: List[Dict[str, str]]

    # INFÉRENCE MACROSCOPIQUE
    intent_T: str
    intent_F: str
    macro_conformance: Optional[bool]

    # CÔTÉ T
    U: Dict[str, str]
    Pre_text: Dict[str, List[str]]
    Pre_mapped: Dict[str, List[str]]
    G_T: Dict[str, List[str]]
    weights: Dict[str, int]

    # CÔTÉ F
    X: List[List[str]]

    # VÉRIFICATION CAUSALE
    current_i: int
    A_i: List[str]

    # RÉSULTATS
    violations: List[Dict[str, Any]]
    contaminated_states: List[str]
    final_score: float
    explanation: str