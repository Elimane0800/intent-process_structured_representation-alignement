"""
Node : precondition_mapping_node
Côté T.
Projette les préconditions textuelles de Pre_text sur U via z-score adaptatif.
Produit Pre_mapped : Dict[str, List[str]]

Même logique z-score que projection_node (v2) pour cohérence.
Lambda légèrement plus élevé (0.8) car les préconditions sont plus précises
sémantiquement que les triplets SPO bruités — on veut moins de faux positifs.

Post-processing déterministe (v2) :
  Après le mapping embedding, on filtre les préconditions mutuellement
  exclusives : si deux états de la même entité se retrouvent tous les deux
  préconditions du même état cible, on retire l'état d'échec/rejet.
  Règle : si "Entity:StateA" et "Entity:StateB" sont préconditions de Uk,
  et que StateA et StateB sont complémentaires (ex: Accepted/Rejected),
  on ne garde que le positif (Accepted, Selected, Provided, etc.).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.state import NeuroSymbolicState

MODEL_NAME        = "all-MiniLM-L6-v2"
ZSCORE_LAMBDA     = 0.8   # Plus strict que la projection (moins de faux positifs)
MIN_SIM_FLOOR     = 0.20  # Plancher absolu

# Suffixes considérés comme "négatifs" — états d'échec ou de rejet
# Un état avec ce suffixe ne peut pas être précondition d'un état "positif"
# si son complémentaire positif est aussi candidat.
_NEGATIVE_SUFFIXES = {
    "Rejected", "Failed", "NotAccepted", "NotSelected",
    "NotProvided", "NotSubmitted", "NotCreated", "NotVerified",
    "NotSent", "Error", "VerificationFailed", "CreationFailed",
    "RegistrationFailed",
}

_POSITIVE_SUFFIXES = {
    "Accepted", "Selected", "Provided", "Submitted", "Created",
    "Verified", "Sent", "Requested", "Entered", "Generated",
    "Registered",
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _get_entity(state_id: str) -> str:
    """Extrait l'entité d'un identifiant 'Entity:State'."""
    return state_id.split(":")[0] if ":" in state_id else state_id


def _get_state_suffix(state_id: str) -> str:
    """Extrait le suffixe d'état d'un identifiant 'Entity:State'."""
    return state_id.split(":")[1] if ":" in state_id else ""


def _is_negative(state_id: str) -> bool:
    suffix = _get_state_suffix(state_id)
    return any(suffix.endswith(neg) for neg in _NEGATIVE_SUFFIXES)


def _is_positive(state_id: str) -> bool:
    suffix = _get_state_suffix(state_id)
    return any(suffix.endswith(pos) for pos in _POSITIVE_SUFFIXES)


def _filter_mutually_exclusive(
    mapped: List[str],
    target_state: str,
) -> List[str]:
    """
    Post-processing déterministe.
    Pour chaque entité représentée dans 'mapped', si on a à la fois
    un état positif ET un état négatif de cette entité, on retire le négatif.
    On retire aussi l'état cible lui-même s'il s'y trouve (auto-référence).
    """
    # Retirer l'auto-référence
    mapped = [s for s in mapped if s != target_state]

    # Grouper par entité
    by_entity: Dict[str, List[str]] = {}
    for s in mapped:
        entity = _get_entity(s)
        by_entity.setdefault(entity, []).append(s)

    result = []
    for entity, states in by_entity.items():
        has_positive = any(_is_positive(s) for s in states)
        for s in states:
            # Si on a un positif ET un négatif pour la même entité,
            # on retire le négatif
            if has_positive and _is_negative(s):
                continue
            result.append(s)

    return result


def precondition_mapping_node(state: NeuroSymbolicState) -> dict:
    """
    Pour chaque état Uk dans Pre_text, projette ses préconditions
    textuelles sur U via z-score adaptatif + post-processing déterministe.
    """
    U        = state.get("U", {})
    Pre_text = state.get("Pre_text", {})

    if not U or not Pre_text:
        return {"Pre_mapped": {}}

    model = SentenceTransformer(MODEL_NAME)

    u_ids        = list(U.keys())
    u_descs      = list(U.values())
    u_embeddings = model.encode(u_descs, convert_to_numpy=True)

    Pre_mapped: Dict[str, List[str]] = {}

    for state_id, preconditions in Pre_text.items():

        # État initial — pas de préconditions
        if not preconditions:
            Pre_mapped[state_id] = []
            continue

        pre_embeddings = model.encode(preconditions, convert_to_numpy=True)

        mapped: List[str] = []

        for pre_emb in pre_embeddings:
            similarities = np.array([
                _cosine_similarity(pre_emb, u_emb)
                for u_emb in u_embeddings
            ])

            mu    = float(np.mean(similarities))
            sigma = float(np.std(similarities))

            if sigma < 1e-6:
                best_idx = int(np.argmax(similarities))
                candidates = (
                    [u_ids[best_idx]]
                    if similarities[best_idx] >= MIN_SIM_FLOOR
                    else []
                )
            else:
                threshold  = mu + ZSCORE_LAMBDA * sigma
                candidates = [
                    u_ids[i]
                    for i, sim in enumerate(similarities)
                    if sim >= threshold and sim >= MIN_SIM_FLOOR
                ]

            for uid in candidates:
                if uid not in mapped:
                    mapped.append(uid)

        # Post-processing déterministe : retirer mutuellement exclusifs
        mapped = _filter_mutually_exclusive(mapped, state_id)

        Pre_mapped[state_id] = mapped

    return {"Pre_mapped": Pre_mapped}