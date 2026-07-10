"""
Node : projection_node
Côté F.
Projette chaque fait de F sur l'espace canonique U via similarité cosinus.
Produit X : List[List[str]] — séquence d'états activés par chaque fait.

Seuillage adaptatif par z-score (v2) :
  Au lieu d'un seuil absolu fixe, on active les états dont la similarité
  est significativement au-dessus de la distribution des similarités du fait.
  Formule : sim(f_i, U_k) >= mu_i + lambda * sigma_i
  où mu_i, sigma_i sont la moyenne et l'écart-type des similarités de f_i
  avec tous les états de U, et lambda est l'hyperparamètre de sharpness.

  Justification scientifique :
  - Robuste à la variance lexicale inter-domaines (triplets SPO vs descriptions
    canoniques institutionnelles).
  - Seul hyperparamètre lambda est interprétable : "combien d'écarts-types
    au-dessus de la moyenne". Justifiable dans un papier A*.
  - Ablation study sur lambda triviale à produire.

  Valeur par défaut lambda=0.5 : activation libérale — on préfère
  les faux positifs aux faux négatifs à ce stade (H3 : F est bruité).
  Augmenter lambda pour plus de précision, diminuer pour plus de rappel.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.state import NeuroSymbolicState

MODEL_NAME = "all-MiniLM-L6-v2"

# Hyperparamètre de sharpness du z-score.
# lambda=0.5 : active les états à 0.5 sigma au-dessus de la moyenne.
# Augmenter (ex: 1.0, 1.5) pour plus de précision, moins de rappel.
# Diminuer (ex: 0.0) pour activer tout ce qui est au-dessus de la moyenne.
ZSCORE_LAMBDA = 0.5

# Seuil absolu plancher : même si le z-score est satisfait,
# on n'active pas un état dont la similarité brute est trop faible.
# Évite d'activer des états sur du bruit pur.
MIN_SIM_FLOOR = 0.20


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _serialize_triplet(fact: dict) -> str:
    subject  = fact.get("subject",  fact.get("Subject",  ""))
    relation = fact.get("relation", fact.get("Relation", ""))
    obj      = fact.get("object",   fact.get("Object",   ""))
    return f"{subject} {relation} {obj}".strip()


def projection_node(state: NeuroSymbolicState) -> dict:
    """
    Projette chaque fait de F sur U via z-score adaptatif.
    Produit X : List[List[str]]
    """
    U = state.get("U", {})
    F = state.get("F", [])

    if not U or not F:
        return {"X": []}

    model = SentenceTransformer(MODEL_NAME)

    u_ids        = list(U.keys())
    u_descs      = list(U.values())
    u_embeddings = model.encode(u_descs, convert_to_numpy=True)

    f_texts      = [_serialize_triplet(fact) for fact in F]
    f_embeddings = model.encode(f_texts, convert_to_numpy=True)

    X: List[List[str]] = []

    for f_idx, f_emb in enumerate(f_embeddings):

        # 1. Calcul de toutes les similarités pour ce fait
        similarities = np.array([
            _cosine_similarity(f_emb, u_emb)
            for u_emb in u_embeddings
        ])

        # 2. Statistiques de la distribution
        mu    = float(np.mean(similarities))
        sigma = float(np.std(similarities))

        # 3. Seuil adaptatif z-score
        # Si sigma ≈ 0 (tous les états sont équidistants), on prend le meilleur
        if sigma < 1e-6:
            # Cas dégénéré : on prend uniquement le meilleur match
            best_idx = int(np.argmax(similarities))
            activated = [u_ids[best_idx]] if similarities[best_idx] >= MIN_SIM_FLOOR else []
        else:
            threshold = mu + ZSCORE_LAMBDA * sigma
            activated = [
                u_ids[i]
                for i, sim in enumerate(similarities)
                if sim >= threshold and sim >= MIN_SIM_FLOOR
            ]

        X.append(activated)

    return {"X": X}