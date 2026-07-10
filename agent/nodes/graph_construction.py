"""
Node : graph_construction_node
Côté T — purement déterministe.
Lit Pre_mapped, construit G_T et weights.
Aucun LLM, aucun embedding.

v3 — Post-processing topologique :
  Après construction de G_T, on détecte et rompt les cycles via DFS.
  Heuristique de rupture : ordre causal canonique du processus.
  Un état "plus tôt" dans le processus ne peut pas dépendre d'un état
  "plus tard" — si c'est le cas, l'arête est un artefact du LLM et on
  la retire.

  Ordre canonique (index croissant = plus tôt dans le processus) :
    Request → Country → PersonalInformation → BusinessDetails
    → CreditCardDetails → VerificationEmail → Verification
    → Account → Password → ErrorNotification

  Justification formelle :
    On cherche à construire un DAG (Directed Acyclic Graph) causal.
    Toute arête u→v telle que rank(v) ≤ rank(u) est une back-edge
    dans l'ordre topologique attendu — on la supprime.
    Ce critère est interprétable, reproductible, et ne dépend d'aucun LLM.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from agent.state import NeuroSymbolicState


# ─────────────────────────────────────────────────────────────
# ORDRE CAUSAL CANONIQUE
# Chaque préfixe d'entité reçoit un rang.
# Plus le rang est faible, plus l'état est tôt dans le processus.
# Les entités absentes reçoivent un rang médian (50) par défaut.
# ─────────────────────────────────────────────────────────────
_ENTITY_RANK: Dict[str, int] = {
    "Request":             0,
    "Country":             1,
    "PersonalInformation": 2,
    "BusinessDetails":     3,
    "CreditCardDetails":   4,
    "VerificationEmail":   5,
    "Verification":        6,
    "Account":             7,
    "Password":            8,
    "ErrorNotification":   9,
}
_DEFAULT_RANK = 50


def _entity_rank(state_id: str) -> int:
    """Retourne le rang causal de l'entité d'un état 'Entity:State'."""
    entity = state_id.split(":")[0] if ":" in state_id else state_id
    return _ENTITY_RANK.get(entity, _DEFAULT_RANK)


def _remove_back_edges(
    G_T: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Passe 1 — suppression rapide des back-edges évidentes.

    Une arête source→target est une back-edge si :
        rank(source) >= rank(target)
    c'est-à-dire que la "cause" est aussi tardive ou plus tardive
    que l'"effet" — ce qui est causalement impossible.

    Retourne (G_T_clean, removed_edges).
    """
    removed: List[Tuple[str, str]] = []
    G_clean: Dict[str, List[str]] = {}

    for source, targets in G_T.items():
        kept = []
        for target in targets:
            if _entity_rank(source) < _entity_rank(target):
                kept.append(target)
            else:
                removed.append((source, target))
        G_clean[source] = kept

    return G_clean, removed


def _detect_cycles_dfs(
    G_T: Dict[str, List[str]]
) -> List[Tuple[str, str]]:
    """
    Passe 2 — détection des cycles résiduels via DFS.

    Retourne la liste des arêtes (source, target) qui forment
    des back-edges dans l'arbre DFS et créent des cycles.
    Ces arêtes seront supprimées.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in G_T}
    back_edges: List[Tuple[str, str]] = []

    def dfs(node: str):
        color[node] = GRAY
        for neighbor in G_T.get(node, []):
            if neighbor not in color:
                color[neighbor] = WHITE
            if color[neighbor] == GRAY:
                # Back-edge détectée : cycle
                back_edges.append((node, neighbor))
            elif color[neighbor] == WHITE:
                dfs(neighbor)
        color[node] = BLACK

    for node in list(G_T.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node)

    return back_edges


def _break_cycles(
    G_T: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Passe 2 — suppression des back-edges DFS résiduelles.

    Pour chaque back-edge (u, v) détectée :
    - Si rank(u) >= rank(v) : on retire u→v  (heuristique rang)
    - Sinon : on retire u→v quand même        (back-edge DFS = cycle certain)

    Itère jusqu'à convergence (le graphe est acyclique).
    """
    all_removed: List[Tuple[str, str]] = []
    max_iterations = 10  # sécurité anti-boucle infinie

    for _ in range(max_iterations):
        back_edges = _detect_cycles_dfs(G_T)
        if not back_edges:
            break
        for (src, tgt) in back_edges:
            if tgt in G_T.get(src, []):
                G_T[src].remove(tgt)
                all_removed.append((src, tgt))

    return G_T, all_removed


def _sync_pre_mapped(
    Pre_mapped: Dict[str, List[str]],
    removed_edges: List[Tuple[str, str]],
) -> Dict[str, List[str]]:
    """
    Synchronise Pre_mapped avec les arêtes supprimées de G_T.

    Une arête supprimée Uj→Uk signifie que Uj n'est plus précondition de Uk.
    On retire donc Uj de Pre_mapped[Uk].
    """
    # removed_edges : (source=Uj, target=Uk)
    # Pre_mapped[Uk] contient Uj → on retire Uj
    removed_set: Set[Tuple[str, str]] = set(removed_edges)
    cleaned: Dict[str, List[str]] = {}

    for target, preconditions in Pre_mapped.items():
        cleaned[target] = [
            p for p in preconditions
            if (p, target) not in removed_set
        ]

    return cleaned


# ─────────────────────────────────────────────────────────────
# NODE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def graph_construction_node(state: NeuroSymbolicState) -> dict:
    """
    Construit le graphe causal institutionnel G_T depuis Pre_mapped,
    puis le rend acyclique via post-processing topologique.

    G_T : Dict[str, List[str]]
        clé   = état source Uj
        valeur = liste des états cibles Uk pour lesquels Uj est précondition
        Lecture : "Uj → Uk" = Uj doit être activé avant Uk.

    weights : Dict[str, int]
        degré sortant de chaque état dans G_T (criticité causale).
    """
    Pre_mapped = state.get("Pre_mapped", {})

    if not Pre_mapped:
        return {"G_T": {}, "weights": {}, "Pre_mapped": {}}

    # ── Étape 1 : Construction brute de G_T ──────────────────
    G_T: Dict[str, List[str]] = defaultdict(list)

    for target_state, preconditions in Pre_mapped.items():
        for source_state in preconditions:
            if target_state not in G_T[source_state]:
                G_T[source_state].append(target_state)

    for state_id in Pre_mapped:
        if state_id not in G_T:
            G_T[state_id] = []

    G_T = dict(G_T)

    n_edges_raw = sum(len(v) for v in G_T.values())

    # ── Étape 2 : Suppression des back-edges par rang ────────
    # Arêtes où rank(source) >= rank(target) → causalement impossibles
    G_T, removed_rank = _remove_back_edges(G_T)

    # ── Étape 3 : Suppression des cycles résiduels (DFS) ─────
    # Cycles qui survivent après le filtrage par rang
    G_T, removed_dfs = _break_cycles(G_T)

    all_removed = removed_rank + removed_dfs
    n_edges_clean = sum(len(v) for v in G_T.values())

    # ── Étape 4 : Synchronisation de Pre_mapped ──────────────
    # On retire de Pre_mapped les préconditions correspondant
    # aux arêtes supprimées — cohérence garantie entre G_T et Pre_mapped
    Pre_mapped_clean = _sync_pre_mapped(Pre_mapped, all_removed)

    # ── Étape 5 : Calcul des weights ─────────────────────────
    weights: Dict[str, int] = {
        state_id: len(successors)
        for state_id, successors in G_T.items()
    }

    # ── Debug log ────────────────────────────────────────────
    print(f"\n   🔧 [graph_construction] Arêtes brutes   : {n_edges_raw}")
    print(f"   🔧 [graph_construction] Supprimées (rang): {len(removed_rank)}")
    print(f"   🔧 [graph_construction] Supprimées (DFS) : {len(removed_dfs)}")
    print(f"   🔧 [graph_construction] Arêtes finales   : {n_edges_clean}")
    if all_removed:
        print(f"   🔧 Arêtes retirées :")
        for src, tgt in sorted(all_removed):
            print(f"      ✂️  {src} → {tgt}")

    return {
        "G_T":       G_T,
        "weights":   weights,
        "Pre_mapped": Pre_mapped_clean,  # version nettoyée, sans cycles
    }