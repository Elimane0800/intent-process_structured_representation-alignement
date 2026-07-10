"""
Fichier : main.py
Point d'entrée de l'Agent Neuro-Symbolique.
- Parse les scénarios séparément (gestion des branches)
- Lance une vérification par scénario
- Traces enrichies : U, Pre_mapped, état vérifié à chaque étape
"""

import pandas as pd
import time
import json
import re
from agent.graph import app


# ─────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────

def parse_scenarios(raw_text: str) -> dict:
    """
    Parse le texte brut en un dictionnaire de scénarios.
    Chaque scénario est une liste ordonnée de triplets SPO.

    Exemple d'entrée :
        Scenario 1: Successful Account Creation
        1. (Personal User, sends, account creation request)
        2. (System, verifies, account)

        Scenario 2: Country Not Accepted
        1. (Personal User, sends, account creation request)
        ...

    Retourne :
        {
          "Scenario 1: Successful Account Creation": [
              {"subject": "Personal User", "relation": "sends", "object": "account creation request"},
              ...
          ],
          ...
        }
    """
    scenarios = {}
    # Découpe sur les en-têtes de scénarios (Scenario N: ...)
    blocks = re.split(r'(Scenario\s+\d+\s*:[^\n]*)', str(raw_text))

    current_label = None
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # En-tête de scénario
        if re.match(r'Scenario\s+\d+\s*:', block):
            current_label = block
            scenarios[current_label] = []
        # Corps de scénario
        elif current_label is not None:
            triplets = re.finditer(r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)", block)
            for m in triplets:
                scenarios[current_label].append({
                    "subject":  m.group(1).strip(),
                    "relation": m.group(2).strip(),
                    "object":   m.group(3).strip(),
                })

    # Fallback : pas de marqueurs de scénario → traite tout comme un seul scénario
    if not scenarios:
        fallback = []
        for m in re.finditer(r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)", str(raw_text)):
            fallback.append({
                "subject":  m.group(1).strip(),
                "relation": m.group(2).strip(),
                "object":   m.group(3).strip(),
            })
        if fallback:
            scenarios["Scenario 1: Default"] = fallback

    return scenarios


# ─────────────────────────────────────────────
# AFFICHAGE DES TRACES ENRICHIES
# ─────────────────────────────────────────────

def print_universe(U: dict):
    """Affiche les états canoniques inférés."""
    print("\n   📚 ÉTATS CANONIQUES (U) :")
    for uid, desc in U.items():
        print(f"      • {uid}")
        print(f"        └─ {desc}")

def print_pre_mapped(Pre_mapped: dict):
    """Affiche les préconditions mappées pour chaque état."""
    print("\n   🔗 PRÉCONDITIONS MAPPÉES (Pre_mapped) :")
    for uid, preconds in Pre_mapped.items():
        if preconds:
            print(f"      • {uid}")
            for p in preconds:
                print(f"        └─ requires : {p}")
        else:
            print(f"      • {uid}  [état initial — aucune précondition]")

def print_causal_step(idx: int, activations: list, A_i: list,
                      violations: list, Pre_mapped: dict):
    """Affiche le détail d'une étape de vérification causale."""
    new_violations = [v for v in violations if v.get("transition_index") == idx]
    
    print(f"   ↳ i={idx} | A_i ({len(A_i)} états) : {A_i if A_i else '∅'}")
    
    if activations:
        print(f"      Tentatives d'activation : {activations}")
        for uid in activations:
            preconds = Pre_mapped.get(uid, [])
            missing = [p for p in preconds if p not in A_i]
            if not missing:
                print(f"      ✅ {uid} — préconditions satisfaites")
            else:
                print(f"      ❌ {uid} — manque : {missing}")
    else:
        print(f"      (aucun état canonique activé par ce fait)")
    
    if new_violations:
        print(f"      🚨 Nouvelles violations : {len(new_violations)}")


# ─────────────────────────────────────────────
# EXÉCUTION D'UN SCÉNARIO
# ─────────────────────────────────────────────

def run_scenario(scenario_label: str, F_list: list, T: str,
                 scenario_index: int, total_scenarios: int) -> dict:
    """
    Lance l'agent sur un scénario unique.
    Retourne le final_state enrichi avec le label du scénario.
    """
    print(f"\n{'='*60}")
    print(f"📋 SCÉNARIO {scenario_index}/{total_scenarios} : {scenario_label}")
    print(f"   {len(F_list)} triplets SPO")
    print(f"{'='*60}")

    initial_state = {
        "T": T,
        "F": F_list,
        "current_i": 0,
        "violations": [],
        "contaminated_states": [],
        "A_i": [],
    }

    final_state = {}
    node_name = "init"
    # On garde Pre_mapped en mémoire pour les traces causales
    current_pre_mapped = {}
    current_X = []

    try:
        for output in app.stream(initial_state):
            for node_name, state_update in output.items():

                node_time = time.time()
                print(f"\n🟢 [NŒUD] : {node_name}")

                # ── Traces enrichies selon le nœud ──────────────────────
                if node_name == "intent_T":
                    print(f"   ↳ Intention T : {state_update.get('intent_T')}")

                elif node_name == "intent_F":
                    print(f"   ↳ Intention F : {state_update.get('intent_F')}")

                elif node_name == "universe_construction":
                    U = state_update.get("U", {})
                    print(f"   ↳ {len(U)} états canoniques construits.")
                    print_universe(U)

                elif node_name == "rules_extraction":
                    Pre_text = state_update.get("Pre_text", {})
                    print(f"   ↳ Préconditions textuelles extraites pour {len(Pre_text)} états.")

                elif node_name == "precondition_mapping":
                    current_pre_mapped = state_update.get("Pre_mapped", {})
                    print(f"   ↳ Mapping embedding terminé.")
                    print_pre_mapped(current_pre_mapped)

                elif node_name == "graph_construction":
                    G_T = state_update.get("G_T", {})
                    weights = state_update.get("weights", {})
                    print(f"   ↳ G_T : {sum(len(v) for v in G_T.values())} arêtes causales.")
                    print(f"   ↳ Poids (degré sortant) :")
                    for uid, w in sorted(weights.items(), key=lambda x: -x[1]):
                        if w > 0:
                            print(f"      • {uid} → poids {w}")

                elif node_name == "projection":
                    current_X = state_update.get("X", [])
                    non_empty = sum(1 for xi in current_X if xi)
                    print(f"   ↳ {len(current_X)} faits projetés, "
                          f"{non_empty} activent au moins un état canonique.")

                elif node_name == "causal_verification":
                    idx = state_update.get("current_i", 1) - 1
                    A_i = state_update.get("A_i", [])
                    violations = state_update.get("violations", [])
                    activations = current_X[idx] if idx < len(current_X) else []
                    print_causal_step(idx, activations, A_i, violations, current_pre_mapped)

                elif node_name == "propagation":
                    contaminated = state_update.get("contaminated_states", [])
                    print(f"   ↳ {len(contaminated)} états suspects après propagation :")
                    for s in contaminated:
                        print(f"      🔴 {s}")

                elif node_name == "aggregation":
                    print(f"   ↳ Score de conformité : {state_update.get('final_score')}")

                elif node_name == "explanation":
                    print(f"   ↳ Rapport d'explication généré.")

                # ── Merge du state ───────────────────────────────────────
                final_state.update(state_update)
                print(f"   ⏱️  {time.time() - node_time:.3f}s")

    except Exception as e:
        import traceback
        print(f"\n❌ CRASH dans [{node_name}] : {e}")
        traceback.print_exc()
        return {"scenario": scenario_label, "error": str(e)}

    final_state["scenario"] = scenario_label
    return final_state


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("🚀 AGENT NEURO-SYMBOLIQUE — VÉRIFICATION PAR SCÉNARIO")
    print("="*60)

    # 1. Chargement CSV
    try:
        df = pd.read_csv("triplet_test_extraction.csv")
        print(f"✅ CSV chargé ({len(df)} lignes).")
    except FileNotFoundError:
        print("❌ 'triplet_test_extraction.csv' introuvable.")
        return

    # 2. Identification des colonnes T et F
    col_t_candidates = [
        col for col in df.columns
        if col.startswith("texte_reformule_") and not col.endswith("_extr_llama_3_3_70b")
    ]
    if not col_t_candidates:
        print("❌ Colonne T introuvable.")
        return

    col_T = col_t_candidates[0]
    col_F = f"{col_T}_extr_llama_3_3_70b"

    if col_F not in df.columns:
        print(f"❌ Colonne F introuvable : {col_F}")
        return

    # 3. Extraction ligne 0
    raw_T = str(df.iloc[0][col_T])
    raw_F_text = str(df.iloc[0][col_F])

    # 4. Parse des scénarios
    scenarios = parse_scenarios(raw_F_text)
    print(f"\n📥 Norme T : {len(raw_T)} caractères")
    print(f"📥 Scénarios détectés : {len(scenarios)}")
    for label, facts in scenarios.items():
        print(f"   • {label} — {len(facts)} triplets")

    # 5. Exécution scénario par scénario
    total_start = time.time()
    all_results = []
    n = len(scenarios)

    for i, (label, facts) in enumerate(scenarios.items(), start=1):
        if not facts:
            print(f"\n⚠️  Scénario '{label}' vide — ignoré.")
            continue
        result = run_scenario(label, facts, raw_T, i, n)
        all_results.append(result)

    total_duration = time.time() - total_start

    # 6. Synthèse globale
    print("\n" + "="*60)
    print("🏆 SYNTHÈSE GLOBALE")
    print("="*60)
    print(f"⏱️  Durée totale : {total_duration:.2f}s\n")

    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        avg_score = sum(r.get("final_score", 0) for r in valid_results) / len(valid_results)
        print(f"📊 Score moyen de conformité : {round(avg_score, 4)}")
        print(f"{'─'*60}")
        for r in valid_results:
            score = r.get("final_score", "N/A")
            nb_v = len(r.get("violations", []))
            nb_c = len(r.get("contaminated_states", []))
            label = r.get("scenario", "?")
            verdict = "✅" if isinstance(score, float) and score >= 0.8 else (
                      "⚠️" if isinstance(score, float) and score >= 0.5 else "❌")
            print(f"{verdict} [{score}] {label}")
            print(f"   Violations: {nb_v} | États contaminés: {nb_c}")

    # 7. Export JSON
    export = {
        "execution_time_seconds": round(total_duration, 2),
        "average_score": round(avg_score, 4) if valid_results else None,
        "scenarios": [
            {
                "scenario": r.get("scenario"),
                "score": r.get("final_score"),
                "violations": r.get("violations", []),
                "contaminated_states": r.get("contaminated_states", []),
                "explanation": r.get("explanation", ""),
            }
            for r in all_results
        ]
    }

    with open("rapport_audit_final.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Rapport sauvegardé dans 'rapport_audit_final.json'")


if __name__ == "__main__":
    main()