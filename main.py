"""
Fichier : main.py
Point d'entrée de l'Agent Neuro-Symbolique.

Architecture deux phases :
  Phase 1 (graph_T) — une seule fois : construction de la référence normative
  Phase 2 (graph_F) — une fois par scénario : vérification causale

Gain : 9x moins d'appels LLM et d'embeddings sur U.
"""

import pandas as pd
import time
import json
import re

from agent.graph import graph_T, graph_F


# ─────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────

def parse_scenarios(raw_text: str) -> dict:
    scenarios = {}
    blocks = re.split(r'(Scenario\s+\d+\s*:[^\n]*)', str(raw_text))
    current_label = None
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if re.match(r'Scenario\s+\d+\s*:', block):
            current_label = block
            scenarios[current_label] = []
        elif current_label is not None:
            for m in re.finditer(r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)", block):
                scenarios[current_label].append({
                    "subject":  m.group(1).strip(),
                    "relation": m.group(2).strip(),
                    "object":   m.group(3).strip(),
                })
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
# AFFICHAGE TRACES
# ─────────────────────────────────────────────

def print_universe(U: dict):
    print("\n   📚 ÉTATS CANONIQUES (U) :")
    for uid, desc in U.items():
        print(f"      • {uid}")
        print(f"        └─ {desc}")

def print_pre_mapped(Pre_mapped: dict):
    print("\n   🔗 PRÉCONDITIONS MAPPÉES (Pre_mapped) :")
    for uid, preconds in Pre_mapped.items():
        if preconds:
            print(f"      • {uid}")
            for p in preconds:
                print(f"        └─ requires : {p}")
        else:
            print(f"      • {uid}  [état initial — aucune précondition]")

def print_causal_step(idx, activations, A_i, violations, Pre_mapped):
    new_v = [v for v in violations if v.get("transition_index") == idx]
    print(f"   ↳ i={idx} | A_i ({len(A_i)}) : {A_i if A_i else '∅'}")
    if activations:
        for uid in activations:
            preconds = set(Pre_mapped.get(uid, []))
            missing  = preconds - set(A_i)
            icon = "✅" if not missing else "❌"
            print(f"      {icon} {uid}" + (f" — manque : {sorted(missing)}" if missing else ""))
    else:
        print("      (aucun état canonique activé par ce fait)")
    if new_v:
        print(f"      🚨 {len(new_v)} nouvelle(s) violation(s)")


# ─────────────────────────────────────────────
# PHASE 1 — CONSTRUCTION NORMATIVE (une seule fois)
# ─────────────────────────────────────────────

def run_phase_T(T: str) -> dict:
    print("\n" + "="*60)
    print("🏗️  PHASE 1 — CONSTRUCTION DE LA RÉFÉRENCE NORMATIVE (T)")
    print("="*60)

    initial = {"T": T, "F": [], "current_i": 0,
               "violations": [], "contaminated_states": [], "A_i": []}

    state_T = {}
    node_name = "init"

    try:
        for output in graph_T.stream(initial):
            for node_name, update in output.items():
                print(f"\n🟢 [NŒUD-T] : {node_name}")

                if node_name == "intent_T":
                    print(f"   ↳ Intention T : {update.get('intent_T')}")

                elif node_name == "universe_construction":
                    U = update.get("U", {})
                    print(f"   ↳ {len(U)} états canoniques construits.")
                    print_universe(U)

                elif node_name == "rules_extraction":
                    print(f"   ↳ Préconditions extraites pour "
                          f"{len(update.get('Pre_text', {}))} états.")

                elif node_name == "precondition_mapping":
                    print(f"   ↳ Mapping embedding terminé.")
                    print_pre_mapped(update.get("Pre_mapped", {}))

                elif node_name == "graph_construction":
                    G_T     = update.get("G_T", {})
                    weights = update.get("weights", {})
                    n_edges = sum(len(v) for v in G_T.values())
                    print(f"   ↳ G_T : {n_edges} arêtes causales.")
                    print("   ↳ Poids critiques :")
                    for uid, w in sorted(weights.items(), key=lambda x: -x[1])[:8]:
                        if w > 0:
                            print(f"      • {uid} → {w}")

                state_T.update(update)

    except Exception as e:
        import traceback
        print(f"\n❌ CRASH Phase T [{node_name}] : {e}")
        traceback.print_exc()

    return state_T


# ─────────────────────────────────────────────
# PHASE 2 — VÉRIFICATION D'UN SCÉNARIO
# ─────────────────────────────────────────────

def run_phase_F(scenario_label: str, F_list: list,
                state_T: dict, idx: int, total: int) -> dict:

    print(f"\n{'='*60}")
    print(f"📋 SCÉNARIO {idx}/{total} : {scenario_label}")
    print(f"   {len(F_list)} triplets SPO")
    print(f"{'='*60}")

    # On injecte le state T complet + les données du scénario
    initial = {
        **state_T,
        "F":                  F_list,
        "current_i":          0,
        "violations":         [],
        "contaminated_states":[],
        "A_i":                [],
    }

    state_F   = {}
    node_name = "init"
    current_X = state_T.get("X", [])   # sera mis à jour après projection

    try:
        for output in graph_F.stream(initial):
            for node_name, update in output.items():
                print(f"\n🟢 [NŒUD-F] : {node_name}")

                if node_name == "intent_F":
                    print(f"   ↳ Intention F : {update.get('intent_F')}")

                elif node_name == "projection":
                    current_X = update.get("X", [])
                    non_empty = sum(1 for xi in current_X if xi)
                    print(f"   ↳ {len(current_X)} faits projetés, "
                          f"{non_empty} activent au moins un état.")

                elif node_name == "causal_verification":
                    i        = update.get("current_i", 1) - 1
                    A_i      = update.get("A_i", [])
                    viols    = update.get("violations", [])
                    acts     = current_X[i] if i < len(current_X) else []
                    Pre_mapped = state_T.get("Pre_mapped", {})
                    print_causal_step(i, acts, A_i, viols, Pre_mapped)

                elif node_name == "propagation":
                    cont = update.get("contaminated_states", [])
                    print(f"   ↳ {len(cont)} états suspects après propagation.")
                    for s in cont:
                        print(f"      🔴 {s}")

                elif node_name == "aggregation":
                    print(f"   ↳ Score : {update.get('final_score')}")

                elif node_name == "explanation":
                    print(f"   ↳ Rapport généré.")

                state_F.update(update)

    except Exception as e:
        import traceback
        print(f"\n❌ CRASH Phase F [{node_name}] : {e}")
        traceback.print_exc()
        return {"scenario": scenario_label, "error": str(e)}

    state_F["scenario"] = scenario_label
    return state_F


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("🚀 AGENT NEURO-SYMBOLIQUE — ARCHITECTURE DEUX PHASES")
    print("="*60)

    try:
        df = pd.read_csv("triplet_test_extraction.csv")
        print(f"✅ CSV chargé ({len(df)} lignes).")
    except FileNotFoundError:
        print("❌ 'triplet_test_extraction.csv' introuvable.")
        return

    col_t_candidates = [
        c for c in df.columns
        if c.startswith("texte_reformule_") and not c.endswith("_extr_llama_3_3_70b")
    ]
    if not col_t_candidates:
        print("❌ Colonne T introuvable.")
        return

    col_T = col_t_candidates[0]
    col_F = f"{col_T}_extr_llama_3_3_70b"

    if col_F not in df.columns:
        print(f"❌ Colonne F introuvable : {col_F}")
        return

    raw_T      = str(df.iloc[0][col_T])
    raw_F_text = str(df.iloc[0][col_F])

    scenarios = parse_scenarios(raw_F_text)
    print(f"\n📥 Norme T      : {len(raw_T)} caractères")
    print(f"📥 Scénarios F  : {len(scenarios)}")
    for label, facts in scenarios.items():
        print(f"   • {label} — {len(facts)} triplets")

    total_start = time.time()

    # ── Phase 1 : une seule fois ──────────────────────────────
    t0     = time.time()
    state_T = run_phase_T(raw_T)
    print(f"\n⏱️  Phase T terminée en {time.time()-t0:.1f}s")

    # ── Phase 2 : une fois par scénario ──────────────────────
    all_results = []
    n = len(scenarios)

    for i, (label, facts) in enumerate(scenarios.items(), start=1):
        if not facts:
            print(f"\n⚠️  '{label}' vide — ignoré.")
            continue
        t0 = time.time()
        result = run_phase_F(label, facts, state_T, i, n)
        print(f"\n⏱️  Scénario {i} terminé en {time.time()-t0:.1f}s")
        all_results.append(result)

    total_duration = time.time() - total_start

    # ── Synthèse ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("🏆 SYNTHÈSE GLOBALE")
    print("="*60)
    print(f"⏱️  Durée totale : {total_duration:.1f}s\n")

    valid  = [r for r in all_results if "error" not in r]
    if valid:
        avg    = sum(r.get("final_score", 0) for r in valid) / len(valid)
        print(f"📊 Score moyen : {round(avg, 4)}")
        print("─"*60)
        for r in valid:
            score = r.get("final_score", "N/A")
            nv    = len(r.get("violations", []))
            nc    = len(r.get("contaminated_states", []))
            lbl   = r.get("scenario", "?")
            icon  = ("✅" if isinstance(score, float) and score >= 0.8
                     else "⚠️"  if isinstance(score, float) and score >= 0.5
                     else "❌")
            print(f"{icon} [{score}] {lbl}")
            print(f"   Violations: {nv} | Contaminés: {nc}")

    # ── Export JSON ───────────────────────────────────────────
    export = {
        "execution_time_seconds": round(total_duration, 2),
        "average_score": round(avg, 4) if valid else None,
        "normative_reference": {
            "intent_T":   state_T.get("intent_T"),
            "nb_states":  len(state_T.get("U", {})),
            "nb_edges":   sum(len(v) for v in state_T.get("G_T", {}).values()),
        },
        "scenarios": [
            {
                "scenario":            r.get("scenario"),
                "score":               r.get("final_score"),
                "intent_F":            r.get("intent_F"),
                "violations":          r.get("violations", []),
                "contaminated_states": r.get("contaminated_states", []),
                "explanation":         r.get("explanation", ""),
            }
            for r in all_results
        ]
    }

    with open("rapport_audit_final.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Rapport sauvegardé dans 'rapport_audit_final.json'")


if __name__ == "__main__":
    main()