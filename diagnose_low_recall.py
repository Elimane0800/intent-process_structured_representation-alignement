"""Diagnostic rapide : pourquoi le rappel sur les operateurs required (remove_activity,
swap_labels) est-il si bas sur ce cas ? Deux hypotheses a trancher, pas a deviner :
(1) le graphe de reference de CE cas est structurellement pauvre en SATISFIED (peu de candidats
    a faire basculer, rappel bas attendu, pas un defaut du mecanisme) ;
(2) un vrai probleme (mutation qui tombe sur une zone non couverte malgre le correctif de
    ciblage, ou une regression introduite ailleurs).

Usage : python3 diagnose_low_recall.py <model_key>
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(__file__)
SINGLE_CASE_DIR = os.path.join(BASE_DIR, "results", "single_case_validation")
MANIFEST_PATH = os.path.join(BASE_DIR, "results", "mutants_single", "manifest.json")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "perturbation_study_single", "results.json")


def main(model: str) -> None:
    baseline_path = os.path.join(SINGLE_CASE_DIR, model, "E_j01__0.bpmn2.xml.json")
    with open(baseline_path) as f:
        baseline = json.load(f)

    alignment = baseline.get("alignment", {})
    reference_graph = baseline.get("reference_graph", {})
    matches = baseline.get("matches", {})

    print(f"=== Baseline [{model}] ===")
    print(f"reference_graph: {len(reference_graph.get('nodes', []))} nodes, "
          f"{len(reference_graph.get('edges', []))} edges")
    print(f"matches: {len(matches)} activites appariees")

    status_counts = {"SATISFIED": 0, "VIOLATED": 0, "UNRESOLVABLE": 0}
    satisfied_terms = []
    for target, entry in alignment.items():
        for t in entry["terms"]:
            status_counts[t["status"]] = status_counts.get(t["status"], 0) + 1
            if t["status"] == "SATISFIED":
                satisfied_terms.append((t["term"], target))

    print(f"Statuts au baseline (niveau terme) : {status_counts}")
    print(f"  -> {len(satisfied_terms)} terme(s) SATISFIED au total "
          f"(seuls ceux-la peuvent basculer et compter comme 'detecte')")
    if satisfied_terms:
        print("  Exemples de termes SATISFIED (term -> target) :")
        for term, target in satisfied_terms[:10]:
            print(f"    {term} -> {target}")

    # Quelles activites ont ete mutees par remove_activity/swap_labels, et ces activites
    # ancrent-elles un milestone qui apparait dans un des termes SATISFIED ci-dessus ?
    satisfied_milestones = {term for term, _ in satisfied_terms} | {tgt for _, tgt in satisfied_terms}
    reverse_matches = {a: m["match"] for a, m in matches.items() if isinstance(m, dict)}

    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        print(f"\n=== Mutants required (manifest) ===")
        for m in manifest:
            if m["operator"] not in ("remove_activity", "swap_labels") or not m["applicable"]:
                continue
            details = m.get("details", {})
            mutated_activities = (
                [details.get("removed_activity")] if m["operator"] == "remove_activity"
                else details.get("swapped", [])
            )
            touches_satisfied = any(
                reverse_matches.get(a) in satisfied_milestones for a in mutated_activities if a
            )
            print(f"  {m['operator']:16} variant={m['variant']}  active(s) mutee(s)={mutated_activities}  "
                  f"touche un terme SATISFIED : {touches_satisfied}")

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)["results"]
        model_results = all_results.get(model, [])
        print(f"\n=== Verdicts perturbation study [{model}] ===")
        for e in model_results:
            if e["operator"] in ("remove_activity", "swap_labels"):
                print(f"  {e['operator']:16} variant={e['variant']}  detected={e.get('detected')}  "
                      f"error={e.get('error')}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b"
    main(model)