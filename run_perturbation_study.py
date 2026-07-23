"""Etude de perturbation controlee : evalue chaque mutant genere par mutate_bpmn.py contre la
reference U/G FIGEE du run v2 correspondant, produit precision/rappel par operateur et le
tableau angle-mort/faux-positifs qui repond au rehibitoire "aucune verite terrain sur les
verdicts".

Principe de controle experimental (deja discute) : SEULE la branche droite (bpmn_to_spo ->
state_matching -> alignment) est rejouee sur chaque mutant. U (state_space) et G
(reference_graph) viennent tels quels du resultat v2 deja produit sur le fichier BPMN de
base -- jamais recalcules. Le texte n'a pas change ; seule la variable manipulee (le BPMN) doit
varier, sinon un changement de verdict ne serait plus attribuable a la mutation (bruit de
non-reproductibilite LLM en plus, cf. pipeline_observations.md limite 3).

Definition operationnelle de "detection" (niveau cas, pas niveau terme) : un mutant est
"detecte" si au moins un couple (target, term) bascule de SATISFIED vers VIOLATED ou vers
UNRESOLVABLE. Ce niveau de granularite est deliberement le meme que le test de controle manuel
deja fait (pipeline_observations.md section 5 : "deux cibles basculent en VIOLATED") -- pas une
localisation precise du terme affecte, hors scope de cette premiere etude.

Trois classes d'operateurs, trois metriques distinctes (cf. mutate_bpmn.py, section taxonomie) :
- required   (remove_activity, swap_labels, cross_case_replace) -> RAPPEL de detection
- forbidden  (insert_activity, shuffle_xml)                     -> TAUX DE FAUX POSITIFS
- blind_spot (rewire_gateway)                                    -> verification structurelle
  directe (le DFG doit etre BYTE-IDENTIQUE au DFG de base -- 'follows' generique, gateway_context
  jamais dans le triplet, decision actee) ; pas d'appel embedding necessaire pour ce cas, le
  controle est mecanique et gratuit.
"""

import json
import os
import re

from dotenv import load_dotenv

# Charge explicitement .env -- ce script importe bpmn_to_spo_node/state_matching_node/
# alignment_node mais JAMAIS agent.models.base_llm (aucun appel LLM necessaire, cf. docstring
# de tete). Or c'est base_llm.py qui appelle load_dotenv() au chargement -- dans run.py, cet
# effet de bord suffit car graph.py importe base_llm avant que state_matching (donc embed())
# ne s'execute. Ici, sans cet import, os.environ ne contient jamais NVIDIA_API_KEY et
# base_nvidia_embedding.embed() leve un KeyError. Ne pas dependre d'un effet de bord d'import
# d'un module non lie -- charger l'environnement explicitement, ici, pour ce script autonome.
load_dotenv()

from agent.nodes.bpmn_to_spo_node import parse_bpmn, resolve_dfg, dfg_to_spo
from agent.nodes.state_matching_node import match_spo_to_u
from agent.nodes.alignment_node import check_alignment

BASE_DIR = os.path.dirname(__file__)
MUTANTS_DIR = os.path.join(BASE_DIR, "results", "mutants")
V2_RESULTS_DIR = os.path.join(BASE_DIR, "results", "dataset_runs_v2")
OUT_DIR = os.path.join(BASE_DIR, "results", "perturbation_study")

# Modeles evalues -- par defaut ceux du run v2 (agent.config.MODELS_TO_RUN au moment de ce run).
# Explicite ici plutot qu'importe : l'etude doit rester reproductible meme si MODELS_TO_RUN
# change plus tard dans le projet pour d'autres runs.
MODELS = ["llama-3.1-8b", "mistral-nemotron", "gpt-oss-20b"]

REQUIRED_OPS = {"remove_activity", "swap_labels", "cross_case_replace"}
FORBIDDEN_OPS = {"insert_activity", "shuffle_xml"}
BLIND_SPOT_OPS = {"rewire_gateway"}


def _v2_result_path(model: str, base_key: str) -> str:
    desc, fname = base_key.split("/", 1)
    return os.path.join(V2_RESULTS_DIR, model, f"{desc}__{fname}.json")


def load_baseline(model: str, base_key: str) -> dict | None:
    """Charge le resultat v2 deja produit pour ce (modele, fichier de base) -- fournit U, G, et
    les aretes DFG de base (pour le controle structurel rapide de rewire_gateway). None si le
    fichier est absent ou si le run de base a echoue avant alignment (pas de reference
    exploitable -- rien a comparer, le cas est signale et saute, jamais suppose)."""
    path = _v2_result_path(model, base_key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    required = ("state_space", "reference_graph", "alignment", "dfg_edges")
    if any(k not in state or not state[k] for k in required):
        return None
    return state


def status_map(alignment: dict) -> dict:
    """(target, term) -> status, a plat -- forme comparable terme a terme entre baseline et
    mutant, independante de l'ordre ou du regroupement par cible."""
    return {
        (target, t["term"]): t["status"]
        for target, entry in alignment.items()
        for t in entry["terms"]
    }


def run_mutant_alignment(mutant_path: str, state_space: dict, reference_graph: dict) -> dict:
    """Rejoue uniquement la branche droite sur le mutant. Necessite un appel embedding
    (match_spo_to_u) -- seul cout reel de cette etude, U et G restant figes."""
    bpmn = parse_bpmn(mutant_path)
    dfg = resolve_dfg(bpmn)
    spo = dfg_to_spo(dfg)
    spo_tuples = [(t["subject"], t["predicate"], t["object"]) for t in spo]
    activities = sorted({a for s, _, o in spo_tuples for a in (s, o) if a and a.strip()})
    matches, process_graph = match_spo_to_u(spo_tuples, state_space)
    alignment = check_alignment(reference_graph, process_graph, matches)
    return {"dfg_edges": dfg, "alignment": alignment}


def dfg_edge_set(dfg_edges: list[dict]) -> set:
    return {
        ((e["source_label"] or "").strip(), (e["target_label"] or "").strip())
        for e in dfg_edges
    }


def classify_required(baseline_status: dict, mutant_status: dict) -> bool:
    """Detecte : au moins un couple present dans les deux ETAIT SATISFIED et ne l'est plus."""
    for key, base_s in baseline_status.items():
        if base_s == "SATISFIED" and mutant_status.get(key) not in ("SATISFIED", None):
            return True
    return False


def classify_forbidden(baseline_status: dict, mutant_status: dict) -> dict:
    """Deux signaux distincts pour les operateurs preservant le comportement :
    - false_positive_violated : un SATISFIED est devenu VIOLATED -- viole directement le
      Theoreme de monotonie en enrichissement (TGMS, section 3.3) ; c'est la mesure stricte.
    - any_regression : un SATISFIED est devenu autre chose (VIOLATED ou UNRESOLVABLE) -- signal
      plus large, attendu a zero pour shuffle_xml (invariance pure, aucune structure ajoutee),
      tolerable en theorie pour insert_activity SEULEMENT si un chemin est rompu par erreur
      d'implementation (ne devrait normalement jamais arriver non plus, cf. docstring de
      mutate_insert_activity : le pont A->New->B preserve toute atteignabilite existante)."""
    fp_violated = False
    any_regression = False
    for key, base_s in baseline_status.items():
        mut_s = mutant_status.get(key)
        if base_s == "SATISFIED" and mut_s not in ("SATISFIED", None):
            any_regression = True
            if mut_s == "VIOLATED":
                fp_violated = True
    return {"false_positive_violated": fp_violated, "any_regression": any_regression}


def run_study(models: list[str] = MODELS, out_dir: str = OUT_DIR) -> dict:
    manifest_path = os.path.join(MUTANTS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"[perturbation] {manifest_path} introuvable -- lancer "
                          f"mutate_bpmn.py d'abord")
    with open(manifest_path) as f:
        manifest = json.load(f)
    applicable = [m for m in manifest if m["applicable"]]
    print(f"[perturbation] {len(applicable)} mutants applicables dans le manifest, "
          f"{len(models)} modele(s)")

    os.makedirs(out_dir, exist_ok=True)
    results = {model: [] for model in models}
    skipped = {model: [] for model in models}

    for model in models:
        print(f"\n########## Modele : {model} ##########")
        baseline_cache: dict[str, dict | None] = {}

        for m in applicable:
            base_key = m["base"]
            if base_key not in baseline_cache:
                baseline_cache[base_key] = load_baseline(model, base_key)
            baseline = baseline_cache[base_key]
            if baseline is None:
                skipped[model].append({"base": base_key, "reason": "no_usable_v2_baseline"})
                continue

            mutant_path = os.path.join(MUTANTS_DIR, m["mutant_file"])
            entry = {"base": base_key, "operator": m["operator"], "variant": m["variant"],
                      "expected": m["expected"], "mutant_file": m["mutant_file"]}

            # Angle mort : controle structurel mecanique, gratuit, pas d'appel embedding --
            # la relation SPO reste "follows" generique quel que soit le type de gateway, donc
            # le DFG doit etre BYTE-IDENTIQUE au DFG de base. Verifie directement plutot que
            # suppose.
            if m["operator"] in BLIND_SPOT_OPS:
                try:
                    bpmn = parse_bpmn(mutant_path)
                    dfg = resolve_dfg(bpmn)
                except Exception as e:
                    entry["error"] = str(e)
                    results[model].append(entry)
                    continue
                base_edges = dfg_edge_set(baseline["dfg_edges"])
                mut_edges = dfg_edge_set(dfg)
                entry["dfg_identical"] = (base_edges == mut_edges)
                entry["detected"] = not entry["dfg_identical"]  # tout ecart serait une surprise
                results[model].append(entry)
                continue

            try:
                out = run_mutant_alignment(mutant_path, baseline["state_space"],
                                            baseline["reference_graph"])
            except Exception as e:
                entry["error"] = str(e)
                results[model].append(entry)
                continue

            baseline_status = status_map(baseline["alignment"])
            mutant_status = status_map(out["alignment"])

            if m["operator"] in REQUIRED_OPS:
                entry["detected"] = classify_required(baseline_status, mutant_status)
            elif m["operator"] in FORBIDDEN_OPS:
                entry.update(classify_forbidden(baseline_status, mutant_status))
            results[model].append(entry)

        print(f"  {len(results[model])} mutants evalues, {len(skipped[model])} references "
              f"v2 absentes/inexploitables (bases sautees) :")
        for s in skipped[model]:
            print(f"    - {s['base']} ({s['reason']})")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "skipped": skipped}, f, indent=2)
    print(f"\n[perturbation] Sauvegarde : {out_path}")
    return results


def summarize(results: dict) -> None:
    print("\n" + "=" * 70)
    print("SYNTHESE -- rappel (required), faux positifs (forbidden), angle mort (blind_spot)")
    print("=" * 70)
    for model, entries in results.items():
        print(f"\n[{model}]")
        by_op: dict[str, list[dict]] = {}
        for e in entries:
            by_op.setdefault(e["operator"], []).append(e)

        for op, group in sorted(by_op.items()):
            errored = [e for e in group if "error" in e]
            valid = [e for e in group if "error" not in e]
            n = len(valid)
            if op in REQUIRED_OPS:
                detected = sum(1 for e in valid if e.get("detected"))
                rate = f"{100*detected/n:.1f}%" if n else "n/a"
                print(f"  {op:20s} REQUIRED   rappel = {detected}/{n} ({rate})"
                      f"{f'  [{len(errored)} erreur(s)]' if errored else ''}")
            elif op in FORBIDDEN_OPS:
                fp = sum(1 for e in valid if e.get("false_positive_violated"))
                reg = sum(1 for e in valid if e.get("any_regression"))
                fp_rate = f"{100*fp/n:.1f}%" if n else "n/a"
                print(f"  {op:20s} FORBIDDEN  faux-positifs(VIOLATED) = {fp}/{n} ({fp_rate})"
                      f"  |  toute regression (incl. UNRESOLVABLE) = {reg}/{n}"
                      f"{f'  [{len(errored)} erreur(s)]' if errored else ''}")
            elif op in BLIND_SPOT_OPS:
                changed = sum(1 for e in valid if not e.get("dfg_identical", True))
                print(f"  {op:20s} BLIND SPOT DFG modifie de facon inattendue = {changed}/{n} "
                      f"(attendu : 0/{n})"
                      f"{f'  [{len(errored)} erreur(s)]' if errored else ''}")
        if errored := [e for e in entries if "error" in e]:
            print(f"    -- echantillon d'erreurs --")
            for e in errored[:3]:
                print(f"    {e['base']}/{e['operator']}: {e['error'][:150]}")


if __name__ == "__main__":
    results = run_study()
    summarize(results)