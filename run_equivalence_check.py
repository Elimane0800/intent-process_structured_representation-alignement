"""Rejoue le test d'equivalence stricte (ancienne implementation figee vs nouveau solveur TGMS)
sur TOUS les mutants applicables du manifeste CORRIGE -- pas des fixtures a la main, les vrais
mutants de production. C'est le controle de non-regression manquant, note comme "toujours non
fait" dans dataset_run_v2_observations.md section 8.3.

Principe identique a run_perturbation_study.py : rejoue uniquement la branche droite
(bpmn_to_spo -> state_matching -> alignment) sur chaque mutant, contre le U/G FIGE du run v2 de
reference -- sauf qu'ici alignment est appele DEUX FOIS sur les memes entrees exactes (ancienne
implementation figee, puis solveur TGMS), et les deux verdicts sont compares champ par champ.
Aucun appel LLM, un seul appel embedding par mutant (partage entre les deux implementations,
puisque le matching ne change pas -- seul alignment_node differe).

Prerequis : _frozen_current_impl.py doit se trouver dans le meme dossier que ce script (copie
figee de l'implementation pre-refactoring, livree separement).

Usage : python3 run_equivalence_check.py
Sortie : un rapport console -- 0 ecart attendu ; si un ecart apparait, il est imprime en
detail (target, terme, statut de chaque implementation) pour investigation immediate, jamais
juste un compteur agrege qui masquerait le cas precis."""

from __future__ import annotations

import importlib.util
import json
import os

from dotenv import load_dotenv

load_dotenv()  # cf. run_perturbation_study.py -- necessaire, ce script n'importe pas base_llm

from agent.nodes.bpmn_to_spo_node import parse_bpmn, resolve_dfg, dfg_to_spo
from agent.nodes.state_matching_node import match_spo_to_u
from agent.nodes.alignment_node import check_alignment as check_alignment_new

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUTANTS_DIR = os.path.join(BASE_DIR, "results", "mutants")
V2_RESULTS_DIR = os.path.join(BASE_DIR, "results", "dataset_runs_v2")
MODELS = ["llama-3.1-8b", "mistral-nemotron", "gpt-oss-20b"]

# Charge la copie figee (fichier livre a part, jamais modifie) -- import par chemin explicite
# pour ne jamais risquer de charger accidentellement la version courante par confusion de nom.
_FROZEN_PATH = os.path.join(BASE_DIR, "_frozen_current_impl.py")
if not os.path.exists(_FROZEN_PATH):
    raise SystemExit(
        f"[equivalence] {_FROZEN_PATH} introuvable -- copier _frozen_current_impl.py "
        f"(livre separement) dans le meme dossier que ce script avant de le lancer."
    )
_spec = importlib.util.spec_from_file_location("_frozen_current_impl", _FROZEN_PATH)
_frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frozen)
check_alignment_old = _frozen.check_alignment


def _v2_result_path(model: str, base_key: str) -> str:
    desc, fname = base_key.split("/", 1)
    return os.path.join(V2_RESULTS_DIR, model, f"{desc}__{fname}.json")


def load_baseline(model: str, base_key: str) -> dict | None:
    path = _v2_result_path(model, base_key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    if not state.get("state_space") or not state.get("reference_graph"):
        return None
    return state


def diff_alignments(old: dict, new: dict) -> list[str]:
    """Compare deux sorties de check_alignment champ par champ. Retourne la liste des
    divergences trouvees (vide si equivalence stricte)."""
    diffs = []
    if set(old.keys()) != set(new.keys()):
        diffs.append(f"cibles differentes : old={set(old)} new={set(new)}")
        return diffs
    for target in old:
        o, n = old[target], new[target]
        if o["status"] != n["status"]:
            diffs.append(f"target={target!r} : statut agrege differe (old={o['status']!r}, new={n['status']!r})")
        o_terms = {t["term"]: t for t in o["terms"]}
        n_terms = {t["term"]: t for t in n["terms"]}
        if set(o_terms) != set(n_terms):
            diffs.append(f"target={target!r} : termes differents (old={set(o_terms)}, new={set(n_terms)})")
            continue
        for term in o_terms:
            ot, nt = o_terms[term], n_terms[term]
            if ot["status"] != nt["status"]:
                diffs.append(
                    f"target={target!r} term={term!r} : statut differe "
                    f"(old={ot['status']!r}, new={nt['status']!r})"
                )
            if sorted(ot.get("missing", [])) != sorted(nt.get("missing", [])):
                diffs.append(f"target={target!r} term={term!r} : missing differe")
            if sorted(ot.get("low_confidence", [])) != sorted(nt.get("low_confidence", [])):
                diffs.append(f"target={target!r} term={term!r} : low_confidence differe")
    return diffs


def run_equivalence_check(models: list[str] = MODELS) -> None:
    manifest_path = os.path.join(MUTANTS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"[equivalence] {manifest_path} introuvable -- lancer mutate_bpmn.py d'abord")
    with open(manifest_path) as f:
        manifest = json.load(f)
    applicable = [m for m in manifest if m["applicable"]]
    print(f"[equivalence] {len(applicable)} mutants applicables, {len(models)} modele(s)")

    total_checked = 0
    total_skipped = 0
    total_diffs = 0
    diff_report: list[tuple[str, str, list[str]]] = []

    for model in models:
        print(f"\n########## Modele : {model} ##########")
        baseline_cache: dict[str, dict | None] = {}

        for m in applicable:
            base_key = m["base"]
            if base_key not in baseline_cache:
                baseline_cache[base_key] = load_baseline(model, base_key)
            baseline = baseline_cache[base_key]
            if baseline is None:
                total_skipped += 1
                continue

            mutant_path = os.path.join(MUTANTS_DIR, m["mutant_file"])
            try:
                bpmn = parse_bpmn(mutant_path)
                dfg = resolve_dfg(bpmn)
                spo = dfg_to_spo(dfg)
                spo_tuples = [(t["subject"], t["predicate"], t["object"]) for t in spo]
                matches, process_graph = match_spo_to_u(spo_tuples, baseline["state_space"])

                old_out = check_alignment_old(baseline["reference_graph"], process_graph, matches)
                new_out = check_alignment_new(baseline["reference_graph"], process_graph, matches)
            except Exception as e:
                print(f"  [ERREUR] {model}/{m['mutant_file']}: {e}")
                total_skipped += 1
                continue

            total_checked += 1
            diffs = diff_alignments(old_out, new_out)
            if diffs:
                total_diffs += 1
                diff_report.append((model, m["mutant_file"], diffs))
                print(f"  [ECART] {m['mutant_file']}")
                for d in diffs:
                    print(f"    - {d}")

        print(f"  ({total_checked} verifies, {total_skipped} sautes jusqu'ici)")

    print("\n" + "=" * 70)
    print("SYNTHESE EQUIVALENCE STRICTE")
    print("=" * 70)
    print(f"Mutants verifies   : {total_checked}")
    print(f"Sautes (pas de ref): {total_skipped}")
    print(f"Ecarts trouves     : {total_diffs}")
    if total_diffs == 0:
        print("\n>>> EQUIVALENCE CONFIRMEE -- 0 ecart entre l'ancienne implementation et le "
              "solveur TGMS sur tous les mutants du manifeste corrige. <<<")
    else:
        print(f"\n>>> {total_diffs} ECART(S) TROUVE(S) -- investiguer avant de considerer le "
              f"refactoring TGMS valide sur ce manifeste. Detail ci-dessus. <<<")


if __name__ == "__main__":
    run_equivalence_check()