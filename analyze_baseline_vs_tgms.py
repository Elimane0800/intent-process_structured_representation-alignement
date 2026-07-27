"""Analyse baseline_llm_judge_results.jsonl et compare directement aux resultats reels TGMS
deja mesures (run_perturbation_study.py, 153 mutants). Lecture seule, aucun appel LLM.

Point critique : les entrees 'inconclusive' (echec de parsing OU echec d'appel API, ex. 429
non retente) sont EXCLUES du denominateur de chaque taux -- jamais comptees comme "non
detecte", sinon un juge purement rate-limite afficherait 0% de rappel a tort. Le nombre
d'inconclusives est toujours affiche a cote du taux pour juger de sa fiabilite (un taux
calcule sur 2 cas sur 12 n'a pas le meme poids qu'un taux calcule sur 12/12)."""

import json
import os
import sys
from collections import defaultdict

REQUIRED_OPS = {"remove_activity", "swap_labels", "cross_case_replace"}
FORBIDDEN_OPS = {"insert_activity", "shuffle_xml"}
BLIND_SPOT_OPS = {"rewire_gateway"}

# Reference TGMS reelle (run_perturbation_study.py, 3 modeles cumules, 153 mutants) -- cf.
# synthese deja publiee. Comparaison volontairement au niveau agrege : TGMS n'a pas de notion
# de "juge", ces chiffres viennent du role "modele sous test" (extraction), pas "modele juge"
# -- la comparaison valide porte sur memes mutants/meme taxonomie/meme definition de detection,
# pas sur "meme role du modele".
TGMS_REFERENCE = {
    "remove_activity": (8, 51),    # 2/12 + 4/18 + 2/21
    "swap_labels": (2, 51),        # 0/12 + 2/18 + 0/21
    "cross_case_replace": (2, 51), # 1/12 + 1/18 + 0/21
    "forbidden_false_positive": (0, 102),
    "blind_spot_detected": (0, 51),
}


def load_entries(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def rate(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "n/a (0 cas exploitable)"


def analyze(path: str) -> None:
    entries = load_entries(path)
    print(f"[{len(entries)} entrees chargees depuis {path}]\n")

    # Regroupe par (judge_model, prompt_variant) -- chaque config rapportee separement, jamais
    # moyennee (cf. discussion sur les 3 variantes de prompt).
    by_config: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        by_config[(e["judge_model"], e["prompt_variant"])].append(e)

    for (judge, variant), group in sorted(by_config.items()):
        print(f"{'='*70}\n{judge}  |  {variant}\n{'='*70}")

        n_inconclusive = sum(1 for e in group if e["inconclusive"])
        n_error = sum(1 for e in group if e.get("error"))
        print(f"  {len(group)} entrees -- {n_inconclusive} inconclusive(s) dont {n_error} "
              f"erreur(s) d'appel API (exclues des taux ci-dessous)")
        if n_inconclusive:
            print(f"  ATTENTION : taux calcules sur un sous-ensemble reduit -- fiabilite a "
                  f"juger au denominateur affiche, pas seulement au pourcentage")

        by_op: dict[str, list[dict]] = defaultdict(list)
        for e in group:
            if not e["inconclusive"]:
                by_op[e["operator"]].append(e)

        for op in sorted(REQUIRED_OPS):
            g = by_op.get(op, [])
            n_flip = sum(1 for e in g if e["flipped_to_violation"])
            tgms_n, tgms_d = TGMS_REFERENCE[op]
            print(f"  {op:20s} REQUIRED   baseline rappel = {rate(n_flip, len(g)):25s} "
                  f"| TGMS (reference) = {rate(tgms_n, tgms_d)}")

        forbidden_g = [e for op in FORBIDDEN_OPS for e in by_op.get(op, [])]
        n_fp = sum(1 for e in forbidden_g if e["flipped_to_violation"])
        tgms_n, tgms_d = TGMS_REFERENCE["forbidden_false_positive"]
        print(f"  {'forbidden (cumul)':20s} FORBIDDEN  baseline faux-positifs = "
              f"{rate(n_fp, len(forbidden_g)):25s} | TGMS (reference) = {rate(tgms_n, tgms_d)}")

        blind_g = by_op.get("rewire_gateway", [])
        n_detected = sum(1 for e in blind_g if e["flipped_to_violation"])
        tgms_n, tgms_d = TGMS_REFERENCE["blind_spot_detected"]
        print(f"  {'rewire_gateway':20s} BLIND SPOT baseline detection    = "
              f"{rate(n_detected, len(blind_g)):25s} | TGMS (reference) = {rate(tgms_n, tgms_d)}")
        if n_detected:
            print(f"    -> le juge a detecte {n_detected} cas que TGMS ne peut structurellement "
                  f"pas voir (angle mort assume XOR<->AND) -- a assumer honnetement, pas a cacher")
        print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "baseline_llm_judge_results.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"introuvable : {path}")
    analyze(path)