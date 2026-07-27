"""Extrait un echantillon lisible des cas 'lost_to_margin' deja produits par
node_type_audit.py (results/node_type_audit/missing_rows_raw.csv) -- pour inspection manuelle
directe, texte-en-main : le concurrent gagnant sur la meme activite est-il reellement le bon
choix (la marge de 0.05 a bien fait de l'exclure), ou une vraie perte (un candidat legitime
ecarte a tort) ?

Stratifie par (model, case) -- au plus MAX_PER_CASE lignes par cas, pour eviter qu'un seul cas
avec beaucoup de milestones domine tout l'echantillon. Trie par score decroissant a l'interieur
de chaque cas -- les plus "proches" d'un ancrage correct en premier, les plus interessants a
juger en priorite.

Aucun calcul, aucun appel LLM -- pure lecture/mise en forme d'un CSV deja produit."""

import csv
import os

IN_PATH = os.path.join(os.path.dirname(__file__), "results", "node_type_audit", "missing_rows_raw.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results", "node_type_audit", "sample_lost_to_margin.txt")

MAX_PER_CASE = 3   # au plus 3 milestones par (model, case) dans l'echantillon
MAX_TOTAL = 30      # plafond global, au cas ou beaucoup de cas distincts existent


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"[sample] {path} introuvable -- lance d'abord node_type_audit.py "
            f"(il produit ce fichier)."
        )
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_sample(rows: list[dict]) -> list[dict]:
    lost = [r for r in rows if r.get("cause") == "lost_to_margin"]
    by_case: dict[tuple[str, str], list[dict]] = {}
    for r in lost:
        key = (r["model"], r["case"])
        by_case.setdefault(key, []).append(r)

    sample = []
    for key in sorted(by_case):
        case_rows = sorted(by_case[key], key=lambda r: float(r["best_score"]), reverse=True)
        sample.extend(case_rows[:MAX_PER_CASE])
        if len(sample) >= MAX_TOTAL:
            break
    return sample[:MAX_TOTAL]


def format_sample(sample: list[dict]) -> str:
    lines = []
    for r in sample:
        winners = r.get("winning_candidates_on_same_activity", "") or "(aucun -- verifier le CSV source)"
        lines.append(
            f"[{r['model']} / {r['case']}]\n"
            f"  Milestone (missing)          : {r['milestone']}\n"
            f"  Meilleure activite BPMN      : \"{r['best_activity_label']}\" (score={r['best_score']})\n"
            f"  Score propre de l'activite   : {r['best_activity_own_best_score']} "
            f"(marge autorisee : jusqu'a {float(r['best_activity_own_best_score']) - 0.05:.4f})\n"
            f"  Concurrent(s) gagnant(s)     : {winners}\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = load_rows(IN_PATH)
    sample = build_sample(rows)
    if not sample:
        raise SystemExit("[sample] aucun cas 'lost_to_margin' trouve dans le CSV -- verifie qu'il "
                          "contient bien la colonne 'cause' (version a jour de node_type_audit.py)")

    text = format_sample(sample)
    print(f"Echantillon de {len(sample)} cas lost_to_margin "
          f"(sur {sum(1 for r in rows if r.get('cause') == 'lost_to_margin')} au total) :\n")
    print(text)

    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"\nEcrit aussi dans {OUT_PATH}")