"""Audit structurel PUR du corpus LRE -- aucun appel LLM, aucun texte lu, uniquement
compute_activity_guards() sur chaque modele .bpmn du corpus. Repond a une question precise
avant de dimensionner LRE_MAX_CASES dans run.py : a quelle frequence l'anomalie
"fork non declare" (proposition 3, cf. bpmn_guards_node.py::find_undeclared_splits) apparait-
elle reellement dans ce corpus, en tout ?

Justification du script separe plutot que d'augmenter LRE_MAX_CASES a l'aveugle : cette
anomalie ne depend QUE de la structure BPMN, jamais du texte ni du LLM -- la mesurer via le
pipeline complet (run.py) gaspillerait le budget d'appels LLM pour une question qui se repond
en quelques secondes, sur tout le corpus, sans lui.

N'importe PAS run.py (qui charge `from agent.graph import graph`, toute la chaine LLM) --
duplique volontairement LRE_ROOT/LRE_SUBSETS ici, avec le meme raisonnement de decouplage deja
applique ailleurs dans ce projet (cf. tgms_solver.py : dupliquer un helper leger plutot que
tirer une dependance lourde pour un seul usage)."""

import csv
import os

from agent.nodes.bpmn_guards_node import compute_activity_guards
from agent.nodes.bpmn_to_spo_node import parse_bpmn

# Duplique de run.py -- volontairement, cf. docstring de tete. AJUSTER EN MEME TEMPS que
# run.py::LRE_ROOT si le layout du depot change (deux constantes independantes, jamais
# synchronisees automatiquement).
LRE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "lre",
    "alignment_model_text-master", "datasets",
)
LRE_SUBSETS = ["NewDataset", "OriginalDataset"]  # les DEUX ici -- contrairement a run.py
                                                    # (restreint a OriginalDataset pour le run
                                                    # LLM), cet audit ne coute rien de plus a
                                                    # couvrir tout le corpus disponible

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_analysis")


def discover_bpmn_files(lre_root: str, subsets: list[str]) -> list[tuple[str, str, str]]:
    """Retourne [(subset, case_name, filepath), ...] -- TOUS les .bpmn de Models/, sans exiger
    de texte correspondant (contrairement a run.py::discover_lre_cases) : cet audit porte sur
    la structure du modele seule, jamais sur la paire texte-modele."""
    files = []
    for subset in subsets:
        models_dir = os.path.join(lre_root, subset, "Models")
        if not os.path.isdir(models_dir):
            print(f"  [audit] ATTENTION : {models_dir} n'existe pas -- sous-ensemble ignore")
            continue
        for fname in sorted(os.listdir(models_dir)):
            if fname.endswith(".bpmn"):
                case_name = fname[: -len(".bpmn")]
                files.append((subset, case_name, os.path.join(models_dir, fname)))
    return files


def audit_corpus(lre_root: str, subsets: list[str]) -> tuple[list[dict], list[dict]]:
    """Retourne (summary_rows, detail_rows). summary_rows : une ligne par sous-ensemble +
    une ligne TOTAL. detail_rows : une ligne PAR FORK non declare trouve (pour inspection
    manuelle ciblee, cf. la meme discipline que l'inspection a la main deja faite sur
    "Rate the company" -- jamais une conclusion tiree du seul chiffre agrege sans regarder
    au moins quelques cas concrets)."""
    files = discover_bpmn_files(lre_root, subsets)
    print(f"[audit] {len(files)} fichier(s) .bpmn decouvert(s) sur {subsets}")

    per_subset = {}
    detail_rows = []

    for subset, case_name, filepath in files:
        bucket = per_subset.setdefault(subset, {
            "total": 0, "parse_error": 0, "no_split": 0, "with_split": 0,
            "total_splits_found": 0,
        })
        bucket["total"] += 1

        try:
            bpmn = parse_bpmn(filepath)
        except Exception as e:
            bucket["parse_error"] += 1
            print(f"  [audit] {subset}/{case_name} : echec de parsing BPMN ({e}) -- ignore, "
                  f"jamais compte comme 'sans fork' ni 'avec fork'")
            continue

        try:
            result = compute_activity_guards(bpmn)
        except Exception as e:
            bucket["parse_error"] += 1
            print(f"  [audit] {subset}/{case_name} : echec de compute_activity_guards ({e}) -- ignore")
            continue

        splits = result.get("undeclared_splits", [])
        if splits:
            bucket["with_split"] += 1
            bucket["total_splits_found"] += len(splits)
            for activity_name in splits:
                detail_rows.append({
                    "subset": subset, "case_name": case_name,
                    "activity_forking": activity_name,
                })
        else:
            bucket["no_split"] += 1

    summary_rows = []
    grand = {"total": 0, "parse_error": 0, "no_split": 0, "with_split": 0, "total_splits_found": 0}
    for subset in sorted(per_subset):
        b = per_subset[subset]
        for k in grand:
            grand[k] += b[k]
        analyzable = b["total"] - b["parse_error"]
        summary_rows.append({
            "subset": subset, "total_files": b["total"], "parse_error": b["parse_error"],
            "analyzable": analyzable,
            "with_undeclared_split": b["with_split"],
            "share_%": round(100 * b["with_split"] / analyzable, 1) if analyzable else None,
            "total_splits_found": b["total_splits_found"],
        })
    analyzable_total = grand["total"] - grand["parse_error"]
    summary_rows.append({
        "subset": "TOTAL", "total_files": grand["total"], "parse_error": grand["parse_error"],
        "analyzable": analyzable_total,
        "with_undeclared_split": grand["with_split"],
        "share_%": round(100 * grand["with_split"] / analyzable_total, 1) if analyzable_total else None,
        "total_splits_found": grand["total_splits_found"],
    })

    return summary_rows, detail_rows


def print_table(title: str, rows: list[dict]):
    print(f"\n=== {title} ===")
    if not rows:
        print("  (aucune donnee)")
        return
    headers = list(rows[0].keys())
    widths = [max(len(str(h)), max((len(str(r.get(h, ""))) for r in rows), default=0)) for h in headers]
    print("  " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    summary, details = audit_corpus(LRE_ROOT, LRE_SUBSETS)

    print_table("Frequence des forks non declares (proposition 3), par sous-ensemble", summary)
    print_table("Detail : chaque activite fork non declare trouvee (subset/cas/activite)", details)

    write_csv(os.path.join(OUT_DIR, "audit_undeclared_splits_summary.csv"), summary)
    write_csv(os.path.join(OUT_DIR, "audit_undeclared_splits_detail.csv"), details)
    print(f"\nCSV ecrits dans {OUT_DIR}")