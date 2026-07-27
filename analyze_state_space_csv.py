"""Verifications de base sur un CSV produit par state_space_to_csv.py.

Usage :
    python analyze_state_space_csv.py                                # prend le CSV le plus recent
    python analyze_state_space_csv.py result_csv/state_space/state_space_run_3.csv

Produit, dans result_csv/state_space/analysis/<nom_du_csv_source>/ :
    - funnel.csv             : brut -> garde (lexname) -> final (dedup), par llm x use_case
    - actor_rejections.csv   : tout ce qui a ete rejete comme ACTOR, par llm x use_case
    - coverage_matrix.csv    : quel final_entity est trouve par quel llm, par use_case
                                (1 = trouve, 0 = absent -- pivot lisible directement)
    - richness.csv           : nb d'entites finales + nb total d'etats finaux, par llm x use_case
                                (repond a la question "richesse de U", cf. pipeline_v2 section 7)

Chaque fichier est ecrase a chaque execution (ce sont des vues derivees d'un seul CSV source
donne en entree, pas des runs a accumuler -- contrairement a state_space_to_csv.py)."""

import glob
import os
import sys

import pandas as pd

CSV_DIR = os.path.join("result_csv", "state_space")


def _latest_csv(directory: str) -> str:
    candidates = glob.glob(os.path.join(directory, "state_space_run_*.csv"))
    if not candidates:
        raise SystemExit(f"Aucun state_space_run_*.csv trouve dans {directory}")
    return max(candidates, key=os.path.getmtime)


def funnel(df: pd.DataFrame) -> pd.DataFrame:
    """brut -> garde (filtre lexical) -> final (post-dedup), par llm x use_case. Le nombre
    d'entites finales peut etre < garde (fusion) mais jamais > garde (la dedup ne cree rien)."""
    grouped = df.groupby(["llm", "use_case"])
    return grouped.apply(
        lambda g: pd.Series({
            "raw_count": len(g),
            "kept_count": (g["kept_lexical"] == True).sum(),  # noqa: E712
            "final_unique_count": g.loc[g["kept_lexical"] == True, "final_entity"].nunique(),
        }),
        include_groups=False,
    ).reset_index()


def actor_rejections(df: pd.DataFrame) -> pd.DataFrame:
    """Tout ce qui a ete juge ACTOR (donc exclu de U) -- utile pour verifier que le piege
    acteur=entite (state_space_observation.md) est traite de facon coherente entre modeles."""
    cols = ["llm", "use_case", "raw_entity", "raw_states", "lexname"]
    return df.loc[df["lexname"] == "ACTOR", cols].sort_values(["use_case", "raw_entity", "llm"])


def coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque use_case, quel final_entity est trouve par quel llm -- 1/0, pivot direct.
    Revele les trous de couverture entre modeles sans relire le JSON entite par entite."""
    kept = df[df["kept_lexical"] == True]  # noqa: E712
    pivot = (
        kept.assign(found=1)
        .pivot_table(index=["use_case", "final_entity"], columns="llm", values="found",
                     aggfunc="max", fill_value=0)
        .reset_index()
    )
    return pivot


def richness(df: pd.DataFrame) -> pd.DataFrame:
    """Nb d'entites finales + nb total d'etats finaux (somme des etats par entite finale
    distincte), par llm x use_case -- la metrique 'richesse de U' discutee en V2 (soupcon de
    sur-fusion sur gpt-oss-20b, 16.2->11.4 etats en moyenne)."""
    kept = df[df["kept_lexical"] == True]  # noqa: E712
    # Une ligne par (llm, use_case, final_entity) avant de compter les etats, pour ne pas
    # compter les etats d'une entite fusionnee plusieurs fois (elle apparait sur plusieurs
    # lignes raw_entity si plusieurs candidats bruts y ont ete fusionnes).
    unique_entities = kept.drop_duplicates(["llm", "use_case", "final_entity"])
    unique_entities = unique_entities.assign(
        n_states=unique_entities["final_states"].apply(
            lambda s: len(s.split(";")) if s else 0
        )
    )
    return (
        unique_entities.groupby(["llm", "use_case"])
        .agg(final_entity_count=("final_entity", "nunique"), total_final_states=("n_states", "sum"))
        .reset_index()
    )


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else _latest_csv(CSV_DIR)
    # dtype=str obligatoire ici : sans lignes d'erreur, kept_lexical ne contient jamais de
    # cellule vide -- pandas infere alors un bool NATIF pour la colonne entiere ('True'/'False'
    # deviennent des Python bool directement), ce qui casse silencieusement le mapping
    # string ci-dessous (toutes les valeurs deviennent NaN, 0 ligne dans tous les rapports
    # filtres sur kept_lexical). Force le parsing en chaine, quel que soit le contenu.
    df = pd.read_csv(input_path, keep_default_na=False, dtype=str)
    # kept_lexical arrive comme chaine "True"/"False"/"" depuis le CSV -- reconverti en bool
    # (None si vide, cas d'erreur de generation).
    df["kept_lexical"] = df["kept_lexical"].map({"True": True, "False": False, "": None})

    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(CSV_DIR, "analysis", stem)
    os.makedirs(out_dir, exist_ok=True)

    reports = {
        "funnel.csv": funnel(df),
        "actor_rejections.csv": actor_rejections(df),
        "coverage_matrix.csv": coverage_matrix(df),
        "richness.csv": richness(df),
    }
    for filename, report_df in reports.items():
        out_path = os.path.join(out_dir, filename)
        report_df.to_csv(out_path, index=False)
        print(f"{filename:22} {len(report_df):4} ligne(s) -> {out_path}")

    print(f"\nSource : {input_path}")