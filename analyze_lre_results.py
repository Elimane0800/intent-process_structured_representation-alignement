"""Analyse les resultats du run LRE (results/lre_runs/) -- ROBUSTE a une couverture partielle :
tous les modeles n'ont pas necessairement tourne sur tous les cas, un run peut avoir ete
interrompu et repris (cf. run.py::RUN_ID, deliberement manuel). Un modele/cas absent est
simplement absent des tableaux, jamais suppose "echoue" ni "a zero" -- meme discipline que le
reste du pipeline (une absence est un statut legitime, jamais une exception silencieuse).

Produit cinq tableaux (+Table H additionnelle), imprimes et exportes en CSV, directement
motives par la discussion sur l'exemple motivant (risque de faux positif du critere permissif
-- reachability plutot que dominance) et par la verification empirique que le mecanisme NOT
s'engage reellement sur des donnees LRE, pas seulement en synthetique :

  A. Couverture/completude par modele (attempted / fatal / incomplete / ok)
  B. Distribution des verdicts (%) par modele -- FAITHFUL / FAITHFUL_WITH_RESERVATIONS /
     NOT_FAITHFUL. LRE etant assume entierement fidele, NOT_FAITHFUL y est directement un
     proxy de faux positif du critere de verification actuel.
  C. Moyennes ecarts/non_verifiable/conforme par cas, par modele -- plus fin que le seul
     verdict (un NOT_FAITHFUL a 1 ecart isole n'est pas le meme signal qu'a 5).
  D. Repartition positif/negatif des termes en echec, par modele -- extrait de
     piece_jointe_alignement_brut (prefixe "NOT " toujours preserve la, jamais du texte
     libre parse) -- mesure si le mecanisme NOT s'engage reellement sur ce dataset.
  E. Couverture de l'analyse structurelle (propositions 1-3) -- activity_guards present,
     forks non declares detectes.
  F. Termes retrogrades par l'analyse structurelle (proposition 2) -- effet reel mesure du
     mecanisme de demotion, via un marqueur litteral genere par notre propre code (jamais du
     texte libre parse).
  G. Comparaison de la distribution des verdicts entre deux run_id (ex. run_1 sans guards vs
     run_2 avec guards), sur les memes cas uniquement.
  H. Diagnostic des cas a verdict change entre les deux run_id de Table G : identifie le
     premier etage du pipeline (state_space / preconditions / graph / matches) deja different
     entre les deux etats sauvegardes, AVANT de regarder le verdict -- pour trancher si un
     changement de verdict est reellement attribuable a activity_guards (Table F/le marqueur
     structurel s'est declenche) ou si un etage en amont a produit un resultat different entre
     les deux runs (non-determinisme LLM/embedding, ou changement de code non isole a la seule
     branche guards) -- auquel cas l'attribution a activity_guards est confondue, pas fiable.
"""

import csv
import json
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_analysis")

# Marqueur litteral et deterministe (genere par NOTRE code dans tgms_solver.py::status_of_
# negated_term, jamais par un LLM -- donc fiable, contrairement a une detection sur du texte
# libre) qui identifie une retrogradation SATISFIED->UNRESOLVABLE due specifiquement a
# l'analyse structurelle (propositions 1-3), et non a un manque de confiance de matching ou a
# une absence d'ancrage. Extrait apres coup depuis un run deja termine -- pas de champ
# structure dedie disponible retroactivement dans des resultats deja generes, ce marqueur est
# le point d'ancrage le plus fiable qui existe reellement dans ces donnees.
STRUCTURAL_DEMOTION_MARKER = "independent structural guard analysis"

# Fixe explicitement le run analyse -- ne jamais melanger silencieusement plusieurs run_id d'un
# meme modele (cf. run.py : RUN_ID est manuel precisement pour eviter ce risque). None = tous
# les run_id trouves sont inclus, avec un avertissement si plusieurs coexistent pour un modele.
TARGET_RUN_ID = "run_2"  # le run avec les propositions 1-3 (activity_guards) -- celui qu'on
                           # veut evaluer maintenant. run_1 reste consultable via Table G/H.

# Ordre des etages du pipeline, du plus en amont au plus en aval -- utilise par Table H. Le
# premier champ de PipelineState qui differe entre deux runs, pour un meme (subset, case_name),
# est rapporte comme la cause la plus probable d'un verdict different (tout ce qui suit en
# herite mecaniquement). "reference_graph" est inclus meme s'il devrait etre une fonction pure
# de state_space + validated_preconditions + branch_conflicts -- une divergence a ce niveau
# alors que les deux etages precedents sont identiques signalerait un bug de non-determinisme
# dans graph_construction lui-meme, pas seulement en amont.
_PIPELINE_STAGES_ORDER = [
    "state_space",              # U (Protocole 3)
    "validated_preconditions",  # Pre(u), avant assemblage en graphe
    "reference_graph",          # G
    "matches",                  # ancrage embedding (branche droite, independante de U/Pre)
    "process_graph",            # G_process, fonction pure de dfg_edges + matches
]


def parse_filename(filename: str):
    """'{subset}__{case_name}__{run_id}.json' -- ancre sur le PREMIER '__' pour le subset et le
    DERNIER '__' pour le run_id, jamais un split naif : case_name peut lui-meme contenir des
    '_' simples (ex. '1066443049_rev6', NewDataset), mais jamais de '__' double (uniquement
    utilise comme separateur par run.py::result_path)."""
    if not filename.endswith(".json"):
        return None
    stem = filename[:-5]
    if "__" not in stem:
        return None
    subset, rest = stem.split("__", 1)
    if "__" not in rest:
        return None
    case_name, run_id = rest.rsplit("__", 1)
    return subset, case_name, run_id


def discover_results(results_dir: str) -> dict:
    """{model: [(subset, case_name, run_id, filepath), ...]} -- ne suppose jamais qu'un modele
    a tourne sur tous les cas."""
    results = defaultdict(list)
    if not os.path.isdir(results_dir):
        print(f"[analyse] ATTENTION : {results_dir} n'existe pas encore.")
        return results
    for model in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for fname in sorted(os.listdir(model_dir)):
            parsed = parse_filename(fname)
            if parsed is None:
                if fname.endswith(".json"):
                    print(f"  [analyse] nom de fichier non reconnu, ignore : {model}/{fname}")
                continue
            subset, case_name, run_id = parsed
            results[model].append((subset, case_name, run_id, os.path.join(model_dir, fname)))
    return results


def load_case(filepath: str):
    """Statuts : 'fatal' (echec definitif, aucun etat exploitable -- cf. run.py::run_one),
    'incomplete' (pipeline execute mais jamais atteint report_node, ex. erreur de node en
    amont), 'ok' (report present, exploitable)."""
    with open(filepath) as f:
        state = json.load(f)
    if "fatal_error" in state:
        return "fatal", state
    if "report" not in state:
        return "incomplete", state
    return "ok", state


def build_tables(results: dict, target_run_id: str | None = None):
    coverage_rows, verdict_rows, summary_rows, polarity_rows = [], [], [], []
    structural_rows = []  # Table E : couverture activity_guards + forks detectes
    demotion_rows = []    # Table F : effet reel de la retrogradation structurelle

    agg = {"n_ok": 0, "faithful": 0, "faithful_reserv": 0, "not_faithful": 0}

    for model in sorted(results):
        entries = results[model]
        run_ids_present = sorted({r for _, _, r, _ in entries})
        if target_run_id is not None:
            entries = [e for e in entries if e[2] == target_run_id]
        elif len(run_ids_present) > 1:
            print(f"  [analyse] ATTENTION [{model}] : plusieurs run_id presents "
                  f"{run_ids_present} -- TARGET_RUN_ID non specifie, tous inclus ensemble "
                  f"(risque de melanger des runs differents). Fixe TARGET_RUN_ID pour restreindre.")

        n_attempted = len(entries)
        n_fatal = n_incomplete = n_ok = 0
        n_faithful = n_faithful_reserv = n_not_faithful = 0
        sum_ecarts = sum_non_verif = sum_conforme = 0
        n_neg_failing = n_pos_failing = 0

        # Table E
        n_with_guards = 0
        n_undeclared_splits_total = 0
        n_cases_with_split = 0

        # Table F
        n_demoted = 0

        for subset, case_name, run_id, filepath in entries:
            try:
                status, state = load_case(filepath)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [analyse] {model}/{subset}__{case_name} : fichier illisible ({e}), ignore")
                n_fatal += 1
                continue

            if status == "fatal":
                n_fatal += 1
                continue
            if status == "incomplete":
                n_incomplete += 1
                continue

            n_ok += 1

            # Table E -- activity_guards vit au niveau racine de l'etat, pas dans report
            ag = state.get("activity_guards")
            if ag is not None:
                n_with_guards += 1
                n_splits_here = len(ag.get("undeclared_splits", []))
                n_undeclared_splits_total += n_splits_here
                if n_splits_here:
                    n_cases_with_split += 1

            report = state["report"]
            verdict = report["verdict"]["verdict"]
            if verdict == "FAITHFUL":
                n_faithful += 1
            elif verdict == "FAITHFUL_WITH_RESERVATIONS":
                n_faithful_reserv += 1
            elif verdict == "NOT_FAITHFUL":
                n_not_faithful += 1

            summary = report["summary"]
            sum_ecarts += summary["ecarts"]
            sum_non_verif += summary["non_verifiable"]
            sum_conforme += summary["conforme"]

            alignment = report.get("piece_jointe_alignement_brut", {})
            for target, entry in alignment.items():
                for term_result in entry.get("terms", []):
                    if term_result["status"] in ("VIOLATED", "UNRESOLVABLE"):
                        if term_result["term"].startswith("NOT "):
                            n_neg_failing += 1
                        else:
                            n_pos_failing += 1
                    # Table F -- ne compte que les UNRESOLVABLE portant le marqueur structurel
                    # (jamais les UNRESOLVABLE "ordinaires", ancrage manquant ou confiance
                    # faible -- cf. STRUCTURAL_DEMOTION_MARKER)
                    if (term_result["status"] == "UNRESOLVABLE"
                            and term_result.get("reason")
                            and STRUCTURAL_DEMOTION_MARKER in term_result["reason"]):
                        n_demoted += 1

        coverage_rows.append({
            "model": model, "attempted": n_attempted, "fatal": n_fatal,
            "incomplete": n_incomplete, "ok": n_ok,
            "completion_%": round(100 * n_ok / n_attempted, 1) if n_attempted else 0.0,
        })

        if n_ok:
            agg["n_ok"] += n_ok
            agg["faithful"] += n_faithful
            agg["faithful_reserv"] += n_faithful_reserv
            agg["not_faithful"] += n_not_faithful

            verdict_rows.append({
                "model": model, "n": n_ok,
                "FAITHFUL_%": round(100 * n_faithful / n_ok, 1),
                "FAITHFUL_WITH_RESERVATIONS_%": round(100 * n_faithful_reserv / n_ok, 1),
                "NOT_FAITHFUL_%": round(100 * n_not_faithful / n_ok, 1),
            })
            summary_rows.append({
                "model": model, "n": n_ok,
                "avg_ecarts": round(sum_ecarts / n_ok, 2),
                "avg_non_verifiable": round(sum_non_verif / n_ok, 2),
                "avg_conforme": round(sum_conforme / n_ok, 2),
            })
            total_failing = n_neg_failing + n_pos_failing
            polarity_rows.append({
                "model": model,
                "negated_terms_failing": n_neg_failing,
                "positive_terms_failing": n_pos_failing,
                "negated_share_%": round(100 * n_neg_failing / total_failing, 1) if total_failing else None,
            })
            structural_rows.append({
                "model": model, "n": n_ok,
                "with_activity_guards_%": round(100 * n_with_guards / n_ok, 1),
                "undeclared_splits_total": n_undeclared_splits_total,
                "cases_with_undeclared_split": n_cases_with_split,
            })
            demotion_rows.append({
                "model": model,
                "terms_demoted_by_structural_analysis": n_demoted,
            })

    if agg["n_ok"]:
        verdict_rows.append({
            "model": "ALL MODELS", "n": agg["n_ok"],
            "FAITHFUL_%": round(100 * agg["faithful"] / agg["n_ok"], 1),
            "FAITHFUL_WITH_RESERVATIONS_%": round(100 * agg["faithful_reserv"] / agg["n_ok"], 1),
            "NOT_FAITHFUL_%": round(100 * agg["not_faithful"] / agg["n_ok"], 1),
        })

    return coverage_rows, verdict_rows, summary_rows, polarity_rows, structural_rows, demotion_rows


def compare_runs(results: dict, run_id_a: str, run_id_b: str) -> list[dict]:
    """Table G : compare la distribution des verdicts entre deux run_id, modele par modele --
    seulement sur les cas ou LES DEUX runs ont un resultat exploitable pour le MEME (subset,
    case_name), jamais un cas present dans un seul des deux (comparaison non biaisee par une
    difference de couverture entre les deux runs)."""
    rows = []
    for model in sorted(results):
        by_case_a, by_case_b = {}, {}
        for subset, case_name, run_id, filepath in results[model]:
            if run_id == run_id_a:
                by_case_a[(subset, case_name)] = filepath
            elif run_id == run_id_b:
                by_case_b[(subset, case_name)] = filepath

        common = sorted(set(by_case_a) & set(by_case_b))
        if not common:
            continue

        counts_a = {"FAITHFUL": 0, "FAITHFUL_WITH_RESERVATIONS": 0, "NOT_FAITHFUL": 0}
        counts_b = dict(counts_a)
        n_changed = 0
        n_valid = 0

        for key in common:
            status_a, state_a = load_case(by_case_a[key])
            status_b, state_b = load_case(by_case_b[key])
            if status_a != "ok" or status_b != "ok":
                continue
            n_valid += 1
            v_a = state_a["report"]["verdict"]["verdict"]
            v_b = state_b["report"]["verdict"]["verdict"]
            counts_a[v_a] += 1
            counts_b[v_b] += 1
            if v_a != v_b:
                n_changed += 1

        if n_valid:
            rows.append({
                "model": model, "n_common_ok": n_valid,
                f"NOT_FAITHFUL_%_{run_id_a}": round(100 * counts_a["NOT_FAITHFUL"] / n_valid, 1),
                f"NOT_FAITHFUL_%_{run_id_b}": round(100 * counts_b["NOT_FAITHFUL"] / n_valid, 1),
                "verdict_changed_%": round(100 * n_changed / n_valid, 1),
            })
    return rows


def _has_structural_marker(state: dict) -> bool:
    """True si au moins un terme de ce cas porte le marqueur de demotion structurelle dans son
    'reason' -- recalcule ICI cas par cas, jamais suppose depuis l'agregat global de Table F
    (qui est global-par-modele, pas par cas)."""
    alignment = state.get("report", {}).get("piece_jointe_alignement_brut", {})
    return any(
        STRUCTURAL_DEMOTION_MARKER in (t.get("reason") or "")
        for entry in alignment.values()
        for t in entry.get("terms", [])
    )


def diagnose_changed_cases(results: dict, run_id_a: str, run_id_b: str) -> list[dict]:
    """Table H : pour chaque cas commun aux deux runs dont le VERDICT differe, identifie le
    premier champ de PipelineState (dans l'ordre _PIPELINE_STAGES_ORDER) qui differe deja entre
    les deux etats sauvegardes -- AVANT de regarder le verdict lui-meme. Comparaison par egalite
    structurelle sur les objets Python deserialises (jamais une comparaison de chaine JSON brute
    -- insensible a l'ordre des cles, qui n'a aucune signification ici).

    Objectif precis : trancher si un verdict change entre run_id_a et run_id_b est attribuable a
    activity_guards (auquel cas STRUCTURAL_DEMOTION_MARKER doit apparaitre pour ce cas precis,
    verifie separement via _has_structural_marker) ou si un etage EN AMONT a deja produit un
    resultat different entre les deux runs -- non-determinisme LLM/embedding, ou changement de
    code non isole a la seule branche guards. Dans ce second cas, l'ecart de verdict n'a rien a
    voir avec activity_guards et toute conclusion tiree de Table G seule serait confondue."""
    rows = []
    for model in sorted(results):
        by_case_a, by_case_b = {}, {}
        for subset, case_name, run_id, filepath in results[model]:
            if run_id == run_id_a:
                by_case_a[(subset, case_name)] = filepath
            elif run_id == run_id_b:
                by_case_b[(subset, case_name)] = filepath

        for key in sorted(set(by_case_a) & set(by_case_b)):
            status_a, state_a = load_case(by_case_a[key])
            status_b, state_b = load_case(by_case_b[key])
            if status_a != "ok" or status_b != "ok":
                continue

            v_a = state_a["report"]["verdict"]["verdict"]
            v_b = state_b["report"]["verdict"]["verdict"]
            if v_a == v_b:
                continue  # Table H ne porte que sur les cas ou le verdict a change

            first_diverging_stage = None
            for stage in _PIPELINE_STAGES_ORDER:
                if state_a.get(stage) != state_b.get(stage):
                    first_diverging_stage = stage
                    break

            rows.append({
                "model": model,
                "case": f"{key[0]}/{key[1]}",
                f"verdict_{run_id_a}": v_a,
                f"verdict_{run_id_b}": v_b,
                "first_diverging_stage": first_diverging_stage or "NONE",
                "structural_marker_present": (
                    _has_structural_marker(state_a) or _has_structural_marker(state_b)
                ),
            })
    return rows


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
    results = discover_results(RESULTS_DIR)
    if not results:
        raise SystemExit(f"[analyse] aucun resultat trouve sous {RESULTS_DIR}")

    print(f"Modeles trouves : {sorted(results.keys())}")
    for model, entries in sorted(results.items()):
        run_ids_seen = sorted({r for _, _, r, _ in entries})
        print(f"  {model} : {len(entries)} fichier(s) de resultat -- run_id present(s) : {run_ids_seen}")

    coverage, verdicts, summaries, polarity, structural, demotion = build_tables(
        results, target_run_id=TARGET_RUN_ID
    )

    print_table("Table A -- Couverture / completude par modele", coverage)
    print_table("Table B -- Distribution des verdicts (%) par modele", verdicts)
    print_table("Table C -- Moyennes ecarts/non_verifiable/conforme par cas", summaries)
    print_table("Table D -- Repartition positif/negatif des termes en echec", polarity)
    print_table("Table E -- Couverture de l'analyse structurelle (propositions 1-3)", structural)
    print_table("Table F -- Termes retrogrades par l'analyse structurelle (propositions 2)", demotion)

    write_csv(os.path.join(OUT_DIR, "table_a_coverage.csv"), coverage)
    write_csv(os.path.join(OUT_DIR, "table_b_verdicts.csv"), verdicts)
    write_csv(os.path.join(OUT_DIR, "table_c_summary_avg.csv"), summaries)
    write_csv(os.path.join(OUT_DIR, "table_d_polarity.csv"), polarity)
    write_csv(os.path.join(OUT_DIR, "table_e_structural_coverage.csv"), structural)
    write_csv(os.path.join(OUT_DIR, "table_f_demotions.csv"), demotion)

    # Table G -- seulement si au moins deux run_id distincts existent quelque part dans les
    # resultats (ex. run_1 = avant propositions 1-3, run_2 = apres) -- sinon rien a comparer,
    # jamais une comparaison inventee a un seul point.
    all_run_ids = sorted({r for entries in results.values() for _, _, r, _ in entries})
    if len(all_run_ids) >= 2:
        run_a, run_b = all_run_ids[0], all_run_ids[1]
        comparison = compare_runs(results, run_a, run_b)
        print_table(f"Table G -- Comparaison {run_a} vs {run_b} (memes cas uniquement)", comparison)
        write_csv(os.path.join(OUT_DIR, "table_g_run_comparison.csv"), comparison)

        # Table H -- diagnostic : POURQUOI chaque cas a verdict change a change. Repond
        # directement a la question laissee ouverte par Table F/G : le mecanisme de demotion
        # structurelle (Table F) peut afficher 0 au global tout en coexistant avec des verdicts
        # changes (Table G) -- Table H determine, cas par cas, si ces changements viennent d'un
        # etage en amont (confondu, rien a voir avec activity_guards) ou non.
        diagnosis = diagnose_changed_cases(results, run_a, run_b)
        print_table(f"Table H -- Diagnostic des verdicts changes ({run_a} vs {run_b})", diagnosis)
        write_csv(os.path.join(OUT_DIR, "table_h_change_diagnosis.csv"), diagnosis)

        if diagnosis:
            n_confounded = sum(1 for r in diagnosis if r["first_diverging_stage"] != "NONE")
            n_clean = len(diagnosis) - n_confounded
            n_clean_with_marker = sum(
                1 for r in diagnosis
                if r["first_diverging_stage"] == "NONE" and r["structural_marker_present"]
            )
            n_clean_without_marker = n_clean - n_clean_with_marker
            print(f"\n[Table H] Sur {len(diagnosis)} cas a verdict change :")
            print(f"  - {n_confounded} ont un etage amont (state_space/preconditions/graph/"
                  f"matches/process_graph) deja different entre les deux runs -- NON "
                  f"attribuable a activity_guards, comparaison confondue sur ces cas.")
            print(f"  - {n_clean} n'ont AUCUNE divergence amont detectee, dont :")
            print(f"      - {n_clean_with_marker} avec le marqueur structurel present -- "
                  f"coherent avec un effet reel d'activity_guards.")
            print(f"      - {n_clean_without_marker} SANS aucun marqueur structurel -- "
                  f"anomalie a investiguer (aucune cause connue du changement de verdict).")
    else:
        print(f"\n[analyse] Table G/H ignorees -- un seul run_id present ({all_run_ids}), rien a comparer.")

    print(f"\nCSV ecrits dans {OUT_DIR}")