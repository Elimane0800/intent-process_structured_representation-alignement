"""Analyses exploratoires sur les runs complets du dataset (results/dataset_runs/) -- un seul
fichier, une fonction par analyse, tout execute et affiche a la suite. Ne modifie rien, lecture
seule sur les resultats deja produits par run_dataset.py."""

import json
import os
import glob
import re
from collections import defaultdict, Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "dataset_runs_v2")
BPMN_ROOT = os.path.join(os.path.dirname(__file__), "text_and_bpmn", "bpmn")
NOTES_DIR = os.path.join(os.path.dirname(__file__), "results", "notes")

MODELS = ["llama-3.1-8b", "mistral-nemotron", "gpt-oss-20b"]


def load_all_runs() -> dict:
    """model -> {(desc, bpmn_file): final_state}

    Recherche RECURSIVE (pas seulement a la racine du dossier modele) -- tolerant a un
    eventuel niveau de nesting supplementaire issu d'un renommage manuel (ex.
    results/dataset_runs/gpt-oss-20b/gpt-oss-20b/*.json au lieu de .../gpt-oss-20b/*.json)."""
    data = {}
    for model in MODELS:
        model_dir = os.path.join(RESULTS_DIR, model)
        data[model] = {}
        if not os.path.isdir(model_dir):
            print(f"  [{model}] dossier introuvable: {model_dir}")
            continue
        paths = glob.glob(os.path.join(model_dir, "**", "*.json"), recursive=True)
        for path in paths:
            fname = os.path.basename(path)[:-5]
            if "__" not in fname:
                continue
            desc, bpmn_file = fname.split("__", 1)
            with open(path) as f:
                data[model][(desc, bpmn_file)] = json.load(f)
    return data


def load_quality_scores() -> dict:
    """(desc, bpmn_file) -> note Zenodo (float), depuis les *.quality.txt du corpus."""
    scores = {}
    for desc_folder in os.listdir(BPMN_ROOT):
        desc_path = os.path.join(BPMN_ROOT, desc_folder)
        if not os.path.isdir(desc_path):
            continue
        for f in os.listdir(desc_path):
            if f.endswith(".quality.txt"):
                bpmn_file = f.replace(".quality.txt", ".bpmn2.xml")
                try:
                    with open(os.path.join(desc_path, f)) as qf:
                        scores[(desc_folder, bpmn_file)] = float(qf.read().strip())
                except (ValueError, OSError):
                    pass
    return scores


def alignment_counts(final_state: dict):
    """(n_conforme, n_ecarts, n_non_verifiable) depuis report.summary, ou None si absent."""
    report = final_state.get("report")
    if not report:
        return None
    s = report.get("summary", {})
    return s.get("conforme", 0), s.get("ecarts", 0), s.get("non_verifiable", 0)


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --- 1. Correlation avec la note Zenodo ---

def analysis_zenodo_correlation(runs, quality_scores):
    section("1. CORRELATION AVEC LA NOTE ZENODO (par modele)")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy indisponible -- pip install scipy --break-system-packages")
        return

    for model in MODELS:
        ratios, quality_vals = [], []
        for key, state in runs[model].items():
            counts = alignment_counts(state)
            if counts is None or key not in quality_scores:
                continue
            conforme, ecarts, non_verif = counts
            total = conforme + ecarts + non_verif
            if total == 0:
                continue
            ratios.append(conforme / total)
            quality_vals.append(quality_scores[key])
        if len(ratios) < 3:
            print(f"  [{model}] pas assez de donnees ({len(ratios)} cas)")
            continue
        rho, pval = spearmanr(ratios, quality_vals)
        print(f"  [{model}] n={len(ratios)}  Spearman rho={rho:.3f}  p={pval:.4f}")


# --- 2. Frequence de VIOLATED ---

def analysis_violated_frequency(runs):
    section("2. FREQUENCE DE VIOLATED (jamais observe a cette echelle avant)")
    for model in MODELS:
        total_checks = 0
        n_violated = 0
        cases_with_violated = []
        for key, state in runs[model].items():
            counts = alignment_counts(state)
            if counts is None:
                continue
            conforme, ecarts, non_verif = counts
            total_checks += conforme + ecarts + non_verif
            n_violated += ecarts
            if ecarts > 0:
                cases_with_violated.append(key)
        if total_checks:
            print(f"  [{model}] VIOLATED: {n_violated}/{total_checks} "
                  f"({100 * n_violated / total_checks:.2f}%)")
        else:
            print(f"  [{model}] aucune donnee")
        if cases_with_violated:
            shown = cases_with_violated[:10]
            more = f" (+{len(cases_with_violated) - 10} autres)" if len(cases_with_violated) > 10 else ""
            print(f"    cas concernes: {shown}{more}")


# --- 3. Etats de U jamais matches (biais structurel) ---

def analysis_never_matched_states(runs):
    section("3. ETATS DE U JAMAIS MATCHES PAR DESCRIPTION (biais structurel)")
    for model in MODELS:
        never_matched = defaultdict(Counter)
        for (desc, bpmn_file), state in runs[model].items():
            alignment = state.get("alignment")
            if not alignment:
                continue
            for target, entry in alignment.items():
                for term in entry.get("terms", []):
                    if term["status"] == "UNRESOLVABLE":
                        for m in term.get("missing", [term["term"]]):
                            never_matched[desc][m] += 1
        print(f"  [{model}]")
        for desc, counter in sorted(never_matched.items()):
            print(f"    {desc}: {counter.most_common(3)}")


# --- 4. Taux de cles de precondition malformees ---

def analysis_malformed_keys(runs):
    section("4. TAUX DE CLES DE PRECONDITION MALFORMEES (par modele)")
    for model in MODELS:
        total, n_malformed = 0, 0
        for key, state in runs[model].items():
            validated = state.get("validated_preconditions")
            if not validated:
                continue
            for target, entry in validated.items():
                total += 1
                if "malformed or unknown target key" in (entry.get("reason") or ""):
                    n_malformed += 1
        pct = 100 * n_malformed / total if total else 0
        print(f"  [{model}] {n_malformed}/{total} entrees malformees ({pct:.1f}%)")


# --- 5. Erreurs internes et rapports manquants ---

def analysis_internal_errors(runs):
    section("5. ERREURS INTERNES ET RAPPORTS MANQUANTS (par modele)")
    for model in MODELS:
        total = len(runs[model])
        with_errors = sum(1 for s in runs[model].values() if s.get("errors"))
        no_report = sum(1 for s in runs[model].values() if not s.get("report"))
        print(f"  [{model}] total={total}  avec_erreurs={with_errors}  sans_rapport={no_report}")


# --- 6. Distribution des raisons de DOWNGRADED ---

def analysis_downgrade_reasons(runs):
    section("6. DISTRIBUTION DES RAISONS DE DOWNGRADED (par modele)")
    patterns = [
        ("malformed or unknown target key", "malformed_target_key"),
        ("references unknown state", "unknown_referenced_state"),
        ("self-reference", "self_reference"),
        ("mutual exclusivity", "mutual_exclusivity"),
        ("quote not found", "quote_not_grounded"),
        ("mixed AND/OR", "mixed_operator"),
    ]
    for model in MODELS:
        reasons = Counter()
        for key, state in runs[model].items():
            validated = state.get("validated_preconditions")
            if not validated:
                continue
            for target, entry in validated.items():
                if entry.get("status") != "DOWNGRADED":
                    continue
                reason = entry.get("reason") or ""
                norm = next((label for pat, label in patterns if pat in reason), "other")
                reasons[norm] += 1
        print(f"  [{model}] {dict(reasons)}")


# --- 7. Frequence des cycles detectes ---

def analysis_cycle_frequency(runs):
    section("7. FREQUENCE DES CYCLES DETECTES (par modele)")
    for model in MODELS:
        cases_with_cycles = [
            (key, len(state["precondition_cycles"]))
            for key, state in runs[model].items()
            if state.get("precondition_cycles")
        ]
        print(f"  [{model}] {len(cases_with_cycles)}/{len(runs[model])} cas avec au moins un cycle")
        for key, n in cases_with_cycles[:10]:
            print(f"    {key}: {n} cycle(s)")


# --- 8. Reproductibilite a temperature=0 ---

def analysis_reproducibility_note():
    section("8. REPRODUCTIBILITE A TEMPERATURE=0")
    print("  Non analysable depuis les runs existants -- necessite de relancer le meme")
    print("  (modele, cas) une seconde fois et de comparer. Point reste ouvert (cf.")
    print("  pipeline_observations.md, limite 3) -- pas traite ici.")


# --- 9. Comparaison inter-modeles : exploitabilite ---

def analysis_model_comparison(runs):
    section("9. COMPARAISON INTER-MODELES : TAUX D'EXPLOITABILITE GLOBAL")
    for model in MODELS:
        tot_s = tot_v = tot_u = 0
        for key, state in runs[model].items():
            counts = alignment_counts(state)
            if counts is None:
                continue
            s, v, u = counts
            tot_s += s
            tot_v += v
            tot_u += u
        total = tot_s + tot_v + tot_u
        if total == 0:
            print(f"  [{model}] aucune donnee")
            continue
        print(f"  [{model}] SATISFIED={100 * tot_s / total:.1f}%  "
              f"VIOLATED={100 * tot_v / total:.1f}%  "
              f"UNRESOLVABLE={100 * tot_u / total:.1f}%")


# --- 10. Accord inter-modeles sur un meme cas ---

def analysis_inter_model_agreement(runs):
    section("10. ACCORD INTER-MODELES SUR UN MEME CAS")
    common_keys = set(runs[MODELS[0]].keys())
    for m in MODELS[1:]:
        common_keys &= set(runs[m].keys())
    print(f"  {len(common_keys)} cas communs aux {len(MODELS)} modeles")

    def dominant(state):
        counts = alignment_counts(state)
        if counts is None:
            return None
        s, v, u = counts
        return max([("SATISFIED", s), ("VIOLATED", v), ("UNRESOLVABLE", u)], key=lambda x: x[1])[0]

    agree = 0
    disagreements = []
    for key in common_keys:
        statuses = [dominant(runs[m][key]) for m in MODELS]
        if len(set(statuses)) == 1:
            agree += 1
        else:
            disagreements.append((key, statuses))
    if common_keys:
        print(f"  Accord total: {agree}/{len(common_keys)} ({100 * agree / len(common_keys):.1f}%)")
    for key, statuses in disagreements[:10]:
        print(f"    Desaccord {key}: {dict(zip(MODELS, statuses))}")


# --- 11. Richesse de U par modele ---

def analysis_u_richness(runs):
    section("11. RICHESSE DE U (nb d'etats extraits) PAR MODELE")
    for model in MODELS:
        sizes = [
            sum(len(v) for v in state["state_space"].values())
            for state in runs[model].values() if state.get("state_space")
        ]
        if sizes:
            print(f"  [{model}] n={len(sizes)}  moyenne={sum(sizes) / len(sizes):.1f}  "
                  f"min={min(sizes)}  max={max(sizes)}")


# --- 12. Complexite du BPMN vs resultat d'alignement ---

def analysis_bpmn_complexity(runs):
    section("12. COMPLEXITE DU BPMN (nb aretes DFG) VS TAUX SATISFIED")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy indisponible")
        return
    for model in MODELS:
        pairs = []
        for key, state in runs[model].items():
            dfg = state.get("dfg_edges")
            counts = alignment_counts(state)
            if dfg is None or counts is None:
                continue
            s, v, u = counts
            total = s + v + u
            if total == 0:
                continue
            pairs.append((len(dfg), s / total))
        if len(pairs) < 3:
            print(f"  [{model}] pas assez de donnees")
            continue
        edges, ratios = zip(*pairs)
        rho, pval = spearmanr(edges, ratios)
        print(f"  [{model}] n={len(pairs)}  Spearman(nb_aretes, taux_SATISFIED)="
              f"rho={rho:.3f} p={pval:.4f}")


# --- 13. Impact de la boucle perdue ---

def analysis_loop_loss_impact(runs):
    section("13. IMPACT DE LA BOUCLE PERDUE SUR L'ALIGNEMENT")
    audit_path = os.path.join(NOTES_DIR, "bpmn_silent_loss_audit.json")
    if not os.path.exists(audit_path):
        print(f"  {audit_path} introuvable -- lancer audit_bpmn_silent_losses.py d'abord")
        return
    with open(audit_path) as f:
        audit = json.load(f)
    loop_files = {
        (desc, fname) for desc, models_ in audit.items()
        for fname, r in models_.items() if r.get("loop_count")
    }

    for model in MODELS:
        with_loop, without_loop = [], []
        for key, state in runs[model].items():
            counts = alignment_counts(state)
            if counts is None:
                continue
            s, v, u = counts
            total = s + v + u
            if total == 0:
                continue
            (with_loop if key in loop_files else without_loop).append(s / total)
        if with_loop and without_loop:
            print(f"  [{model}] avec boucle perdue (n={len(with_loop)}): "
                  f"SATISFIED moyen={sum(with_loop) / len(with_loop):.2f}")
            print(f"           sans (n={len(without_loop)}):              "
                  f"SATISFIED moyen={sum(without_loop) / len(without_loop):.2f}")


# --- 14-15. Variance par description ---

def analysis_per_description_variance(runs):
    section("14-15. VARIANCE PAR DESCRIPTION (intra-cas)")
    for model in MODELS:
        by_desc = defaultdict(list)
        for (desc, bpmn_file), state in runs[model].items():
            counts = alignment_counts(state)
            if counts is None:
                continue
            s, v, u = counts
            total = s + v + u
            if total:
                by_desc[desc].append(s / total)
        print(f"  [{model}]")
        for desc, ratios in sorted(by_desc.items()):
            if len(ratios) > 1:
                mean = sum(ratios) / len(ratios)
                var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
                print(f"    {desc}: n={len(ratios)} moyenne={mean:.2f} variance={var:.3f}")


# --- 16. Compression du rapport par cause racine ---

def analysis_report_compression(runs):
    section("16. RATIO DE COMPRESSION DU REGROUPEMENT PAR CAUSE RACINE")
    for model in MODELS:
        raw_failures = clustered_causes = 0
        for key, state in runs[model].items():
            report = state.get("report")
            if not report:
                continue
            for item in report.get("ecarts", []) + report.get("non_verifiable", []):
                clustered_causes += 1
                raw_failures += item.get("affected_count", 1)
        if clustered_causes:
            print(f"  [{model}] {raw_failures} echecs bruts -> {clustered_causes} causes "
                  f"distinctes (ratio {raw_failures / clustered_causes:.2f}x)")


# --- 17. Citations manquantes ---

def analysis_missing_citations(runs):
    section("17. FREQUENCE DES CITATIONS MANQUANTES DANS L'ALIGNEMENT")
    for model in MODELS:
        total = missing = 0
        for key, state in runs[model].items():
            alignment = state.get("alignment")
            if not alignment:
                continue
            for target, entry in alignment.items():
                total += 1
                if not entry.get("quote"):
                    missing += 1
        if total:
            print(f"  [{model}] {missing}/{total} citations manquantes "
                  f"({100 * missing / total:.1f}%)")


# --- 18. Detail des erreurs par node (comptage + echantillon de messages reels) ---

def analysis_error_breakdown(runs):
    section("18. DETAIL DES ERREURS PAR NODE (comptage + echantillon de messages)")
    for model in MODELS:
        node_counts = Counter()
        node_samples = defaultdict(list)
        for key, state in runs[model].items():
            for err in state.get("errors", []):
                node_name = err.split(":", 1)[0].strip()
                node_counts[node_name] += 1
                if len(node_samples[node_name]) < 5:
                    node_samples[node_name].append((key, err))

        print(f"  [{model}] cas avec erreurs par node : {dict(node_counts)}")
        for node_name, samples in node_samples.items():
            print(f"\n    --- {node_name} : {node_counts[node_name]} cas au total, "
                  f"echantillon de {len(samples)} ---")
            for key, err in samples:
                truncated = err if len(err) <= 300 else err[:300] + "... (tronque)"
                print(f"      {key}: {truncated}")


# --- 19. Re-analyse sur le sous-ensemble propre (isole le signal du bruit d'infrastructure) ---

def analysis_clean_subset(runs, quality_scores):
    section("19. RE-ANALYSE SUR LE SOUS-ENSEMBLE PROPRE (sans erreur interne, rapport present)")
    clean = {
        model: {
            key: state for key, state in runs[model].items()
            if not state.get("errors") and state.get("report")
        }
        for model in MODELS
    }
    for model in MODELS:
        total = len(runs[model])
        n_clean = len(clean[model])
        pct = 100 * n_clean / total if total else 0
        print(f"  [{model}] {n_clean}/{total} cas propres ({pct:.1f}%)")

    print("\n  --- Correlation Zenodo (sous-ensemble propre) ---")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("    scipy indisponible")
        return clean

    for model in MODELS:
        ratios, quality_vals = [], []
        for key, state in clean[model].items():
            counts = alignment_counts(state)
            if counts is None or key not in quality_scores:
                continue
            conforme, ecarts, non_verif = counts
            total = conforme + ecarts + non_verif
            if total == 0:
                continue
            ratios.append(conforme / total)
            quality_vals.append(quality_scores[key])
        if len(ratios) < 3:
            print(f"    [{model}] pas assez de donnees ({len(ratios)} cas)")
            continue
        rho, pval = spearmanr(ratios, quality_vals)
        print(f"    [{model}] n={len(ratios)}  Spearman rho={rho:.3f}  p={pval:.4f}")

    print("\n  --- Exploitabilite globale (sous-ensemble propre) ---")
    for model in MODELS:
        tot_s = tot_v = tot_u = 0
        for key, state in clean[model].items():
            counts = alignment_counts(state)
            if counts is None:
                continue
            s, v, u = counts
            tot_s += s
            tot_v += v
            tot_u += u
        total = tot_s + tot_v + tot_u
        if total == 0:
            print(f"    [{model}] aucune donnee")
            continue
        print(f"    [{model}] SATISFIED={100 * tot_s / total:.1f}%  "
              f"VIOLATED={100 * tot_v / total:.1f}%  "
              f"UNRESOLVABLE={100 * tot_u / total:.1f}%")

    return clean


# --- 20. Accord cas par cas (comptage direct, pas un coefficient) ---

def analysis_case_level_agreement(runs, quality_scores):
    section("20. MATRICE DE CONFUSION : METHODE vs NOTE ZENODO (pas juste le taux d'accord)")
    print("  Seuils : Zenodo 'bon' si note>=3.5, 'mauvais' si note<=1.5 (entre les deux exclu)")
    print("           Methode 'bon' si taux SATISFIED>=0.5, 'mauvais' si <0.5")

    for model in MODELS:
        tp = fp = fn = tn = excluded_mid = 0  # zenodo_bon/mauvais x methode_bon/mauvais
        for key, state in runs[model].items():
            counts = alignment_counts(state)
            if counts is None or key not in quality_scores:
                continue
            conforme, ecarts, non_verif = counts
            total = conforme + ecarts + non_verif
            if total == 0:
                continue
            ratio = conforme / total
            q = quality_scores[key]
            if 1.5 < q < 3.5:
                excluded_mid += 1
                continue
            q_good = q >= 3.5
            r_good = ratio >= 0.5
            if q_good and r_good:
                tp += 1
            elif q_good and not r_good:
                fn += 1
            elif not q_good and r_good:
                fp += 1
            else:
                tn += 1

        total_classified = tp + fp + fn + tn
        n_method_good = tp + fp
        n_method_bad = fn + tn
        n_zenodo_good = tp + fn
        n_zenodo_bad = fp + tn

        print(f"\n  [{model}] (n={total_classified}, exclus intermediaires={excluded_mid})")
        print(f"    Repartition METHODE : bon={n_method_good} ({100*n_method_good/total_classified:.1f}%)  "
              f"mauvais={n_method_bad} ({100*n_method_bad/total_classified:.1f}%)")
        print(f"    Repartition ZENODO  : bon={n_zenodo_good} ({100*n_zenodo_good/total_classified:.1f}%)  "
              f"mauvais={n_zenodo_bad} ({100*n_zenodo_bad/total_classified:.1f}%)")
        print(f"    Matrice :                Zenodo bon   Zenodo mauvais")
        print(f"      Methode bon      :      {tp:>5}         {fp:>5}")
        print(f"      Methode mauvais  :      {fn:>5}         {tn:>5}")
        if n_zenodo_bad > 0:
            print(f"    Rappel sur 'mauvais' (tn / zenodo_mauvais) : {tn}/{n_zenodo_bad} "
                  f"({100*tn/n_zenodo_bad:.1f}%) -- la methode detecte-t-elle les vrais mauvais cas ?")
        else:
            print(f"    Aucun cas Zenodo 'mauvais' dans cet echantillon -- rappel non calculable")


# --- 21. Detection systematique des matchs antonymes (activite vs etat) via WordNet ---

def _content_words(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower().replace("_", " ")))


_STOPWORDS = {"the", "a", "an", "of", "to", "for", "is", "are", "be", "been", "on", "in", "at",
              "and", "or", "with", "by", "your", "you", "job", "process", "application"}


def find_antonym_mismatches(matches: dict):
    """Pour chaque (activite, etat matche), verifie via WordNet si un mot-cle de l'activite est
    un antonyme direct d'un mot-cle du nom d'etat -- meme mecanisme (WordNet, deterministe) deja
    utilise ailleurs dans le pipeline (lexnames ACTOR/ENTITY), applique ici a la relation
    d'antonymie plutot qu'a la classification lexicale."""
    from nltk.corpus import wordnet as wn

    findings = []
    for activity_label, m in matches.items():
        state = m["match"]
        entity, _, state_name = state.partition(".")
        activity_words = _content_words(activity_label) - _STOPWORDS
        state_words = (_content_words(state_name) | _content_words(entity)) - _STOPWORDS

        for aw in activity_words:
            for synset in wn.synsets(aw):
                for lemma in synset.lemmas():
                    for antonym in lemma.antonyms():
                        ant_word = antonym.name().lower()
                        if ant_word in state_words:
                            findings.append({
                                "activity": activity_label, "state": state,
                                "activity_word": aw, "state_word": ant_word, "score": m["score"],
                            })
    return findings


def analysis_antonym_mismatches(runs):
    section("21. DETECTION SYSTEMATIQUE DES MATCHS ANTONYMES (activite vs etat, via WordNet)")
    for model in MODELS:
        total_findings = 0
        cases_with_finding = []
        for key, state in runs[model].items():
            matches = state.get("matches")
            if not matches:
                continue
            findings = find_antonym_mismatches(matches)
            if findings:
                total_findings += len(findings)
                cases_with_finding.append((key, findings))

        print(f"  [{model}] {len(cases_with_finding)} cas avec au moins un match antonyme, "
              f"{total_findings} matchs antonymes au total")
        for key, findings in cases_with_finding[:10]:
            for f in findings[:3]:
                print(f"    {key}: {f['activity']!r} -> {f['state']} "
                      f"(mot activite={f['activity_word']!r} vs mot etat={f['state_word']!r}, "
                      f"score={f['score']:.2f})")


# --- 21. Matchs a faible confiance impliques dans des SATISFIED/VIOLATED (proxy pratique) ---
#
# La detection par antonymie WordNet a ete testee et rejetee : sur le seul cas reel connu
# (Unemployed matche a job.permanent, score 0.29), WordNet ne relie "permanent" a "unemployed"/
# "employed" par aucune relation d'antonymie, meme en elargissant aux synonymes -- limite de
# l'outil (contradiction contextuelle en deux sauts, pas un antonyme lexical direct), pas un bug
# du script. Approche remplacee par un signal deja disponible sans rien construire de nouveau :
# le score de confiance du match lui-meme etait deja bas (0.29) sur le cas connu -- on verifie
# si un seuil bas repere ce type de cas a moindre cout que l'antonymie.

LOW_SCORE_THRESHOLD = 0.35


def find_low_confidence_conclusions(alignment: dict, matches: dict) -> list:
    """Pour chaque terme SATISFIED ou VIOLATED (jamais UNRESOLVABLE, qui n'a par definition pas
    de match), verifie si le terme ou la cible repose sur un match de score bas -- ces
    conclusions sont probablement les plus a risque d'etre comme le cas Unemployed/job.permanent."""
    # etat -> meilleur score parmi les activites qui y ont ete matchees
    best_score_for_state = {}
    for activity_label, m in matches.items():
        state = m["match"]
        if state not in best_score_for_state or m["score"] > best_score_for_state[state]:
            best_score_for_state[state] = m["score"]

    findings = []
    for target, entry in alignment.items():
        for term_result in entry["terms"]:
            if term_result["status"] not in ("SATISFIED", "VIOLATED"):
                continue
            term = term_result["term"]
            term_score = best_score_for_state.get(term)
            target_score = best_score_for_state.get(target)
            worst = min(s for s in (term_score, target_score) if s is not None) \
                if (term_score is not None or target_score is not None) else None
            if worst is not None and worst < LOW_SCORE_THRESHOLD:
                findings.append({
                    "target": target, "term": term, "status": term_result["status"],
                    "term_score": term_score, "target_score": target_score,
                })
    return findings


def analysis_low_confidence_conclusions(runs):
    section("21. CONCLUSIONS (SATISFIED/VIOLATED) REPOSANT SUR UN MATCH A FAIBLE CONFIANCE")
    print(f"  Seuil : score < {LOW_SCORE_THRESHOLD} sur le terme ou la cible impliquee")
    for model in MODELS:
        total_conclusions = 0
        total_low_confidence = 0
        examples = []
        for key, state in runs[model].items():
            alignment = state.get("alignment")
            matches = state.get("matches")
            if not alignment or not matches:
                continue
            for entry in alignment.values():
                for t in entry["terms"]:
                    if t["status"] in ("SATISFIED", "VIOLATED"):
                        total_conclusions += 1
            findings = find_low_confidence_conclusions(alignment, matches)
            total_low_confidence += len(findings)
            for f in findings[:2]:
                examples.append((key, f))

        pct = 100 * total_low_confidence / total_conclusions if total_conclusions else 0
        print(f"\n  [{model}] {total_low_confidence}/{total_conclusions} conclusions "
              f"({pct:.1f}%) reposent sur un match a faible confiance")
        for key, f in examples[:8]:
            print(f"    {key}: {f['target']} <- {f['term']} [{f['status']}] "
                  f"(scores: terme={f['term_score']}, cible={f['target_score']})")
            

# --- 22. Confiance du matching vs accord inter-modeles (validation sans ground truth externe) ---

def analysis_confidence_vs_agreement(runs):
    """Teste si les cas avec un taux eleve de conclusions a faible confiance (score matching
    < LOW_SCORE_THRESHOLD) montrent un desaccord inter-modeles plus frequent que les cas a
    confiance elevee. Restreint aux cas ou les trois modeles ont un rapport complet (pas de
    None), pour ne pas confondre desaccord de jugement et incompletude de donnees (point ouvert
    de la section 10/12)."""
    section("22. CONFIANCE DU MATCHING VS ACCORD INTER-MODELES")

    # cas communs aux trois modeles, avec state_space/alignment/matches presents (rapport complet)
    common_keys = set(runs[MODELS[0]].keys())
    for m in MODELS[1:]:
        common_keys &= set(runs[m].keys())

    complete_keys = []
    for key in common_keys:
        if all(
            alignment_counts(runs[m][key]) is not None
            and runs[m][key].get("alignment")
            and runs[m][key].get("matches")
            for m in MODELS
        ):
            complete_keys.append(key)

    print(f"  {len(complete_keys)}/{len(common_keys)} cas communs avec rapport complet "
          f"sur les trois modeles (avant filtrage, {len(common_keys)} cas partagent juste la cle)")

    if not complete_keys:
        print("  Aucun cas exploitable -- arret de l'analyse")
        return

    def dominant(state):
        counts = alignment_counts(state)
        s, v, u = counts
        return max([("SATISFIED", s), ("VIOLATED", v), ("UNRESOLVABLE", u)], key=lambda x: x[1])[0]

    def low_confidence_fraction(state):
        """Reutilise find_low_confidence_conclusions (section 21) -- pas de nouvelle logique."""
        alignment = state.get("alignment")
        matches = state.get("matches")
        total_conclusions = sum(
            1 for entry in alignment.values() for t in entry["terms"]
            if t["status"] in ("SATISFIED", "VIOLATED")
        )
        if total_conclusions == 0:
            return None
        findings = find_low_confidence_conclusions(alignment, matches)
        return len(findings) / total_conclusions

    rows = []
    for key in complete_keys:
        fractions = [low_confidence_fraction(runs[m][key]) for m in MODELS]
        fractions = [f for f in fractions if f is not None]
        if not fractions:
            continue
        avg_low_conf = sum(fractions) / len(fractions)
        statuses = [dominant(runs[m][key]) for m in MODELS]
        agree = len(set(statuses)) == 1
        rows.append((key, avg_low_conf, agree))

    if not rows:
        print("  Aucune fraction de confiance calculable -- arret")
        return

    # Repartition en deux groupes par la mediane, plutot qu'un seuil arbitraire fixe
    sorted_fracs = sorted(r[1] for r in rows)
    median = sorted_fracs[len(sorted_fracs) // 2]
    print(f"\n  n={len(rows)} cas classables, mediane du taux de faible confiance = {median:.3f}")

    high_group = [r for r in rows if r[1] >= median]
    low_group = [r for r in rows if r[1] < median]

    def agreement_rate(group):
        if not group:
            return None
        return sum(1 for _, _, agree in group if agree) / len(group)

    rate_high = agreement_rate(high_group)
    rate_low = agreement_rate(low_group)

    print(f"\n  Table de contingence (accord = meme statut dominant sur les 3 modeles) :")
    print(f"    Groupe taux de faible confiance ELEVE (n={len(high_group)}) : "
          f"accord={sum(1 for *_, a in high_group if a)}/{len(high_group)} "
          f"({100 * rate_high:.1f}%)" if rate_high is not None else "    (groupe eleve vide)")
    print(f"    Groupe taux de faible confiance FAIBLE (n={len(low_group)}) : "
          f"accord={sum(1 for *_, a in low_group if a)}/{len(low_group)} "
          f"({100 * rate_low:.1f}%)" if rate_low is not None else "    (groupe faible vide)")

    if rate_high is not None and rate_low is not None:
        diff = rate_low - rate_high
        print(f"\n  Ecart d'accord (groupe faible - groupe eleve) : {100 * diff:+.1f} points")
        if diff > 0:
            print("  -> Direction attendue : les cas a faible confiance desaccordent plus souvent.")
        else:
            print("  -> Direction INATTENDUE : pas de signal, ou signal inverse -- a ne pas ignorer.")

    # Test de significativite simple (chi2), si scipy disponible et effectifs suffisants
    try:
        from scipy.stats import chi2_contingency
        a = sum(1 for *_, ag in high_group if ag)
        b = len(high_group) - a
        c = sum(1 for *_, ag in low_group if ag)
        d = len(low_group) - c
        if min(a, b, c, d) >= 0 and (a + b) > 0 and (c + d) > 0:
            table = [[a, b], [c, d]]
            chi2, p, dof, _ = chi2_contingency(table)
            print(f"\n  Chi2={chi2:.3f}  p={p:.4f}  (table : {table})")
    except ImportError:
        print("\n  scipy indisponible -- pas de test chi2, ecart brut ci-dessus seulement")

    # Details des cas de desaccord dans le groupe a faible confiance elevee, pour inspection manuelle
    disagreements_high = [(k, f) for k, f, ag in high_group if not ag]
    print(f"\n  Echantillon de desaccords dans le groupe a confiance faible elevee "
          f"({len(disagreements_high)} cas) -- a inspecter manuellement en priorite :")
    for key, frac in disagreements_high[:10]:
        print(f"    {key}: taux faible confiance moyen={frac:.2f}")


# --- 23. Distribution des scores de matching -- bimodalite pour calibrer un seuil non supervise ---
#
# Teste si les scores de similarite cosinus produits par state_matching_node (deja calcules,
# aucun nouvel appel LLM/embedding necessaire) ont une forme bimodale -- un pic de "bons
# matches" et un pic de "matches rates". Si oui, la vallee entre les deux pics donne un seuil
# defendable sans dependre du seul cas connu (Unemployed/job.permanent, score 0.29) ni du petit
# echantillon Zenodo "mauvais" (0 a 6 cas, section 3bis).

def collect_all_match_scores(runs) -> dict:
    """model -> liste de tous les scores de matches (bruts, avant tout filtrage SATISFIED/
    VIOLATED/UNRESOLVABLE) -- la distribution complete, pas seulement celle des conclusions
    retenues, pour juger honnetement de la forme globale."""
    scores = {model: [] for model in MODELS}
    for model in MODELS:
        for key, state in runs[model].items():
            matches = state.get("matches")
            if not matches:
                continue
            for activity_label, m in matches.items():
                score = m.get("score")
                if score is not None:
                    scores[model].append(score)
    return scores


def _gmm_bimodality(scores_arr):
    """Compare GMM a 1 vs 2 composantes par BIC -- plus bas = meilleur ajustement. Si le modele
    a 2 composantes gagne nettement, la distribution est plausiblement bimodale. Retourne
    (bic_1, bic_2, means_2, weights_2, threshold) ou threshold est le point de croisement des
    deux gaussiennes (calcule numeriquement sur une grille), None si non calculable."""
    import numpy as np
    from sklearn.mixture import GaussianMixture

    X = scores_arr.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
    bic1, bic2 = gmm1.bic(X), gmm2.bic(X)

    means = gmm2.means_.flatten()
    order = means.argsort()
    means_sorted = means[order]
    weights_sorted = gmm2.weights_[order]
    stds_sorted = (gmm2.covariances_.flatten() ** 0.5)[order]

    # recherche du seuil : point ou les deux densites ponderees s'egalisent, entre les deux moyennes
    grid = np.linspace(means_sorted[0], means_sorted[1], 500)

    def dens(x, mean, std, weight):
        return weight * (1.0 / (std * (2 * np.pi) ** 0.5)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    d_low = dens(grid, means_sorted[0], stds_sorted[0], weights_sorted[0])
    d_high = dens(grid, means_sorted[1], stds_sorted[1], weights_sorted[1])
    diff = d_low - d_high
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    threshold = grid[sign_changes[0]] if len(sign_changes) else None

    return bic1, bic2, means_sorted, weights_sorted, threshold


def analysis_matching_score_bimodality(runs):
    section("23. DISTRIBUTION DES SCORES DE MATCHING -- BIMODALITE (calibration non supervisee)")
    try:
        import numpy as np
    except ImportError:
        print("  numpy indisponible -- pip install numpy --break-system-packages")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        has_plot = True
    except ImportError:
        print("  matplotlib indisponible -- histogramme non trace (pip install matplotlib)")
        has_plot = False
    try:
        from sklearn.mixture import GaussianMixture  # noqa: F401
        has_gmm = True
    except ImportError:
        print("  scikit-learn indisponible -- test GMM saute (pip install scikit-learn)")
        has_gmm = False

    all_scores = collect_all_match_scores(runs)

    for model in MODELS:
        scores = all_scores[model]
        n = len(scores)
        if n < 20:
            print(f"  [{model}] n={n} -- trop peu de scores pour un diagnostic fiable")
            continue
        arr = np.array(scores)
        print(f"\n  [{model}] n={n}  moyenne={arr.mean():.3f}  ecart-type={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")

        q10, q25, q50 = np.percentile(arr, [10, 25, 50])
        print(f"    percentiles : p10={q10:.3f}  p25={q25:.3f}  mediane={q50:.3f}")
        pct_below_035 = 100 * (arr < LOW_SCORE_THRESHOLD).sum() / n
        print(f"    part des scores < {LOW_SCORE_THRESHOLD} (seuil actuel) : {pct_below_035:.1f}%")

        bic1 = bic2 = threshold = None
        if has_gmm:
            bic1, bic2, means_sorted, weights_sorted, threshold = _gmm_bimodality(arr)
            print(f"    GMM 1 composante  : BIC={bic1:.1f}")
            print(f"    GMM 2 composantes : BIC={bic2:.1f}  "
                  f"moyennes={means_sorted[0]:.3f}/{means_sorted[1]:.3f}  "
                  f"poids={weights_sorted[0]:.2f}/{weights_sorted[1]:.2f}")
            if bic2 < bic1:
                print(f"    -> 2 composantes preferees par BIC (delta={bic1 - bic2:.1f}) "
                      f"-- distribution plausiblement bimodale")
                if threshold is not None:
                    print(f"    -> Seuil suggere (croisement des deux gaussiennes) : {threshold:.3f}")
                else:
                    print(f"    -> Croisement non trouve numeriquement sur la grille testee")
            else:
                print(f"    -> 1 composante preferee par BIC -- pas de signal clair de bimodalite, "
                      f"un seuil relatif (percentile) est probablement plus defendable qu'un seuil "
                      f"absolu ici")

        if has_plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(arr, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
            ax.axvline(LOW_SCORE_THRESHOLD, color="red", linestyle="--",
                       label=f"seuil actuel ({LOW_SCORE_THRESHOLD})")
            if has_gmm and bic2 is not None and bic1 is not None and bic2 < bic1 and threshold is not None:
                ax.axvline(threshold, color="green", linestyle="--",
                           label=f"seuil suggere GMM ({threshold:.3f})")
            ax.set_title(f"Distribution des scores de matching -- {model} (n={n})")
            ax.set_xlabel("Score de similarite cosinus")
            ax.set_ylabel("Frequence")
            ax.legend()
            out_path = os.path.join(NOTES_DIR, f"matching_score_histogram_{model}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"    Histogramme sauvegarde : {out_path}")


# --- 24. Section 21 recalculee avec le seuil GMM (au lieu du seuil fixe 0.35) ---
#
# Reutilise find_low_confidence_conclusions (section 21), mais avec un seuil par modele issu
# du croisement des deux gaussiennes de la section 23, au lieu du seuil fixe LOW_SCORE_THRESHOLD
# calibre sur un seul cas connu. Affiche l'ancien et le nouveau resultat cote a cote pour
# mesurer l'ecart, pas juste remplacer un chiffre par un autre sans comparaison.

def find_low_confidence_conclusions_with_threshold(alignment: dict, matches: dict, threshold: float) -> list:
    """Identique a find_low_confidence_conclusions (section 21), mais avec un seuil parametrable
    au lieu du seuil global LOW_SCORE_THRESHOLD -- pour comparer plusieurs seuils sans dupliquer
    toute la logique de section 21."""
    best_score_for_state = {}
    for activity_label, m in matches.items():
        state = m["match"]
        if state not in best_score_for_state or m["score"] > best_score_for_state[state]:
            best_score_for_state[state] = m["score"]

    findings = []
    for target, entry in alignment.items():
        for term_result in entry["terms"]:
            if term_result["status"] not in ("SATISFIED", "VIOLATED"):
                continue
            term = term_result["term"]
            term_score = best_score_for_state.get(term)
            target_score = best_score_for_state.get(target)
            worst = min(s for s in (term_score, target_score) if s is not None) \
                if (term_score is not None or target_score is not None) else None
            if worst is not None and worst < threshold:
                findings.append({
                    "target": target, "term": term, "status": term_result["status"],
                    "term_score": term_score, "target_score": target_score,
                })
    return findings


def analysis_low_confidence_gmm_threshold(runs):
    section("24. SECTION 21 RECALCULEE AVEC LE SEUIL GMM (par modele, vs seuil fixe 0.35)")
    try:
        import numpy as np
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("  numpy/scikit-learn indisponible -- section sautee")
        return

    all_scores = collect_all_match_scores(runs)

    for model in MODELS:
        scores = all_scores[model]
        if len(scores) < 20:
            print(f"  [{model}] pas assez de scores pour calibrer un seuil GMM")
            continue

        arr = np.array(scores)
        bic1, bic2, means_sorted, weights_sorted, gmm_threshold = _gmm_bimodality(arr)

        if bic2 >= bic1 or gmm_threshold is None:
            print(f"  [{model}] pas de seuil GMM fiable (bimodalite non confirmee) -- "
                  f"section sautee pour ce modele")
            continue

        print(f"\n  [{model}] seuil fixe actuel = {LOW_SCORE_THRESHOLD}  |  "
              f"seuil GMM = {gmm_threshold:.3f}")

        total_conclusions = 0
        total_old = 0
        total_new = 0
        new_only_examples = []

        for key, state in runs[model].items():
            alignment = state.get("alignment")
            matches = state.get("matches")
            if not alignment or not matches:
                continue

            for entry in alignment.values():
                for t in entry["terms"]:
                    if t["status"] in ("SATISFIED", "VIOLATED"):
                        total_conclusions += 1

            old_findings = find_low_confidence_conclusions(alignment, matches)  # seuil 0.35, section 21
            new_findings = find_low_confidence_conclusions_with_threshold(
                alignment, matches, gmm_threshold
            )
            total_old += len(old_findings)
            total_new += len(new_findings)

            # cas flagges par le nouveau seuil mais pas par l'ancien -- la vraie zone de divergence
            old_pairs = {(f["target"], f["term"]) for f in old_findings}
            for f in new_findings:
                if (f["target"], f["term"]) not in old_pairs:
                    new_only_examples.append((key, f))

        pct_old = 100 * total_old / total_conclusions if total_conclusions else 0
        pct_new = 100 * total_new / total_conclusions if total_conclusions else 0

        print(f"    Ancien (seuil {LOW_SCORE_THRESHOLD})  : {total_old}/{total_conclusions} "
              f"conclusions flaggees ({pct_old:.1f}%)")
        print(f"    Nouveau (seuil {gmm_threshold:.3f}) : {total_new}/{total_conclusions} "
              f"conclusions flaggees ({pct_new:.1f}%)")
        print(f"    Ecart : {pct_new - pct_old:+.1f} points  "
              f"({total_new - total_old:+d} conclusions supplementaires flaggees)")

        print(f"    Echantillon de conclusions flaggees UNIQUEMENT par le nouveau seuil "
              f"({len(new_only_examples)} au total, zone 0.35-{gmm_threshold:.2f}) :")
        for key, f in new_only_examples[:8]:
            print(f"      {key}: {f['target']} <- {f['term']} [{f['status']}] "
                  f"(scores: terme={f['term_score']}, cible={f['target_score']})")


import csv
import random
 
# --- 25. Quantification exhaustive des causes d'erreur (remplace l'echantillon de 5 messages) ---
#
# Ferme le point ouvert de dataset_run_observations.md section 2 : seuls 5 messages par node
# avaient ete examines a la main. Ici, pattern-matching sur TOUS les messages d'erreur de TOUS
# les cas (y compris les fatal_error de run.py), pour produire le tableau causes x modeles
# exhaustif -- necessaire avant d'ecrire "l'inexploitabilite est majoritairement
# infrastructurelle" dans le papier.
 
_ERROR_PATTERNS = [
    ("429", "rate_limit_429"),
    ("add_in_arc", "pm4py_add_arc_bug"),
    ("add_out_arc", "pm4py_add_arc_bug"),
    ("raw response=''", "empty_response"),
    ("raw response=\"\"", "empty_response"),
    ("parse failed", "json_parse_failed"),          # JSON tronque/malforme AVEC contenu -- teste
    ("Expecting value", "json_parse_failed"),        # apres empty_response, l'ordre compte
    ("KeyError", "cascade_missing_key"),             # champ absent car echec en amont
]
 
 
def _classify_error(err: str) -> str:
    for pat, label in _ERROR_PATTERNS:
        if pat in err:
            return label
    return "other"
 
 
def analysis_error_cause_quantification(runs):
    section("25. QUANTIFICATION EXHAUSTIVE DES CAUSES D'ERREUR (tous les messages, pas 5)")
    for model in MODELS:
        cause_counts = Counter()          # occurrences de messages, toutes erreurs confondues
        cause_by_node = defaultdict(Counter)
        fatal_count = 0
        other_samples = []
        for key, state in runs[model].items():
            if "fatal_error" in state:    # echec definitif ecrit par run.py apres MAX_RETRIES
                fatal_count += 1
                cause = _classify_error(str(state["fatal_error"]))
                cause_counts[f"fatal::{cause}"] += 1
                continue
            for err in state.get("errors", []):
                node_name = err.split(":", 1)[0].strip()
                cause = _classify_error(err)
                cause_counts[cause] += 1
                cause_by_node[node_name][cause] += 1
                if cause == "other" and len(other_samples) < 5:
                    other_samples.append((key, err[:200]))
 
        total = sum(cause_counts.values())
        print(f"\n  [{model}] {total} messages d'erreur classes, "
              f"{fatal_count} cas en echec definitif (fatal_error)")
        for cause, n in cause_counts.most_common():
            print(f"    {cause:25s} {n:5d}  ({100 * n / total:.1f}%)" if total else "")
        # Repartition par node uniquement pour les causes non-cascade -- c'est la ou la cause
        # RACINE se lit (les nodes aval ne font que propager la cle manquante).
        print(f"    -- par node (hors cascade) --")
        for node_name, counter in sorted(cause_by_node.items()):
            non_cascade = {c: n for c, n in counter.items() if c != "cascade_missing_key"}
            if non_cascade:
                print(f"    {node_name:20s} {dict(non_cascade)}")
        if other_samples:
            print(f"    -- echantillon des messages 'other' (a re-patterner si frequents) --")
            for key, msg in other_samples:
                print(f"      {key}: {msg}")
 
 
# --- 26. Correlation Zenodo recalculee sur le sous-ensemble BPMN structurellement propre ---
#
# Ferme le point ouvert 4 de dataset_run_observations.md section 20 : la note Zenodo inclut un
# axe "well-formed" que le pipeline ne mesure pas. Ici on retire les BPMN structurellement
# suspects (boucle perdue par PM4Py, bug add_in_arc) et on recalcule Spearman -- si rho reste
# ~0 sur ce sous-ensemble, l'explication "axe well-formed" perd de sa force ; s'il monte, elle
# se renforce. Dans les deux cas, la conclusion Zenodo du papier devient defendable.
 
def _load_loop_files() -> set:
    audit_path = os.path.join(NOTES_DIR, "bpmn_silent_loss_audit.json")
    if not os.path.exists(audit_path):
        return set()
    with open(audit_path) as f:
        audit = json.load(f)
    return {
        (desc, fname) for desc, files_ in audit.items()
        for fname, r in files_.items() if r.get("loop_count")
    }
 
 
def _find_pm4py_bug_keys(runs) -> set:
    """Cas dont bpmn_to_spo a echoue sur add_in_arc/add_out_arc -- identiques pour les trois
    modeles (verifie en section 2 des observations), donc l'union suffit."""
    keys = set()
    for model in MODELS:
        for key, state in runs[model].items():
            for err in state.get("errors", []):
                if "add_in_arc" in err or "add_out_arc" in err:
                    keys.add(key)
    return keys
 
 
def analysis_zenodo_clean_bpmn_subset(runs, quality_scores):
    section("26. CORRELATION ZENODO SUR SOUS-ENSEMBLE BPMN STRUCTURELLEMENT PROPRE")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy indisponible -- section sautee")
        return
 
    loop_files = _load_loop_files()
    pm4py_bug = _find_pm4py_bug_keys(runs)
    if not loop_files:
        print("  ATTENTION: bpmn_silent_loss_audit.json introuvable -- filtre boucle inactif")
    print(f"  Exclusions structurelles : {len(loop_files)} fichiers avec boucle perdue, "
          f"{len(pm4py_bug)} fichiers bug PM4Py add_arc")
 
    for model in MODELS:
        ratios, quality_vals = [], []
        for key, state in runs[model].items():
            if key in loop_files or key in pm4py_bug:
                continue
            counts = alignment_counts(state)
            if counts is None or key not in quality_scores:
                continue
            conforme, ecarts, non_verif = counts
            total = conforme + ecarts + non_verif
            if total == 0:
                continue
            ratios.append(conforme / total)
            quality_vals.append(quality_scores[key])
        if len(ratios) < 3:
            print(f"  [{model}] pas assez de donnees ({len(ratios)} cas)")
            continue
        rho, pval = spearmanr(ratios, quality_vals)
        print(f"  [{model}] n={len(ratios)}  Spearman rho={rho:.3f}  p={pval:.4f}  "
              f"(comparer a la section 1 : rho global inchange = l'axe well-formed "
              f"n'explique pas le rho~0)")
 
 
# --- 27. Controle du facteur de confusion complexite/boucle perdue (section 11 vs 10) ---
#
# Ferme le point ouvert 5 : le resultat contre-intuitif (boucle perdue -> SATISFIED plus eleve)
# n'a jamais ete controle par la taille du BPMN. Stratification par terciles du nombre d'aretes
# DFG : si l'ecart avec/sans boucle disparait a taille comparable, c'etait la confusion
# complexite ; s'il persiste dans chaque strate, l'effet est reel et a expliquer.
 
def analysis_loop_complexity_confounder(runs):
    section("27. BOUCLE PERDUE vs SATISFIED, STRATIFIE PAR TAILLE DE DFG (controle de confusion)")
    loop_files = _load_loop_files()
    if not loop_files:
        print("  bpmn_silent_loss_audit.json introuvable -- section sautee")
        return
 
    for model in MODELS:
        rows = []  # (n_edges, satisfied_ratio, has_loop)
        for key, state in runs[model].items():
            dfg = state.get("dfg_edges")
            counts = alignment_counts(state)
            if dfg is None or counts is None:
                continue
            s, v, u = counts
            total = s + v + u
            if total == 0:
                continue
            rows.append((len(dfg), s / total, key in loop_files))
        if len(rows) < 9:
            print(f"  [{model}] pas assez de donnees ({len(rows)} cas)")
            continue
 
        sizes = sorted(r[0] for r in rows)
        t1 = sizes[len(sizes) // 3]
        t2 = sizes[2 * len(sizes) // 3]
 
        def stratum(n_edges):
            return "petit" if n_edges <= t1 else ("moyen" if n_edges <= t2 else "grand")
 
        print(f"\n  [{model}] terciles de taille DFG : petit<= {t1}  moyen<= {t2}  grand> {t2}")
        for name in ("petit", "moyen", "grand"):
            with_loop = [r[1] for r in rows if stratum(r[0]) == name and r[2]]
            without = [r[1] for r in rows if stratum(r[0]) == name and not r[2]]
            wl = f"{sum(with_loop) / len(with_loop):.2f} (n={len(with_loop)})" if with_loop else "n=0"
            wo = f"{sum(without) / len(without):.2f} (n={len(without)})" if without else "n=0"
            print(f"    strate {name:6s} : SATISFIED moyen avec boucle = {wl:18s} sans = {wo}")
        print(f"    -> si l'ecart avec/sans s'annule strate par strate, l'effet de la section 11 "
              f"etait la confusion taille/complexite, pas la perte de boucle elle-meme")
 
 
# --- 28. Dump complet des cas VIOLATED (prerequis materiel de l'inspection qualitative) ---
#
# Ferme le point ouvert de la section 4 : seule la LISTE des cas VIOLATED existait, jamais leur
# contenu. Ecrit un markdown lisible avec tout le contexte necessaire pour juger a la main si
# chaque VIOLATED est un vrai desordre du BPMN ou un artefact de matching.
 
def _reverse_best_labels(matches: dict) -> dict:
    """state -> (label activite, score) du meilleur match -- meme logique que
    report_node.label_for_state, recopiee pour garder ce script autonome."""
    reverse = {}
    for label, m in matches.items():
        state = m["match"]
        if state not in reverse or m["score"] > reverse[state][1]:
            reverse[state] = (label, m["score"])
    return reverse
 
 
def dump_violated_cases(runs):
    section("28. DUMP DES CAS VIOLATED -> results/notes/violated_cases_dump.md")
    os.makedirs(NOTES_DIR, exist_ok=True)
    out_path = os.path.join(NOTES_DIR, "violated_cases_dump.md")
    n_total = 0
    with open(out_path, "w") as f:
        f.write("# Cas VIOLATED -- dump complet pour inspection qualitative\n\n")
        f.write("Pour chaque terme VIOLATED : la cible, le terme, la citation qui a justifie la "
                "precondition, les labels d'activite BPMN matches (avec score) des deux cotes, "
                "et les aretes du graphe process touchant ces etats. Objectif : trancher "
                "'vrai desordre' vs 'artefact de matching' cas par cas.\n\n")
        for model in MODELS:
            f.write(f"\n## Modele : {model}\n\n")
            for key, state in sorted(runs[model].items()):
                alignment = state.get("alignment")
                matches = state.get("matches")
                if not alignment or not matches:
                    continue
                reverse = _reverse_best_labels(matches)
                process_graph = state.get("process_graph") or []
                for target, entry in alignment.items():
                    for t in entry.get("terms", []):
                        if t["status"] != "VIOLATED":
                            continue
                        n_total += 1
                        term = t["term"]
                        t_lab = reverse.get(term, ("(aucun label)", None))
                        g_lab = reverse.get(target, ("(aucun label)", None))
                        f.write(f"### {key[0]}/{key[1]}\n")
                        f.write(f"- cible   : `{target}`  <- activite matchee : "
                                f"\"{g_lab[0]}\" (score={g_lab[1]})\n")
                        f.write(f"- terme   : `{term}`  <- activite matchee : "
                                f"\"{t_lab[0]}\" (score={t_lab[1]})\n")
                        f.write(f"- operateur : {entry.get('operator')}  |  citation : "
                                f"\"{entry.get('quote')}\"\n")
                        edges_touch = [
                            f"{e['from']} -> {e['to']} "
                            f"(\"{e.get('activity_from')}\" -> \"{e.get('activity_to')}\")"
                            for e in process_graph
                            if e.get("from") in (term, target) or e.get("to") in (term, target)
                        ]
                        f.write(f"- aretes process touchant ces etats ({len(edges_touch)}) :\n")
                        for line in edges_touch[:12]:
                            f.write(f"    - {line}\n")
                        if len(edges_touch) > 12:
                            f.write(f"    - ... (+{len(edges_touch) - 12})\n")
                        f.write("\n")
    print(f"  {n_total} termes VIOLATED dumpes -> {out_path}")
 
 
# --- 29. Echantillon d'annotation zone 0.35-0.49 -> CSV (point 10 des observations) ---
#
# Transforme le point le mieux borne de la liste ouverte en tache directe : 30 conclusions
# tirees aleatoirement (seed fixe, reproductible) dans la zone entre le seuil actuel et le
# seuil GMM, exportees avec colonnes d'annotation vides. L'annotation manuelle de ce fichier
# donne le taux de precision reel du signal de confiance -- LA donnee qui decide si
# l'architecture du matching doit changer.
 
ANNOTATION_ZONE = (0.35, 0.49)
ANNOTATION_SAMPLE_SIZE = 30
 
 
def dump_annotation_sample(runs):
    section(f"29. ECHANTILLON D'ANNOTATION ZONE {ANNOTATION_ZONE[0]}-{ANNOTATION_ZONE[1]} "
            f"-> results/notes/annotation_sample.csv")
    lo, hi = ANNOTATION_ZONE
    pool = []
    for model in MODELS:
        for key, state in runs[model].items():
            alignment = state.get("alignment")
            matches = state.get("matches")
            if not alignment or not matches:
                continue
            best = {}
            for label, m in matches.items():
                st = m["match"]
                if st not in best or m["score"] > best[st][1]:
                    best[st] = (label, m["score"])
            for target, entry in alignment.items():
                for t in entry.get("terms", []):
                    if t["status"] not in ("SATISFIED", "VIOLATED"):
                        continue
                    term = t["term"]
                    ts = best.get(term, (None, None))[1]
                    gs = best.get(target, (None, None))[1]
                    present = [s for s in (ts, gs) if s is not None]
                    if not present:
                        continue
                    worst = min(present)
                    if lo <= worst < hi:
                        pool.append({
                            "model": model,
                            "case": f"{key[0]}/{key[1]}",
                            "target": target,
                            "target_activity": best.get(target, ("", None))[0],
                            "target_score": gs,
                            "term": term,
                            "term_activity": best.get(term, ("", None))[0],
                            "term_score": ts,
                            "status": t["status"],
                            "quote": entry.get("quote") or "",
                        })
 
    rng = random.Random(0)  # seed fixe : echantillon reproductible d'un lancement a l'autre
    sample = pool if len(pool) <= ANNOTATION_SAMPLE_SIZE else rng.sample(pool, ANNOTATION_SAMPLE_SIZE)
 
    os.makedirs(NOTES_DIR, exist_ok=True)
    out_path = os.path.join(NOTES_DIR, "annotation_sample.csv")
    fields = ["model", "case", "target", "target_activity", "target_score",
              "term", "term_activity", "term_score", "status", "quote",
              "annotation_match_correct", "annotation_commentaire"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            row["annotation_match_correct"] = ""   # a remplir : oui / non
            row["annotation_commentaire"] = ""
            writer.writerow(row)
    print(f"  {len(pool)} conclusions dans la zone, {len(sample)} echantillonnees (seed=0) "
          f"-> {out_path}")
    print(f"  Colonnes a remplir a la main : annotation_match_correct (oui/non), "
          f"annotation_commentaire")


def analysis_quote_verbatim_rate(runs):
    section("30. TAUX DE FIDELITE VERBATIM DES CITATIONS (par modele-redacteur)")
    for model in MODELS:
        counts = Counter()  # True / False / None (None = pas de citation a verifier)
        violations_sample = []
        for key, state in runs[model].items():
            report = state.get("report")
            if not report:
                continue
            for item in report.get("ecarts", []) + report.get("non_verifiable", []):
                v = item.get("quote_verbatim")
                counts[v] += 1
                if v is False and len(violations_sample) < 5:
                    violations_sample.append((key, item.get("root_cause"), item.get("quote"),
                                               item.get("explanation")))
 
        checkable = counts[True] + counts[False]
        if checkable == 0:
            print(f"  [{model}] aucune entree avec citation verifiable "
                  f"(0 rapport produit, ou aucun ecart/non-verifiable, ou champ absent -- "
                  f"redemarrer le run apres le patch report_node.py si le champ manque)")
            continue
 
        rate = 100 * counts[True] / checkable
        print(f"  [{model}] {counts[True]}/{checkable} citations verbatim ({rate:.1f}%)  "
              f"-- {counts[False]} violation(s), {counts[None]} entree(s) sans citation "
              f"a verifier (exclues du taux)")
        if violations_sample:
            print(f"    -- echantillon des violations (citation non reproduite mot pour mot) --")
            for key, root, quote, expl in violations_sample:
                print(f"    {key}: cause={root!r}")
                print(f"      citation attendue : {quote!r}")
                print(f"      explication generee : {(expl or '')[:160]}...")
 

def main():
    print("Chargement des runs...")
    runs = load_all_runs()
    for model in MODELS:
        print(f"  [{model}] {len(runs[model])} fichiers charges")

    print("Chargement des notes Zenodo...")
    quality_scores = load_quality_scores()
    print(f"  {len(quality_scores)} notes chargees")

    analysis_zenodo_correlation(runs, quality_scores)
    analysis_violated_frequency(runs)
    analysis_never_matched_states(runs)
    analysis_malformed_keys(runs)
    analysis_internal_errors(runs)
    analysis_downgrade_reasons(runs)
    analysis_cycle_frequency(runs)
    analysis_reproducibility_note()
    analysis_model_comparison(runs)
    analysis_inter_model_agreement(runs)
    analysis_u_richness(runs)
    analysis_bpmn_complexity(runs)
    analysis_loop_loss_impact(runs)
    analysis_per_description_variance(runs)
    analysis_report_compression(runs)
    analysis_missing_citations(runs)
    analysis_error_breakdown(runs)
    analysis_clean_subset(runs, quality_scores)
    analysis_case_level_agreement(runs, quality_scores)
    analysis_low_confidence_conclusions(runs)
    analysis_confidence_vs_agreement(runs)
    analysis_matching_score_bimodality(runs)
    analysis_low_confidence_gmm_threshold(runs)
    analysis_error_cause_quantification(runs)
    analysis_zenodo_clean_bpmn_subset(runs, quality_scores)
    analysis_loop_complexity_confounder(runs)
    dump_violated_cases(runs)
    dump_annotation_sample(runs)
    analysis_quote_verbatim_rate(runs)

    section("TERMINE")


if __name__ == "__main__":
    main()