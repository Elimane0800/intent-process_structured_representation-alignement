"""Rapport final : traduit la sortie de check_alignment() en langage metier. Un appel LLM par
CAUSE RACINE distincte (pas par item non satisfait) -- plusieurs echecs qui partagent la meme
cause (ex. un etat jamais couvert par le process, dont dependent plusieurs autres verifications
en cascade) ne produisent qu'une seule explication, avec la liste des cibles affectees ajoutee
par du code, jamais par le LLM. Assemblage en trois sections par du code deterministe.

La cause racine est trouvee en remontant recursivement dans le graphe de Pre lui-meme
(alignment) -- pas une heuristique textuelle : si le terme fautif d'un echec est lui-meme une
cible verifiee ailleurs (donc lui-meme en echec), la vraie cause est encore plus en amont, et
ainsi de suite jusqu'a un terme qui n'est plus lui-meme une cible controlee (une vraie feuille),
ou jusqu'a un point ou plusieurs causes distinctes convergent (auquel cas on s'arrete plutot que
de fusionner a tort deux problemes independants).

Le detail brut de l'alignement est conserve en piece jointe, jamais remplace par la prose.

Ordre des sections fixe, jamais reordonne par le LLM : ecarts (VIOLATED) d'abord, puis
non-verifiable (UNRESOLVABLE) dans une section separee, puis conforme (SATISFIED) en dernier,
groupe et bref -- jamais developpe phrase par phrase comme les deux premieres."""

import json
import os
import re

from agent.models.base_llm import get_llm
from agent.prompt.report_prompt import (
    REPORT_PROMPT_VIOLATED,
    REPORT_PROMPT_UNRESOLVABLE,
    REPORT_PROMPT_VIOLATED_NEGATED,
    REPORT_PROMPT_UNRESOLVABLE_NEGATED,
)

ALIGNMENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "alignment")
MATCHING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_matching")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "report")

_MISSING_PLACEHOLDER = "(no clear process activity corresponds to this)"
_NO_QUOTE_PLACEHOLDER = "(no quote available)"


def _is_negated(term: str) -> bool:
    return term.startswith("NOT ")


def _strip_not(term: str) -> str:
    """Copie locale du meme helper que precondition_node_with_retry.py / graph_node.py /
    tgms_solver.py -- meme discipline de duplication assumee dans tout ce projet. Necessaire
    ICI specifiquement parce qu'un terme de precondition niee ('NOT job.continued') traverse
    tgms_solver.py jusqu'a alignment_node.py SANS jamais etre reintegre a un vrai nom d'etat --
    'term' dans une entree TermVerdict porte le prefixe "NOT " tel quel (par design, cf.
    status_of_negated_term), et ce fichier est le premier maillon de la chaine qui produit du
    texte destine a un lecteur humain plutot que des donnees structurees pour un autre node.
    Sans ce strip systematique, "NOT " fuit litteralement dans le rapport metier -- exactement
    le jargon interne que _REPORT_RULES interdit explicitement."""
    return term[len("NOT "):] if _is_negated(term) else term


def reverse_labels(matches: dict) -> dict:
    """entity.state -> liste (label, score) des activites BPMN qui y ont ete matchees (peut
    etre vide). Garde le score pour permettre de choisir le candidat le plus confiant quand
    plusieurs activites matchent le meme etat, plutot que de les concatener dans le rapport.

    CORRECTIF (bug bloquant, meme categorie que celui deja trouve et corrige dans
    tgms_from_pipeline) : state_matching_node.py ne renvoie plus, par activite, un match unique
    a plat ({"match": ..., "score": ...}) -- il renvoie desormais {"activity_label": ...,
    "matches": [{"match", "score"}, ...], "clusters": [...]} (top-k relatif + clustering
    additif). L'ancien code (`m["match"]` directement sur une entree de matches.values())
    levait un KeyError des la premiere execution avec ce nouveau format -- corrige en iterant
    explicitement sur la liste "matches" de chaque activite, meme structure que
    tgms_from_pipeline."""
    reverse: dict = {}
    for activity_id, entry in matches.items():
        activity_label = entry.get("activity_label", activity_id)
        for m in entry.get("matches", []):
            reverse.setdefault(m["match"], []).append((activity_label, m["score"]))
    return reverse


def label_for_state(state: str, reverse: dict) -> str:
    """Un seul label, jamais une concatenation -- plusieurs activites matchees au meme etat
    donnent un texte illisible pour un lecteur metier. Garde le score le plus eleve ; les
    autres candidats restent visibles dans la piece jointe brute, pas dans la prose."""
    candidates = reverse.get(state)
    if not candidates:
        return _MISSING_PLACEHOLDER
    best_label, _ = max(candidates, key=lambda c: c[1])
    return best_label


def display_label_for_state(state: str, reverse: dict) -> str:
    """label_for_state() avec repli automatique vers _readable_state() si aucun label BPMN
    n'existe -- jamais le placeholder brut lui-meme.

    Bug trouve a l'echelle (v2, section 30 quote_verbatim -- 8 violations sur 3 modeles,
    toutes de cette forme, jamais un cas de traduction) : le garde 'si placeholder, replier
    sur _readable_state()' existait deja pour missing_description dans
    generate_cluster_explanation, mais AUCUN AUTRE site n'en beneficiait -- target_label dans
    les deux prompts (VIOLATED et UNRESOLVABLE), le champ root_cause, la liste
    affected_activities et le texte 'Cette meme cause affecte aussi' utilisaient tous
    label_for_state() nu. Resultat observe litteralement dans la prose destinee a
    l'utilisateur : "avant que l'activite \"(aucune activite du processus ne correspond
    clairement a cela)\" puisse se produire" -- le placeholder technique traite comme un nom
    d'activite reel, entre guillemets, par le LLM redacteur qui n'a aucune raison de le
    reconnaitre comme special.

    Fix : un seul point de garde, applique systematiquement (plus fiable qu'un garde
    duplique a chaque site d'appel, source exacte de l'omission initiale)."""
    label = label_for_state(state, reverse)
    return label if label != _MISSING_PLACEHOLDER else _readable_state(state)


def _readable_state(state: str) -> str:
    """Repli quand aucun label d'activite n'existe pour decrire ce que le texte exigeait --
    rendu lisible a partir de entity.state lui-meme (mieux que rien, mais moins fiable qu'un
    vrai label BPMN)."""
    entity, _, s = state.partition(".")
    return f"{entity.replace('_', ' ')} {s.replace('_', ' ')}"


def quote_reproduced_verbatim(explanation: str, quote: str) -> bool | None:
    """Controle mecanique de la regle 'citation reproduite mot pour mot, jamais traduite' --
    le controle apres coup propose dans report_observations.md section 3.3, jamais implemente
    jusqu'ici (la regle de prompt seule etait efficace a 5/6 cas observes, sans mesure
    possible a l'echelle). Deterministe : la citation, normalisee en espacement, doit etre un
    substring de l'explication normalisee. Une traduction ou une paraphrase entre guillemets
    echoue necessairement ce test. Ne bloque jamais -- flag mesure, pas rejet, coherent avec
    la discipline du pipeline (le rapport part quand meme, le taux par redacteur devient une
    metrique). None si aucune citation n'etait disponible (rien a verifier)."""
    if not quote or quote == _NO_QUOTE_PLACEHOLDER:
        return None
    normalize = lambda s: " ".join(s.split())
    return normalize(quote) in normalize(explanation)


def find_root_term(term: str, alignment: dict, seen: set | None = None) -> str:
    """Remonte a la vraie cause : si `term` est lui-meme une cible controlee ailleurs (donc
    lui-meme en echec), la cause reelle est encore plus en amont. S'arrete des que : (a) `term`
    n'est plus lui-meme une cible controlee (feuille), ou (b) plusieurs causes distinctes
    convergent (ne jamais fusionner deux problemes independants sous une seule cause).

    Ne suit un sous-terme que s'il est LUI-MEME reellement manquant (present dans son propre
    champ 'missing'), jamais par defaut -- bug reel corrige ici : un terme peut avoir le statut
    UNRESOLVABLE alors que CE terme est bel et bien matche dans le process, simplement parce que
    la CIBLE de son propre contrde (la cle `term` elle-meme, une entree du dict alignment) ne
    l'est pas. Suivre ce sous-terme comme s'il etait la cause menait a accuser la mauvaise
    activite (observe concretement : 'job_offer.sent' introuvable, la cause remontait a tort
    vers 'job_application.received', qui etait pourtant bien matche).

    CORRECTIF (bug de traversee, pas seulement d'affichage) : `term` est desormais
    systematiquement deprefixe de "NOT " des l'entree, ET dans sa propre boucle de recursion
    avant toute comparaison. Sans ce strip, deux problemes distincts se produisaient pour un
    terme nie ('NOT review.visible') : (1) `alignment.get("NOT review.visible")` ne trouve
    jamais de cle (les cibles du dict alignment ne sont jamais prefixees), donc `entry` est
    toujours None et la fonction retourne prematurement le terme encore prefixe comme "racine",
    sans jamais chercher plus loin ; (2) plus insidieux, `t["term"] in missing` comparait un
    terme prefixe ("NOT review.visible") a une liste `missing` qui contient toujours des noms
    d'etat propres (cf. status_of_negated_term), donc cette comparaison echouait TOUJOURS pour
    un sous-terme nie meme quand il etait reellement la bonne piste a suivre -- pas juste une
    fuite d'affichage, une vraie fausse negative dans la logique de remontee."""
    term = _strip_not(term)
    seen = seen if seen is not None else set()
    if term in seen:
        return term  # garde-fou anti-cycle, ne devrait pas arriver sur un G bien forme
    seen = seen | {term}

    entry = alignment.get(term)
    if entry is None or entry["status"] == "SATISFIED":
        return term

    failing_subterms = []
    for t in entry["terms"]:
        if t["status"] == "SATISFIED":
            continue
        clean_subterm = _strip_not(t["term"])
        missing = t.get("missing", [])
        # Un sous-terme n'est une piste a remonter que s'il est LUI-MEME dans sa propre liste
        # missing -- sinon c'est la cible de son controle (donc `term` lui-meme) qui est en
        # cause, pas ce sous-terme, et le suivre serait une fausse piste. Comparaison sur la
        # forme deprefixee des deux cotes (missing l'est deja par construction, t["term"] ne
        # l'etait pas -- cf. note ci-dessus).
        if not missing or clean_subterm in missing:
            failing_subterms.append(clean_subterm)

    if not failing_subterms:
        return term

    roots = {find_root_term(t, alignment, seen) for t in failing_subterms}
    return next(iter(roots)) if len(roots) == 1 else term


def status_for_root(root: str, failures: list[dict], alignment: dict) -> str:
    for f in failures:
        if f["term"] == root:
            return f["status"]
    if root in alignment:
        return alignment[root]["status"]
    return "UNRESOLVABLE"  # repli conservateur, ne devrait normalement pas etre atteint


def collect_failures(alignment: dict) -> list[dict]:
    """Une entree par element reellement manquant (pas une entree par terme non satisfait --
    quand la cible elle-meme est ce qui manque, pas le terme, c'est elle qu'il faut tracer).

    Porte desormais un flag "negated" par entree -- ne PAS le confondre avec un simple
    probleme d'etiquette : une precondition niee ("NOT X") n'a pas seulement besoin d'un label
    different plus loin dans le pipeline, elle a besoin d'une PHRASE differente
    (REPORT_PROMPT_VIOLATED_NEGATED / REPORT_PROMPT_UNRESOLVABLE_NEGATED), parce que "X doit
    avoir eu lieu avant Y" et "X ne doit PAS avoir eu lieu avant Y" ne sont pas la meme
    affirmation. "term"/"root" sont toujours deprefixes ici -- plus jamais de "NOT " dans les
    donnees qui circulent en aval vers l'affichage (cf. _strip_not, find_root_term)."""
    failures = []
    for target, entry in alignment.items():
        for term_result in entry["terms"]:
            if term_result["status"] == "SATISFIED":
                continue
            raw_term = term_result["term"]
            negated = _is_negated(raw_term)
            clean_term = _strip_not(raw_term)
            missing = term_result.get("missing", [])
            # VIOLATED n'a pas de 'missing' (les deux etats existent, juste pas de chemin) --
            # dans ce cas on retombe sur le terme lui-meme (deja deprefixe), comportement
            # inchange pour le cas positif ; missing est deja deprefixe par construction pour
            # le cas negatif (cf. status_of_negated_term dans tgms_solver.py).
            elements = missing if missing else [clean_term]
            for element in elements:
                failures.append({
                    "target": target,
                    "term": clean_term,
                    "negated": negated,
                    "status": term_result["status"],
                    "root": find_root_term(element, alignment),
                    "quote": entry.get("quote"),
                })
    return failures


def cluster_by_root(failures: list[dict]) -> dict:
    clusters = {}
    for f in failures:
        clusters.setdefault(f["root"], []).append(f)
    return clusters


def generate_cluster_explanation(
    root: str, representative_target: str, status: str, quote: str, negated: bool, reverse: dict, llm
) -> str:
    """Choisit le gabarit NEGATED plutot que standard des que la cause racine correspond a une
    exigence d'ABSENCE ("NOT X") plutot que de presence -- jamais un simple choix d'etiquette,
    cf. docstring de collect_failures. Utiliser le mauvais gabarit produirait une explication
    grammaticalement correcte mais affirmant l'inverse exact de ce que le texte source exige."""
    root_label = display_label_for_state(root, reverse)
    target_label = display_label_for_state(representative_target, reverse)
    if status == "VIOLATED":
        template = REPORT_PROMPT_VIOLATED_NEGATED if negated else REPORT_PROMPT_VIOLATED
        prompt = template.format(target_label=target_label, term_label=root_label, quote=quote)
    else:
        template = REPORT_PROMPT_UNRESOLVABLE_NEGATED if negated else REPORT_PROMPT_UNRESOLVABLE
        prompt = template.format(
            target_label=target_label, missing_description=root_label, quote=quote
        )
    return llm.invoke(prompt).content.strip()


def compute_verdict(summary: dict) -> dict:
    """Verdict calcule par du code deterministe, jamais demande au LLM -- coherent avec tout ce
    projet (le LLM-as-judge a ete identifie tot comme non fiable pour ce role). Trois valeurs,
    jamais reduites a un score continu ni a une moyenne ponderee -- c'est precisement le defaut
    reproche a la grille Mangler des le debut de ce projet : un score 0-5 agregeant plusieurs
    criteres (couverture, parallelisme, bonne formation) sans ponderation documentee, donc
    invisible et non falsifiable. Ici, un seul ecart suffit a faire basculer le verdict --
    aucun seuil de tolerance ni poids de gravite invente, meme discipline que
    CONFIDENCE_THRESHOLD/CLUSTER_SIMILARITY_THRESHOLD (calibres empiriquement, jamais poses
    comme des poids de gravite a priori) :

      - FIDELE : aucun ecart, aucun cas non verifiable -- chaque precondition non triviale du
        texte a ete confirmee dans le processus observe.
      - FIDELE_SOUS_RESERVE : aucun ecart, mais au moins une precondition n'a pu etre rattachee
        a aucune activite du processus -- ni confirmee, ni infirmee. Ne jamais confondre avec
        une confirmation de fidelite complete (meme prudence que le statut UNRESOLVABLE
        lui-meme, jamais force vers SATISFIED ou VIOLATED en amont).
      - NON_FIDELE : au moins un ecart -- un seul suffit.

    Le champ "perimetre" rappelle explicitement, dans le rapport lui-meme, une limite posee des
    le debut de ce projet par inspection manuelle du corpus (cf. l'analyse du modele Camunda 42) :
    cette fidelite porte sur l'atteignabilite causale des exigences du texte, pas sur la qualite
    generale du modele -- un defaut de multiplicite (une activite qui s'execute deux fois a
    cause d'un AND-split/XOR-join mal apparie) reste hors champ, meme si le processus reste par
    ailleurs bien forme.

    Valeurs de sortie (verdict/justification/perimetre) en anglais, comme le reste du rapport
    depuis le changement de langue de report_prompt.py -- seule cette docstring developpeur
    reste en francais, meme convention que partout ailleurs dans ce projet."""
    ecarts = summary["ecarts"]
    non_verifiable = summary["non_verifiable"]

    if ecarts > 0:
        verdict = "NOT_FAITHFUL"
        justification = (
            f"{ecarts} gap(s) found between the source text's requirements and the observed "
            f"process -- at least one precondition asserted by the text is contradicted by "
            f"the model's actual sequence of activities."
        )
    elif non_verifiable > 0:
        verdict = "FAITHFUL_WITH_RESERVATIONS"
        justification = (
            f"No gap found, but {non_verifiable} requirement(s) from the text could not be "
            f"associated with any clear activity in the observed process -- neither confirmed "
            f"nor refuted. This verdict is NOT a confirmation of full faithfulness."
        )
    else:
        verdict = "FAITHFUL"
        justification = (
            "Every non-trivial precondition asserted by the source text was confirmed in the "
            "observed process's sequence of activities."
        )

    return {
        "verdict": verdict,
        "justification": justification,
        "scope": (
            "This verdict covers the causal reachability of the requirements expressed in the "
            "source text, not the general quality of the process model -- a multiplicity "
            "defect (an activity executed more than once due to a mismatched branching "
            "structure) is not covered by this verdict, even if the process is otherwise "
            "well-formed."
        ),
    }


def build_report(alignment: dict, matches: dict, llm) -> dict:
    reverse = reverse_labels(matches)
    failures = collect_failures(alignment)
    clusters = cluster_by_root(failures)

    violated, unresolvable, satisfied = [], [], []
    for root, items in clusters.items():
        status = status_for_root(root, failures, alignment)
        quote = next((i["quote"] for i in items if i["root"] == root and i["quote"]), None) or _NO_QUOTE_PLACEHOLDER
        if quote == _NO_QUOTE_PLACEHOLDER:
            print(f"    [report] WARNING: no quote available for root cause {root!r} -- "
                  f"graph_construction may not have been regenerated with the quote fix")
        # Meme idiome que la selection de `quote` juste au-dessus : un representant parmi les
        # items de ce cluster, pas un nouveau calcul separe. Determine si l'EXPLICATION de ce
        # cluster doit utiliser le gabarit NEGATED (exigence d'absence) ou standard.
        negated = next((i["negated"] for i in items if i["root"] == root), False)

        representative_target = items[0]["target"]
        explanation = generate_cluster_explanation(
            root, representative_target, status, quote, negated, reverse, llm
        )
        affected = sorted({display_label_for_state(i["target"], reverse) for i in items})
        if len(affected) > 1:
            # Ajoute par du code, jamais par le LLM -- la liste des cibles affectees ne doit
            # jamais dependre d'une reformulation libre.
            others = [a for a in affected if a != display_label_for_state(representative_target, reverse)]
            if others:
                explanation += f" (This same root cause also affects: {', '.join(others)}.)"

        entry = {
            "root_cause": display_label_for_state(root, reverse),
            "quote": quote,
            "explanation": explanation,
            "quote_verbatim": quote_reproduced_verbatim(explanation, quote),
            "affected_activities": affected,
            "affected_count": len(items),
        }
        if entry["quote_verbatim"] is False:
            print(f"    [report] WARNING: quote not reproduced verbatim in explanation for "
                  f"root cause {root!r} -- likely translated or paraphrased (rule violation, "
                  f"flagged, not blocked)")
        (violated if status == "VIOLATED" else unresolvable).append(entry)

    for target, entry in alignment.items():
        for term_result in entry["terms"]:
            if term_result["status"] == "SATISFIED":
                # Un terme NIE peut tout a fait etre SATISFIED (cf. tgms_solver.py) -- meme
                # discipline que collect_failures : deprefixer avant tout lookup de label, et
                # marquer l'absence par du code (jamais par le LLM, cette section reste "groupee
                # et breve", sans generation de prose) plutot que de laisser fuir "NOT" tel quel.
                raw_term = term_result["term"]
                term_label = display_label_for_state(_strip_not(raw_term), reverse)
                if _is_negated(raw_term):
                    term_label = f"absence of: {term_label}"
                satisfied.append({
                    "target": display_label_for_state(target, reverse),
                    "term": term_label,
                })

    summary = {
        "ecarts": len(violated),
        "non_verifiable": len(unresolvable),
        "conforme": len(satisfied),
    }
    return {
        "verdict": compute_verdict(summary),
        "summary": summary,
        "ecarts": violated,
        "non_verifiable": unresolvable,
        "conforme": satisfied,
        "piece_jointe_alignement_brut": alignment,
    }


if __name__ == "__main__":
    existing_alignment_runs = [f for f in os.listdir(ALIGNMENT_DIR) if re.match(r"run_\d+\.json$", f)]
    if not existing_alignment_runs:
        raise SystemExit(f"[report] no alignment run found in {ALIGNMENT_DIR}")
    latest_alignment_run = max(existing_alignment_runs, key=lambda f: int(re.match(r"run_(\d+)\.json$", f).group(1)))
    print(f"Loading alignment results from {latest_alignment_run}")
    with open(os.path.join(ALIGNMENT_DIR, latest_alignment_run)) as f:
        alignment_data = json.load(f)

    existing_matching_runs = [f for f in os.listdir(MATCHING_DIR) if re.match(r"run_\d+\.json$", f)]
    if not existing_matching_runs:
        raise SystemExit(f"[report] no state_matching run found in {MATCHING_DIR}")
    latest_matching_run = max(existing_matching_runs, key=lambda f: int(re.match(r"run_(\d+)\.json$", f).group(1)))
    print(f"Loading state_matching results from {latest_matching_run}")
    with open(os.path.join(MATCHING_DIR, latest_matching_run)) as f:
        matching_data = json.load(f)

    REPORT_MODEL = "gpt-5.5"  # explicite : herite du defaut de base_llm.py sinon,
                                            # sans rapport avec le modele "sous test" aligne.
    print(f"Report writer model: {REPORT_MODEL}")
    llm = get_llm(temperature=0, model=REPORT_MODEL)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for prompt_name, per_spo in alignment_data.items():
        results[prompt_name] = {}
        for spo_filename, alignment in per_spo.items():
            matches = matching_data[prompt_name][spo_filename]["matches"]
            report = build_report(alignment, matches, llm)
            results[prompt_name][spo_filename] = report

            print(f"\n########## {spo_filename} [{prompt_name}] ##########")
            print(f"VERDICT: {report['verdict']['verdict']} -- {report['verdict']['justification']}")
            print(f"Ecarts: {report['summary']['ecarts']}  "
                  f"Non verifiable: {report['summary']['non_verifiable']}  "
                  f"Conforme: {report['summary']['conforme']}")
            for item in report["ecarts"]:
                print(f"  [ECART] ({item['affected_count']} etape(s) affectee(s)) {item['explanation']}")
            for item in report["non_verifiable"]:
                print(f"  [NON VERIFIABLE] ({item['affected_count']} etape(s) affectee(s)) {item['explanation']}")

    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")