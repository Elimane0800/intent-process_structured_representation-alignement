"""Run sur le dataset LRE (van der Aa et al., "Comparing textual descriptions to process
models" -- alignment_model_text) : associe chaque texte source a SON modele BPMN (appariement
1:1 par nom de fichier de base, propriete du dataset -- contrairement au corpus Mangler ou un
texte a 8-11 modeles BPMN differents). Invoque le pipeline complet une fois par (modele LLM x
sous-ensemble x cas).

Toutes les ground truths de ce dataset sont considerees fidèles par construction (peer-associated,
pas de note d'expert 0-5 a comparer comme pour Mangler) -- ce run sert donc a mesurer si le
critere d'alignement actuel (atteignabilite permissive, path_exists) produit des verdicts
globalement coherents avec ca, AVANT d'investir dans un critere plus strict
(_project_process_clause / bpmn_guards_node.py, delibere mais pas encore code -- cf. discussion).
Un taux eleve de NOT_FAITHFUL sur ce dataset serait un signal fort a interroger en premier.

Structure du dataset (les deux sous-ensembles partagent exactement la meme forme de dossier,
seule la convention de nommage des cas differe -- identifiants numeriques de revision pour
NewDataset, noms de processus/"ModelN-M" pour OriginalDataset) :
  {subset}/Texts/{case_name}.txt
  {subset}/Models/{case_name}.bpmn
  {subset}/Groundtruth/{case_name}.json   -- jamais lu par ce script (aucune comparaison
                                              automatique programmee ici ; verite terrain
                                              assumee fidele, cf. note ci-dessus)

Stockage incremental : un fichier par cas, ecrit des que le resultat est obtenu -- jamais tout
en memoire jusqu'a la fin. Si le run s'arrete (crash, interruption), relancer ce script reprend
exactement la ou il s'etait arrete : chaque cas deja present sur disque est ignore. Repertoire de
resultats NEUF et distinct de celui du corpus Mangler (results/dataset_runs_v2) -- dataset
different, jamais melanger les deux dans le meme dossier.

Retry borne avec backoff exponentiel, erreurs transitoires (429) detectees APRES un invoke()
reussi (pas seulement par try/except) -- meme mecanisme, meme rationale que la version Mangler
de ce script (cf. son historique : 162/215 cas mistral-nemotron perdus sur v1 faute de ce
controle). Rien de nouveau ici, report fidele de ce qui a deja ete prouve necessaire.

Configurable : SINGLE_RUN pour ne traiter qu'un seul cas precis (subset, case_name),
independamment de la boucle sur tout le dataset."""

import json
import os
import time

from agent.graph import graph
from agent.config import MODELS_TO_RUN

# Racine du dataset LRE. Chemin relatif a la position reelle de ce script (AAAI/run.py) vers
# BPMN/datasets/lre/alignment_model_text-master/datasets -- "lre" est un dossier FRERE de
# AAAI/ (tous deux sous BPMN/datasets/... et BPMN/AAAI/ respectivement), pas un enfant de AAAI/
# -- d'ou le ".." avant de redescendre dans datasets/lre. AJUSTER si le layout reel differe
# encore (ex. deplacement futur du depot).
LRE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "lre",
    "alignment_model_text-master", "datasets",
)

# Les deux sous-ensembles partagent la meme forme de dossier (Groundtruth/Image/Models/Texts,
# appariement 1:1 par nom de fichier de base) -- traites uniformement par discover_lre_cases().
# Restreint a OriginalDataset seul pour ce run -- decommenter "NewDataset" pour l'inclure.
LRE_SUBSETS = ["OriginalDataset"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")

# Identifiant de run -- granularite supplementaire pour pouvoir relancer plusieurs fois SANS
# ecraser ni fusionner un run precedent avec le suivant (ex. mesurer la variance d'une meme
# configuration sur plusieurs executions completes). Volontairement PAS auto-incremente a
# chaque invocation du script (contrairement au motif run_N.json utilise ailleurs dans ce
# projet pour un seul fichier combine par execution) -- ici, un auto-increment casserait la
# reprise sur crash : redemarrer le script apres une interruption creerait un NOUVEAU run au
# lieu de reprendre celui en cours, puisque already_done() ne verrait alors jamais les fichiers
# du run precedent. Change RUN_ID toi-meme quand tu veux un run complet entierement frais ;
# laisse-le inchange pour reprendre un run deja en cours apres une interruption.
RUN_ID = "run_2"

STATE_SPACE_PROMPT_NAME = "one_shot"
PRECONDITION_PROMPT_NAME = "zero_shot"

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 5
BACKOFF_MAX_SECONDS = 120

# Motifs d'erreur de node consideres transitoires -- identique a la version Mangler de ce
# script, meme rationale (cf. docstring de tete).
TRANSIENT_ERROR_PATTERNS = ("429", "Too Many Requests")

# Aucun fichier confirme malforme sur ce dataset a ce jour (contrairement au corpus Mangler, ou
# EXCLUDED_MALFORMED a ete construit par audit explicite -- audit_bpmn_silent_losses.py +
# verify_reconstruction.py). Vide par defaut, jamais pre-rempli par supposition -- a completer
# UNIQUEMENT si un audit equivalent est mene sur ce dataset et confirme des cas irrecuperables,
# meme discipline que pour Mangler (jamais une exclusion devinee).
EXCLUDED_MALFORMED: set[str] = set()

# Si None, boucle sur tout le dataset (les deux sous-ensembles, moins les exclus) x
# MODELS_TO_RUN. Si renseigne, ne traite QUE ce cas precis -- ex. ("NewDataset", "1066443049_rev6").
SINGLE_RUN: tuple[str, str] | None = None

# Filtre optionnel de developpement, meme esprit que DESCRIPTIONS_SUBSET dans la version
# Mangler -- absent par defaut (None = dataset complet, ce que demande ce run). Renseigner avec
# un set de (subset, case_name) pour restreindre a un sous-ensemble de test avant un run complet.
LRE_CASE_FILTER: set[tuple[str, str]] | None = None

# Limite le nombre de cas traites aux N premiers (ordre deterministe : discover_lre_cases trie
# deja alphabetiquement par cas, jamais un tirage aleatoire) -- validation rapide avant un run
# complet sur tout le dataset, meme esprit que LRE_CASE_FILTER mais par compte plutot que par
# selection nominative. None = pas de limite (tout le dataset, apres LRE_CASE_FILTER eventuel).
LRE_MAX_CASES: int | None = 10


def discover_lre_cases(lre_root: str, subsets: list[str]) -> list[tuple[str, str, str]]:
    """Retourne une liste de (subset, case_name, bpmn_filename). Appariement par nom de fichier
    de base entre Texts/ et Models/ -- jamais suppose : un texte sans modele correspondant (ou
    l'inverse) est explicitement signale et exclu, jamais silencieusement ignore ni force."""
    cases = []
    for subset in subsets:
        texts_dir = os.path.join(lre_root, subset, "Texts")
        models_dir = os.path.join(lre_root, subset, "Models")
        if not os.path.isdir(texts_dir) or not os.path.isdir(models_dir):
            print(f"  [lre] WARNING: subset {subset!r} missing Texts/ or Models/ under {lre_root} -- skipped")
            continue

        text_names = {os.path.splitext(f)[0] for f in os.listdir(texts_dir) if f.endswith(".txt")}
        model_files = {os.path.splitext(f)[0]: f for f in os.listdir(models_dir) if f.endswith(".bpmn")}

        for name in sorted(text_names):
            key = f"{subset}/{name}"
            if key in EXCLUDED_MALFORMED:
                continue
            if name not in model_files:
                print(f"  [lre] WARNING: {subset}/{name} has a text but no matching .bpmn model -- skipped")
                continue
            cases.append((subset, name, model_files[name]))

        orphan_models = sorted(set(model_files) - text_names)
        for name in orphan_models:
            print(f"  [lre] WARNING: {subset}/{name} has a .bpmn model but no matching text -- skipped")

    return cases


def load_text(subset: str, case_name: str) -> str:
    """Tente UTF-8 en premier (le cas normal, silencieux). Certains textes du corpus LRE
    (dataset academique 2017-2018, souvent transcrit/edite sous Windows) ne sont PAS encodes en
    UTF-8 -- accents, guillemets courbes, tirets cadratins en Windows-1252 plutot qu'en UTF-8,
    d'ou un octet comme 0xd5 invalide en UTF-8 mais valide en cp1252. Repli sur cp1252 --
    surensemble de Latin-1, capable de decoder N'IMPORTE QUELLE sequence d'octets sans jamais
    lever d'erreur (contrairement a errors='ignore'/'replace', qui perdrait des caracteres sans
    le dire), donc pas de risque de crash en cascade, et un decodage plausible pour la famille
    de corpus la plus probable ici. Le repli est signale, jamais silencieux."""
    text_path = os.path.join(LRE_ROOT, subset, "Texts", f"{case_name}.txt")
    try:
        with open(text_path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        print(f"  [lre] WARNING: {subset}/{case_name}.txt is not valid UTF-8 ({e}) -- "
              f"falling back to cp1252 (most likely encoding for this corpus)")
        with open(text_path, encoding="cp1252") as f:
            return f.read()


def result_path(model_key: str, subset: str, case_name: str) -> str:
    model_dir = os.path.join(RESULTS_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{subset}__{case_name}__{RUN_ID}.json")


def already_done(model_key: str, subset: str, case_name: str) -> bool:
    return os.path.exists(result_path(model_key, subset, case_name))


def _backoff_delay(attempt: int) -> float:
    return min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)


def _transient_node_errors(final_state: dict) -> list[str]:
    """Erreurs de node capturees par les wrappers (donc invisibles au try/except) qui
    correspondent a un probleme transitoire d'infrastructure, pas de contenu."""
    return [
        e for e in final_state.get("errors", [])
        if any(pattern in e for pattern in TRANSIENT_ERROR_PATTERNS)
    ]


def run_one(model_key: str, subset: str, case_name: str, bpmn_filename: str) -> None:
    out_path = result_path(model_key, subset, case_name)
    # CORRECTIF : load_text() vivait jusqu'ici HORS de toute protection -- en amont meme de la
    # boucle retry/backoff qui protege deja graph.invoke(). Un seul texte source dans une
    # encodage inattendue (cf. load_text) levait une exception non rattrapee qui remontait
    # jusqu'au script principal et tuait TOUTE l'execution -- tous les cas restants, tous
    # modeles confondus, perdus, alors que la philosophie de tout ce fichier est precisement
    # de ne jamais laisser un seul cas faire tomber le run entier (meme discipline que la
    # capture d'erreur de _with_error_capture cote pipeline). Le repli cp1252 de load_text()
    # absorbe deja la cause la plus probable ; ce garde-fou couvre le reste (fichier manquant,
    # permissions, etc.) sans jamais bloquer les cas suivants.
    try:
        text = load_text(subset, case_name)
    except Exception as e:
        with open(out_path, "w") as f:
            json.dump({"fatal_error": f"load_text failed: {e}"}, f, indent=2)
        print(f"  [{model_key}] {subset}/{case_name} -- ECHEC DEFINITIF (texte source "
              f"illisible: {e})")
        return
    bpmn_path = os.path.join(LRE_ROOT, subset, "Models", bpmn_filename)

    initial_state = {
        # Prefixe par le sous-ensemble -- jamais utilise pour la logique du pipeline elle-meme
        # (state.py : "jamais utilise pour la logique, uniquement pour la tracabilite"), evite
        # simplement une collision entre un cas "Model10-1" (OriginalDataset) et un homonyme
        # eventuel d'un autre sous-ensemble.
        "case_name": f"{subset}/{case_name}",
        "text": text,
        "bpmn_path": bpmn_path,
        "state_space_prompt_name": STATE_SPACE_PROMPT_NAME,
        "precondition_prompt_name": PRECONDITION_PROMPT_NAME,
        "state_space_model": model_key,
        "precondition_model": model_key,
        # Meme decision que la version Mangler : chaque modele redige SON PROPRE rapport,
        # comparable entre redacteurs grace au controle mecanique quote_verbatim.
        "report_model": model_key,
        # NB : bpmn_description_index / bpmn_model_index / bpmn_expert_score (champs
        # PipelineState specifiques au corpus Zenodo/Mangler) ne sont pas renseignes ici --
        # sans objet pour LRE (pas de note d'expert 0-5, pas de "plusieurs modeles par
        # description"). Champs NotRequired, absence normale, jamais une erreur.
    }

    last_error = None
    last_state = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            final_state = graph.invoke(initial_state)
        except Exception as e:
            last_error = e
            print(f"  [{model_key}] {subset}/{case_name} -- tentative {attempt + 1}/"
                  f"{MAX_RETRIES + 1} echouee: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(_backoff_delay(attempt))
            continue

        last_state = final_state
        transient = _transient_node_errors(final_state)
        if transient and attempt < MAX_RETRIES:
            delay = _backoff_delay(attempt)
            print(f"  [{model_key}] {subset}/{case_name} -- erreur(s) transitoire(s) dans "
                  f"l'etat ({len(transient)}), tentative {attempt + 1}/{MAX_RETRIES + 1}, "
                  f"retry dans {delay:.0f}s")
            time.sleep(delay)
            continue

        with open(out_path, "w") as f:
            json.dump(final_state, f, indent=2, default=str, ensure_ascii=False)
        errors = final_state.get("errors")
        if not errors:
            status = "OK"
        elif transient:
            status = (f"OK PARTIEL -- erreurs transitoires persistantes apres "
                      f"{MAX_RETRIES + 1} tentatives: {errors}")
        else:
            status = f"OK avec erreurs internes de node: {errors}"
        print(f"  [{model_key}] {subset}/{case_name} -- {status}")
        return

    if last_state is not None:
        with open(out_path, "w") as f:
            json.dump(last_state, f, indent=2, default=str, ensure_ascii=False)
        print(f"  [{model_key}] {subset}/{case_name} -- ECHEC apres {MAX_RETRIES + 1} "
              f"tentatives, dernier etat partiel ecrit (errors: {last_state.get('errors')})")
        return
    with open(out_path, "w") as f:
        json.dump({"fatal_error": str(last_error)}, f, indent=2)
    print(f"  [{model_key}] {subset}/{case_name} -- ECHEC DEFINITIF "
          f"apres {MAX_RETRIES + 1} tentatives")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cases = discover_lre_cases(LRE_ROOT, LRE_SUBSETS)
    if LRE_CASE_FILTER is not None:
        cases = [c for c in cases if (c[0], c[1]) in LRE_CASE_FILTER]
    if LRE_MAX_CASES is not None:
        cases = cases[:LRE_MAX_CASES]
    print(f"LRE dataset: {len(cases)} (text, model) pair(s) discovered across {LRE_SUBSETS}"
          f"{' (filtered)' if LRE_CASE_FILTER is not None else ''}"
          f"{f' (limited to first {LRE_MAX_CASES})' if LRE_MAX_CASES is not None else ''}")

    if SINGLE_RUN is not None:
        subset, case_name = SINGLE_RUN
        matching = [c for c in cases if c[0] == subset and c[1] == case_name]
        if not matching:
            raise SystemExit(f"SINGLE_RUN {SINGLE_RUN!r} not found among discovered LRE cases")
        _, _, bpmn_filename = matching[0]
        for model_key in MODELS_TO_RUN:
            run_one(model_key, subset, case_name, bpmn_filename)
        raise SystemExit(0)

    for model_key in MODELS_TO_RUN:
        print(f"\n########## Modele : {model_key} ##########")
        for subset, case_name, bpmn_filename in cases:
            if already_done(model_key, subset, case_name):
                print(f"  [{model_key}] {subset}/{case_name} -- deja fait, ignore")
                continue
            run_one(model_key, subset, case_name, bpmn_filename)