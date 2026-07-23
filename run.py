"""Run sur l'ensemble du corpus (text_and_bpmn/bpmn/) : associe chaque modele BPMN a son texte
source, invoque le pipeline complet une fois par (modele LLM x description x modele BPMN).

Stockage incremental : un fichier par paire, ecrit des que le resultat est obtenu -- jamais tout
en memoire jusqu'a la fin. Si le run s'arrete (crash, interruption), relancer ce script reprend
exactement la ou il s'etait arrete : chaque paire deja presente sur disque est ignoree.

Retry borne avec backoff exponentiel : une paire qui echoue est retentee MAX_RETRIES fois
(delai 5s, 10s, 20s, 40s, 80s, plafonne a BACKOFF_MAX_SECONDS) avant d'etre marquee en echec
definitif (et ecrite comme telle, pour ne pas la retenter indefiniment a chaque relance sans le
savoir).

Point crucial, decouvert sur le run v1 : les erreurs transitoires (429 rate limit) ne font
JAMAIS planter graph.invoke() -- le confinement d'erreur des wrappers de node les capture dans
state['errors'], le run est ecrit comme "fait" du premier coup et already_done() le saute a
jamais. C'est la cause directe des 19.1% d'exploitabilite de mistral-nemotron sur v1 (162/215
cas tues par 429 a state_space, jamais retentes). Le retry doit donc inspecter state['errors']
apres un invoke() reussi, pas seulement attraper les exceptions -- c'est ce que fait run_one
desormais. Si les tentatives s'epuisent avec des erreurs transitoires restantes, le meilleur
etat obtenu est ecrit quand meme (resultat partiel explicite, jamais perdu).

Configurable : SINGLE_RUN pour ne traiter qu'une paire precise, independamment de la boucle sur
tout le corpus -- utile pour tester/relancer un seul cas."""

import json
import os
import time

from agent.graph import graph
from agent.config import MODELS_TO_RUN

BPMN_ROOT = os.path.join(os.path.dirname(__file__), "text_and_bpmn", "bpmn")
# v2 : repertoire NEUF, obligatoire -- (1) le pipeline a change (demotion par confiance, filtre
# start/end, deduplication de U), tous les cas doivent etre recalcules pour la comparaison
# avant/apres ; (2) reutiliser dataset_runs ferait sauter tous les cas via already_done(),
# y compris les 429 de v1 ecrits comme "faits". v1 reste intact comme reference.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "dataset_runs_v2")

STATE_SPACE_PROMPT_NAME = "one_shot"
PRECONDITION_PROMPT_NAME = "zero_shot"

MAX_RETRIES = 5                 # releve de 2 -> 5 : avec le backoff exponentiel, 5 tentatives
                                 # couvrent ~2.5 min de fenetre, suffisant pour laisser retomber
                                 # une limite de debit persistante (cause n.1 de perte v1)
BACKOFF_BASE_SECONDS = 5
BACKOFF_MAX_SECONDS = 120

# Motifs d'erreur de node consideres transitoires : un invoke() "reussi" dont state['errors']
# contient l'un d'eux est retente comme un echec. Volontairement limite au rate limiting --
# les reponses vides de gpt-oss-20b n'y figurent PAS (cause non diagnostiquee, cf.
# dataset_run_observations.md section 20 point 2 : les retenter en aveugle brulerait des appels
# sans diagnostic ; a reevaluer une fois la cause identifiee).
TRANSIENT_ERROR_PATTERNS = ("429", "Too Many Requests")

# Confirme non exploitables (audit_bpmn_silent_losses.py + verify_reconstruction.py, sur les
# 223 fichiers du corpus -- 100% des flux signales irrecuperables sur chacun des 8, pas un
# echantillon partiel). Exclus explicitement plutot que de produire un resultat trompeur.
EXCLUDED_MALFORMED = {
    "E_j01/10.bpmn2.xml",
    "E_j05/9.bpmn2.xml",
    "G_g03/0.bpmn2.xml",
    "M_g01/3.bpmn2.xml",
    "M_g02/5.bpmn2.xml",
    "M_j02/5.bpmn2.xml",
    "R_g01/0.bpmn2.xml",
    "R_j03/2.bpmn2.xml",
}

# Si None, boucle sur tout le corpus (moins les exclus) x MODELS_TO_RUN. Si renseigne, ne
# traite QUE cette paire precise, independamment de la boucle -- ex. ("E_j01", "0.bpmn2.xml").
SINGLE_RUN: tuple[str, str] | None = None

# Sous-ensemble de developpement -- validation des fixes v2 et prototypage, JAMAIS la source
# des chiffres du papier (selection stratifiee : 1+ description par famille E/G/M/R/V/X,
# couverture des 3 familles d'artefacts v1, un texte dense). Chiffres finaux : corpus complet
# (DESCRIPTIONS_SUBSET = None). Criteres fixes avant execution.
DESCRIPTIONS_SUBSET: set[str] | None = {
    "E_j02", "E_j03", "G_g03", "M_g01", "R_j02", "V_k09", "X_g01",
}


def load_text(desc_folder: str) -> str:
    text_path = os.path.join(BPMN_ROOT, f"{desc_folder}.txt")
    with open(text_path, encoding="utf-8") as f:
        return f.read()


def result_path(model_key: str, desc_folder: str, bpmn_filename: str) -> str:
    model_dir = os.path.join(RESULTS_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{desc_folder}__{bpmn_filename}.json")


def already_done(model_key: str, desc_folder: str, bpmn_filename: str) -> bool:
    return os.path.exists(result_path(model_key, desc_folder, bpmn_filename))


def _backoff_delay(attempt: int) -> float:
    return min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)


def _transient_node_errors(final_state: dict) -> list[str]:
    """Erreurs de node capturees par les wrappers (donc invisibles au try/except) qui
    correspondent a un probleme transitoire d'infrastructure, pas de contenu."""
    return [
        e for e in final_state.get("errors", [])
        if any(pattern in e for pattern in TRANSIENT_ERROR_PATTERNS)
    ]


def run_one(model_key: str, desc_folder: str, bpmn_filename: str) -> None:
    out_path = result_path(model_key, desc_folder, bpmn_filename)
    text = load_text(desc_folder)
    bpmn_path = os.path.join(BPMN_ROOT, desc_folder, bpmn_filename)

    initial_state = {
        "case_name": desc_folder,
        "text": text,
        "bpmn_path": bpmn_path,
        "state_space_prompt_name": STATE_SPACE_PROMPT_NAME,
        "precondition_prompt_name": PRECONDITION_PROMPT_NAME,
        "state_space_model": model_key,
        "precondition_model": model_key,
        # Decision v2 : chaque modele redige SON PROPRE rapport -- pipeline de bout en bout par
        # modele, evaluation complete incluant la generation finale. Revient sur le defaut
        # herite silencieusement (DEFAULT_REPORT_MODEL_KEY, qui faisait rediger tous les
        # rapports v1 par llama-3.1-8b quel que soit le modele sous test). La prose devient
        # comparable entre redacteurs grace au controle mecanique quote_verbatim
        # (report_node.quote_reproduced_verbatim) -- sans lui, trois redacteurs seraient
        # compares sans instrument.
        "report_model": model_key,
    }

    last_error = None
    last_state = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            final_state = graph.invoke(initial_state)
        except Exception as e:
            # Crash reel (rare : les nodes confinent leurs propres erreurs) -- meme backoff.
            last_error = e
            print(f"  [{model_key}] {desc_folder}/{bpmn_filename} -- tentative {attempt + 1}/"
                  f"{MAX_RETRIES + 1} echouee: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(_backoff_delay(attempt))
            continue

        # invoke() a reussi -- mais un 429 confine dans state['errors'] est un echec
        # transitoire deguise, pas un resultat : on retente au lieu de l'ecrire comme fait
        # (cause directe de la perte de 162 cas nemotron sur v1, cf. docstring de tete).
        last_state = final_state
        transient = _transient_node_errors(final_state)
        if transient and attempt < MAX_RETRIES:
            delay = _backoff_delay(attempt)
            print(f"  [{model_key}] {desc_folder}/{bpmn_filename} -- erreur(s) transitoire(s) "
                  f"dans l'etat ({len(transient)}), tentative {attempt + 1}/{MAX_RETRIES + 1}, "
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
        print(f"  [{model_key}] {desc_folder}/{bpmn_filename} -- {status}")
        return

    # Crash a toutes les tentatives. Si au moins un invoke() a produit un etat (crash apres des
    # tentatives transitoires), on ecrit ce meilleur etat partiel plutot que de tout perdre.
    if last_state is not None:
        with open(out_path, "w") as f:
            json.dump(last_state, f, indent=2, default=str, ensure_ascii=False)
        print(f"  [{model_key}] {desc_folder}/{bpmn_filename} -- ECHEC apres {MAX_RETRIES + 1} "
              f"tentatives, dernier etat partiel ecrit (errors: {last_state.get('errors')})")
        return
    with open(out_path, "w") as f:
        json.dump({"fatal_error": str(last_error)}, f, indent=2)
    print(f"  [{model_key}] {desc_folder}/{bpmn_filename} -- ECHEC DEFINITIF "
          f"apres {MAX_RETRIES + 1} tentatives")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    description_folders = sorted(
        d for d in os.listdir(BPMN_ROOT) if os.path.isdir(os.path.join(BPMN_ROOT, d))
    )

    if SINGLE_RUN is not None:
        desc_folder, bpmn_filename = SINGLE_RUN
        for model_key in MODELS_TO_RUN:
            run_one(model_key, desc_folder, bpmn_filename)
        raise SystemExit(0)

    for model_key in MODELS_TO_RUN:
        print(f"\n########## Modele : {model_key} ##########")
        for desc_folder in description_folders:
            if DESCRIPTIONS_SUBSET is not None and desc_folder not in DESCRIPTIONS_SUBSET:
                continue
            desc_path = os.path.join(BPMN_ROOT, desc_folder)
            bpmn_files = sorted(f for f in os.listdir(desc_path) if f.endswith(".bpmn2.xml"))
            for bpmn_filename in bpmn_files:
                key = f"{desc_folder}/{bpmn_filename}"
                if key in EXCLUDED_MALFORMED:
                    continue
                if already_done(model_key, desc_folder, bpmn_filename):
                    print(f"  [{model_key}] {desc_folder}/{bpmn_filename} -- deja fait, ignore")
                    continue
                run_one(model_key, desc_folder, bpmn_filename)