"""Validation d'un seul cas (une description, un fichier BPMN) sur TOUS les modeles de
MODELS_TO_RUN (config.py, 11 entrees actuellement) -- deliberement SEPARE de run.py :

- Sortie ecrite dans results/single_case_validation/, jamais dans results/dataset_runs/ ni
  results/dataset_runs_v2/ -- ce run est une validation ponctuelle, pas une entree du corpus
  officiel, et ne doit jamais etre confondu avec l'un ou l'autre ni ecraser quoi que ce soit.

- Plusieurs modeles de MODELS_TO_RUN n'ont jamais ete smoke-testes avec base_llm.py dans ce
  projet (point reste en suspens depuis le debut) -- des echecs isoles sont attendus et
  informatifs, pas une raison d'arreter le run : chaque modele est traite independamment, un
  echec sur l'un n'empeche jamais les autres de continuer (cf. run_one, capture large des
  exceptions).

Usage : ajuster CASE ci-dessous puis `python3 run_single_case_validation.py`."""

import json
import os
import traceback

from agent.graph import graph
from agent.config import MODELS_TO_RUN

BPMN_ROOT = os.path.join(os.path.dirname(__file__), "text_and_bpmn", "bpmn")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "single_case_validation")

# Le cas a valider -- E_j03 (Work Accident), le fichier BPMN uploade correspond a "0.bpmn2.xml"
# dans l'arborescence du projet (E_j03/0.bpmn2.xml).
CASE: tuple[str, str] = ("E_j01", "0.bpmn2.xml")

# Tous les modeles actuellement configures (config.py) -- y compris ceux jamais
# smoke-testes ; c'est precisement l'objet de ce run de validation large.
MODELS = MODELS_TO_RUN

STATE_SPACE_PROMPT_NAME = "one_shot"
PRECONDITION_PROMPT_NAME = "zero_shot"


def load_text(desc_folder: str) -> str:
    with open(os.path.join(BPMN_ROOT, f"{desc_folder}.txt"), encoding="utf-8") as f:
        return f.read()


def run_one(model_key: str, desc_folder: str, bpmn_filename: str) -> None:
    model_dir = os.path.join(OUT_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, f"{desc_folder}__{bpmn_filename}.json")

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
        "report_model": model_key,  # chaque modele redige son propre rapport (V2)
    }

    print(f"[{model_key}] lancement sur {desc_folder}/{bpmn_filename} ...")
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        # Echec complet (ex. modele jamais smoke-teste, plomberie base_llm.py absente pour ce
        # provider) -- ecrit explicitement pour diagnostic, jamais silencieux, et n'arrete
        # jamais la boucle sur les autres modeles.
        with open(out_path, "w") as f:
            json.dump({"fatal_error": str(e), "traceback": traceback.format_exc()}, f, indent=2)
        print(f"[{model_key}] ECHEC COMPLET : {e}")
        return

    with open(out_path, "w") as f:
        json.dump(final_state, f, indent=2, default=str, ensure_ascii=False)

    errors = final_state.get("errors")
    status = "OK" if not errors else f"OK avec erreurs internes de node: {errors}"
    print(f"[{model_key}] termine -- {status}")
    print(f"[{model_key}] ecrit dans : {out_path}")


if __name__ == "__main__":
    desc_folder, bpmn_filename = CASE
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[validation] {len(MODELS)} modele(s) a tester : {MODELS}\n")
    for model_key in MODELS:
        run_one(model_key, desc_folder, bpmn_filename)
    print(f"\nTermine. Resultats dans : {OUT_DIR}/<modele>/{desc_folder}__{bpmn_filename}.json")