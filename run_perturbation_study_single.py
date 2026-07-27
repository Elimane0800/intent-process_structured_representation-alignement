"""Version single-case de run_perturbation_study.py -- evalue les mutants generes par
mutate_bpmn_single.py (tous issus d'UN SEUL fichier BPMN de base) contre le baseline single-case
(results/single_case_validation/), pas le run complet du corpus (results/dataset_runs_v2/).
"Single" porte sur le BPMN de base, pas sur les modeles : chaque mutant reste evalue contre
TOUS les modeles de MODELS_TO_RUN, exactement comme la version corpus complet.

Reutilise integralement la logique testee de run_perturbation_study.py (load_baseline,
run_mutant_alignment, classify_required/classify_forbidden, run_study, summarize) -- zero
reimplementation, zero risque de divergence. Seuls les repertoires d'entree/sortie changent,
obtenus en substituant les constantes de module AVANT d'appeler run_study() :
load_baseline()/_v2_result_path() lisent V2_RESULTS_DIR depuis le namespace du module au
moment de l'appel (resolution Python standard des variables globales), pas au moment de
l'import -- la substitution est donc effective sans toucher au fichier original (verifie
directement, pas suppose).

Modeles : la liste COMPLETE de agent.config.MODELS_TO_RUN (pas le sous-ensemble de 3 modeles
code en dur dans la version corpus-complet) -- c'est ce que run_single_case_validation.py a
reellement produit comme baselines sur disque ; utiliser moins de modeles laisserait des
baselines exploitables inutilisees, en utiliser d'autres echouerait faute de fichier."""

import os

import run_perturbation_study as base
from agent.config import MODELS_TO_RUN as MODELS

BASE_DIR = os.path.dirname(__file__)
MUTANTS_DIR = os.path.join(BASE_DIR, "results", "mutants_single")
SINGLE_CASE_RESULTS_DIR = os.path.join(BASE_DIR, "results", "single_case_validation")
OUT_DIR = os.path.join(BASE_DIR, "results", "perturbation_study_single")


if __name__ == "__main__":
    base.MUTANTS_DIR = MUTANTS_DIR
    base.V2_RESULTS_DIR = SINGLE_CASE_RESULTS_DIR
    base.OUT_DIR = OUT_DIR

    results = base.run_study(models=MODELS, out_dir=OUT_DIR)
    base.summarize(results)