"""Version single-case de mutate_bpmn.py -- genere des mutants pour UN SEUL fichier BPMN (CASE
ci-dessous, aligne EXACTEMENT sur celui de run_single_case_validation.py -- jamais un choix
independant, pour garantir que mutate/perturbation/single_case_validation portent tous sur le
meme cas sans risque de divergence). "Single" porte sur le BPMN, pas sur les modeles :
plusieurs mutants (plusieurs perturbations de ce meme fichier, jusqu'a 3 variantes x 6
operateurs) sont generes, chacun evalue ensuite contre TOUS les modeles de MODELS_TO_RUN --
exactement comme la version corpus complet, seul le nombre de fichiers de base change (un seul
au lieu du sous-ensemble de developpement).

Reutilise integralement la logique testee de mutate_bpmn.py (aucune reimplementation des 6
operateurs, aucun risque de divergence) : ce script importe generate_mutants() et se contente
de restreindre base_files a une seule entree.

Limite honnete, pas un bug : avec un seul fichier de base, cross_case_replace n'a AUCUN label
d'activite etranger a puiser -- le mecanisme original tire ses intrus des AUTRES cas presents
dans le meme batch (cf. docstring de mutate_bpmn.py). Consequence directe et attendue :
cross_case_replace ressort "applicable": False pour ses 3 variantes dans le manifest -- exactement
le meme mecanisme de repli que pour n'importe quel autre operateur structurellement inapplicable,
rien n'est invente pour combler ce trou."""

import os

from mutate_bpmn import generate_mutants
from agent.config import MODELS_TO_RUN as MODELS

BASE_DIR = os.path.dirname(__file__)
BPMN_ROOT = os.path.join(BASE_DIR, "text_and_bpmn", "bpmn")
OUT_DIR = os.path.join(BASE_DIR, "results", "mutants_single")
# Baseline single-case (results/single_case_validation/<model>/<desc>__<file>.json), PAS le run
# complet du corpus (results/dataset_runs_v2/) -- c'est le seul repertoire qui contient
# reellement une reference pour le cas unique traite ici, mais pour TOUS les modeles deja
# passes par run_single_case_validation.py.
V2_RESULTS_DIR = os.path.join(BASE_DIR, "results", "single_case_validation")

# Aligne EXACTEMENT sur CASE dans run_single_case_validation.py -- ne jamais laisser diverger,
# sinon _guard_relevant_activities_union() ne trouvera aucune baseline exploitable pour aucun
# modele.
CASE: tuple[str, str] = ("E_j01", "0.bpmn2.xml")


if __name__ == "__main__":
    desc, fname = CASE
    base_key = f"{desc}/{fname}"
    print(f"[mutate_single] cas : {base_key}  --  {len(MODELS)} modele(s) : {MODELS}")

    base_files = {base_key: os.path.join(BPMN_ROOT, desc, fname)}
    generate_mutants(base_files, OUT_DIR, V2_RESULTS_DIR, MODELS)