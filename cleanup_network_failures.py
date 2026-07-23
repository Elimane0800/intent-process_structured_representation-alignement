"""Nettoyage des cas 'OK avec erreurs internes' dus a une panne reseau transitoire (survenue
ENTRE deux lancements de run.py, donc jamais vue par le retry/backoff interne a run_one -- celui
-ci ne protege que les echecs survenant PENDANT une invocation, pas une coupure qui touche
plusieurs cas d'affilee avant que le script ne soit relance).

already_done() ne regarde que l'existence du fichier, pas son contenu -- un cas ecrit avec
'Connection error.' en cascade (state_space echoue -> tous les nodes en aval echouent sur cle
manquante) est donc silencieusement saute a la relance, jamais retente. Ce script supprime ces
fichiers precis pour que la prochaine execution de run.py les retraite normalement -- rien
d'autre n'est touche (les vrais succes et les echecs de CONTENU, ex. parse failed/reponses
vides, restent en l'etat : ils ne sont pas transitoires, les retenter en aveugle ne changerait
rien et masquerait un vrai probleme de contenu sous un ravalement reseau)."""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "dataset_runs_v2")

# Motif strict : uniquement les pannes reseau transitoires entre deux lancements, jamais les
# erreurs de contenu (parse failed, reponses vides, 429 -- ce dernier est deja gere par le
# backoff DANS run_one, pas par ce script).
NETWORK_ERROR_PATTERNS = ("Connection error", "Connection reset", "Connection aborted",
                           "Read timed out", "Timeout")


def is_network_cascade_failure(state: dict) -> bool:
    if "fatal_error" in state:
        return any(p in str(state["fatal_error"]) for p in NETWORK_ERROR_PATTERNS)
    errors = state.get("errors", [])
    return any(any(p in e for p in NETWORK_ERROR_PATTERNS) for e in errors)


def main(dry_run: bool = True) -> None:
    if not os.path.isdir(RESULTS_DIR):
        print(f"[cleanup] {RESULTS_DIR} introuvable -- rien a faire")
        return

    to_delete = []
    for model_dir in sorted(os.listdir(RESULTS_DIR)):
        model_path = os.path.join(RESULTS_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(model_path, fname)
            try:
                with open(fpath) as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [cleanup] SKIP {model_dir}/{fname} -- illisible ({e}), inspection manuelle requise")
                continue
            if is_network_cascade_failure(state):
                to_delete.append(fpath)

    print(f"[cleanup] {len(to_delete)} fichier(s) en echec de cascade reseau detecte(s) "
          f"sur {RESULTS_DIR}")
    for fpath in to_delete:
        rel = os.path.relpath(fpath, RESULTS_DIR)
        if dry_run:
            print(f"  [dry-run] supprimerait: {rel}")
        else:
            os.remove(fpath)
            print(f"  supprime: {rel}")

    if dry_run and to_delete:
        print(f"\n[cleanup] DRY RUN -- rien supprime. Relancer avec --confirm pour supprimer "
              f"reellement ces {len(to_delete)} fichier(s), puis relancer run.py pour les "
              f"retraiter.")


if __name__ == "__main__":
    main(dry_run="--confirm" not in sys.argv)