"""Verification croisee : pour llama-3.1-8b, les 3 fichiers de base a zero detection totale
sur les operateurs required (E_j02/5, M_g01/10, X_g01/0) sont-ils explicables par des cles de
precondition malformees (34.2% documente pour ce modele, dataset_run_observations.md section
4.6) plutot que par un vrai defaut de rappel de l'alignement ?

Principe du test, dans l'ordre :
1. Charger le manifest de mutants (results/mutants/manifest.json) -- retrouver, pour chaque
   mutant "required" applicable de ces 3 fichiers, quelle activite/etat a ete mute (details).
2. Charger le run precondition_extraction_with_retry de llama-3.1-8b pour chacun de ces 3
   fichiers -- retrouver le statut de la cible (target) qui aurait du encoder la precondition
   testee par cette mutation.
3. Verifier : est-ce que la cible en question est UNRESOLVED avec une raison de type
   "cle malformee" / "target invalide" (peu importe le libelle exact, matching flou sur mots-cles) ?

IMPORTANT -- ce script NE DEVINE PAS le schema exact de vos fichiers (plusieurs noms de champs
possibles selon la version du pipeline : "validated"/"downgraded", "status"/"state",
"reason"/"downgrade_reason"). Il essaie plusieurs cles plausibles et SIGNALE EXPLICITEMENT
quand aucune ne correspond, plutot que de conclure sur une supposition -- meme discipline que
le reste du projet ("jamais un rejet silencieux"). Si la sortie dit "SCHEMA NON RECONNU", il
faut ajuster les noms de champs dans _find_target_status() ci-dessous a la structure reelle de
vos fichiers avant de faire confiance au resultat.

Usage :
    python check_llama_malformed_keys.py \
        --manifest results/mutants/manifest.json \
        --precondition-dir results/precondition_extraction_with_retry \
        --v2-results-dir results/dataset_runs_v2 \
        --model llama-3.1-8b \
        --stuck-bases E_j02/5.bpmn2.xml M_g01/10.bpmn2.xml X_g01/0.bpmn2.xml
"""

import argparse
import json
import os
import re

MALFORMED_KEYWORDS = [
    "malformed", "malforme", "invalid target", "target invalide", "not a valid",
    "n'est pas un", "cle invalide", "invalid key", "unknown target",
]

REQUIRED_OPS = {"remove_activity", "swap_labels", "cross_case_replace"}


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_precondition_run_for_model(precondition_dir: str, model: str) -> dict | None:
    """Cherche un fichier de run precondition_extraction_with_retry associe a `model`.
    Essaie plusieurs conventions de nommage plausibles avant d'abandonner. Retourne None
    (jamais un dict vide) si rien n'est trouve -- l'appelant doit distinguer explicitement
    "pas de fichier trouve" de "fichier trouve mais vide"."""
    if not os.path.isdir(precondition_dir):
        print(f"[check] ATTENTION : {precondition_dir} n'existe pas ou n'est pas un dossier")
        return None
    candidates = [
        f for f in os.listdir(precondition_dir)
        if model.replace("-", "").replace(".", "") in f.replace("-", "").replace(".", "")
        or model in f
    ]
    if not candidates:
        print(f"[check] ATTENTION : aucun fichier dans {precondition_dir} ne mentionne "
              f"'{model}' dans son nom -- ajustez find_precondition_run_for_model() a votre "
              f"convention de nommage reelle (ex. un seul run_N.json partage entre modeles ?)")
        return None
    if len(candidates) > 1:
        print(f"[check] ATTENTION : plusieurs candidats trouves pour '{model}' : {candidates} "
              f"-- utilisation du premier ({candidates[0]}), verifiez que c'est le bon")
    return load_json(os.path.join(precondition_dir, candidates[0]))


def _looks_malformed(reason: str) -> bool:
    reason_low = (reason or "").lower()
    return any(kw in reason_low for kw in MALFORMED_KEYWORDS)


def find_target_status(precondition_data, base_key: str, target_state: str) -> dict:
    """Cherche le statut de `target_state` (ex. "job_application.rated") dans les donnees
    precondition chargees, pour le cas `base_key`. Retourne un dict avec au minimum
    {"found": bool, "status": str|None, "reason": str|None, "looks_malformed": bool}.

    Essaie plusieurs schemas plausibles (liste d'entrees avec 'target', dict indexe par
    target, structure imbriquee par description/desc). Si RIEN ne correspond, retourne
    found=False avec un message explicite -- ne jamais renvoyer un statut suppose."""
    if precondition_data is None:
        return {"found": False, "status": None, "reason": None, "looks_malformed": False,
                "note": "precondition_data est None (fichier absent ou illisible)"}

    desc = base_key.split("/")[0]

    # Schema A : liste plate d'entrees avec des champs "target"/"status"/"reason", parfois
    # sous une cle "validated" + "downgraded", parfois groupees par description.
    def scan(obj):
        if isinstance(obj, dict):
            if "target" in obj and obj.get("target") == target_state:
                return obj
            for v in obj.values():
                r = scan(v)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for item in obj:
                r = scan(item)
                if r is not None:
                    return r
        return None

    # Tente d'abord de restreindre a la description concernee si la structure le permet
    scoped = precondition_data
    if isinstance(precondition_data, dict) and desc in precondition_data:
        scoped = precondition_data[desc]

    entry = scan(scoped) or scan(precondition_data)
    if entry is None:
        return {"found": False, "status": None, "reason": None, "looks_malformed": False,
                "note": f"aucune entree avec target=='{target_state}' trouvee -- schema non "
                        f"reconnu ou etat absent du run"}

    status = entry.get("status") or entry.get("state") or entry.get("verdict")
    reason = entry.get("reason") or entry.get("downgrade_reason") or entry.get("note")
    return {"found": True, "status": status, "reason": reason,
            "looks_malformed": _looks_malformed(reason)}


def mutant_target_state(mutant_entry: dict, v2_results_dir: str, model: str) -> str | None:
    """Retrouve l'etat (entity.state) ancre par l'activite mutee, via le run v2 de reference
    deja utilise pour le ciblage guard-pertinent (memes fichiers que
    _guard_relevant_activities dans mutate_bpmn.py). Retourne None si introuvable -- jamais
    une supposition."""
    base_key = mutant_entry["base"]
    details = mutant_entry.get("details", {})
    activity = (details.get("removed_activity") or details.get("replaced_activity")
                or (details.get("swapped") or [None])[0])
    if activity is None:
        return None

    path = os.path.join(v2_results_dir, model, base_key.replace("/", "__") + ".json")
    v2_data = load_json(path)
    if v2_data is None:
        return None
    matches = v2_data.get("matches", {})
    m = matches.get(activity)
    if isinstance(m, dict):
        return m.get("match")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--precondition-dir", required=True)
    ap.add_argument("--v2-results-dir", required=True)
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--stuck-bases", nargs="+", required=True,
                     help="Fichiers de base a zero detection a investiguer, ex. E_j02/5.bpmn2.xml")
    args = ap.parse_args()

    manifest = load_json(args.manifest)
    if manifest is None:
        raise SystemExit(f"[check] manifest introuvable : {args.manifest}")

    precondition_data = find_precondition_run_for_model(args.precondition_dir, args.model)

    stuck_keys = set()
    for b in args.stuck_bases:
        # normalise "E_j02/5.bpmn2.xml" -> "E_j02/5.bpmn2.xml" (deja le format du manifest)
        stuck_keys.add(b)

    print(f"\n[check] Modele : {args.model}")
    print(f"[check] Fichiers investigues : {sorted(stuck_keys)}\n")

    any_malformed_explains_gap = False
    total_checked = 0

    for entry in manifest:
        if entry.get("base") not in stuck_keys:
            continue
        if entry.get("operator") not in REQUIRED_OPS:
            continue
        if not entry.get("applicable"):
            continue

        total_checked += 1
        base_key = entry["base"]
        op = entry["operator"]
        variant = entry["variant"]

        target_state = mutant_target_state(entry, args.v2_results_dir, args.model)
        if target_state is None:
            print(f"  [{base_key} | {op} variant {variant}] -- impossible de retrouver l'etat "
                  f"ancre par l'activite mutee (match absent ou run v2 illisible) -- ne compte "
                  f"pas comme preuve dans un sens ou l'autre")
            continue

        status_info = find_target_status(precondition_data, base_key, target_state)

        if not status_info["found"]:
            print(f"  [{base_key} | {op} variant {variant}] cible='{target_state}' -- "
                  f"SCHEMA NON RECONNU ({status_info['note']}) -- ajustez find_target_status() "
                  f"a la structure reelle de vos fichiers avant de conclure quoi que ce soit")
            continue

        malformed_tag = " <-- CLE MALFORMEE" if status_info["looks_malformed"] else ""
        print(f"  [{base_key} | {op} variant {variant}] cible='{target_state}' "
              f"status={status_info['status']!r} reason={status_info['reason']!r}"
              f"{malformed_tag}")
        if status_info["looks_malformed"]:
            any_malformed_explains_gap = True

    print(f"\n[check] {total_checked} mutation(s) required investiguee(s) sur les fichiers "
          f"bloques.")
    if any_malformed_explains_gap:
        print("[check] AU MOINS UNE cible correspond a une raison de type cle malformee -- "
              "l'hypothese 'cause en amont (precondition), pas un defaut de l'alignement' "
              "a au moins un appui concret. Reste a verifier si ca explique TOUTES les "
              "non-detections ou seulement certaines.")
    else:
        print("[check] AUCUNE cible investiguee ne correspond a une raison de type cle "
              "malformee (parmi celles au schema reconnu) -- l'hypothese 'clefs malformees' "
              "n'est PAS confirmee par ce test ; le rappel faible sur ces 3 fichiers reste "
              "a expliquer autrement (vrai defaut de rappel, ou autre cause en amont non "
              "testee ici).")


if __name__ == "__main__":
    main()