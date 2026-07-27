"""Aplatit un run JSON de state_space_node (results/state_space_protocol3/run_N.json) en un CSV
lisible -- une ligne par candidat brut genere par le LLM (etape A), avec son sort a travers les
trois etapes du node : generation (candidates), filtre lexical WordNet (validation_report),
deduplication referentielle (state_space_dedup).

Usage :
    python state_space_to_csv.py                          # prend le run le plus recent
    python state_space_to_csv.py chemin/vers/run_17.json   # run precis

Sortie : result_csv/state_space/state_space_run_N.csv -- N incremental, jamais d'ecrasement
(meme discipline que les autres repertoires de resultats du projet)."""

import csv
import glob
import json
import os
import re
import sys

STATE_SPACE_DIR = os.path.join("results", "state_space_protocol3")
OUT_DIR = os.path.join("result_csv", "state_space")

COLUMNS = [
    "prompt_name", "llm", "use_case",
    "raw_entity", "raw_states",
    "lexname", "kept_lexical", "rejection_reason",
    "dedup_rule", "final_entity", "final_states",
]


def _latest_run(directory: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(directory, "run_*.json")),
        key=lambda p: int(re.search(r"run_(\d+)\.json$", p).group(1)),
    )
    if not candidates:
        raise SystemExit(f"Aucun run_*.json trouve dans {directory}")
    return candidates[-1]


def _resolve_final_entity(raw_name: str, absorbed_to_final: dict) -> str:
    """Suit la chaine d'absorption jusqu'a l'entite finale -- une fusion regle 1 peut ensuite
    etre elle-meme absorbee par la regle 2 (cf. commentaire 'model -> 3d_model ->
    printed_3d_model' dans state_space_node_v3.py). Garde-fou anti-cycle par precaution, ne
    devrait jamais se declencher sur un rapport bien forme."""
    seen = set()
    current = raw_name
    while current in absorbed_to_final and current not in seen:
        seen.add(current)
        current = absorbed_to_final[current]
    return current


def flatten_case(prompt_name: str, llm: str, use_case: str, entry: dict) -> list[dict]:
    candidates = entry.get("candidates", {})
    if not isinstance(candidates, dict) or "error" in candidates:
        error_msg = candidates.get("error", "unknown") if isinstance(candidates, dict) else "unknown"
        return [{
            "prompt_name": prompt_name, "llm": llm, "use_case": use_case,
            "raw_entity": "", "raw_states": "", "lexname": "", "kept_lexical": "",
            "rejection_reason": f"ERROR: {error_msg}",
            "dedup_rule": "", "final_entity": "", "final_states": "",
        }]

    validation_report = entry.get("validation_report", {})
    state_space = entry.get("state_space", {})
    dedup_report = entry.get("state_space_dedup") or {"merged": {}, "ambiguous_not_merged": {}}

    # Reverse map : nom absorbe -> entite finale, toutes regles confondues -- permet de
    # resoudre les chaines de fusion par iteration (cf. _resolve_final_entity).
    absorbed_to_final: dict[str, str] = {}
    dedup_rule_for_final: dict[str, str] = {}
    for final_name, merge_entry in dedup_report.get("merged", {}).items():
        dedup_rule_for_final[final_name] = merge_entry.get("rule", "")
        for absorbed_name in merge_entry.get("absorbed", []):
            absorbed_to_final[absorbed_name] = final_name

    rows = []
    for raw_entity, raw_states in candidates.items():
        validation = validation_report.get(raw_entity, {})
        kept = validation.get("kept")
        row = {
            "prompt_name": prompt_name, "llm": llm, "use_case": use_case,
            "raw_entity": raw_entity,
            "raw_states": ";".join(raw_states) if isinstance(raw_states, list) else str(raw_states),
            "lexname": validation.get("f_lex") or "",
            "kept_lexical": kept if kept is not None else "",
            "rejection_reason": validation.get("reason", ""),
            "dedup_rule": "", "final_entity": "", "final_states": "",
        }
        if kept:
            final_entity = _resolve_final_entity(raw_entity, absorbed_to_final)
            row["final_entity"] = final_entity
            row["final_states"] = ";".join(state_space.get(final_entity, []))
            row["dedup_rule"] = dedup_rule_for_final.get(final_entity, "")
        rows.append(row)
    return rows


def flatten_run(data: dict) -> list[dict]:
    rows = []
    for prompt_name, per_model in data.items():
        for llm, per_case in per_model.items():
            for use_case, entry in per_case.items():
                rows.extend(flatten_case(prompt_name, llm, use_case, entry))
    return rows


def next_output_path() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(OUT_DIR) if re.match(r"state_space_run_\d+\.csv$", f)]
    next_idx = max(
        [int(re.match(r"state_space_run_(\d+)\.csv$", f).group(1)) for f in existing],
        default=0,
    ) + 1
    return os.path.join(OUT_DIR, f"state_space_run_{next_idx}.csv")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else _latest_run(STATE_SPACE_DIR)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = flatten_run(data)
    out_path = next_output_path()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Charge : {input_path}")
    print(f"{len(rows)} ligne(s) ecrite(s) -> {out_path}")