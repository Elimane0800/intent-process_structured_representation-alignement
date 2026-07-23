"""Audit du corpus BPMN complet (text_and_bpmn/bpmn/), en XML brut -- pas via PM4Py, qui
absorbe silencieusement les pertes qu'on cherche justement a quantifier :
1. standardLoopCharacteristics / multiInstanceLoopCharacteristics (boucles), jamais exposees
   par le modele d'objets BPMN de PM4Py.
2. sequenceFlow sans sourceRef/targetRef en attribut -- flux "casse" en apparence, mais
   potentiellement reconstructible via les balises <incoming>/<outgoing> des activites
   elles-memes (verifie sur M_g02/5 : la reconstruction echoue completement sur ce cas precis,
   32 flux sur 34 n'ont aucune trace de connectivite nulle part dans le fichier -- ce module
   verifie maintenant si c'est un cas isole ou general a l'ensemble du corpus).

Ne transforme rien, ne construit aucun DFG -- verifie uniquement l'ampleur du probleme."""

import json
import os
import xml.etree.ElementTree as ET

BPMN_ROOT = os.path.join(os.path.dirname(__file__), "text_and_bpmn", "bpmn")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "notes")


def audit_file(path: str) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    loop_count = 0
    dangling_flow_ids = []
    truly_broken_flow_ids = []

    # Index incoming/outgoing de TOUS les elements (pas seulement sequenceFlow) -- c'est la
    # ou la connectivite peut survivre meme quand sourceRef/targetRef manque sur le flux lui-meme.
    flow_source = {}
    flow_target = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "sequenceFlow":
            continue
        el_id = el.get("id")
        if el_id is None:
            continue
        for child in el:
            child_tag = child.tag.split("}")[-1]
            if child_tag == "outgoing" and child.text:
                flow_source[child.text.strip()] = el_id
            elif child_tag == "incoming" and child.text:
                flow_target[child.text.strip()] = el_id

    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("standardLoopCharacteristics", "multiInstanceLoopCharacteristics"):
            loop_count += 1
        elif tag == "sequenceFlow":
            flow_id = el.get("id")
            has_src_attr = el.get("sourceRef") is not None
            has_tgt_attr = el.get("targetRef") is not None
            if has_src_attr and has_tgt_attr:
                continue
            dangling_flow_ids.append(flow_id)

            src = el.get("sourceRef") or flow_source.get(flow_id)
            tgt = el.get("targetRef") or flow_target.get(flow_id)
            if not (src and tgt):
                truly_broken_flow_ids.append(flow_id)

    return {
        "loop_count": loop_count,
        "dangling_flow_ids": dangling_flow_ids,
        "truly_broken_flow_ids": truly_broken_flow_ids,
    }


if __name__ == "__main__":
    description_folders = sorted(
        d for d in os.listdir(BPMN_ROOT) if os.path.isdir(os.path.join(BPMN_ROOT, d))
    )

    report = {}
    total_files = 0
    files_with_loop = 0
    files_with_dangling = 0
    files_with_truly_broken = 0
    files_with_error = 0

    for desc_folder in description_folders:
        desc_path = os.path.join(BPMN_ROOT, desc_folder)
        bpmn_files = sorted(f for f in os.listdir(desc_path) if f.endswith(".bpmn2.xml"))
        report[desc_folder] = {}
        for bpmn_filename in bpmn_files:
            total_files += 1
            path = os.path.join(desc_path, bpmn_filename)
            try:
                result = audit_file(path)
            except Exception as e:
                result = {"error": str(e)}
                files_with_error += 1
            report[desc_folder][bpmn_filename] = result
            if result.get("loop_count"):
                files_with_loop += 1
            if result.get("dangling_flow_ids"):
                files_with_dangling += 1
            if result.get("truly_broken_flow_ids"):
                files_with_truly_broken += 1

    print(f"Total fichiers analyses : {total_files}")
    print(f"Fichiers avec au moins une boucle (perdue par PM4Py) : "
          f"{files_with_loop} ({100 * files_with_loop / total_files:.1f}%)")
    print(f"Fichiers avec au moins un flux sans attribut sourceRef/targetRef : "
          f"{files_with_dangling} ({100 * files_with_dangling / total_files:.1f}%)")
    print(f"Fichiers avec au moins un flux VRAIMENT irrecuperable "
          f"(ni attribut, ni incoming/outgoing nulle part) : "
          f"{files_with_truly_broken} ({100 * files_with_truly_broken / total_files:.1f}%)")
    if files_with_error:
        print(f"Fichiers en erreur de parsing XML : {files_with_error}")

    print("\n--- Detail : fichiers avec boucle ---")
    for desc, models in report.items():
        for fname, r in models.items():
            if r.get("loop_count"):
                print(f"  {desc}/{fname} -- {r['loop_count']} boucle(s)")

    print("\n--- Detail : fichiers avec flux VRAIMENT irrecuperables ---")
    for desc, models in report.items():
        for fname, r in models.items():
            if r.get("truly_broken_flow_ids"):
                n_dangling = len(r["dangling_flow_ids"])
                n_broken = len(r["truly_broken_flow_ids"])
                print(f"  {desc}/{fname} -- {n_broken}/{n_dangling} flux signales "
                      f"restent irrecuperables apres reconstruction")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "bpmn_silent_loss_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")