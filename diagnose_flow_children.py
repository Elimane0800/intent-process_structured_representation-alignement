"""Diagnostic definitif : pour chaque sequenceFlow deja signale comme 'casse' (attribut
sourceRef/targetRef absent), verifie s'il existe comme ELEMENT ENFANT a la place -- variante
BPMN valide, jamais rencontree dans les fichiers Signavio traites a la main jusqu'ici."""

import os
import xml.etree.ElementTree as ET

BPMN_ROOT = os.path.join(os.path.dirname(__file__), "text_and_bpmn", "bpmn")

FLAGGED = {
    "E_j01/10.bpmn2.xml": ["sid-F132AA3F-5C7D-415D-9474-87CAEED35E23"],
    "E_j05/9.bpmn2.xml": ["sid-279FE039-23E8-4B9F-980E-503B38712493"],
    "G_g03/0.bpmn2.xml": None,   # None = tous les sequenceFlow incomplets du fichier
    "M_g01/3.bpmn2.xml": None,
    "M_g02/5.bpmn2.xml": None,
    "M_j02/5.bpmn2.xml": None,
    "R_g01/0.bpmn2.xml": None,
    "R_j03/2.bpmn2.xml": None,
}

if __name__ == "__main__":
    total_truly_broken = 0
    total_child_element = 0

    for rel_path, ids in FLAGGED.items():
        path = os.path.join(BPMN_ROOT, rel_path)
        tree = ET.parse(path)
        root = tree.getroot()

        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if tag != "sequenceFlow":
                continue
            has_src_attr = el.get("sourceRef") is not None
            has_tgt_attr = el.get("targetRef") is not None
            if has_src_attr and has_tgt_attr:
                continue  # deja bien forme, pas concerne

            child_tags = {c.tag.split("}")[-1] for c in el}
            has_src_child = "sourceRef" in child_tags
            has_tgt_child = "targetRef" in child_tags

            if (has_src_attr or has_src_child) and (has_tgt_attr or has_tgt_child):
                total_child_element += 1
            else:
                total_truly_broken += 1
                print(f"VRAIMENT casse : {rel_path} -- {el.get('id')} "
                      f"(src attr={has_src_attr} child={has_src_child}, "
                      f"tgt attr={has_tgt_attr} child={has_tgt_child})")

    print()
    print(f"Total via element enfant (faux positif de mon script initial) : {total_child_element}")
    print(f"Total vraiment casse (ni attribut ni element)                  : {total_truly_broken}")