"""Verifie si la connectivite manquante des sequenceFlow (ni sourceRef ni targetRef en
attribut) peut etre reconstruite via les balises <incoming>/<outgoing> des activites
elles-memes -- hypothese testee sur M_g02/5.bpmn2.xml."""

import xml.etree.ElementTree as ET
import sys

path = sys.argv[1]
tree = ET.parse(path)
root = tree.getroot()

# Etape 1 : construire flow_id -> (source_activity_id, target_activity_id) via les balises
# <incoming>/<outgoing> de CHAQUE element (pas seulement sequenceFlow lui-meme)
flow_source = {}
flow_target = {}

for el in root.iter():
    tag = el.tag.split("}")[-1]
    if tag == "sequenceFlow":
        continue  # on ignore volontairement les attributs sourceRef/targetRef ici
    el_id = el.get("id")
    if el_id is None:
        continue
    for child in el:
        child_tag = child.tag.split("}")[-1]
        if child_tag == "outgoing" and child.text:
            flow_source[child.text.strip()] = el_id
        elif child_tag == "incoming" and child.text:
            flow_target[child.text.strip()] = el_id

# Etape 2 : pour chaque sequenceFlow sans sourceRef/targetRef en attribut, verifier si la
# reconstruction via incoming/outgoing comble le trou
total = 0
reconstructed = 0
still_broken = []

for el in root.iter():
    tag = el.tag.split("}")[-1]
    if tag != "sequenceFlow":
        continue
    total += 1
    flow_id = el.get("id")
    has_src_attr = el.get("sourceRef") is not None
    has_tgt_attr = el.get("targetRef") is not None
    if has_src_attr and has_tgt_attr:
        continue

    src = el.get("sourceRef") or flow_source.get(flow_id)
    tgt = el.get("targetRef") or flow_target.get(flow_id)

    if src and tgt:
        reconstructed += 1
    else:
        still_broken.append((flow_id, src, tgt))

print(f"Total sequenceFlow : {total}")
print(f"Reconstructibles via incoming/outgoing : {reconstructed}")
print(f"Toujours casses meme apres reconstruction : {len(still_broken)}")
for flow_id, src, tgt in still_broken:
    print(f"  {flow_id} -- src={src} tgt={tgt}")