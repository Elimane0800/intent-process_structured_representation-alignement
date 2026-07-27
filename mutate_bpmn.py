"""Etude de perturbation controlee : generation de mutants BPMN avec verite terrain par
construction. Repond au manque central identifie pour AAAI (aucune precision/rappel sur les
verdicts) sans aucune annotation humaine : chaque mutant sait ce que la methode DOIT detecter
(operateurs nuisibles), NE DOIT PAS flaguer (operateurs preservant le comportement -- la claim
"G_process a le droit d'etre plus riche" devient un resultat teste), ou NE PEUT PAS detecter
par construction (angle mort documente, pas cache).

Methodologie standard (mutation testing / injection de bruit en conformance checking),
continuation directe du test de controle manuel deja fait (pipeline_observations.md section 5,
arete retiree sur mistral-nemotron/1_1) et du point ouvert n.11 (negatifs synthetiques).

Taxonomie -- 6 operateurs, 3 classes d'attente :

  NUISIBLES (expected="required" -- une detection est attendue) :
  - remove_activity      : une activite exigee par le texte disparait du modele (pont recable,
                            le graphe reste connexe). Detection attendue : l'etat ancre sur
                            cette activite perd sa couverture -> UNRESOLVABLE (ou cascade).
  - swap_labels          : deux activites sequentiellement ordonnees echangent leurs labels --
                            equivalent exact d'une inversion d'ordre, sans toucher aux flux
                            (bien plus robuste que recabler le XML). Detection attendue :
                            VIOLATED sur les preconditions qui exigeaient l'ordre original.
  - cross_case_replace   : le label d'une activite est remplace par une activite d'une AUTRE
                            description (intrus semantique). Detection attendue : l'etat perd
                            son ancre (score bas -> demotion/UNRESOLVABLE).

  ANGLE MORT ASSUME (expected="blind_spot" -- 0% de detection attendu, par construction) :
  - rewire_gateway       : XOR <-> AND. La relation SPO est volontairement "follows" generique,
                            gateway_context hors du triplet (decision actee, bpmn_to_spo) --
                            cette classe de defaut est HORS PERIMETRE de l'abstraction par
                            atteignabilite. L'inclure et rapporter 0% avec l'explication est
                            plus honnete (et plus solide face aux reviewers) que l'omettre.

  PRESERVANT LE COMPORTEMENT (expected="forbidden" -- toute nouvelle detection est un FAUX
  POSITIF) :
  - insert_activity      : une activite supplementaire inseree sur une arete A->B (le modele
                            s'enrichit -- teste la claim centrale "chemin, jamais arete
                            directe" : les preconditions via A->New->B doivent rester
                            SATISFIED).
  - shuffle_xml          : permutation de l'ordre des elements freres dans le XML (ordre
                            documentaire, sans semantique BPMN). Verdicts attendus strictement
                            identiques -- test d'invariance pur.

Sortie : pour chaque fichier de base, des mutants <base>__<operator>_<n>.bpmn2.xml + un
manifest.json qui porte la verite terrain (operateur, parametres exacts, attente). SEPARATION
STRICTE d'avec Zenodo : les mutants n'ont pas de note experte et n'entrent JAMAIS dans une
correlation -- validite interne uniquement, tableau separe dans le papier.

Correctif du biais de ciblage (post-perturbation-study v1, cf. dataset_run_v2_observations.md
section 8) : les operateurs REQUIRED (remove_activity/swap_labels/cross_case_replace)
choisissaient une activite au hasard parmi TOUTES les activites nommees du BPMN, sans verifier
si l'etat qu'elle ancre est effectivement REFERENCE PAR UN GUARD de G. Une part inconnue des
mutants tombait donc sur une zone du graphe non verifiee par aucune precondition --
structurellement indetectable, independamment de la qualite du mecanisme de verification.
Preuve du biais : sur le premier run, les detections se concentraient sur 1-2 fichiers de base
par modele au lieu d'etre dispersees -- signe direct que beaucoup de mutations tombaient hors
zone verifiee.

Fix : restreindre le pool de candidats de ces trois operateurs a l'UNION, sur les modeles
fournis, des activites dont l'etat apparie apparait comme terme OU cible d'au moins une arete
de reference_graph. Volontairement une UNION et pas un jeu de mutants par modele : a ce stade
du developpement, les 3 modeles sont testes ensemble (comme partout ailleurs dans le projet),
pas comme une comparaison inter-modeles -- un seul manifest, un seul jeu de mutants, chacun
toujours evalue contre les 3 modeles, exactement comme avant le correctif. Seule la SELECTION
de l'activite mutee change ; l'evaluation et le reporting restent identiques.

Determinisme : seed fixe par (fichier, operateur, index) -- memes mutants a chaque execution,
reproductibles pour le papier.
"""

import copy
import json
import os
import random
import re
import xml.etree.ElementTree as ET

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
ET.register_namespace("", BPMN_NS)

# Tags d'activites considerees comme mutables (jamais les evenements ni les gateways -- les
# operateurs portent sur les activites, l'unite que le matching consomme).
ACTIVITY_TAGS = {"task", "usertask", "servicetask", "manualtask", "scripttask", "sendtask",
                  "receivetask", "businessruletask", "calltask"}
GATEWAY_TAGS = {"exclusivegateway", "parallelgateway", "inclusivegateway", "eventbasedgateway"}


def _guard_relevant_activities(model: str, base_key: str, v2_results_dir: str) -> set[str] | None:
    """Charge le run v2 de reference pour (modele, base_key) et retourne l'ensemble des labels
    d'activite dont l'etat apparie apparait comme terme OU cible d'au moins une arete de
    reference_graph -- le pool de candidats legitime pour les operateurs required (cf. docstring
    de tete). Retourne None (pas un ensemble vide) si le run de reference est absent ou
    inexploitable -- distingue explicitement "aucune activite pertinente trouvee" (ensemble
    vide, cas reel possible) de "impossible a determiner" (None, l'appelant doit sauter, jamais
    supposer un comportement par defaut)."""
    path = os.path.join(v2_results_dir, model, base_key.replace("/", "__") + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    reference_graph = state.get("reference_graph")
    matches = state.get("matches")
    if not reference_graph or not reference_graph.get("edges") or not matches:
        return None
    guard_milestones = {e["from"] for e in reference_graph["edges"]} | \
                       {e["to"] for e in reference_graph["edges"]}
    return {
        activity for activity, m in matches.items()
        if isinstance(m, dict) and m.get("match") in guard_milestones
    }


def _guard_relevant_activities_union(models: list[str], base_key: str,
                                      v2_results_dir: str) -> set[str] | None:
    """Union, sur tous les modeles fournis, des activites guard-pertinentes pour base_key --
    UN seul mutant par variante est genere a partir de cette union (pas un jeu par modele) :
    on veut juste eviter de muter une activite hors de toute zone verifiee, pas produire une
    analyse comparative modele par modele (hors scope a ce stade du developpement -- 3 modeles
    testes ensemble, comme partout ailleurs dans le projet, pas 3 etudes separees).

    None seulement si AUCUN modele n'a de run v2 exploitable pour cette base (rien a cibler,
    jamais suppose non restreint). Si certains modeles ont une reference et d'autres non, l'union
    porte sur ceux qui en ont -- les autres sont simplement absents, non bloquants."""
    found_any = False
    union: set[str] = set()
    for model in models:
        allowed = _guard_relevant_activities(model, base_key, v2_results_dir)
        if allowed is not None:
            found_any = True
            union |= allowed
    return union if found_any else None



def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _iter_with_parent(root):
    for parent in root.iter():
        for child in list(parent):
            yield parent, child


def _load(path: str) -> ET.ElementTree:
    return ET.parse(path)


def _elements(root, tags: set) -> list:
    return [el for el in root.iter() if _local(el.tag) in tags]


def _flows(root) -> list:
    return [el for el in root.iter() if _local(el.tag) == "sequenceflow"]


def _named_activities(root) -> list:
    return [el for el in _elements(root, ACTIVITY_TAGS)
            if (el.get("name") or "").strip()]


def _save_mutant(tree: ET.ElementTree, out_path: str) -> None:
    tree.write(out_path, xml_declaration=True, encoding="utf-8")


# --- Operateurs nuisibles -------------------------------------------------------------------

def mutate_remove_activity(tree: ET.ElementTree, rng: random.Random,
                            allowed_labels: set[str] | None = None) -> dict | None:
    """Supprime une activite nommee a exactement 1 flux entrant et 1 sortant (pont propre,
    deterministe -- pas de choix semantique sur le recablage), recable entrant -> cible du
    sortant, supprime les deux flux originaux et le noeud. Les blocs <incoming>/<outgoing> des
    voisins sont mis a jour pour rester coherents (certains parseurs les lisent).

    allowed_labels : si fourni, restreint le pool aux activites dont le label y figure --
    correctif du biais de ciblage (cf. docstring de tete), jamais un filtre silencieux : si la
    restriction vide le pool, retourne None comme n'importe quel autre cas non applicable."""
    root = tree.getroot()
    flows = _flows(root)
    by_source, by_target = {}, {}
    for f in flows:
        by_source.setdefault(f.get("sourceRef"), []).append(f)
        by_target.setdefault(f.get("targetRef"), []).append(f)

    candidates = [a for a in _named_activities(root)
                  if len(by_target.get(a.get("id"), [])) == 1
                  and len(by_source.get(a.get("id"), [])) == 1]
    # Garde multi-pools : jamais une activite encore referencee par un messageFlow -- la
    # supprimer laisserait une reference pendante que PM4Py transforme en chaine brute
    # (exactement le bug add_in_arc/add_out_arc diagnostique par ailleurs). Decouvert par test
    # sur E_j03/3 (pools Organisation/Insurance Company relies par messages).
    msg_refs = {f.get("sourceRef") for f in root.iter() if _local(f.tag) == "messageflow"} | \
               {f.get("targetRef") for f in root.iter() if _local(f.tag) == "messageflow"}
    candidates = [a for a in candidates if a.get("id") not in msg_refs]
    if allowed_labels is not None:
        candidates = [a for a in candidates if (a.get("name") or "").strip() in allowed_labels]
    if not candidates:
        return None
    victim = rng.choice(candidates)
    vid, vname = victim.get("id"), victim.get("name")
    incoming, outgoing = by_target[vid][0], by_source[vid][0]
    new_target = outgoing.get("targetRef")

    incoming.set("targetRef", new_target)  # le pont : entrant pointe sur la cible du sortant
    for parent, child in _iter_with_parent(root):
        if child is outgoing or child is victim:
            parent.remove(child)
    # Coherence des balises <incoming>/<outgoing> du noeud cible du pont
    for el in root.iter():
        if el.get("id") == new_target:
            for sub in list(el):
                if _local(sub.tag) == "incoming" and (sub.text or "").strip() == outgoing.get("id"):
                    sub.text = incoming.get("id")
    return {"removed_activity": vname, "removed_id": vid}


def mutate_swap_labels(tree: ET.ElementTree, rng: random.Random,
                        allowed_labels: set[str] | None = None) -> dict | None:
    """Echange les labels de deux activites nommees reliees par un chemin sequentiel court
    (A -> B direct, ou A -> gateway -> B) -- l'ordre des deux activites nommees s'inverse
    exactement, structure de flux intacte. Prefere une paire directe si disponible.

    allowed_labels : si fourni, ne retient une paire que si AU MOINS UN des deux labels y
    figure (pas les deux -- l'echange affecte l'ancrage des deux etats, un seul suffit a rendre
    la mutation potentiellement detectable ; exiger les deux videraient le pool inutilement)."""
    root = tree.getroot()
    acts = {a.get("id"): a for a in _named_activities(root)}
    flows = _flows(root)
    direct_pairs = [(f.get("sourceRef"), f.get("targetRef")) for f in flows
                    if f.get("sourceRef") in acts and f.get("targetRef") in acts]
    if not direct_pairs:
        # via un gateway unique : A -> g -> B
        gw_ids = {g.get("id") for g in _elements(root, GATEWAY_TAGS)}
        into_gw = {}
        for f in flows:
            if f.get("targetRef") in gw_ids and f.get("sourceRef") in acts:
                into_gw.setdefault(f.get("targetRef"), []).append(f.get("sourceRef"))
        for f in flows:
            if f.get("sourceRef") in gw_ids and f.get("targetRef") in acts:
                for a in into_gw.get(f.get("sourceRef"), []):
                    direct_pairs.append((a, f.get("targetRef")))
    direct_pairs = [(a, b) for a, b in direct_pairs if a != b]
    if allowed_labels is not None:
        direct_pairs = [
            (a, b) for a, b in direct_pairs
            if (acts[a].get("name") or "").strip() in allowed_labels
            or (acts[b].get("name") or "").strip() in allowed_labels
        ]
    if not direct_pairs:
        return None
    a_id, b_id = rng.choice(sorted(direct_pairs))
    a, b = acts[a_id], acts[b_id]
    a_name, b_name = a.get("name"), b.get("name")
    a.set("name", b_name)
    b.set("name", a_name)
    return {"swapped": [a_name, b_name], "ids": [a_id, b_id]}


def mutate_cross_case_replace(tree: ET.ElementTree, rng: random.Random,
                               foreign_labels: list[str],
                               allowed_labels: set[str] | None = None) -> dict | None:
    """Remplace le label d'une activite par un label d'activite d'une AUTRE description --
    intrus semantiquement etranger au texte de ce cas.

    allowed_labels : si fourni, restreint le pool de victimes possibles (l'etat qui PERD son
    ancre doit etre guard-pertinent, sinon la mutation est indetectable par construction)."""
    root = tree.getroot()
    candidates = _named_activities(root)
    if allowed_labels is not None:
        candidates = [a for a in candidates if (a.get("name") or "").strip() in allowed_labels]
    if not candidates or not foreign_labels:
        return None
    victim = rng.choice(candidates)
    original = victim.get("name")
    intruder = rng.choice(sorted(set(foreign_labels) - {original}))
    victim.set("name", intruder)
    return {"replaced_activity": original, "intruder_label": intruder, "id": victim.get("id")}


def mutate_rewire_gateway(tree: ET.ElementTree, rng: random.Random) -> dict | None:
    """XOR <-> AND sur un gateway de branchement. Angle mort ASSUME de la methode (relation
    'follows' insensible au type de gateway, decision actee) -- 0% de detection attendu."""
    root = tree.getroot()
    gws = _elements(root, {"exclusivegateway", "parallelgateway"})
    if not gws:
        return None
    victim = rng.choice(gws)
    old = _local(victim.tag)
    new_tag = ("{%s}parallelGateway" % BPMN_NS) if old == "exclusivegateway" \
        else ("{%s}exclusiveGateway" % BPMN_NS)
    victim.tag = new_tag
    return {"gateway_id": victim.get("id"), "from": old, "to": _local(new_tag)}


# --- Operateurs preservant le comportement (controles negatifs) ------------------------------

_NEUTRAL_LABELS = [
    # Labels d'activites administratives plausibles dans n'importe quel processus, jamais
    # exiges par un texte -- l'enrichissement legitime type que la methode ne doit pas punir.
    "Document the step", "Archive the record", "Log the activity", "Update the tracking sheet",
]


def mutate_insert_activity(tree: ET.ElementTree, rng: random.Random) -> dict | None:
    """Insere une activite neutre sur une arete A->B existante (A->New->B). Le modele devient
    plus riche que le texte -- aucune nouvelle detection ne doit apparaitre (la verification
    par CHEMIN doit absorber l'intermediaire, c'est la claim centrale testee)."""
    root = tree.getroot()
    process = next((el for el in root.iter() if _local(el.tag) == "process"), None)
    flows = [f for f in _flows(root)]
    if process is None or not flows:
        return None
    flow = rng.choice(sorted(flows, key=lambda f: f.get("id") or ""))
    label = rng.choice(_NEUTRAL_LABELS)
    new_id = f"mut_task_{rng.randrange(10**9)}"
    new_flow_id = f"mut_flow_{rng.randrange(10**9)}"
    old_target = flow.get("targetRef")

    task = ET.SubElement(process, "{%s}task" % BPMN_NS,
                          {"id": new_id, "name": label, "completionQuantity": "1",
                           "startQuantity": "1"})
    ET.SubElement(process, "{%s}sequenceFlow" % BPMN_NS,
                   {"id": new_flow_id, "sourceRef": new_id, "targetRef": old_target})
    flow.set("targetRef", new_id)
    return {"inserted_label": label, "on_flow": flow.get("id"), "before": old_target}


def mutate_shuffle_xml(tree: ET.ElementTree, rng: random.Random) -> dict | None:
    """Permute l'ordre des elements freres de chaque <process> -- ordre purement documentaire
    en BPMN, aucune semantique. Verdicts attendus strictement identiques (invariance)."""
    root = tree.getroot()
    shuffled = 0
    for el in root.iter():
        if _local(el.tag) == "process":
            children = list(el)
            if len(children) > 1:
                rng.shuffle(children)
                for c in list(el):
                    el.remove(c)
                for c in children:
                    el.append(c)
                shuffled += len(children)
    return {"shuffled_elements": shuffled} if shuffled else None


# --- Orchestration ---------------------------------------------------------------------------

OPERATORS = {
    "remove_activity": (mutate_remove_activity, "required"),
    "swap_labels": (mutate_swap_labels, "required"),
    "cross_case_replace": (mutate_cross_case_replace, "required"),
    "rewire_gateway": (mutate_rewire_gateway, "blind_spot"),
    "insert_activity": (mutate_insert_activity, "forbidden"),
    "shuffle_xml": (mutate_shuffle_xml, "forbidden"),
}
REQUIRED_OPS = {"remove_activity", "swap_labels", "cross_case_replace"}


def collect_activity_labels(path: str) -> list[str]:
    root = _load(path).getroot()
    return sorted({a.get("name").strip() for a in _named_activities(root)})


def generate_mutants(base_files: dict[str, str], out_dir: str, v2_results_dir: str,
                      models: list[str], variants_per_operator: int = 3) -> list[dict]:
    """base_files : {"desc/fichier.bpmn2.xml": chemin} -- genere variants_per_operator mutants
    par operateur et par fichier (seeds distincts, UN SEUL jeu de mutants, pas un par modele),
    ecrit le manifest de verite terrain. Les labels etrangers de cross_case_replace viennent
    des AUTRES fichiers fournis.

    Correctif du biais de ciblage (cf. docstring de tete) : les operateurs required sont
    restreints a l'union, sur `models`, des activites guard-pertinentes -- evite de muter une
    activite hors de toute zone verifiee par un guard, sans dupliquer les mutants par modele
    (chaque mutant reste evalue contre les 3 modeles, exactement comme avant)."""
    os.makedirs(out_dir, exist_ok=True)
    labels_by_case = {key: collect_activity_labels(p) for key, p in base_files.items()}
    manifest = []

    for key, path in sorted(base_files.items()):
        desc = key.split("/")[0]
        foreign = sorted({l for k, ls in labels_by_case.items()
                          if k.split("/")[0] != desc for l in ls})
        safe = key.replace("/", "__").replace(".bpmn2.xml", "")

        allowed = _guard_relevant_activities_union(models, key, v2_results_dir)
        if allowed is None:
            print(f"  [mutate] {key} : aucun run v2 exploitable pour aucun des {len(models)} "
                  f"modeles -- operateurs required non generes pour ce cas (jamais suppose "
                  f"non restreint)")
        elif not allowed:
            print(f"  [mutate] {key} : 0 activite guard-pertinente trouvee (union de "
                  f"{len(models)} modele(s)) -- aucun mutant required genere pour ce cas")

        for op_name, (op_fn, expected) in OPERATORS.items():
            needs_restriction = op_name in REQUIRED_OPS
            if needs_restriction and allowed is None:
                for i in range(variants_per_operator):
                    manifest.append({"base": key, "operator": op_name, "variant": i,
                                      "expected": expected, "applicable": False,
                                      "reason": "no_v2_reference"})
                continue

            for i in range(variants_per_operator):
                rng = random.Random(f"{key}|{op_name}|{i}")  # deterministe et reproductible
                tree = _load(path)
                if op_name == "cross_case_replace":
                    details = op_fn(tree, rng, foreign, allowed if needs_restriction else None)
                elif needs_restriction:
                    details = op_fn(tree, rng, allowed)
                else:
                    details = op_fn(tree, rng)
                if details is None:
                    manifest.append({"base": key, "operator": op_name, "variant": i,
                                      "expected": expected, "applicable": False})
                    continue
                mutant_name = f"{safe}__{op_name}_{i}.bpmn2.xml"
                _save_mutant(tree, os.path.join(out_dir, mutant_name))
                manifest.append({"base": key, "operator": op_name, "variant": i,
                                  "expected": expected, "applicable": True,
                                  "mutant_file": mutant_name, "details": details})

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    applicable = sum(1 for m in manifest if m["applicable"])
    print(f"\n[mutate] {applicable} mutant(s) generes ({len(manifest) - applicable} "
          f"non applicables), manifest: {os.path.join(out_dir, 'manifest.json')}")
    return manifest


if __name__ == "__main__":
    # Exemple sur le sous-ensemble de developpement -- adapter BPMN_ROOT/BASE_FILES en usage
    # reel (les chemins ci-dessous correspondent a l'arborescence du projet).
    BPMN_ROOT = os.path.join(os.path.dirname(__file__), "text_and_bpmn", "bpmn")
    OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "mutants")
    V2_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "dataset_runs_v2")
    MODELS = ["llama-3.1-8b", "mistral-nemotron", "gpt-oss-20b"]
    BASE_FILES = {
        f"{desc}/{fname}": os.path.join(BPMN_ROOT, desc, fname)
        for desc, fname in [
            ("E_j02", "1.bpmn2.xml"), ("E_j02", "5.bpmn2.xml"), ("E_j03", "3.bpmn2.xml"),
            ("M_g01", "10.bpmn2.xml"), ("R_j02", "6.bpmn2.xml"), ("V_k09", "2.bpmn2.xml"),
            ("X_g01", "0.bpmn2.xml"),
        ]
    }
    generate_mutants(BASE_FILES, OUT_DIR, V2_RESULTS_DIR, MODELS)