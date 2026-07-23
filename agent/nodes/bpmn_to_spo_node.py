"""BPMN -> DFG -> SPO, en utilisant le parseur BPMN de PM4Py (couverture plus large que notre
parseur XML fait main) plutot que sa decouverte de DFG par simulation (rejetee : non
deterministe, cf. discussion -- "Loops and choices mean finite simulation cannot guarantee
every possible path").

Meme traversee que la version precedente : les gateways ne sont jamais reifies comme noeuds,
relation SPO fixee a "follows".

Extension du meme principe (ajoutee apres l'inspection des 61 VIOLATED du run dataset, cf.
violated_cases_dump.md) : les evenements Start/End ne sont plus reifies non plus. Un evenement
de depart/fin nomme ("Start", "end process", ou le titre du processus porte par l'evenement
initial) n'est pas une activite -- c'est un delimiteur de cycle de vie, exactement comme un
gateway est un delimiteur de routage. Les faire entrer dans le matching produisait des
appariements absurdes a la source de plusieurs faux VIOLATED (ex. l'evenement de depart
"Printing a 3D model" matche a 0.85 sur l'etat 3d_model.printed, accusant trois desordres
inexistants sur un seul cas). Le critere est le TYPE de noeud PM4Py (StartEvent/EndEvent, un
test d'instance), jamais une liste de labels a maintenir -- meme discipline que le rejet des
listes ouvertes partout ailleurs dans le pipeline. Les evenements intermediaires (timers,
messages) restent reifies : certains portent un vrai contenu d'activite, les filtrer serait un
jugement de contenu, pas de structure -- limite documentee, a reexaminer sur donnees si les
artefacts residuels le justifient.

Consequence structurelle assumee : les aretes partant d'un StartEvent ou arrivant a un EndEvent
disparaissent du DFG (rien en amont d'un start, rien en aval d'un end -- la traversee n'a rien
a relier a travers eux). L'atteignabilite entre activites reelles est inchangee : un
source/puits retire ne modifie aucun chemin entre les autres noeuds.
"""

import json
import sys
import types

# Contournement : pm4py importe cvxopt.glpk au chargement (pour ses solveurs d'alignement/
# conformance-checking bases sur la programmation lineaire), meme si on ne s'en sert jamais ici
# -- on utilise uniquement read_bpmn() et le modele d'objets BPMN, jamais ces solveurs. Sur
# certains systemes (ex. macOS, cvxopt compile sans le support GLPK), "from cvxopt import glpk"
# echoue au chargement de pm4py et bloque meme le simple parsing BPMN qu'on veut faire.
# Enregistrer un module factice AVANT l'import de pm4py suffit a satisfaire l'import -- sans
# danger pour notre usage, puisqu'aucune fonction GLPK n'est jamais reellement appelee.
try:
    from cvxopt import glpk  # noqa: F401 -- si ca marche deja, rien a faire
except ImportError:
    import cvxopt
    _glpk_stub = types.ModuleType("cvxopt.glpk")
    sys.modules["cvxopt.glpk"] = _glpk_stub
    cvxopt.glpk = _glpk_stub

import pm4py

GATEWAY_TYPES = (
    pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway,
    pm4py.objects.bpmn.obj.BPMN.ParallelGateway,
    pm4py.objects.bpmn.obj.BPMN.InclusiveGateway,
    pm4py.objects.bpmn.obj.BPMN.EventBasedGateway,
)

# Noeuds traverses sans jamais etre reifies dans le DFG : les gateways (routage) et les
# evenements Start/End (delimiteurs de cycle de vie) -- cf. docstring de tete. Les classes de
# base StartEvent/EndEvent couvrent leurs sous-classes PM4Py (NormalStartEvent,
# MessageStartEvent, NormalEndEvent, ...) via isinstance.
PASSTHROUGH_TYPES = GATEWAY_TYPES + (
    pm4py.objects.bpmn.obj.BPMN.StartEvent,
    pm4py.objects.bpmn.obj.BPMN.EndEvent,
)


def parse_bpmn(path: str):
    """Parse via PM4Py, apres retrait des elements de pure documentation (<association> et
    <textAnnotation>) du XML.

    Double motivation, diagnostiquee sur les fichiers reels du corpus (run dataset v1) :

    1. CRASH ('str' object has no attribute 'add_in_arc'/'add_out_arc', 24/223 fichiers,
       identique pour les trois modeles LLM) : Signavio attache des commentaires aux ARETES
       (association sequenceFlow -> textAnnotation, BPMN valide). PM4Py n'enregistre que les
       noeuds dans son dictionnaire de references ; la reference vers un sequenceFlow reste une
       chaine brute, sur laquelle Association.__init__ appelle add_out_arc -> AttributeError.
       Verifie sur E_j02/5 : parse OK apres retrait, 22 noeuds / 26 flux recuperes.

    2. POLLUTION SILENCIEUSE (fichiers qui parsaient sans erreur, ex. E_j02/1 : 5 TextAnnotation
       reifiees en noeuds + 5 Association reifiees en flux) : resolve_dfg suit get_flows() sans
       distinguer le type de flux -- une association task -> annotation devenait une arete DFG
       vers une pseudo-activite (souvent de label vide, mais pas toujours), jamais un flux de
       controle reel.

    Les deux problemes ont la meme cause (artefacts de documentation traites comme du flux) et
    le meme fix : les retirer avant parse. Aucune perte semantique pour le DFG -- une annotation
    n'est jamais du controle de flux. Le retrait se fait sur une copie temporaire, le fichier
    source n'est jamais modifie."""
    import os
    import tempfile
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1].lower()
            if tag in ("association", "textannotation"):
                parent.remove(child)
                removed += 1
    if removed == 0:
        return pm4py.read_bpmn(path)
    print(f"    [bpmn_to_spo] {removed} element(s) de documentation (association/"
          f"textAnnotation) retire(s) avant parse -- cf. docstring de parse_bpmn")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".bpmn2.xml", delete=False) as f:
            tree.write(f, xml_declaration=True, encoding="utf-8")
            tmp_path = f.name
        return pm4py.read_bpmn(tmp_path)
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)


def resolve_dfg(bpmn) -> list[dict]:
    nodes = {n.get_id(): n for n in bpmn.get_nodes()}
    outgoing = {}
    for flow in bpmn.get_flows():
        outgoing.setdefault(flow.get_source().get_id(), []).append(flow)

    def next_activities(node_id: str, ctx: list[str], visited_passthrough: frozenset[str]):
        results = []
        for flow in outgoing.get(node_id, []):
            target = flow.get_target()
            step_ctx = ctx + ([flow.get_name()] if flow.get_name() else [])
            if isinstance(target, PASSTHROUGH_TYPES):
                if target.get_id() in visited_passthrough:
                    continue
                results.extend(
                    next_activities(target.get_id(), step_ctx + [type(target).__name__],
                                     visited_passthrough | {target.get_id()})
                )
            else:
                results.append((target.get_id(), step_ctx))
        return results

    edges = []
    for node_id, node in nodes.items():
        if isinstance(node, PASSTHROUGH_TYPES):
            continue
        for target_id, ctx in next_activities(node_id, [], frozenset()):
            edges.append({
                "source_id": node_id,
                "source_label": node.get_name(),
                "target_id": target_id,
                "target_label": nodes[target_id].get_name(),
                "gateway_context": ctx,
            })
    return edges


def dfg_to_spo(edges: list[dict]) -> list[dict]:
    return [
        {
            "subject": e["source_label"],
            "subject_id": e["source_id"],
            "predicate": "follows",
            "object": e["target_label"],
            "object_id": e["target_id"],
        }
        for e in edges
    ]


if __name__ == "__main__":
    import os
    import re

    BPMN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "text_and_bpmn", "bpmn")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bpmn_dfg_spo")

    # Confirme via audit_bpmn_silent_losses.py + verify_reconstruction.py sur les 223 fichiers
    # du corpus : ces 8 ont au moins un sequenceFlow sans sourceRef/targetRef en attribut ET
    # sans trace de connectivite recuperable via incoming/outgoing nulle part dans le fichier
    # (100% des flux signales irrecuperables sur chacun des 8, pas un echantillon partiel).
    # Exclus explicitement plutot que de produire un DFG partiel/trompeur silencieusement.
    EXCLUDED_MALFORMED = {
        "E_j01/10.bpmn2.xml",
        "E_j05/9.bpmn2.xml",
        "G_g03/0.bpmn2.xml",
        "M_g01/3.bpmn2.xml",
        "M_g02/5.bpmn2.xml",
        "M_j02/5.bpmn2.xml",
        "R_g01/0.bpmn2.xml",
        "R_j03/2.bpmn2.xml",
    }

    description_folders = sorted(
        d for d in os.listdir(BPMN_ROOT) if os.path.isdir(os.path.join(BPMN_ROOT, d))
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for desc_folder in description_folders:
        desc_path = os.path.join(BPMN_ROOT, desc_folder)
        bpmn_files = sorted(f for f in os.listdir(desc_path) if f.endswith(".bpmn2.xml"))
        results[desc_folder] = {}

        for bpmn_filename in bpmn_files:
            key = f"{desc_folder}/{bpmn_filename}"
            if key in EXCLUDED_MALFORMED:
                results[desc_folder][bpmn_filename] = {
                    "excluded": "connectivite structurelle irrecuperable (verifie via "
                                 "audit_bpmn_silent_losses.py + verify_reconstruction.py)"
                }
                print(f"[{key}] EXCLU -- connectivite irrecuperable, cf. resultats/notes")
                continue

            bpmn_path = os.path.join(desc_path, bpmn_filename)
            try:
                bpmn = parse_bpmn(bpmn_path)
                dfg = resolve_dfg(bpmn)
                spo = dfg_to_spo(dfg)
                results[desc_folder][bpmn_filename] = {"dfg": dfg, "spo": spo}
                print(f"[{desc_folder}/{bpmn_filename}] dfg_edges={len(dfg)} spo_triples={len(spo)}")
            except Exception as e:
                results[desc_folder][bpmn_filename] = {"error": str(e)}
                print(f"[{desc_folder}/{bpmn_filename}] ERROR: {e}")

    existing = [f for f in os.listdir(RESULTS_DIR) if re.match(r"run_\d+\.json$", f)]
    next_idx = max([int(re.match(r"run_(\d+)\.json$", f).group(1)) for f in existing], default=0) + 1
    out_path = os.path.join(RESULTS_DIR, f"run_{next_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")