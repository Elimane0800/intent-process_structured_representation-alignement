"""Orchestration LangGraph du pipeline complet, sur un run (un texte, un cas, un fichier BPMN).

Topologie :

  START -> state_space -> precondition -> graph_construction --\\
    |            |                                                \\
    |            \\-----------------------------------------------> alignment -> report -> END
    |             \\-> state_matching --------------------------> /
  START -> bpmn_to_spo ------------------------------------------/

Deux jointures :
- state_matching attend state_space (U) ET bpmn_to_spo (SPO) -- SYMETRIQUE (les deux
  predecesseurs sont a 1 saut de START), pas de defer necessaire, teste et confirme.
- alignment attend graph_construction (G, 3 sauts) ET state_matching (2 sauts) -- ASYMETRIQUE,
  `defer=True` necessaire (teste et confirme -- et empiler un second defer sur state_matching
  casserait la synchronisation, egalement teste).

Chaque node ci-dessous est un wrapper autour d'une fonction deja livree et testee ailleurs
(agent/nodes/) -- ce fichier ne reimplemente aucune logique, il cable seulement l'etat entre
elles. Chaque wrapper capture ses exceptions dans state['errors'] plutot que de faire planter
tout le run -- coherent avec la discipline actee dans agent/state.py.
"""

from langgraph.graph import StateGraph, START, END

from agent.state import PipelineState
from agent.models.base_llm import get_llm

from agent.nodes.state_space_node_v3 import (
    generate_candidates, filter_state_space, generate_exclusive_groups, deduplicate_state_space,
)
from agent.prompt.state_space_prompt_permissive import (
    STATE_SPACE_PROMPT_PERMISSIVE,
    STATE_SPACE_PROMPT_PERMISSIVE_FEWSHOT,
    STATE_SPACE_PROMPT_PERMISSIVE_TWOSHOT,
    STATE_SPACE_PROMPT_PERMISSIVE_COT,
)

from agent.nodes.precondition_node_with_retry import run_naive_with_retry
from agent.prompt.precondition_prompt import (
    PRECONDITION_PROMPT,
    PRECONDITION_PROMPT_FEWSHOT,
    PRECONDITION_PROMPT_TWOSHOT,
    PRECONDITION_PROMPT_COT,
)

from agent.nodes.graph_node import build_graph
from agent.nodes.bpmn_to_spo_node import parse_bpmn, resolve_dfg, dfg_to_spo
from agent.nodes.bpmn_guards_node import compute_activity_guards
from agent.nodes.state_matching_node import match_dfg_to_u
from agent.nodes.alignment_node import check_alignment
from agent.nodes.report_node import build_report


_STATE_SPACE_PROMPTS = {
    "zero_shot": STATE_SPACE_PROMPT_PERMISSIVE,
    "one_shot": STATE_SPACE_PROMPT_PERMISSIVE_FEWSHOT,
    "two_shot": STATE_SPACE_PROMPT_PERMISSIVE_TWOSHOT,
    "cot": STATE_SPACE_PROMPT_PERMISSIVE_COT,
}

_PRECONDITION_PROMPTS = {
    "zero_shot": PRECONDITION_PROMPT,
    "one_shot": PRECONDITION_PROMPT_FEWSHOT,
    "two_shot": PRECONDITION_PROMPT_TWOSHOT,
    "cot": PRECONDITION_PROMPT_COT,
}

from agent.config import DEFAULT_REPORT_MODEL_KEY


def _with_error_capture(node_name: str, fn):
    """Une exception dans un node devient une entree dans state['errors'], jamais un crash de
    tout le run. Ne renvoie QUE la nouvelle erreur (pas l'historique complet) -- le champ
    'errors' utilise un reducteur operator.add (agent/state.py) qui fusionne automatiquement les
    ecritures de plusieurs nodes executes en parallele ; recopier l'historique ici produirait des
    doublons a chaque fusion. Les noeuds en aval recoivent un etat partiel en cas d'echec -- a
    eux de traiter un champ manquant comme un signal legitime (deja le cas partout ailleurs dans
    ce pipeline: UNRESOLVED, UNRESOLVABLE, etc. sont des statuts, pas des exceptions)."""
    def wrapped(state: PipelineState) -> dict:
        try:
            return fn(state)
        except Exception as e:
            return {"errors": [f"{node_name}: {e}"]}
    return wrapped


# --- Branche gauche : texte -> U -> Pre -> G ---

def state_space_node(state: PipelineState) -> dict:
    prompt_name = state.get("state_space_prompt_name", "zero_shot")
    template = _STATE_SPACE_PROMPTS[prompt_name]
    llm = get_llm(temperature=0, model=state.get("state_space_model")) if state.get("state_space_model") \
        else get_llm(temperature=0)
    candidates = generate_candidates(state["text"], llm, template)
    filtered, report = filter_state_space(candidates)
    # Etape A2 (Protocol 3.2) -- second appel LLM decouple pour exclusive_groups, absent de ce
    # wrapper jusqu'ici (integration jamais faite alors que state_space_node_v3.py la produit
    # depuis plusieurs revisions). Ordre etabli et jamais improvise : APRES le filtrage lexical
    # (Etape B, jamais de raisonnement sur des acteurs sur le point d'etre ecartes), AVANT la
    # dedup (Etape C, dont la logique de fusion absorbe deja exclusive_groups exactement comme
    # elle absorbe "states"). Sans cet appel, exclusive_groups restait silencieusement vide sur
    # tout run reel passant par ce graphe -- desactivant du meme coup branch_conflicts en aval.
    with_groups, exclusive_groups_report = generate_exclusive_groups(state["text"], filtered, llm)
    # Etape C (deduplication referentielle deterministe) -- fusionne les formes de surface d'un
    # meme referent AVANT que Pre/matching ne voient U, pour que les deux branches partagent un
    # vocabulaire unique. Chaque fusion/ambiguite est tracee dans state_space_dedup, jamais
    # silencieuse -- cf. state_space_node_v3.deduplicate_state_space.
    deduped, dedup_report = deduplicate_state_space(with_groups)
    return {
        "candidates": candidates,
        "validation_report": report,
        "exclusive_groups_report": exclusive_groups_report,
        "state_space": deduped,
        "state_space_dedup": dedup_report,
    }


def precondition_node(state: PipelineState) -> dict:
    prompt_name = state.get("precondition_prompt_name", "zero_shot")
    template = _PRECONDITION_PROMPTS[prompt_name]
    llm = get_llm(temperature=0, model=state.get("precondition_model")) if state.get("precondition_model") \
        else get_llm(temperature=0)
    output = run_naive_with_retry(state["text"], state["state_space"], llm, template)
    return {
        "raw_preconditions": output["raw"],
        "validated_preconditions": output["validated"],
        "precondition_cycles": output["cycles"],
        # Jusqu'ici jamais extrait de la sortie du node alors que run_naive_with_retry le
        # produit depuis l'ajout de check_branch_coherence -- perdu silencieusement avant meme
        # d'atteindre graph_construction_node, qui sait pourtant deja le consommer (troisieme
        # parametre optionnel de build_graph).
        "branch_conflicts": output["branch_conflicts"],
    }


def graph_construction_node(state: PipelineState) -> dict:
    graph = build_graph(
        state["state_space"], state["validated_preconditions"], state.get("branch_conflicts", [])
    )
    return {"reference_graph": graph}


# --- Branche droite : BPMN -> DFG -> SPO ---

def bpmn_to_spo(state: PipelineState) -> dict:
    # Calcule aussi activity_guards ici, PAS dans un node separe -- reutilise le meme objet
    # bpmn deja parse par parse_bpmn() une seule fois (jamais un second parsing). L'objet BPMN
    # brut de pm4py n'est jamais mis dans l'etat (non serialisable proprement en JSON, cf.
    # json.dump(..., default=str) dans run.py -- une stringification avec default=str serait
    # perdue/irrecuperable) ; seul le resultat structure et serialisable de
    # compute_activity_guards() y entre.
    bpmn = parse_bpmn(state["bpmn_path"])
    dfg = resolve_dfg(bpmn)
    spo = dfg_to_spo(dfg)
    activity_guards = compute_activity_guards(bpmn)
    return {"dfg_edges": dfg, "spo_triples": spo, "activity_guards": activity_guards}


# --- Jointure 1 : U (branche gauche) + SPO (branche droite) -- symetrique, pas de defer ---

def state_matching_node(state: PipelineState) -> dict:
    # CORRECTIF (bloquant) : match_dfg_to_u() attend le DFG riche (source_id/source_label/
    # target_id/target_label/gateway_context), PAS une projection en triplets SPO aplatis --
    # c'est meme la raison d'etre de ce fichier ("Exploite le DFG complet pour recuperer le
    # gateway_context", cf. state_matching_node.py). L'ancien code projetait spo_triples en
    # tuples (subject, predicate, object) et appelait une fonction match_spo_to_u() qui
    # n'existe nulle part dans le code reel -- ImportError au chargement du module, avant meme
    # qu'un seul node ne tourne. dfg_edges (calcule par bpmn_to_spo, jamais utilise jusqu'ici en
    # aval) est la seule entree correcte ici ; spo_triples reste calcule et stocke dans l'etat
    # pour tracabilite, mais n'est consomme par aucun node de ce graphe.
    matches, process_graph = match_dfg_to_u(state["dfg_edges"], state["state_space"])
    return {"matches": matches, "process_graph": process_graph}


# --- Jointure : G (branche gauche) + matches/G_process (state_matching) ---

def alignment_node(state: PipelineState) -> dict:
    # .get() plutot que [] -- activity_guards peut etre absent si bpmn_to_spo a echoue avant
    # de l'ecrire (cf. _with_error_capture) ; check_alignment le traite comme None (renforcement
    # structurel simplement indisponible, jamais un blocage -- meme discipline que partout
    # ailleurs : une absence est un statut legitime, jamais une exception).
    alignment = check_alignment(
        state["reference_graph"], state["process_graph"], state["matches"],
        activity_guards=state.get("activity_guards"),
    )
    return {"alignment": alignment}


def report_node(state: PipelineState) -> dict:
    model = state.get("report_model", DEFAULT_REPORT_MODEL_KEY)
    llm = get_llm(temperature=0, model=model)
    report = build_report(state["alignment"], state["matches"], llm)
    return {"report": report}


def build_pipeline():
    g = StateGraph(PipelineState)

    g.add_node("state_space", _with_error_capture("state_space", state_space_node))
    g.add_node("precondition", _with_error_capture("precondition", precondition_node))
    g.add_node("graph_construction", _with_error_capture("graph_construction", graph_construction_node))
    g.add_node("bpmn_to_spo", _with_error_capture("bpmn_to_spo", bpmn_to_spo))
    # state_matching : jointure symetrique (state_space et bpmn_to_spo sont tous deux a 1 saut
    # de START) -- pas de defer necessaire, teste et confirme des le premier essai.
    g.add_node("state_matching", _with_error_capture("state_matching", state_matching_node))
    # alignment : defer=True necessaire -- seule jointure asymetrique du graphe
    # (graph_construction a 3 sauts depuis START, state_matching a 2). Teste et confirme :
    # sans defer, alignment se declenche des que le premier des deux predecesseurs finit, avec
    # l'autre encore manquant.
    g.add_node("alignment", _with_error_capture("alignment", alignment_node), defer=True)
    g.add_node("report", _with_error_capture("report", report_node))

    g.add_edge(START, "state_space")
    g.add_edge("state_space", "precondition")
    g.add_edge("precondition", "graph_construction")

    g.add_edge(START, "bpmn_to_spo")
    g.add_edge("state_space", "state_matching")
    g.add_edge("bpmn_to_spo", "state_matching")

    g.add_edge("graph_construction", "alignment")
    g.add_edge("state_matching", "alignment")

    g.add_edge("alignment", "report")
    g.add_edge("report", END)

    return g.compile()


graph = build_pipeline()