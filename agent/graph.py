"""
Fichier : graph.py
Orchestrateur LangGraph — deux sous-graphes compilés séparément.

graph_T  : Phase normative (une seule fois par texte T)
           intent_T → universe_construction → rules_extraction
           → precondition_mapping → graph_construction

graph_F  : Phase vérification (une fois par scénario F)
           intent_F → projection → causal_verification (boucle)
           → propagation → aggregation → explanation
"""

from langgraph.graph import StateGraph, START, END

from agent.state import NeuroSymbolicState
from agent.edges import router_verification

from agent.nodes.intent_T          import intent_inference_T_node
from agent.nodes.intent_F          import intent_inference_F_node
from agent.nodes.universe_construction import universe_construction_node
from agent.nodes.rules_extraction  import rules_extraction_node
from agent.nodes.precondition_mapping  import precondition_mapping_node
from agent.nodes.graph_construction    import graph_construction_node
from agent.nodes.projection        import projection_node
from agent.nodes.causal_verification   import causal_verification_node
from agent.nodes.propagation       import propagation_node
from agent.nodes.aggregation       import aggregation_node
from agent.nodes.explanation       import explanation_node


def build_graph_T():
    """
    Sous-graphe T : construction de la référence normative.
    Entrée  : T (texte normatif)
    Sorties : intent_T, U, Pre_text, Pre_mapped, G_T, weights
    """
    wf = StateGraph(NeuroSymbolicState)

    wf.add_node("intent_T",             intent_inference_T_node)
    wf.add_node("universe_construction",universe_construction_node)
    wf.add_node("rules_extraction",     rules_extraction_node)
    wf.add_node("precondition_mapping", precondition_mapping_node)
    wf.add_node("graph_construction",   graph_construction_node)

    wf.add_edge(START,                  "intent_T")
    wf.add_edge("intent_T",             "universe_construction")
    wf.add_edge("universe_construction","rules_extraction")
    wf.add_edge("rules_extraction",     "precondition_mapping")
    wf.add_edge("precondition_mapping", "graph_construction")
    wf.add_edge("graph_construction",   END)

    return wf.compile()


def build_graph_F():
    """
    Sous-graphe F : vérification d'un scénario.
    Entrée  : F + tout le state T déjà calculé (U, Pre_mapped, G_T, weights)
    Sorties : intent_F, X, violations, contaminated_states, final_score, explanation
    """
    wf = StateGraph(NeuroSymbolicState)

    wf.add_node("intent_F",           intent_inference_F_node)
    wf.add_node("projection",         projection_node)
    wf.add_node("causal_verification",causal_verification_node)
    wf.add_node("propagation",        propagation_node)
    wf.add_node("aggregation",        aggregation_node)
    wf.add_node("explanation",        explanation_node)

    wf.add_edge(START,              "intent_F")
    wf.add_edge("intent_F",         "projection")
    wf.add_edge("projection",       "causal_verification")

    wf.add_conditional_edges(
        "causal_verification",
        router_verification,
        {
            "continue":  "causal_verification",
            "end_loop":  "propagation",
        }
    )

    wf.add_edge("propagation",  "aggregation")
    wf.add_edge("aggregation",  "explanation")
    wf.add_edge("explanation",  END)

    return wf.compile()


# Instances exportées
graph_T = build_graph_T()
graph_F = build_graph_F()