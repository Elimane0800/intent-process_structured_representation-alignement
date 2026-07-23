"""Etat partage du pipeline complet (agent/graph.py) -- un objet par execution bout-en-bout : un
texte source, un cas, un modele BPMN precis, une combinaison prompt/modele precise. Pas un
conteneur pour plusieurs prompts/modeles/cas a la fois -- orchestrer plusieurs combinaisons est
le role d'un script qui invoque le graphe plusieurs fois, pas de l'etat lui-meme.

Convention : chaque champ est rempli progressivement au fil des nodes. Un champ absent
(NotRequired) signifie "pas encore atteint dans ce run", jamais une erreur -- coherent avec le
reste du pipeline, ou une absence est toujours un statut legitime (UNRESOLVED, UNRESOLVABLE,
etc.), jamais une exception silencieuse.

Chaque type ci-dessous reprend exactement les cles produites par le node correspondant, deja
livre et teste -- pas une schematisation a priori. Ou les noms different legerement de ceux
utilises en interne par un node (ex. "graph" -> "process_graph" ici, pour eviter la confusion
avec agent/graph.py), c'est note explicitement.
"""

from typing import Annotated, Literal, NotRequired, TypedDict
import operator

# --- Branche gauche : texte source -> U (Protocole 3) -> Pre -> G (graph_construction) ---

ValidationReportEntry = TypedDict("ValidationReportEntry", {
    "f_lex": str | None,       # "ACTOR" | "ENTITY" | "INDETERMINATE" | None (stopword)
    "kept": bool,
    "reason": NotRequired[str],  # present seulement si kept=False pour raison "stopword"
})

# Rapport de l'etape C de state_space_node_v3 (deduplicate_state_space) : fusion deterministe
# des formes de surface d'un meme referent ('job_applications'+'job_application',
# 'model'->'3d_model'). Chaque fusion est tracee ici ; les cas d'anaphore ambigue (un nom
# general contenu dans PLUSIEURS specifiques, ex. 'account' sous 'bank_account' ET
# 'battle_net_account') ne sont jamais fusionnes, seulement signales.
DedupMergeEntry = TypedDict("DedupMergeEntry", {
    "absorbed": list[str],  # noms de surface fusionnes dans la cle (le nom conserve)
    "rule": str,             # "morphological_identity" | "anaphoric_containment" | combinaison
    "states": list[str],     # liste d'etats resultante (union ordonnee, dedupliquee)
})

StateSpaceDedupReport = TypedDict("StateSpaceDedupReport", {
    "merged": dict[str, DedupMergeEntry],          # cle = nom conserve
    "ambiguous_not_merged": dict[str, list[str]],  # nom general -> specifiques candidats
})

# Sortie brute de extract_preconditions() (precondition_node_with_retry.py), avant validation --
# {"precondition": str, "quote": str | None} uniquement, jamais de "status"/"reason"/"retried"
# a ce stade (ajoutes seulement par _validate_one()/retry_downgraded(), cf. ValidatedPrecondition
# ci-dessous). Peut aussi etre une simple chaine ("INITIAL") au lieu d'un dict, tolere par
# _validate_one() en amont -- forme canonique produite par le LLM reste le dict.
RawPrecondition = TypedDict("RawPrecondition", {
    "precondition": str,
    "quote": str | None,
})

ValidatedPrecondition = TypedDict("ValidatedPrecondition", {
    "precondition": str,        # "INITIAL" | "UNRESOLVED" | "entity.state AND/OR ..."
    "quote": str | None,
    "status": Literal["OK", "DOWNGRADED"],
    "reason": str | None,
    "retried": NotRequired[bool],  # present seulement si passe par retry_downgraded()
})

# Une arete de graph_construction (graph_node.py::build_edges). "from"/"to" sont des mots-cles
# Python -- TypedDict fonctionnel obligatoire ici pour matcher exactement les cles reelles.
GraphEdge = TypedDict("GraphEdge", {
    "from": str,
    "to": str,
    "type": Literal["sequence", "cross_entity_guard"],
    "operator": Literal["AND", "OR"],
    "quote": str,   # jamais None par construction -- _validate_one() l'exige pour accepter OK
})

GraphNode = TypedDict("GraphNode", {
    "id": str,       # "entity.state"
    "weight": float,  # centralite de degre -- utilise uniquement par les visualisations PNG,
                       # jamais par l'alignement (cf. alignment_observations.md, section 3.2)
})

ReferenceGraph = TypedDict("ReferenceGraph", {
    "nodes": list[GraphNode],
    "edges": list[GraphEdge],
    "cycles": list[list[str]],
})


# --- Branche droite : BPMN -> DFG -> SPO -> state_matching ---

DFGEdge = TypedDict("DFGEdge", {
    "source_id": str,
    "source_label": str,
    "target_id": str,
    "target_label": str,
    "gateway_context": list[str],  # conserve pour un usage futur (Micro), jamais utilise
                                     # dans le triplet SPO lui-meme (relation = "follows" fixe)
    "is_loop": bool,
})

# Forme canonique sur disque (ex. job_application_spo.json) -- garde les IDs BPMN, necessaires
# pour desambiguer des labels identiques (ex. deux activites "Unemployed" distinctes, start et
# end). state_matching_node consomme une projection en tuples (subject, predicate, object) a
# partir de cette forme dans son __main__ -- la forme riche ci-dessous reste la source de
# verite dans l'etat, pas les tuples (qui perdent les IDs).
SPOTriple = TypedDict("SPOTriple", {
    "subject": str,
    "subject_id": str,
    "predicate": str,   # toujours "follows" actuellement -- decision actee, gateway_context
                          # (typage XOR/AND/boucle) volontairement laisse hors de ce champ
    "object": str,
    "object_id": str,
})

MatchResult = TypedDict("MatchResult", {
    "match": str,    # entity.state retenu (meilleur score cosinus)
    "score": float,
})

# Arete de G_process (state_matching_node.py::build_process_state_graph) -- deja exprimee dans
# le vocabulaire entity.state de U, pas dans les labels d'activite bruts (c'est le matching qui
# fait cette traduction). "graph" dans la sortie brute du node -> "process_graph" ici pour ne
# pas entrer en collision avec agent/graph.py.
ProcessGraphEdge = TypedDict("ProcessGraphEdge", {
    "from": str,
    "to": str,
    "activity_from": str,   # label BPMN brut, pour tracabilite / le rapport final
    "activity_to": str,
})


# --- Micro : alignement + rapport ---

TermCheck = TypedDict("TermCheck", {
    "term": str,
    "status": Literal["SATISFIED", "VIOLATED", "UNRESOLVABLE"],
    "reason": str | None,
    "missing": list[str],   # etats sans correspondant dans G_process ([] si les deux existent)
                              # -- produit par check_term depuis toujours, absent de ce type
                              # jusqu'ici (incoherence corrigee, pas un nouveau champ)
    "low_confidence": NotRequired[list[str]],  # present seulement si un candidat VIOLATED a ete
                                                 # retrograde en UNRESOLVABLE parce qu'un de ses
                                                 # deux etats repose sur un match de score <
                                                 # CONFIDENCE_THRESHOLD (alignment_node) --
                                                 # liste les etats fautifs, scores dans 'reason'
})

AlignmentEntry = TypedDict("AlignmentEntry", {
    "operator": Literal["AND", "OR"],
    "quote": str | None,   # None seulement si graph_construction pas encore regenere avec le
                             # fix quote -- cf. report_observations.md, section 3.1
    "terms": list[TermCheck],
    "status": Literal["SATISFIED", "VIOLATED", "UNRESOLVABLE"],
})

ReportItem = TypedDict("ReportItem", {
    "root_cause": str,
    "quote": str,
    "explanation": str,
    "quote_verbatim": bool | None,  # controle mecanique (report_node.quote_reproduced_verbatim):
                                      # la citation apparait-elle mot pour mot dans l'explication ?
                                      # False = traduite/paraphrasee (violation flaggee, jamais
                                      # bloquee) ; None = aucune citation disponible a verifier
    "affected_activities": list[str],
    "affected_count": int,
})

ReportSatisfiedItem = TypedDict("ReportSatisfiedItem", {
    "target": str,
    "term": str,
})

ReportSummary = TypedDict("ReportSummary", {
    "ecarts": int,
    "non_verifiable": int,
    "conforme": int,
})

Report = TypedDict("Report", {
    "summary": ReportSummary,
    "ecarts": list[ReportItem],
    "non_verifiable": list[ReportItem],
    "conforme": list[ReportSatisfiedItem],
    "piece_jointe_alignement_brut": dict[str, AlignmentEntry],
})


# --- Etat global ---

class PipelineState(TypedDict):
    # Identification du run -- jamais utilise pour la logique elle-meme, uniquement pour la
    # tracabilite et pour retrouver/reproduire un run precis.
    case_name: str                              # "job_application" | "maternity_leave" | "work_accident" | ...
    text: str                                    # T, le texte source d'intention

    state_space_run: NotRequired[str]            # ex. "run_17.json", U de reference pin explicitement
    state_space_prompt_name: NotRequired[str]    # "zero_shot" | "one_shot"
    state_space_model: NotRequired[str]

    precondition_prompt_name: NotRequired[str]   # "zero_shot" | "one_shot" | "two_shot" | "cot"
                                                   # -- NB: n'implique PAS quel U a ete utilise,
                                                   # cf. alignment_observations.md section 2.2
    precondition_model: NotRequired[str]

    bpmn_path: NotRequired[str]
    bpmn_description_index: NotRequired[int]     # index de description (corpus Zenodo)
    bpmn_model_index: NotRequired[int]           # index du modele BPMN parmi les 8-11 du cas
    bpmn_expert_score: NotRequired[int]          # note Zenodo 0-5, si connue -- jamais utilisee
                                                   # par le pipeline lui-meme, comparaison externe
                                                   # a la main uniquement

    report_model: NotRequired[str]               # modele redacteur du rapport final -- distinct
                                                   # du modele "sous test" (precondition_model)

    # --- Branche gauche ---
    candidates: NotRequired[dict[str, list[str]]]           # sortie brute de l'etape A (U)
    validation_report: NotRequired[dict[str, ValidationReportEntry]]
    state_space: NotRequired[dict[str, list[str]]]           # U valide ET deduplique (etape C)
    state_space_dedup: NotRequired[StateSpaceDedupReport]    # trace des fusions referentielles

    raw_preconditions: NotRequired[dict[str, RawPrecondition]]
    validated_preconditions: NotRequired[dict[str, ValidatedPrecondition]]
    precondition_cycles: NotRequired[list[list[str]]]

    reference_graph: NotRequired[ReferenceGraph]              # G

    # --- Branche droite ---
    dfg_edges: NotRequired[list[DFGEdge]]
    spo_triples: NotRequired[list[SPOTriple]]

    matches: NotRequired[dict[str, MatchResult]]              # cle = label d'activite BPMN brut
    process_graph: NotRequired[list[ProcessGraphEdge]]         # G_process

    # --- Micro ---
    alignment: NotRequired[dict[str, AlignmentEntry]]
    report: NotRequired[Report]

    # Erreurs accumulees, jamais une exception qui casse tout le run -- chaque node qui echoue
    # ajoute ici plutot que de faire planter le graphe entier, meme discipline que le reste du
    # pipeline (rejet jamais silencieux, mais jamais bloquant non plus quand evitable).
    # Annotated[..., operator.add] est necessaire ici et nulle part ailleurs dans cet etat :
    # c'est le seul champ que plusieurs nodes executes EN PARALLELE (meme superstep) peuvent
    # ecrire simultanement -- sans reducteur explicite, LangGraph rejette la deuxieme ecriture
    # concurrente sur la meme cle (InvalidUpdateError), teste et confirme.
    errors: NotRequired[Annotated[list[str], operator.add]]