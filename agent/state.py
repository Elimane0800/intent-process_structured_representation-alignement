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
    "head_used": str,          # A.2 -- mot-tete utilise pour la decision lexicale, trace pour
                                 # audit ulterieur (jamais present avant Protocol 3.1)
    "reason": NotRequired[str],  # present seulement si kept=False pour raison "stopword"
    "exclusive_group_warnings": NotRequired[list[list[str]]],  # paires exclusive_groups dont
                                 # un des deux etats n'existe pas dans "states" -- signalees ici,
                                 # jamais silencieusement laissees passer ni silencieusement
                                 # supprimees (cf. filter_state_space -> _sanitize_exclusive_groups)
})
# NB : validation_report porte aussi une cle reservee "__meta__" (jamais un vrai nom de
# candidat), agregat {"total_candidates": int, "indeterminate_kept_count": int} -- cf. A.8,
# state_space_node_v3.py::validate_candidates. Ne correspond pas a la forme ci-dessus ;
# tout code qui iterait validation_report.items() en supposant que chaque cle est un candidat
# doit l'ignorer explicitement.

# Rapport de l'etape C de state_space_node_v3 (deduplicate_state_space) : fusion deterministe
# des formes de surface d'un meme referent ('job_applications'+'job_application',
# 'model'->'3d_model'). Chaque fusion est tracee ici ; les cas d'anaphore ambigue (un nom
# general contenu dans PLUSIEURS specifiques, ex. 'account' sous 'bank_account' ET
# 'battle_net_account') ne sont jamais fusionnes, seulement signales.
DedupMergeEntry = TypedDict("DedupMergeEntry", {
    "absorbed": list[str],       # noms de surface fusionnes dans la cle (le nom conserve)
    "rule": str,                  # "morphological_identity" | "anaphoric_containment" | combinaison
    "states": list[str],          # liste d'etats resultante (union ordonnee, dedupliquee)
    "exclusive_groups": list[list[str]],  # union des paires exclusives des entites fusionnees
    "concurrent_with": list[str],          # union resolue a travers la table de renommage
})

StateSpaceDedupReport = TypedDict("StateSpaceDedupReport", {
    "merged": dict[str, DedupMergeEntry],          # cle = nom conserve
    "ambiguous_not_merged": dict[str, list[str]],  # nom general -> specifiques candidats
})

# Schema imbrique depuis Protocol 3.2 (state_space_node_v3.py) -- remplace l'ancienne liste
# plate d'etats. "candidates" (sortie brute de l'etape A) ET "state_space" (U final, post
# etapes B/A2/C) partagent cette meme forme structurelle -- seule la richesse du contenu
# differe selon l'etape (ex. exclusive_groups vide juste apres l'etape A, l'etape A2 ne
# s'executant qu'ensuite).
StateSpaceEntity = TypedDict("StateSpaceEntity", {
    "states": list[str],
    "exclusive_groups": list[list[str]],  # paires d'etats mutuellement exclusifs de CETTE
                                            # entite -- vide par defaut, jamais suppose par
                                            # defaut a l'exclusivite (cf. state_space_prompt_
                                            # permissive.py, la regle "most states ... are NOT
                                            # mutually exclusive")
    "concurrent_with": list[str],           # autres entites explicitement marquees simultanees
                                              # dans le texte (marqueur textuel explicite requis)
})

# Sortie de generate_exclusive_groups() (Etape A2, Protocol 3.2) -- second appel LLM decouple,
# jamais fusionne avec l'etape A elle-meme (cf. state_space_prompt_permissive.py, section
# EXCLUSIVE_GROUPS_PROMPT). "accepted"/"rejected" tracent chaque proposition individuellement,
# rejetee jamais silencieusement (etat inconnu, entite inconnue, citation manquante, paire trop
# courte -- chaque raison possible cf. state_space_node_v3.py::generate_exclusive_groups).
ExclusiveGroupClaim = TypedDict("ExclusiveGroupClaim", {
    "entity": str,
    "states": list[str],
    "quote": str,
})

ExclusiveGroupRejection = TypedDict("ExclusiveGroupRejection", {
    "entity": NotRequired[str],
    "states": NotRequired[list[str]],
    "quote": NotRequired[str],
    "reason": str,
})

ExclusiveGroupsReport = TypedDict("ExclusiveGroupsReport", {
    "accepted": list[ExclusiveGroupClaim],
    "rejected": list[ExclusiveGroupRejection],
})

# Sortie brute de extract_preconditions() (precondition_node_with_retry.py), avant validation --
# {"precondition": str, "quote": ...} uniquement, jamais de "status"/"reason"/"retried"
# a ce stade (ajoutes seulement par _validate_one()/retry_downgraded(), cf. ValidatedPrecondition
# ci-dessous). Peut aussi etre une simple chaine ("INITIAL") au lieu d'un dict, tolere par
# _validate_one() en amont -- forme canonique produite par le LLM reste le dict.
#
# "quote" accepte desormais soit une citation unique (str, le cas par defaut), soit une liste
# ordonnee de segments verbatim (list[str]) -- pour une regle dont la justification est batie
# par accumulation sur plusieurs phrases non adjacentes plutot que sur un seul segment continu
# (cf. precondition_prompt.py, la regle "ordered list of verbatim segments"). Les deux formes
# sont validees par _validate_quote_grounding(), jamais devinees.
RawPrecondition = TypedDict("RawPrecondition", {
    "precondition": str,
    "quote": str | list[str] | None,
})

ValidatedPrecondition = TypedDict("ValidatedPrecondition", {
    "precondition": str,        # "INITIAL" | "UNRESOLVED" | "entity.state AND entity.state ..."
                                  # | "entity.state OR entity.state ..." | "(entity.state AND
                                  # entity.state) OR entity.state ..." (DNF a un niveau, v3) --
                                  # un terme peut desormais porter un prefixe "NOT " (v4, unaire,
                                  # jamais applique a un groupe parenthese)
    "quote": str | list[str] | None,   # cf. RawPrecondition -- meme forme polymorphe
    "status": Literal["OK", "DOWNGRADED"],
    "reason": str | None,
    "retried": NotRequired[bool],  # present seulement si passe par retry_downgraded()
})

# Sortie de check_branch_coherence() (precondition_node_with_retry.py) -- verification
# ADDITIVE, jamais bloquante : deux cibles marquees exclusive_groups qui partagent exactement
# la meme precondition acceptee, une contradiction directe de leur exclusivite declaree. Ne
# downgrade jamais rien, ne mute jamais "validated_preconditions" -- coherent avec la decision
# actee que exclusive_groups reste un signal bruite, jamais une autorite (cf.
# state_space_observations_2.md).
BranchConflict = TypedDict("BranchConflict", {
    "entity": str,
    "exclusive_group": list[str],
    "conflicting_targets": list[str],
    "shared_precondition": str,
    "reason": str,
})

# Une arete de graph_construction (graph_node.py::build_edges). "from"/"to" sont des mots-cles
# Python -- TypedDict fonctionnel obligatoire ici pour matcher exactement les cles reelles.
#
# v3 : "operator" (AND/OR unique par cible) remplace par "clause" (index DNF) -- la grammaire
# de Pre n'interdit plus le melange AND/OR pour une meme cible (precondition_prompt.py, v3),
# une cible peut donc porter plusieurs groupes d'aretes. Toutes les aretes qui partagent le
# meme (to, clause) forment un groupe ET ; des index de clause differents pour la meme cible
# sont les OU-alternatives d'une DNF a un niveau. Reconstruire Pre(u) exactement demande donc
# de regrouper par "to" PUIS par "clause" (cf. tgms_solver.py::tgms_from_pipeline), plus par
# "to" seul.
GraphEdge = TypedDict("GraphEdge", {
    "from": str,
    "to": str,
    "type": Literal["sequence", "cross_entity_guard"],
    "clause": int,  # index de la clause ET a laquelle appartient cette arete (0-based)
    "quote": str | list[str],  # forme polymorphe depuis v5 de precondition_prompt.py -- cf.
                                 # RawPrecondition/ValidatedPrecondition ; jamais None par
                                 # construction, _validate_one() l'exige pour accepter OK
    "negated": bool,  # v4 -- polarite du terme d'origine ("NOT entity.state"). Un terme nie
                        # pointe vers le MEME noeud pour la connectivite (cf. build_edges,
                        # detect_cycles), seule cette polarite distingue les deux cas -- sans
                        # elle, tgms_solver.py traitait "X AND NOT Y" comme "X AND Y" (bug
                        # trouve et corrige, cf. tgms_from_pipeline).
})

ExclusivePair = TypedDict("ExclusivePair", {
    "entity": str,
    "nodes": list[str],  # exactement les deux (ou plus) noeuds "entity.state" declares
                           # exclusive_groups pour cette entite -- relation XOR ENTRE deux
                           # noeuds cibles, jamais une arete causale (cf. build_figure,
                           # connecteur pointille distinct des aretes AND/OR/NOT)
})

GraphNode = TypedDict("GraphNode", {
    "id": str,       # "entity.state"
    "weight": float,  # centralite de degre -- calcule mais plus utilise par l'encodage visuel
                       # des visualisations PNG depuis la refonte AND/OR/NOT/XOR (decision
                       # explicite : le canal visuel sert desormais a la structure logique et
                       # aux noeuds en conflit de branche, pas au poids) ; jamais utilise par
                       # l'alignement non plus (cf. alignment_observations.md, section 3.2).
                       # Conserve dans la sortie pour un usage futur eventuel.
})

ReferenceGraph = TypedDict("ReferenceGraph", {
    "nodes": list[GraphNode],
    "edges": list[GraphEdge],
    "cycles": list[list[str]],  # v4 -- ne signale plus un cycle compose entierement d'aretes
                                  # negatives (deux branches mutuellement exclusives exprimees
                                  # via NOT reciproque n'est pas une circularite causale, cf.
                                  # detect_cycles) ; signale toujours un cycle contenant au
                                  # moins une arete positive
    "branch_conflicts": list[BranchConflict],  # propage tel quel depuis precondition_node_
                                                  # with_retry.py -- jamais recalcule ici
    "exclusive_pairs": list[ExclusivePair],     # toutes les paires exclusive_groups declarees
                                                   # du state space, conflictuelles ou non
})


# --- Branche droite : BPMN -> DFG -> SPO -> state_matching ---

DFGEdge = TypedDict("DFGEdge", {
    "source_id": str,
    "source_label": str,
    "target_id": str,
    "target_label": str,
    "gateway_context": list[str],  # conserve pour un usage futur (Micro), jamais utilise
                                     # dans le triplet SPO lui-meme (relation = "follows" fixe)
})
# NB : pas de champ "is_loop" -- n'existe dans aucune sortie reelle de resolve_dfg()
# (bpmn_to_spo_node.py). Absent depuis toujours cote code ; retire ici pour ne plus laisser
# supposer un champ qui n'a jamais existe.

# Forme canonique sur disque (ex. job_application_spo.json) -- garde les IDs BPMN, necessaires
# pour desambiguer des labels identiques (ex. deux activites "Unemployed" distinctes, start et
# end). NB : state_matching_node NE consomme PAS cette forme (ni sa projection en tuples) --
# match_dfg_to_u() prend directement dfg_edges (le DFG riche, avec gateway_context), jamais les
# triplets SPO, ni dans le node du graphe agent/graph.py ni dans le __main__ de
# state_matching_node.py lui-meme. spo_triples reste calcule et stocke dans l'etat pour
# tracabilite/usage futur, mais n'est actuellement consomme par aucun node du pipeline.
SPOTriple = TypedDict("SPOTriple", {
    "subject": str,
    "subject_id": str,
    "predicate": str,   # toujours "follows" actuellement -- decision actee, gateway_context
                          # (typage XOR/AND/boucle) volontairement laisse hors de ce champ
    "object": str,
    "object_id": str,
})

# Forme top-k + clustering (state_matching_node.py::match_activities_to_states) -- remplace
# l'ancienne forme a match unique par activite. Cle du dict englobant (PipelineState.matches)
# = identifiant d'activite BPMN contextualise (node_id), PAS le label brut.
MatchCandidate = TypedDict("MatchCandidate", {
    "match": str,    # entity.state retenu
    "score": float,   # similarite cosinus, avant tout seuillage de confiance en aval
})

ActivityMatchEntry = TypedDict("ActivityMatchEntry", {
    "activity_label": str,
    "matches": list[MatchCandidate],   # 0 a N candidats retenus (seuil absolu + marge top-k
                                          # relative de 0.05, cf. MATCHING_THRESHOLD)
    "clusters": list[list[str]],        # structure locale AND-of-OR (chaque sous-liste = un
                                          # cluster de quasi-synonymes/OR ; les clusters entre
                                          # eux = AND) -- additif, non consomme par le solveur
                                          # BFS actuel (cf. tgms_solver.py, activity_clusters)
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
    "source_id": str,        # id BPMN brut de l'activite source -- reellement produit par
                               # build_process_state_graph, absent de ce type jusqu'ici
    "target_id": str,
})


# --- Micro : alignement + rapport ---

TermCheck = TypedDict("TermCheck", {
    "term": str,             # peut porter le prefixe "NOT " (v4, tgms_solver.py) -- meme
                               # noeud sous-jacent que sa forme positive pour l'ancrage/la
                               # connectivite, seule la polarite differe (cf. status_of_
                               # negated_term)
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

ClauseCheck = TypedDict("ClauseCheck", {
    "terms": list[str],   # termes de cette clause ET (une des OR-alternatives de la DNF)
    "status": Literal["SATISFIED", "VIOLATED", "UNRESOLVABLE"],  # agregation AND de la clause
})

AlignmentEntry = TypedDict("AlignmentEntry", {
    "operator": Literal["AND", "OR", "MIXED"],  # v3 : label resume derive de "clauses",
                                                   # jamais utilise par la logique elle-meme --
                                                   # "MIXED" pour une vraie DNF (A AND B) OR C
    "quote": str | list[str] | None,   # forme polymorphe, propagee telle quelle depuis
                             # GraphEdge.quote -- None seulement si graph_construction pas
                             # encore regenere avec le fix quote -- cf. report_observations.md,
                             # section 3.1
    "terms": list[TermCheck],  # aplatissement de "clauses" ci-dessous, meme ordre -- inchange
                                 # depuis v2, c'est ce que consomme report_node.py, jamais la
                                 # structure "clauses" elle-meme
    "clauses": NotRequired[list[ClauseCheck]],  # v3, structure DNF explicite -- additif,
                                                   # absent si produit par une version anterieure
                                                   # de tgms_solver (pas casse pour autant,
                                                   # "terms" suffit toujours a report_node.py)
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

# --- Proposition 2/3 : sortie de bpmn_guards_node.compute_activity_guards ---
# Analyse structurelle pure (dominance/post-dominance + detection de fork non declare),
# aucun LLM. Additive dans le pipeline : renforce (jamais ne remplace) la verification par
# atteignabilite -- cf. tgms_solver.py::status_of_negated_term.
ActivityGuardEntry = TypedDict("ActivityGuardEntry", {
    "status": Literal["OK", "STRUCTURE_UNRESOLVED"],
    "clauses": NotRequired[list[list[str]]],   # present seulement si status == "OK"
    "forks_undeclared": NotRequired[bool],      # present seulement sur le noeud D'ORIGINE d'un
                                                   # fork non declare (activite a out_degree > 1
                                                   # sans gateway) -- ses successeurs recoivent
                                                   # STRUCTURE_UNRESOLVED par propagation, sans
                                                   # ce flag, cf. bpmn_guards_node.py
})

ActivityGuardsResult = TypedDict("ActivityGuardsResult", {
    "activities": dict[str, ActivityGuardEntry],  # cle = label d'activite BPMN (pas un id)
    "undeclared_splits": list[str],                 # labels des activites-origine d'un fork
                                                        # non declare -- meme information que
                                                        # forks_undeclared=True ci-dessus, exposee
                                                        # aussi a plat pour un acces direct
})


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
    candidates: NotRequired[dict[str, StateSpaceEntity]]     # sortie brute de l'etape A (U)
    validation_report: NotRequired[dict[str, ValidationReportEntry]]
    exclusive_groups_report: NotRequired[ExclusiveGroupsReport]  # sortie de l'etape A2 (second
                                                                    # appel LLM decouple)
    state_space: NotRequired[dict[str, StateSpaceEntity]]    # U valide ET deduplique (etape C)
    state_space_dedup: NotRequired[StateSpaceDedupReport]     # trace des fusions referentielles

    raw_preconditions: NotRequired[dict[str, RawPrecondition]]
    validated_preconditions: NotRequired[dict[str, ValidatedPrecondition]]
    precondition_cycles: NotRequired[list[list[str]]]
    branch_conflicts: NotRequired[list[BranchConflict]]        # sortie de check_branch_coherence,
                                                                   # transitoire ici -- finit
                                                                   # aussi dans reference_graph
                                                                   # une fois graph_construction
                                                                   # execute (meme donnee, deux
                                                                   # emplacements par commodite
                                                                   # d'acces en aval)

    reference_graph: NotRequired[ReferenceGraph]              # G

    # --- Branche droite ---
    dfg_edges: NotRequired[list[DFGEdge]]
    spo_triples: NotRequired[list[SPOTriple]]                  # calcule, trace, mais non
                                                                   # consomme par aucun node
                                                                   # actuel du pipeline (cf.
                                                                   # SPOTriple ci-dessus)
    activity_guards: NotRequired[ActivityGuardsResult]          # proposition 1/2/3 -- calcule
                                                                   # par le meme node que
                                                                   # dfg_edges (reutilise le
                                                                   # meme bpmn deja parse, jamais
                                                                   # un second parsing), absent
                                                                   # si bpmn_to_spo a echoue

    matches: NotRequired[dict[str, ActivityMatchEntry]]        # cle = id d'activite BPMN
                                                                   # contextualise (node_id),
                                                                   # PAS le label brut
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