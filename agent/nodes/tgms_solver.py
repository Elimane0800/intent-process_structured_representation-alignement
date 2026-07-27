"""Twin-Graph Milestone System -- solveur de fidelite texte/processus confidence-gated.

Objet formel et solveur generique motives par une question de nouveaute precise (le mecanisme
d'alignment_node.py est-il un objet formel propre au probleme, ou une methode existante
redeguisee ?) -- voir la note de formalisme dediee (*A Twin-Graph Milestone System*) pour les
definitions completes, le theoreme de reduction vers la semantique de point fixe GSM
(Damaggio, Hull, Vaculin, BPM 2011), et les preuves des deux theoremes de monotonie.

Ce fichier consolide en un seul module ce qui vivra a terme dans un package standalone
separe (types/solveur/adaptateurs) -- deliberement PAS encore packages independamment :
integration et validation dans le pipeline d'abord, packaging seulement une fois tout
confirme. Zero import agent.* malgre tout, par discipline (garde la porte ouverte au
packaging futur sans reecriture).

============================================================================================
TYPES
============================================================================================
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


def _is_negated(term: str) -> bool:
    return term.startswith("NOT ")


def _strip_not(term: str) -> str:
    """Strips exactly one leading 'NOT ' if present, else returns the term unchanged. Copy of
    the same helper in precondition_node_with_retry.py / graph_node.py -- same duplication
    discipline this file already claims for itself ("zero import agent.*", see module
    docstring): a negated term references the SAME underlying milestone for identity/anchoring
    purposes, only its truth value is inverted."""
    return term[len("NOT "):] if _is_negated(term) else term


class Status(str, Enum):
    """Statut a trois valeurs, jamais reduit a deux (Definition 4 de la note, status_tau).
    Herite de str : comparaisons/serialisation JSON transparentes avec les chaines historiques
    du pipeline ('SATISFIED' == Status.SATISFIED est True), aucune migration requise cote
    consommateurs existants (report_node.py, analyze_dataset_runs.py)."""
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNRESOLVABLE = "UNRESOLVABLE"


@dataclass(frozen=True)
class Guard:
    """G(m) -- Definition 1, ETENDUE en v3 pour lever une restriction non justifiee. La v1/v2
    du fragment n'admettait qu'une conjonction ou une disjonction PURE (jamais mixte) -- un
    choix qui s'est revele etre une limite de la grammaire d'extraction (precondition_prompt.py),
    pas une limite reelle du domaine : le texte source exprime authentiquement des regles
    melangees ('(A et B) ou C', ex. 'claim.paid' dans precondition_prompt.py, Example 2).

    Represente maintenant comme une DNF a un seul niveau : une disjonction (OR) de clauses,
    chaque clause etant elle-meme une conjonction PURE (AND) d'un ou plusieurs termes. Un AND
    pur redevient le cas degenere a une seule clause ; un OR pur redevient le cas ou chaque
    clause est un singleton -- retrocompatible sans exception avec tout guard deja construit
    sous l'ancienne definition."""
    clauses: tuple[tuple[str, ...], ...]  # OR de clauses ; chaque clause = AND de termes

    def __post_init__(self) -> None:
        if not self.clauses:
            raise ValueError("un guard doit contenir au moins une clause")
        for clause in self.clauses:
            if not clause:
                raise ValueError("une clause ne peut pas etre vide")

    @property
    def terms(self) -> tuple[str, ...]:
        """Tous les termes, toutes clauses confondues, a plat -- pour les usages qui n'ont
        besoin que de l'ensemble des references (ex. verification d'auto-reference), pas de la
        structure logique elle-meme."""
        return tuple(t for clause in self.clauses for t in clause)


@dataclass
class TwinGraphMilestoneSystem:
    """TGMS = (M, G, A, Gproc, phi, c) -- Definition 3 de la note.

    - milestones (M)      : ensemble des milestones (entity.state du texte).
    - guards (G)           : m -> Guard(m), un guard par milestone non-INITIAL/UNRESOLVED.
    - process_edges (Gproc): aretes du graphe d'execution, sur le vocabulaire des ACTIVITES
                              (pas des milestones) -- c'est precisement ce qui distingue le
                              TGMS d'un schema GSM unique (Definition 3, remarque 3.1) : deux
                              vocabulaires distincts, relies par phi, pas un seul graphe.
    - anchor (phi)          : activite -> milestone, PARTIELLE et NON-INJECTIVE (plusieurs
                              activites peuvent ancrer le meme milestone ; une activite peut
                              n'ancrer aucun milestone -- absente de ce dict dans ce cas).
    - confidence (c)        : activite -> score dans [0,1] (cosinus de l'embedding dans
                              l'instantiation actuelle -- le TGMS lui-meme reste agnostique
                              a la source du score).
    """
    milestones: frozenset[str]
    guards: dict[str, Guard]
    process_edges: frozenset[tuple[str, str]]
    anchor: dict[str, str]           # activite -> milestone (absente si non ancree)
    confidence: dict[str, float]     # activite -> score

    # --- Champs additionnels (activite -> ...), preserves pour un usage FUTUR uniquement ---
    # `anchor` ci-dessus n'est PAS reellement "activite -> milestone" dans l'instantiation
    # pipeline actuelle (cf. tgms_from_pipeline) : c'est une identite milestone -> milestone,
    # parce que process_edges est deja projete au niveau milestone par
    # state_matching_node.build_process_state_graph AVANT d'arriver ici. Ca suffit au solveur
    # BFS actuel (solve()/path_exists), qui ne raisonne que sur des milestones. Mais ca perd,
    # structurellement, l'identite de l'activite BPMN brute -- information necessaire plus tard
    # pour projeter un Guard-process (issu de bpmn_guards_node.py, exprime au vocabulaire des
    # ACTIVITES) vers le vocabulaire des milestones. Ces deux champs portent CETTE information,
    # en plus de `anchor`/`process_edges` qui restent inchanges et continuent d'alimenter le
    # solveur BFS existant sans aucune modification de comportement (non-regression garantie).
    # NON CONSOMMES par solve() aujourd'hui -- prepares pour _project_process_clause (a venir),
    # produits et testes isolement avant d'etre branches dans la logique d'agregation.
    activity_matches: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # activite BPMN brute -> milestones qu'elle ancre (ordre = ordre de retour du matcher).

    activity_clusters: dict[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    # activite BPMN brute -> structure locale AND-of-OR (chaque tuple interne = un cluster de
    # quasi-synonymes, OR ; les clusters eux-memes sont AND entre eux) -- cf.
    # state_matching_node.py::_cluster_matches_by_similarity pour la justification du critere.

    # --- Champs proposition 2 (desormais CONSOMMES par status_of_negated_term) ---
    matches: dict | None = None
    # Copie brute de l'argument `matches` de tgms_from_pipeline (format top-k+clusters de
    # state_matching_node.py : {activity_id: {"activity_label":..., "matches":[...], ...}}).
    # anchor/confidence ci-dessus en derivent deja une forme aplatie (milestone -> score, sans
    # l'identite de l'activite), insuffisante pour repondre a "quelles activites ancrent CE
    # milestone" (necessaire pour croiser avec activity_guards, indexe par LABEL d'activite,
    # pas par milestone) -- cf. _anchoring_activities.

    activity_guards: dict | None = None
    # Sortie de bpmn_guards_node.compute_activity_guards -- {"activities": {label: {"status":
    # "OK"|"STRUCTURE_UNRESOLVED", ...}}, "undeclared_splits": [...]}. Analyse purement
    # structurelle (dominance/post-dominance + forks non declares, proposition 3), aucun LLM.
    # None si non fournie -- le solveur retombe alors integralement sur l'atteignabilite seule,
    # comportement identique a avant cette proposition (non-regression garantie).

    def __post_init__(self) -> None:
        for target, guard in self.guards.items():
            for term in guard.terms:
                if _strip_not(term) == target:
                    raise ValueError(
                        f"auto-reference interdite (Definition 2, regle 4 du fragment) : "
                        f"{target!r} refere a lui-meme dans son propre guard"
                    )
        unanchored_refs = set(self.anchor.values()) - self.milestones
        if unanchored_refs:
            raise ValueError(f"phi ancre vers des milestones inconnus de M : {unanchored_refs}")


@dataclass(frozen=True)
class TermVerdict:
    """Verdict pour un seul terme d'un guard -- Status_tau(m' -> m), Definition 4."""
    term: str
    status: Status
    missing: tuple[str, ...] = ()        # sous-ensemble de {term, target} sans AUCUN ancrage
    low_confidence: tuple[str, ...] = () # sous-ensemble de {term, target} ancre mais sous tau
    reason: str | None = None


@dataclass(frozen=True)
class ClauseVerdict:
    """Verdict pour une seule clause-ET du guard (un des OR-alternatives de la DNF) --
    agregation AND des TermVerdict qui la composent. Nouveau en v3 (n'existait pas quand un
    guard n'avait qu'un seul niveau plat)."""
    terms: tuple[TermVerdict, ...]
    status: Status


@dataclass(frozen=True)
class GuardVerdict:
    """Verdict agrege pour un milestone cible -- une entree de check_alignment().

    v3 : porte desormais `clauses` (un ClauseVerdict par OR-alternative de la DNF) plutot
    qu'une liste plate de TermVerdict -- necessaire pour representer fidelement un guard mixte
    AND/OR (cf. Guard). `terms` reste expose comme propriete (aplatissement de `clauses`,
    meme ordre qu'avant) pour que tout consommateur qui n'a besoin que de la liste plate des
    verdicts individuels (ex. report_node.py, qui ne lit jamais la structure logique elle-meme,
    seulement le statut de chaque terme) continue de fonctionner sans aucune modification.
    `operator` reste un label resume ('AND'/'OR'/'MIXED', derive de la forme des clauses) --
    informatif seulement, jamais utilise par la logique d'agregation elle-meme (qui opere sur
    `clauses` directement, cf. solve())."""
    target: str
    operator: str
    status: Status
    clauses: tuple[ClauseVerdict, ...] = field(default_factory=tuple)

    @property
    def terms(self) -> tuple[TermVerdict, ...]:
        return tuple(t for c in self.clauses for t in c.terms)


"""
============================================================================================
SOLVEUR -- point fixe monotone (guards AND/OR purs) + BFS confidence-gated
============================================================================================

Theoreme de reduction (note, section 2.3, Thm 2.3) : quand phi est totale, injective et
c = 1 partout (cas degenere a un seul graphe), ce solveur calcule exactement le plus petit
point fixe de l'operateur de consequence immediate d'un schema GSM (Damaggio, Hull, Vaculin,
BPM 2011) restreint au fragment Milestone/Guard. Rien de nouveau dans ce cas -- la nouveaute
est dans la gestion de phi/c au cas general (twin-graph, correspondance incertaine).
"""


def confident_activities(tgms: TwinGraphMilestoneSystem, tau: float) -> set[str]:
    """A_tau -- Definition 3bis : activites dont la confiance atteint le seuil. Documente
    A_tau pour inspection/diagnostic externe -- NON utilise pour le test d'ancrage du solveur
    (cf. anchored_milestones), seulement pour la decision de demotion d'un candidat VIOLATED."""
    return {a for a, score in tgms.confidence.items() if score >= tau}


def anchored_milestones(tgms: TwinGraphMilestoneSystem) -> set[str]:
    """Milestones ayant au moins UNE activite ancree, quelle que soit sa confiance --
    correspond exactement a `matched_states` dans l'implementation historique
    (matched_states = {m["match"] for m in matches.values()}, jamais filtre par seuil).

    Point de precision important, corrige apres test d'equivalence stricte contre
    l'implementation de reference (le premier jet de ce solveur filtrait par tau ici, ce qui
    divergeait du comportement reel sur un cas connu -- un etat ancre a 0.08 de confiance
    etait traite comme 'aucune activite appariee' au lieu de 'appariee mais peu confiante').
    Dans l'implementation reelle, tau ne joue AUCUN role dans la distinction missing/ancre --
    il n'intervient que plus tard, pour decider si un candidat VIOLATED doit etre retrograde
    (cf. status_of_term). Consequence notable, plus forte que ce que la note de formalisme
    enoncait initialement : un verdict SATISFIED est entierement INVARIANT a tau (ni le test
    d'ancrage ni l'atteignabilite ne dependent de tau), pas seulement 'jamais transforme en
    VIOLATED par une hausse de tau' -- la note doit etre corrigee en consequence (Definition
    3bis/4) avant integration finale au papier."""
    return set(tgms.anchor.values())


def best_confidence(tgms: TwinGraphMilestoneSystem) -> dict[str, float]:
    """milestone -> meilleur score de confiance parmi les activites qui l'ancrent (le maximum,
    pas la moyenne -- coherent avec label_for_state()/build_state_scores() de l'implementation
    historique : un seul bon appariement suffit a rendre l'etat fiable comme point d'ancrage)."""
    best: dict[str, float] = {}
    for activity, milestone in tgms.anchor.items():
        score = tgms.confidence.get(activity, 0.0)
        if milestone not in best or score > best[milestone]:
            best[milestone] = score
    return best


def _build_adjacency(edges: frozenset[tuple[str, str]]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
    return adjacency


def path_exists(adjacency: dict[str, list[str]], source: str, target: str) -> bool:
    """BFS sur le graphe process, jamais restreint par tau -- justifie par la preuve du
    Theoreme de monotonie en confiance (note, Thm 3.5) : la structure des aretes de Gproc est
    independante de tau, seule l'eligibilite des points d'ancrage (les deux EXTREMITES du
    chemin recherche) depend du seuil, et meme cette eligibilite n'entre en jeu qu'apres coup
    (demotion), jamais dans le calcul du chemin lui-meme."""
    if source == target:
        return True
    visited = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def path_exists_avoiding(adjacency: dict[str, list[str]], source: str, target: str, avoid: str) -> bool:
    """Comme path_exists, mais traite `avoid` comme retire du graphe -- la cible reste-t-elle
    atteignable depuis `source` SANS jamais passer par `avoid` ? Reutilise le MEME primitif BFS
    que path_exists (aucune notion nouvelle ou plus faible d'"absence" inventee pour l'occasion)
    -- seule la traversee de `avoid` est bloquee.

    C'est ce qui fonde la verification d'un terme NOT, par extension directe du raisonnement
    deja applique a la main sur le corpus BPMN plus tot dans ce projet (ex. modele Camunda 41,
    verification qu'un etat est reellement atteignable ou bloque par la structure du modele --
    pas devine) : le modele doit offrir une VRAIE route alternative vers la cible qui n'oblige
    pas `avoid` a se produire, pour correspondre a la semantique "unless" du texte -- pas
    seulement l'absence non prouvee d'un chemin quelconque."""
    if source == avoid:
        return False
    if source == target:
        return True
    visited = {source, avoid}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor == avoid:
                continue
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def status_of_term(
    tgms: TwinGraphMilestoneSystem,
    term: str,
    target: str,
    anchored: set[str],
    confidence: dict[str, float],
    adjacency: dict[str, list[str]],
    tau: float,
) -> TermVerdict:
    """Status_tau(m' -> m) -- Definition 4 corrigee : le test d'ancrage (`missing`) est
    INDEPENDANT de tau ; seule la decision de retrograder un candidat VIOLATED en
    UNRESOLVABLE (demotion) depend de tau."""
    missing = tuple(s for s in (term, target) if s not in anchored)
    if missing:
        return TermVerdict(term=term, status=Status.UNRESOLVABLE, missing=missing,
                            reason=f"no match in process for: {list(missing)}")

    if path_exists(adjacency, term, target):
        return TermVerdict(term=term, status=Status.SATISFIED)

    # Candidat VIOLATED -- demotion si un point d'ancrage est sous tau (les deux SONT ancres,
    # cf. `missing` ci-dessus, mais leur confiance peut rester en-dessous du seuil).
    weak = tuple(s for s in (term, target) if confidence.get(s, 0.0) < tau)
    if weak:
        detail = ", ".join(f"{s}={confidence.get(s, 0.0):.3f}" for s in weak)
        return TermVerdict(
            term=term, status=Status.UNRESOLVABLE, low_confidence=weak,
            reason=f"would be VIOLATED, but match confidence below {tau} for: {detail} -- "
                   f"cannot conclude on an uncertain match",
        )
    return TermVerdict(term=term, status=Status.VIOLATED,
                        reason="both states matched, no path in process")


def _anchoring_activities(milestone: str, matches: dict) -> set[str]:
    """Labels des activites BPMN qui ancrent ce milestone -- meme logique que
    report_node.reverse_labels, dupliquee ici par la meme discipline de decouplage deja
    affirmee en tete de ce fichier (aucun import agent.*). Necessaire pour croiser un milestone
    (vocabulaire de U/Pre) avec activity_guards (indexe par LABEL d'activite BPMN, vocabulaire
    de bpmn_guards_node.py) -- deux vocabulaires distincts, comme partout ailleurs dans ce
    pipeline, jamais fusionnes sans passer explicitement par matches."""
    return {
        entry.get("activity_label", aid)
        for aid, entry in matches.items()
        if isinstance(entry, dict)
        and any(c.get("match") == milestone for c in entry.get("matches", []))
    }


def status_of_negated_term(
    negated_term: str,
    positive_co_terms: tuple[str, ...],
    target: str,
    anchored: set[str],
    confidence: dict[str, float],
    adjacency: dict[str, list[str]],
    tau: float,
    matches: dict | None = None,
    activity_guards: dict | None = None,
) -> TermVerdict:
    """Status_tau pour un terme negatif ('NOT Y') a l'interieur d'une clause-ET -- fonde sur le
    MEME primitif d'atteignabilite que status_of_term (path_exists_avoiding, une simple
    extension de path_exists), jamais une notion separee et plus faible d'"absence". Extension
    directe, non arbitraire, du raisonnement deja applique a la main sur le corpus BPMN plus
    tot dans ce projet (verification d'atteignabilite structurelle, ex. modele Camunda 41) :
    'X AND NOT Y -> cible' est verifie en demandant si un chemin de X vers la cible existe qui
    n'oblige PAS a passer par Y -- une vraie route alternative dans le modele, pas une absence
    supposee.

    Necessite au moins un terme POSITIF ancre dans la MEME clause pour servir de point de
    depart -- 'NOT Y' seul n'a aucune notion de "atteignable depuis ou" dans un graphe qui
    n'encode que des relations 'follows' positives. Un NOT isole sans terme positif dans sa
    clause est UNRESOLVABLE, jamais devine a partir d'un ensemble de sources arbitraire."""
    missing = tuple(s for s in (negated_term, target) if s not in anchored)
    label = f"NOT {negated_term}"
    if missing:
        return TermVerdict(term=label, status=Status.UNRESOLVABLE, missing=missing,
                            reason=f"no match in process for: {list(missing)}")

    anchored_co_terms = tuple(t for t in positive_co_terms if t in anchored)
    if not anchored_co_terms:
        return TermVerdict(
            term=label, status=Status.UNRESOLVABLE,
            reason="negated term has no anchored positive co-term in the same AND-clause to "
                   "check reachability from -- cannot verify a negation without a real "
                   "anchor, never guessed",
        )

    avoidable_from = [c for c in anchored_co_terms
                       if path_exists_avoiding(adjacency, c, target, negated_term)]
    if avoidable_from:
        # CORRECTIF (proposition 2, additif) : un chemin qui evite negated_term dans le DFG
        # aplati (relation "follows" generique, gateways jamais reifies -- cf. bpmn_to_spo_node.
        # py) prouve seulement qu'UNE route existe qui ne force pas negated_term -- jamais que
        # le modele GARANTIT l'exclusivite entre target et negated_term. Cas concret qui motive
        # ce correctif, verifie sur un vrai modele : une tache BPMN avec deux flux sortants
        # directs, SANS gateway, fait diverger le jeton vers les deux issues a la fois --
        # "un chemin evite negated_term" reste techniquement vrai (les deux branches sont
        # disjointes dans le DFG aplati), mais l'exclusivite que le texte affirme n'est
        # structurellement garantie par rien dans le modele (cf. bpmn_guards_node.py::
        # find_undeclared_splits, proposition 3). Quand activity_guards est disponible, on
        # verifie donc que l'analyse structurelle independante (dominance/post-dominance, zero
        # LLM) confirme au moins l'ABSENCE de contre-indication -- si elle est elle-meme
        # incapable de conclure (STRUCTURE_UNRESOLVED, y compris propage depuis un fork non
        # declare) sur les activites qui ancrent target ou negated_term, le SATISFIED n'est pas
        # invente : retrograde en UNRESOLVABLE, jamais force vers VIOLATED (on ne sait pas non
        # plus que c'est faux, seulement qu'on ne peut pas l'affirmer) -- meme asymetrie que la
        # demotion par confiance ci-dessous, appliquee ici a une deuxieme source d'incertitude.
        if activity_guards is not None and matches is not None:
            affected = _anchoring_activities(target, matches) | _anchoring_activities(negated_term, matches)
            activities = activity_guards.get("activities", {})
            unresolved = sorted(a for a in affected if activities.get(a, {}).get("status") == "STRUCTURE_UNRESOLVED")
            if unresolved:
                return TermVerdict(
                    term=label, status=Status.UNRESOLVABLE,
                    reason=(
                        f"a path avoiding {negated_term!r} exists in the flattened process "
                        f"graph, but independent structural guard analysis (dominance/"
                        f"post-dominance, no LLM) cannot confirm exclusivity around: "
                        f"{unresolved} -- reachability alone does not establish that the "
                        f"model guarantees these outcomes are mutually exclusive"
                    ),
                )
        return TermVerdict(term=label, status=Status.SATISFIED)

    # Candidat VIOLATED -- tous les chemins des co-termes positifs ancres vers la cible
    # passent par negated_term. Meme demotion par confiance que le cas positif.
    weak = tuple(s for s in (negated_term, target) if confidence.get(s, 0.0) < tau)
    if weak:
        detail = ", ".join(f"{s}={confidence.get(s, 0.0):.3f}" for s in weak)
        return TermVerdict(
            term=label, status=Status.UNRESOLVABLE, low_confidence=weak,
            reason=f"would be VIOLATED (every path from an anchored co-term to the target "
                   f"requires {negated_term!r} to occur first, contradicting the text's "
                   f"negation), but match confidence below {tau} for: {detail} -- cannot "
                   f"conclude on an uncertain match",
        )
    return TermVerdict(
        term=label, status=Status.VIOLATED,
        reason=f"every path from an anchored co-term to the target requires {negated_term!r} "
               f"to occur first -- the process forces what the text says must not happen",
    )


def _combine(statuses: list[Status], operator: str) -> Status:
    """Regle d'agregation generique sur une liste de Status -- AND : UNRESOLVABLE domine
    VIOLATED (on ne peut rien conclure tant qu'un element reste indecidable) ; OR : SATISFIED
    domine, sinon UNRESOLVABLE si un element est indecidable, VIOLATED seulement si tous les
    elements sont resolus et tous violes.

    v3 : generalisee depuis l'ancienne `aggregate(term_verdicts, operator)` pour operer sur de
    simples Status plutot que sur des TermVerdict -- reutilisee maintenant a DEUX niveaux dans
    solve() (AND a l'interieur d'une clause, OR entre les clauses d'un guard mixte), la meme
    regle de domination s'appliquant identiquement aux deux echelles."""
    if operator == "AND":
        if Status.UNRESOLVABLE in statuses:
            return Status.UNRESOLVABLE
        if Status.VIOLATED in statuses:
            return Status.VIOLATED
        return Status.SATISFIED
    else:  # OR
        if Status.SATISFIED in statuses:
            return Status.SATISFIED
        if Status.UNRESOLVABLE in statuses:
            return Status.UNRESOLVABLE
        return Status.VIOLATED


def _describe_operator(clauses: tuple[tuple[str, ...], ...]) -> str:
    """Label resume purement informatif ('AND'/'OR'/'MIXED') derive de la forme des clauses --
    jamais consulte par la logique d'agregation elle-meme (solve() opere sur `clauses`
    directement). Une seule clause = AND pur (ou terme seul) ; plusieurs clauses toutes
    singleton = OR pur ; sinon, un vrai melange."""
    if len(clauses) == 1:
        return "AND"
    if all(len(c) == 1 for c in clauses):
        return "OR"
    return "MIXED"


def solve(tgms: TwinGraphMilestoneSystem, tau: float) -> dict[str, GuardVerdict]:
    """Point d'entree du solveur -- calcule Status_tau pour chaque guard de G. Un seul passage,
    pas d'iteration de point fixe explicite necessaire : le fragment restreint (DNF a un
    niveau, guards non-recursifs) rend le calcul direct terme par terme.

    v3 : agregation a DEUX niveaux au lieu d'un seul -- AND a l'interieur de chaque clause,
    puis OR entre les clauses. Un guard pur AND (une seule clause) ou pur OR (clauses
    singleton) retombe exactement sur l'ancien comportement a un seul niveau (verifie par test
    d'equivalence, cf. tests de ce module).

    v4 (cette revision) : gere desormais les termes negatifs ('NOT Y') a l'interieur d'une
    clause -- jusqu'ici silencieusement perdus par tgms_from_pipeline (e['from'] deja
    NOT-strippe par graph_node.py, e['negated'] jamais lu), ce qui faisait traiter
    'X AND NOT Y' comme 'X AND Y', un contresens semantique, pas seulement une perte
    d'information. Chaque clause est desormais partitionnee en termes positifs (verifies
    exactement comme avant, status_of_term inchange) et termes negatifs (status_of_negated_term,
    qui utilise les co-termes POSITIFS de la MEME clause comme points d'ancrage -- voir sa
    docstring). Une clause sans aucun terme negatif retombe exactement sur le comportement
    precedent (positive_terms == clause_terms, negative_terms == ()) -- non-regression totale
    pour tout guard deja construit avant ce changement."""
    confidence = best_confidence(tgms)
    anchored = anchored_milestones(tgms)
    adjacency = _build_adjacency(tgms.process_edges)

    results: dict[str, GuardVerdict] = {}
    for target, guard in tgms.guards.items():
        clause_verdicts = []
        for clause_terms in guard.clauses:
            positive_terms = tuple(t for t in clause_terms if not _is_negated(t))
            negative_terms = tuple(_strip_not(t) for t in clause_terms if _is_negated(t))

            term_verdicts = tuple(
                status_of_term(tgms, term, target, anchored, confidence, adjacency, tau)
                for term in positive_terms
            ) + tuple(
                status_of_negated_term(neg, positive_terms, target, anchored, confidence, adjacency, tau,
                                        matches=tgms.matches, activity_guards=tgms.activity_guards)
                for neg in negative_terms
            )
            clause_status = _combine([t.status for t in term_verdicts], "AND")
            clause_verdicts.append(ClauseVerdict(terms=term_verdicts, status=clause_status))

        overall_status = _combine([cv.status for cv in clause_verdicts], "OR")
        results[target] = GuardVerdict(
            target=target,
            operator=_describe_operator(guard.clauses),
            status=overall_status,
            clauses=tuple(clause_verdicts),
        )
    return results


"""
============================================================================================
ADAPTATEURS -- format dict historique (reference_graph/process_graph/matches) <-> TGMS
============================================================================================

Note d'instantiation : la Definition 3 pose phi: A -> M au niveau ACTIVITE (le vocabulaire
brut des labels BPMN). Le pipeline actuel realise cette projection activite->milestone plus
tot, dans state_matching_node.build_process_state_graph() : process_graph est deja un graphe
sur le vocabulaire des MILESTONES. Cet adaptateur instancie donc le TGMS general avec phi =
IDENTITE sur les milestones apparies -- une instance valide de la Definition 3, pas un
raccourci qui la contredit : le mapping activite->milestone general est realise
structurellement par l'etape en amont du pipeline, pas recalcule ici.
"""


def tgms_from_pipeline(
    reference_graph: dict,
    process_graph: list[dict],
    matches: dict,
    activity_guards: dict | None = None,
) -> tuple[TwinGraphMilestoneSystem, dict[str, str | None]]:
    """Construit un TGMS a partir du format dict du pipeline. Retourne aussi un dict
    target -> quote (metadonnee de report_node.py, hors du TGMS formel -- jamais utilisee par
    le solveur, seulement pour reconstituer le format de sortie historique).

    activity_guards (proposition 2, additif) : cf. TwinGraphMilestoneSystem.activity_guards --
    propage tel quel sur le TGMS construit, None si non fourni (comportement identique a avant
    cette proposition).

    v3 : graph_node.py::build_edges tague desormais chaque arete avec un index `clause` (DNF)
    plutot qu'un `operator` unique par cible -- necessaire pour representer un guard mixte.
    Repli explicite conserve pour tout reference_graph deja genere avant ce changement (aretes
    avec 'operator' mais sans 'clause') : reconstruit exactement l'ancienne semantique plate
    (AND -> une seule clause avec tous les termes ; OR -> une clause PAR terme, chacun son
    propre OR-alternative) -- non-regression garantie sur les graphes deja sur disque, pas
    seulement sur les nouveaux."""
    milestones = frozenset(node["id"] for node in reference_graph["nodes"])

    per_target: dict[str, list[dict]] = {}
    quotes: dict[str, str | None] = {}
    for edge in reference_graph["edges"]:
        per_target.setdefault(edge["to"], []).append(edge)
        quotes.setdefault(edge["to"], edge.get("quote"))

    guards = {}
    for target, target_edges in per_target.items():
        if all("clause" in e for e in target_edges):
            by_clause: dict[int, list[str]] = {}
            for e in target_edges:
                # CORRECTIF (bug bloquant, decouvert par inspection, pas par un run) :
                # graph_node.py::build_edges stocke deja e["from"] NOT-strippe (le noeud brut,
                # pour la connectivite du graphe) et porte la polarite SEPAREMENT dans
                # e["negated"]. Cette ligne ne lisait jusqu'ici que e["from"], jamais
                # e["negated"] -- "X AND NOT Y" devenait silencieusement "X AND Y" dans le
                # Guard, un contresens semantique (le solveur cherchait un chemin vers Y comme
                # si c'etait une exigence positive), pas seulement une perte d'information.
                # Fixe en reappliquant le prefixe "NOT " ici -- meme convention de
                # representation textuelle que precondition_prompt.py/precondition_node_with_
                # retry.py, desormais geree en aval par status_of_negated_term (cf. solve()).
                term = ("NOT " if e.get("negated") else "") + e["from"]
                by_clause.setdefault(e["clause"], []).append(term)
            ordered_clauses = tuple(tuple(by_clause[idx]) for idx in sorted(by_clause))
        else:
            # Format historique (pre-DNF) : un seul 'operator' partage par toutes les aretes
            # de cette cible. Anterieur a la fois a la structure DNF ET a NOT -- aucun champ
            # "negated" n'existe sur ces aretes, rien a preserver ici, comportement inchange.
            operator = target_edges[0].get("operator", "AND")
            if operator == "AND":
                ordered_clauses = (tuple(e["from"] for e in target_edges),)
            else:
                ordered_clauses = tuple((e["from"],) for e in target_edges)
        guards[target] = Guard(clauses=ordered_clauses)

    process_edges = frozenset((edge["from"], edge["to"]) for edge in process_graph)

    # CORRECTIF (bug bloquant) : state_matching_node.py ne renvoie plus, par activite, un
    # match unique a plat ({"match": ..., "score": ...}) -- il renvoie desormais
    # {"activity_label": ..., "matches": [{"match", "score"}, ...], "clusters": [...]}
    # (top-k relatif + clustering additif). L'ancien code (`m["match"]`, `m["score"]`
    # directement sur une entree de matches.values()) levait un KeyError des la premiere
    # execution avec ce nouveau format -- corrige en iterant explicitement sur la liste
    # "matches" de chaque activite.
    matched_milestones: set[str] = set()
    confidence: dict[str, float] = {}
    activity_matches: dict[str, tuple[str, ...]] = {}
    activity_clusters: dict[str, tuple[tuple[str, ...], ...]] = {}

    for activity_id, entry in matches.items():
        activity_match_list = entry.get("matches", [])
        milestones_for_activity = []
        for m in activity_match_list:
            state, score = m["match"], m["score"]
            matched_milestones.add(state)
            milestones_for_activity.append(state)
            if state not in confidence or score > confidence[state]:
                confidence[state] = score
        if milestones_for_activity:
            activity_matches[activity_id] = tuple(milestones_for_activity)

        clusters = entry.get("clusters", [])
        if clusters:
            activity_clusters[activity_id] = tuple(tuple(cluster) for cluster in clusters)

    # `anchor` reste une identite milestone -> milestone (voir note dans la docstring de
    # TwinGraphMilestoneSystem) : c'est ce que consomme le solveur BFS existant, inchange.
    anchor = {milestone: milestone for milestone in matched_milestones}

    tgms = TwinGraphMilestoneSystem(
        milestones=milestones, guards=guards, process_edges=process_edges,
        anchor=anchor, confidence=confidence,
        activity_matches=activity_matches, activity_clusters=activity_clusters,
        matches=matches, activity_guards=activity_guards,
    )
    return tgms, quotes


def verdicts_to_pipeline_alignment(
    verdicts: dict[str, GuardVerdict],
    quotes: dict[str, str | None],
) -> dict:
    """Reconstitue le format dict historique de check_alignment(), consomme sans modification
    par report_node.py/analyze_dataset_runs.py :
    {target: {"operator", "quote", "terms": [{"term","status","reason","missing", ["low_confidence"]}], "status"}}.

    v3 : ajoute `clauses` de facon PUREMENT ADDITIVE (structure DNF explicite : OR de clauses,
    chaque clause son statut AND-agrege et ses termes) -- jamais consomme par report_node.py
    (qui ne lit que "terms"/"quote"/"status", inchange), disponible pour un consommateur futur
    qui voudrait afficher ou auditer la structure logique complete plutot que la liste aplatie.
    `terms` reste exactement l'aplatissement de `clauses` dans le meme ordre qu'avant (propriete
    GuardVerdict.terms) -- zero changement de comportement pour report_node.py."""
    alignment: dict = {}
    for target, verdict in verdicts.items():
        terms = []
        for t in verdict.terms:
            entry = {
                "term": t.term,
                "status": t.status.value,
                "reason": t.reason,
                "missing": list(t.missing),
            }
            if t.low_confidence:
                entry["low_confidence"] = list(t.low_confidence)
            terms.append(entry)
        alignment[target] = {
            "operator": verdict.operator,
            "quote": quotes.get(target),
            "terms": terms,
            "status": verdict.status.value,
            "clauses": [
                {"terms": [t.term for t in c.terms], "status": c.status.value}
                for c in verdict.clauses
            ],
        }
    return alignment