import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "graph_construction")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
PRECONDITION_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "precondition_extraction_with_retry")

# v2 vs v1 -- two integration fixes, catching up with upstream changes this file had fallen
# behind on. Neither is a new feature of THIS file; both are this file re-synchronizing with a
# grammar/schema change that already happened elsewhere, the same category of bug already
# documented (and fixed) twice before in this project:
#
#   (a) state_space shape -- build_nodes/build_edges used to do `for state in states` directly
#       on an entity's raw value, assuming the legacy bare-list shape. Since Protocol 3.2
#       (state_space_node_v3.py), that value is {"states": [...], "exclusive_groups": [...],
#       "concurrent_with": [...]}, so iterating it directly yields its KEYS, not real state
#       labels -- silently producing bogus nodes/targets like "job.states". This is the EXACT
#       same bug _entity_states() was introduced to fix in precondition_node_with_retry.py; it
#       was simply never ported here, because this file deliberately duplicates rather than
#       imports (see parse_precondition's docstring below) and duplication was not kept in sync.
#       Fixed the same way: a local _entity_states() helper, used everywhere state_space is
#       iterated.
#
#   (b) NOT (v4) -- parse_precondition/_split_top_level here were still the pre-v4 copy: they
#       happily parse a "NOT entity.state" term (NOT is a term-content prefix, never a
#       separator, so the DNF splitter itself needed no change -- same observation already made
#       in precondition_node_with_retry.py when NOT was added there). But build_edges compared
#       the raw, un-stripped term against valid_targets, so every negated term failed that
#       check and was silently dropped (a print, never an edge) -- a real dependency the
#       precondition graph is missing purely because this file did not catch up. Fixed by
#       stripping "NOT " (via _strip_not) before validating/typing a term, and by carrying the
#       polarity forward on the edge itself ("negated": bool) rather than discarding it -- a
#       negated term still references the SAME underlying node for graph purposes, exactly as
#       precondition_node_with_retry.py's own build_graph() already does. detect_cycles() is
#       updated to match: a cycle built entirely of negated edges is how this grammar expresses
#       two mutually exclusive branch outcomes, not a genuine circular dependency, so it is no
#       longer reported -- mirrors detect_cycles() in precondition_node_with_retry.py exactly,
#       for the same structural reason (never tuned to a specific text or backbone).


def load_precondition_run(run_filename: str) -> dict:
    """Loads a run from precondition_extraction_with_retry. Each entry already contains both
    the state space (nodes) and the validated preconditions (edges) -- nothing else to load."""
    with open(os.path.join(PRECONDITION_DIR, run_filename)) as f:
        return json.load(f)


# --- Deterministic graph construction, no LLM ---

def _entity_states(value) -> list[str]:
    """Extracts the plain list of state labels from a state-space entity's value. Accepts both
    the legacy bare-list shape and the {"states": [...], "exclusive_groups": [...],
    "concurrent_with": [...]} shape state_space_node_v3.py has produced since Protocol 3.2.

    Copy of precondition_node_with_retry.py::_entity_states -- same duplication discipline as
    parse_precondition below (this file stays autonomous rather than importing), so any future
    fix to one copy must be mirrored to the other by hand."""
    if isinstance(value, dict):
        return list(value.get("states", []))
    return list(value)


def build_nodes(state_space: dict) -> list[str]:
    return [f"{entity}.{state}" for entity, value in state_space.items() for state in _entity_states(value)]


def _split_top_level(s: str, sep: str) -> list[str]:
    """Coupe `s` sur `sep` uniquement au niveau 0 de parenthesage -- dupliquee depuis
    precondition_node_with_retry.py::parse_precondition, PAS importee : ce fichier reste
    autonome par discipline explicite du projet (deja le cas pour la detection de cycles,
    recopiee plutot qu'importee, cf. docstring de detect_cycles ci-dessous)."""
    parts, depth, buf, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif depth == 0 and s[i:i + len(sep)] == sep:
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _is_negated(term: str) -> bool:
    return term.startswith("NOT ")


def _strip_not(term: str) -> str:
    """Strips exactly one leading 'NOT ' if present, else returns the term unchanged. Copy of
    precondition_node_with_retry.py::_strip_not -- see that file for the rationale (a malformed
    'NOT' with nothing after it is left untouched on purpose, so it fails the valid_targets
    check naturally like any other unknown term)."""
    return term[len("NOT "):] if _is_negated(term) else term


def parse_precondition(precondition: str) -> tuple[list[tuple[str, ...]], str | None]:
    """Parse une precondition en DNF a un seul niveau (liste de clauses-ET jointes par OU).
    Copie exacte de precondition_node_with_retry.py::parse_precondition -- toute correction
    faite ici doit etre repercutee la-bas et vice versa (meme discipline de duplication
    assumee que pour detect_cycles). Voir ce fichier pour la grammaire complete commentee.

    NOT (v4) ne change rien ici structurellement, exactement comme note dans la version source :
    "NOT entity.state" est un prefixe sur le CONTENU d'un terme, jamais un separateur -- il
    traverse le split sur " AND "/" OR " intact et se retrouve comme un seul element de tuple.
    L'interpretation de ce prefixe (existence, type d'arete, cycles) est geree en aval, dans
    build_edges/detect_cycles, jamais ici -- ce qui explique pourquoi cette fonction elle-meme
    n'avait en realite pas besoin d'etre modifiee pour v4, seuls ses appelants l'avaient."""
    raw_clauses = _split_top_level(precondition, " OR ")

    if len(raw_clauses) == 1:
        raw = raw_clauses[0].strip()
        if raw.startswith("(") and raw.endswith(")"):
            raw = raw[1:-1].strip()
        elif raw.startswith("(") or raw.endswith(")"):
            return [], f"unbalanced parentheses: {raw!r}"
        terms = tuple(t.strip() for t in raw.split(" AND "))
        if any(not t for t in terms):
            return [], f"empty term in precondition: {precondition!r}"
        return [terms], None

    clauses: list[tuple[str, ...]] = []
    for raw in raw_clauses:
        raw = raw.strip()
        if not raw:
            return [], "empty clause between OR"
        if raw.startswith("(") or raw.endswith(")"):
            if not (raw.startswith("(") and raw.endswith(")")):
                return [], f"unbalanced parentheses in clause: {raw!r}"
            inner = raw[1:-1].strip()
            terms = tuple(t.strip() for t in inner.split(" AND "))
            if len(terms) < 2:
                return [], f"parenthesized clause must contain at least two AND-terms: {raw!r}"
        else:
            if " AND " in raw:
                return [], (f"AND used without parentheses in a clause combined with OR "
                            f"(ambiguous): {raw!r} -- wrap AND-groups in parentheses")
            if "(" in raw or ")" in raw:
                return [], f"unbalanced parentheses: {raw!r}"
            terms = (raw,)
        clauses.append(terms)
    return clauses, None


def build_edges(state_space: dict, validated: dict) -> list[dict]:
    """For every state with a non-INITIAL, non-UNRESOLVED precondition, one edge per term of the
    precondition, across every AND-clause of its (possibly mixed) DNF structure.

    v2: a precondition accepted by Pre's validation was always a PURE AND or a PURE OR (mixed
    logic never survived that validation) -- a single 'operator' tag per target was therefore
    enough to reconstruct Pre(u) exactly. v3 lifts that restriction (parse_precondition now
    accepts a one-level DNF, i.e. an OR of AND-clauses, see precondition_node_with_retry.py) --
    an unjustified restriction, not a real grammar limit: the text can genuinely express
    "(A AND B) OR C". A single flat 'operator' can no longer represent that shape, so each edge
    now carries a 'clause' index instead: all edges sharing the same (target, clause) form one
    AND-group, and different clause indices for the same target are the OR-alternatives.
    Reconstructing Pre(u) exactly from the flattened edge list means grouping by 'to', then by
    'clause' -- still without going back to the raw validated Pre entry.

    v4 (this revision): every term is validated and typed on its NOT-stripped form
    (_strip_not), catching up with the same negation support already added upstream in
    precondition_prompt.py/precondition_node_with_retry.py. Each edge now also carries
    "negated": bool, recording the polarity that produced it -- a negated term still points at
    the SAME underlying node for graph/connectivity purposes ("A requires NOT B" is still an
    edge B -> A, just a negative one), exactly as precondition_node_with_retry.py's own
    build_graph() already does. Before this fix, a "NOT entity.state" term was compared
    UN-stripped against valid_targets, always failed that check, and was silently dropped (a
    print, never an edge) -- every negative dependency the upstream grammar can express since
    v4 was invisible to the graph, not because it was invalid, but because this file had not
    caught up.

    Each edge still carries the 'quote' that grounded the WHOLE precondition (not a single
    clause) -- unchanged: _validate_one() never accepts a precondition (status 'OK') without a
    verified quote covering the whole rule, so any edge reaching this point always has a real,
    grounded quote, never None.

    Typed 'sequence' if the term belongs to the same entity as the target (progression within one
    entity's own lifecycle), 'cross_entity_guard' otherwise (dependency on another entity). Entity
    extraction now also happens on the NOT-stripped term/target -- comparing "NOT entity.state"
    against valid_targets, or splitting it on "." for its entity, previously either always failed
    (see above) or would have read the entity name as "NOT entity", never the real one.

    Defensive: both the target key and every term are checked against the actual state space --
    upstream validation (Pre) should already guarantee terms are valid, but the target key itself
    was never checked there, so a malformed or hallucinated key in the raw LLM output could
    otherwise slip through as an edge endpoint that is never a proper node. A malformed DNF
    string is also defended against here (should never happen -- Pre's own validation already
    rejects it before this point) rather than assumed impossible."""
    valid_targets = {f"{entity}.{state}" for entity, value in state_space.items() for state in _entity_states(value)}
    edges = []
    for target, entry in validated.items():
        precondition = entry.get("precondition") if isinstance(entry, dict) else entry
        quote = entry.get("quote") if isinstance(entry, dict) else None
        if precondition in ("INITIAL", "UNRESOLVED", None):
            continue
        if target not in valid_targets:
            print(f"    [graph] skipping edge(s) to unknown target: {target!r}")
            continue
        target_entity = target.split(".")[0]

        clauses, parse_error = parse_precondition(precondition)
        if parse_error:
            # Should never happen -- Pre's own validation already rejects a malformed DNF
            # string into UNRESOLVED before this point. Defensive only: skip rather than guess
            # a structure.
            print(f"    [graph] skipping edge(s): malformed precondition for {target!r}: "
                  f"{precondition!r} ({parse_error})")
            continue

        for clause_idx, clause in enumerate(clauses):
            for term in clause:
                node = _strip_not(term)
                if node not in valid_targets:
                    print(f"    [graph] skipping edge: unknown term {term!r} -> {target!r}")
                    continue
                term_entity = node.split(".")[0]
                edge_type = "sequence" if term_entity == target_entity else "cross_entity_guard"
                edges.append({
                    "from": node, "to": target, "clause": clause_idx,
                    "type": edge_type, "quote": quote, "negated": _is_negated(term),
                })
    return edges


def compute_node_weights(nodes: list[str], edges: list[dict]) -> dict:
    """Structural importance, computed once the graph is fully assembled: degree centrality
    (in-degree + out-degree), normalized by the maximum degree observed in this graph. This is
    distinct from precondition confidence/grounding, which is already established upstream in Pre."""
    degree = {node: 0 for node in nodes}
    for edge in edges:
        degree[edge["from"]] = degree.get(edge["from"], 0) + 1
        degree[edge["to"]] = degree.get(edge["to"], 0) + 1

    max_degree = max(degree.values(), default=0)
    if max_degree == 0:
        return {node: 0.0 for node in nodes}
    return {node: round(degree.get(node, 0) / max_degree, 3) for node in nodes}


def _adjacency(edges: list[dict]) -> dict:
    """adjacency: node -> list of (neighbor, is_negated) pairs, one per edge, carrying the
    polarity along -- needed by detect_cycles() to tell a cycle made entirely of negated edges
    apart from one containing at least one positive dependency (see its docstring)."""
    adj: dict[str, list[tuple[str, bool]]] = {}
    for edge in edges:
        adj.setdefault(edge["from"], []).append((edge["to"], edge.get("negated", False)))
    return adj


def detect_cycles(edges: list[dict]) -> list[list[str]]:
    """DFS-based cycle detection over the precondition graph.

    v4: only reports a cycle if at least one edge along it is a POSITIVE (non-negated)
    dependency -- mirrors detect_cycles() in precondition_node_with_retry.py exactly, and for
    the same structural reason (never tuned to a specific text or backbone): a cycle made
    ENTIRELY of negated edges (e.g. "A requires NOT B" and "B requires NOT A") is exactly how
    this grammar expresses two mutually exclusive outcomes of the same branch point, not a
    genuine circular causal chain. Before this fix, this file had no notion of edge polarity at
    all (see _adjacency's previous version), so it could never have made this distinction in
    the first place -- any all-negation branch pattern already reaching this function would
    have been flagged as a false-positive cycle."""
    adj = _adjacency(edges)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    cycles = []

    def dfs(node, path, path_edges):
        color[node] = GRAY
        path.append(node)
        for neighbor, is_negated in adj.get(node, []):
            if color.get(neighbor, WHITE) == WHITE:
                path_edges.append(is_negated)
                dfs(neighbor, path, path_edges)
                path_edges.pop()
            elif color.get(neighbor) == GRAY:
                cycle_start = path.index(neighbor)
                cycle_edges = path_edges[cycle_start:] + [is_negated]
                if not all(cycle_edges):  # at least one positive (non-negated) edge present
                    cycles.append(path[cycle_start:] + [neighbor])
        path.pop()
        color[node] = BLACK

    for node in list(adj.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node, [], [])
    return cycles


def _exclusive_pairs(state_space: dict) -> list[dict]:
    """Every exclusive_groups pair declared in the state space, as graph-node id pairs -- e.g.
    {"entity": "job", "nodes": ["job.permanent", "job.not_permanent"]}. This is the XOR
    relationship BETWEEN two target nodes (an entity can hold only one of them), distinct from
    an AND/OR edge (a dependency FROM a term TO a target) -- drawn separately in build_figure as
    an undirected connector, never as a directed edge. Returned regardless of whether this
    specific pair is an active branch_conflicts entry -- build_figure styles a pair differently
    depending on whether it also appears in branch_conflicts (a real, detected problem) or not
    (a declared-but-currently-uncontradicted exclusivity, still worth showing so the reader sees
    the relationship exists at all)."""
    pairs = []
    for entity, value in state_space.items():
        groups = value.get("exclusive_groups", []) if isinstance(value, dict) else []
        for group in groups:
            nodes = [f"{entity}.{state}" for state in group]
            if len(nodes) >= 2:
                pairs.append({"entity": entity, "nodes": nodes})
    return pairs


def build_graph(state_space: dict, validated: dict, branch_conflicts: list | None = None) -> dict:
    nodes = build_nodes(state_space)
    edges = build_edges(state_space, validated)
    weights = compute_node_weights(nodes, edges)
    cycles = detect_cycles(edges)
    return {
        "nodes": [{"id": n, "weight": weights[n]} for n in nodes],
        "edges": edges,
        "cycles": cycles,
        # Both new: previously computed upstream by check_branch_coherence() in
        # precondition_node_with_retry.py and silently dropped here -- the exact same
        # integration gap already fixed twice in this file for state_space shape and NOT,
        # just discovered later since branch_conflicts was added to the upstream file after
        # those two fixes. "branch_conflicts" is the subset of "exclusive_pairs" (below) that
        # validate_preconditions actually found sharing an identical accepted precondition; kept
        # as its own field (rather than only inferred by cross-referencing exclusive_pairs
        # elsewhere) so nothing downstream has to recompute what the precondition layer already
        # determined.
        "branch_conflicts": branch_conflicts or [],
        "exclusive_pairs": _exclusive_pairs(state_space),
    }


# --- Layout: Directed-Follows-Graph style (layered left-to-right), not force-directed ---
#
# spring_layout minimizes distances/crossings via attraction-repulsion and has no notion of
# "before/after", even though the edges are directed -- that's why the old plots didn't read
# as a process flow. Here x = causal depth (longest-path distance from a source, à la Sugiyama/
# process-mining DFG layouts), y = position within that depth "layer", refined with a few
# barycenter sweeps to reduce edge crossings. Cycles are broken (classic DFS back-edge
# classification) *only* to compute this acyclic skeleton for positioning -- every original
# edge, including the back edges, is still drawn afterwards; a back edge simply ends up pointing
# right-to-left in the final layout, which reads naturally as "this loops back".

def _compute_layered_positions(G, spacing_x: float = 220.0, spacing_y: float = 110.0) -> dict:
    import networkx as nx

    if len(G.nodes) == 0:
        return {}

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in G.nodes}
    back_edges = set()

    def dfs(u):
        color[u] = GRAY
        for v in G.successors(u):
            if color[v] == WHITE:
                dfs(v)
            elif color[v] == GRAY:
                back_edges.add((u, v))
        color[u] = BLACK

    for n in G.nodes:
        if color[n] == WHITE:
            dfs(n)

    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes)
    dag.add_edges_from(e for e in G.edges if e not in back_edges)

    # Layer = longest path from a source. Processing in topological order guarantees every
    # predecessor's layer is already final when a node is visited.
    layer = {n: 0 for n in G.nodes}
    for n in nx.topological_sort(dag):
        preds = list(dag.predecessors(n))
        if preds:
            layer[n] = max(layer[p] for p in preds) + 1

    layers: dict[int, list] = {}
    for n, l in layer.items():
        layers.setdefault(l, []).append(n)

    # Crossing reduction: order nodes within each layer by the average order of their
    # neighbors, alternating forward/backward sweeps. Not an optimal Sugiyama solve, but cheap
    # and good enough at this graph size.
    order = {n: i for l in sorted(layers) for i, n in enumerate(layers[l])}

    def sweep(forward: bool):
        layer_ids = sorted(layers) if forward else sorted(layers, reverse=True)
        for l in layer_ids:
            nodes_here = layers[l]

            def neighbors_of(n):
                return (list(dag.predecessors(n)) + list(G.predecessors(n))) if forward \
                    else (list(dag.successors(n)) + list(G.successors(n)))

            bary = {}
            for n in nodes_here:
                neigh = [nb for nb in neighbors_of(n) if nb in order]
                bary[n] = (sum(order[nb] for nb in neigh) / len(neigh)) if neigh else order[n]
            nodes_here.sort(key=lambda n: bary[n])
            for i, n in enumerate(nodes_here):
                order[n] = i

    for _ in range(4):
        sweep(forward=True)
        sweep(forward=False)

    pos = {}
    for l, nodes_here in layers.items():
        ordered = sorted(nodes_here, key=lambda n: order[n])
        n_here = len(ordered)
        for i, n in enumerate(ordered):
            x = l * spacing_x
            y = (i - (n_here - 1) / 2) * spacing_y
            pos[n] = (x, y)
    return pos


_GATEWAY_OFFSET_OR = 50.0
_GATEWAY_OFFSET_AND = 95.0
_GATEWAY_STACK_Y = 22.0


def _clause_groups_for_target(edges: list[dict], target: str) -> dict[int, list[dict]]:
    """Every incoming edge of `target`, grouped by clause index -- edges sharing a clause index
    are the terms of one AND-group; different clause indices are OR-alternatives. Mirrors
    exactly how build_edges() itself groups terms (see its docstring), just re-grouped by
    target here for layout purposes."""
    groups: dict[int, list[dict]] = {}
    for e in edges:
        if e["to"] == target:
            groups.setdefault(e["clause"], []).append(e)
    return groups


def _draw_real_edge(ax, p0, p1, edge: dict, edge_style: dict) -> None:
    """Draws one edge that corresponds to an actual precondition term (source -> a gateway, or
    source -> target directly when there is no gateway). Color/linestyle come from the edge's
    "type" (sequence vs cross_entity_guard) exactly as before. Negation is now rendered as a
    Petri-net-style INHIBITOR ARC -- no arrowhead, a small hollow circle near the endpoint
    instead -- rather than the previous, easy-to-miss arrowhead-size trick: a hollow circle is a
    pre-existing, standard notation for "prevents/NOT", not an invented one, and reads
    unambiguously even at a glance or when printed in grayscale."""
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.pyplot as plt

    style = edge_style[edge["type"]]
    negated = edge.get("negated", False)
    arrow = FancyArrowPatch(
        p0, p1, arrowstyle=("-" if negated else "-|>"), mutation_scale=14,
        color=style["color"], linestyle=style["linestyle"],
        linewidth=1.5, shrinkA=14, shrinkB=(20 if negated else 14),
        connectionstyle="arc3,rad=0.05", zorder=1,
    )
    ax.add_patch(arrow)
    if negated:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dist = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        ux, uy = dx / dist, dy / dist
        cx, cy = p1[0] - ux * 16, p1[1] - uy * 16
        ax.add_patch(plt.Circle((cx, cy), radius=5, facecolor="white",
                                 edgecolor=style["color"], linewidth=1.5, zorder=2))


def _draw_structural_edge(ax, p0, p1) -> None:
    """Gateway-to-gateway or gateway-to-target connector -- purely compositional (reconstructs
    the DNF shape so the reader can see it at a glance), never itself a grounded precondition
    term. Deliberately neutral (thin, gray) so it never competes visually with the real,
    colored/styled edges that carry actual meaning."""
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=10,
        color="#999999", linestyle="solid", linewidth=1.0,
        shrinkA=10, shrinkB=14, zorder=1,
    )
    ax.add_patch(arrow)


def _draw_and_gateway(ax, xy) -> None:
    """AND-gateway glyph (diamond + "&"), BPMN/Petri-net convention for 'all of these must hold
    together' -- placed where 2+ edges of the same AND-clause converge before continuing on
    (either straight to the target, or into an OR-gateway if the target's precondition mixes
    AND and OR)."""
    from matplotlib.patches import RegularPolygon
    diamond = RegularPolygon(xy, numVertices=4, radius=9, orientation=0.785398,
                              facecolor="#F4E8D8", edgecolor="#8A6D3B", linewidth=1.3, zorder=3)
    ax.add_patch(diamond)
    ax.annotate("&", xy, ha="center", va="center", fontsize=8, fontweight="bold",
                color="#8A6D3B", zorder=4)


def _draw_or_gateway(ax, xy) -> None:
    """OR-gateway glyph (circle + "∨"), placed where 2+ alternative clauses (each possibly
    itself an AND-group) converge on the same target -- any ONE of the incoming branches is
    independently sufficient."""
    import matplotlib.pyplot as plt
    circle = plt.Circle(xy, radius=9, facecolor="#E3EEF9", edgecolor="#3B6E8A", linewidth=1.3, zorder=3)
    ax.add_patch(circle)
    ax.annotate("∨", xy, ha="center", va="center", fontsize=9, fontweight="bold",
                color="#3B6E8A", zorder=4)


def _draw_incoming_structure(ax, pos: dict, target: str, edges: list[dict], edge_style: dict) -> None:
    """Draws everything feeding into `target`, reconstructing its DNF precondition shape exactly
    (see _clause_groups_for_target): a bare single term draws directly; a single AND-clause of
    2+ terms draws through one AND-gateway; 2+ clauses (an OR, each clause itself possibly an
    AND-group) draw through one OR-gateway that all the (possibly AND-gated) alternatives feed
    into. No gateway at all is drawn for a target with no incoming edges (INITIAL/UNRESOLVED)."""
    groups = _clause_groups_for_target(edges, target)
    if not groups:
        return
    tx, ty = pos[target]

    if len(groups) == 1:
        (_, group_edges), = groups.items()
        if len(group_edges) == 1:
            e = group_edges[0]
            _draw_real_edge(ax, pos[e["from"]], (tx, ty), e, edge_style)
        else:
            and_xy = (tx - _GATEWAY_OFFSET_AND, ty)
            _draw_and_gateway(ax, and_xy)
            _draw_structural_edge(ax, and_xy, (tx, ty))
            for e in group_edges:
                _draw_real_edge(ax, pos[e["from"]], and_xy, e, edge_style)
        return

    or_xy = (tx - _GATEWAY_OFFSET_OR, ty)
    _draw_or_gateway(ax, or_xy)
    _draw_structural_edge(ax, or_xy, (tx, ty))
    sorted_groups = sorted(groups.items())
    n = len(sorted_groups)
    for i, (_, group_edges) in enumerate(sorted_groups):
        y_off = (i - (n - 1) / 2) * _GATEWAY_STACK_Y
        if len(group_edges) == 1:
            e = group_edges[0]
            _draw_real_edge(ax, pos[e["from"]], or_xy, e, edge_style)
        else:
            and_xy = (tx - _GATEWAY_OFFSET_OR - _GATEWAY_OFFSET_AND, ty + y_off)
            _draw_and_gateway(ax, and_xy)
            _draw_structural_edge(ax, and_xy, or_xy)
            for e in group_edges:
                _draw_real_edge(ax, pos[e["from"]], and_xy, e, edge_style)


def _draw_xor_connectors(ax, pos: dict, graph: dict) -> None:
    """Undirected connectors between sibling nodes declared exclusive_groups in the state space
    -- a fundamentally different relationship from an AND/OR/NOT edge (it is not a causal
    dependency, it is "these two cannot both hold"), so drawn with its own visual language:
    dashed, unheaded, behind everything else (low zorder). Styled neutrally (purple, thin) when
    the pair is only a DECLARED exclusivity with no detected problem, and prominently (red,
    thick, warning label) when the pair also appears in branch_conflicts -- i.e. the precondition
    layer found both branches sharing an identical accepted precondition, a real contradiction
    of their own declared exclusivity, not merely a state-space claim."""
    import matplotlib.pyplot as plt

    conflict_pairs = {frozenset(c["conflicting_targets"]) for c in graph.get("branch_conflicts", [])}
    for pair in graph.get("exclusive_pairs", []):
        a, b = pair["nodes"][0], pair["nodes"][1]
        if a not in pos or b not in pos:
            continue
        is_conflict = frozenset([a, b]) in conflict_pairs
        (xa, ya), (xb, yb) = pos[a], pos[b]
        color = "#D62728" if is_conflict else "#9467BD"
        lw = 2.2 if is_conflict else 1.1
        ax.plot([xa, xb], [ya, yb], linestyle=(0, (4, 3)), color=color, linewidth=lw, zorder=0.5)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        label = "\u26a0 XOR conflict" if is_conflict else "XOR"
        ax.annotate(
            label, (mx, my), ha="center", va="center", fontsize=7, color=color,
            fontweight="bold" if is_conflict else "normal",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, linewidth=0.8),
            zorder=0.6,
        )


def build_figure(graph: dict, title: str):
    """Builds a static matplotlib Figure (networkx used only for the DAG/topo-sort machinery
    behind the layered layout).

    Node size/color no longer encode structural weight (degree centrality) -- dropped on
    purpose: it was not a useful reading at this stage of the project, and spending a color
    channel on it left nothing free to encode what actually matters here. Nodes are now a
    uniform neutral style, EXCEPT nodes involved in an active branch_conflicts entry, which get
    a bold red highlight -- the one node-level signal worth calling out visually. `weight` is
    still computed and still present in the returned graph JSON (compute_node_weights is
    unchanged) for anything downstream that wants it; it is simply no longer spent on this
    figure's node styling.

    The logical structure of each target's precondition (AND / OR / NOT) is now drawn
    explicitly rather than left for the reader to infer from a flat edge list:
      - AND: a diamond gateway ("&") where 2+ terms of one AND-clause converge.
      - OR: a circle gateway ("∨") where 2+ alternative clauses converge.
      - NOT: an inhibitor-arc-style hollow circle instead of an arrowhead on that one edge.
      - XOR (state-space exclusive_groups, a relationship between two TARGET nodes, not an
        edge): a dashed undirected connector, neutral by default, red with a warning label if
        it also appears in branch_conflicts (both branches share an accepted precondition,
        contradicting their own declared exclusivity).
    Edge color still distinguishes sequence (same-entity progression, blue) from
    cross_entity_guard (dependency on another entity, red/orange dashed) -- unchanged, a
    different semantic axis from AND/OR/NOT/XOR and kept separate rather than overloaded."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    import networkx as nx

    G = nx.DiGraph()
    for node in graph["nodes"]:
        G.add_node(node["id"])
    for edge in graph["edges"]:
        G.add_edge(edge["from"], edge["to"])

    pos = _compute_layered_positions(G)

    node_ids = list(G.nodes)
    node_x = [pos[n][0] for n in node_ids]
    node_y = [pos[n][1] for n in node_ids]

    max_x = max(node_x, default=0)
    max_y_span = max((abs(y) for y in node_y), default=0)
    width_px = max(1200, int(max_x + 500))
    height_px = max(700, int(2 * max_y_span + 300))
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    edge_style = {
        "sequence": dict(color="#4C78A8", linestyle="solid"),
        "cross_entity_guard": dict(color="#E45756", linestyle="dashed"),
    }

    _draw_xor_connectors(ax, pos, graph)
    for target in node_ids:
        _draw_incoming_structure(ax, pos, target, graph["edges"], edge_style)

    conflict_nodes = {t for c in graph.get("branch_conflicts", []) for t in c["conflicting_targets"]}
    node_facecolors = ["#FADBD8" if n in conflict_nodes else "#E8ECF1" for n in node_ids]
    node_edgecolors = ["#C0392B" if n in conflict_nodes else "#333333" for n in node_ids]
    node_linewidths = [2.2 if n in conflict_nodes else 1.0 for n in node_ids]
    ax.scatter(node_x, node_y, s=320, c=node_facecolors, edgecolors=node_edgecolors,
               linewidths=node_linewidths, zorder=5)
    for n, (x, y) in pos.items():
        ax.annotate(n, (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, zorder=6)

    legend_handles = [
        Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], lw=1.5, label=etype)
        for etype, s in edge_style.items()
    ]
    legend_handles += [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#F4E8D8",
               markeredgecolor="#8A6D3B", markersize=9, label="AND gateway"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#E3EEF9",
               markeredgecolor="#3B6E8A", markersize=9, label="OR gateway"),
        Line2D([0], [0], color="#666666", linestyle="solid", lw=1.5, marker="o",
               markerfacecolor="white", markeredgecolor="#666666", markersize=6,
               markevery=[1], label="NOT (inhibitor)"),
        Line2D([0], [0], color="#9467BD", linestyle=(0, (4, 3)), lw=1.2, label="XOR (declared)"),
        Line2D([0], [0], color="#D62728", linestyle=(0, (4, 3)), lw=2.2, label="XOR conflict"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("process progression →")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(min(node_x, default=0) - 160, max(node_x, default=0) + 100)
    ax.set_ylim(min(node_y, default=0) - 80, max(node_y, default=0) + 80)
    fig.tight_layout()
    return fig


def save_graph_visualizations(graph: dict, title: str, png_path: str) -> None:
    """Builds the figure and saves it as a static PNG. (matplotlib is static -- no interactive
    HTML export for now; revisit with e.g. mpld3 if that's needed later.)"""
    import matplotlib.pyplot as plt

    fig = build_figure(graph, title)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_files = sorted(
        f for f in os.listdir(PRECONDITION_DIR) if re.match(r"run_\d+\.json$", f)
    )
    if not run_files:
        raise FileNotFoundError(f"No run_*.json files found in {PRECONDITION_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    for run_filename in run_files:
        print(f"\n########## {run_filename} ##########")
        data = load_precondition_run(run_filename)
        run_stem = run_filename.replace(".json", "")

        results = {}
        for prompt_name in data:
            results[prompt_name] = {}
            for model_name in data[prompt_name]:
                results[prompt_name][model_name] = {}
                for case_name, entry in data[prompt_name][model_name].items():
                    if "error" in entry:
                        results[prompt_name][model_name][case_name] = {"error": entry["error"]}
                        print(f"[{prompt_name}][{model_name}][{case_name}] SKIPPED (upstream error)")
                        continue

                    state_space = entry["state_space_used"]
                    validated = entry["validated"]
                    branch_conflicts = entry.get("branch_conflicts", [])
                    graph = build_graph(state_space, validated, branch_conflicts)
                    results[prompt_name][model_name][case_name] = graph

                    print(f"[{prompt_name}][{model_name}][{case_name}] "
                          f"nodes={len(graph['nodes'])} edges={len(graph['edges'])} "
                          f"cycles={len(graph['cycles'])} branch_conflicts={len(graph['branch_conflicts'])}")

                    safe_model = model_name.replace("/", "_")
                    viz_title = f"{run_stem} | {prompt_name} | {model_name} | {case_name}"
                    viz_stem = f"{run_stem}_{prompt_name}_{safe_model}_{case_name}"
                    save_graph_visualizations(
                        graph, viz_title,
                        png_path=os.path.join(VISUALIZATION_DIR, f"{viz_stem}.png"),
                    )

        # output filename mirrors the source filename -- keeps a clear 1:1 mapping between a
        # precondition run and its resulting graph, rather than a separately incrementing counter
        out_path = os.path.join(RESULTS_DIR, run_filename)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {out_path}")