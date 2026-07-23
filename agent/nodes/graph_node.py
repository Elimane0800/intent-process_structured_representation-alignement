import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "graph_construction")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
PRECONDITION_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "precondition_extraction_with_retry")


def load_precondition_run(run_filename: str) -> dict:
    """Loads a run from precondition_extraction_with_retry. Each entry already contains both
    the state space (nodes) and the validated preconditions (edges) -- nothing else to load."""
    with open(os.path.join(PRECONDITION_DIR, run_filename)) as f:
        return json.load(f)


# --- Deterministic graph construction, no LLM ---

def build_nodes(state_space: dict) -> list[str]:
    return [f"{entity}.{state}" for entity, states in state_space.items() for state in states]


def build_edges(state_space: dict, validated: dict) -> list[dict]:
    """For every state with a non-INITIAL, non-UNRESOLVED precondition, one edge per term of the
    precondition -- whether the terms are joined by AND (conjunction) or OR (disjunction). A
    precondition accepted by Pre's validation is always a PURE AND or a PURE OR (mixed logic
    never survives that validation), so a single alternation split is safe regardless of which
    operator produced it.

    Each edge now carries the 'operator' that produced it ('AND' or 'OR') and the 'quote' that
    grounded it. Because the grammar forbids mixing AND/OR in a single precondition, every edge
    pointing to the same target necessarily shares the same operator AND the same quote (the
    quote justifies the whole precondition, not a single term) -- there is only ever one group
    per target, so both tags are enough to reconstruct Pre(u) exactly from the flattened edge
    list, without going back to the raw validated Pre entry.

    No defensive handling needed for a missing quote: _validate_one() never accepts a
    precondition (status 'OK') without a verified quote passing _validate_quote() first, so any
    edge reaching this point always has a real, grounded quote -- never None.

    Typed 'sequence' if the term belongs to the same entity as the target (progression within one
    entity's own lifecycle), 'cross_entity_guard' otherwise (dependency on another entity).

    Defensive: both the target key and every term are checked against the actual state space --
    upstream validation (Pre) should already guarantee terms are valid, but the target key itself
    was never checked there, so a malformed or hallucinated key in the raw LLM output could
    otherwise slip through as an edge endpoint that is never a proper node."""
    valid_targets = {f"{entity}.{state}" for entity, states in state_space.items() for state in states}
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

        has_and = " AND " in precondition
        has_or = " OR " in precondition
        if has_and and has_or:
            # Should never happen -- Pre's own validation already rejects mixed AND/OR into
            # UNRESOLVED before this point. Defensive only: skip rather than guess an operator.
            print(f"    [graph] skipping edge(s): mixed AND/OR precondition for {target!r}: {precondition!r}")
            continue
        operator = "OR" if has_or else "AND"

        for term in re.split(r" AND | OR ", precondition):
            if term not in valid_targets:
                print(f"    [graph] skipping edge: unknown term {term!r} -> {target!r}")
                continue
            term_entity = term.split(".")[0]
            edge_type = "sequence" if term_entity == target_entity else "cross_entity_guard"
            edges.append({"from": term, "to": target, "type": edge_type, "operator": operator, "quote": quote})
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
    adj = {}
    for edge in edges:
        adj.setdefault(edge["from"], []).append(edge["to"])
    return adj


def detect_cycles(edges: list[dict]) -> list[list[str]]:
    """DFS-based cycle detection over the precondition graph."""
    adj = _adjacency(edges)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    cycles = []

    def dfs(node, path):
        color[node] = GRAY
        path.append(node)
        for neighbor in adj.get(node, []):
            if color.get(neighbor, WHITE) == WHITE:
                dfs(neighbor, path)
            elif color.get(neighbor) == GRAY:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
        path.pop()
        color[node] = BLACK

    for node in list(adj.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node, [])
    return cycles


def build_graph(state_space: dict, validated: dict) -> dict:
    nodes = build_nodes(state_space)
    edges = build_edges(state_space, validated)
    weights = compute_node_weights(nodes, edges)
    cycles = detect_cycles(edges)
    return {
        "nodes": [{"id": n, "weight": weights[n]} for n in nodes],
        "edges": edges,
        "cycles": cycles,
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


def build_figure(graph: dict, title: str):
    """Builds a static matplotlib Figure (networkx used only for the DAG/topo-sort machinery
    behind the layered layout). Node size/color reflect structural weight; edge style
    distinguishes sequence (same-entity progression, solid) from cross_entity_guard
    (dependency on another entity, dashed). Edges are drawn as arrows via FancyArrowPatch so
    the direction of the flow stays visible even without interactivity."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import networkx as nx

    G = nx.DiGraph()
    for node in graph["nodes"]:
        G.add_node(node["id"], weight=node["weight"])
    for edge in graph["edges"]:
        G.add_edge(edge["from"], edge["to"], type=edge["type"])

    pos = _compute_layered_positions(G)

    node_ids = list(G.nodes)
    node_x = [pos[n][0] for n in node_ids]
    node_y = [pos[n][1] for n in node_ids]
    node_weight = [G.nodes[n].get("weight", 0.0) for n in node_ids]

    # Figure size grows with the number of layers / max layer width so a big process doesn't
    # get squeezed into a fixed canvas -- that's what made the old spring_layout plots feel
    # cramped and tangled on larger graphs too.
    max_x = max(node_x, default=0)
    max_y_span = max((abs(y) for y in node_y), default=0)
    width_px = max(1200, int(max_x + 400))
    height_px = max(700, int(2 * max_y_span + 300))
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    cmap = mpl.colormaps["YlOrRd"]
    norm = mcolors.Normalize(vmin=0, vmax=max(node_weight, default=1) or 1)

    edge_style = {
        "sequence": dict(color="#4C78A8", linestyle="solid"),
        "cross_entity_guard": dict(color="#E45756", linestyle="dashed"),
    }
    for u, v, data in G.edges(data=True):
        style = edge_style[data["type"]]
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=14,
            color=style["color"], linestyle=style["linestyle"],
            linewidth=1.5, shrinkA=14, shrinkB=14,
            connectionstyle="arc3,rad=0.05", zorder=1,
        )
        ax.add_patch(arrow)

    node_sizes = [80 + 400 * w for w in node_weight]
    scatter = ax.scatter(
        node_x, node_y, s=node_sizes,
        c=node_weight, cmap=cmap, norm=norm,
        edgecolors="#333333", linewidths=1, zorder=2,
    )
    for n, (x, y) in pos.items():
        ax.annotate(n, (x, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, zorder=3)

    fig.colorbar(scatter, ax=ax, label="weight", shrink=0.8)

    legend_handles = [
        Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], lw=1.5, label=etype)
        for etype, s in edge_style.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("process progression →")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(min(node_x, default=0) - 100, max(node_x, default=0) + 100)
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
                    graph = build_graph(state_space, validated)
                    results[prompt_name][model_name][case_name] = graph

                    print(f"[{prompt_name}][{model_name}][{case_name}] "
                          f"nodes={len(graph['nodes'])} edges={len(graph['edges'])} cycles={len(graph['cycles'])}")

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