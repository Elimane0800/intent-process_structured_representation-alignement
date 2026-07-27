"""Solveur INDEPENDANT (aucun import de tgms_solver.py -- reimplementation de zero, pas une
reutilisation) produisant trois scores agreges : CORRECTNESS, COMPLETENESS, RELEVANCE. Le
matching delta (texte<->modele) est assure par un LLM (GPT-5.5), meme prompt/garde-fous que
llm_matching_experiment_guardrails.py (grounding lexical, no-match autorise, vote majoritaire
sur 3 repetitions) -- dupliques ici plutot qu'importes, meme discipline de decouplage que tout
le reste de ce projet.

=== Definitions precises, ancrees dans le formalisme pose en debut de discussion ===

Rappel : Fidelite(M, T) = comparaison(psi(M), phi(T)), avec psi/phi deux fonctions
d'interpretation, et deux axes -- structurel (Rel, profil comportemental) et semantique (delta,
denotation). Les trois scores ci-dessous sont des agregats CALCULABLES sur des donnees deja
produites par ce pipeline, pas trois notions nouvelles :

  COMPLETENESS = surjectivite de delta = |milestones de U ancres par >=1 activite de M| /
                 |milestones de U total|.
                 Repond a : "tout ce que le texte affirme est-il represente quelque part dans
                 le modele ?" -- LIMITE ASSUMEE : ne distingue pas encore une vraie absence
                 d'un artefact de matching non recupere (cf. Table O/lost_to_margin, discute
                 plus tot) -- ce score mesure delta tel qu'il est aujourd'hui, pas une verite
                 corrigee du bruit de matching.

  RELEVANCE = injectivite de delta = |activites de M ancrees par >=1 milestone de U| /
              |activites de M total|.
              Repond a : "le modele contient-il des elements que rien dans le texte ne
              justifie ?" -- axe JAMAIS mesure avant cette experience dans ce pipeline (la
              verification allait uniquement texte -> modele, jamais modele -> texte).

  CORRECTNESS = accord de Rel(psi(M)) avec Rel(phi(T)) = |termes SATISFIED| /
                (|termes SATISFIED| + |termes VIOLATED|), UNRESOLVABLE EXCLU du denominateur
                (choix explicite : ne pas confondre "on ne sait pas" avec "faux" -- une
                precision-like, pas un rappel).
                Repond a : "ce que le modele permet de faire contredit-il une contrainte du
                texte ?" -- calcule par le solveur de reachability reecrit ci-dessous.

=== Solveur de reachability -- reimplementation independante ===

Grammaire de reference_graph (confirmee en lisant graph_node.py directement) :
  edges: [{"from": milestone_terme, "to": milestone_cible, "clause": int, "negated": bool, ...}]
  -- regrouper par "to" PUIS par "clause" reconstruit la DNF exacte de Pre(cible) : les aretes
  d'un meme (to, clause) forment un groupe ET, les differents "clause" pour un meme "to" sont
  les alternatives OU.

Semantique retenue, deliberement plus simple que tgms_solver.py (pas de score de confiance
continu ici -- le LLM rend une decision BINAIRE, matche ou non, donc pas de notion de "faible
confiance" a arbitrer) :
  - terme POSITIF : SATISFIED si un CHEMIN existe, dans le graphe de process (construit depuis
    dfg_edges + matches), du terme vers la cible ; MISSING/UNRESOLVABLE si l'un des deux
    (terme ou cible) n'est ancre par aucune activite ; VIOLATED sinon (les deux ancres, aucun
    chemin).
  - terme NEGATIF ("NOT X") : semantique d'atteignabilite seule (pas d'activity_guards ici,
    limite assumee, coherente avec le comportement de repli documente dans alignment_node.py
    pour le cas ou activity_guards est absent) -- SATISFIED si AUCUN chemin de X vers la cible
    (l'absence est respectee), VIOLATED si un chemin existe malgre la negation, UNRESOLVABLE si
    l'un des deux est missing.
  - clause (groupe ET) : UNRESOLVABLE domine VIOLATED domine SATISFIED (un seul terme
    manquant/faux invalide toute la clause) -- meme regle de dominance que tgms_solver.py,
    reimplementee independamment, pas importee (comparabilite d'interpretation, independance
    d'implementation).
  - guard (groupe OU de clauses) : SATISFIED domine UNRESOLVABLE domine VIOLATED (une seule
    clause vraie suffit)."""

import csv
import json
import os
import re
from collections import Counter, deque

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.nodes.state_matching_node import _build_contextualized_activities, _entity_states, render_candidate
from agent.nodes.state_matching_node import build_process_state_graph
from agent.models.base_llm import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "three_score_solver")

RUN_ID = "run_2"
MODEL_NAME = "gpt-5.5"
N_REPEATS = 3

CASES: list[tuple[str, str, str]] = [
    ("gpt-5.5", "OriginalDataset", "ClaimsCreation"),
    ("gpt-5.5", "OriginalDataset", "BicycleManufacturing"),
]


# --- Matching delta par LLM -- duplique de llm_matching_experiment_guardrails.py ---

MATCHING_PROMPT_TEMPLATE = """You are given a business process described by a BPMN diagram, and a list of \
"milestones" (facts of the form entity.state) that were extracted independently from the \
textual description of this same process.

Your task: for each BPMN activity below, decide which milestone(s) from the list -- if any -- \
that activity realizes.

IMPORTANT rules, read carefully:
- Not every milestone needs to be assigned to an activity. Many milestones will correspond to \
NOTHING in this diagram (e.g. generic process lifecycle facts like "a process instance was \
created/finished", or a milestone that is really a duplicate/paraphrase of another one already \
assigned elsewhere). Leaving a milestone unassigned is the CORRECT and EXPECTED answer whenever \
you are not confident -- do not force a plausible-sounding but weak connection.
- A milestone should only be assigned to an activity if it describes that SPECIFIC activity's \
own action or its direct, immediate outcome -- not merely the same general topic or a nearby \
step in the process.
- For every milestone you DO assign, you must give a short "reason" (max 15 words) that \
explicitly refers to words actually present in that activity's own label.

BPMN activities (id, label, and surrounding routing context in [] where available), in \
approximate process order:
{activities_block}

Milestones (entity.state, rendered as plain text for readability):
{milestones_block}

Respond with ONLY a JSON object, no prose, no markdown fences. Structure:
{{"activity_id_1": [{{"milestone": "entity.state1", "reason": "short reason referencing the activity's own label"}}], "activity_id_2": []}}
"""


def call_llm(prompt: str) -> str:
    llm = get_llm(model=MODEL_NAME)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def parse_llm_response(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def is_grounded(reason: str, activity_label: str) -> bool:
    if not reason or not reason.strip():
        return False
    label_words = {w.lower() for w in _WORD_RE.findall(activity_label)}
    reason_words = {w.lower() for w in _WORD_RE.findall(reason)}
    return bool(label_words & reason_words)


def build_matches_from_llm_guarded(parsed: dict, contextualized: dict, valid_activity_ids: set[str]) -> dict:
    matches = {aid: {"activity_label": contextualized[aid]["label"], "matches": []} for aid in valid_activity_ids}
    for aid, entries in parsed.items():
        if not isinstance(entries, list) or aid not in valid_activity_ids:
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "milestone" not in entry:
                continue
            milestone, reason = entry.get("milestone"), entry.get("reason", "")
            if isinstance(milestone, str) and is_grounded(reason, contextualized[aid]["label"]):
                matches[aid]["matches"].append({"match": milestone, "score": 1.0})
    return matches


def majority_vote_matches(repeats_matches: list[dict], contextualized: dict, min_votes: int = 2) -> dict:
    counts: Counter = Counter()
    for m in repeats_matches:
        for aid, data in m.items():
            for entry in data["matches"]:
                counts[(aid, entry["match"])] += 1
    consensus = {aid: {"activity_label": contextualized[aid]["label"], "matches": []} for aid in contextualized}
    for (aid, milestone), n in counts.items():
        if n >= min_votes:
            consensus[aid]["matches"].append({"match": milestone, "score": 1.0})
    return consensus


# --- Solveur de reachability independant (reimplementation de zero) ---

def build_milestone_graph(process_graph_edges: list[dict]) -> dict[str, set[str]]:
    """process_graph_edges : sortie de build_process_state_graph (liste de {"from":, "to": ...}
    en vocabulaire milestone, deja projetee depuis le DFG d'activites). Retourne une adjacence
    simple milestone -> {milestones atteignables en un pas}."""
    adj: dict[str, set[str]] = {}
    for e in process_graph_edges:
        adj.setdefault(e["from"], set()).add(e["to"])
    return adj


def path_exists(adj: dict[str, set[str]], source: str, target: str) -> bool:
    """BFS simple -- reimplementation independante, volontairement naive (pas d'optimisation,
    le corpus est petit), pour ne partager aucun code avec tgms_solver.py."""
    if source == target:
        return True
    visited = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, ()):
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def status_of_term(term: str, target: str, negated: bool, anchored: set[str], adj: dict) -> str:
    if term not in anchored or target not in anchored:
        return "UNRESOLVABLE"
    reachable = path_exists(adj, term, target)
    if negated:
        return "SATISFIED" if not reachable else "VIOLATED"
    return "SATISFIED" if reachable else "VIOLATED"


def _combine_and(statuses: list[str]) -> str:
    if "UNRESOLVABLE" in statuses:
        return "UNRESOLVABLE"
    if "VIOLATED" in statuses:
        return "VIOLATED"
    return "SATISFIED"


def _combine_or(statuses: list[str]) -> str:
    if "SATISFIED" in statuses:
        return "SATISFIED"
    if "UNRESOLVABLE" in statuses:
        return "UNRESOLVABLE"
    return "VIOLATED"


def solve_independent(reference_graph: dict, process_graph_edges: list[dict], anchored: set[str]) -> dict:
    """Retourne {"term_statuses": [...], "clause_statuses": [...], "guard_statuses": {target: status}}."""
    adj = build_milestone_graph(process_graph_edges)

    by_target: dict[str, dict[int, list[dict]]] = {}
    for e in reference_graph["edges"]:
        by_target.setdefault(e["to"], {}).setdefault(e["clause"], []).append(e)

    term_statuses, clause_statuses, guard_statuses = [], [], {}

    for target, clauses in by_target.items():
        clause_results = []
        for clause_idx, edges in clauses.items():
            term_results = []
            for e in edges:
                status = status_of_term(e["from"], e["to"], e["negated"], anchored, adj)
                term_results.append(status)
                term_statuses.append({
                    "target": target, "clause": clause_idx, "term": e["from"],
                    "negated": e["negated"], "status": status,
                })
            clause_status = _combine_and(term_results)
            clause_statuses.append({"target": target, "clause": clause_idx, "status": clause_status})
            clause_results.append(clause_status)
        guard_statuses[target] = _combine_or(clause_results)

    return {"term_statuses": term_statuses, "clause_statuses": clause_statuses, "guard_statuses": guard_statuses}


# --- Trois scores ---

def compute_three_scores(
    term_statuses: list[dict], state_space: dict, contextualized: dict, matches: dict
) -> dict:
    all_milestones = {f"{e}.{s}" for e, v in state_space.items() for s in _entity_states(v)}
    anchored_milestones = {
        m["match"] for data in matches.values() for m in data["matches"]
    }
    completeness = len(anchored_milestones & all_milestones) / len(all_milestones) if all_milestones else None

    all_activities = set(contextualized.keys())
    matched_activities = {aid for aid, data in matches.items() if data["matches"]}
    relevance = len(matched_activities) / len(all_activities) if all_activities else None

    n_sat = sum(1 for t in term_statuses if t["status"] == "SATISFIED")
    n_viol = sum(1 for t in term_statuses if t["status"] == "VIOLATED")
    n_unres = sum(1 for t in term_statuses if t["status"] == "UNRESOLVABLE")
    correctness = n_sat / (n_sat + n_viol) if (n_sat + n_viol) > 0 else None

    return {
        "completeness": round(completeness, 3) if completeness is not None else None,
        "relevance": round(relevance, 3) if relevance is not None else None,
        "correctness": round(correctness, 3) if correctness is not None else None,
        "n_terms_satisfied": n_sat, "n_terms_violated": n_viol, "n_terms_unresolvable": n_unres,
    }


# --- Orchestration ---

def load_case_state(model: str, subset: str, case_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, model, f"{subset}__{case_name}__{RUN_ID}.json")
    if not os.path.exists(path):
        raise SystemExit(f"[three_score] fichier introuvable : {path}")
    with open(path) as f:
        state = json.load(f)
    for required in ("dfg_edges", "state_space", "reference_graph"):
        if required not in state:
            raise SystemExit(f"[three_score] {path} : champ '{required}' manquant")
    return state


def build_prompt(dfg_edges: list[dict], state_space: dict) -> tuple[str, dict]:
    contextualized = _build_contextualized_activities(dfg_edges)
    valid = {k: v for k, v in contextualized.items() if v["context_text"].strip()}
    activities_block = "\n".join(
        f"- id={aid} | \"{v['label']}\" | context: {v['context_text']}" for aid, v in valid.items()
    )
    candidates = [f"{e}.{s}" for e, value in state_space.items() for s in _entity_states(value)]
    milestones_block = "\n".join(f"- {c}  (\"{render_candidate(*c.split('.', 1))}\")" for c in candidates)
    prompt = MATCHING_PROMPT_TEMPLATE.format(activities_block=activities_block, milestones_block=milestones_block)
    return prompt, valid


def run() -> list[dict]:
    rows = []
    for model, subset, case_name in CASES:
        case_label = f"{subset}/{case_name}"
        state = load_case_state(model, subset, case_name)
        dfg_edges, state_space, reference_graph = state["dfg_edges"], state["state_space"], state["reference_graph"]

        prompt, contextualized = build_prompt(dfg_edges, state_space)
        valid_set = set(contextualized.keys())

        repeats_matches = []
        for repeat_idx in range(1, N_REPEATS + 1):
            print(f"[three_score] {case_label} -- repeat {repeat_idx}/{N_REPEATS} -- appel LLM ({MODEL_NAME})...")
            raw = call_llm(prompt)
            parsed = parse_llm_response(raw)
            if parsed is None:
                print("  [three_score] JSON invalide, repeat ignore")
                continue
            repeats_matches.append(build_matches_from_llm_guarded(parsed, contextualized, valid_set))

        if len(repeats_matches) < 2:
            print(f"  [three_score] {case_label} : pas assez de repetitions exploitables, cas ignore")
            continue

        matches = majority_vote_matches(repeats_matches, contextualized, min_votes=2)
        anchored = {m["match"] for data in matches.values() for m in data["matches"]}

        process_graph_edges = build_process_state_graph(dfg_edges, matches)
        solved = solve_independent(reference_graph, process_graph_edges, anchored)
        scores = compute_three_scores(solved["term_statuses"], state_space, contextualized, matches)

        rows.append({"case": case_label, **scores})

    return rows


def print_table(title: str, table: list[dict]):
    print(f"\n=== {title} ===")
    if not table:
        print("  (aucune donnee)")
        return
    headers = list(table[0].keys())
    widths = [max(len(str(h)), max((len(str(r.get(h, ""))) for r in table), default=0)) for h in headers]
    print("  " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in table:
        print("  " + " | ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = run()
    if not rows:
        raise SystemExit("[three_score] aucun resultat produit")

    print_table("Table AA -- Correctness / Completeness / Relevance (consensus 2/3, GPT-5.5)", rows)

    with open(os.path.join(OUT_DIR, "three_scores.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultats dans {OUT_DIR}")