"""Solveur COMPLETEMENT AGNOSTIQUE du pipeline actuel : aucun import de state_matching_node.py,
bpmn_to_spo_node.py, alignment_node.py, tgms_solver.py, report_node.py. Seules deux donnees deja
produites separement sont reutilisees comme INTRANTS (pas comme methode) : state_space (U, deja
valide par le Protocole 3, hors du champ de ce test) et le texte source T -- necessaire pour le
role de pertinence, cf. plus bas.

=== Definitions, cherchees dans la litterature plutot que recyclees de nos propres statuts ===

CORRECTNESS (RECOVER -- requirements engineering ; extraction bio-medicale) : une correspondance
affirmee (milestone, activite) est correcte si elle est IMPLIQUEE par le contenu reel de
l'activite, VERIFIEE INDEPENDAMMENT de qui l'a affirmee -- jamais une simple absence de
contradiction dans un graphe. D'ou l'architecture a deux roles : un EXTRACTEUR propose des
correspondances, un VERIFICATEUR AVEUGLE (ne voit jamais le raisonnement de l'extracteur, juge
chaque paire seule) les confirme ou les rejette. Correctness = paires confirmees / paires
proposees -- une precision de l'extraction, pas une atteignabilite.

COMPLETENESS (OpenIE -- "extract ALL information [available]") : part des milestones de U ayant
au moins une correspondance CONFIRMEE (post-verification, jamais les propositions brutes de
l'extracteur -- sinon une extraction bruyante gonflerait artificiellement ce score).

RELEVANCE (litterature LLM-judge/5W1H -- distincte de correctness : le contenu est-il pertinent
au sujet, pas seulement correct) : dans notre cas, une activite BPMN peut etre legitime SANS
jamais avoir de milestone dedie dans U (le texte peut l'impliquer sans la nommer explicitement --
U lui-meme n'est pas necessairement exhaustif). D'ou un troisieme role, un JUGE DE PERTINENCE,
qui retourne au TEXTE SOURCE BRUT (jamais aux milestones deja extraits) pour chaque activite
SANS correspondance confirmee, et demande : le texte decrit-il ou implique-t-il clairement cette
etape, meme sans la nommer ? Relevance = (activites avec >=1 correspondance confirmee + activites
non-matchees mais jugees pertinentes au texte) / total des activites.

=== Cout ===
Contrairement aux experiences precedentes (1 appel par repetition), celle-ci fait PLUSIEURS
appels par cas : 1 (extraction) + 1 par paire proposee (verification) + 1 par activite non
matchee (pertinence). Sur un petit cas (~15-20 activites/milestones), ça peut faire 15-25 appels
LLM par cas -- borne ici a 2 cas, 1 seule passe chacun (pas de repetitions), pour un premier test
avant de decider s'il faut passer a l'echelle."""

import csv
import json
import os
import re
import sys
import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.models.base_llm import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "agnostic_three_score_solver")

RUN_ID = "run_2"
MODEL_NAME = "gpt-5.5"
N_REPEATS = 3

TARGET_MODEL_DIR = "gpt-5.5"  # dossier sous results/lre_runs/ a scanner -- les cas y sont
                                # deja tous produits avec ce meme modele pour state_space/Pre,
                                # coherent avec MODEL_NAME utilise ici pour l'extraction/
                                # verification/pertinence


def parse_filename(filename: str):
    if not filename.endswith(".json"):
        return None
    stem = filename[:-5]
    if "__" not in stem:
        return None
    subset, rest = stem.split("__", 1)
    if "__" not in rest:
        return None
    case_name, run_id = rest.rsplit("__", 1)
    return subset, case_name, run_id


def discover_cases(model_dir: str, run_id: str) -> list[tuple[str, str, str]]:
    """Retourne [(model, subset, case_name), ...] -- auto-detecte depuis le disque, meme
    principe que les scripts precedents de cette session, plutot qu'une liste figee a la main."""
    dir_path = os.path.join(RESULTS_DIR, model_dir)
    if not os.path.isdir(dir_path):
        raise SystemExit(f"[agnostic] dossier introuvable : {dir_path}")
    cases = []
    for fname in sorted(os.listdir(dir_path)):
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        subset, case_name, found_run_id = parsed
        if found_run_id == run_id:
            cases.append((model_dir, subset, case_name))
    return cases


CASES: list[tuple[str, str, str]] = discover_cases(TARGET_MODEL_DIR, RUN_ID)


def call_llm(prompt: str) -> str:
    llm = get_llm(model=MODEL_NAME)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def parse_json_response(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# --- Parsing BPMN minimal, ecrit ici -- volontairement independant de bpmn_to_spo_node.py
# (qui encode des choix de conception specifiques a l'approche embedding : passthrough des
# gateways, format de gateway_context). Ici on ne veut que : la liste des activites (id, label)
# et l'ordre approximatif du process (via les sequenceFlow bruts), rien de plus. ---

def _stub_glpk():
    try:
        from cvxopt import glpk  # noqa: F401
    except ImportError:
        import cvxopt
        stub = types.ModuleType("cvxopt.glpk")
        sys.modules["cvxopt.glpk"] = stub
        cvxopt.glpk = stub


def parse_bpmn_minimal(bpmn_path: str) -> list[dict]:
    """Retourne [{"id":, "label":, "order_hint": int}] pour les activites (Task/SubProcess),
    dans un ordre approximatif de process (index de decouverte par parcours des sequenceFlow
    depuis les start events). Aucune notion de gateway_context, de passthrough ou de DFG --
    volontairement minimal, seulement ce qui est necessaire pour donner au LLM une vue
    d'ensemble du process, sans imposer le format de representation de l'approche embedding."""
    _stub_glpk()
    import xml.etree.ElementTree as ET
    import tempfile
    import pm4py

    tree = ET.parse(bpmn_path)
    root = tree.getroot()
    removed = False
    for parent in root.iter():
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1].lower()
            if tag in ("association", "textannotation"):
                parent.remove(child)
                removed = True
    if removed:
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            tree.write(f, xml_declaration=True, encoding="utf-8")
            tmp_path = f.name
        bpmn = pm4py.read_bpmn(tmp_path)
        os.unlink(tmp_path)
    else:
        bpmn = pm4py.read_bpmn(bpmn_path)

    ACTIVITY_TYPES = (pm4py.objects.bpmn.obj.BPMN.Task, pm4py.objects.bpmn.obj.BPMN.SubProcess)
    activities = {n.get_id(): n.get_name() for n in bpmn.get_nodes() if isinstance(n, ACTIVITY_TYPES)}

    outgoing: dict[str, list] = {}
    for flow in bpmn.get_flows():
        outgoing.setdefault(flow.get_source().get_id(), []).append(flow.get_target().get_id())

    order_hint: dict[str, int] = {}
    counter = 0
    starts = [n.get_id() for n in bpmn.get_nodes() if isinstance(n, pm4py.objects.bpmn.obj.BPMN.StartEvent)]
    visited = set()
    stack = list(starts)
    while stack:
        node_id = stack.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in activities and node_id not in order_hint:
            order_hint[node_id] = counter
            counter += 1
        stack.extend(outgoing.get(node_id, []))

    return [
        {"id": aid, "label": label or "", "order_hint": order_hint.get(aid, 9999)}
        for aid, label in activities.items() if label and label.strip()
    ]


def _entity_states(value) -> list[str]:
    if isinstance(value, dict):
        return list(value.get("states", []))
    return list(value)


# --- Role 1 : EXTRACTEUR ---

EXTRACTOR_PROMPT = """You are given a list of BPMN process activities (in approximate process order) and a \
list of milestones (facts of the form entity.state) extracted independently from the textual \
description of the same process.

For each milestone, propose AT MOST ONE activity that most plausibly realizes it -- or none if \
you are not confident. Do not force a match.

Activities (id, label, approximate order):
{activities_block}

Milestones:
{milestones_block}

Respond with ONLY a JSON object, no prose: {{"milestone1": "activity_id_or_null", "milestone2": "activity_id_or_null", ...}}
"""


def run_extractor(activities: list[dict], milestones: list[str]) -> dict[str, str | None]:
    activities_block = "\n".join(f"- id={a['id']} | \"{a['label']}\" | order={a['order_hint']}" for a in activities)
    milestones_block = "\n".join(f"- {m}" for m in milestones)
    prompt = EXTRACTOR_PROMPT.format(activities_block=activities_block, milestones_block=milestones_block)
    raw = call_llm(prompt)
    parsed = parse_json_response(raw)
    if parsed is None:
        return {}
    return {m: (v if isinstance(v, str) and v.lower() != "null" else None) for m, v in parsed.items() if m in milestones}


# --- Role 2 : VERIFICATEUR AVEUGLE (batche -- un seul appel pour TOUTES les paires du cas,
# jamais un appel par paire) ---
#
# Compromis assume : le verificateur ne voit toujours PAS le raisonnement de l'extracteur (le
# principal facteur de biais d'auto-confirmation qu'on voulait eviter), mais voit desormais
# toutes les paires ENSEMBLE dans le meme appel plutot qu'une seule a la fois -- risque mineur
# de contamination entre jugements voisins, accepte pour reduire le cout de 1-appel-par-paire a
# 1-appel-par-cas. A degrader vers la version non-batchee si ce risque s'avere significatif en
# pratique (comparer un sous-echantillon en mode isole vs batche si besoin de trancher).

VERIFIER_PROMPT_BATCH = """Below is a list of claimed correspondences between BPMN process activities and \
milestones (facts about the process). For EACH claim, judge independently, based ONLY on the \
activity's own label given below (not on general process knowledge) -- does this SPECIFIC \
activity genuinely and directly realize this specific milestone ?

Claims:
{claims_block}

Respond with ONLY a JSON array, no prose, one entry per claim in the same order:
[{{"milestone": "...", "activity_id": "...", "confirmed": true or false, "justification": "one short sentence"}}, ...]
"""


def run_verifier_batch(pairs: list[tuple[str, str, str]]) -> dict[str, bool]:
    """pairs : liste de (milestone, activity_id, activity_label). Retourne {milestone: bool}."""
    if not pairs:
        return {}
    claims_block = "\n".join(
        f'{i+1}. milestone="{m}" | activity_id="{aid}" | activity_label="{label}"'
        for i, (m, aid, label) in enumerate(pairs)
    )
    prompt = VERIFIER_PROMPT_BATCH.format(claims_block=claims_block)
    raw = call_llm(prompt)
    parsed = parse_json_response(raw)
    if not isinstance(parsed, list):
        return {m: False for m, _, _ in pairs}  # reponse inexploitable -- rien confirme par defaut
    result = {}
    for entry in parsed:
        if isinstance(entry, dict) and "milestone" in entry:
            result[entry["milestone"]] = bool(entry.get("confirmed", False))
    # toute paire absente de la reponse (parsing partiel) reste non confirmee par defaut
    return {m: result.get(m, False) for m, _, _ in pairs}


# --- Role 3 : JUGE DE PERTINENCE (batche -- un seul appel pour TOUTES les activites non
# matchees du cas) ---

RELEVANCE_PROMPT_BATCH = """Here is a textual description of a business process:
\"\"\"
{text}
\"\"\"

Below is a list of steps from a BPMN diagram of (supposedly) the same process, none of which \
were explicitly matched to a specific fact extracted from the text. For EACH step, judge \
independently -- does the text above describe or clearly imply this step, even if it does not \
name it explicitly (e.g. because the text describes it at a coarser or finer level of detail) ? \
Answer strictly based on the text given, not on general assumptions.

Steps:
{steps_block}

Respond with ONLY a JSON array, no prose, one entry per step in the same order:
[{{"activity_id": "...", "relevant": true or false, "justification": "one short sentence"}}, ...]
"""


def run_relevance_judge_batch(text: str, activities: list[dict]) -> set[str]:
    """Retourne l'ensemble des activity_id juges pertinents par rapport au texte."""
    if not activities:
        return set()
    steps_block = "\n".join(f'{i+1}. activity_id="{a["id"]}" | label="{a["label"]}"' for i, a in enumerate(activities))
    prompt = RELEVANCE_PROMPT_BATCH.format(text=text, steps_block=steps_block)
    raw = call_llm(prompt)
    parsed = parse_json_response(raw)
    if not isinstance(parsed, list):
        return set()
    return {
        entry["activity_id"] for entry in parsed
        if isinstance(entry, dict) and entry.get("relevant") is True and "activity_id" in entry
    }


# --- Orchestration ---

def load_case_state(model: str, subset: str, case_name: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, model, f"{subset}__{case_name}__{RUN_ID}.json")
    if not os.path.exists(path):
        print(f"  [agnostic] {subset}/{case_name} : fichier introuvable ({path}), cas ignore")
        return None
    with open(path) as f:
        state = json.load(f)
    if "fatal_error" in state:
        print(f"  [agnostic] {subset}/{case_name} : fatal_error en amont, cas ignore")
        return None
    missing = [k for k in ("state_space", "bpmn_path", "text") if k not in state]
    if missing:
        print(f"  [agnostic] {subset}/{case_name} : champ(s) manquant(s) {missing}, cas ignore")
        return None
    if not os.path.exists(state["bpmn_path"]):
        print(f"  [agnostic] {subset}/{case_name} : bpmn_path introuvable ({state['bpmn_path']}), cas ignore")
        return None
    return state


def run_case_once(model: str, subset: str, case_name: str, state: dict, repeat_idx: int) -> dict:
    case_label = f"{subset}/{case_name}"
    text, bpmn_path, state_space = state["text"], state["bpmn_path"], state["state_space"]

    activities = parse_bpmn_minimal(bpmn_path)
    milestones = [f"{e}.{s}" for e, v in state_space.items() for s in _entity_states(v)]

    print(f"[agnostic] {case_label} repeat {repeat_idx} -- extraction ({len(activities)} activites, {len(milestones)} milestones)...")
    proposed = run_extractor(activities, milestones)
    proposed_pairs = {m: aid for m, aid in proposed.items() if aid}

    print(f"[agnostic] {case_label} repeat {repeat_idx} -- verification (1 appel groupe) de {len(proposed_pairs)} paire(s) proposee(s)...")
    activity_by_id = {a["id"]: a["label"] for a in activities}
    pairs_to_verify = [
        (m, aid, activity_by_id[aid]) for m, aid in proposed_pairs.items() if aid in activity_by_id
    ]
    verified = run_verifier_batch(pairs_to_verify)
    confirmed_pairs = {m: (aid if verified.get(m, False) else None) for m, aid in proposed_pairs.items()}

    confirmed_milestones = {m for m, aid in confirmed_pairs.items() if aid}
    confirmed_activities = {aid for aid in confirmed_pairs.values() if aid}

    unmatched_activities = [a for a in activities if a["id"] not in confirmed_activities]
    print(f"[agnostic] {case_label} repeat {repeat_idx} -- pertinence (1 appel groupe) pour {len(unmatched_activities)} activite(s) non matchee(s)...")
    relevant_unmatched = run_relevance_judge_batch(text, unmatched_activities)

    n_proposed = len(proposed_pairs)
    n_confirmed = len(confirmed_milestones)
    correctness = n_confirmed / n_proposed if n_proposed else None
    completeness = n_confirmed / len(milestones) if milestones else None
    n_relevant_activities = len(confirmed_activities) + len(relevant_unmatched)
    relevance = n_relevant_activities / len(activities) if activities else None

    return {
        "case": case_label, "repeat": repeat_idx,
        "n_activities": len(activities), "n_milestones": len(milestones),
        "n_pairs_proposed": n_proposed, "n_pairs_confirmed": n_confirmed,
        "n_activities_relevant_unmatched": len(relevant_unmatched),
        "correctness": round(correctness, 3) if correctness is not None else None,
        "completeness": round(completeness, 3) if completeness is not None else None,
        "relevance": round(relevance, 3) if relevance is not None else None,
        "confirmed_milestones": sorted(confirmed_milestones),  # PAS juste le compte -- necessaire
                                                                  # pour distinguer convergence
                                                                  # (memes milestones a chaque
                                                                  # fois) de coincidence
                                                                  # numerique (meme COMPTE, milestones
                                                                  # differents), cf. discussion.
    }


def run_case_repeated(model: str, subset: str, case_name: str) -> tuple[list[dict], dict, dict] | None:
    """Rejoue run_case_once() N_REPEATS fois, independamment (aucun etat partage entre
    repetitions -- chaque appel LLM est independant, pas de vote majoritaire ici : on veut
    justement voir si CHAQUE repetition, seule, donne un score coherent). Retourne
    (repeat_rows, summary, overlap_row), ou None si le cas doit etre ignore (cf.
    load_case_state)."""
    case_label = f"{subset}/{case_name}"
    state = load_case_state(model, subset, case_name)
    if state is None:
        return None

    repeat_rows = [run_case_once(model, subset, case_name, state, i) for i in range(1, N_REPEATS + 1)]

    def agg(key):
        vals = [r[key] for r in repeat_rows if r[key] is not None]
        if not vals:
            return {"mean": None, "min": None, "max": None}
        return {"mean": round(sum(vals) / len(vals), 3), "min": round(min(vals), 3), "max": round(max(vals), 3)}

    summary = {"case": case_label}
    for key in ("correctness", "completeness", "relevance"):
        stats = agg(key)
        summary[f"{key}_mean"] = stats["mean"]
        summary[f"{key}_min"] = stats["min"]
        summary[f"{key}_max"] = stats["max"]
        summary[f"{key}_range"] = (
            round(stats["max"] - stats["min"], 3) if stats["min"] is not None else None
        )

    # Recouvrement entre repetitions -- repond DIRECTEMENT a "convergence vraie ou coincidence
    # numerique" : le NOYAU STABLE (intersection des 3 ensembles de milestones confirmes) vs
    # l'UNION (tout ce qui a ete confirme au moins une fois) -- un ratio noyau/union proche de 1
    # = memes milestones a chaque fois (convergence) ; proche de 0 = memes COMPTES mais
    # milestones differents (coincidence numerique, la stabilite de Table AC serait trompeuse).
    sets = [set(r["confirmed_milestones"]) for r in repeat_rows]
    core = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    overlap_row = {
        "case": case_label,
        "n_core_stable": len(core), "n_union_ever_confirmed": len(union),
        "core_over_union_ratio": round(len(core) / len(union), 3) if union else None,
        "core_milestones": sorted(core),
        "confirmed_only_sometimes": sorted(union - core),
    }

    return repeat_rows, summary, overlap_row


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

    print(f"[agnostic] {len(CASES)} cas decouverts sous results/lre_runs/{TARGET_MODEL_DIR}/ "
          f"pour {RUN_ID} -- {N_REPEATS} repetitions chacun, 3 appels LLM par repetition "
          f"(estimation : ~{len(CASES) * N_REPEATS * 3} appels au total)")

    all_repeat_rows, all_summaries, all_overlaps = [], [], []
    n_skipped = 0
    for model, subset, case_name in CASES:
        result = run_case_repeated(model, subset, case_name)
        if result is None:
            n_skipped += 1
            continue
        repeat_rows, summary, overlap_row = result
        all_repeat_rows.extend(repeat_rows)
        all_summaries.append(summary)
        all_overlaps.append(overlap_row)

    print(f"\n[agnostic] {len(all_summaries)}/{len(CASES)} cas traites avec succes "
          f"({n_skipped} ignore(s))")

    if not all_repeat_rows:
        raise SystemExit("[agnostic] aucun cas exploitable -- rien a rapporter")

    print_table("Table AB -- Detail par repetition (extracteur + verificateur aveugle + juge de pertinence)",
                [{k: v for k, v in r.items() if k != "confirmed_milestones"} for r in all_repeat_rows])
    print_table("Table AC -- Stabilite des scores sur 3 repetitions independantes (mean/min/max/range)", all_summaries)
    print_table("Table AD -- Recouvrement reel entre repetitions (convergence vs coincidence numerique)",
                [{k: v for k, v in r.items() if k not in ("core_milestones", "confirmed_only_sometimes")} for r in all_overlaps])

    for row in all_overlaps:
        print(f"\n[{row['case']}] noyau stable (confirme les 3 fois) : {row['core_milestones']}")
        print(f"[{row['case']}] confirme SEULEMENT parfois (1 ou 2 fois sur 3) : {row['confirmed_only_sometimes']}")

    # Table AE -- agrege sur TOUS les cas (moyenne des moyennes par cas -- chaque cas pese
    # pareil, independamment de son nombre de milestones/activites, coherent avec la lecture
    # "performance typique sur un cas" plutot que "performance ponderee par la taille du cas").
    def overall_mean(key):
        vals = [s[key] for s in all_summaries if s[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    table_ae = [{
        "n_cases": len(all_summaries),
        "correctness_mean_of_means": overall_mean("correctness_mean"),
        "completeness_mean_of_means": overall_mean("completeness_mean"),
        "relevance_mean_of_means": overall_mean("relevance_mean"),
        "avg_core_over_union_ratio": round(
            sum(r["core_over_union_ratio"] for r in all_overlaps if r["core_over_union_ratio"] is not None)
            / len([r for r in all_overlaps if r["core_over_union_ratio"] is not None]), 3
        ) if all_overlaps else None,
    }]
    print_table("Table AE -- Agrege sur tous les cas (moyenne des moyennes par cas)", table_ae)

    with open(os.path.join(OUT_DIR, "agnostic_three_scores_repeats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_repeat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_repeat_rows)
    with open(os.path.join(OUT_DIR, "agnostic_three_scores_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)
    with open(os.path.join(OUT_DIR, "agnostic_three_scores_overlap.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_overlaps[0].keys()))
        writer.writeheader()
        writer.writerows(all_overlaps)
    with open(os.path.join(OUT_DIR, "agnostic_three_scores_overall.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_ae[0].keys()))
        writer.writeheader()
        writer.writerows(table_ae)
    print(f"\nResultats dans {OUT_DIR}")