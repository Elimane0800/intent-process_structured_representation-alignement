"""Diagnostic cible : POURQUOI ClaimsCreation se degrade (FAITHFUL_WITH_RESERVATIONS ->
NOT_FAITHFUL) sous la regle 'dissimilarite' sur 4/5 modeles, alors que l'agrege global
ameliore SATISFIED_%. Dump, pour chaque modele, les matches RETENUS EN PLUS par la regle
dissimilarite (jamais retenus par la marge actuelle) sur ClaimsCreation -- le candidat en trop
est probablement parmi eux."""

import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.nodes.state_matching_node import (
    _build_contextualized_activities, _entity_states, render_candidate, cosine,
)
from agent.models.base_nvidia_embedding import embed
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lre_runs")
RUN_ID = "run_2"
MATCHING_THRESHOLD_FIXED = 0.25
CLUSTER_SIMILARITY_THRESHOLD = 0.80

MODELS = ["gpt-5.5", "gpt-oss-20b", "mistral-nemotron", "nemotron-3-super-120b", "llama-3.1-8b"]
CASE = "OriginalDataset/ClaimsCreation"


def matches_current_rule(activity_ids, activity_vecs, candidates, candidate_vecs, threshold):
    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        best = max(scores) if scores else -1
        matches[aid] = {candidates[i] for i, s in enumerate(scores) if s >= threshold and s >= best - 0.05}
    return matches


def matches_dissimilarity_rule(activity_ids, activity_vecs, candidates, candidate_vecs, threshold, cluster_threshold):
    matches = {}
    for aid, vec in zip(activity_ids, activity_vecs):
        scores = [cosine(vec, cvec) for cvec in candidate_vecs]
        above = sorted([(i, s) for i, s in enumerate(scores) if s >= threshold], key=lambda x: -x[1])
        retained = []
        for idx, score in above:
            if not retained or not any(cosine(candidate_vecs[idx], candidate_vecs[r]) >= cluster_threshold for r in retained):
                retained.append(idx)
        matches[aid] = {candidates[i] for i in retained}
    return matches


for model in MODELS:
    path = os.path.join(RESULTS_DIR, model, f"{CASE.replace('/', '__')}__{RUN_ID}.json")
    if not os.path.exists(path):
        print(f"[{model}] fichier introuvable ({path}), ignore")
        continue
    with open(path) as f:
        state = json.load(f)
    if any(k not in state for k in ("dfg_edges", "state_space")):
        print(f"[{model}] champs manquants, ignore")
        continue

    dfg_edges, state_space = state["dfg_edges"], state["state_space"]
    contextualized = _build_contextualized_activities(dfg_edges)
    valid = {k: v for k, v in contextualized.items() if v["context_text"].strip()}
    candidates = [f"{e}.{s}" for e, v in state_space.items() for s in _entity_states(v)]
    candidate_texts = [render_candidate(*c.split(".", 1)) for c in candidates]
    activity_ids = list(valid.keys())
    activity_texts = [" ".join(valid[a]["context_text"].split()) for a in activity_ids]

    if not activity_texts or not candidate_texts:
        print(f"[{model}] pas assez de donnees, ignore")
        continue

    activity_vecs = np.array(embed(activity_texts, input_type="query"))
    candidate_vecs = np.array(embed(candidate_texts, input_type="query"))

    m_current = matches_current_rule(activity_ids, activity_vecs, candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED)
    m_alt = matches_dissimilarity_rule(
        activity_ids, activity_vecs, candidates, candidate_vecs, MATCHING_THRESHOLD_FIXED, CLUSTER_SIMILARITY_THRESHOLD
    )

    print(f"\n########## {model} / {CASE} -- nouveaux matches (dissimilarite only) ##########")
    any_new = False
    for aid in activity_ids:
        new = m_alt.get(aid, set()) - m_current.get(aid, set())
        if new:
            any_new = True
            label = valid[aid]["label"]
            print(f"  \"{label}\" gagne : {sorted(new)}  (avait deja : {sorted(m_current.get(aid, set()))})")
    if not any_new:
        print("  (aucun nouveau match -- le changement de verdict vient d'ailleurs, pas du matching)")