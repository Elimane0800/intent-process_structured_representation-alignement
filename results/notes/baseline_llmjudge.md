# Observations — Baseline LLM-as-judge (premier run)

Notes sur le premier run de `run_baseline_llm_judge.py`, analysé via `analyze_baseline_vs_tgms.py`.
1134 entrées chargées (3 modèles-juges × 3 variantes de prompt × 126 mutants applicables =
1134 lignes attendues — comptage exhaustif, cohérent). Comparaison directe contre les résultats
réels TGMS déjà mesurés (`run_perturbation_study.py`, 153 mutants, cf.
`dataset_run_v2_observations.md` section 8).

**Statut de ce document : premier run, avec des lacunes d'infrastructure connues et non
corrigées à ce stade (cf. section 4).** Les résultats des sections 2-3 restent valides pour les
configurations propres ; ceux de `mistral-nemotron` et `gpt-oss-20b/cot` sont à re-mesurer
après correctif avant toute conclusion définitive les concernant.

---

## 1. Périmètre et méthode

**Réutilisation stricte du cadre du perturbation study**, décidée avant tout code : mêmes 153
mutants, même taxonomie `required`/`forbidden`/`blind_spot`, même définition de détection au
niveau du cas (le verdict du juge bascule-t-il de `NO_VIOLATION` sur le BPMN de base à
`VIOLATION` sur le mutant ?). Juges = les trois modèles de `MODELS_TO_RUN`
(`llama-3.1-8b`, `mistral-nemotron`, `gpt-oss-20b`). Entrée du juge : texte source + **XML BPMN
brut** du mutant (choix explicite plutôt que le DFG extrait, pour ne pas retirer par
construction le type de gateway et laisser une vraie chance au juge de détecter ce que TGMS ne
peut pas voir sur `blind_spot`).

**Trois variantes de prompt testées séparément** (`zero_shot`, `one_shot`, `cot`), jamais
moyennées — même discipline que `state_space_prompt_permissive.py`/`precondition_prompt.py`
partout ailleurs dans le projet, où l'effet du prompt est documenté comme non-monotone.

**Stockage** : JSONL append-only à la racine (`baseline_llm_judge_results.jsonl`), une ligne
par `(juge, variante, mutant)`.

---

## 2. Résultat central pour la thèse du papier — confirmé, net

Sur `forbidden` (`insert_activity` + `shuffle_xml`), là où TGMS est prouvé formellement et
vérifié empiriquement à **0/102 faux positifs**, la baseline produit des faux positifs massifs
et très instables selon la configuration :

| Juge | Variante | Faux positifs |
|---|---|---|
| `llama-3.1-8b` | `cot` | **9/11 (81.8%)** |
| `gpt-oss-20b` | `one_shot` | 9/24 (37.5%) |
| `gpt-oss-20b` | `zero_shot` | 9/30 (30.0%) |
| `llama-3.1-8b` | `zero_shot` | 5/36 (13.9%) |
| `llama-3.1-8b` | `one_shot` | 0/36 (0.0%) |
| `mistral-nemotron` | `zero_shot` | 0/21 (0.0%) — n réduit, cf. section 4 |

**Lecture directe** : un jugement LLM holistique n'a aucune garantie de monotonie sur
l'enrichissement du graphe process — contrairement à TGMS (Théorème 3.7, prouvé et vérifié sans
exception). Le taux dépend fortement de la configuration (`llama-3.1-8b` va de 0% à 81.8%
selon la seule variante de prompt), ce qui est en soi un argument supplémentaire : même le
meilleur réglage possible d'un juge reste **imprévisible**, une propriété que TGMS n'a pas.

---

## 3. Résultat honnête à assumer — détection sur l'angle mort structurel

Sur `rewire_gateway` (angle mort assumé de TGMS, XOR↔AND indiscernable par construction dans la
relation `"follows"` du DFG), la baseline détecte parfois ce que TGMS ne peut structurellement
pas voir :

| Juge | Variante | Détection |
|---|---|---|
| `llama-3.1-8b` | `cot` | 4/6 (66.7%) |
| `gpt-oss-20b` | `zero_shot` | 5/15 (33.3%) |
| `gpt-oss-20b` | `one_shot` | 4/12 (33.3%) |
| `llama-3.1-8b` | `zero_shot` | 1/18 (5.6%) |

**À intégrer sans le minimiser** : c'est une vraie limite de l'abstraction état-centrique de
TGMS, contrebalancée par sa garantie formelle de monotonie — jamais un point à cacher dans le
papier. Le contraste avec la section 2 (le même juge produit aussi des faux positifs massifs
sur `forbidden`) doit être présenté ensemble : le juge voit parfois plus, mais sans garantie
d'être fiable quand il le fait.

---

## 4. Découverte non prévue — CoT dégrade `llama-3.1-8b`, pas ne l'améliore pas

Sur `forbidden`, `llama-3.1-8b` passe de 13.9% de faux positifs en `zero_shot` à **81.8% en
`cot`** — l'inverse de l'intuition standard ("le raisonnement explicite réduit les
hallucinations"). C'est exactement le résultat que l'inspection à un seul prompt aurait
manqué — la justification a posteriori de la décision de tester les trois variantes séparément.
À creuser qualitativement (inspecter `mutant_raw_response` des cas concernés) avant d'écrire une
explication dans le papier ; à ce stade, uniquement le fait est établi, pas la cause.

---

## 5. Lacunes d'infrastructure trouvées — non corrigées à ce stade

**`mistral-nemotron` : signal quasi entièrement inexploitable.** 126/126 `inconclusive` sur
`cot` et `one_shot` (100% erreurs `429`), 58/126 sur `zero_shot`. Cause : `run_baseline_llm_judge.py`
n'a **aucun retry/backoff**, alors même que `mistral-nemotron` est documenté depuis le tout
premier run dataset (`dataset_run_observations.md`) comme rate-limitant agressivement — un
oubli de report du correctif déjà appliqué à `run.py`. Les taux de la section 2-3 pour ce juge
ne doivent pas être lus comme des résultats : `n/a` ou `n` trop faible pour être significatif.

**`gpt-oss-20b/cot` : 116/126 `inconclusive` (92%), zéro erreur API.** Différent du cas
`mistral-nemotron` — pas du rate limiting, un échec de format de sortie (le marqueur
`FINAL_VERDICT:` jamais atteint). Hypothèse, cohérente avec un mode d'échec déjà documenté pour
ce modèle ailleurs dans le projet (réponses vides/tronquées, cf. gain apporté par
`reasoning_effort="low"` sur le run principal) : le budget de raisonnement s'épuise avant
d'atteindre le marqueur final sur un prompt CoT sans contrôle du raisonnement. Le script de
baseline n'applique actuellement aucun `reasoning_effort` au juge. Non corrigé à ce stade.

---

## 6. Ce qui reste ouvert

1. Corriger le retry/backoff exponentiel dans `run_baseline_llm_judge.py` (même discipline que
   `run.py`) — `_load_done_keys` doit exclure les entrées en erreur pour permettre une relance
   automatique sans toucher aux ~900 lignes déjà propres.
2. Appliquer `reasoning_effort="low"` au juge `gpt-oss-20b`, au moins pour la variante `cot`.
3. Relancer uniquement les configurations cassées (`mistral-nemotron` toutes variantes,
   `gpt-oss-20b/cot`) une fois le fix en place.
4. Inspecter qualitativement les `mutant_raw_response` des cas `llama-3.1-8b/cot` sur
   `forbidden` pour comprendre la dégradation (section 4) avant d'en écrire l'explication.
5. Une fois toutes les configurations propres : agréger un tableau final unique
   (juge × variante × classe d'opérateur) pour le papier, avec le TGMS de référence en colonne
   fixe — pas encore fait, prématuré tant que 2 des 9 configurations restent partiellement ou
   totalement invalides.