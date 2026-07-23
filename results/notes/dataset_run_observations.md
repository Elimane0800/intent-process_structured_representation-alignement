# Observations — Run à trois modèles sur le dataset complet (215 cas)

Notes factuelles sur le premier run du pipeline complet à l'échelle du corpus entier
(`text_and_bpmn/bpmn/`, 24 descriptions, 223 modèles BPMN, 8 exclus pour connectivité
irrécupérable — cf. `bpmn_silent_loss_audit.json`), sur trois modèles LLM :
`meta/llama-3.1-8b-instruct`, `mistralai/mistral-nemotron`, `openai/gpt-oss-20b` (clé courte
`llama-3.1-8b`, `mistral-nemotron`, `gpt-oss-20b`). 215 paires (modèle BPMN × texte) par modèle
LLM, produites par `run_dataset.py`, analysées par `analyze_dataset_runs.py`.

Ce document présente les résultats bruts, organisés par thème, sans trancher les questions
encore ouvertes qu'ils soulèvent — chaque section se termine par ce qui reste à vérifier plutôt
que par une conclusion.

---

## 1. Volume de données exploitables par modèle

| Modèle | Cas propres (avec rapport, sans erreur interne) | Total | % |
|---|---|---|---|
| `llama-3.1-8b` | 141 | 215 | 65.6% |
| `mistral-nemotron` | 41 | 215 | 19.1% |
| `gpt-oss-20b` | 91 | 215 | 42.3% |

Toutes les analyses statistiques qui suivent (corrélations, taux agrégés) portent uniquement sur
ces sous-ensembles propres — `alignment_counts()` filtre implicitement sur la présence d'un
`report`, donc les analyses "brutes" et l'analyse explicitement "sur sous-ensemble propre"
(section 19 du script) produisent des chiffres identiques. Point méthodologique confirmé, pas
supposé : ce n'est pas un filtrage supplémentaire, c'est le même filtrage appliqué partout.

---

## 2. Erreurs par node — répartition et nature

| Modèle | precondition | graph_construction | alignment | report | bpmn_to_spo | state_matching | state_space |
|---|---|---|---|---|---|---|---|
| `llama-3.1-8b` | 52 | 52 | 70 | 74 | 24 | 42 | 20 |
| `mistral-nemotron` | 168 | 168 | 174 | 174 | 24 | 169 | 162 |
| `gpt-oss-20b` | 113 | 113 | 124 | 124 | 24 | 101 | 85 |

(Comptage de cas affectés, pas d'occurrences — un cas peut apparaître dans plusieurs colonnes
par effet de cascade : un échec `state_space` entraîne mécaniquement un échec dans tous les
nodes en aval qui en dépendent.)

**Trois causes distinctes identifiées dans les messages d'erreur bruts** :

1. **Rate limiting (`429 Too Many Requests`)** — dominant pour `mistral-nemotron` sur
   `state_space`/`precondition` (échantillons examinés à la main : 5/5 messages `state_space`
   pour ce modèle sont des 429). `run_dataset.py` retente 2 fois avec un délai fixe de 5
   secondes, insuffisant pour laisser retomber la limite dans ce cas.
2. **`bpmn_to_spo` : `'str' object has no attribute 'add_in_arc'`/`'add_out_arc'`** — exactement
   24 cas pour les **trois** modèles, mêmes fichiers BPMN dans les trois cas (ex. `M_j03/4`,
   `M_j02/2`, `E_j02/5`, `E_j03/4`, `G_j01/1`). Indépendant du LLM (ce node n'en appelle
   aucun) — un problème de parsing PM4Py propre à certains fichiers du corpus, distinct des 8
   déjà exclus pour connectivité cassée. Non encore diagnostiqué en détail.
3. **Réponses vides (`raw response=''`)** — observé uniquement sur `gpt-oss-20b`, sur
   `state_space` et `precondition`. Cause non identifiée (troncature, filtrage de contenu,
   paramètre de génération) — non encore investiguée.

Un quatrième mode d'échec (JSON mal formé/tronqué avec contenu réel, distinct d'une réponse
vide) apparaît sur `llama-3.1-8b` et occasionnellement `gpt-oss-20b`.

**Ouvert** : la part relative de chacune des trois causes dans le total des erreurs par modèle
n'a pas été quantifiée précisément (seuls des échantillons de 5 messages par node ont été
examinés à la main) — un comptage systématique par pattern de message reste à faire avant de
prioriser un correctif.

---

## 3. Corrélation entre couverture (SATISFIED) et note experte Zenodo

| Modèle | n | Spearman ρ | p |
|---|---|---|---|
| `llama-3.1-8b` | 111 | 0.012 | 0.898 |
| `mistral-nemotron` | 41 | 0.079 | 0.625 |
| `gpt-oss-20b` | 88 | 0.000 | 0.998 |

Aucune corrélation détectable, sur les trois modèles, sur l'échantillon propre disponible.

**Ce que la note Zenodo mesure, vérifié dans la documentation du dataset** (Bernard et al.,
*"Conversational Process Modeling"*, arXiv:2304.11065 — dataset source des mêmes descriptions/
modèles) : *"Each model was evaluated by a modelling expert using a quality value from 0 to 5,
to reflect how well the textual description has been transformed into a BPMN model, i.e., all
tasks and decisions from the textual description are in the BPMN, tasks which can run in
parallel have been correctly identified, and the BPMN model is well-formed."* Les modèles
eux-mêmes ont été produits par des *"modelling novices"*, pas des experts.

Cette définition recoupe en grande partie ce que le pipeline mesure (couverture des tâches/
décisions, identification correcte du parallélisme) — la note ne semble donc pas mesurer un
axe de qualité totalement disjoint de ce qu'on vérifie. Mais elle inclut aussi *"well-formed"*
(bonne formation BPMN au sens structurel), un axe que le pipeline actuel ne vérifie pas du tout
(la validité structurelle du BPMN lui-même n'est jamais évaluée, seulement sa fidélité au
texte une fois parsé).

**Ouvert** : la part de variance de la note Zenodo attribuable à la bonne formation BPMN
(non mesurée par le pipeline) plutôt qu'à la fidélité texte↔modèle (ce que le pipeline mesure)
n'est pas quantifiable avec les données actuelles. Un filtrage sur les modèles structurellement
propres (0 boucle perdue, 0 flux cassé) avant de recalculer la corrélation n'a pas été fait.

### 3bis. Matrice de confusion (méthode bon/mauvais vs Zenodo bon/mauvais)

Le coefficient de corrélation seul ne dit pas si la méthode discrimine réellement les cas, ou si
un taux d'accord apparent tient uniquement à un déséquilibre de classes des deux côtés. Seuils
utilisés : Zenodo "bon" si note ≥ 3.5, "mauvais" si note ≤ 1.5 (cas intermédiaires exclus) ;
méthode "bon" si taux SATISFIED ≥ 0.5, "mauvais" sinon.

**Répartition des classes, chaque modèle** :

| Modèle | n classé | Zenodo bon | Zenodo mauvais | Méthode bon | Méthode mauvais |
|---|---|---|---|---|---|
| `llama-3.1-8b` | 104 | 98 (94.2%) | 6 (5.8%) | 80 (76.9%) | 24 (23.1%) |
| `mistral-nemotron` | 37 | 37 (100%) | 0 (0%) | 23 (62.2%) | 14 (37.8%) |
| `gpt-oss-20b` | 80 | 76 (95.0%) | 4 (5.0%) | 45 (56.2%) | 35 (43.8%) |

**Matrices de confusion** :

`llama-3.1-8b` (n=104) :

| | Zenodo bon | Zenodo mauvais |
|---|---|---|
| **Méthode bon** | 75 | 5 |
| **Méthode mauvais** | 23 | 1 |

`mistral-nemotron` (n=37) — aucun cas Zenodo "mauvais" dans l'échantillon propre :

| | Zenodo bon | Zenodo mauvais |
|---|---|---|
| **Méthode bon** | 23 | 0 |
| **Méthode mauvais** | 14 | 0 |

`gpt-oss-20b` (n=80) :

| | Zenodo bon | Zenodo mauvais |
|---|---|---|
| **Méthode bon** | 45 | 0 |
| **Méthode mauvais** | 31 | 4 |

**Rappel sur la classe "mauvais"** (vrais mauvais détectés / total vrais mauvais) :

| Modèle | Vrais mauvais détectés | Total vrais mauvais | Rappel |
|---|---|---|---|
| `llama-3.1-8b` | 1 | 6 | 16.7% |
| `mistral-nemotron` | — | 0 | non calculable |
| `gpt-oss-20b` | 4 | 4 | 100% |

**Constat factuel principal de cette section** : dans les trois échantillons, la classe Zenodo
"mauvais" (note ≤ 1.5) est très minoritaire (0 à 6 cas sur 37 à 104) — bien en dessous de ce qui
permettrait de conclure statistiquement sur la capacité de la méthode à détecter les mauvais
modèles, dans un sens comme dans l'autre. Les deux rappels calculables (16.7% et 100%) vont dans
des directions opposées et reposent chacun sur un nombre de cas trop faible (6 et 4) pour
trancher. Ce déséquilibre de classe, plutôt qu'un défaut de la méthode elle-même, est la
limite la plus probable pour expliquer l'absence de signal de la section 3 — sans que ce soit
non plus démontré : `ρ≈0` a été calculé sur l'ensemble continu des notes (n=41 à 111), pas
seulement sur les extrêmes, et reste la mesure la plus fiable disponible à ce stade.

**Ouvert** : la méthode ne dit pas "toujours bon" (23% à 44% des cas classés "mauvais" par la
méthode), ce qui écarte l'hypothèse d'un biais de prédiction constante. Mais on ne sait pas
encore si le fait de classer 23 à 31 cas comme "mauvais" (méthode) alors que Zenodo les note
"bon" reflète une méthode trop sévère, une définition de seuil (0.5) mal calibrée, ou une vraie
divergence de jugement entre les deux — non tranché.

---

## 4. Fréquence de VIOLATED (mécanisme calibré en amont, jamais observé à cette échelle)

| Modèle | VIOLATED | Total vérifications | % |
|---|---|---|---|
| `llama-3.1-8b` | 15 | 695 | 2.16% |
| `mistral-nemotron` | 7 | 289 | 2.42% |
| `gpt-oss-20b` | 19 | 593 | 3.20% |

Taux comparable et non nul sur les trois modèles — première observation de `VIOLATED` en
conditions réelles à grande échelle (jusqu'ici seulement calibré par test de contrôle construit,
cf. `pipeline_observations.md` section 5).

**Ouvert** : aucune inspection qualitative des cas `VIOLATED` réels n'a été faite à ce stade
(seule la liste des cas concernés a été extraite, pas leur contenu détaillé) — impossible de
dire à ce stade si ces désordres détectés sont des vrais désordres du BPMN ou des artefacts
(ex. mismatch de granularité au matching).

---

## 5. Répartition SATISFIED / VIOLATED / UNRESOLVABLE, par modèle

| Modèle | SATISFIED | VIOLATED | UNRESOLVABLE |
|---|---|---|---|
| `llama-3.1-8b` | 71.4% | 2.2% | 26.5% |
| `mistral-nemotron` | 47.1% | 2.4% | 50.5% |
| `gpt-oss-20b` | 49.2% | 3.2% | 47.6% |

**Ces trois répartitions ne sont pas directement comparables entre elles** : elles portent sur
des sous-ensembles de cas propres de tailles et de compositions différentes (141, 41, 91 cas
respectivement, cf. section 1), pas sur les mêmes 215 cas. Une différence entre modèles ici peut
refléter une différence de composition d'échantillon plutôt qu'une différence réelle de
comportement du modèle.

---

## 6. Taux de clés de précondition malformées

| Modèle | Malformées | Total entrées | % |
|---|---|---|---|
| `llama-3.1-8b` | 700 | 2049 | 34.2% |
| `mistral-nemotron` | 0 | 796 | 0.0% |
| `gpt-oss-20b` | 0 | 1478 | 0.0% |

Confirme à l'échelle un pattern déjà observé sur 2 cas isolés (`pipeline_observations.md`,
section 3) : `llama-3.1-8b` est seul concerné par ce mode d'échec, 0% pour les deux autres
modèles sur 215 cas chacun.

### Distribution des raisons de `DOWNGRADED`

| Raison | `llama-3.1-8b` | `mistral-nemotron` | `gpt-oss-20b` |
|---|---|---|---|
| malformed_target_key | 700 | 0 | 0 |
| mutual_exclusivity | 44 | 0 | 1 |
| quote_not_grounded | 26 | 6 | 1 |
| mixed_operator | 13 | 0 | 0 |
| self_reference | 7 | 0 | 0 |
| unknown_referenced_state | 7 | 0 | 0 |

---

## 7. Cycles de précondition détectés

| Modèle | Cas avec ≥1 cycle | Total |
|---|---|---|
| `llama-3.1-8b` | 33 | 215 |
| `mistral-nemotron` | 0 | 215 |
| `gpt-oss-20b` | 2 | 215 |

Confirme à l'échelle que le cycle réel observé auparavant (un seul cas, `pipeline_observations.md`
section 4) n'était pas isolé pour `llama-3.1-8b` (33/215, 15.3%) — mais reste rare pour les deux
autres modèles.

---

## 8. Richesse de U (nombre d'états extraits)

| Modèle | n | Moyenne | Min | Max |
|---|---|---|---|---|
| `llama-3.1-8b` | 194 | 11.8 | 4 | 80 |
| `mistral-nemotron` | 53 | 16.4 | 6 | 51 |
| `gpt-oss-20b` | 130 | 16.2 | 5 | 82 |

`n` ici compte les cas où `state_space` a été produit, indépendamment du reste du pipeline —
différent des `n` des sections précédentes.

---

## 9. États de U jamais matchés (biais structurel) — un motif par description, pas aléatoire

Pour chaque description et chaque modèle, les états `U` les plus fréquemment `UNRESOLVABLE`
sont **spécifiques à la description**, pas dispersés aléatoirement — ex. `E_j03` :
`insured_employment.gainful`/`group.protected` (21 occurrences chacun, `llama-3.1-8b`),
`agricultural_establishment.covered`/`forestry_establishment.covered` (25 chacun,
`mistral-nemotron`) ; `V_k09` : `order.withdrawn` (31, `llama-3.1-8b`). Détail complet dans la
sortie de `analyze_dataset_runs.py`, section 3.

**Ouvert** : à déterminer si ces motifs récurrents par description reflètent un vrai manque de
couverture dans le process (le concept n'est jamais modélisé dans aucun des 8-11 BPMN de cette
description) ou une extraction `U` trop permissive côté texte (l'état existe dans `U` mais ne
correspond à rien de réaliste à matcher).

---

## 10. Complexité du BPMN (nombre d'arêtes DFG) vs taux SATISFIED

| Modèle | n | Spearman ρ | p |
|---|---|---|---|
| `llama-3.1-8b` | 111 | 0.165 | 0.084 |
| `mistral-nemotron` | 41 | 0.014 | 0.931 |
| `gpt-oss-20b` | 88 | 0.084 | 0.439 |

Signal faible et non significatif au seuil conventionnel (0.05) pour les trois modèles ; le plus
proche de la significativité (`llama-3.1-8b`, p=0.084) reste au-dessus du seuil.

---

## 11. Impact de la boucle perdue (PM4Py) sur le taux SATISFIED

| Modèle | Avec boucle perdue (n, SATISFIED moyen) | Sans (n, SATISFIED moyen) |
|---|---|---|
| `llama-3.1-8b` | 17, 0.70 | 94, 0.64 |
| `mistral-nemotron` | 10, 0.68 | 31, 0.46 |
| `gpt-oss-20b` | 11, 0.66 | 77, 0.45 |

Direction cohérente sur les trois modèles : les cas avec boucle perdue ont un taux SATISFIED
**plus élevé**, pas plus bas — contre-intuitif par rapport à l'hypothèse initiale ("perdre de
l'information dégrade le résultat").

**Ouvert** : aucun contrôle de confusion effectué (les BPMN avec boucle sont peut-être
systématiquement plus simples/petits par ailleurs, ce qui expliquerait le taux plus élevé
indépendamment de la perte elle-même — cf. section 10, où la complexité seule montre déjà un
effet dans le même sens). Non démêlé.

---

## 12. Accord inter-modèles sur les mêmes cas

215 cas communs aux trois modèles (mêmes clés `desc/fichier`, indépendamment du contenu produit).
**Accord total (même statut dominant SATISFIED/VIOLATED/UNRESOLVABLE) : 58/215 (27.0%)**.

Ce chiffre est fortement influencé par les taux de complétude très différents (section 1) : un
"désaccord" est souvent `{'llama-3.1-8b': 'SATISFIED', 'mistral-nemotron': None, 'gpt-oss-20b':
'SATISFIED'}` — c'est-à-dire un cas où `mistral-nemotron` n'a simplement pas produit de rapport
(échec en amont), pas un vrai désaccord de jugement entre trois rapports complets.

**Ouvert** : le taux d'accord recalculé uniquement sur les cas où les trois modèles ont produit
un rapport complet (sans `None`) n'a pas été isolé séparément — le chiffre actuel mélange
désaccord de jugement et incomplétude de données. **Partiellement traité en section 17
ci-dessous**, bien que sur un échantillon trop petit pour trancher (n=10).

---

## 13. Ratio de compression du regroupement par cause racine (`report_node`)

| Modèle | Échecs bruts | Causes distinctes | Ratio |
|---|---|---|---|
| `llama-3.1-8b` | 556 | 199 | 2.79× |
| `mistral-nemotron` | 378 | 153 | 2.47× |
| `gpt-oss-20b` | 696 | 301 | 2.31× |

Cohérent sur les trois modèles (2.3× à 2.8×) — confirme à l'échelle que `find_root_term`
regroupe effectivement plusieurs échecs en cascade sous une cause commune, pas seulement sur les
1-2 exemples testés initialement (`report_observations.md`).

---

## 14. Citations manquantes dans l'alignement

| Modèle | Manquantes | Total | % |
|---|---|---|---|
| `llama-3.1-8b` | 0 | 734 | 0.0% |
| `mistral-nemotron` | 0 | 342 | 0.0% |
| `gpt-oss-20b` | 0 | 790 | 0.0% |

Aucune régression détectée à l'échelle sur le fix `quote` (cf. `alignment_observations.md`
section 2.1) — 0% sur les trois modèles, 1866 vérifications au total.

---

## 15. Piste antonymie WordNet pour la détection des matchs sémantiquement incohérents — abandonnée

Idée testée avant de passer à l'approche par score de confiance (sections 16-20) : détecter
systématiquement, via les lexnames et antonymes WordNet (mécanisme déjà utilisé ailleurs dans le
pipeline pour `state_space_node`), les paires activité/état matchées qui seraient des antonymes
directs — ex. une activité contenant "unemployed" matchée à un état contenant "employed".

**Testée contre le seul cas réel confirmé disponible** (`Unemployed` matché à `job.permanent`,
score 0.29, cf. `state_matching_observations.md`) avant tout déploiement à l'échelle : **0 match
antonyme détecté sur ce cas**, y compris en élargissant la recherche aux antonymes des synonymes.
Cause identifiée, pas un bug du script : WordNet relie `"unemployed"` à `"employed"` par
antonymie lexicale directe, mais ne relie pas `"permanent"` à l'un ou l'autre — la contradiction
sémantique dans ce cas passe par une inférence en deux sauts ("permanent" implique "employé", qui
contredit "unemployed") que WordNet, limité aux antonymes lexicaux directs, ne capture pas.

**Conclusion retenue** : limite structurelle de l'outil, pas un défaut de mise en œuvre à
corriger. La piste est abandonnée, remplacée par l'exploitation d'un signal déjà disponible sans
rien construire de nouveau — le score de confiance du match lui-même (sections 16-20).

---

## 16. Conclusions reposant sur un match à faible confiance (score < 0.35)

Signal de repli après l'abandon de la piste antonymie (section 15) : le match fautif connu
(`Unemployed → job.permanent`) avait un score de confiance bas (0.29), disponible sans calcul
supplémentaire. Seuil initial choisi : score < 0.35 sur le terme ou la cible impliquée dans une
conclusion `SATISFIED`/`VIOLATED` (les `UNRESOLVABLE` n'ont par définition pas de match).

| Modèle | Conclusions à faible confiance | Total conclusions | % |
|---|---|---|---|
| `llama-3.1-8b` | 139 | 534 | 26.0% |
| `mistral-nemotron` | 64 | 144 | 44.4% |
| `gpt-oss-20b` | 69 | 321 | 21.5% |

Seuil de vigilance fixé à l'avance (>10-15%) franchi, largement, sur les trois modèles — ce qui
justifie de rouvrir la discussion sur un seuil de confiance en aval du matching, écartée dans
`state_matching_observations.md` section 6.3.

**Réserve posée dès cette section, confirmée ensuite (section 19)** : un score bas signale que
l'embedding est incertain, ce qui est un sur-ensemble de "le match est faux" — une paraphrase
correcte mais formulée différemment du candidat `U` peut recevoir un score bas sans être une
erreur. Le seuil 0.35 n'a été calibré que sur un seul cas connu ; le distinguer d'un vrai
indicateur d'erreur nécessite une validation indépendante de ce point de calibration unique.

**Ouvert à ce stade** : quelle part de ce 21-44% correspond à de vraies erreurs de matching
plutôt qu'à des paraphrases correctes sous-notées par l'embedding — non trancher sans inspection
qualitative ciblée.

---

## 17. Confiance du matching vs accord inter-modèles — tentative de validation sans ground truth

Piste explorée pour valider le signal de la section 16 sans dépendre de Zenodo (dont
l'échantillon "mauvais" est trop petit, section 3bis) ni d'annotation humaine lourde : si le
score de confiance capture un vrai signal d'erreur, les cas à taux élevé de conclusions à faible
confiance devraient montrer un désaccord inter-modèles plus fréquent que les cas à confiance
haute — sur le **statut final** (SATISFIED/VIOLATED/UNRESOLVABLE), qui est dans un vocabulaire
commun aux trois modèles, contrairement à `U`/`Pre` bruts qui ne le sont pas.

**Restriction nécessaire avant le calcul** : uniquement les cas où les trois modèles ont produit
un rapport complet (sans `None`), pour ne pas confondre désaccord de jugement et incomplétude de
données (point ouvert de la section 12).

**Résultat** : 10 cas sur 215 satisfont cette restriction — reflet direct du faible taux
d'exploitabilité de `mistral-nemotron` (19.1%, section 1), qui limite mécaniquement toute
comparaison à trois rapports complets.

| Groupe (médiane du taux de faible confiance = 0.250) | n | Accord |
|---|---|---|
| Taux de faible confiance élevé | 5 | 2/5 (40.0%) |
| Taux de faible confiance faible | 5 | 3/5 (60.0%) |

Écart de +20.0 points dans la direction attendue (les cas à faible confiance désaccordent plus),
mais **chi²=0.000, p=1.0000** sur une table parfaitement symétrique (2,3 / 3,2) — non
interprétable statistiquement sur un échantillon de cette taille.

**Limite supplémentaire, distincte de la taille d'échantillon** : les trois pipelines LLM
partagent le **même backend d'embedding** (`nvidia/llama-nemotron-embed-1b-v2`,
`state_matching_observations.md` section 3.5) — un biais systématique de ce moteur
apparaîtrait de façon identique dans les trois modèles simultanément. Un accord inter-modèles
élevé ne prouverait donc pas l'absence d'erreur de matching, seulement l'absence d'erreur
spécifique à la génération LLM (extraction, formulation) en amont du matching partagé.

**Ouvert** : le volume actuel (n=10) ne permet aucune conclusion, positive ou négative. Point
non résolu, dépendant de la levée des problèmes d'infrastructure listés en section 2 (rate
limiting notamment) pour augmenter le nombre de cas à trois rapports complets.

---

## 18. Distribution des scores de matching — test de bimodalité (calibration non supervisée)

Alternative à la calibration sur un seul cas connu (section 16) ou sur l'échantillon Zenodo
"mauvais" trop restreint (section 3bis) : tester si la distribution complète des scores de
matching (tous les scores produits par `state_matching_node`, avant tout filtrage sur le statut
de la conclusion) a une forme bimodale — un pic de "bons matches" et un pic de "matches ratés".
Si oui, la vallée entre les deux pics donne un seuil calibré sur la structure de la distribution
elle-même, sans dépendre de Zenodo ni d'un cas isolé annoté. Méthode : comparaison d'un GMM
(Gaussian Mixture Model) à 1 vs 2 composantes par critère BIC, sur l'ensemble des scores bruts
de `matches` par modèle.

| Modèle | n | Moyenne | Écart-type | Médiane | % scores < 0.35 (seuil actuel) |
|---|---|---|---|---|---|
| `llama-3.1-8b` | 2811 | 0.450 | 0.220 | 0.434 | 36.5% |
| `mistral-nemotron` | 745 | 0.400 | 0.207 | 0.369 | 46.6% |
| `gpt-oss-20b` | 1844 | 0.492 | 0.222 | 0.485 | 29.7% |

**Premier constat, avant même le GMM** : le seuil actuel (0.35) est situé juste en dessous de la
médiane pour les trois modèles — 30% à 47% de **tous** les scores de matching (pas seulement
ceux menant à une conclusion retenue) sont sous ce seuil. Un seuil censé repérer une minorité de
cas anormaux ne devrait pas capturer une aussi grande part de la population totale.

**Résultat du test GMM** :

| Modèle | BIC (1 composante) | BIC (2 composantes) | Moyennes (2 comp.) | Poids | Seuil suggéré (croisement) |
|---|---|---|---|---|---|
| `llama-3.1-8b` | -523.1 | -765.7 | 0.294 / 0.642 | 0.55 / 0.45 | 0.472 |
| `mistral-nemotron` | -217.7 | -308.0 | 0.288 / 0.633 | 0.67 / 0.33 | 0.487 |
| `gpt-oss-20b` | -298.3 | -456.2 | 0.315 / 0.673 | 0.51 / 0.49 | 0.491 |

Le BIC préfère nettement 2 composantes sur les trois modèles (écarts de 90 à 243), avec des
moyennes de composante basse remarquablement proches entre modèles (0.29-0.32) et cohérentes
avec le score du cas connu (0.29) — bimodalité statistiquement confirmée, stable au changement de
vocabulaire `U` d'un modèle à l'autre.

**Réserve immédiate, avant d'agir sur ce résultat** : les trois modèles partagent le même
backend d'embedding (section 17) — la stabilité de la bimodalité entre modèles confirme que la
propriété est robuste au changement de vocabulaire, mais ne garantit pas que la composante basse
correspond à de vraies erreurs de matching plutôt qu'à une propriété générale du comportement de
ce moteur d'embedding face à la reformulation lexicale (paraphrase). Distinction tranchée en
section 19.

**Le point le plus actionnable de cette section** : le seuil suggéré par le croisement des deux
gaussiennes (≈0.47-0.49) est **nettement plus haut** que le seuil actuel de 0.35, sur les trois
modèles de façon cohérente. Testé directement en section 19.

---

## 19. Recalcul avec le seuil GMM — le nouveau seuil invalidé par inspection manuelle

Section 16 recalculée avec le seuil issu du croisement des deux gaussiennes (section 18) au lieu
du seuil fixe 0.35, pour mesurer l'écart concret entre les deux calibrations.

| Modèle | Seuil fixe (0.35) | Seuil GMM | Écart |
|---|---|---|---|
| `llama-3.1-8b` | 139/534 (26.0%) | 269/534 (50.4%, seuil=0.472) | +24.3 points |
| `mistral-nemotron` | 64/144 (44.4%) | 85/144 (59.0%, seuil=0.487) | +14.6 points |
| `gpt-oss-20b` | 69/321 (21.5%) | 122/321 (38.0%, seuil=0.491) | +16.5 points |

Le seuil GMM flag entre 38% et 59% de **toutes** les conclusions — une majorité ou quasi-majorité
selon le modèle. Un signal qui capture la majorité des cas cesse d'être discriminant.

**Inspection manuelle des conclusions flaggées uniquement par le nouveau seuil** (zone
0.35-0.49, échantillon examiné directement dans les résultats imprimés) — plusieurs exemples
identifiés comme des matchs sémantiquement corrects, simplement formulés différemment du
candidat `U` :

- `process.continued <- machine.restarted` (score cible = 0.3775, `llama-3.1-8b`) — relation
  logique correcte (une machine redémarrée entraîne la continuation du processus).
- `character_name.chosen <- character_name.available` (score cible = 0.4566, `gpt-oss-20b`) —
  même famille lexicale, relation évidente et correcte.
- `status_update.written <- offer.accepted` (score cible = 0.4621, `llama-3.1-8b`) — correct,
  une offre acceptée entraîne la rédaction d'une mise à jour de statut.
- `registration.completed <- course.selected` (score cible = 0.4485, `gpt-oss-20b`) — correct,
  relation logique claire.

**Conclusion tirée de cette inspection, révisant l'interprétation de la section 18** : la
bimodalité détectée par le GMM est statistiquement réelle et stable, mais elle sépare
vraisemblablement "chevauchement lexical fort" (mots identiques ou proches, scores hauts,
composante ~0.6-0.7) de "chevauchement lexical faible malgré une relation sémantique correcte"
(paraphrase, verbe différent, composante basse ~0.3) — **pas** "bons matches" de "mauvais
matches". Le seul cas faux confirmé (`Unemployed → job.permanent`, 0.29) tombe dans la
composante basse, mais les exemples ci-dessus, également dans cette composante, montrent que
beaucoup de bons matches y tombent aussi. Le seuil GMM (~0.48), adopté tel quel pour un rejet
automatique, produirait probablement un taux de faux positifs (bons matches signalés comme
suspects) plus élevé que le gain réel en détection de vraies erreurs — il dégraderait la
précision du signal plutôt que de l'améliorer.

**Conclusion méthodologique retenue** : la calibration non supervisée par bimodalité (section
18), bien qu'exécutée correctement et confirmant un phénomène statistique réel, ne suffit pas
seule à définir un seuil opérationnel de rejet — elle doit être complétée par une inspection
qualitative de ce que chaque composante représente réellement avant d'être actionnée. Le seuil
fixe initial (0.35), bien que calibré sur un seul cas, n'est pas invalidé par ce test — au
contraire, il reste pour l'instant le candidat le plus prudent des deux, dans l'attente d'une
annotation ciblée (point ouvert ci-dessous).

**Ouvert** : la zone 0.35-0.49 comporte 130 + 21 + 53 = 204 cas au total sur les trois modèles,
tous imprimables directement par le script actuel (section 24 de `analyze_dataset_runs.py`). Un
échantillon aléatoire de 20-30 cas dans cette zone spécifique, annoté manuellement en "match
sémantiquement correct malgré le score bas" vs "vraie erreur", donnerait un taux de précision
réel du signal de confiance — travail non encore fait, mais désormais ciblé sur une zone précise
plutôt que sur l'ensemble de la distribution.

---

## 20. Synthèse des points laissés ouverts (à traiter, pas à conclure ici)

1. Diagnostiquer précisément la cause de `bpmn_to_spo: add_in_arc/add_out_arc` sur les 24
   fichiers communs aux trois modèles.
2. Investiguer les réponses vides spécifiques à `gpt-oss-20b`.
3. Corriger le retry sur rate-limit (backoff exponentiel plutôt que délai fixe) si l'objectif est
   de maximiser le volume de cas exploitables — sans garantie que ça change la section 3
   (corrélation Zenodo), puisque les cas actuellement propres montrent déjà le même résultat.
   **Ce point conditionne aussi la section 17** (accord inter-modèles sur cas complets, n=10
   actuellement, trop petit pour trancher) — plus de cas exploitables pour `mistral-nemotron`
   en particulier augmenterait directement la taille de cet échantillon.
4. Isoler l'effet "bonne formation BPMN" (mesuré par Zenodo, non mesuré par le pipeline) de
   l'effet "fidélité au texte" (mesuré par les deux) avant de conclure sur la section 3.
5. Contrôler la confusion complexité/boucle perdue (section 10 vs 11).
6. Recalculer l'accord inter-modèles (section 12) sur le sous-ensemble à trois rapports complets
   — amorcé en section 17, mais non concluant sur l'échantillon actuel (n=10).
7. Déterminer si les motifs de section 9 reflètent un vrai manque de couverture processus ou une
   sur-extraction de `U`.
8. Reproductibilité à `temperature=0` toujours non testée (cf. `pipeline_observations.md`,
   limite 3) — aucune donnée de ce run ne permet de la vérifier (chaque cas n'a été exécuté
   qu'une fois par modèle).
9. Le déséquilibre de classe Zenodo (section 3bis, 0 à 6 cas "mauvais" par modèle sur 37 à 104)
   limite la puissance de tout test binaire bon/mauvais sur ce corpus — augmenter la couverture
   (plus de modèles BPMN testés) est la seule voie identifiée pour lever cette limite, pas un
   changement de méthode d'analyse.
10. **Nouveau** : annoter manuellement un échantillon (20-30 cas) dans la zone de score
    0.35-0.49 (section 19) pour établir un taux de précision réel du signal de confiance de
    matching, avant de considérer tout ajustement du seuil de rejet en aval du matching. C'est
    désormais le point le plus concret et le mieux borné de cette liste — contrairement aux
    points 3, 6 et 9, il ne dépend d'aucun autre correctif préalable et peut être traité
    directement avec les données déjà produites.
11. Si l'annotation du point 10 confirme un taux d'erreur significatif dans la composante basse
    du GMM, envisager une méthode de calibration alternative fondée sur des négatifs
    synthétiques (permutation contrôlée d'activités entre cas, dans l'esprit du calibrage déjà
    appliqué à `alignment_node`, cf. `pipeline_observations.md` section 3.6) plutôt qu'une
    bimodalité non supervisée, qui s'est révélée insuffisante seule pour distinguer erreur
    sémantique et simple variation lexicale.