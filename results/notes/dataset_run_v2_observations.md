# Observations — Run dataset v2 (sous-ensemble de développement)

Notes de vérification empirique des trois correctifs appliqués au pipeline après l'inspection
qualitative des 61 cas `VIOLATED` du run v1 (`dataset_run_observations.md`, section citée comme
"v1" dans tout ce document) : démotion par confiance (`alignment_node.py`), filtre des
événements Start/End (`bpmn_to_spo_node.py`), déduplication référentielle de `U`
(`state_space_node_v3.py`, étape C). S'y ajoutent le backoff exponentiel et le préprocessing
anti-documentation (`run.py`, `bpmn_to_spo_node.py`) et la rédaction du rapport par le modèle
sous test lui-même (`report_model=model_key`), plutôt que par un rédacteur fixe hérité par
défaut.

**Statut de ce document : diagnostic sur le sous-ensemble de développement, pas les chiffres du
papier.** Comme convenu (`analyze_dataset_runs.py`, discussion sur la validité statistique du
subset), ce run sert à valider que les correctifs produisent l'effet attendu — pas à mesurer un
taux généralisable. Un run complet sur les 24 descriptions reste nécessaire avant toute
affirmation chiffrée dans le papier.

---

## 1. Périmètre du run

**Sous-ensemble** : 7 descriptions (`E_j02`, `E_j03`, `G_g03`, `M_g01`, `R_j02`, `V_k09`,
`X_g01`), sélectionnées par stratification (une par famille de préfixe, couverture des trois
familles d'artefacts identifiées en v1) — jamais par convenance. 60 fichiers BPMN par modèle,
trois modèles (`llama-3.1-8b`, `mistral-nemotron`, `gpt-oss-20b`), identiques à v1 pour permettre
une comparaison appariée.

**Incident de run signalé et traité** : une coupure réseau survenue entre deux lancements a
produit des cascades `Connection error.` sur un grand nombre de cas (visibles dans les logs
intermédiaires), invisibles à `already_done()` (qui ne vérifie que l'existence du fichier, pas
son contenu). Nettoyés via `cleanup_network_failures.py` puis retraités par une relance de
`run.py` — les chiffres ci-dessous sont ceux du run après nettoyage.

**Comparaison avec v1** : v1 portait sur les 215 cas du corpus complet (24 descriptions). Les
taux (pourcentages, rho) sont comparables entre v1 et v2 ; **les décomptes bruts ne le sont
pas** — v2 porte sur un sous-ensemble 3,6× plus petit, non aléatoire.

---

## 2. Vérification empirique des trois correctifs

### 2.1 Démotion par confiance — confirmée, sans exception

**Fait vérifié, pas estimé** : sur les 26 termes `VIOLATED` restants dans le dump v2 (tous
modèles confondus), **0 cas** repose sur un score de matching inférieur au seuil de 0.35, côté
terme ou côté cible. Score minimum observé sur l'ensemble des 26 cas : **0.3534**. Comptage
exhaustif, pas un échantillon (`violated_cases_dump.md`, recompté par script indépendant du
pipeline d'analyse).

Comparaison directe avec v1 : le dump v1 (61 cas) contenait de nombreux `VIOLATED` reposant sur
des scores extrêmes (`0.08` pour *"Brag to friends"*, `0.13` pour *"Pickerl Issuance"*, `0.19`
pour *"twitter_account.connected"*, etc.) — la famille d'artefacts identifiée comme "score
effondré" dans l'inspection v1. **Cette famille a disparu intégralement du dump v2**, résultat
exact attendu du mécanisme de démotion (`check_term`, `alignment_node.py`) et de sa garantie
formelle (monotonie en confiance — cf. note de formalisme TGMS, Théorème 3.5).

**Non vérifié à ce stade** : la fréquence de démotion elle-même (combien de candidats `VIOLATED`
ont été effectivement rétrogradés en `UNRESOLVABLE`) n'est pas encore comptée — `check_term`
pose le champ `low_confidence` sur chaque terme concerné, mais aucune fonction d'analyse ne
l'agrège pour l'instant. À ajouter si ce chiffre est utile pour le papier (ratio artefacts
supprimés / candidats totaux).

### 2.2 Filtre Start/End — confirmé sur le cas qui l'a motivé

Le triplet d'artefacts `M_g01/10.bpmn2.xml` causé par l'événement de départ nommé *"Printing a
3D model"* (matché à 0.85 sur `3d_model.printed` en v1, trois fausses accusations `VIOLATED` sur
ce seul cas) **a disparu du dump v2**. Le seul `VIOLATED` restant sur `M_g01/10.bpmn2.xml`
(`color.checked` / `color.in_stock`, scores 0.4648/0.5952) est un cas structurellement différent,
entre deux vraies activités du process, jamais lié à un événement de délimitation.

### 2.3 Déduplication référentielle — pas de contre-exemple, pas de preuve positive directe

Aucune paire d'entités quasi-dupliquées (motif `car`/`car_service`, `3d_model`/`model` observé
en v1) n'apparaît dans les 26 cas `VIOLATED` du dump v2. Cohérent avec le fonctionnement attendu
du correctif, mais **pas une preuve positive isolée** — le sous-ensemble ne contient peut-être
simplement aucune occurrence de ce motif indépendamment de la déduplication. À confirmer sur le
run complet, où la fréquence de fusions réelles (`state_space_dedup.merged`) pourra être comptée
et corrélée à la disparition (ou non) de ce type d'artefact.

---

## 3. Gain d'exploitabilité — l'effet le plus fort mesuré dans ce run

| Modèle | v1 (215 cas, %) | v2 (60 cas, %) |
|---|---|---|
| `llama-3.1-8b` | 65.6% | **76.7%** |
| `mistral-nemotron` | 19.1% | **91.7%** |
| `gpt-oss-20b` | 42.3% | **100.0%** |

Attribution, par cause, permise par la quantification exhaustive (section 25 de
`analyze_dataset_runs.py`) :

- **`mistral-nemotron`** : le run v1 perdait 162/215 cas à des 429 jamais retentés (le run
  écrivait le cas comme "fait" au premier échec transitoire). Le run v2 ne montre plus que 3-5
  cas avec erreur interne sur 60, tous des `Error code: 500` isolés (panne serveur ponctuelle,
  pas un motif de rate limiting) — cohérent avec le fix retry-sur-erreurs-confinées de
  `run_one`.
- **`gpt-oss-20b`** : 0 erreur sur 60 cas, 0 cas sans rapport. v1 montrait 109 réponses vides
  (`raw response=''`) attribuées à l'hypothèse d'un budget de raisonnement épuisé avant le
  contenu (`precondition_observation.md`, section 11). Le run v2 introduit
  `reasoning_effort="low"` pour ce modèle — **la disparition totale des réponses vides est
  cohérente avec cette hypothèse, mais n'a pas été isolée d'un effet du backoff seul** (les deux
  changements sont arrivés dans le même run). Non tranché entre les deux causes à ce stade — un
  test isolant `reasoning_effort` seul (sans backoff, ou vice versa) trancherait, non fait ici.
- **`llama-3.1-8b`** : gain plus modeste (65.6% → 76.7%), cohérent avec le fait que sa cause
  d'échec dominante (clés de précondition malformées, 34.2% en v1, 25.6% en v2) est un problème
  de contenu du modèle, pas d'infrastructure — non affecté par le backoff, seul le préprocessing
  anti-associations peut avoir marginalement aidé en réduisant les échecs `bpmn_to_spo` en
  cascade.

---

## 4. Mode d'échec nouvellement identifié — grammaire sans négation

**Observation, pas encore documentée avant ce run.** Un cluster de `VIOLATED` récurrent sur
`G_g03` (personnage WoW) apparaît chez les trois modèles, avec une structure commune : le texte
source exprime une condition **négative** (*"As soon as you have an active WoW subscription. If
not, you can select the payment method."*), et la précondition extraite pour
`payment_method.selected` référence `battle_net_account.active` comme terme requis — alors que
le texte dit littéralement l'inverse (le paiement se sélectionne quand l'abonnement n'est **pas**
actif).

**Cause probable, structurelle, pas un bug de matching** : la grammaire des guards
(`_validate_one`, Definition 2) n'admet que des conjonctions/disjonctions pures d'états
positifs — aucune négation représentable. Face à une condition textuelle négative, le LLM
d'extraction semble substituer la précondition positive la plus plausible (l'état
"complémentaire" du bon sens), produisant une précondition syntaxiquement valide mais
sémantiquement inversée. Ceci est indépendant du matching et des trois correctifs de cette
session — c'est un problème de `precondition_node_with_retry`, pas de `state_matching_node` ni
`alignment_node`.

**Portée observée** : concentré sur `G_g03` dans ce sous-ensemble (texte riche en constructions
conditionnelles imbriquées — *"If not... If you choose... If you choose..."*). Non quantifié à
l'échelle du corpus complet — à vérifier si ce motif se généralise ou reste spécifique à ce type
de texte. **Non corrigé, seulement documenté** : ajouter la négation à la grammaire serait un
changement de scope significatif (Definition 2 du papier), à ne considérer que si la fréquence
sur le corpus complet le justifie.

---

## 4bis. Bug trouvé et corrigé — fuite du placeholder `_MISSING_PLACEHOLDER` dans la prose

**Découvert par le taux `quote_verbatim` (section 30, analyse post-run) — un effet de bord de
la mesure, pas son objectif initial.** Sur 373 entrées vérifiées (94 llama + 132 nemotron + 147
gpt-oss), 8 violations (`llama-3.1-8b` : 7/94, `gpt-oss-20b` : 1/147, `mistral-nemotron` : 0/132)
ne correspondent **pas** au bug de traduction/paraphrase déjà documenté
(`report_observations.md`, section 3.3) — un mécanisme distinct, jamais identifié jusqu'ici.

**Cause exacte, vérifiée sur le code** (`report_node.py`) : `label_for_state()` retourne le
littéral `_MISSING_PLACEHOLDER` (`"(aucune activite du processus ne correspond clairement a
cela)"`) quand un état n'a reçu aucune activité BPMN appariée. Un garde existait déjà pour
protéger `missing_description` dans `generate_cluster_explanation`, mais **cinq autres sites
d'appel n'en bénéficiaient pas** : `target_label` (dans les deux templates de prompt,
`VIOLATED` et `UNRESOLVABLE`), le champ `root_cause`, la liste `affected_activities`, et le texte
"Cette même cause affecte aussi". Résultat observé littéralement dans la prose générée par le
LLM rédacteur : *"avant que l'activité "(aucune activité du processus ne correspond clairement à
cela)" puisse se produire"* — le placeholder technique traité comme un nom d'activité réel, le
LLM n'ayant aucune raison de le reconnaître comme spécial. Le flag `quote_verbatim=False` sur ces
cas est un effet secondaire cohérent : le LLM, confronté à un "nom d'activité" absurde, produit
une prose qui s'éloigne de la citation plutôt que de l'ancrer dessus.

**Corrigé** : introduction de `display_label_for_state()`, un point de garde unique
(repli vers `_readable_state()` si le placeholder est retourné), substitué aux six sites
d'appel de `label_for_state()` qui en étaient dépourvus — au lieu de dupliquer le garde
site par site (source exacte de l'omission initiale). Vérifié sur une reconstruction exacte
du cas réel `gpt-oss-20b/G_g03/4.bpmn2.xml` (`iban.entered` jamais matché) : avant fix,
`label_for_state` retourne le placeholder brut ; après fix, `"iban entered"` — comportement
inchangé sur les états correctement matchés.

**Non encore fait** : ce fix n'a pas encore tourné sur un run — les 8 cas ci-dessus datent
d'avant le correctif. À vérifier sur la prochaine exécution que `quote_verbatim` grimpe vers
100% sur ces 8 entrées précises (elles ne testaient jamais la règle de traduction, seulement
cette fuite — le taux de fidélité verbatim pourrait ne pas capturer *toutes* les erreurs
possibles de traduction si aucune n'était présente dans cet échantillon).

---

## 5. Hypothèse ouverte, non tranchée — cycles et déduplication

Taux de cycles de précondition détectés :

| Modèle | v1 | v2 |
|---|---|---|
| `llama-3.1-8b` | 33/215 (15.3%) | 14/60 (**23.3%**) |
| `gpt-oss-20b` | 2/215 (0.9%) | 7/60 (**11.7%**) |
| `mistral-nemotron` | 0/215 | 0/60 |

Hausse notable pour deux modèles sur trois. Deux hypothèses non départagées :
1. **Composition du sous-ensemble** — `G_g03` et `R_j02`, présents dans ce subset, pourraient
   être structurellement plus propices aux cycles indépendamment de tout changement de pipeline.
2. **Effet secondaire de la déduplication** — fusionner deux entités référentiellement
   identiques peut recréer artificiellement un cycle absent avant fusion (si `A` dépendait de
   `B` et `B` de `C`, fusionner `C` dans `A` referme la boucle `A → B → A`).

**Non tranché ici.** Un test ciblé (comparer les cycles détectés avec et sans l'étape de
déduplication activée, sur les mêmes cas) trancherait — pas fait dans ce run. À traiter avant
d'écrire quoi que ce soit sur les cycles dans le papier, puisque l'hypothèse 2 impliquerait un
compromis du correctif de déduplication non identifié jusqu'ici.

---

## 6. Corrélation Zenodo — signal à ne pas sur-interpréter

| Modèle | v1 (n) | v1 rho | v2 (n) | v2 rho | v2 p |
|---|---|---|---|---|---|
| `llama-3.1-8b` | 111 | 0.012 | 38 | **0.322** | 0.049 |
| `mistral-nemotron` | 41 | 0.079 | 55 | 0.063 | 0.648 |
| `gpt-oss-20b` | 88 | 0.000 | 57 | 0.048 | 0.725 |

Le saut pour `llama-3.1-8b` (quasi nul → 0.322, marginalement significatif) **ne doit pas être
reporté comme un résultat** : le sous-ensemble a été choisi par stratification délibérée (couvrir
les familles d'artefacts), jamais tiré aléatoirement — un ρ calculé dessus n'estime pas la
corrélation réelle sur le corpus. Section 26 (contrôle sur sous-ensemble BPMN structurellement
propre) confirme la même instabilité : `rho=0.304, p=0.072` pour llama sur ce sous-ensemble
encore plus restreint (n=36) — cohérent avec un effet d'échantillon, pas un signal robuste.
**Seul le run complet (24 descriptions) pourra trancher.**

---

## 7. Ce qui reste ouvert après ce run

1. **Comparaison appariée stricte v1/v2** — non faite ici. Les chiffres ci-dessus comparent v2
   au run v1 complet (215 cas), pas à un sous-ensemble v1 filtré sur les 7 mêmes descriptions.
   Une comparaison rigoureuse nécessite de refiltrer v1 sur `DESCRIPTIONS_SUBSET` avant de
   comparer terme à terme — pas fait, les tendances ci-dessus (exploitabilité, disparition des
   artefacts à score bas) sont qualitativement claires mais pas quantitativement appariées.
2. **Taux `quote_verbatim` par modèle-rédacteur** — calculé (section 30, ci-dessus) :
   `llama-3.1-8b` 92.6% (87/94), `mistral-nemotron` 100% (132/132), `gpt-oss-20b` 99.3%
   (146/147). Les 8 violations identifiées ne sont pas le bug de traduction déjà documenté
   mais un bug distinct (fuite de placeholder, section 4bis), **trouvé et corrigé** dans cette
   session. À revérifier sur un run avec le fix pour confirmer sa résolution et pour évaluer si
   le bug de traduction original (`report_observations.md` 3.3) est présent ou non à l'échelle
   du run v2 — l'échantillon actuel de violations ne le teste pas.
3. **Fréquence de démotion elle-même** (candidats `VIOLATED` → `UNRESOLVABLE`, pas seulement
   "0 cas restant sous le seuil") — non comptée, cf. section 2.1.
4. **Cause exacte du gain gpt-oss-20b** (backoff vs `reasoning_effort` vs les deux) — non isolée,
   cf. section 3.
5. **Cycles et déduplication** — hypothèse non tranchée, cf. section 5.
6. **Négation dans la grammaire des guards** — fréquence sur le corpus complet non mesurée,
   cf. section 4.
7. **Rappel de l'étude de perturbation (classe `required`)** — premier run fait (section 9),
   mais le biais de génération identifié (mutants non ciblés sur les activités réellement
   couvertes par un guard) rend le chiffre actuel non interprétable comme mesure pure du
   mécanisme. Correctif de ciblage identifié, pas encore implémenté.

---

## 8. Étude de perturbation — premier run, résultat mixte et diagnostiqué

**Contexte** : `mutate_bpmn.py` génère 126 mutants applicables sur les 7 BPMN de base retenus
(3 variantes × 6 opérateurs), `run_perturbation_study.py` rejoue uniquement la branche droite
(`bpmn_to_spo → state_matching → alignment`) sur chaque mutant, contre le `U`/`G` figés du run
v2 correspondant (aucun appel LLM, seulement des appels d'embedding — cf. docstring de tête du
script pour le principe de contrôle expérimental).

### 9.1 Résultat solide — classe `forbidden` et `blind_spot`, zéro exception

| Opérateur | Modèles cumulés | Résultat |
|---|---|---|
| `insert_activity` + `shuffle_xml` (forbidden) | 3 modèles, 102 mutants | **0 faux positif, 0 régression** (`false_positive_violated` et `any_regression` à 0 partout) |
| `rewire_gateway` (blind_spot) | 3 modèles, 51 mutants | **0 changement de DFG inattendu** (attendu et confirmé) |

Confirmation empirique directe, sans exception sur l'échantillon testé, du Théorème de
monotonie en enrichissement (note de formalisme TGMS, section 3.3, Corollaire 3.8) : aucune
activité ajoutée n'a dégradé un verdict `SATISFIED`, sur trois modèles et deux opérateurs
distincts. Et confirmation mécanique de l'angle mort assumé sur le type de gateway (la relation
`"follows"` du DFG est bien insensible à XOR/AND, comme actée dans `bpmn_to_spo_node.py`).

### 9.2 Résultat faible et diagnostiqué — rappel sur la classe `required`

| Opérateur | `llama-3.1-8b` | `mistral-nemotron` | `gpt-oss-20b` |
|---|---|---|---|
| `remove_activity` | 2/12 (16.7%) | 4/18 (22.2%) | 2/21 (9.5%) |
| `swap_labels` | 0/12 (0.0%) | 2/18 (11.1%) | 0/21 (0.0%) |
| `cross_case_replace` | 1/12 (8.3%) | 1/18 (5.6%) | 0/21 (0.0%) |

**Cause probable identifiée, pas encore corrigée — un biais de génération des mutants, pas
(nécessairement) une faiblesse du mécanisme de vérification.** Inspection du détail brut
(`results/perturbation_study/results.json`) : les détections ne sont pas dispersées
uniformément entre les 7 fichiers de base, elles se concentrent sur 1 à 2 fichiers par modèle
(`V_k09/2` pour `llama-3.1-8b`, `M_g01/10` et `V_k09/2` pour `gpt-oss-20b`) — zéro détection sur
les autres bases, quel que soit l'opérateur. Si le mécanisme était simplement peu sensible, une
dispersion plus uniforme serait attendue.

Cause structurelle probable : `mutate_bpmn.py` choisit l'activité mutée **au hasard parmi
toutes les activités nommées du BPMN**, sans vérifier si l'état qu'elle ancre est effectivement
référencé comme terme ou cible d'un guard dans `reference_graph`. Une part inconnue des mutants
touche donc des activités dont l'état n'est jamais vérifié par aucune précondition — mutations
structurellement indétectables, indépendamment de la qualité du mécanisme d'alignement. Le
rappel mesuré ici mélange donc deux quantités distinctes : la vraie capacité de détection, et
la proportion de mutations tombées sur une zone non couverte du graphe.

**Non corrigé dans ce document** : nécessite de cibler la génération de mutants sur les
activités dont l'état apparaît dans `reference_graph` (information disponible dans le run v2
mais non consommée par `mutate_bpmn.py`). Complication de conception non résolue : les mutants
sont aujourd'hui partagés entre les 3 modèles (un seul jeu, indépendant du modèle, car la
mutation ne touche que le BPMN) ; cibler par pertinence de guard rend la mutation dépendante du
`U`/`G` de chaque modèle — décision à prendre entre (a) un jeu de mutants par modèle (rappel
propre, 3× plus de mutants) ou (b) cibler l'union des activités pertinentes sur les 3 modèles
(un seul jeu, rappel encore partiellement dilué pour les modèles dont le guard ne couvre pas
l'activité choisie).

---

## 9. Portée de la validation

Un sous-ensemble de 7 descriptions sur 24, sélectionné par stratification délibérée, trois
modèles. Suffisant pour confirmer que les mécanismes de démotion et de filtre start/end
produisent l'effet structurel attendu (preuve directe, comptage exhaustif sur les cas présents).
Insuffisant pour toute affirmation de taux généralisable (Zenodo, taux `VIOLATED` à l'échelle,
fréquence du mode d'échec par négation) — ces questions attendent le run complet sur les 24
descriptions, qui reste la prochaine étape avant l'étude de perturbation et les baselines.