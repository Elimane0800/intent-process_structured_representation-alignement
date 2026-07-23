# Observations — Alignment Module

Notes de conception et de test pour `alignment_node.py`, le bloc Micro (partiel) de
l'architecture : vérifie que chaque precondition de `G` (texte, via Protocole 3 + Pre +
graph_construction) est positionnée correctement dans `G_process` (BPMN, via DFG → SPO →
state_matching), sans jamais exiger que les deux graphes soient égaux.

---

## 1. Principe retenu

Une description textuelle peut sous-spécifier un processus — `G_process` a le droit d'être
plus riche que `G`. L'alignement ne vérifie donc jamais l'égalité des deux graphes, seulement
une observation positionnelle : pour chaque précondition non triviale de `G`, les états
qu'elle requiert sont-ils atteignables, via un **chemin** (pas une arête directe), avant l'état
cible, dans `G_process` ?

**Chemin plutôt qu'arête directe** : décision explicite, pas un choix par défaut. Une
précondition textuelle du type "A doit tenir avant B" n'affirme jamais que rien d'autre ne peut
se passer entre les deux — exiger une arête directe imposerait une contrainte que le texte
source n'a jamais posée.

**Trois issues par terme, jamais réduites à deux** :
- `SATISFIED` : chemin trouvé dans `G_process`.
- `VIOLATED` : les deux états existent dans `G_process`, aucun chemin entre eux.
- `UNRESOLVABLE` : un des deux états n'a aucun correspondant dans `G_process`.

`UNRESOLVABLE` reste délibérément ambigu entre deux causes distinctes, jamais démêlées
automatiquement :
1. Sous-spécification légitime du texte (le processus fait plus que ce que le texte décrit —
   cas normal, pas une faille).
2. Échec d'extraction en amont (Protocole 3 n'a pas capturé un concept que le texte porte
   réellement — une vraie faille, juste située ailleurs dans le pipeline).

Un signal de confiance permettant de distinguer les deux automatiquement a été envisagé puis
explicitement écarté (voir section 5) — pas encore nécessaire au stade actuel.

---

## 2. Source de `G` : `graph_construction`, pas `precondition_extraction_with_retry` directement

### 2.1 Le problème de l'aplatissement AND/OR

`graph_node.py::build_edges` aplatissait chaque précondition en arêtes individuelles sans
encoder l'opérateur (`AND` vs `OR`) — cohérent avec la définition formelle de `G` posée tôt
dans le projet (`E = {(u_i, u) : u_i ∈ Pre(u)}`), mais devenu insuffisant une fois `OR`
introduit dans `Pre`. Sans l'opérateur, impossible de savoir si une cible avec plusieurs
préconditions entrantes exige que *toutes* tiennent ou qu'*une seule* suffise.

**Fix appliqué** : chaque arête de `graph_construction` porte désormais `"operator"`
(`"AND"`/`"OR"`). La grammaire de `Pre` interdisant tout mélange AND/OR pour une même cible,
toutes les arêtes pointant vers une même cible partagent nécessairement le même opérateur — un
seul groupe par cible, donc regrouper par `"to"` suffit à reconstruire `Pre(u)` exactement,
sans retourner au `validated` brut.

**Fix additionnel, même logique** : chaque arête porte aussi désormais `"quote"` (la citation
qui a justifié la précondition dans `Pre`) — nécessaire pour que `report_node.py` puisse
produire des explications ancrées textuellement. Garanti non-`None` par construction :
`_validate_one()` n'accepte jamais une précondition sans citation vérifiée.

### 2.2 Bug trouvé : incompatibilité de vocabulaire entre `zero_shot`/`zero_shot`

`precondition_node_with_retry.py` charge `STATE_SPACES = load_state_spaces(run_filename=
"run_17.json")` **une seule fois**, avec le `prompt_name` par défaut de la fonction
(`"one_shot"`). Les clés `PROMPTS = {"zero_shot": ..., "one_shot": ..., ...}` de ce fichier
désignent le prompt utilisé pour **générer les préconditions**, pas celui utilisé pour générer
`U` — `U` est toujours celui de `run_17` en `one_shot` (vocabulaire singulier
`job_application`), quelle que soit la clé du `run_N.json` résultant.

`state_matching_node.py`, lui, charge bien un `U` différent par prompt (son bucket `zero_shot`
utilise le vocabulaire pluriel `job_applications`, chargé explicitement avec
`prompt_name="zero_shot"`). Comparer `G` (toujours vocabulaire singulier) au bucket `zero_shot`
du matching (vocabulaire pluriel) produisait donc un résultat **identique et vide de sens**
pour n'importe quel modèle BPMN testé — `UNRESOLVABLE` partout, `0/0/7` que le modèle soit
noté 5/5 ou 0/5 par ailleurs, sans jamais rien mesurer.

**Fix** : le bucket `zero_shot` du matching est explicitement ignoré (`skipping bucket(s)
['zero_shot'] -- vocabulary mismatch with G`), avec message clair plutôt qu'un résultat
silencieusement faux. Seul le bucket `one_shot` est comparable à `G` en l'état.

### 2.3 Garantie de correspondance run_N ↔ run_17

`precondition_node_with_retry.py` pin `run_filename="run_17.json"` dans son propre `__main__` —
donc *tout* `run_N.json` de `precondition_extraction_with_retry` (et par extension de
`graph_construction`, qui en mirrore le nom 1:1) a été construit avec le `U` de `run_17`, par
construction du code, pas par supposition. Un sanity check mécanique reste en place (comparaison
des nœuds de `G` contre le `state_space` réel de `run_17`) au cas où ce hardcode changerait sans
que ce document soit mis à jour.

---

## 3. Score de conformité et poids : décidés absents, pas juste reportés

### 3.1 Score de conformité

Décision : **pas de score agrégé, ni maintenant ni dans l'architecture cible.** Trois raisons,
pas une simple prudence :
1. Des checks structurellement non-discriminants (`job_offer.sent`/`received`, cf. section 4)
   pèseraient dans n'importe quelle moyenne, indépendamment de la qualité réelle du modèle testé.
2. `VIOLATED` n'a jamais été observé sur aucun run réel (cf. section 4) — un score le
   refléterait avec la même confiance apparente qu'un check correctement calibré.
3. Deux usages distincts identifiés pour un éventuel score, avec des besoins différents :
   - **Usage "audit d'un BPMN contre son intention"** (l'objectif du papier) : n'a pas besoin
     d'un score, seulement de savoir *quoi* est fidèle et *quoi* ne l'est pas — le rapport
     détaillé (section 5) sert directement cet usage, un score y retire de l'information plutôt
     que d'en ajouter.
   - **Usage "comparer plusieurs modèles / valider la méthode contre une note experte"** (ce
     qu'on a fait à la main avec les notes Zenodo, section 4.2) : un scalaire aurait un usage
     réel ici (corrélation Spearman/Pearson), mais seulement comme *statistique d'évaluation du
     papier*, jamais comme sortie du système — calculable à la main sur les résultats bruts,
     sans coder de fonction d'agrégation officielle.

L'architecture initiale prévoyait un score, mais toujours accompagné d'un rapport d'audit — signe
que le score n'a jamais été le vrai livrable, seulement un résumé optionnel.

### 3.2 Poids structurel (`weight`, centralité de degré)

Jamais utilisé par l'alignement ni par le rapport — n'a de rôle que dans les visualisations PNG
déjà produites par `graph_node.py::build_figure` (taille/couleur des nœuds). Toujours calculé,
jamais retiré du code : le retirer casserait les visualisations existantes pour un gain nul ici.

### 3.3 Propagation des échecs

Prévue dans l'architecture initiale comme étape séparée avant l'agrégation pondérée. Jamais
codée comme mécanisme à part dans `alignment_node.py` : à cette échelle, les causes racines
d'échecs en cascade sont directement repérables dans la sortie de `check_alignment()` par simple
recherche de clé (un terme en échec qui est lui-même une cible en échec ailleurs dans le même
dict) — pas besoin d'un mécanisme dédié. Formalisé plus tard, mais dans `report_node.py`
(`find_root_term`, voir `report_observations.md`), pas ici : l'alignement produit les faits
bruts, le regroupement par cause appartient à la présentation, pas à la vérification elle-même.

---

## 4. Résultats empiriques — `job_application`, 2 modèles BPMN sur 8-11 disponibles

### 4.1 Biais structurel identifié : `job_offer.sent`/`job_offer.received`

Ces deux checks ressortent `UNRESOLVABLE` de façon identique sur les deux modèles testés
(`1_1`, `1_10`) — aucune activité, dans aucun des deux BPMN, ne modélise l'envoi d'une offre
côté entreprise (le process est vu du point de vue du candidat, qui *reçoit* des offres, n'en
envoie jamais). Ces deux checks sont donc structurellement invivables pour n'importe quel
modèle de ce corpus, indépendamment de sa qualité — à exclure d'une évaluation comparative, pas
un signal de qualité du modèle testé.

### 4.2 Comparaison contre la note experte Zenodo (0-5)

Une fois les deux checks non-discriminants retirés (5 checks réellement discriminants sur 7) :

| | `job_application.rated` | `job_application.reported` | `job_interview.negotiated` | `probation_phase.entered` | `process.ended` | **Score** | **Zenodo** |
|---|---|---|---|---|---|---|---|
| `1_1` | SAT | SAT | SAT | SAT | SAT | **5/5** | 5/5 |
| `1_10` | SAT | UNRES | UNRES | SAT | UNRES | **2/5** | 0/5 |

`1_1` tombe exactement sur la note attendue une fois le biais retiré — pas juste "dans le bon
sens", une correspondance exacte, sur un seul cas. `1_10` ne tombe pas au plancher attendu (2/5
au lieu de 0/5) — deux lectures possibles, non tranchées : le modèle peut être globalement
mauvais (jugé par Zenodo sur d'autres critères que la fidélité aux préconditions testées) sans
échouer sur *tout* ; ou l'alignement reste trop indulgent quelque part.

**Signal secondaire, non validé statistiquement (2 points de données)** : le *taux* de
`UNRESOLVABLE` (indépendamment de `VIOLATED`, qui ne s'est jamais déclenché) semble corréler
avec la note Zenodo — 2/8 checks au niveau terme pour `1_1`, 6/8 pour `1_10`. Piste à tester sur
davantage de modèles avant d'en tirer une conclusion, pas encore une preuve.

### 4.3 `VIOLATED` jamais observé — point ouvert, non résolu

Sur tous les runs réels effectués jusqu'ici, `VIOLATED` n'est jamais apparu une seule fois,
quel que soit le modèle BPMN ou le prompt testé — chaque échec est soit `SATISFIED`, soit
`UNRESOLVABLE`. Deux lectures possibles, non tranchées :
1. Réaliste pour des graphes de cette taille (peu d'occasions d'un vrai désordre séquentiel).
2. Le check `path_exists` est trop permissif quelque part.

**Test de contrôle proposé, jamais exécuté** : construire un cas où une arête de `G_process`
est délibérément inversée, confirmer que `check_alignment` détecte bien un `VIOLATED` dans ce
cas construit. À faire avant de généraliser les résultats de la section 4.2 à d'autres cas.

---

## 5. Portée de la validation

Repose sur **un seul cas (`job_application`), deux modèles BPMN sur 8-11 disponibles pour ce
cas dans le corpus Zenodo, sur 24 descriptions au total**. Preuve de faisabilité sur un
échantillon, pas une validation à l'échelle du corpus. Le test de contrôle sur `VIOLATED`
(section 4.3) devrait être une priorité avant toute extension à d'autres cas.