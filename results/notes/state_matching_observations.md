# Observations — State Matching Module

Notes de conception et de test pour `state_matching_node.py`, le module qui apparie chaque
activité BPMN (côté processus observé, branche droite de l'architecture) à un état de `U`
(côté texte d'intention, déjà produit par Protocole 3) — la première brique concrète de
Mapping(2).

---

## 1. Portée du module

Entrée : des triplets SPO `(activité1, "follows", activité2)`, produits par la transformation
BPMN → DFG → SPO en amont. Sortie : pour chaque activité unique, le candidat `entity.state` de
`U` le plus proche par similarité cosinus, plus un graphe côté processus reconstruit
automatiquement en réutilisant l'ordre des activités déjà porté par les triplets SPO (chaque
arête `follows` devient une arête entre les deux états matchés, sans étape supplémentaire).

Ce que ce module ne fait pas : il ne décompose jamais une activité en `(entité, état)`
synthétiques avant comparaison (approche "extraction" explicitement écartée, voir section 2.2).
Il ne filtre jamais un candidat avant l'embedding (aucune couche déterministe en amont).

---

## 2. Deux approches envisagées pour Mapping(2)

### 2.1 Approche retenue : appariement direct par embedding, nœud par nœud

Chaque texte d'activité est embeddé tel quel et comparé par cosinus à chaque candidat
`entity.state` de `U` (rendu en texte : `job_application confirmed`). Le meilleur score est
retenu. Aucun seuil de décision, aucune classification `AMBIGUOUS`/`NO_MATCH` — le meilleur
candidat est toujours renvoyé, y compris quand aucun n'est réellement bon (cf. section 6.1).

### 2.2 Approche écartée pour l'instant : alignement structurel DFG ↔ Graphe d'état

Formulation explorée mais non implémentée : poser le problème comme du graph matching inexact
(type QAP — quadratic assignment problem), optimisant conjointement un coût de correspondance
nœud à nœud (la similarité embedding ci-dessus) et un coût de cohérence structurelle (les
voisins de deux nœuds appariés doivent eux-mêmes être appariables de façon cohérente).

Argument en sa faveur, non réfuté : c'est la seule des deux approches qui pourrait corriger le
mode d'échec documenté par ReGA (similarité sémantique proche du hasard sur des paires
structurellement symétriques, cf. related-work) — un appariement nœud par nœud isolé reste
aveugle à ce problème par construction.

Argument contre, décisif pour le report : complexité algorithmique et d'implémentation d'un
ordre de grandeur supérieure (NP-difficile en général, formulation à trancher entre
activités↔nœuds de `G` ou activités↔arêtes de `G` — ontologiquement plus défendable mais plus
lourd à poser), sans gain démontré sur les échecs réellement observés jusqu'ici (`Unemployed`,
`Rate company`, cf. section 6.1) qui sont des candidats *manquants*, pas des candidats
*ambigus entre eux* — donc pas le mode d'échec que l'alignement structurel corrigerait.

**Statut : mis de côté, pas invalidé.** Le `gateway_context` (type de gateway BPMN traversé :
XOR/AND/boucle) a été délibérément conservé séparément du triplet SPO (`relation` reste
`"follows"` générique) précisément pour ne pas fermer cette porte — c'est l'ingrédient qui
manquerait pour reconstruire un coût d'arête comparable côté DFG si cette approche est reprise
plus tard, typiquement au niveau du bloc Micro plutôt qu'ici.

---

## 3. Backend d'embedding : deux modèles testés, un seul avec preuve positive

### 3.1 Échec `baai/bge-m3` — jamais validé, pas juste jamais préféré

Trois tentatives, trois échecs, aucune n'a produit de résultat exploitable :

1. Premier essai (avant tout traitement défensif) : `500 Internal Server Error` générique dès
   le premier appel multi-textes.
2. Après découpage en lots de 8 + retry : échec identique et systématique (3 tentatives) sur un
   batch précis, contenant les labels bruts du BPMN `job_application_v2` (retours à la ligne
   et espaces parasites jamais nettoyés avant l'appel, ex. `"company rates\n application "`).
   Le retry ne changeait rien car l'échec n'était pas transitoire.
3. Le fix de nettoyage d'espacement (`" ".join(text.split())` avant embedding) a été ajouté à
   ce moment, mais **jamais retesté avec `bge-m3`** — le travail a basculé directement sur le
   test d'un second modèle avant validation.

**Conclusion actée : `bge-m3` n'a aucune donnée positive ni négative exploitable.** Toute
affirmation antérieure du type "bge-m3 fonctionne mieux" formulée pendant ce travail était une
erreur d'attribution (confusion avec les résultats obtenus plus tard sur l'autre modèle) — pas
un résultat réel. À ne pas réutiliser comme référence.

### 3.2 Bascule vers `nvidia/llama-nemotron-embed-1b-v2`

Modèle testé en second, à la suite de l'échec non résolu de `bge-m3`. Fonctionne
mécaniquement (pas d'erreur serveur une fois le nettoyage d'espacement en place), mais introduit
un problème de conception distinct : voir section 4.

---

## 4. `query` vs `passage` — pourquoi tout est en `query`

`llama-nemotron-embed-1b-v2` est un modèle de retrieval asymétrique (famille E5/BGE), entraîné
sur des paires **question courte / paragraphe long** (type MS MARCO), pas un embedding
symétrique façon BERT/Sentence-BERT. Il expose un paramètre `input_type` (`"query"` ou
`"passage"`) qui change la représentation produite pour un même texte — pas une étiquette
administrative, un traitement différent selon la valeur.

**Premier réglage testé, suivant l'usage documenté par NVIDIA** : activités en `query`,
candidats `U` en `passage` (le candidat étant traité comme "ce qui contient l'information").
Résultat : dégradation nette et mesurable par rapport à un test antérieur tout-`query`, sur les
mêmes activités :

| Activité | tout-`query` | `query`/`passage` |
|---|---|---|
| `Application process finished` | `process.ended` (0.59, correct) | `job_application.confirmed` (0.28, faux) |
| `Send job application` | `job_application.reported` (0.70, plausible) | `job_offer.sent` (0.38, faux) |
| `Negotiate job interview` | `job_interview.negotiated` (0.85) | même match (0.40) — score écrasé |

Explication retenue : le mécanisme `query`/`passage` encode une asymétrie de longueur et de
rôle (question vs document) apprise à l'entraînement. Nos deux côtés (activité BPMN, candidat
`entity.state`) sont tous les deux courts (2 à 4 mots) et de même nature — ni l'un ni l'autre
n'est un "document long". Forcer un côté en `passage` le fait passer par une transformation
calibrée pour un cas qui ne correspond pas au nôtre, dégradant la représentation.

**Décision : `input_type="query"` des deux côtés.** Confirmé stable sur deux exécutions
strictement identiques (résultats reproduits à la décimale près, cf. section 5) — écartant une
hypothèse initiale, erronée, de non-déterminisme côté serveur : l'écart observé entre deux runs
successifs était entièrement dû à ce changement de code (`passage` introduit puis retiré), pas
à une instabilité de l'API.

**Écart assumé à documenter dans le papier** : cet usage n'est pas celui documenté par NVIDIA
pour ce modèle. Le choix est justifié empiriquement (résultats mesurés sur `job_application`),
pas par la documentation du modèle — à formuler ainsi explicitement, pas à présenter comme un
usage standard.

---

## 5. Résultats empiriques — `job_application` (1 cas sur 24, 2 modèles BPMN sur 8-11)

Tout-`query`, `nvidia/llama-nemotron-embed-1b-v2`, `run_17` (Protocole 3), les deux variantes
de prompt (`zero_shot`, `one_shot`) :

- **Matchs corrects à haute confiance** (>0.7) : `Negotiate job interview` → `job_interview.negotiated`
  (0.85), `Report job application` → `job_applications.reported` (0.72-0.82),
  `Receive potential job offers`/`Recieve potential job offer` → `*.received` (0.75-0.87),
  `Probation phase finished` → `probation_phase.completed` (0.89 en `zero_shot`, où l'état
  existe — cf. section 6.2 pour le cas `one_shot`).
- **Reproductibilité vérifiée** : deux exécutions successives, code identique, résultats
  identiques à la décimale près sur les 25 activités testées (`1_1` + `1_10`, deux prompts).

---

## 6. Limites connues, non résolues — à ne pas laisser silencieuses

### 6.1 Candidats manquants dans `U` — problème structurel, pas un défaut du matching

`'Unemployed'` (0.20-0.29, toujours faux, quel que soit le prompt) et `'Rate company'`
(0.13-0.27, toujours faux) n'ont **aucun candidat correct disponible** dans `U` :
- Aucun état "sans emploi"/"recherche d'emploi" n'a été capturé par Protocole 3.
- `company` a été explicitement filtré comme `ACTOR` en amont (validation lexicale WordNet),
  donc aucun `entity.state` plausible n'existe pour une activité qui porte sur cette entité.

Ces deux échecs sont stables à travers les deux backends d'embedding testés et les deux
variantes de prompt — confirmation qu'aucun choix de modèle d'embedding ne les corrigera.
Le fix, s'il est fait, appartient à Protocole 3 (état space), pas à ce module.

### 6.2 Sensibilité de phrasé sur des candidats quasi identiques

`'Write job application'` matche `job_applications.rated` (0.60) en `zero_shot` mais
`job_application.reported` (0.66) en `one_shot` — alors que les state spaces ne diffèrent que
par le pluriel de l'entité (`job_applications` vs `job_application`, mêmes trois états sinon).
Un simple pluriel/singulier suffit à faire basculer le meilleur candidat retenu. À documenter
comme limite de robustesse, pas à masquer si le papier revendique une méthode stable.

De même, `Probation phase finished` matche correctement `probation_phase.completed` (0.89) en
`zero_shot` (où cet état existe) mais est forcé vers `probation_phase.entered` (0.80, faux, avec
un score trompeusement élevé) en `one_shot`, où `U` ne contient que `["entered"]` pour cette
entité — encore un cas de candidat manquant plutôt qu'un défaut du matching lui-même, mais qui
illustre que la qualité du matching dépend directement de la richesse de `U` en amont.

### 6.3 Aucun signal de confiance exploitable en aval

Décision actée explicitement : pas de seuil, pas de statut `AMBIGUOUS`/`NO_MATCH`. Conséquence
directe : un match faux à 0.13-0.29 (section 6.1) sort dans le même format qu'un match correct à
0.85-0.89, sans rien pour les distinguer autrement qu'en lisant chaque score individuellement.
Tolérable tant que ce module est testé isolément ; **à reconsidérer avant que sa sortie
n'alimente le bloc Micro**, où un mauvais match pèserait autant qu'un bon dans tout ce qui suit.

---

## 7. Portée de la validation — à ne pas sur-généraliser

Tout ce qui précède repose sur **un seul cas (`job_application`) et deux modèles BPMN sur les
8-11 disponibles pour ce cas dans le corpus Zenodo**, sur 24 descriptions au total. C'est une
preuve de faisabilité sur un échantillon, pas une validation à l'échelle du corpus. Les deux
limites de la section 6 sont chacune rattachées à une cause identifiée (couverture de `U`,
sensibilité de phrasé) plutôt qu'à un échec générique du matching — mais rien ne garantit que ce
diagnostic se généralise sans tester d'autres cas.