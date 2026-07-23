# Observations — Pipeline complet (agent/graph.py)

Notes de conception et de test pour l'orchestration LangGraph du pipeline entier (`agent/graph.py`,
`agent/state.py`, `agent/config.py`) et pour les premiers runs bout-en-bout en conditions
réelles, sur plusieurs modèles. Contrairement aux notes précédentes (par module), celle-ci
synthétise ce que l'exécution du pipeline complet révèle — des propriétés qui n'étaient pas
visibles en testant chaque node isolément.

---

## 1. Architecture d'orchestration

### 1.1 Topologie

```
START -> state_space -> precondition -> graph_construction --\
              |                                                \
              \-----------------------------------------------> alignment -> report -> END
START -> [spo_triples, fourni a l'invocation] --------------> state_matching --/
```

`spo_triples` est une **entrée** du graphe, jamais calculée par un node — chargée depuis un JSON
déjà produit (`tests_bloc2/*_spo.json`), pas de node BPMN en direct dans ce pipeline à ce stade.

Deux profondeurs différentes convergent vers `alignment` : `graph_construction` (3 sauts depuis
`START`) et `state_matching` (2 sauts). C'est la seule vraie jointure asymétrique du graphe.

### 1.2 Bug de synchronisation trouvé et corrigé : `defer` empilé

Un simple `add_edge` en fan-in déclenche le nœud récepteur dès que son premier prédécesseur
termine, pas quand tous ont terminé, si les deux branches n'ont pas la même profondeur —
confirmé par test direct (`langgraph` 1.2.9). `defer=True` corrige ce cas pour **une** jointure
asymétrique isolée. Mais empiler deux `defer=True` consécutifs (à la fois sur `state_matching`
et `alignment`) réintroduit le même bug : le second se déclenche avant que le premier ait fini —
confirmé par test direct sur une reproduction minimale de la topologie réelle.

**Fix retenu** : `defer=True` uniquement sur `alignment` (la seule jointure réellement
asymétrique). `state_matching`, dont les deux prédécesseurs (`state_space`, `spo_triples`) sont
à la même profondeur, n'en a jamais eu besoin — jointure symétrique, stable sans `defer` dès le
premier test.

### 1.3 Bug de concurrence trouvé et corrigé : écriture parallèle sur `errors`

Deux nodes exécutés dans le même superstep (ex. `precondition` et `state_matching`, tous deux
déclenchés après `state_space`) qui échouent simultanément lèvent `InvalidUpdateError` s'ils
écrivent tous les deux dans le même champ `errors` sans réducteur explicite — confirmé en
conditions réelles, pas seulement en test synthétique.

**Fix** : `errors: NotRequired[Annotated[list[str], operator.add]]` dans `PipelineState` — seul
champ de tout l'état qui a besoin d'un réducteur, puisque c'est le seul que plusieurs nodes
parallèles peuvent écrire simultanément. Chaque wrapper de node renvoie uniquement sa propre
nouvelle erreur (`{"errors": [f"{node_name}: {e}"]}`), jamais l'historique complet — le
réducteur s'occupe de la fusion, le renvoyer soi-même produirait des doublons à chaque fusion.

### 1.4 Confinement d'erreur en cascade, vérifié bout-en-bout

Chaque wrapper de node capture ses exceptions dans `errors` plutôt que de faire planter tout le
run. Vérifié sur un vrai run raté (`llama-3.1-8b`/1_1, réponse LLM tronquée par `max_tokens`) :
`precondition` échoue, puis `graph_construction`, `alignment`, `report` échouent chacun
proprement avec une clé manquante explicite dans `errors` — jamais un crash, jamais un résultat
silencieusement faux.

---

## 2. Configuration centralisée (`agent/config.py`)

Catalogue de modèles avec leurs paramètres (`max_tokens`, `temperature`), une liste `MODELS_TO_RUN`
que les scripts d'orchestration parcourent. `base_llm.py::get_llm()` accepte soit une clé de ce
catalogue soit un nom de modèle brut (compatibilité avec le code existant qui appelle
`get_llm(temperature=0, model="meta/llama-3.1-8b-instruct")` directement) — vérifié que les deux
formes résolvent correctement `max_tokens`/`temperature` par défaut.

`report_model` reste indépendant de `MODELS_TO_RUN` — le rédacteur du rapport n'est pas
nécessairement le modèle "sous test", décision actée dans `report_observations.md`.

---

## 3. Résultats empiriques cross-modèles — job_application, 1_1 et 1_10

Trois modèles testés en conditions réelles sur le pipeline complet : `openai/gpt-oss-120b`
(défaut), `meta/llama-3.1-8b-instruct`, `mistralai/mistral-nemotron`.

### 3.1 `llama-3.1-8b` : échec quasi total, deux modes distincts

- **Run 1** : 100% des clés de précondition malformées sur les deux cas (`1_1`, `1_10`) — bare
  entity names ou expressions `AND`/`OR` entières utilisées comme clé JSON.
- **Run 2** (`1_1`) : réponse JSON qui boucle sur les mêmes paires clé-valeur répétées jusqu'à
  couper au `max_tokens` (8192) sans jamais fermer le JSON — `json.loads` échoue purement et
  simplement, aucune précondition récupérable.
- **Run 3** (`1_10`) : premier cycle réel en conditions de production — une boucle de 9 nœuds
  (`job_application.continued → ... → job_application.permanent → job_application.continued`),
  jusque-là seulement documentée comme cas de test manuel isolé (`precondition_observation.md`,
  section 13-14). `detect_cycles` fonctionne correctement dessus, l'alignement continue de
  tourner — mais confirme empiriquement que la Proposition 1 de la note related-work ("bien
  formé sous hypothèse (iii)") reste une hypothèse non garantie, pas un risque théorique.
- **Non-reproductibilité à `temperature=0`** : deux runs sur `1_10`, mêmes paramètres exacts,
  `state_space` complètement différent (6 entités séparées vs 1 entité fourre-tout à 12 états,
  avec une duplication interne — `job_application.rated_by_applicant` apparaît deux fois dans
  `candidates`). `temperature=0` ne garantit pas la stabilité de granularité d'un run à l'autre
  pour ce modèle — implication directe pour toute comparaison de runs entre eux.

### 3.2 `mistral-nemotron` : chaîne complète propre, deux fois sur `1_1`

`0 écart, 0 non-vérifiable, 5 conforme` — première fois que la chaîne complète (`U → Pre → G` →
matching → alignement) produit un résultat entièrement exploitable sans aucun trou, reproduit
une seconde fois avec ce modèle sur `1_1`. Sur `1_10` : `2 non-vérifiable, 2 conforme` —
toujours partiel, jamais parfait sur le BPMN noté 0/5 par Zenodo.

### 3.3 Signal Zenodo, confirmé une quatrième fois, indépendamment

Le taux de `non_verifiable` suit la note Zenodo du modèle BPMN testé, sur deux modèles LLM
différents (`gpt-oss-120b` informellement plus tôt, `mistral-nemotron` deux fois) — plus une
coïncidence isolée à ce stade, un pattern reproductible sur les données disponibles.

---

## 4. Bugs de robustesse trouvés en conditions réelles (pas en test synthétique)

- **Validation du `target`** (`_MALFORMED_TARGET_PREFIX`) : conçue après avoir observé 100% de
  clés malformées sur `llama-3.1-8b`. Confirmée fonctionner à 100% de détection sur les runs
  suivants.
- **Garde anti-retry sur cible malformée** : nécessaire pour éviter de gaspiller des appels LLM
  sur une correction structurellement impossible (retenter une précondition pour une clé qui
  n'a jamais été une vraie cible). A nécessité plusieurs itérations avant d'être effectivement
  appliqué côté utilisateur — la fonction complète a dû être redonnée en remplacement intégral
  plutôt qu'en patch partiel, plus fiable.
- **Citation traduite ou fabriquée dans le rapport** : règle "jamais de traduction" efficace à
  5/6 cas observés, pas 6/6 — un cas a contourné la règle en produisant une paraphrase française
  entre guillemets plutôt qu'en traduisant directement la vraie citation. Confirme qu'une règle
  de prompt oriente le comportement sans le garantir à 100%, cohérent avec toutes les
  observations précédentes sur ce point dans ce projet.

---

## 5. Test de contrôle : calibrage du mécanisme `VIOLATED`

Point resté ouvert sur plusieurs tours : `VIOLATED` ne s'était jamais déclenché sur aucun run
réel, toutes conditions confondues — ambigu entre deux hypothèses non tranchées (graphes trop
petits pour qu'un vrai désordre survienne, ou `path_exists` trop permissif). Publier un
mécanisme à trois voies dont une n'a jamais été vérifiée aurait été une faille méthodologique,
pas une nuance à mentionner en passant.

**Test 1 — cas construit minimal.** Deux états matchés dans le process, mais reliés dans le sens
inverse de ce que `Pre` exige (aucun chemin possible dans le bon sens). Résultat :
`VIOLATED`, avec la raison correcte (`"both states matched, no path in process"`) — le mécanisme
fonctionne sur un cas jouet, mais ça ne suffit pas à trancher entre les deux hypothèses.

**Test 2 — décisif : casser une vraie arête du graphe `mistral-nemotron`/`1_1`.** Repris le
`reference_graph` et le `process_graph` réels du run qui avait produit `5 conforme, 0 écart`, et
retiré une seule arête réelle (`job_offer.negotiated → probation_phase.entered`). Résultat :
deux cibles basculent en `VIOLATED`, pas une — `probation_phase.entered` (directement privée de
son seul chemin) et `job_application.rated` (en cascade, son unique chemin passait par la cible
maintenant coupée) — tandis que les trois autres cibles non affectées (`job_offer.sent`,
`job_offer.negotiated`, `job.permanent`) restent correctement `SATISFIED`.

**Conclusion, tranchée, pas juste "probablement correct"** : sur un graphe dense et réel,
retirer un seul lien suffit à révéler un désordre, avec la bonne propagation en cascade —
**hypothèse "mécanisme trop permissif" réfutée**, pas seulement non confirmée.
L'absence de `VIOLATED` observée jusqu'ici sur les runs réels s'explique par le fait que, sur les
deux modèles BPMN testés, le texte et le processus étaient effectivement bien ordonnés l'un par
rapport à l'autre — pas par une faille de détection. Le mécanisme à trois voies est maintenant
vérifié dans ses trois branches, pas seulement deux sur trois.

---

## 6. Thèse neurosymbolique — ce que le pipeline complet démontre, factuellement

### 6.1 Ce qui fonctionne, avec preuve

1. **La validation déterministe généralise sans adaptation par modèle** — le même code a
   correctement traité trois modèles très différents, avec un taux de détection de 100% sur les
   erreurs de format qu'il visait, sans aucune logique spécifique à un modèle.
2. **Le confinement d'erreur en cascade tient bout-en-bout**, jusqu'au node final, jamais un
   résultat silencieusement faux.
3. **Quand la génération est correcte, la chaîne complète produit un résultat exploitable et
   cohérent** — vérifié deux fois avec `mistral-nemotron`.
4. **Le signal de qualité corrèle avec une note externe indépendante**, de façon reproductible
   sur plusieurs modèles et plusieurs cas.
5. **Le mécanisme `VIOLATED` détecte correctement un vrai désordre, avec la bonne cascade** —
   vérifié par calibrage actif (section 5), pas seulement absent des données observées.

### 6.2 Ce qui ne fonctionne pas, avec preuve

1. **La couche symbolique contient les dégâts, elle ne répare pas la génération** — face à un
   modèle structurellement incapable de respecter le schéma de sortie, elle produit un rejet
   quasi total, pas un résultat dégradé mais utilisable.
2. **Le retry est structurellement impuissant sur les erreurs de format**, seulement sur les
   erreurs de contenu — a dû être explicitement exclu plutôt que rendu fonctionnel.
3. **`temperature=0` ne garantit pas la reproductibilité** pour au moins un modèle testé — remet
   en question la stabilité de toute comparaison de runs pour ce modèle. Non vérifié pour les
   modèles sur lesquels repose la démonstration positive (`mistral-nemotron`, `gpt-oss-120b`) —
   reste un point ouvert, pas résolu par le test de la section 5.
4. **Une seule lacune d'extraction se propage en cascade** à travers la structure en chaîne AND
   — un trou isolé fait basculer plusieurs cibles en aval en `UNRESOLVABLE`, pas seulement la
   cible directement concernée. Confirmé aussi vrai pour `VIOLATED` (section 5, test 2).
5. **La couche générative finale reste le maillon le moins contrôlable**, malgré des contraintes
   explicites et un ancrage textuel fourni.

### 6.3 Thèse retenue pour le papier

La valeur du pipeline neurosymbolique est **asymétrique**, pas générale : il excelle à confiner
et diagnostiquer l'échec de génération (jamais silencieux, toujours tracé jusqu'à sa cause
racine, mécanisme de détection calibré et vérifié dans ses trois branches), mais n'améliore pas
la qualité de la génération sous-jacente elle-même. Un modèle incapable de respecter le schéma
de sortie traverse la couche symbolique en produisant strictement zéro résultat exploitable,
indépendamment de la quantité de validation en aval. Une formulation plus nuancée et plus
défendable que "le neurosymbolique corrige les faiblesses du LLM" — c'est "le neurosymbolique
révèle et contient ces faiblesses, sans les corriger."

**Portée de la thèse de fidélité, précisée par le test de la section 5** : la méthode ne se
limite plus à vérifier la *couverture* des états requis (SATISFIED/UNRESOLVABLE) — elle vérifie
aussi la *fidélité de l'ordre causal* (VIOLATED), avec un mécanisme dont le calibrage a été
vérifié activement, pas seulement supposé. La thèse complète ("fidélité entre texte et modèle
vérifiable par appariement d'états") est formulable pour le papier sur cette base — à condition
de ne jamais présenter cette vérification comme allant plus loin que ce qui a été testé (un cas,
deux modèles BPMN, cf. section 7).

---

## 7. Portée de la validation

Un seul cas (`job_application`), deux modèles BPMN (`1_1`, `1_10`) sur 8-11 disponibles pour ce
cas, trois modèles LLM testés sur le pipeline complet. Les patterns observés (corrélation
Zenodo, asymétrie de la valeur neurosymbolique, non-reproductibilité de `llama-3.1-8b`, calibrage
du mécanisme `VIOLATED`) sont cohérents à travers plusieurs runs indépendants et un test de
contrôle actif, mais reposent sur un échantillon encore modeste — à traiter comme des pistes
solides et un mécanisme vérifié, pas des résultats statistiquement établis à grande échelle.