# Observations — Graph Construction Module

Notes de recherche pour le troisième module du pipeline, `graph_node.py`, qui assemble l'espace
d'états `U` (Protocole 3) et les préconditions validées (`Pre`, naïf + validation + retry) en un
graphe orienté, purement déterministe — aucun appel LLM.

---

## 1. Conception initiale

**Entrée** : un seul fichier `results/precondition_extraction_with_retry/run_N.json` — chaque
entrée contient déjà `state_space_used` (nœuds) et `validated` (préconditions, donc arêtes) côte à
côte, pas besoin de recharger `U` séparément depuis `state_space_protocol3`.

**Ce que le module produit** :
- **Nœuds** : chaque `entité.état` de `state_space_used`, à plat.
- **Arêtes** : une par terme de conjonction, pour chaque état avec une précondition non-`INITIAL`/
  `UNRESOLVED`. Typées `sequence` (terme de la même entité que la cible — progression naturelle du
  cycle de vie) ou `cross_entity_guard` (entité différente — dépendance inter-objets).
- **Poids des nœuds** : centralité de degré (in-degree + out-degree), normalisée par le degré
  maximum observé dans le graphe — calculée une fois le graphe assemblé, distincte de la confiance
  d'extraction déjà établie en amont dans `Pre`. Décision actée tôt dans la conversation : ne jamais
  fusionner "importance structurelle" et "fiabilité de l'extraction" en un seul nombre.
- **Détection de cycles** : DFS, réutilisée telle quelle du design déjà validé pour `Pre`.

**Fichier autonome** : demandé explicitement par l'utilisateur, aucune dépendance vers les fichiers
`Pre` — la logique de détection de cycles est recopiée, pas importée.

**Boucle sur tous les runs disponibles** : le `__main__` traite tous les `run_*.json` trouvés dans
`precondition_extraction_with_retry/`, pas un seul fichier fixé en dur. Sortie sauvegardée sous le
même nom de fichier dans `graph_construction/` — traçabilité 1:1 entre un run `Pre` et son graphe.

---

## 2. Bug découvert au premier run réel : nœud cible halluciné

**Symptôme** : `KeyError: 'weight'` au moment du rendu, provenant de `networkx` qui crée
silencieusement un nœud sans l'attribut `weight` attendu.

**Cause exacte** : `_validate_one()` (module `Pre`) vérifie que les **termes** d'une précondition
référencent des états valides, mais ne vérifie jamais que la **clé cible** (`target`) elle-même est
un `entity.state` réel de `state_space`. Une clé hallucinée par le LLM en amont (jamais filtrée à
la source) devient une destination d'arête ici, sans avoir jamais été ajoutée comme nœud légitime.

**Correctif appliqué, à deux niveaux** :
1. `build_edges()` prend maintenant `state_space` en paramètre et filtre explicitement toute arête
   dont la cible ou un terme n'existe pas dans les états valides — avec un message affiché
   (`[graph] skipping edge(s) to unknown target: ...`), jamais un rejet silencieux.
2. `render_graph_html()`/`build_figure()` rendu défensif (`.get(..., 0.0)`) comme second filet de
   sécurité, indépendant de la qualité des données en amont.

**Ce bug révèle un problème plus large, documenté aussi côté `Pre`** (`precondition_observation.md`,
section 14) : la validation de `Pre` ne couvre jamais le `target` lui-même, seulement ses termes de
précondition — fix identifié là-bas, pas encore appliqué à la source.

---

## 3. Visualisation : deux formats, deux obstacles techniques résolus

**Demande** : visualisation en plus du JSON de résultats, plotly plutôt que matplotlib, formats
HTML **et** PNG.

**Obstacle 1 — export PNG (`kaleido`)** : la version récente de `kaleido` (≥1.0) nécessite Chrome
installé séparément, non disponible dans le sandbox de test. Contournement initial avec
`kaleido==0.2.1` (autonome, pas de dépendance Chrome) — fonctionnait dans le sandbox, mais a révélé
un second problème une fois testé sur la machine réelle de l'utilisateur.

**Obstacle 2 — chemin contenant des espaces** : `kaleido==0.2.1` embarque un script shell interne
(`cd /chemin/vers/kaleido`) qui ne met pas le chemin entre guillemets — cassé par les espaces dans
le chemin du projet (`.../thèse CIFRE Devoteam/...`). Bug connu de cette version sur macOS, pas
propre à ce projet. **Résolu** en repassant à `kaleido>=1.0.0` (pilote Chrome directement en Python,
pas de script shell fragile), avec une étape de configuration unique côté utilisateur :
`uv run plotly_get_chrome` pour télécharger Chrome une fois.

---

## 4. Deux défauts de rendu signalés par l'utilisateur, corrigés

**Absence de direction visible.** `go.Scatter(mode="lines")` trace de simples segments, sans
flèche — alors que les données sous-jacentes sont bien orientées (`from`/`to` sur chaque arête). Le
graphe ne "ressemblait" à rien de causal malgré des données correctement orientées. **Corrigé** par
l'ajout d'annotations plotly (une flèche par arête, légèrement raccourcie pour ne pas disparaître
sous les marqueurs de nœuds, colorée selon le type d'arête).

**Poids en valeurs grossières (0/0.5/1), donnant l'impression d'une classification plutôt qu'un
calcul.** Vérifié et confirmé : c'est un vrai calcul continu (`degré/degré_max`), pas une
classification déguisée — mais sur un graphe peu dense (beaucoup d'états `INITIAL`/`UNRESOLVED`
sans arête), le degré maximum observé peut être très petit, réduisant mécaniquement le nombre de
valeurs distinctes possibles. Confirmé par un test contrôlé (graphe à 5 arêtes → poids
`0/0.333/0.667/1.0`, plus granulaire). **Transparence ajoutée** : le degré brut est maintenant
affiché dans l'info-bulle au survol (`weight=X (degree=Y)`), pour que la mécanique du calcul soit
vérifiable directement sur les données réelles, pas seulement expliquée.

---

## 5. Changement de layout : de force-directed à Directed-Follows-Graph

**Problème signalé** : `spring_layout` (attraction/répulsion, sans notion de direction) plaçait les
nœuds sans aucun rapport avec le flux causal du processus — le graphe ne "parlait" pas comme un
diagramme de processus, malgré des données orientées correctement.

**Solution retenue** : layout par génération topologique (`nx.topological_generations`), dans
l'esprit des Directed-Follows-Graphs utilisés en process mining (Disco, Celonis, PM4Py) — chaque
nœud placé selon sa profondeur causale (nombre de préconditions à remonter), gauche = début du
processus, droite = fin, avec un étalement vertical au sein de chaque génération pour éviter les
recouvrements.

**Repli explicite prévu** : si le graphe contient un cycle, `topological_generations` échoue
(nécessite un DAG) — repli automatique sur `spring_layout` dans ce cas précis, avec la limitation
assumée que cette situation ne représente alors plus un flux lisible, cohérent avec le fait qu'un
cycle signale de toute façon une incohérence à traiter en amont (cf. `precondition_observation.md`,
section 14, sur les cycles révélés par le retry à contexte étroit).

---

## 6. Questions ouvertes / prochaines étapes

- **Répercuter le fix `target not in valid_targets`** identifié côté `Pre`
  (`precondition_observation.md`, section 15) — le filtre défensif de `graph_node.py` protège déjà
  le graphe final, mais corriger à la source éviterait la pollution du rapport intermédiaire
  `validated`.
- **Tester le rendu sur un graphe contenant un cycle réel** (maintenant qu'on sait que le retry à
  contexte étroit en produit, cf. `precondition_observation.md` section 14) — vérifier que le repli
  `spring_layout` se déclenche proprement et reste lisible, pas seulement fonctionnel.
- **Support `OR` pas encore répercuté dans `graph_node.py`** — le module actuel suppose des
  conjonctions (`AND`) uniquement pour construire les arêtes ; à mettre à jour une fois le support
  `OR` stabilisé côté `Pre` (cf. `precondition_observation.md`, section 13), pour que
  `compute_node_weights`/`detect_cycles`/le rendu restent cohérents avec la distinction `AND`/`OR`.
- Vérifier si la granularité grossière des poids (section 4) reste un problème pratique une fois
  testée sur des graphes plus denses (`work_accident` avec le support `OR`, potentiellement plus
  d'arêtes qu'auparavant) — pourrait résoudre le problème de lui-même sans intervention
  supplémentaire.