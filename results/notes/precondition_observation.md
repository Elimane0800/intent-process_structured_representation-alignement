# Observations — Precondition Extraction Module (Pre)

Notes de recherche pour le second module du pipeline, `Pre`, qui prend en entrée l'espace
d'états `U` (déjà validé par le Protocole 3) et le texte source, et produit, pour chaque état,
la précondition qui légitime son atteinte.

---

## 1. Définition du problème

`Pre` répond à la question : *pour chaque état d'une entité dans `U`, quels états (de la même
entité ou d'entités différentes) doivent déjà être atteints pour qu'on puisse légitimement dire
que cet état est atteint ?* Contrairement à `U`, la tâche ne consiste pas à énumérer des
candidats déjà présents dans le texte, mais à **construire des relations** entre des éléments
déjà fixés.

**Nature de la tâche, positionnée dans la littérature IA générale (pas BPM/process mining)** :
extraction de relations causales, additive, strictement ancrée dans le texte — à l'intersection de
l'extraction de relations (NLP), des contraintes déclaratives (process mining/DECLARE, non retenu
comme cadre principal — trop activité-centrique), et de l'extraction précondition/effet
(planification symbolique, PDDL). Champ actif en IA générale sous le nom de "neurosymbolic
LLM-planning", avec des précédents à AAAI (workshops 2025) et dans les revues IEEE/ACM.

---

## 2. Décisions de formalisation, actées avant tout code

- **Additif uniquement (`AND`)** : `OR` est explicitement hors périmètre. Une précondition
  disjonctive doit être marquée `UNRESOLVED`, jamais forcée dans une conjonction. Décision motivée
  à l'origine par la complexité de propagation d'échec en aval (SAT vs parcours de graphe simple) —
  nuancée ensuite : un `OR` de conjonctions indépendantes n'exige pas de solveur SAT complet, juste
  une vérification d'existence par groupe. La décision de rester additif est maintenue malgré cette
  nuance, pour garder le scope étroit et défendable.
- **Exclusivité mutuelle héritée de `U`** : deux états de la même entité ne peuvent jamais
  apparaître ensemble dans une même précondition (contradiction logique directe).
- **Rester strictement état-centrique** : pas d'objet "activité" ou "événement" comme nœud de
  graphe, même si cela laisse une information non capturée (voir frame problem ci-dessous). Décision
  prise explicitement en écartant un format concurrent (proposé par un collègue) qui fusionnait
  activités, préconditions et jalons en un seul jugement composite par appel LLM — jugé trop proche
  des architectures ayant échoué sur `U` (jugement composite en un seul passage).
- **Frame problem reconnu comme limite explicite, non traitée à ce stade** : rien ne garantit
  aujourd'hui que les états non concernés par une transition restent stables. Un ancrage théorique a
  été identifié (Event Calculus, Kowalski & Sergot 1986 — prédicats `Initiates`/`Terminates`/
  `HoldsAt` avec axiome de persistance explicite) comme formalisme construit spécifiquement pour ce
  problème, plutôt qu'un algorithme de graphe générique. Non implémenté à ce stade — noté comme
  piste de formalisation future, pas comme composant actif.
- **Le formalisme reste un fragment restreint inspiré de GSM** (Milestones + Guards conjonctives),
  pas une implémentation complète — à formuler ainsi explicitement dans le papier, pas présenté comme
  du GSM standard.

---

## 3. Architectures envisagées avant de coder le test naïf

Plusieurs protocoles à plusieurs étages ont été conçus et discutés, avant de décider de commencer
par la méthode la plus simple possible (voir section 4) :

- **Protocole A** : identification des états de début/fin (LLM, tâche bornée) → génération
  combinatoire réduite aux états intermédiaires (en respectant l'exclusivité mutuelle) →
  classification LLM contextualisée avec ancrage textuel obligatoire → solveur de cohérence
  déterministe (absence de cycle/contradiction).
- **Protocole B** : construction incrémentale par ordre topologique, réduction transitive appliquée
  en continu pendant la construction plutôt qu'en passe séparée après coup.
- **Piste sieve (connecteurs discursifs PDTB)** testée puis explicitement écartée : vérification
  empirique sur les 3 cas de test — `maternity_leave` ne contient **aucun** connecteur `Contingency`
  (if/unless/when/after/before/once/must...), alors que `job_application` en contient plusieurs. Un
  filtre qui échoue totalement sur un tiers du corpus de test n'est pas un filtre fiable ; retiré du
  protocole plutôt que rendu conditionnel (l'activation conditionnelle par seuil aurait recréé
  exactement le type de règle ad hoc déjà écartée ailleurs).
- **Piste filtrage par similarité sémantique (vectorizer/embeddings)** proposée puis explicitement
  rejetée sur le plan théorique : la proximité sémantique de deux états (ex. `company_review` et
  `company_rating`, lexicalement proches) ne garantit aucune relation causale entre eux, et
  inversement une relation causale réelle peut relier des états lexicalement très éloignés (ex.
  `accident.fatal` → `labour_inspectorate.notified`, aucun mot partagé). Confusion entre deux axes
  indépendants (proximité de sens vs structure causale) — non retenue.
- **Décision finale** : commencer par la méthode la plus naïve possible (aucun filtre, aucun
  solveur, juste le LLM avec un prompt exhaustif) pour établir une ligne de base avant d'ajouter de
  la complexité — cohérent avec la discipline appliquée sur `U` (ne pas empiler des mécanismes sans
  preuve empirique qu'ils sont nécessaires).

---

## 4. Le prompt naïf : conception et bug corrigé

`agent/prompt/precondition_prompt.py` — structure Definitions + Rules, au même niveau
d'exhaustivité que `STATE_SPACE_PROMPT_V4`. Entrée : texte source + `state_space` déjà validé
(JSON). Sortie : pour chaque état, une valeur parmi `INITIAL`, `UNRESOLVED`, ou une conjonction
`entité.état AND entité.état...`. Contrainte stricte : ne référencer que des `entité.état` déjà
présents dans le state space fourni — aucune invention possible par construction du prompt.

**Bug découvert au premier test** : violation d'auto-référence — un état cité comme sa propre
précondition (`self_employed_person.reported <- self_employed_person.reported AND
accident.ocurred`), observé 3 fois sur le premier run (`work_accident`, zero-shot, llama-8b).
**Corrigé** par l'ajout d'une règle explicite dans `_PRECONDITION_DEFINITION` : *"A state can never
be part of its own precondition... mark it UNRESOLVED instead."* Vérifié éliminé sur l'ensemble des
runs suivants (0 violation sur 12 runs, 4 familles de modèles).

**Bug de chargement découvert en cours de route** : `load_latest_state_spaces()` prenait
initialement la première combinaison prompt/modèle trouvée dans le dernier run Protocole 3
disponible — corrigé pour cibler explicitement `one_shot` par défaut. Limite résiduelle non encore
corrigée : la fonction prend toujours "le dernier run Protocole 3 disponible au moment de
l'exécution", pas un `U` de référence fixé une fois pour toutes — ce qui a causé une comparaison
faussée entre runs de `Pre` (voir section 5).

---

## 5. Résultats empiriques

**Découverte centrale : la qualité de `Pre` est bottleneck par la qualité de `U`, pas par le
prompt de `Pre` lui-même.**

Deux `U` très différents ont circulé selon le moment d'exécution du Protocole 3 en amont :
- **`U` dégénéré** (6 entités : `work_accident`, `accident`, `risk`, `defect`, `doctor_visit`,
  `person`) — `work_accident` et `accident` portent des états quasi identiques (doublons),
  `risk: [serious, immediate, discovered]` confond des attributs qualificatifs indépendants avec un
  cycle de vie. Sur ce `U`, `Pre` produit soit un collapse quasi total en `UNRESOLVED` (12-14
  états sur 14, stable à travers llama-70b, mistral-nemotron, gpt-oss-20b), soit un échec de
  parsing pur (troncature, observé sur mistral-large-3-675b et gpt-oss-20b en zero-shot).
- **`U` propre** (`job_application` : 6-8 entités bien délimitées ; `maternity_leave` : 7 entités
  cohérentes, run `run_17`/one_shot de gpt-oss-120b identifié comme référence) — `Pre` produit des
  chaînes causales linéaires cohérentes, un usage approprié d'`INITIAL` et d'`UNRESOLVED`
  (réservé aux cas génuinement temporels comme `review.visible` après "1 an"), et — chez
  mistral-large-3-675b — de vraies conjonctions inter-entités bien formées
  (`maternity_leave.extended <- maternity_leave.taken AND extension.decided`).

**Interprétation** : le taux d'`UNRESOLVED` massif n'est pas un défaut du prompt de `Pre` — c'est
un **révélateur fiable** de la qualité de `U` en amont. Face à un `U` fait de doublons et
d'attributs déguisés en états, qui n'a structurellement aucune vraie relation causale interne à
découvrir, `Pre` refuse correctement de fabriquer des relations fictives plutôt que d'halluciner.
C'est un comportement désirable, mais qui empêche de juger `Pre` isolément tant que le `U` fourni
n'est pas contrôlé et fixé.

**Comportement stable et cohérent sur `U` propre, à travers 5 familles de modèles** (llama-70b,
mistral-nemotron, gpt-oss-20b, gpt-oss-120b, mistral-large-3-675b) — chaînes causales plausibles,
zéro violation d'auto-référence après correctif, usage disciplinée d'`UNRESOLVED`/`INITIAL`. Premier
module du pipeline où le prompt engineering seul (sans architecture de validation à plusieurs
étages) semble suffire pour obtenir le comportement recherché — contraste net avec `U`, qui a
nécessité 4 versions de définition et une couche symbolique externe avant de se stabiliser.

**Réserves avant de généraliser cette conclusion** :
1. Jamais testé sur un `U` à la fois **dense et propre** — tous les échecs viennent d'un `U`
   dégénéré, jamais d'un `U` riche mais bien construit. Le test décisif (`Pre` sur le `U` de
   référence `run_17`/one_shot pour `work_accident`) reste à faire, en fixant explicitement le `U`
   utilisé plutôt que de laisser le loader prendre "le dernier run disponible".
2. **Aucun mécanisme d'ancrage textuel dans le prompt naïf actuel** — les chaînes produites sont
   plausibles et cohérentes avec l'ordre du texte, mais rien ne garantit qu'elles sont réellement
   extraites de phrases spécifiques plutôt qu'inférées par défaut logique ("l'ordre naturel d'un tel
   processus"). Risque analogue à celui déjà documenté et corrigé sur `U` via
   `validate_grounding()` — pas encore transposé à `Pre`.

---

## 6. Protocole A implémenté et testé : échec net

Implémentation fidèle du Protocole A tel qu'acté : génération combinatoire de **tous** les
candidats pour **tous** les états (aucune étape start/end, contrairement à une première tentative
erronée qui avait mélangé les deux protocoles — corrigée après relecture). Classification LLM
**indépendante de chaque candidat** (pas de sélection unique), avec ancrage textuel obligatoire
(réutilisation de la logique de `validate_grounding` : substring exact puis fuzzy via `difflib`).
Solveur de cohérence limité à la détection de cycles (pas de notion start/end dans ce protocole).

**Volume mesuré empiriquement avant de lancer** : sur un `U` dense de type `work_accident` (8
entités, 11 états), jusqu'à **575 candidats pour un seul état**, dans un seul appel LLM (pas de
batching à ce stade, conformément à la décision de ne rien borner a priori).

**Résultat sur llama-3.1-8b** : 9 états sur 10 en échec de parsing (troncature avant la fin du
JSON attendu). Le seul état "réussi" ne l'est que parce que la réponse s'est coupée juste après
avoir traité le seul candidat marqué `valid`, avant les 283 autres — un artefact de troncature, pas
un jugement fiable. Et le contenu lui-même est causalement absurde
(`job_application.confirmed <- job.permanent AND process.ended`, un état terminal cité comme
précondition d'un état précoce). **Diagnostic confirmé par l'inspection du JSON complet** (pas
seulement la sortie console) : ce n'est pas un problème de capacité de jugement du modèle, c'est un
échec structurel — le volume de candidats empêche le modèle de terminer sa réponse avant la limite
de tokens.

---

## 7. Protocole B implémenté et testé : échec pire que le Protocole A

Conception validée avant codage : étape 0 (identification start/end par LLM, comme dans une
tentative précédente conservée pour cette partie) → batching des candidats en lots de taille fixe
(30) pour les états non-début → **contexte incrémental à deux niveaux**, explicitement demandé :
(1) cohérence intra-état — les candidats déjà validés dans un lot précédent du même état sont
montrés au lot suivant ; (2) enchaînement inter-états — les états déjà résolus (dans l'ordre du
`state_space`) deviennent des "faits établis" montrés comme contexte aux états suivants. Testé
fonctionnellement (batching + enchaînement) avec un LLM simulé avant tout appel réel, validé
correct.

**Résultat sur llama-3.1-8b, run partiel** : pire que le Protocole A, mais de façon plus insidieuse
— **JSON syntaxiquement propre**, ce qui masque le problème de fond. `job_application.confirmed`
reçoit **23 préconditions valides simultanément**, dont plusieurs conjonctions à 4-5 termes,
largement redondantes entre elles. Le solveur détecte de vrais **cycles logiques directs** entre
deux états de la même entité (`job_application.confirmed → job_application.rated →
job_application.confirmed`), et une **contradiction start/end** malgré le mécanisme de contexte
incrémental censé la prévenir (`maternity_leave.extended` déclaré END à l'étape 0, puis utilisé
comme précondition valide de `maternity_leave.taken`, un état de la même entité).

**Interprétation** : le contexte incrémental n'empêche pas le modèle de se contredire lui-même — il
continue d'halluciner des relations qui vont à l'encontre de ce qu'il a déjà établi. Demander "ce
candidat est-il plausible" (classification faible) plutôt que "quelle est LA précondition réelle"
(question forte, comme dans le prompt naïf) semble structurellement pousser vers une sur-validation
généreuse, indépendamment du batching ou du contexte fourni.

**Décision, validée explicitement par l'utilisateur** : abandon des Protocoles A et B pour `Pre`.
Aucun des deux n'a produit de résultat exploitable, et le second (plus sophistiqué) est pire que le
premier — l'ajout de mécanique n'a pas amélioré la fiabilité, il l'a dégradée. Retour à la version
naïve comme seule approche retenue.

---

## 8. Couche de validation mécanique ajoutée au prompt naïf

Le prompt naïf (section 4) ne demandait aucune justification — bon comportement observé, mais
aucune garantie mécanique que les chaînes produites sont réellement ancrées dans le texte plutôt
que plausibles par défaut. Corrigé en deux temps, sans changer l'architecture à un seul appel LLM
(qui reste la seule configuration ayant produit des résultats cohérents jusqu'ici) :

- **Prompt modifié** : chaque précondition non-`INITIAL`/`UNRESOLVED` doit désormais être
  accompagnée d'une citation verbatim. Format de sortie : `{"entity.state": {"precondition": ...,
  "quote": ...}}`.
- **`validate_preconditions()`** (nouvelle couche, aucun LLM) : quatre vérifications indépendantes
  sur la sortie brute du modèle — existence des références dans `U`, absence d'auto-référence,
  exclusivité mutuelle (deux états de la même entité dans une conjonction), ancrage textuel de la
  citation (réutilise la même logique que le Protocole A). Toute violation est **rétrogradée en
  `UNRESOLVED` avec une raison explicite** — jamais rejetée silencieusement. Détection de cycles
  appliquée au graphe final (réutilise `detect_cycles`/`build_graph` du Protocole A, sans
  duplication de code).
- Testé unitairement sur des cas construits à la main (référence inexistante, exclusivité mutuelle
  violée, auto-référence, précondition légitime) avant tout run réel — les 4 cas se comportent
  comme attendu.
- **Bug corrigé au passage** : `load_state_spaces()` accepte maintenant un `run_filename` explicite
  (fixé sur `run_17.json`), au lieu de prendre "le dernier run Protocole 3 disponible" — condition
  nécessaire pour comparer les modèles sur un `U` identique, déjà identifiée comme bug en section 5
  mais non corrigée avant ce point.

---

## 9. Résultats avec validation, 9 runs, 3 familles de modèles

**La couche de validation remplit son rôle** : elle intercepte des erreurs de fond réelles sans
jamais dégrader un résultat correct. Cas le plus net : llama-3.1-8b sur `work_accident` (un run
précis) — 17 rétrogradations sur 22 états (77%), dont 9 violations d'exclusivité mutuelle, 4
auto-références, 4 références à des états inexistants. Ce sont des erreurs que le prompt seul,
même avec la règle explicite déjà ajoutée en section 4, laisse parfois passer — la couche
mécanique les rattrape de façon fiable.

**Cycles quasi absents** (1 seul cas sur l'ensemble des 9 runs × 6 combinaisons cas/prompt) —
confirme que le jugement contextualisé unique du prompt naïf ne produit quasiment jamais de
contradictions logiques internes, contrairement aux Protocoles A/B qui en généraient
systématiquement.

**Par modèle** :
- **mistral-nemotron** : le plus stable des trois — résultats **identiques mot pour mot** sur 3
  répétitions (mêmes préconditions, mêmes citations, mêmes rétrogradations) sur `work_accident`.
  Rétrogradations limitées et bien identifiées : 3 tentatives de `OR` explicite
  (`work_accident.fatal OR work_accident.serious_injury`) correctement interceptées comme
  références invalides par la couche de validation, sans intervention supplémentaire nécessaire.
- **llama-3.1-8b** : le moins stable — le même cas (`work_accident`) varie de 3 à 17
  rétrogradations selon le run, avec apparition/disparition d'échecs de parsing. Cohérent avec les
  observations précédentes sur ce modèle (le plus faible de l'échantillon).
- **gpt-oss-20b** : `work_accident` échoue en parsing dans 5 tentatives sur 6, souvent par réponse
  vide — troncature déjà documentée comme limite propre à ce modèle sur les textes longs (cf.
  Protocole 1, section 6 du document `state_space_observation.md`), indépendante du prompt ou de la
  validation.

**Limite structurelle découverte, non détectable par la validation actuelle** : sur
mistral-nemotron (les 3 runs identiques), le graphe produit sur `work_accident` est topologiquement
pauvre — **18 états différents partagent tous la même précondition unique**
(`work_accident.considered`), formant une étoile plutôt qu'une chaîne causale. Chaque citation est
individuellement authentique (vérifiée par `validate_preconditions`), mais l'ensemble ne restitue
aucune vraie structure de dépendances entre états spécifiques. Cause probable : `U` contient un état
extrêmement générique (`work_accident.considered`, "l'accident est considéré comme tel") auquel
presque tout le texte peut légitimement se raccrocher, agissant comme un aimant causal. La
validation actuelle garantit l'authenticité d'une citation, jamais sa **pertinence relative** face à
une alternative plus spécifique et plus proche — un trou distinct de celui déjà documenté (§5,
réserve 2), plus subtil qu'une simple absence d'ancrage.

---

- **Corriger `load_latest_state_spaces()`** pour accepter un chemin de fichier `U` de référence
  fixe (`run_17`/one_shot), au lieu de prendre le dernier run Protocole 3 disponible — condition
  préalable à toute comparaison fiable entre modèles sur `Pre`.
- **Tester `Pre` sur `work_accident` avec le `U` propre fixé**, pour trancher si le prompt seul
  suffit aussi sur un cas dense, ou si l'architecture à plusieurs étages (Protocole A/B) redevient
  nécessaire à ce niveau de complexité.
- **Ajouter un mécanisme d'ancrage textuel** (citation vérifiée par état/précondition, même
  principe que `validate_grounding()` sur `U`) avant de considérer les chaînes causales produites
  comme fiables, pas seulement plausibles.
---

## 10. Retry unique ajouté : contexte étroit, jamais de régression, corrige les erreurs syntaxiques

Suite à la découverte du problème d'étoile (section 9), décision de tester un mécanisme de
correction — nommé explicitement "self-refine" par l'utilisateur avant validation du principe.
Distinction actée avant codage, cruciale par rapport aux deux self-refine déjà testés et abandonnés
ailleurs (`resolve_conflicts` sur `U`, contexte incrémental du Protocole B) : celui-ci est
**unique** (un seul essai, jamais de boucle) et **borné par erreur détectée** (contexte étroit —
état concerné, raison exacte du rejet, liste des états valides restants — jamais le graphe entier).
Propriété structurelle recherchée et vérifiée : le résultat du retry repasse par exactement la même
validation mécanique que la proposition initiale, donc il ne peut jamais dégrader un résultat déjà
correct — au pire, retour à `UNRESOLVED`, jamais pire.

**Erreur de conception corrigée avant tout run réel** : la première implémentation modifiait le
fichier naïf existant et importait depuis lui, sans validation préalable de l'utilisateur — corrigé
sur demande explicite en un fichier `precondition_node_with_retry.py` totalement autonome (toute la
logique du naïf + validation recopiée dedans, aucun import croisé), le fichier naïf original restant
intact et jamais modifié.

**Résultat sur 9 runs (3 modèles × 3 répétitions)** : **zéro régression observée**, la garantie
architecturale tient empiriquement. Taux de correction très inégal selon le modèle :

| Modèle | Rétrogradations retentées | Corrigées | Taux |
|---|---|---|---|
| llama-3.1-8b | 8 | 5 | 63% |
| mistral-nemotron | 21 | 4 | 19% |
| gpt-oss-20b | 0 (bloqué en amont par troncature) | — | n/a |

**Interprétation** : le retry corrige bien les erreurs **mécaniques et locales** (auto-référence,
exclusivité mutuelle — le profil dominant chez llama-8b), mais ne peut rien contre un **biais
structurel de raisonnement** (le problème du graphe en étoile chez mistral-nemotron, qui reproduit
vraisemblablement la même construction bancale même avec un contexte de retry différent) ni contre
une **panne technique en amont** (gpt-oss-20b, dont l'appel initial échoue avant même de produire
quoi que ce soit à valider). Confirme empiriquement la prédiction faite avant codage : un seul
mécanisme ne peut pas couvrir des causes d'erreur de nature différente.

---

## 11. Investigation de `base_llm.py` : troncature, modèles indisponibles, config par modèle

Suite aux échecs de parsing persistants sur `gpt-oss-20b` (`raw response=''`), examen des fiches
modèle officielles NVIDIA pour chaque modèle de la liste `MODELS`. Deux découvertes :

- **`gpt-oss-20b`/`gpt-oss-120b` exposent un champ `reasoning_content` séparé du champ `content`**
  visible — confirmé dans la documentation officielle (exemple de code NVIDIA affichant les deux
  séparément). Hypothèse retenue : le budget de tokens peut s'épuiser dans le raisonnement interne
  avant que `content` ne commence à s'écrire, ce que `langchain_openai.ChatOpenAI` ne peut pas
  détecter (seul `.content` est lu par le pipeline actuel).
- **Confirmation qu'un petit modèle peut recevoir un `max_tokens` élevé** — les valeurs "par
  défaut" des fiches NVIDIA (1024 pour llama-8b/70b, 4096 pour gpt-oss, 16384 pour kimi) reflètent
  des choix d'exemple de documentation, pas des limites techniques du modèle. `max_tokens` relevé de
  4096 à une valeur plus généreuse pour tous les modèles (changement effectué par l'utilisateur,
  hors session).

**Deux modèles supplémentaires testés, tous deux indisponibles pour des raisons d'infrastructure,
non liées à notre code** :
- **`moonshotai/kimi-k2.6`** : `Error code: 404 - Function ... Not Found for account`. Confirmé
  depuis l'interface NVIDIA elle-même (pas seulement l'API) : `Invalid URL: Cannot parse
  function_id with value None`. Diagnostic définitif : modèle listé au catalogue mais sans
  déploiement d'infrastructure actif côté NVIDIA — aucun correctif possible côté client. Retiré des
  tests jusqu'à déploiement effectif par NVIDIA.
- **`thinkingmachines/inkling`** : pas d'erreur HTTP, mais réponses vides intermittentes et
  imprévisibles (contrairement au pattern reproductible de `gpt-oss-20b`) — y compris une
  reformulation tronquée en plein milieu de phrase. Probable instabilité du déploiement NVIDIA pour
  ce modèle (récent), pas un problème de configuration identifié à ce stade.
- **`mistral-medium-3.5-128b`** : sa fiche officielle utilise un paramètre `reasoning_effort`
  (`"high"`), non standard OpenAI, non transmis par `get_llm()` actuel (qui ne connaît que les
  paramètres standard de `ChatOpenAI`). Non testé après correction — nécessiterait `model_kwargs`
  ou équivalent pour transmettre ce paramètre ; laissé en l'état, non prioritaire.

**Résultat après augmentation de `max_tokens` sur `gpt-oss-20b` (run_10)** : amélioration nette mais
incomplète. `job_application`/`maternity_leave` : parfaits sur les deux prompts. `work_accident` :
**`one_shot` réussit intégralement (23 états, 0 rétrogradation), `zero_shot` échoue toujours**
(`raw response=''` persiste). Le budget de tokens à lui seul n'explique donc pas tout — l'écart
zero/one-shot sur ce modèle, déjà présent avant l'augmentation, persiste après.

---

## 12. Décision : `one_shot` retenu comme configuration de référence pour `Pre`

Comparaison consolidée de `zero_shot` vs `one_shot` sur `work_accident`, à travers l'ensemble des
runs disponibles (retry batch, section 10, + run post-`max_tokens`, section 11) :

| Modèle / run | zero_shot | one_shot |
|---|---|---|
| llama-3.1-8b (run_1) | 2 retentés, 1 corrigé, 1 encore UNRESOLVED | 0 rétrogradation |
| llama-3.1-8b (run_2) | 4 retentés, 3 corrigés, 1 encore UNRESOLVED | 1 retenté, corrigé, 0 restant |
| llama-3.1-8b (run_3) | ERROR (parsing complet) | 0 rétrogradation |
| mistral-nemotron (run_4) | 5 retentés, 1 corrigé, 4 restants | 3 retentés, 0 corrigé, 3 restants |
| mistral-nemotron (run_5) | 3 retentés, 0 corrigé | 3 retentés, 0 corrigé (égal) |
| mistral-nemotron (run_6) | 5 retentés, 1 corrigé, 4 restants | 3 retentés, 0 corrigé, 3 restants |
| gpt-oss-20b (run_7-9) | ERROR (3/3 runs) | ERROR (2/3), 1 succès partiel |
| gpt-oss-20b (run_10, max_tokens relevé) | ERROR persiste | 23 états, 0 rétrogradation |

**Sur 8 comparaisons directes, `one_shot` égale ou surpasse `zero_shot` systématiquement, jamais
l'inverse** — à travers les 3 familles de modèles testées, pas un artefact d'un seul modèle.
Interprétation cohérente avec l'observation déjà faite sur `U` (Protocole 1) : un exemple concret
contraint la forme de sortie attendue avant que le modèle commence à raisonner sur le contenu,
réduisant le budget gaspillé en exploration de structure — effet d'autant plus marqué que le texte
source est dense (`work_accident`).

**Décision** : `one_shot` devient la configuration par défaut pour toute évaluation ou comparaison
de modèles sur `Pre` à partir de maintenant. `zero_shot` reste conservé dans le code comme point de
comparaison pour une ablation dans le papier (quantifier l'écart plutôt que le simplement
constater), mais n'est plus traité comme une alternative équivalente pour la suite des expériences.

---

- **Traiter le problème du graphe en étoile** (section 9) — la validation actuelle ne peut pas
  détecter qu'une citation authentique justifie une précondition trop générique plutôt qu'une
  alternative plus spécifique. Piste à explorer : pénaliser ou signaler les états dont le
  in-degree (nombre de fois cités comme précondition d'autre chose) est anormalement élevé, comme
  heuristique de détection plutôt que de correction automatique.
- **`gpt-oss-20b` reste structurellement fragile sur `work_accident`** — limite de troncature déjà
  documentée sur ce modèle, indépendante de `Pre`. Pas de correctif prévu côté `Pre` ; à noter comme
  limite du modèle, pas du protocole.
- **`llama-3.1-8b` reste le moins stable des trois modèles testés** — cohérent avec son statut de
  modèle le plus faible de l'échantillon dans toutes les expériences précédentes (`U` et `Pre`). Ne
  pas chercher à le stabiliser artificiellement ; le documenter comme limite connue.
---

## 13. Extension du scope additif : support de `OR`, revenant sur une décision antérieure

Décision initiale (section 2) : `OR` explicitement hors périmètre, toute disjonction textuelle
marquée `UNRESOLVED`. Réexaminé après constat empirique répété que plusieurs rejets `UNRESOLVED`
sur `work_accident` correspondaient à de vraies disjonctions textuelles explicites (*"leads to a
fatality **or** serious injury"*) que le prompt interdisait par construction, pas par manque de
capacité du modèle — confirmé par l'observation que le modèle produisait parfois le `OR` en clair
malgré l'interdiction, avant d'être mécaniquement rejeté comme référence inconnue.

**Trois points de code modifiés, identifiés comme nécessaires ensemble avant de commencer** (un
seul changement isolé aurait laissé le comportement identique à l'ancien) :
1. **`precondition_prompt.py`** — l'interdiction explicite de `OR` retirée, remplacée par une
   règle l'autorisant avec un exemple few-shot dédié.
2. **Parseur/validateur** — reconnaissance de `" OR "` en plus de `" AND "`, chaque terme validé
   individuellement quel que soit l'opérateur reliant le groupe.
3. **`graph_node.py::build_edges`** — même reconnaissance, pour que les arêtes résultantes restent
   cohérentes avec la structure `AND`/`OR` choisie.

**Résultat fonctionnel : objectif atteint.** Sur le run testé, `OR` accepté correctement
(`work_accident.reported_to_inspectorate <- work_accident.fatal OR work_accident.serious_injury`),
alors que ce même cas était systématiquement rejeté avant (`references unknown state(s)`). `OR`
entre deux états de la même entité accepté sans déclencher la règle d'exclusivité mutuelle —
cohérent, cette règle reste explicitement scopée à `AND` seul. Zéro rétrogradation restante sur
`work_accident`, contre 3-4 auparavant, toutes dues à des `OR` mal interprétés.

**Fréquence réelle de `OR`, mesurée précisément pour éviter toute surestimation** : sur 66 états
au total (3 cas × 2 prompts sur ce run), seulement 8 `OR` maximum — concentrés exclusivement sur
`work_accident` (le seul des trois textes contenant réellement des règles disjonctives explicites),
`job_application` et `maternity_leave` restant à 0 `OR` sur 20 états. Le mécanisme ne "bascule" donc
pas systématiquement vers `OR` — il reste rare et localisé aux textes qui en contiennent
véritablement.

---

## 14. Trois problèmes révélés par ce même run, indépendants du mécanisme OR lui-même

**Bug de validation non détecté jusqu'ici : la clé cible (`target`) n'est jamais vérifiée.**
Observé : `"job.permanent OR job_application.rated"` accepté comme clé JSON, alors que ce n'est
même pas un `entity.state` valide — une string composite hallucinée par le modèle. `_validate_one`
vérifie les termes de la précondition contre `valid_targets`, mais jamais le `target` lui-même. Le
trou existait déjà avant le support `OR` mais restait invisible tant que le modèle ne produisait
que des clés simples ; il devient visible avec l'apparition de clés composites. `graph_node.py`
s'en protège déjà correctement (filtre défensif ajouté en aval, section graphe), mais `validated`
lui-même (le rapport intermédiaire sauvegardé) contient l'entrée invalide sans alerte. **Fix identifié,
non encore appliqué** : rejeter `target` d'emblée dans `_validate_one` si absent de `valid_targets`,
avant même d'examiner la précondition.

**Régression observée sur `one_shot`/`work_accident`, mistral-nemotron : 17/23 états retombés en
`INITIAL` contre 6/23 avant le changement de prompt.** Comparé à l'ancien comportement du même
modèle sur le même cas (stable auparavant), ce n'est pas une caractéristique normale du one-shot —
c'est une dégradation nouvelle. Hypothèse retenue : l'exemple `OR` a été intégré au même exemple
few-shot existant plutôt que dans un exemple séparé, diluant le point pédagogique original (chaîne
de propagation longue depuis un état déclencheur générique) au profit du nouveau point pédagogique
(`OR`) — un seul exemple devant désormais illustrer deux mécanismes différents. Piste de correction
identifiée, non encore tranchée : séparer les deux exemples plutôt que les fusionner.

**Limite du retry à contexte étroit, exposée concrètement pour la première fois** : 4 cycles à 2
nœuds détectés sur ce run (`work_accident.fatal ↔ person_with_accident_insurance.killed`), chacun
individuellement bien formé (référence valide, ancrage vérifié) mais contradictoire une fois
assemblé. Confirme empiriquement la limite déjà nommée au moment de concevoir le retry (section 10) :
chaque décision est prise isolément, sans connaissance de ce que les états voisins contiennent déjà.
`detect_cycles` fonctionne comme prévu — en diagnostic post-hoc, pas en prévention à l'acceptation.
Question ouverte, non tranchée : passer à une vérification d'acyclicité avant acceptation casserait
le principe "un seul essai, jamais de boucle" déjà validé comme garde-fou contre la dérive.

**`cot` peu fiable sur ce modèle** : 2 cas sur 3 échouent au parsing, le 3ᵉ qui passe
(`maternity_leave`, le texte le plus court) produit quand même des réponses contradictoires avec le
texte (`maternity_leave.taken <- INITIAL` alors que le texte décrit une planification préalable
explicite). Hypothèse retenue, cohérente avec les limites déjà documentées sur les modèles
reasoning (section 11, `gpt-oss-20b`) : le prompt CoT demande un raisonnement explicite en 6 étapes
par état, ce qui peut dépasser `max_tokens` sur les cas à beaucoup d'états avant même d'atteindre le
marqueur de fin.

---

- **Traiter le problème du graphe en étoile** (section 9) — non résolu par le retry (section 10),
  puisqu'il n'est pas détecté comme une violation par la validation mécanique actuelle. Piste à
  explorer : pénaliser ou signaler les états dont le in-degree (nombre de fois cités comme
  précondition d'autre chose) est anormalement élevé, comme heuristique de détection plutôt que de
  correction automatique.
---

## 15. Questions ouvertes / prochaines étapes

- **Fix identifié, non encore appliqué : rejeter `target` si absent de `valid_targets` dans
  `_validate_one`** (section 14) — priorité la plus immédiate, bug simple et localisé, déjà
  diagnostiqué précisément.
- **Régression `one_shot`/`work_accident` sur mistral-nemotron** (section 14) — piste retenue :
  séparer l'exemple `OR` de l'exemple de chaîne longue existant plutôt que les fusionner. Non encore
  testé.
- **Cycles contradictoires révélés par le retry à contexte étroit** (section 14) — décision à
  prendre : rester en détection post-hoc (`detect_cycles`, déjà en place) ou passer à une
  vérification d'acyclicité avant acceptation, au prix de casser le principe "un seul essai, jamais
  de boucle" déjà validé comme garde-fou.
- **`cot` peu fiable sur les cas à beaucoup d'états** (section 14) — piste : `max_tokens` dédié plus
  généreux pour cette variante spécifiquement, ou raisonnement condensé à une ligne par état plutôt
  que le détail explicite des 6 étapes.
- **`gpt-oss-20b` reste fragile sur `zero_shot`/`work_accident`** malgré l'augmentation de
  `max_tokens` — persiste même après correction du budget de tokens, ce qui indique une cause plus
  profonde que la seule troncature (probablement liée à l'absence de guidage structurel du
  zero-shot sur ce modèle reasoning). Résolu en pratique par la décision de la section 12 (`one_shot`
  comme référence), pas par une correction du modèle lui-même.
- **`llama-3.1-8b` reste le moins stable des trois modèles testés**, mais bénéficie le plus du
  retry (63% de correction) — cohérent avec son profil d'erreurs majoritairement syntaxiques plutôt
  que structurelles.
- **`kimi-k2.6` indisponible pour raison d'infrastructure NVIDIA** — à retester périodiquement, pas
  de correctif possible côté code.
- **`mistral-medium-3.5-128b` jamais testé avec son paramètre `reasoning_effort`** — nécessiterait
  d'étendre `get_llm()` pour transmettre des paramètres non standard (`model_kwargs` ou équivalent).
  Non prioritaire à ce stade.
- **Confirmer si le prompt naïf + validation + retry + OR tient aussi sur d'autres `U`** (pas
  seulement `run_17`) — tous les tests jusqu'ici utilisent le même `U` de référence ; un second `U`
  propre indépendant permettrait de vérifier que les bons résultats ne sont pas spécifiques à ce `U`
  précis.
- **Protocoles A et B abandonnés pour `Pre`**, mais le code reste disponible
  (`precondition_node_protocolA.py`, `precondition_node_protocolB.py`) — à garder en archive pour
  le papier (résultats négatifs documentés, utile pour justifier pourquoi l'architecture simple a
  été retenue plutôt que présumée par défaut).
- Ajouter un mécanisme d'ancrage pour `U` reformulé/reformulation-aware si le pipeline évolue vers
  un `U` différent de `run_17` à l'avenir — actuellement `run_filename` est fixé en dur dans le
  code, à paramétrer proprement si plusieurs `U` de référence doivent être comparés
  systématiquement.