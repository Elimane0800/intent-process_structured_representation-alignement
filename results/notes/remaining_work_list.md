# Remaining Work List — vers une soumission AAAI

Consolidation de tout ce qui reste ouvert à travers l'ensemble des documents du projet
(`pipeline_v3.pdf`, `tgms_formalism.pdf`, `dataset_run_v2_observations.md`,
`baseline_llm_judge_observations.md`, `tgms_formalism_to_code.md`), plus les points soulevés
mais jamais formellement enregistrés. Structuré en trois niveaux : **rédhibitoires**
(sans quoi le papier risque un rejet quasi certain), **fortement recommandés** (affaiblissent
significativement le papier sans être des motifs de rejet automatique), et **décisions à
trancher maintenant** (questions de conception, pas du travail d'exécution).

---

## Niveau 1 — Rédhibitoires (bloquants pour toute soumission)

### 1.1 Run complet sur les 24 descriptions, configuration V2
Tout résultat actuel (exploitabilité, taux VIOLATED, rappel perturbation, Zenodo) porte sur un
sous-ensemble de développement de 7 descriptions, choisi par stratification délibérée — jamais
un échantillon représentatif. C'est la dépendance de tout le reste de cette liste : plusieurs
points ci-dessous (Zenodo, richesse de guards, négation, AND/OR mixte) ne peuvent être tranchés
qu'avec ce run.

### 1.2 Baseline de similarité structurelle directe
Jamais commencée. Sans elle, la claim centrale du papier (le passage à une abstraction
état-centrique résout le problème de granularité texte-clause ↔ activité-BPMN) reste une
affirmation, jamais mesurée contre l'alternative la plus évidente qu'un reviewer proposera.

### 1.3 Baseline LLM-as-judge — terminer et nettoyer le premier run
Actuellement contaminé : `mistral-nemotron` quasi entièrement `inconclusive` sur deux variantes
de prompt sur trois (100% erreurs 429 — retry/backoff jamais ajouté à
`run_baseline_llm_judge.py`, alors que le pipeline principal l'a) ; `gpt-oss-20b`/`cot` à 92%
`inconclusive` (probable épuisement du budget de raisonnement, `reasoning_effort` jamais
appliqué au juge). Sans ce fix, deux des neuf configurations (3 juges × 3 variantes) restent
inutilisables — le résultat le plus fort obtenu jusqu'ici (faux positifs 0% TGMS vs jusqu'à
81.8% baseline) doit être reconfirmé sur des données propres avant d'être cité.

### 1.4 Corrélation Zenodo — toujours à zéro, jamais expliquée
ρ≈0 depuis le tout premier run à l'échelle (v1, 215 cas, déséquilibre de classe sévère : 0 à 6
cas "mauvais" sur 37-104). Le run v2 sur sous-ensemble ne permet pas de trancher (échantillon
non aléatoire). C'est un rédhibitoire déclaré dès le début du projet — sans un signal de
validité externe (accord avec un jugement humain/Zenodo), le papier revendique de la fidélité
sans jamais la confronter à un jugement indépendant. Trois issues possibles, à trancher après
le run complet (item 1.1) : (a) le déséquilibre de classe se résorbe à l'échelle et une vraie
corrélation apparaît ; (b) il ne se résorbe pas, et il faut soit élargir la collecte de cas
"mauvais" activement, soit remplacer cette validation par une autre stratégie et le dire
explicitement dans le papier ; (c) proposer une explication structurelle honnête si (a) et (b)
échouent tous les deux.

### 1.5 Étude de perturbation — étendre l'échelle
7 fichiers de base, 3 variantes par opérateur → intervalles de confiance très larges sur le
rappel (`2/12` peut plausiblement représenter n'importe quoi entre ~2% et ~48%). Nécessaire
pour toute affirmation de rappel citable, et pour vérifier si l'asymétrie de richesse de guards
entre modèles (item 1.6) est systématique ou spécifique à un cas.

### 1.6 Asymétrie de richesse de guards entre modèles — comprendre et neutraliser dans la mesure
Trouvé sur `M_g01/10` : `mistral-nemotron` extrait 16 guards / 12 cibles distinctes,
`llama-3.1-8b` 9/5, `gpt-oss-20b` 5/5 — sur un nombre **identique** d'activités matchées (16
partout). Le rappel `required` mesuré confond donc sensibilité du mécanisme et richesse de
`Pre` extraite par chaque modèle. Non tranché : systématique sur les 6 autres bases, ou cas
isolé ? Si systématique, décider entre rapporter un rappel brut (honnête mais confondu) ou
normalisé (isolé mais méthodologiquement plus complexe à justifier) — cf. item 3.5.

### 1.7 Hypothèse de granularité modèle/texte — trancher la cause exacte
Sur `V_k09/2`, l'hypothèse de fusion d'activités (une tâche du modèle accomplissant
plusieurs états à la fois) est **infirmée** — les activités concernées sont distinctes dans le
modèle. Redirigé vers une hypothèse de résolution d'entité (`order.withdrawn` vs
`product.withdrawn`, le texte suggérant clairement `product`). Non confirmée : nécessite
d'inspecter le `state_space` réel (`U`) de ce cas pour voir si `product.withdrawn` existe comme
candidat. Cette investigation doit être refaite sur d'autres cas avant de généraliser une
conclusion sur la granularité — un seul cas infirmé ne clôt pas la question pour le corpus
entier.

### 1.8 Relecture finale du formalisme TGMS
Une passe de relecture a déjà eu lieu et a trouvé une vraie erreur (le théorème de réduction
vers GSM, corrigé — voir `tgms_formalism.pdf`, remplacé par une position plus modeste et
honnête). Une seconde passe indépendante (par un tiers si possible, sinon une seconde relecture
à froid) reste recommandée avant intégration finale au papier, étant donné qu'une erreur a déjà
été trouvée une fois après une première rédaction qui semblait complète.

---

## Niveau 2 — Fortement recommandés (affaiblissent le papier, pas rédhibitoires seuls)

### 2.1 Reproductibilité à `temperature=0`
Jamais testée sur aucun des trois modèles. Un reviewer demandera si les résultats sont stables
d'un run à l'autre.

### 2.2 Isoler la cause exacte du gain d'exploitabilité `gpt-oss-20b`
Backoff exponentiel et `reasoning_effort="low"` ont été ajoutés dans le même run — le gain
(42.3%→100%) leur est attribué conjointement, jamais isolé individuellement.

### 2.3 Hypothèse cycles/déduplication
Le taux de cycles de précondition a augmenté sur le sous-ensemble v2 (llama 15.3%→23.3%,
gpt-oss 0.9%→11.7%) après l'ajout de la déduplication référentielle. Hypothèse non tranchée :
composition du sous-ensemble, ou effet secondaire réel de la fusion d'entités qui referme des
boucles absentes avant fusion. Test ciblé (dedup on/off sur les mêmes cas) jamais fait.

### 2.4 Risque de sur-fusion de `U` pour `gpt-oss-20b`
Richesse moyenne de `U` en recul net pour ce modèle spécifiquement (16.2→11.4, un cas à 1 seul
état) après ajout de la déduplication. Possible sur-fusion propre à ce modèle, jamais vérifiée
directement sur le rapport `state_space_dedup`.

### 2.5 Accord inter-modèles — recalculer proprement
La hausse observée (27%→40%) est probablement expliquée mécaniquement par la baisse des
rapports manquants (gain d'exploitabilité), pas par une vraie convergence de jugement — biais de
complétude jamais isolé statistiquement.

### 2.6 Licence AGPL v3 de PM4Py
Point de vigilance non tranché depuis le tout début (contexte CIFRE) — à clarifier avant toute
publication ou mise à disposition du code.

### 2.7 Score de conformité agrégé — absence à justifier explicitement dans le papier
Décision délibérée de ne jamais implémenter de score global unique — vaut la peine d'un
paragraphe explicite dans le papier plutôt que de laisser un reviewer se demander pourquoi il
manque.

---

## Niveau 3 — Décisions à trancher maintenant (conception, pas exécution)

### 3.1 Grammaire des guards : la séparation stricte AND/OR est-elle vraiment tenable ?
**Nouveau point, jamais audité.** Toute l'architecture actuelle (`_validate_one` règle 2,
`group_edges_by_target`, la Definition 1 du formalisme TGMS) suppose qu'un guard pour une
cible donnée est **soit** une conjonction pure, **soit** une disjonction pure — jamais mixte
(ex. `(A AND B) OR C`, ou `A AND (B OR C)`). Cette hypothèse n'a jamais été vérifiée contre le
corpus réel. Deux risques distincts si elle est fausse dans des cas réels :
- Si le texte exprime réellement une précondition imbriquée, `_validate_one` la rejette ou la
  dégrade en `UNRESOLVED` — perte silencieuse d'information réelle, pas une erreur détectée.
- Si plusieurs guards distincts pour la même cible sont incorrectement fusionnés sous un seul
  opérateur par `group_edges_by_target` (qui suppose l'unicité de l'opérateur par cible), le
  graphe de référence lui-même serait mal construit.
**À trancher** : auditer la fréquence de préconditions réellement imbriquées sur le corpus
complet (item 1.1) avant de décider. Si négligeable, documenter l'hypothèse comme confirmée et
la garder. Si significatif, décider entre (a) étendre la grammaire à des expressions booléennes
imbriquées (changement de scope substantiel, touche `graph_node.py`, `alignment_node.py`/
`tgms_solver.py`, et la Definition 1 du formalisme) ou (b) détecter et signaler explicitement
les cas mixtes comme `UNRESOLVABLE` plutôt que de les fusionner silencieusement sous un mauvais
opérateur (changement mineur, perte d'expressivité assumée et documentée).

### 3.2 Négation dans la grammaire des guards
Mode d'échec identifié sur le sous-ensemble (cluster `G_g03`, cas "if not X") — grammaire
actuelle incapable d'exprimer une condition négative, le LLM d'extraction produit une
précondition positive plausible mais sémantiquement inversée. Fréquence à l'échelle du corpus
complet non mesurée. **À trancher après item 1.1** : si fréquent, extension de grammaire
nécessaire (couplée à la décision 3.1, puisque les deux touchent la même Definition 1) ; si
rare, documenter comme limite connue et ne pas étendre.

### 3.3 XOR — côté texte et côté process
Décision déjà posée explicitement (note TGMS, paragraphe XOR) : intégration conditionnée à
preuve empirique. Côté process (`rewire_gateway`), le perturbation study confirme l'angle mort
exactement comme prédit (0% détection, par construction) — **pas** un coût observé qui
déclencherait la condition. Côté texte, recherche de motif faite seulement sur le sous-ensemble
de 7 descriptions, aucune vraie occurrence de guard XOR multi-termes trouvée. **À trancher** :
refaire la recherche de motif texte sur le corpus complet (item 1.1) avant de considérer la
question close ; si toujours absent, documenter la décision de ne pas intégrer comme
définitivement close pour cette version du papier.

### 3.4 Granularité activité/état — top-k ou décomposition ?
Suite à l'item 1.7 : si l'hypothèse de résolution d'entité est confirmée comme cause générale
(pas seulement `V_k09`), une vraie décision d'architecture se pose pour `state_matching_node`.
Deux pistes identifiées, jamais tranchées : (a) retenir les $k$ meilleurs candidats au-dessus
d'un second seuil au lieu du seul meilleur (reste déterministe, pas de nouvel appel LLM,
risque de sur-génération d'ancrages) ; (b) décomposer les activités composites avant embedding
(plus proche de la cause réelle, réintroduit un appel LLM à une étape actuellement
purement symbolique). **À trancher** seulement après confirmation de la cause sur plusieurs cas,
pas sur la base d'un seul cas infirmé/orienté.

### 3.5 Présentation du rappel de perturbation — brut ou normalisé par richesse de guards ?
Suite à l'item 1.6. Si l'asymétrie de richesse de guards entre modèles se confirme
systématique, décider si le papier rapporte le rappel brut (honnête sur ce qui a été mesuré,
mais mélange deux causes) ou un rappel conditionné aux mutations tombant sur une zone
effectivement couverte par un guard (isole la question "le mécanisme rate-t-il de vrais
désordres visibles", méthodologiquement plus propre, demande une justification plus longue dans
le papier).

### 3.6 Théorème de réduction TGMS — s'arrêter à la version affaiblie, ou investir dans une preuve restreinte réelle ?
La version actuelle (`tgms_formalism.pdf`) énonce une parenté méthodologique avec GSM
(Remark, pas Théorème) après qu'une tentative d'identité formelle se soit révélée fausse. Une
vraie réduction formelle resterait possible sous une hypothèse structurelle supplémentaire
(guards sans référence croisée entre eux, profondeur 1) — jamais formulée ni prouvée. **À
trancher** : investir le temps de formuler et prouver cette version restreinte pour un théorème
plus fort dans le papier, ou assumer la version actuelle (plus modeste, déjà solide via les deux
théorèmes de monotonie) comme suffisante. Pas bloquant pour soumettre, mais change la force de
la section formalisme.

### 3.7 Timing du packaging standalone TGMS
Prototype déjà existant (`tgms/` en package pip-installable, testé en isolation). Délibérément
mis de côté jusqu'à validation complète. **À trancher** : packager avant ou après soumission —
n'affecte pas le contenu du papier, seulement la disponibilité du code pour reproductibilité/
review.

### 3.8 Frontière formalisme / implémentation dans le papier — tranchée, à respecter partout en rédigeant
**Décision actée** : la section "Formalisme" du papier ne contient que l'objet TGMS abstrait
$(M, \mathcal G, A, \Gproc, \varphi, c)$, le solveur $\mathrm{sat}_R$, et les deux théorèmes de
monotonie (confiance, enrichissement du graphe process). Le reste — pipeline neurosymbolique,
extraction, calibrage, résultats — va en section évaluation/implémentation. Implications
précises à respecter pendant la rédaction, pas juste un découpage de convenance :

- **N'entrent PAS dans le formalisme** : le seuil $\tau=0.35$ (décision empirique, jamais un
  résultat prouvé) ; toute affirmation sur la qualité de l'extraction de $\mathcal G$ depuis le
  texte (aucune preuve possible qu'un LLM extrait correctement — seulement des mesures) ; le
  Remark de parenté avec GSM reste dans le formalisme mais explicitement comme Remark, jamais
  présenté comme un résultat porteur (cf. item 3.6).
- **Phrase de pont obligatoire, à écrire explicitement entre les deux sections** : les
  théorèmes prouvent des propriétés *sous hypothèse* que $\mathcal G$ respecte la grammaire de
  la Definition 1 (guards purs AND/OR, jamais mixtes, jamais auto-référentiels). Le pipeline ne
  *prouve* jamais que l'extraction produit toujours des guards conformes — il le **garantit
  mécaniquement** (`_validate_one` rejette/dégrade tout ce qui ne respecte pas la grammaire
  plutôt que de l'accepter silencieusement). Sans cette phrase explicite, un reviewer demande
  "comment savez-vous que vos théorèmes s'appliquent aux vraies données ?" sans réponse déjà
  écrite. Se connecte directement à l'item 3.1 : si l'audit du corpus complet révèle des
  préconditions réellement imbriquées, cette phrase de pont doit refléter honnêtement ce que le
  système en fait (rejetées en `UNRESOLVABLE`, jamais mal-interprétées silencieusement), pas
  prétendre que le cas ne se présente jamais.
- **Placement de la preuve d'équivalence stricte (324 mutants, 0 écart)** : ni pure théorie
  (c'est une mesure empirique) ni pure implémentation (elle valide la conformité du code à
  l'objet formel) — va en section évaluation/résultats, présentée explicitement comme le pont
  inverse : preuve que l'implémentation instancie fidèlement l'objet formel de la section
  formalisme, pas seulement l'inverse (théorèmes qui s'appliquent aux données).
- **Règle à appliquer partout en rédigeant la section pipeline** : jamais d'environnement
  `\theorem` dedans. Toute affirmation y est appuyée par une mesure, jamais par une preuve —
  évite le piège de dignifier une observation empirique en pseudo-théorème pour paraître plus
  rigoureux, cohérent avec la discipline déjà tenue ailleurs dans le projet (score de
  conformité jamais implémenté, seuil GMM rejeté par inspection plutôt que gardé par confort
  théorique).

---

1. Run complet 24 descriptions (1.1) — débloque 1.4, 1.6, 3.1, 3.2, 3.3.
2. En parallèle : fix retry/backoff + `reasoning_effort` sur la baseline LLM-judge (1.3),
   ré-inspection `state_space` pour trancher 1.7.
3. Une fois 1.1 disponible : étendre le perturbation study (1.5), démarrer la baseline de
   similarité structurelle (1.2).
4. Trancher 3.1/3.2/3.3 sur données du corpus complet, décider extensions de grammaire si
   nécessaires.
5. Seconde relecture du formalisme (1.8) en parallèle, indépendante du reste.
6. Niveau 2 en continu, au fil de l'eau, pas bloquant pour une première version du papier.