# Observations — State Space Extraction Node

Notes de recherche accumulées pendant le développement de `agent/nodes/state_space_node.py`.
Objectif du node : extraire, à partir d'un texte d'intention en langage naturel, l'espace d'états
fini d'un processus — défini comme un ensemble d'entités, chacune avec une liste finie et
mutuellement exclusive d'états de cycle de vie.

Ces observations sont empiriques, obtenues par comparaison qualitative de sorties de modèles
(pas de ground truth indépendant à ce stade — voir section "Limite méthodologique actuelle").

---

## 1. Setup

- **Modèles testés** : `meta/llama-3.1-8b-instruct`, `meta/llama-3.3-70b-instruct`,
  `openai/gpt-oss-20b`, `openai/gpt-oss-120b` (via NVIDIA NIM), + un test manuel hors pipeline
  sur Gemini 3.1 Pro (zéro-shot identique).
- **Cas de test** : 3 textes d'intention en langage naturel — `job_application` (processus de
  candidature), `maternity_leave` (congé parental), `work_accident` (déclaration d'accident du
  travail, texte long et multi-clauses).
- **Prompts comparés** : `zero_shot`, `one_shot` (exemple bibliothèque, une seule entité),
  `two_shot` (bibliothèque + exemple order/shipment, deux entités distinctes).
- **Température** : fixée à 0 après un premier run où le même modèle, même texte, même prompt,
  produisait des sorties différentes d'un run à l'autre — nécessaire pour isoler la variance
  inter-modèles de la variance de sampling.

---

## 2. Définition opérationnelle retenue : "entité" vs "acteur"

Décision prise en amont du prompt engineering : l'espace d'états ne doit pas être défini sur
n'importe quelle variable, mais sur des **entités** — des objets/cas identifiables qui persistent
et progressent dans un cycle de vie (ex: `paper: [in_progress, submitted, reviewed, accepted]`).

Exclus explicitement de la définition :
- attributs/valeurs portées par une entité (score, date, compte)
- flags booléens
- rôles ou acteurs, sauf si le texte les traite eux-mêmes comme sujet du lifecycle

Ce vocabulaire ("entity" plutôt que "objet métier") suit la littérature *artifact-centric /
data-centric BPM* (Hull et al., GSM) pour rester dans un cadrage académique standard plutôt
que consulting.

---

## 3. Observation principale : le piège "acteur = entité" est universel et résistant

Sur les 3 cas testés, **tous les modèles NVIDIA testés** (8B → 120B) transforment à un moment
donné un acteur institutionnel (`company`, `employer`, `employee`, `school`) en entité à part
entière, avec des "états" qui sont en réalité des événements juxtaposés et non mutuellement
exclusifs (ex: `company: ['application_received', 'application_rated', 'offer_sent', ...]` —
une company peut être dans plusieurs de ces conditions simultanément, donc ce ne sont pas des
états au sens défini).

- Ce piège persiste **indépendamment de la taille du modèle** (présent de 8B à 120B).
- Il persiste **à travers les trois variantes de prompt** (zero/one/two-shot), y compris dans le
  two-shot où l'exemple 2 exclut explicitement un acteur analogue (`customer`, `store`).
- Seul **Gemini 3.1 Pro**, testé en zero-shot strictement identique, évite systématiquement ce
  piège sur les 3 cas — en le contournant structurellement (ex: invente une entité tierce
  `Company Review` plutôt que de forcer `company` à porter un lifecycle qu'elle n'a pas).

**Interprétation** : ce n'est probablement pas un problème de spécification du prompt (sinon le
two-shot l'aurait résolu), mais une limite de raisonnement propre à certains modèles sur cette
distinction précise. Un signal potentiellement publiable : une contrainte négative simple en
langage naturel ("X n'est pas une entité sauf si...") reste sous-déterminée pour plusieurs LLMs
même à taille importante, ce qui renforce l'argument qu'un mécanisme de vérification indépendant
du modèle générateur est nécessaire — pas seulement un meilleur prompt.

---

## 4. Observation : l'effet du few-shot n'est pas monotone

Zero-shot → one-shot améliore nettement les cas simples (`job_application`, `maternity_leave`) :
réduction du bruit, entités plus resserrées, `company: [unrated, rated]` apparaît enfin propre
chez llama-3.3-70b.

Mais one-shot **dégrade** le cas complexe (`work_accident`, texte long, multi-acteurs) :
- **Collapse en une seule entité fourre-tout** chez llama-3.1-8b et gpt-oss-120b — les "états"
  deviennent des catégories de contexte (`insured_employment`, `agricultural_forest`, ...) et non
  un lifecycle.
- **Apparition de flags booléens** chez llama-3.3-70b (`injured_person: [able_to_work,
  unable_to_work]`), alors que le prompt les interdit explicitement — absent en zero-shot,
  apparu après l'ajout du one-shot.

**Interprétation** : l'exemple bibliothèque (une seule entité) transmet un biais de forme
implicite ("une entité suffit généralement"), en plus du principe qu'il est censé illustrer. Sur
un texte simple ce biais est inoffensif, sur un texte structurellement dense il pousse le modèle
à sur-généraliser vers une décomposition trop pauvre. Conséquence méthodologique : **un exemple
few-shot n'est jamais un vecteur neutre du principe qu'on veut enseigner** — il transmet aussi sa
propre forme (nombre d'entités, longueur des listes, style), et cet effet doit être testé, pas
supposé négligeable.

Le two-shot (ajout d'un exemple à deux entités distinctes, acteurs explicitement exclus) a été
construit spécifiquement pour tester si ce biais se corrige.

**Verdict two-shot (résultat négatif)** : sur `work_accident`, les sorties one-shot et two-shot
sont quasi identiques pour les deux modèles comparés en profondeur (llama-3.1-8b, gpt-oss-20b) —
le collapse mono-entité chez llama-8b persiste à l'identique, le collapse binaire
(`reported`/`unreported` partout) persiste à l'identique chez gpt-oss-20b. Sur `maternity_leave`,
le two-shot fait même **régresser** llama-8b par rapport au one-shot (retour à `parent` seul comme
entité unique, alors que le one-shot avait produit `maternity_leave` comme entité artifact-centric).

**Conséquence** : l'hypothèse initiale ("le one-shot biaise vers 1 seule entité, un exemple
multi-entités devrait corriger ça") est invalidée par les données. Le problème sur les textes
denses n'est donc pas structurellement un biais de forme transmis par l'exemple — c'est plus
probablement une limite de raisonnement des modèles face à la densité du texte source elle-même,
que le nombre d'exemples few-shot ne compense pas. Ajouter un troisième exemple a été jugé peu
susceptible de changer ce constat ; décision prise de ne pas continuer à empiler des exemples et
de tester un levier différent (chain-of-thought, sections 6-7).

---

## 5. Observation : la complexité du texte dégrade la conformité plus que la taille du modèle ne la corrige

gpt-oss-120b est propre sur `maternity_leave` (texte court, un seul acteur) et se dégrade
fortement sur `work_accident` (texte long, clauses imbriquées, acteurs institutionnels multiples).
Le facteur de risque dominant semble être la **densité structurelle du texte source**, pas
la taille du modèle — un signal à creuser plutôt que "quel modèle choisir".

---

## 6. Chain-of-thought : problème technique (troncature silencieuse)

Prompt `cot` ajouté : raisonnement explicite en 5 étapes (lister les sujets candidats, trancher
entité/acteur avec justification, lister les états candidats par entité, vérifier l'exclusivité
mutuelle par paire, écrire la réponse finale) suivi d'un marqueur `FINAL_ANSWER:` puis le JSON
strict. Parsing adapté pour découper sur ce marqueur avant de parser.

**Premier run** : échec systématique sur tous les modèles/cas testés en `cot`, avec l'erreur
`Expecting value: line 1 column 1 (char 0)` — c'est-à-dire une chaîne vide reçue par `json.loads`,
pas un JSON malformé.

**Diagnostic** : `get_llm()` ne fixait aucun `max_tokens` explicite. Le prompt CoT étant nettement
plus long à produire (5 étapes + JSON), le budget de tokens par défaut de l'endpoint NVIDIA NIM
était probablement insuffisant, en particulier pour les modèles reasoning natifs (`gpt-oss-*`),
dont le raisonnement interne (invisible, indépendant du CoT explicite demandé) consomme le même
budget que le contenu visible — le budget peut s'épuiser avant qu'aucun `content` ne soit émis.

**Correctif appliqué** : `max_tokens` rendu paramétrable dans `get_llm()`, défaut remonté à 4096,
puis à 8192. Erreur de parsing rendue informative (inclut la réponse brute) pour diagnostiquer les
échecs futurs sans deviner. Après correctif, les erreurs de troncature diminuent mais ne
disparaissent pas totalement — `gpt-oss-20b` échoue encore, y compris en **zero_shot** sur
`work_accident` (prompt qu'il gérait sans problème auparavant), sans corrélation simple avec la
longueur du prompt. Confirme l'hypothèse d'un problème structurel des modèles reasoning face au
budget de tokens, pas un problème spécifique au prompt CoT.

---

## 7. Chain-of-thought : résultat qualitatif (négatif)

Une fois les erreurs de troncature réduites, le CoT n'améliore la qualité sur aucun des deux modes
d'échec qu'il visait à corriger, et introduit un nouveau mode d'échec :

- **Nouveau mode d'échec — l'utilisateur lui-même devient une entité.** Chez llama-3.1-8b sur
  `job_application` : `'You': ['Receive job offers', 'No longer have to report', 'In probation
  phase', 'Rated by company']` — pire que le zero-shot, ce ne sont pas des états mais des
  événements de log attribués au narrateur du texte.
- **Le flag booléen déguisé reparaît malgré l'étape de vérification dédiée.** `'Probation phase':
  ['In probation phase', 'Not in probation phase']` — l'étape 4 du raisonnement (vérification
  explicite d'exclusivité mutuelle par paire) ne suffit pas à l'empêcher.
- **Perte de couverture sur texte dense.** Sur `work_accident`, le CoT de llama-3.1-8b produit
  3 entités contre 10-12 en zero-shot, avec disparition complète de `employer`,
  `self-employed_person`, `school` — pas une simplification propre, un raisonnement qui semble
  s'arrêter avant d'avoir traité tous les acteurs du texte.
- **Le collapse binaire persiste à l'identique.** Chez gpt-oss-20b sur `work_accident` :
  `work_accident: [unreported, reported]`, `incident: [unreported, reported]`, `risk:
  [unreported, reported]`, `defect: [unreported, reported]` — motif rigoureusement identique à
  celui déjà observé en one-shot et two-shot pour ce modèle sur ce cas.

**Conclusion** : le CoT, comme le few-shot, échoue à corriger les modes d'échec ciblés sur les
textes structurellement denses, et en ajoute un nouveau. Deux stratégies de prompting
indépendantes (exemples et raisonnement explicite) échouent sur les mêmes points précis : entités-
acteurs, exclusivité mutuelle, complétude de couverture. **Décision : arrêter l'itération sur le
prompt seul.** Ce résultat négatif, documenté sur deux leviers de prompting distincts, renforce
l'argument central du papier — la fidélité de l'extraction NL → structure n'est pas garantie par
la capacité brute du modèle ni par des techniques de prompting standard, et nécessite un mécanisme
de vérification/correction indépendant en aval.

---

## 8. Limite méthodologique actuelle : pas de ground truth indépendant

Toutes les comparaisons ci-dessus sont des jugements qualitatifs par observation croisée des
sorties de plusieurs modèles — pas une évaluation contre une vérité de référence externe. Décision
prise : pas de gold set annoté manuellement à ce stade (coût jugé disproportionné). Alternative
retenue : boucle **HITL différée** — génération automatisée en batch, puis validation/correction
humaine a posteriori sur les résultats stockés, produisant un gold set comme sous-produit de
l'itération plutôt que comme campagne dédiée. Non encore implémentée dans le pipeline.

Piste secondaire évoquée : ground truth **synthétique par construction** — partir d'un espace
d'états fixé à la main, générer un texte à partir de lui, tester si l'extraction le retrouve. Utile
pour valider la mécanique du pipeline, mais ne teste pas la robustesse face à l'ambiguïté du texte
réel (l'essentiel du problème).

---

## 9. Reformulation utile du problem statement

Le problème de fidélité NL → structure, qu'on cherche à formaliser pour le papier, se manifeste
déjà à l'étape amont : extraire un "état des lieux" fiable depuis du texte n'est pas garanti même
avec le plus gros modèle disponible, et deux modèles corrects par ailleurs peuvent diverger
structurellement sur la même tâche à prompt identique. Cette difficulté observée empiriquement sur
le propre pipeline de construction de données est cohérente avec — et renforce — le problem
statement du papier.

---

## 10. Reformulation pré-extraction (inspirée de Nour Eldin et al., 2026)

Nouveau levier testé, distinct du prompting sur le texte source : une étape de **reformulation**
en amont de l'extraction, insérée entre le texte brut et le prompt zero-shot d'extraction. Approche
inspirée du papier *"A Decomposed Hybrid Approach to Business Process and Data Modeling with
LLMs"* (Nour Eldin, Dalmas, Gaaloul, IJCIS 2026), qui utilise une étape de reformulation similaire
en amont de son pipeline d'extraction BPMN pour réduire l'ambiguïté avant extraction.

**Prompt de reformulation contraint** (`agent/prompt/reformulation_prompt.py`) : instruction
explicite de ne pas inventer de règles/acteurs/entités non présents dans le texte original, de ne
pas résoudre les ambiguïtés en devinant, et d'expliciter quels objets persistent et progressent
dans le temps — sans changer le contenu. Décision de conception : le **texte reformulé est
systématiquement loggé** à côté de l'espace d'états extrait (`results/state_space/run_N.json`,
clé `reformulated_text`), pour pouvoir tracer si un gain de qualité vient d'une clarification
légitime ou d'un contenu halluciné par la reformulation elle-même — risque identifié avant
implémentation, le papier source admettant lui-même que sa reformulation peut compléter les
descriptions vagues avec les connaissances pré-entraînées du modèle.

**Résultat : effet non uniforme, un troisième levier de prompting à ne pas généraliser sans
prudence.**

| Modèle | Effet de la reformulation sur `work_accident` |
|---|---|
| gpt-oss-120b | Nette amélioration — 7 entités cohérentes, disparition du collapse binaire observé en two-shot |
| llama-3.1-8b | Neutre — moins fragmenté qu'en zero-shot (6 entités vs 12), mais piège acteur=entité toujours présent (`employees`, `employer`) |
| llama-3.3-70b | **Régression nette** — quasiment toutes les entités deviennent des flags booléens (`employee: [injured, not_injured, deceased]`, `employer: [notified, not_notified]`), pire que son zero-shot d'origine |

**Interprétation** : comme le few-shot et le CoT avant elle, la reformulation n'est pas un
correctif universel — son effet dépend fortement du modèle en aval, pas seulement de la qualité
du texte produit. Le risque anticipé (la reformulation peut simplifier le texte d'une façon qui
pousse vers le pattern binaire plutôt que de le corriger) se confirme concrètement chez
llama-3.3-70b.

---

## 11. Bug découvert : échec silencieux de la reformulation, auto-démonstration du problème étudié

Sur `gpt-oss-20b`, la reformulation a échoué silencieusement sur les 3 cas : `reformulate_text()`
a reçu une réponse vide (probable troncature par budget de raisonnement interne, cf. section 6),
cette chaîne vide a été injectée telle quelle dans le prompt d'extraction, et le modèle a produit
un JSON techniquement valide à partir de rien : `{'': []}`. Aucune exception levée — le parsing
JSON réussit puisque `{"": []}` est syntaxiquement correct.

**Ce résultat est noté explicitement car il reproduit, dans le pipeline de construction de données
lui-même, exactement le phénomène étudié par le papier** : un artefact syntaxiquement valide qui
ne réalise aucune fidélité à l'intention source, sans aucun signal d'alerte au moment de la
production. C'est le gap validity/fidelity du problem statement, observé ici non pas chez un
système tiers (NaLa2BPMN, ProMoAI, GIVUP) mais dans notre propre code.

Correctif nécessaire (non encore appliqué à la date de cette note) : `reformulate_text()` doit
vérifier que la sortie n'est pas vide et lever une erreur explicite plutôt que de laisser une
chaîne vide se propager silencieusement vers l'étape suivante.

---

## 12. Diagnostic affiné de l'échec CoT sur texte dense : explosion combinatoire, pas seulement budget de tokens

Grâce à l'erreur de parsing rendue informative (section 6), la réponse brute tronquée de
llama-3.1-8b sur `work_accident` en CoT est maintenant visible dans les logs. Elle montre
précisément où le raisonnement casse : l'étape 1 énumère environ 50 sujets candidats (le texte
`work_accident` étant riche en acteurs et circonstances), puis l'étape 4 — vérification
d'exclusivité mutuelle "pour chaque paire de conditions" — tente d'énumérer explicitement toutes
les paires de conditions candidates pour la seule entité `work accident`, rédigées en toutes
lettres une à une.

**Ce n'est plus uniquement un problème de `max_tokens` insuffisant** (diagnostic de la section 6) :
c'est un défaut de conception du prompt CoT lui-même. L'étape 4, formulée comme une vérification
par paire, scale quadratiquement (C(n,2)) avec le nombre de candidats retenus à l'étape 1. Sur un
texte dense comme `work_accident`, ça la rend structurellement inexploitable quel que soit le
budget de tokens alloué — augmenter `max_tokens` repousse la limite sans traiter la cause.

---

## 13. Reformulation renforcée (one-shot) : nouveau mode d'échec, pas de correction

Suite à la section 10, le prompt de reformulation a été renforcé (`reformulation_prompt_oneshot.py`) :
instruction explicite de signaler l'ambiguïté en prose plutôt que de la résoudre silencieusement
(ex. `"the text does not specify whether..."`), contrainte anti-invention de seuils/délais
numériques, et un exemple one-shot (location de voiture, hors du champ sémantique des 3 cas de
test) démontrant ce signalement explicite. `reformulate_text()` corrigée en même temps : lève
désormais une erreur explicite sur réponse vide (correctif de la section 11 appliqué), au lieu de
laisser passer une chaîne vide silencieusement.

**Résultat sur `work_accident` (llama-3.1-8b), le cas ciblé : régression vers un nouveau collapse,
plus dégénéré que les précédents.**

- `reformulated_zero_shot` : 8 entités actor-centric, verbeux et redondant mais avec des états
  substantiels tirés du texte (`'work accident'`, `'serious and immediate risk to safety and
  health'`...).
- `reformulated_one_shot` : 9 entités, **presque toutes réduites à `['active', 'inactive']`** —
  y compris pour `Labour Inspectorate`, `competent directorate`, `insured party`, qui n'ont
  conceptuellement aucun sens à porter un état "actif/inactif" au sens du texte.

**Interprétation** : ce nouveau collapse est plus vide de contenu que le collapse binaire
`reported`/`unreported` observé aux sections précédentes — ce dernier portait au moins une
information réelle extraite du texte. `active`/`inactive` est un label générique appliqué
mécaniquement, sans lien apparent avec le contenu. Hypothèse : l'instruction "signale l'ambiguïté
plutôt que de la résoudre" a été détournée par le modèle — au lieu de signaler en prose comme
démontré dans l'exemple, il semble avoir satisfait la contrainte formellement en assignant un état
binaire fourre-tout à chaque entité floue, sans produire l'information que l'instruction visait à
préserver.

**Signal positif isolé** : sur `maternity_leave`, le one-shot renforcé corrige effectivement le
comportement ciblé — bascule de l'entité-acteur `parent` (zero-shot) vers l'entité artifact-centric
`leave`. Fonctionne sur le cas simple, pas sur le cas dense.

**Effet collatéral observé** : le CoT échoue de nouveau sur `work_accident` (`parse failed`) malgré
`max_tokens=8192`, mais la troncature intervient cette fois *dans le JSON final* et non plus dans
le raisonnement étape par étape (cf. section 12) — signe que le point de rupture peut se déplacer
d'un run à l'autre, à `temperature=0` pourtant. Le déterminisme strict de l'endpoint reste à
vérifier (cf. question ouverte ci-dessous).

**Bilan cumulé** : quatre leviers de prompting testés indépendamment (few-shot, CoT, reformulation
zero-shot, reformulation one-shot renforcée) échouent tous à corriger le comportement sur texte
dense, chacun avec un mode de dégradation différent (collapse mono-entité, événements non exclusifs,
collapse binaire sémantique, collapse binaire vide de sens). C'est un résultat négatif consolidé,
pas un accident isolé — argument empirique de plus en plus solide pour la nécessité d'un mécanisme
de vérification/correction post-extraction plutôt que d'un raffinement supplémentaire du prompt.

---

## 14. Questions ouvertes / prochaines étapes

- ~~Correctif prioritaire : réponse vide de `reformulate_text()`~~ — **appliqué** (section 13) :
  lève désormais `ValueError` explicite au lieu de propager une chaîne vide.
- **Prompting abandonné comme levier universel, quatre résultats négatifs indépendants** —
  few-shot, chain-of-thought, reformulation zero-shot et reformulation one-shot renforcée
  échouent tous sur `work_accident`, chacun avec un mode de dégradation distinct (cf. section 13).
  Prochain levier à explorer : un mécanisme de vérification/correction post-extraction, indépendant
  du prompting.
- Si le CoT est repris plus tard, revoir la formulation de l'étape de vérification d'exclusivité
  mutuelle pour éviter l'énumération explicite par paire (coût quadratique) — par exemple, demander
  une vérification groupée par entité plutôt que par paire de conditions (cf. section 12).
- Vérifier le déterminisme réel de l'endpoint NVIDIA NIM à `temperature=0` — le point de rupture du
  CoT sur `work_accident` s'est déplacé entre deux runs identiques en apparence (cf. section 13).
- Intégrer Gemini comme provider dans `base_llm.py` pour le faire tourner dans le pipeline
  automatisé (mêmes prompts, mêmes cas, écriture dans le même `results/state_space/`), au lieu
  d'un test manuel hors pipeline — pertinent puisque Gemini est, à ce stade, le seul modèle testé
  à éviter le piège acteur=entité de façon stable.
- Implémenter la boucle HITL différée (script de review post-génération) pour commencer à
  accumuler un gold set validé sans bloquer l'itération.
- Investiguer le comportement de troncature silencieuse des modèles reasoning (`gpt-oss-*`) face
  au budget de tokens, indépendamment du CoT — apparaît aussi en zero-shot sur `work_accident`.
- **Positionnement littérature confirmé par Hermes (USENIX Security 2024) et RFCNLP (IEEE S&P
  2022)** : même un pipeline supervisé, lourdement outillé (grammaire dédiée, ~2800h d'annotation
  humaine pour Hermes) ne résout pas l'erreur d'extraction à la source — les deux déplacent la
  correction vers une vérification a posteriori contre un référentiel externe (tests de conformité
  3GPP pour Hermes). Nous n'avons pas d'équivalent à ce référentiel pour du texte métier généraliste,
  ce qui rend la case vide ciblée par la thèse (vérification de fidélité sans gold standard
  externe préexistant) plus difficile, et confirme sa pertinence.