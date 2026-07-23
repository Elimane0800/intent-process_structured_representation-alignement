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

## 14. Mécanisme neurosymbolique post-extraction : citation, ancrage, conflict resolution

Suite au constat de six résultats négatifs sur le prompting seul (sections 4, 7, 10, 13), changement de
levier : au lieu de continuer à raffiner le prompt d'extraction, ajout d'un mécanisme de
vérification/correction **post-extraction**, inspiré de deux papiers de la revue de littérature
(Hemmer et al. 2025, validation neurosymbolique par couches ; Li et al. 2024, G&O — décomposition
génération/organisation en tours séparés). Trois fonctions ajoutées à `state_space_node.py`,
appliquées après coup sur les 6 variantes de prompting déjà existantes, sans modifier ces dernières :

- **`extract_evidence()`** — second appel LLM : pour chaque entité déjà extraite, demande une citation
  verbatim de la ou les phrases du texte source qui la justifient. Design délibéré : on ne demande pas
  au modèle de justifier le *nom* de l'entité (construction du modèle, rarement un span littéral du
  texte — le fuzzy-match sur le nom testé initialement était un signal faible), mais la *preuve*
  censée la soutenir, qui elle doit être un span réel.
- **`validate_grounding()`** — pure Python, aucun appel LLM. Vérifie que chaque citation est un span
  exact (ou quasi-exact, `difflib`) du texte source. Teste si la justification prétendue par le modèle
  est réelle, pas si l'entité elle-même est fondée.
- **`resolve_conflicts()`** — second tour LLM (self-refine) : montre au modèle sa propre extraction et
  lui redemande de trancher entité/acteur et exclusivité mutuelle selon la même définition, avec
  interdiction explicite d'ajouter de nouvelles entités. Testé malgré une réserve documentée avant
  implémentation : la littérature sur le self-refine (ex. Huang et al. 2023) montre que l'auto-critique
  sans vérificateur externe peut se convaincre elle-même d'un changement non réellement meilleur — donc
  traité comme une expérience à évaluer avec la même rigueur que les précédentes, pas comme un
  correctif présupposé fiable.

---

## 15. Résultats du mécanisme neurosymbolique (run_13) : deux nouveaux modes d'échec, un signal positif isolé

**Conflict resolution : échec majoritaire (~8/18 cas), mais d'un type nouveau — échec de clôture, pas
de jugement.** Dans plusieurs cas ratés (ex. `maternity_leave`, zero_shot), le modèle identifie
*correctement* en prose que `parent` est un acteur et que les états ne sont pas mutuellement
exclusifs — un raisonnement juste — mais son JSON final ne reflète pas cette analyse, ou du texte
supplémentaire est généré après le JSON, cassant le parsing. Contrairement aux modes d'échec
précédents (collapse, binaire, actor=entity — des erreurs de *jugement*), celui-ci est une erreur
*d'exécution* : le modèle voit juste mais ne clôture pas proprement sa sortie structurée. Corollaire :
quand le parsing réussit, la correction est parfois réellement substantielle et positive — ex.
`reformulated_zero_shot`/`job_application` : `job_applicant` passe de 7 états mêlant actions et états à
3 états épurés après conflict resolution. Le mécanisme n'est donc pas inutile en soi ; son taux
d'échec actuel vient principalement d'un problème de format de sortie, pas de raisonnement.

**Limite découverte dans `validate_grounding` : la recombinaison de fragments réels peut tromper le
vérificateur.** Sur `reformulated_one_shot`/`work_accident`, plusieurs citations passent
`grounding: true` alors qu'elles recombinent des fragments individuellement exacts du texte source
d'une façon qui **altère le sens** (ex. omission silencieuse d'une négation en combinant deux clauses
disjointes). Chaque fragment est un vrai substring, donc le test passe, mais la citation résultante ne
représente pas fidèlement ce que dit le texte à cet endroit. C'est une limite de conception à noter
explicitement : `validate_grounding` teste l'authenticité des fragments, pas la fidélité sémantique de
leur recombinaison — un mode de "triche" subtil découvert empiriquement plutôt qu'anticipé.

**Défaut de conception à corriger : le grounding sur les variantes reformulées compare au mauvais
texte.** `postprocess(reformulated, state_space, llm)` compare la citation au texte **reformulé**, pas
au texte source original. Une dérive de la reformulation elle-même (déjà documentée sections 10-13)
ne serait donc jamais détectée par ce grounding tel qu'implémenté — il valide la fidélité au texte
intermédiaire, pas à l'intention d'origine.

**Signal positif isolé, à l'inverse** : sur ce même cas (`reformulated_one_shot`/`job_application`),
`grounding` détecte correctement `job_offer`, `job_applicant`, `company` comme `false` — ces citations
sont en réalité des paraphrases plutôt que des extraits, et le vérificateur les flague correctement.
Preuve que le mécanisme fonctionne comme prévu au moins dans certains cas, avant même correction des
deux limites ci-dessus.

**Bug de format identifié et à corriger en priorité** : le prompt de conflict resolution ne force
aucun marqueur de fin (contrairement au CoT, section 6, qui utilise `FINAL_ANSWER:`). Ajouter un tel
marqueur devrait éliminer la majorité des échecs de parsing observés ici, puisque le raisonnement en
amont est souvent correct — le problème est la clôture, pas le contenu.

---

## 16. Changement de paradigme : Protocole 2 — extraction permissive + validation symbolique déterministe

Après six résultats négatifs indépendants sur le prompting seul (few-shot, CoT, reformulation ×2,
définitions V1/V2/V3), et un biais confirmé stable à travers quatre familles de modèles (llama-8b,
llama-70b, mistral-nemotron, gpt-oss-120b) sur le même cas (`company`/`job_application`, résistant
même à un prompt V3 spécifiquement conçu pour son cas d'ambiguïté mesurée), décision de changer de
nature d'intervention plutôt que de continuer à raffiner le prompt.

**Diagnostic préalable (déterminant, pas hasard) — biais confirmé, pas bruit.** Test de répétition
(même prompt, `temperature=0`, 3-5 runs identiques) sur llama-8b, llama-70b, mistral-nemotron :
`company` apparaît **identique à travers toutes les répétitions**, sur les trois modèles. Ce n'est
donc pas un problème de variance d'échantillonnage — un mécanisme de consensus par sampling
(self-consistency / "Monte-Carlo" au sens échantillonnage à `temperature>0`, distinct du MC Dropout
qui nécessite un accès aux poids) n'aurait aucune prise dessus, puisqu'il n'y a essentiellement
aucune variance à exploiter. Décision de ne pas poursuivre cette piste.

**Architecture retenue**, formalisée à partir d'un diagramme de pipeline plus large fourni par
l'utilisateur (Macro : intent inference → U (state space) → Pre (préconditions) → Mapping(1) → G ;
Micro : alignement avec le BPMN observé → conformity score). Décision explicite de ne **pas**
adopter la restructuration façon Saccon et al. (primitives locales → construction algorithmique de
`U` par atteignabilité) : la granularité résultante d'une construction déterministe ne serait pas
garantie compatible avec celle d'un BPMN réellement généré, observé dans la branche parallèle du
pipeline (`Mapping(2)`) — un problème de mismatch de formalisme, plus profond que celui de fidélité
d'extraction, que ce protocole ne veut pas introduire.

**Protocole retenu** : découpler la tâche difficile (jugement de granularité) en deux étapes de
nature différente.

- **Étape A — génération permissive (LLM)** : le prompt (`agent/prompt/state_space_prompt_permissive.py`,
  4 variantes zero/one/two-shot/cot) ne demande plus aucun jugement entité/acteur — seulement une
  énumération sur-inclusive de tout candidat associé à un changement de condition dans le texte.
  Toute la mécanique de règle d'exclusion (présente en V1/V2/V3) est retirée du prompt.
- **Étape B — validation symbolique déterministe (aucun LLM, aucun entraînement)** :
  `agent/nodes/state_space_node_v2.py`. Deux signaux, calculés sur le texte source brut :
  - `f_srl` — ratio objet/(sujet+objet) du mot-tête du candidat, via dependency parsing spaCy
    (`en_core_web_sm`). Note de terminologie : ce n'est pas du véritable Semantic Role Labeling
    (rôles sémantiques stables, type PropBank/Agent-Patient) mais une approximation syntaxique de
    surface (fonction grammaticale locale) — distinction vérifiée en confrontant à la littérature
    SRL (Gildea & Jurafsky 2002 ; "LLMs Can Also Do Well" ACL 2025, arXiv 2506.05385, qui montre
    qu'un LLM zero-shot est très peu fiable sur la tâche SRL elle-même — 10 à 21% F1 sans
    fine-tuning — confirmant que ce jugement ne doit pas être délégué au LLM lui-même).
  - `f_wn` — classification par hyperonyme WordNet (0 = organisation/personne, 1 = artefact/document/
    événement, 0.5 = indéterminé), du sens le plus fréquent du mot-tête, via NLTK.
  - **Règle de vote sans poids calibrés** (délibérément préférée à une somme pondérée `w₁f_srl+w₂f_wn`
    dont les coefficients auraient été choisis à dire d'expert — jugé trop fragile face à un
    reviewer AAAI) : rejet uniquement si les deux signaux s'accordent (`f_srl ≤ 0.5` ET `f_wn = 0`).
    Par défaut, un candidat est conservé (le rappel prime sur la précision à ce stade — cohérent
    avec l'objectif reformulé : accepter une erreur résiduelle d'exhaustivité, éliminer l'erreur de
    jugement).

**Bugs corrigés en cours de route** :
- `_head_word()` ne lemmatisait pas son résultat — un candidat pluriel/composite (`"company
  reviews"`) ne matchait jamais `token.lemma_` (singulier) calculé sur le texte, cassant
  silencieusement `f_srl` (`None` systématique). Corrigé en lemmatisant la tête via spaCy avant
  comparaison.
- Absence de filtre pour les pronoms/mots vides (`You`, `process`, `it`...) produits par CoT —
  ajout d'une liste noire (`STOP_CANDIDATES`) appliquée avant le calcul des signaux.

**Pistes explorées puis explicitement abandonnées, pour éviter l'accumulation de rustines** :
un troisième signal basé sur un modèle NLI zero-shot (`typeform/distilbert-base-uncased-mnli`,
puis `facebook/bart-large-mnli`) a été testé pour corriger spécifiquement le cas `maternity_leave`
(voir §19). Résultat mitigé : corrige bien le cas visé mais introduit un nouveau biais sur un cas
qui fonctionnait déjà (`application` classé à tort comme acteur à 66-71%). Décision explicite de ne
**pas** l'intégrer, même de façon conditionnelle (n'activer NLI que si `f_srl` est peu fiable) —
reconnu comme un début d'accumulation de rustines ad hoc sans principe unificateur, répétant au
niveau symbolique l'anti-pattern déjà observé et abandonné au niveau du prompting (patcher
réactivement chaque cas d'échec découvert plutôt que d'évaluer l'architecture globale). Retour
délibéré à la règle à deux signaux, plus simple et plus défendable.

---

## 17. Résultats du Protocole 2 : succès reproductible sur le biais central, deux limites caractérisées

Testé sur 5 runs (3× llama-3.1-8b, 2× llama-3.3-70b), 4 prompts (zero/one/two-shot/cot), 3 cas.

**Succès stable et reproductible** :
- `company`/`companies` rejeté correctement sur `job_application` dans **10/10 occurrences**
  exploitables, à travers les deux modèles et tous les prompts (à l'exception d'un candidat
  composite `"company reviews"` produit par un `zero_shot` sans exemple — cf. limite de
  granularité de l'étape A, non un défaut de la couche B). C'est le premier résultat qui corrige,
  de façon reproductible, le biais qui avait résisté à 9 stratégies de prompting sur 4 familles de
  modèles.
- `employee` rejeté correctement sur `work_accident` dans **10/10 occurrences** (`f_srl=0.0,
  f_wn=0.0` — accord net des deux signaux), un succès nouveau non observé en Protocole 1.

**Limite 1 — dépendance à la fréquence d'occurrence, caractérisée précisément.**
`Company` sur `maternity_leave` : gardé à tort dans **5/5 runs**, de façon parfaitement
reproductible (`f_srl=1.0` à chaque fois). Cause identifiée : le texte ne mentionne "company" que
1-2 fois, toutes en position objet syntaxique (`"Notify Social Security, Company in time"`,
`"Gather information from companies"`) — aucune occurrence sujet dans ce texte précis, contrairement
à `job_application` où le texte dit explicitement *"Companies have to confirm..."*. Le signal `f_srl`
mesure un **comportement syntaxique dans le texte courant**, pas une propriété stable du candidat —
sur un texte court à faible redondance, ce signal devient peu fiable, voire trompeur (ici il pointe
activement dans le mauvais sens, pas juste "indéterminé"). Limite reproductible et caractérisable
("le signal se dégrade en dessous d'un certain nombre d'occurrences"), documentée comme telle plutôt
que patchée.

**Limite 2 — seuil de rejet trop strict pour certains acteurs sémantiquement nets.**
`employer` gardé à tort dans **10/10 occurrences**, avec un ratio stable et reproductible
(`f_srl≈0.67`, jamais ≤0.5) malgré `f_wn=0.0` net (WordNet identifie clairement "employer" comme
`person.n.01`). Le comportement syntaxique de ce candidat est mixte mais pas assez tranché pour
franchir le seuil actuel, même si le signal sémantique est sans ambiguïté. Contrairement à la
Limite 1 (signal peu fiable), ici le signal sémantique est fiable mais le seuil de la règle de vote
ne l'exploite pas assez — piste de correction potentielle (ex: rejet si `f_wn=0.0` seul, quelle que
soit `f_srl`, au lieu d'exiger l'accord des deux) volontairement non appliquée à ce stade pour éviter
de re-basculer vers l'accumulation de règles ad hoc sans validation plus large.

**Confirmation indépendante, mode d'échec de l'étape A déjà documenté en Protocole 1** :
`work_accident` continue de produire des échecs de parsing fréquents (troncature, cf. section 6) et
une sur-fragmentation en candidats contextuels redondants (`"work accident (report by
self-employed person)"`, observé en two-shot) — ce protocole ne corrige pas ces deux problèmes, qui
relèvent de l'étape A (qualité de génération du LLM), hors du périmètre de la couche B.

---

## 18. Signal NLI zero-shot : piste testée, écartée après diagnostic

Hypothèse testée : un signal indépendant de la fréquence d'occurrence (contrairement à `f_srl`)
pourrait corriger la Limite 1 ci-dessus. Un modèle NLI zero-shot classifie chaque candidat contre
deux hypothèses ("organisation/acteur" vs "document/dossier suivi dans le temps"), sans besoin de
plusieurs occurrences dans le texte.

- `typeform/distilbert-base-uncased-mnli` (léger) : échoue nettement — classe `Company`/
  `maternity_leave` à 87% "document" (faux, dans le mauvais sens) et `application`/`job_application`
  (cas le plus évident du corpus) à 71% "acteur" (faux). Modèle trop petit pour cette distinction
  fine.
- `facebook/bart-large-mnli` (référence standard) : corrige bien le cas visé (`Company`/
  `maternity_leave` → 76% "acteur", correct), mais introduit un nouveau biais sur le cas qui
  fonctionnait déjà (`application` → 66% "acteur", faux) — signe d'un biais de label plutôt que
  d'un jugement sémantique robuste (probablement lié à la formulation asymétrique des deux
  hypothèses).

**Décision** : ne pas intégrer, même conditionnellement. Le gain sur la Limite 1 ne compense pas le
risque de casser un cas déjà résolu, et l'ajout aurait constitué un troisième patch réactif sur la
couche symbolique — reconnu explicitement comme reproduisant, côté symbolique, l'anti-pattern déjà
abandonné côté prompting. La règle à deux signaux (`f_srl`+`f_wn`) est conservée telle quelle, avec
ses deux limites documentées en section 17 plutôt que masquées.

---

## 19. Protocole 3 : validation par lexnames WordNet, remplace SRL+hypernymes manuels

Motivation directe : le Protocole 2 (§16-18) nécessitait de maintenir `ACTOR_HYPERNYMS`/
`ENTITY_HYPERNYMS`, deux listes de synsets choisies à la main — un problème de généralité reconnu
explicitement : chaque nouveau cas de test ou domaine métier risquait de révéler un acteur non
couvert, obligeant à étendre la liste indéfiniment ("à chaque nouveau use case il faudra une
correction après coup — c'est infaisable").

**Solution retenue** : les **lexnames** WordNet — une taxonomie **fixe et fermée** de 25 catégories
sémantiques (`noun.person`, `noun.group`, `noun.artifact`, `noun.act`, ...) qui couvre déjà tout
WordNet. Le mapping acteur/entité (`{noun.person, noun.group} → ACTOR`, `{noun.artifact,
noun.communication, noun.act, noun.event, noun.cognition} → ENTITY`) est choisi une seule fois sur
un ensemble borné, et ne grandit jamais avec de nouveaux cas de test — contrairement à une liste de
synsets. `agent/nodes/state_space_node_v3.py` : un seul signal `f_lex` remplace `f_srl`+`f_wn`,
règle de rejet simplifiée (`rejeté ⟺ f_lex == "ACTOR"`), suppression de la dépendance à spaCy
(`f_srl` abandonné avec elle).

**Résultat, sur 4 runs (3× llama-8b, 1× mistral-nemotron)** : nette amélioration par rapport au
Protocole 2. `company` rejeté 15/16 occurrences (le seul raté étant un candidat composite
`"company reviews"`, un problème d'étape A, pas de la couche B). **`employer` rejeté 4/4** — corrige
la Limite 2 du Protocole 2 (`f_srl≈0.67`, jamais assez bas pour franchir le seuil), sans aucun
réglage de seuil, juste par le changement de nature du signal. `director`, `school`, `student(s)`,
`employee(s)` rejetés systématiquement, y compris des candidats **jamais anticipés ni ajoutés à la
main** — preuve concrète de la généralité recherchée.

**Limite résiduelle identifiée : `person` non résolu.** `wn.synsets("person")[0]` (`person.n.01`,
"a human being") est lui-même étiqueté `noun.Tops` par WordNet, pas `noun.person` — une bizarrerie
de construction de la ressource, pas un défaut de la méthode. `self-employed person` et `person`
seul échappent donc au filtre.

---

## 20. Protocole 4 : vote sur les N premiers sens (tentative de correction de la limite `person`)

Deux corrections automatiques testées pour `person`, sans réintroduire de liste de mots :
- Vote majoritaire sur les 3 premiers sens (`top_n=3`) plutôt que le seul sens dominant.
- Pondération par fréquence d'usage sur tous les sens (`lemma.count()`).

**Résultat : aucune des deux ne corrige `person`** — vérifié empiriquement, aucun des sens de
"person" n'est jamais étiqueté `noun.person`, quelle que soit la fenêtre ou la pondération choisie.
Confirmé comme limite structurelle de WordNet, documentée en commentaire dans
`agent/nodes/state_space_node_v4.py`, plutôt que patchée par un cas spécial (qui aurait recréé le
problème de liste ouverte que le Protocole 3 cherchait justement à éliminer).

**Le vote top-3 reste adopté pour son bénéfice indépendant** : `director`, `customer`, `student`,
`worker` basculent proprement en ACTOR alors qu'ils ne l'étaient pas toujours avec le seul sens
dominant (Protocole 3) — amélioration générale au-delà du cas `person`, retenue malgré l'échec sur
le cas qui l'avait motivée.

---

## 21. Retour au Protocole 1 : définition V4, premier résultat de prompting nettement positif

Après trois versions de définition infructueuses sur le jugement direct par le LLM (V1 circulaire,
V2 flip test binaire insuffisant sur les cas mixtes, V3 règle de départage appliquée de façon
incohérente), une V4 a été écrite en empruntant le niveau d'exhaustivité d'un guide d'annotation
formel — inspirée d'un prompt de reformulation externe examiné en parallèle (`REFORMULATION_PROMPT_V2`,
non adopté tel quel : format activité/précondition/jalon jugé trop composite, cf. décision en
section suivante) : structure **Definitions** (entité, état explicite/implicite, acteur) +
**Classification rules** (règle de majorité d'occurrences plutôt qu'un test sur une seule occurrence,
checklist actionnable finale) + **Rules** de sortie tout aussi détaillées (jamais de sortie
minimaliste sous prétexte de "zero-shot" — zero-shot signifie seulement l'absence d'exemple, pas la
pauvreté des consignes).

**Résultat sur `job_application` : 9/9 runs, 3 familles de modèles indépendantes (llama-3.1-8b,
mistral-nemotron, gpt-oss-20b), zéro occurrence de `company` seul comme entité.** Premier résultat
de prompting robuste sur ce cas précis, après 3 tentatives négatives (V1/V2/V3) et 4 familles de
modèles ayant toutes montré le biais comme stable (§16). Le mécanisme observé va au-delà du simple
rejet : le LLM applique correctement la règle *"if the process tracks that received outcome, extract
that outcome as its own entity"* — `company_review`, `company_rating` apparaissent spontanément
comme entités séparées, exactement la décomposition visée depuis le début, produite sans
intervention de la couche symbolique.

**Généralisation partielle aux deux autres cas** : `work_accident` et `maternity_leave` propres sur
6-7 runs sur 9, avec des exceptions concentrées sur llama-3.1-8b (le modèle le plus faible) —
`company`/`employer`/`director`/`school` réapparaissent sur 2 de ses 3 runs, malgré une
`temperature=0` identique (cohérent avec le non-déterminisme de l'endpoint déjà documenté en
section 14). Plusieurs échecs indépendants de la qualité du prompt observés en parallèle : rate
limiting (429), troncature déjà documentée sur les modèles reasoning (`gpt-oss-*`).

**Bilan** : V4 est la variante de prompting la plus robuste testée à ce jour, sans être une garantie
absolue — la limite résiduelle se concentre sur le modèle le plus faible de l'échantillon, pas
distribuée aléatoirement. À valider par des répétitions supplémentaires sur gpt-oss-120b et llama-70b
avant de la considérer comme définitivement stable.

**Décision prise en parallèle** : le format externe qui a inspiré le niveau d'exhaustivité de V4
(`REFORMULATION_PROMPT_V2`, activités numérotées avec préconditions/jalons intégrés) n'a pas été
adopté comme remplacement de l'architecture `U`/`Pre` à deux étapes. Diagnostic : il réintroduit un
jugement composite en un seul appel (granularité d'activité + condition, simultanément) — exactement
le type de tâche qui a produit les échecs documentés en sections 3-15 — et autorise `OR` dans les
préconditions, alors que le pipeline reste volontairement additif (`AND` uniquement) pour ne pas
complexifier la propagation d'échec en aval (SAT vs parcours de graphe simple). Seule la technique
d'écriture (exhaustivité structurée définitions/règles) a été retenue et transposée à la définition
entité/acteur du Protocole 1, pas le format de sortie.

---

---

## 22. Prompt permissif discipliné (étape A) + Protocole 3 : élimination du bruit de format, meilleur résultat combiné à ce jour

**Bug identifié** : l'étape A du Protocole 2/3 produisait occasionnellement des états ou des noms de
candidats sous forme de clauses entières recopiées du texte plutôt que de labels courts — exemple
observé : `{"childcare_facility": ["taking a child to or collecting them from is considered part
of the work accident conditions"]}`. Ce n'est pas un problème de jugement entité/acteur (délégué à
la couche B), c'est un problème de discipline de format, indépendant, qui casse silencieusement
toute la chaîne en aval (validation symbolique, citation d'evidence, construction de graphe).

**Correctif** : réécriture complète de `agent/prompt/state_space_prompt_permissive.py`. Séparation
explicite de deux blocs indépendants — `_CANDIDATE_DEFINITION` (quoi inclure, toujours permissif)
et `_FORMAT_RULES` (comment nommer, jamais négociable, même pour un candidat incertain). Le contre-
exemple fautif réel (`childcare_facility`) est intégré tel quel dans le prompt comme exemple
`WRONG`/`RIGHT` explicite, plutôt qu'un exemple générique abstrait — cohérent avec l'efficacité déjà
observée des contre-exemples concrets ailleurs dans ce document. Règles ajoutées : longueur
maximale (1-4 mots), interdiction des amorces de clause (gérondif, "the fact that", "it is
considered"), snake_case, et une clause de repli explicite ("si vous ne pouvez pas compresser en
label court sans inventer, omettez plutôt que d'écrire une phrase"). Les 4 variantes (zero/one/two-
shot/cot) reçoivent le même niveau de détail — pas de version "allégée" pour le zero-shot.

**Résultat sur 12 runs (5 familles de modèles : mistral-medium-3.5-128b, llama-3.1-8b,
mistral-nemotron, gpt-oss-20b, gpt-oss-120b), 1125 candidats évalués au total** :

- **Zéro violation de format détectée** (aucun état ni candidat sous forme de phrase/clause), sur
  l'ensemble des 12 runs, toutes variantes et modèles confondus — le bug ciblé est éliminé.
- **309/1125 candidats (27%) rejetés par la couche B**, avec une cohérence inter-modèle remarquable :
  `school`, `professional_association`, `chamber_of_labour`, `trade_union_federation`, `guild`,
  `fire_brigade`, `employer`, `employee`, `labour_inspectorate`, `accident_insurance_provider`,
  `directorate`, `university`, `insured_party` rejetés de façon quasi identique à travers les 5
  familles de modèles sur `work_accident`.
- **`company`/`parent` sur `maternity_leave` corrigé de façon systématique** — confirme que la
  Limite 1 du Protocole 2 (§17, signal `f_srl` trompeur par manque d'occurrences) reste résolue par
  le Protocole 3 (lexnames) même avec ce nouveau prompt d'étape A.
- **`gpt-oss-120b`, auparavant instable (troncatures documentées en section 6), propre sur
  l'ensemble de ses runs** — aucune régression liée à la longueur accrue du nouveau prompt (2523 à
  4311 caractères selon la variante).

**Interprétation : la combinaison est meilleure que chacun des deux leviers pris isolément.**
Le prompt discipliné seul (sans couche B) resterait, comme V4 en Protocole 1, sensible au modèle
utilisé — un LLM pourrait toujours juger `company` comme légitime malgré de bonnes définitions
(observé sur llama-8b en Protocole 1, section 21). La couche B seule (Protocole 3 avec l'ancien
prompt permissif) perdait du signal sur les candidats mal formés (noms composites masquant le mot-
tête problématique, section 19). Ensemble : l'étape A produit des candidats propres et exhaustifs,
la couche B tranche le jugement difficile de façon reproductible et indépendante du modèle — c'est
la première configuration testée où la robustesse ne dépend plus principalement de la qualité
intrinsèque du LLM choisi.

**Point résiduel à vérifier** : `run_18`/`zero_shot`/`maternity_leave` rejette `information_gathering`
comme `ACTOR` — probablement un faux rejet (candidat qui ressemble à une activité, mal classé par le
lexname dominant de son mot-tête). À investiguer avant de considérer ce protocole comme stable.

---

## 23. Questions ouvertes / prochaines étapes

- **Investiguer le faux rejet `information_gathering`** (section 22) — vérifier si cest un cas isolé ou un mode déchec récurrent du signal lexname sur des candidats nommés comme des activités plutôt que des objets.
- **Le prompt permissif discipliné + Protocole 3 est la meilleure configuration testée à ce jour** — candidat naturel pour devenir le pipeline de référence.

- **V4 à confirmer sur davantage de modèles** — testée sur llama-8b, mistral-nemotron, gpt-oss-20b ;
  reste à tester sur gpt-oss-120b et llama-70b avant de la considérer comme la définition de
  référence du Protocole 1 (cf. section 21).
- **Limite résiduelle de V4 concentrée sur llama-3.1-8b** — non aléatoire, cohérente avec le pattern
  déjà observé (les petits modèles résistent davantage à toute intervention de prompting). Ne pas
  chercher une V5 tant que la cause précise (capacité de raisonnement conditionnel insuffisante vs.
  non-déterminisme de l'endpoint) n'est pas isolée.
- **`person` non résolu dans les Protocoles 3/4** — limite structurelle de WordNet documentée (le
  mot n'est jamais étiqueté `noun.person` lui-même), acceptée comme telle plutôt que patchée par un
  cas spécial qui romprait la généralité recherchée (cf. section 20).
- **Décider si le Protocole 1 (prompting seul, désormais avec V4) ou le Protocole 3 (validation
  symbolique par lexnames) devient le pipeline de référence** — les deux montrent maintenant de bons
  résultats sur `company`/`employer`, obtenus par des mécanismes différents (meilleure définition vs.
  filtrage externe indépendant du LLM). Un test croisé (V4 comme étape A du Protocole 3, au lieu du
  prompt permissif actuel) pourrait combiner les deux gains plutôt que de choisir entre eux.
- **Limite 1 du Protocole 2 (dépendance à la fréquence)** — piste de correction à évaluer sans
  précipitation : pondérer la confiance de `f_srl` par le nombre brut d'occurrences plutôt que par
  le seul ratio, ou basculer vers `f_wn` seul quand `f_srl` repose sur moins de 2-3 occurrences.
  Ne pas re-tester NLI comme substitut (cf. section 18, décision prise).
- **Limite 2 du Protocole 2 (seuil trop strict pour `employer`)** — résolue indirectement par le
  Protocole 3 (lexnames), qui n'a plus ce seuil. Non prioritaire de la corriger dans le Protocole 2
  lui-même si le Protocole 3/4 devient la référence.
- **Qualité de l'étape A reste un facteur, même en Protocole 2/3** — `zero_shot` sans exemple produit
  des candidats composites (`"company reviews"`) qui contournent le filtre B en dissimulant le mot-
  tête problématique ; `work_accident` continue à produire échecs de parsing et sur-fragmentation,
  hérités du Protocole 1, hors du périmètre de la couche B.
- Tester si l'étape A permissive (Protocole 2/3) bénéficierait elle-même d'être reformulée avec le
  niveau d'exhaustivité de V4, plutôt que le prompt permissif minimal actuel.
- Tester le Protocole 2/3 sur gpt-oss-120b (déjà testé en Protocole 1) pour vérifier si les limites
  caractérisées (section 17) et les succès (section 20) se reproduisent à l'identique.
- Implémenter la boucle HITL différée pour commencer à accumuler un gold set validé — les rapports
  de validation (`validation_report`, Protocoles 2/3) et les citations `evidence` (Protocole 1)
  offrent tous deux une base de traçabilité utile pour cette revue humaine.
- **Correctif prioritaire (Protocole 1, toujours en attente)** : ajouter un marqueur de fin explicite
  (type `FINAL_ANSWER:`) au prompt de conflict resolution pour éliminer les échecs de parsing dus à
  du texte parasite après le JSON (cf. section 15) — pertinence réduite si le Protocole 3 devient le
  pipeline principal, à trancher.
- **`validate_grounding` à renforcer** (Protocole 1) : ne détecte pas la recombinaison trompeuse de
  fragments individuellement authentiques.
- Intégrer Gemini comme provider dans `base_llm.py` — Gemini reste le seul modèle observé (test
  manuel hors pipeline) à éviter nativement le piège acteur=entité en zero-shot simple ; à vérifier
  si V4 (Protocole 1) ou le Protocole 3 rendent cette différence de modèle moins déterminante.
- **Positionnement littérature confirmé par Hermes (USENIX Security 2024) et RFCNLP (IEEE S&P
  2022)** : même un pipeline supervisé, lourdement outillé (grammaire dédiée, ~2800h d'annotation
  humaine pour Hermes) ne résout pas l'erreur d'extraction à la source — les deux déplacent la
  correction vers une vérification a posteriori contre un référentiel externe (tests de conformité
  3GPP pour Hermes). Nous n'avons pas d'équivalent à ce référentiel pour du texte métier généraliste,
  ce qui rend la case vide ciblée par la thèse (vérification de fidélité sans gold standard
  externe préexistant) plus difficile, et confirme sa pertinence.