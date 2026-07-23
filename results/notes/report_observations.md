# Observations — Report Module

Notes de conception et de test pour `report_node.py`, le dernier node du pipeline : traduit la
sortie de `check_alignment()` en langage métier pour l'utilisateur final. Dernier maillon sans
validateur mécanique en aval — ce que le LLM écrit part directement à l'utilisateur.

---

## 1. Principe retenu : structurer et expliquer, jamais conclure

Rejet explicite de l'idée d'un "verdict" en prose libre. Un verdict reproduirait, sous une
forme moins visible, exactement le risque déjà écarté pour le score de conformité
(`alignment_observations.md`, section 3.1) — une agrégation informelle et non auditée,
potentiellement moins rigoureuse qu'un score explicitement refusé.

**Structure fixe, trois sections, ordre jamais réordonné par le LLM :**
1. **Écarts** (`VIOLATED`) — en premier, ce qui compte le plus.
2. **Non vérifiable** (`UNRESOLVABLE`) — section séparée, jamais mélangée à la précédente ;
   formulée comme "ni confirmation ni infirmation", pas comme un problème de même nature qu'un
   écart réel.
3. **Conforme** (`SATISFIED`) — en dernier, groupé et bref, pas développé phrase par phrase.

**Le LLM structure et explique, il ne conclut jamais lui-même** — pas de jugement de conformité
globale généré librement, uniquement la formulation en langage clair de faits déjà tranchés par
`check_alignment()`.

**La distinction à trois voies doit survivre dans le texte** — risque nommé explicitement dès
la conception : qu'une reformulation fusionne `VIOLATED` et `UNRESOLVABLE` sous un vague
"problème détecté". Les prompts l'interdisent par construction (deux templates séparés,
formulations différentes).

**Le détail brut (`piece_jointe_alignement_brut`) est toujours joint**, jamais remplacé par la
prose — permet à qui doute de recroiser directement contre `check_alignment()`.

---

## 2. Granularité : par cause racine, pas par item

### 2.1 Premier design : un appel LLM par terme non satisfait

Version initiale : une explication par terme en échec. Problème découvert à l'usage, pas
anticipé en conception : plusieurs échecs partagent souvent la même cause racine (ex.
`job_offer.sent`/`job_offer.received` jamais couverts, dont dépendent en cascade
`job_interview.negotiated`, `probation_phase.entered`, etc.) — le rapport produisait alors 6
paragraphes quasi redondants là où il n'y a que 2-3 causes distinctes, noyant l'information la
plus utile (la vraie cause commune) sous la répétition.

### 2.2 Fix : regroupement par cause racine avant génération

`find_root_term()` remonte récursivement dans `alignment` lui-même (pas une heuristique
textuelle) : si le terme fautif d'un échec est lui-même une cible vérifiée ailleurs (donc
lui-même en échec), la vraie cause est encore plus en amont — jusqu'à un terme qui n'est plus
lui-même une cible contrôlée (une feuille), ou jusqu'à ce que plusieurs causes distinctes
convergent (auquel cas on s'arrête plutôt que de fusionner à tort deux problèmes indépendants).

Un seul appel LLM par cause racine distincte. La liste des cibles affectées par une même cause
est ajoutée **par du code**, jamais par le LLM (`"Cette meme cause affecte aussi : ..."`) —
cohérent avec le principe déjà appliqué ailleurs dans le pipeline (génération locale par LLM,
structure/agrégation déterministe).

Vérifié sur un cas reproduisant exactement un scénario réel observé (6 vérifications en échec,
2 causes racines réelles) : correctement réduit à 2 explications, pas 6.

---

## 3. Bugs trouvés et corrigés, dans l'ordre où ils sont apparus

### 3.1 `"None"` littéral dans le texte destiné à l'utilisateur

Cause : `entry.get("quote")` renvoyait `None` silencieusement quand `graph_construction`
n'avait pas encore été régénéré avec le fix `quote` (`alignment_observations.md`, section 2.1),
et ce `None` Python s'est retrouvé injecté tel quel dans le prompt puis répété par le LLM.
Incohérent avec la discipline "jamais un rejet silencieux" appliquée partout ailleurs dans le
pipeline — un oubli de ma part, pas une décision.

**Fix** : avertissement explicite (`WARNING: no quote for target ...`) dès qu'un `quote` est
manquant, placeholder `"(aucune citation disponible)"` substitué avant tout envoi au LLM — plus
jamais de `None` littéral possible dans le texte généré.

**Diagnostic downstream, pas un bug en soi** : la persistance du warning sur plusieurs
itérations a révélé un problème de séquencement pur (runs `alignment` générés avant la
régénération de `graph_construction`, jamais un bug de propagation) — confirmé en inspectant
directement `precondition_extraction_with_retry` → `graph_construction` → `alignment` à chaque
étape plutôt que de deviner où était la rupture.

### 3.2 Labels concaténés illisibles

`label_for_state()` concaténait tous les labels d'activité BPMN matchés à un même état avec
`" / "` (ex. *"company want you to work for them / continue to receive job offers / receive
new potential job offers"*, lu comme une seule activité par un lecteur non technique).

**Fix** : un seul label retenu, celui du score de matching le plus élevé — les autres
candidats restent visibles dans la pièce jointe brute, jamais dans la prose.

### 3.3 Lien causal inventé entre citation et condition manquante

Le plus sérieux des trois, une vraie violation de la règle "n'invente aucun fait, aucune
cause" déjà présente dans le prompt initial — la règle seule n'a pas suffi.

**Symptôme observé** : le LLM déduisait un sens plausible à partir du nom de l'état manquant
(ex. `job_application.confirmed` → "confirmation de l'entreprise reçue"), puis présentait la
citation associée comme si elle confirmait cette déduction — alors que la citation portait sur
un fait différent (ex. *"You have to regularly report..."*, sur le fait de **reporter**, pas de
**recevoir une confirmation**).

**Cause structurelle** : la citation attachée à une cause racine justifie toujours pourquoi la
cible *en aval* a besoin d'une précondition (elle vient de l'entrée `Pre` de la cible), jamais
ce que la précondition *elle-même* signifie précisément. Rien n'empêchait le LLM de combler cet
écart par une inférence plausible.

**Fix, en deux règles ajoutées** :
1. La citation justifie seulement *qu'*une condition est exigée, jamais présentée comme
   expliquant *ce qu'*elle signifie.
2. Interdiction explicite des connecteurs qui inventent un lien logique non donné
   ("ce qui suggère que", "ce qui indique que", "ce qui impose que", "comme le montre").

**Vérifié sur données réelles, pas seulement en théorie** : comparaison avant/après sur les
mêmes cas — le nouveau texte dit *"la citation ... montre qu'une condition est exigée, mais..."*
au lieu de présenter la citation comme description directe de la condition. Un cas ambigu
inspecté séparément (`"stay to work permanently"` / citation *"If a job becomes permanent..."*)
confirmé comme légitime, pas une résurgence du bug : la citation et l'état manquant y
correspondent réellement, littéralement, contrairement aux cas corrigés.

**Statut** : fix par prompt, pas par contrôle mécanique — aucune garantie que le comportement
ne réapparaisse pas sur d'autres cas. Si ça se reproduit, la vraie solution proposée mais non
implémentée serait un contrôle après coup (vérifier que les mots-clés de la condition manquante
apparaissent dans la citation, sinon forcer une formulation neutre), pas une nouvelle itération
de prompt.

**Régression mineure notée en même temps, non corrigée** : sur au moins un cas, le nouveau
texte omet de nommer explicitement la condition manquante (saute directement à la citation) —
moins informatif qu'avant, mais une omission, pas une invention. À surveiller si ça se répète.

---

## 4. Modèle rédacteur vs modèle sous test

Point de confusion identifié et corrigé : le modèle qui rédige le rapport (`report_node.py`,
`get_llm(temperature=0)`) et le modèle dont on évalue l'extraction (`graph_construction`, ex.
`meta/llama-3.1-8b-instruct`) sont deux rôles indépendants, tournant potentiellement sur des
modèles différents, sans que ce soit affiché nulle part par défaut — un défaut hérité
silencieusement de `base_llm.py` (`MODEL_NAME = "openai/gpt-oss-120b"`), pas un choix explicite.

**Fix** : le modèle rédacteur est maintenant fixé explicitement (`REPORT_MODEL` en haut du
`__main__`) et affiché dans les logs à chaque exécution.

---

## 5. Limites connues, non résolues

- **`VIOLATED` jamais rendu en pratique** — comme documenté dans `alignment_observations.md`
  (section 4.3), le prompt `REPORT_PROMPT_VIOLATED` n'a encore jamais été exercé sur une vraie
  donnée, seulement sur des cas synthétiques. Son comportement réel sur un cas `VIOLATED` réel
  reste à vérifier une fois le test de contrôle de l'alignement effectué.
- **`status_for_root()` — heuristique de repli non stress-testée** : détermine le statut
  représentatif d'une cause racine en cherchant d'abord un item où `term == root` dans la liste
  des échecs, avec repli sur le statut de `alignment[root]` si absent. Fonctionne sur tous les
  cas réels rencontrés jusqu'ici, mais n'a pas été testée sur une structure de cascade plus
  profonde ou plus irrégulière que celles observées.
- **Portée** : mêmes limites que `alignment_node.py` — un seul cas, deux modèles BPMN testés.