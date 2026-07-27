# TGMS — du formalisme au code validé

Document de synthèse : capture le jalon complet allant de la question de nouveauté ("est-ce de
l'ASP déguisé en GSM ?") jusqu'à un solveur réellement intégré dans le pipeline et validé sans
exception sur des données de production réelles. Cette preuve n'existe pour l'instant que dans
l'historique de conversation — ce document la rend citable et persistante avant qu'elle ne se
perde.

---

## 1. La question à laquelle ce jalon répond

*"Un reviewer pourrait dire qu'on atteindrait les mêmes résultats avec une composition
d'outils existants (ASP, seuillage). Comment se défendre, malgré le fait qu'on montre
clairement l'utilité de TGMS ?"*

Réponse structurée en trois temps, chacun maintenant étayé par une pièce concrète :

1. **Concéder honnêtement ce qui n'est pas nouveau** — le calcul de base (guards AND/OR purs,
   atteignabilité) coïncide, dans le cas dégénéré à un seul graphe, avec la sémantique de point
   fixe GSM établie par Damaggio, Hull et Vaculín (BPM 2011). Énoncé comme théorème de
   réduction, pas caché.
2. **Isoler précisément ce qui ne se réduit à rien de connu** — la correspondance
   $(\varphi, c)$ entre deux graphes indépendamment dérivés, absente de GSM (schéma unique),
   d'ASP (pas de correspondance graduée sur le grounding) et de la logique floue (qui grade la
   vérité d'une formule, pas la légitimité d'une identification entre vocabulaires).
3. **Prouver des propriétés qui n'existent que pour cet objet précis** — deux théorèmes de
   monotonie, maintenant confirmés empiriquement sans exception (section 4).

---

## 2. Ce qui est prouvé formellement (note dédiée : `tgms_formalism.pdf`)

| Résultat | Statut |
|---|---|
| Théorème de réduction vers la sémantique de point fixe GSM (cas dégénéré) | Prouvé |
| Théorème de monotonie en confiance — **SATISFIED invariant à τ**, pas seulement jamais transformé en VIOLATED | Prouvé (version renforcée, cf. section 3) |
| Théorème de monotonie en enrichissement du graphe process | Prouvé |
| Positionnement explicite contre GSM, ASP, logique floue, LLM-as-judge | Fait, un paragraphe chacun |
| Paragraphe XOR réservé (intégration conditionnée à la preuve empirique) | Fait |

## 3. Un écart trouvé et corrigé — la preuve n'est pas restée statique

En construisant le solveur générique (`tgms_solver.py`) à partir de la note, le premier jet
faisait dépendre le test d'ancrage (`missing`) du seuil de confiance τ — cohérent avec la
Definition 3bis telle qu'écrite initialement, mais **incohérent avec le comportement réel de
`alignment_node.py`**, où seule la décision de rétrograder un candidat VIOLATED dépend de τ, le
test d'ancrage lui-même étant inconditionnel.

Le test d'équivalence stricte (section 4) a détecté l'écart immédiatement, sur le cas réel qui
avait motivé la démotion (`resources.gathered`, confiance 0.0814) : le solveur corrigé
distinguait `UNRESOLVABLE` (missing) de l'ancien comportement attendu. Corrigé dans le code
*et* dans la note (`tgms_formalism.pdf`, Remark "Where τ actually enters — corrected from an
earlier draft") — la version corrigée du théorème est **plus forte**, pas plus faible : un
verdict SATISFIED est invariant à τ, propriété qu'on n'aurait pas pu énoncer sous l'ancienne
définition.

C'est un point à assumer explicitement plutôt qu'à minimiser : la preuve formelle et
l'implémentation se sont mutuellement corrigées, dans les deux sens, via un instrument de
vérification (le test d'équivalence) construit précisément pour ça.

## 4. Validation empirique — équivalence stricte, trois niveaux croissants

**Niveau 1 — fixtures construites à la main.** Cas synthétiques couvrant l'agrégation AND/OR,
comparés champ par champ (statut, `missing`, `low_confidence`) entre l'implémentation figée de
référence et le nouveau solveur.

**Niveau 2 — cas réels isolés.** Le cas v1 qui a motivé la démotion (score 0.0814,
`materials.farmed`/`resources.gathered`) et deux cas v2 confirmés VIOLATED après démotion
(`E_j02/0`, `V_k09/2`) — contre-test anti-sur-correction : la démotion ne doit pas rétrograder
des VIOLATED légitimes.

**Niveau 3 — production réelle, 153 mutants.** Rejeu complet de l'étude de perturbation avec
`alignment_node.py` refactorisé (délègue à `tgms_solver.py`), contre les résultats déjà publiés
de l'ancienne implémentation. **Résultat : zéro écart, entrée par entrée, sur les 153 mutants**
— rappel identique à la décimale près sur les trois modèles et les six opérateurs, taux de
faux positifs `forbidden` identique (0/102 avant et après), angle mort `blind_spot` identique
(0/51 avant et après).

C'est le niveau de preuve le plus fort disponible sans toucher au corpus complet : de vrais
appels d'embedding, le vrai câblage `graph.py → alignment_node.py → tgms_solver.py`, pas
seulement des fixtures isolées.

**Portée à noter honnêtement** : ce test de niveau 3 porte sur le manifeste de mutants
*biaisé* (le biais de ciblage n'était pas encore corrigé au moment de ce test) — ce qui ne
l'invalide pas comme test d'équivalence (il compare deux implémentations sur les *mêmes*
entrées, peu importe la qualité de ces entrées), mais signifie qu'il faudra le rejouer une fois
sur le manifeste corrigé, comme simple contrôle de non-régression, pas pour re-découvrir un
écart.

## 5. Pourquoi ceci répond à l'objection du reviewer, concrètement

| Argument de l'objection | Réponse, maintenant avec preuve |
|---|---|
| "Une composition atteindrait les mêmes résultats" | Possible sur l'échantillon testé — mais aucune composition ne donne un théorème de réduction vers Damaggio et al., ni les deux garanties de monotonie prouvées pour *tout* $\tau$/$G_{\mathrm{proc}}$, pas seulement les cas observés. |
| "Le solveur est juste du code, pas un objet formel" | L'objet formel (types, solveur générique) existe indépendamment de l'intégration pipeline (`tgms_solver.py` sans import `agent.*`), testé en équivalence stricte contre le comportement réel. |
| "Les preuves papier ne correspondent peut-être pas au code" | Un écart réel existait, trouvé par le test d'équivalence, corrigé dans les deux sens — documenté, pas caché. |
| "Composer serait plus simple" | Composer n'élimine pas le besoin de définir l'interface confiance/logique crisp — ça la cache dans du code de glue non prouvé, au lieu de la formaliser. |

## 6. Ce qui reste à faire avant que ce jalon soit définitivement clos

1. Rejouer le test d'équivalence de niveau 3 sur le manifeste de mutants corrigé (union des
   activités guard-pertinentes, cf. `dataset_run_v2_observations.md` section 8.2) — attendu :
   même conclusion (zéro écart), simple contrôle de non-régression.
2. Packaging standalone du solveur (`tgms/` en package pip-installable, déjà prototypé,
   volontairement mis de côté jusqu'à validation complète — cf. discipline du projet : jamais
   de packaging avant que tout soit validé).
3. Seconde relecture des preuves formelles par un tiers avant intégration finale au papier.