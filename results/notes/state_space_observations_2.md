# State Space — Observations 2

Journal de suivi des observations empiriques et des décisions de conception sur la couche state
space (`state_space_prompt_permissive.py` + `state_space_node_v3.py`, Protocol 3.2 — post-
découplage `exclusive_groups` en second appel LLM). Objectif : ne rien retrancher, tracer ce qui
a été tranché et pourquoi, avant de passer à la couche précondition.

Principe directeur, rappelé explicitement car il conditionne toutes les décisions ci-dessous :
**le pipeline doit rester agnostique au cas d'usage et au backbone.** Aucune correction ne doit
être calée sur le pattern d'échec d'un modèle précis ni sur le texte job_application utilisé
comme banc d'essai. On cherche un comportement récurrent et une robustesse de conception, pas un
score parfait sur un backbone donné — **on ne peut pas sauver tous les modèles**, et ce n'est pas
l'objectif.

---

## 1. Ce qui est acquis (validé sur plusieurs runs réels, pas seulement en test unitaire)

| Point | Statut | Preuve |
|---|---|---|
| A.1 — lexname top-3, unanimité sur sens informatifs seulement | ✅ Stable | `company` → ACTOR sur 100% des runs (17/17 backbones cumulés) |
| A.2 — `head_used` tracé | ✅ Stable | Présent sur tous les runs |
| A.6 — Rule 2 resserrée (modificateurs génériques) | ✅ Comportement documenté et assumé | `3d_model`/`work_accident` → `ambiguous_not_merged`, testé |
| A.8 — `__meta__` agrégat INDETERMINATE | ✅ Stable | Présent sur tous les runs |
| Sanitize des références fantômes (`exclusive_groups` → état inexistant) | ✅ **Confirmé en conditions réelles** | `nemotron-3-nano-30b` (doc 48) a halluciné `company_rating.continued` → rejeté avec raison exacte, sans intervention manuelle |
| Découplage Step A / Step A2 (deux appels LLM séparés) | ✅ Gain net sur l'adoption | 2/7 → 6/10 backbones utilisant `exclusive_groups` |
| `concurrent_with` | ✅ Aucun faux positif observé sur 3 runs cumulés | Reste vide sauf quand un marqueur textuel explicite existe |

---

## 2. Décision tranchée dans cet échange : le bruit résiduel sur `exclusive_groups` est accepté, pas poursuivi

### Observation
Sur le run le plus récent (10 backbones), le découplage a résolu le problème d'adoption mais
**pas entièrement le problème de sur-généralisation** :

- 1 backbone (`gpt-oss-20b`) : capture exacte de la bonne branche, avec citation complète de la
  clause "unless" comme preuve. Résultat cible.
- 1 backbone (`gpt-oss-120b`) : paire discutable (`permanent`/`ended` — plausible mais imprécis,
  "ended" ressemble plus à une conséquence qu'à une alternative).
- 2 backbones (`llama-3.3-nemotron-super-49b`, `mistral-nemotron`) : paires défendables mais
  imprécises (captent `temporary`/`permanent` ou `probationary`/`permanent` plutôt que la vraie
  branche liée à la note ≤ C).
- **2 backbones (`llama-3.1-8b-instruct`, `nemotron-3-nano-30b`) reproduisent, à l'identique, le
  pattern de sur-généralisation qu'on avait corrigé pour `mistral-nemotron` en révision 2** :
  `[confirmed, rated]` marqué exclusif alors que c'est une séquence — le contre-exemple précis
  que la règle renforcée cite littéralement.

### Décision
**On n'ira pas plus loin sur le prompt pour fermer ce résidu.** Deux raisons, actées ensemble :

1. **Principe d'agnosticité** : corriger davantage pour rattraper spécifiquement
   `llama-3.1-8b-instruct` ou `nemotron-3-nano-30b` reviendrait à sur-ajuster le prompt à des
   échecs de modèles précis plutôt qu'à des lacunes de conception générales — contraire au
   principe déjà posé dans cet échange ("le code doit être agnostique au cas d'usage et au
   modèle").
2. **Le garde-fou structurel absorbe déjà le risque le plus grave** (référence à un état
   inexistant) — confirmé en conditions réelles ce run-ci. Le risque résiduel (mauvaise paire,
   mais structurellement valide) est un problème de **fiabilité du signal**, pas d'intégrité du
   pipeline.

### Conséquence directe pour la couche précondition (à ne pas oublier en la concevant)
**`exclusive_groups` doit être traité comme un signal probable mais bruité, jamais comme une
vérité automatiquement fiable.** Estimation de fiabilité sur ce run : environ 1 paire correcte
franche sur 6 (`gpt-oss-20b`), le reste allant de l'imprécis au franchement faux. Si la couche
précondition (ou une future vérification de cohérence de branche, cf. point B.10 des corrections
précédentes) s'appuie sur `exclusive_groups` pour détecter des incohérences entre deux
préconditions, elle doit prévoir une marge d'erreur ou un mécanisme pour ignorer/contredire un
`exclusive_groups` mal posé — ne jamais lui faire une confiance absolue implicite.

**Point ouvert, à trancher plus tard, pas maintenant** : est-ce que la couche précondition peut
*elle-même* redériver l'exclusivité de façon plus fiable, à partir de la structure causale
propre (deux états chacun gardés par des conditions qui s'excluent logiquement), sans dépendre
du signal `exclusive_groups` du state space ? Si oui, `exclusive_groups` redevient un indice de
recoupement plutôt qu'une source de vérité — hypothèse à vérifier une fois le fichier précondition
en main, pas à assumer maintenant.

---

## 3. Autres observations mineures notées sur ce run, non traitées (statut : à surveiller)

- **Renommage silencieux de candidats singleton en cas de normalisation** (ex. `company_reviews`
  → `company_review` chez `nemotron-3-nano-30b`, `job_offers` → `job_offer` sur le même run) :
  cohérent avec le design (Rule 1 ne fusionne que s'il y a plusieurs membres du même groupe
  normalisé ; un singleton renommé silencieusement n'entre pas dans `merged`), mais reste
  potentiellement surprenant à l'audit humain. Pas un bug — noté pour référence.
- **Chaînes de `concurrent_with` mal alignées après filtrage ACTOR** (`nemotron-3-nano-30b` :
  `company` était `concurrent_with: [job_application]` mais `company` est filtré comme ACTOR ;
  `job_application` gardait pourtant une référence propre à `job_offer`, pas à `company` — donc
  pas de fuite observée ici, mais le cas mériterait un test dédié : que se passe-t-il si l'ENTITÉ
  qui reste après filtrage pointait, elle, vers une entité qui a été filtrée ? Actuellement le
  resolver ne garde que les références vers des entités présentes dans `deduped` — donc
  structurellement déjà couvert. Pas d'action requise, juste noté comme confirmé plutôt que
  supposé.)
- **Deux backbones sur 10 n'ont pas produit `job` ou `job_offer` du tout dans leur state_space
  final** (`qwen3.6-27b` a perdu `company` — normal, ACTOR filtré, mais aussi n'a jamais eu de
  candidat représentant la branche permanent/continued). Confirme, une fois de plus, la variance
  inter-backbone déjà documentée avant le découplage — non spécifique à `exclusive_groups`.

---

## 4. Prochaine étape

Fichier précondition (`precondition_node_with_retry.py` + prompts associés) à examiner avec, en
tête, la contrainte du point 2 ci-dessus : ne pas supposer `exclusive_groups` fiable par défaut.