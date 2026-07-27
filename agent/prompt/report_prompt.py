# Explication en langage metier d'un ecart (VIOLATED) ou d'un cas non verifiable
# (UNRESOLVABLE) detecte par l'alignement. Un appel par cause racine -- jamais un rapport
# complet genere en un seul appel, pour eviter qu'un seul gros appel fusionne les categories ou
# derive vers un ton conclusif global sur l'ensemble du processus.
#
# Contrainte stricte, actee explicitement dans la conception : ne jamais utiliser le vocabulaire
# interne du pipeline (entity.state, AND/OR, SATISFIED/VIOLATED/UNRESOLVABLE, "Pre", "U",
# "matching", "noeud", "graphe") -- uniquement le vocabulaire metier du process (noms
# d'activites BPMN, citation du texte source). La personne visee connait les processus metier,
# pas notre formalisme interne.
#
# Regle ajoutee apres observation concrete (invention) : la citation attachee a une cause
# racine justifie TOUJOURS pourquoi la cible en aval a besoin d'une precondition, jamais ce que
# cette precondition signifie precisement -- elle vient de l'entree de Pre pour la cible, pas
# d'une description de l'etat manquant lui-meme. Sans regle explicite, le LLM comble cet ecart
# en deduisant un sens plausible a partir du nom de l'etat manquant, puis presente cette
# deduction comme si la citation la confirmait.
#
# Deuxieme regle ajoutee apres une observation distincte (traduction) : la citation elle-meme
# doit rester verbatim dans sa langue source, jamais traduite ni paraphrasee -- observe
# concretement quand le rapport etait encore redige en francais (v1 de ce fichier) : "A job
# interview can be negotiated." reproduit comme "Une entretien d'embauche peut etre negocie"
# (traduit, et fautif en plus). "Verbatim" dans les regles ci-dessous ne visait que le contenu,
# pas explicitement la langue -- ecart corrige en le rendant explicite dans la regle, et ce
# malgre le changement de langue du rapport lui-meme ci-dessous (le principe reste valide quelle
# que soit la langue de redaction : ne jamais reformuler une citation).
#
# CHANGEMENT DE LANGUE (cette revision) : le contenu envoye au LLM (regles + gabarits) passe du
# francais a l'anglais -- ce fichier etait le SEUL prompt de tout le pipeline redige en francais
# (state_space_prompt_permissive.py, precondition_prompt.py, precondition_prompt_retry.py sont
# tous en anglais, seuls leurs commentaires developpeur sont en francais, meme convention gardee
# ici). Le rapport final est donc desormais redige en anglais. report_node.py a ete mis a jour
# en consequence pour que les chaines generees PAR LE CODE (placeholders, texte d'agregation
# "cette meme cause affecte aussi", justification du verdict) restent dans la meme langue que le
# texte produit par le LLM -- un rapport qui melangerait les deux serait pire que
# l'incoherence de depart.

_REPORT_RULES = """Rules:
- Never use internal technical jargon: no "entity.state", "AND"/"OR", "SATISFIED"/"VIOLATED"/"UNRESOLVABLE", "Pre", "U", "matching", "node", "graph", or "state" in its technical sense.
- Rely only on the information given below. Never invent a fact, an activity, or a cause -- if information is missing, say so explicitly rather than guessing or filling the gap.
- The quote provided only justifies the fact that a condition is required before this step -- it does not necessarily describe what that condition precisely means. Never present the quote as explaining or confirming the meaning of the missing condition; if its content does not explicitly describe that condition, cite it only to show that a requirement exists, without rephrasing what it would mean.
- Never use a connector that invents a logical link not explicitly given (e.g. "which suggests that", "which indicates that", "which implies that", "as shown by") between the quote and the missing condition -- describe the two facts separately if their link is not explicit in the quote itself.
- The quote must always be reproduced word for word, exactly as given -- NEVER translate or paraphrase it, even if it happens to be in a different language than the rest of your answer. A quote stays exactly as written, in full, between quotation marks, without any modification.
- One to two short sentences, never a long paragraph.
- Use the process activity names exactly as given, never a rewording that changes their meaning.
- Never conclude on the overall conformance of the process -- describe only this specific case, nothing more."""


REPORT_PROMPT_VIOLATED = """The source text requires that the activity "{{term_label}}" must already have taken place before the activity "{{target_label}}" can occur.

Quote from the source text associated with this requirement, TO BE REPRODUCED WORD FOR WORD WITHOUT TRANSLATING IT (justifies that a condition is required, not necessarily what it precisely means): "{{quote}}"

In the observed process, both activities do exist, but nothing in the process flow makes "{{term_label}}" actually precede "{{target_label}}".

{rules}

Write the explanation of this gap for a non-technical reader, familiar with business processes but not with our internal tooling.""".format(rules=_REPORT_RULES)


REPORT_PROMPT_UNRESOLVABLE = """The source text requires that a specific condition already be met before the activity "{{target_label}}" can occur: {{missing_description}}

Quote from the source text associated with this requirement, TO BE REPRODUCED WORD FOR WORD WITHOUT TRANSLATING IT (justifies that a condition is required, not necessarily what it precisely means): "{{quote}}"

This condition could not be associated with any clear activity in the observed process -- it is therefore impossible to verify whether the requirement is met or not, either way.

{rules}

Write the explanation of this case for a non-technical reader, being explicit that this is neither a confirmation nor a refutation of conformance -- just an inability to verify.""".format(rules=_REPORT_RULES)


# Ajout (apres decouverte du contresens NOT -- cf. report_node.py) : une precondition niee
# ("NOT X") n'a pas seulement besoin d'un label different, elle a besoin d'une PHRASE
# differente. REPORT_PROMPT_VIOLATED/UNRESOLVABLE affirment tous deux, dans leur formulation
# meme, que l'activite manquante DEVAIT avoir eu lieu -- correct pour un terme positif, un
# contresens pour un terme negatif (le texte source exige au contraire qu'elle NE SE SOIT PAS
# produite). Utiliser le meme gabarit pour les deux aurait masque le probleme derriere une
# etiquette propre plutot que de le corriger : le lecteur aurait lu une explication grammaticalement
# correcte mais affirmant l'inverse de ce que le texte dit reellement. Deux gabarits distincts,
# jamais un parametre conditionnel a l'interieur d'un seul prompt (le pipeline evite deja ce
# genre de branchement implicite ailleurs -- precondition_prompt.py separe par exemple ses
# variantes zero/few/two-shot plutot que de les fusionner en un seul prompt parametrise).

REPORT_PROMPT_VIOLATED_NEGATED = """The source text requires that the activity "{{term_label}}" must NOT have taken place before the activity "{{target_label}}" can occur.

Quote from the source text associated with this requirement, TO BE REPRODUCED WORD FOR WORD WITHOUT TRANSLATING IT (justifies that an absence is required, not necessarily what it precisely means): "{{quote}}"

In the observed process, both activities do exist, but "{{term_label}}" consistently occurs before "{{target_label}}" -- never otherwise.

{rules}

Write the explanation of this gap for a non-technical reader, familiar with business processes but not with our internal tooling. Be explicit that the text's requirement is an ABSENCE requirement, not a presence requirement.""".format(rules=_REPORT_RULES)


REPORT_PROMPT_UNRESOLVABLE_NEGATED = """The source text requires that a specific condition must NOT be met before the activity "{{target_label}}" can occur: the absence of {{missing_description}}

Quote from the source text associated with this requirement, TO BE REPRODUCED WORD FOR WORD WITHOUT TRANSLATING IT (justifies that an absence is required, not necessarily what it precisely means): "{{quote}}"

This condition could not be associated with any clear activity in the observed process -- it is therefore impossible to verify whether this absence is respected or not, either way.

{rules}

Write the explanation of this case for a non-technical reader, being explicit that this is neither a confirmation nor a refutation of conformance -- just an inability to verify an ABSENCE requirement.""".format(rules=_REPORT_RULES)