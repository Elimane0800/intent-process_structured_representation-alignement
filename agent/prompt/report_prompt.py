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
# Deuxieme regle ajoutee apres une observation distincte (traduction) : le rapport est redige en
# francais, mais la citation elle-meme doit rester verbatim dans sa langue source (le texte
# d'intention est en anglais) -- observe concretement : "A job interview can be negotiated."
# reproduit comme "Une entretien d'embauche peut etre negocie" (traduit, et fautif en plus).
# "Verbatim" dans les regles ci-dessous ne visait que le contenu, pas explicitement la langue --
# ecart corrige en le rendant explicite.

_REPORT_RULES = """Regles :
- N'utilise jamais de jargon technique interne : pas de "entity.state", "AND"/"OR", "SATISFIED"/"VIOLATED"/"UNRESOLVABLE", "Pre", "U", "matching", "noeud", "graphe", "etat" au sens technique.
- Base-toi uniquement sur les informations fournies ci-dessous. N'invente aucun fait, aucune activite, aucune cause -- si une information manque, dis-le explicitement plutot que de deviner ou de combler.
- La citation fournie justifie seulement le fait qu'une condition est exigee avant cette etape -- elle ne decrit pas forcement ce que cette condition signifie precisement. Ne presente jamais la citation comme expliquant ou confirmant le sens de la condition manquante ; si son contenu ne decrit pas explicitement cette condition, cite-la uniquement pour montrer qu'une exigence existe, sans reformuler ce qu'elle signifierait.
- N'utilise aucun connecteur qui invente un lien logique non donne explicitement (ex. "ce qui suggere que", "ce qui indique que", "ce qui impose que", "comme le montre") entre la citation et la condition manquante -- decris les deux faits separement si leur lien n'est pas explicite dans la citation elle-meme.
- La citation doit toujours etre reproduite mot pour mot, dans sa langue d'origine -- NE LA TRADUIS JAMAIS, meme si le reste de ta reponse est en francais et la citation en anglais. Une citation en anglais reste en anglais, integralement, entre guillemets, sans aucune modification.
- Une a deux phrases courtes, jamais un paragraphe long.
- Utilise les noms d'activites du processus tels que donnes, jamais une reformulation qui en change le sens.
- Ne conclus jamais sur la conformite globale du processus -- decris uniquement ce cas precis, rien de plus."""


REPORT_PROMPT_VIOLATED = """Le texte source exige que l'activite "{{term_label}}" ait deja eu lieu avant que l'activite "{{target_label}}" puisse se produire.

Citation du texte source associee a cette exigence, A REPRODUIRE MOT POUR MOT SANS LA TRADUIRE (justifie qu'une condition est exigee, pas forcement ce qu'elle signifie precisement) : "{{quote}}"

Dans le processus observe, les deux activites existent bien, mais rien dans l'enchainement du processus ne fait que "{{term_label}}" precede effectivement "{{target_label}}".

{rules}

Ecris l'explication de cet ecart pour un lecteur non technique, familier des processus metier mais pas de notre outillage interne.""".format(rules=_REPORT_RULES)


REPORT_PROMPT_UNRESOLVABLE = """Le texte source exige qu'une condition precise soit deja remplie avant que l'activite "{{target_label}}" puisse se produire : {{missing_description}}

Citation du texte source associee a cette exigence, A REPRODUIRE MOT POUR MOT SANS LA TRADUIRE (justifie qu'une condition est exigee, pas forcement ce qu'elle signifie precisement) : "{{quote}}"

Cette condition n'a pu etre associee a aucune activite claire du processus observe -- il est donc impossible de verifier si l'exigence est respectee ou non, dans un sens comme dans l'autre.

{rules}

Ecris l'explication de ce cas pour un lecteur non technique, en etant explicite que ceci n'est ni une confirmation ni une infirmation de conformite -- juste une impossibilite de verifier.""".format(rules=_REPORT_RULES)