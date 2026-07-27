"""Baseline LLM-as-judge : evalue un jugement LLM holistique (texte + XML BPMN brut, aucun
passage par U/G/matching) sur les MEMES mutants que run_perturbation_study.py, avec la MEME
taxonomie required/forbidden/blind_spot et la meme metrique case-level (le verdict a-t-il
"bascule" du cas de base vers le mutant ?).

Pourquoi reutiliser exactement le cadre du perturbation study (decide avant tout code, cf.
discussion) :
- Comparaison directe, chiffre contre chiffre, avec TGMS deja mesure sur les memes 153 mutants.
- Verite terrain deja etablie par construction (mutate_bpmn.py), pas de nouvelle verite a
  inventer pour cette baseline.
- Question empirique explicitement ouverte : sur blind_spot (XOR<->AND), notre pipeline est
  aveugle PAR CONSTRUCTION (la relation "follows" ne porte jamais le type de gateway). Un LLM
  qui lit le XML entier n'a pas cette limite structurelle -- s'il detecte ce que nous ne
  pouvons pas detecter, c'est un resultat a assumer honnetement dans le papier, pas a cacher.

Entree du juge : texte source T + XML BPMN BRUT du mutant (option (a), choisie explicitement
plutot que le DFG extrait (b) -- (b) aurait deja perdu le type de gateway comme notre pipeline,
tuant la question blind_spot ci-dessus avant meme de la poser).

Prompt ferme, verdict binaire (VIOLATION / NO_VIOLATION) -- comparable a detected/
false_positive_violated/dfg_identical du perturbation study, jamais un verdict a trois valeurs
ou une localisation precise (disproportionne pour une baseline, le but est de montrer l'absence
de garantie du jugement brut, pas de le rendre aussi riche que TGMS).

Mise a jour : UN SEUL prompt etait un point faible identifie -- un reviewer peut toujours
objecter qu'un baseline LLM-as-judge n'a echoue/reussi qu'a cause d'un prompt particulier, pas
du principe "juger sans decomposition symbolique". Trois variantes ajoutees (zero_shot,
one_shot, cot), meme discipline d'ecriture que les prompts state_space/precondition deja
existants dans le projet : blocs Definition/Rules separes, contre-exemple WRONG/RIGHT concret
plutot qu'abstrait pour le one-shot, marqueur FINAL_ANSWER: obligatoire pour le CoT (sans quoi
le parsing casse sur du texte parasite apres la reponse -- deja documente sur state_space_node,
cf. observations, section 6/12). Le CoT est volontairement borne a 4 etapes courtes, jamais une
verification par paire (le meme piege qui a fait exploser combinatoirement le CoT sur
work_accident cote state_space n'a aucune raison de ne pas se reproduire ici si le nombre
d'etapes scale avec le nombre d'activites du BPMN).

Stockage : JSONL append-only a la racine du projet (baseline_llm_judge_results.jsonl) -- une
ligne par (modele_juge, variante_prompt, mutant_file), jamais de reecriture du fichier entier
(contrairement a un JSON classique qui demanderait charger-modifier-reecrire a chaque
execution). Deja-fait detecte et saute par defaut (meme discipline que run.py/already_done()) ;
ALLOW_RERUN=True accumule une nouvelle ligne timestampee a cote de l'ancienne, jamais
d'ecrasement en place -- permet un futur test de reproductibilite sans perdre l'historique.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()  # cf. run_perturbation_study.py : necessaire, ce script n'importe pas base_llm
                # via un autre module qui le ferait deja en effet de bord

from agent.config import MODELS_TO_RUN
from agent.models.base_llm import get_llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUTANTS_DIR = os.path.join(BASE_DIR, "results", "mutants")
BPMN_ROOT = os.path.join(BASE_DIR, "text_and_bpmn", "bpmn")
OUT_PATH = os.path.join(BASE_DIR, "baseline_llm_judge_results.jsonl")

# Juges = les modeles sous test actuels (agent.config.MODELS_TO_RUN) -- comparaison directe
# avec les memes modeles evalues comme "sous test" ailleurs dans le projet. Redefinissable
# independamment si on veut un jour juger avec des modeles differents (ex. plus costauds).
JUDGE_MODELS: list[str] = list(MODELS_TO_RUN)

ALLOW_RERUN = False  # False : saute les (juge, variante, mutant) deja presents dans le JSONL
                      # (comme already_done()). True : ajoute une nouvelle ligne timestampee a
                      # cote de l'ancienne, jamais d'ecrasement -- pour test de reproductibilite.

REQUIRED_OPS = {"remove_activity", "swap_labels", "cross_case_replace"}
FORBIDDEN_OPS = {"insert_activity", "shuffle_xml"}
BLIND_SPOT_OPS = {"rewire_gateway"}


# ---------------------------------------------------------------------------
# Bloc partage, meme principe de separation que _CANDIDATE_DEFINITION/_FORMAT_RULES
# dans state_space_prompt_permissive.py : ce qui definit la tache (jamais negociable
# entre variantes) separe de ce qui varie (exemple, raisonnement explicite).
# ---------------------------------------------------------------------------

_JUDGE_DEFINITION = """You are reviewing whether a BPMN process model faithfully implements a textual process description.

The model is allowed to be richer than the text: it may contain extra activities, extra steps, or extra detail not mentioned in the text at all -- that is never, by itself, a violation. The text is allowed to under-specify the process; the model has the right to do more than the text describes.

A violation exists only if the model actively contradicts a requirement stated in the text -- specifically:
- an activity, decision, or outcome that the text says must happen before another does NOT happen before it in this model (wrong order, or the required activity is missing entirely), or
- a step, condition, or outcome the text explicitly requires is entirely absent from the model, with no equivalent activity standing in for it.

Do not flag a violation merely because the model is more detailed, uses different wording for the same step, or includes parallel/alternative branches the text does not mention. Do not flag a violation because you are unsure -- only flag one if you can point to a specific requirement in the text that the model actively contradicts or omits."""

_JUDGE_OUTPUT_RULES = """Output format, never negotiable:
- Respond with EXACTLY one of the two words VIOLATION or NO_VIOLATION on the first line.
- Follow it with a short (1-2 sentence) justification on the next line, naming the specific requirement involved.
- No other text, no markdown formatting, no bullet points, no repetition of the question."""


JUDGE_PROMPT_ZERO_SHOT = """{definition}

{output_rules}

Text description:
{{text}}

BPMN model (raw XML):
{{bpmn_xml}}

Does this BPMN model contradict a requirement stated in the text?""".format(
    definition=_JUDGE_DEFINITION, output_rules=_JUDGE_OUTPUT_RULES
)


JUDGE_PROMPT_ONE_SHOT = """{definition}

{output_rules}

Example:
Text description: "A library manages book loans. A member must be registered before they can borrow a book. Once a book is borrowed, it becomes checked out. The member must return the book before it can be checked out again by anyone else."

BPMN model (raw XML), summarized here for the example only -- the real input below always gives you full raw XML, never a summary: a process where "Borrow book" can be reached directly from the start event, with no "Register member" activity anywhere in the model, and no gateway or condition referencing registration.

Correct judgment for this example:
VIOLATION
The text requires registration before borrowing, but the model has no registration activity at all and allows borrowing directly from the start -- the required activity is entirely absent, not just reworded or reordered.

Contrast (why this would NOT be a violation): if the model instead had "Register member" then "Borrow book" then ALSO an extra "Send welcome email" activity in between, that would be NO_VIOLATION -- the extra activity does not contradict the text, it only adds detail the text never ruled out.

Now judge the real case below, following the same standard: only a contradiction or a missing required step counts as a violation, never extra detail, different wording, or your own uncertainty.

Text description:
{{text}}

BPMN model (raw XML):
{{bpmn_xml}}

Does this BPMN model contradict a requirement stated in the text?""".format(
    definition=_JUDGE_DEFINITION, output_rules=_JUDGE_OUTPUT_RULES
)


JUDGE_PROMPT_COT = """{definition}

Work through the following steps explicitly before answering. Keep each step short -- a few
bullet points at most, never an exhaustive pairwise enumeration of every activity against every
other activity.

Step 1 -- List the ordering/presence requirements the text actually states or clearly implies
(e.g. "X must happen before Y", "Y cannot happen without X first"). Do not invent requirements
the text does not support.
Step 2 -- For each requirement from Step 1, check the BPMN model: is the required activity
present, and does it occur in the required position relative to the other activity? Note only
whether it holds or not -- do not describe the entire model.
Step 3 -- Decide: is there at least one requirement from Step 1 that the model actively
contradicts or entirely omits (not just rephrases, reorders in a way the text never actually
constrained, or supplements with extra detail)?
Step 4 -- State the single verdict that follows from Step 3.

{output_rules}

After Step 4, output the line "FINAL_ANSWER:" followed immediately by the two-line answer
described above (verdict word, then justification) -- nothing after it.

Text description:
{{text}}

BPMN model (raw XML):
{{bpmn_xml}}""".format(definition=_JUDGE_DEFINITION, output_rules=_JUDGE_OUTPUT_RULES)


# Variantes bouclees par run_baseline -- meme principe que PROMPTS = {{"zero_shot":..., ...}}
# ailleurs dans le projet (precondition_node_with_retry.py). Cle courte reutilisee telle quelle
# dans les entrees JSONL (champ "prompt_variant").
PROMPT_VARIANTS: dict[str, str] = {
    "zero_shot": JUDGE_PROMPT_ZERO_SHOT,
    "one_shot": JUDGE_PROMPT_ONE_SHOT,
    "cot": JUDGE_PROMPT_COT,
}


def _load_manifest() -> list[dict]:
    path = os.path.join(MUTANTS_DIR, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"[baseline] {path} introuvable -- lancer mutate_bpmn.py d'abord")
    with open(path) as f:
        return json.load(f)


def _load_base_text(desc: str) -> str:
    with open(os.path.join(BPMN_ROOT, f"{desc}.txt"), encoding="utf-8") as f:
        return f.read()


def _load_bpmn_xml(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _parse_verdict(raw: str, prompt_variant: str) -> tuple[str | None, str]:
    """Retourne (verdict, raw_response). verdict est 'VIOLATION', 'NO_VIOLATION', ou None si
    la reponse ne respecte pas le format attendu -- jamais suppose silencieusement, trace
    explicitement pour diagnostic (meme discipline que le reste du pipeline : un echec de
    parsing n'est jamais un rejet silencieux).

    Pour le CoT : on ne parse qu'apres le marqueur FINAL_ANSWER:, exactement comme le parsing
    CoT deja en place cote state_space_node -- le raisonnement en amont contient forcement les
    mots VIOLATION/NO_VIOLATION en discussion libre, les chercher sur l'ensemble du texte
    donnerait de faux positifs de parsing."""
    text = raw.strip()
    if prompt_variant == "cot":
        marker = "FINAL_ANSWER:"
        if marker not in text:
            return None, raw
        text = text.split(marker, 1)[1].strip()

    first_line = text.splitlines()[0].strip().upper() if text else ""
    if first_line.startswith("NO_VIOLATION"):
        return "NO_VIOLATION", raw
    if first_line.startswith("VIOLATION"):
        return "VIOLATION", raw
    return None, raw


def _judge(llm, text: str, bpmn_xml: str, prompt_variant: str) -> dict:
    prompt_template = PROMPT_VARIANTS[prompt_variant]
    prompt = prompt_template.format(text=text, bpmn_xml=bpmn_xml)
    t0 = time.monotonic()
    response = llm.invoke(prompt)
    elapsed = time.monotonic() - t0
    verdict, raw = _parse_verdict(response.content, prompt_variant)
    usage = getattr(response, "response_metadata", {}).get("token_usage", {}) \
        if hasattr(response, "response_metadata") else {}
    return {
        "verdict": verdict,
        "raw_response": raw,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_chars": len(prompt),
        "token_usage": usage or None,
    }


def _load_done_keys(out_path: str) -> set[tuple[str, str, str]]:
    """Cle = (judge_model, prompt_variant, mutant_file) -- la variante fait partie de la cle
    de deduplication, jamais fusionnee avec le modele : un meme (modele, mutant) doit pouvoir
    etre juge sous les trois variantes independamment."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((
                entry.get("judge_model"),
                entry.get("prompt_variant"),
                entry.get("mutant_file"),
            ))
    return done


def _append_jsonl(out_path: str, entry: dict) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_baseline(
    judge_models: list[str] = JUDGE_MODELS,
    prompt_variants: dict[str, str] = PROMPT_VARIANTS,
    out_path: str = OUT_PATH,
) -> None:
    manifest = _load_manifest()
    applicable = [m for m in manifest if m["applicable"]]
    print(
        f"[baseline] {len(applicable)} mutants applicables, {len(judge_models)} juge(s), "
        f"{len(prompt_variants)} variante(s) de prompt ({', '.join(prompt_variants)})"
    )

    done_keys = _load_done_keys(out_path) if not ALLOW_RERUN else set()
    if done_keys:
        print(f"[baseline] {len(done_keys)} paire(s) (juge, variante, mutant) deja presentes "
              f"-- sautees (ALLOW_RERUN=False)")

    text_cache: dict[str, str] = {}
    # (judge_model, prompt_variant, base_key) -> verdict entry -- le verdict de base depend de
    # la variante de prompt autant que du modele, donc la variante fait partie de la cle de
    # cache, pas seulement de la cle de deduplication JSONL.
    base_verdict_cache: dict[tuple[str, str, str], dict] = {}

    for judge_model in judge_models:
        print(f"\n########## Juge : {judge_model} ##########")
        llm = get_llm(model=judge_model, temperature=0)

        for prompt_variant in prompt_variants:
            print(f"  ---- variante : {prompt_variant} ----")

            for m in applicable:
                key = (judge_model, prompt_variant, m["mutant_file"])
                if key in done_keys:
                    continue

                base_key = m["base"]
                desc, fname = base_key.split("/", 1)

                if desc not in text_cache:
                    text_cache[desc] = _load_base_text(desc)
                text = text_cache[desc]

                base_cache_key = (judge_model, prompt_variant, base_key)
                if base_cache_key not in base_verdict_cache:
                    base_path = os.path.join(BPMN_ROOT, desc, fname)
                    if not os.path.exists(base_path):
                        print(f"    [SKIP base introuvable] {base_key}")
                        base_verdict_cache[base_cache_key] = {"verdict": None}
                    else:
                        base_xml = _load_bpmn_xml(base_path)
                        try:
                            base_verdict_cache[base_cache_key] = _judge(
                                llm, text, base_xml, prompt_variant
                            )
                        except Exception as e:
                            print(f"    [ERREUR base] {base_key}: {e}")
                            base_verdict_cache[base_cache_key] = {
                                "verdict": None, "error": str(e)
                            }
                base_result = base_verdict_cache[base_cache_key]

                mutant_path = os.path.join(MUTANTS_DIR, m["mutant_file"])
                try:
                    mutant_xml = _load_bpmn_xml(mutant_path)
                    mutant_result = _judge(llm, text, mutant_xml, prompt_variant)
                    error = None
                except Exception as e:
                    mutant_result = {
                        "verdict": None, "raw_response": None,
                        "elapsed_seconds": None, "prompt_chars": None, "token_usage": None,
                    }
                    error = str(e)

                base_v = base_result.get("verdict")
                mutant_v = mutant_result.get("verdict")
                flipped_to_violation = (base_v == "NO_VIOLATION" and mutant_v == "VIOLATION")
                inconclusive = base_v is None or mutant_v is None or base_v == "VIOLATION"

                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "judge_model": judge_model,
                    "prompt_variant": prompt_variant,
                    "base": base_key,
                    "operator": m["operator"],
                    "variant": m["variant"],
                    "expected": m["expected"],
                    "mutant_file": m["mutant_file"],
                    "base_verdict": base_v,
                    "mutant_verdict": mutant_v,
                    "flipped_to_violation": flipped_to_violation,
                    "inconclusive": inconclusive,
                    "mutant_raw_response": mutant_result.get("raw_response"),
                    "mutant_elapsed_seconds": mutant_result.get("elapsed_seconds"),
                    "mutant_prompt_chars": mutant_result.get("prompt_chars"),
                    "mutant_token_usage": mutant_result.get("token_usage"),
                    "error": error,
                }
                _append_jsonl(out_path, entry)

                status = (
                    "INCONCLUSIVE" if inconclusive
                    else ("FLIP" if flipped_to_violation else "no-flip")
                )
                print(
                    f"    [{m['operator']:20s}] {m['mutant_file']:45s} "
                    f"base={base_v} mutant={mutant_v} -> {status}"
                )

    print(f"\n[baseline] Resultats accumules dans {out_path}")


if __name__ == "__main__":
    run_baseline()