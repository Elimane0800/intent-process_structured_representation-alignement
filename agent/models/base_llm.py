import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent.config import MODELS, PROVIDERS

load_dotenv()
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"

# Resolution inverse nom brut -> config : le meme modele doit recevoir la meme configuration
# qu'il soit designe par sa cle courte ('gpt-oss-20b', chemin pipeline via graph.py) ou par son
# nom API brut ('openai/gpt-oss-20b', anciens scripts) -- sans ca, le controle du reasoning ne
# serait actif que sur un des deux chemins, silencieusement.
_CONFIG_BY_NAME = {cfg["name"]: cfg for cfg in MODELS.values()}

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


class _ReasoningSafeChat:
    """Wrapper minimal autour de ChatOpenAI pour les modeles dont le reasoning se controle par
    system prompt (famille llama-nemotron v1.x) : (1) injecte le system prompt de controle
    (ex. '/no_think') devant tout prompt passe en chaine -- le pipeline n'envoie sinon aucun
    system prompt, laissant ces modeles en reasoning ON par defaut ; (2) retire defensivement
    tout bloc <think>...</think> residuel du content avant de le rendre (la fiche modele
    precise qu'un bloc <think></think> peut apparaitre meme sans raisonnement necessaire).
    Tous les sites d'appel du pipeline utilisent llm.invoke(chaine).content uniquement -- le
    wrapper couvre exactement ce contrat, rien de plus."""

    def __init__(self, llm: ChatOpenAI, system_prompt: str):
        self._llm = llm
        self._system_prompt = system_prompt

    def invoke(self, input, **kwargs):
        from langchain_core.messages import HumanMessage, SystemMessage
        if isinstance(input, str):
            input = [SystemMessage(content=self._system_prompt), HumanMessage(content=input)]
        response = self._llm.invoke(input, **kwargs)
        content = response.content
        if isinstance(content, str) and "<think>" in content:
            response.content = _THINK_BLOCK.sub("", content)
        return response


def get_llm(temperature: float | None = None, model: str | None = None,
            max_tokens: int | None = None, reasoning_effort: str | None = None):
    """`model` accepte soit une cle de agent.config.MODELS (ex. 'llama-3.1-8b'), soit un nom de
    modele brut tel qu'attendu par l'API NVIDIA (ex. 'meta/llama-3.1-8b-instruct') -- garde la
    compatibilite avec le code existant qui appelle get_llm(temperature=0, model=model_name)
    avec un nom brut (state_space_node_v3.py, precondition_node_with_retry.py). Si `model`
    resout vers une config connue (par cle OU par nom brut), temperature/max_tokens et le
    controle du reasoning en sont tires ; sinon, replis sur les anciens defauts.

    Quatre mecanismes de controle du reasoning, selectionnes par la config du modele (cf.
    agent/config.py, jamais envoyes aux modeles qui ne les exposent pas) :
    - reasoning_effort (gpt-oss, mistral-medium, OpenAI GPT-5.x) -> extra_body {'reasoning_effort': ...}
    - reasoning_system_prompt (llama-nemotron v1.x) -> wrapper _ReasoningSafeChat
    - extra_body (nemotron-3) -> transmis tel quel (ex. chat_template_kwargs.enable_thinking)
    - verbosity (OpenAI GPT-5.x uniquement) -> extra_body {'verbosity': ...}
    Retour : ChatOpenAI, ou _ReasoningSafeChat (meme contrat .invoke(...).content).

    Cas particulier OpenAI GPT-5.x (reasoning) : ces modeles rejettent le parametre
    temperature avec une erreur 400 des qu'il differe du defaut (=1). Quand la config du
    modele porte temperature=None (cf. agent/config.py), le kwarg temperature n'est pas
    transmis du tout a ChatOpenAI plutot que d'y forcer une valeur."""
    config = MODELS.get(model) or _CONFIG_BY_NAME.get(model)
    if config is not None:
        resolved_name = config["name"]
        default_temperature = config["temperature"]
        default_max_tokens = config["max_tokens"]
    else:
        resolved_name = model or DEFAULT_MODEL_NAME
        default_temperature = 0.2
        default_max_tokens = 8192
        config = _CONFIG_BY_NAME.get(resolved_name, {})

    extra_body = dict(config.get("extra_body", {}))
    resolved_reasoning = reasoning_effort if reasoning_effort is not None \
        else config.get("reasoning_effort")
    if resolved_reasoning is not None:
        extra_body["reasoning_effort"] = resolved_reasoning
    # Mecanisme 4 (OpenAI GPT-5.x) : verbosity n'a pas d'override par argument de get_llm
    # comme reasoning_effort en a un -- pas de besoin identifie a ce jour, cf. seul le
    # catalogue de config.py le pilote pour l'instant.
    resolved_verbosity = config.get("verbosity")
    if resolved_verbosity is not None:
        extra_body["verbosity"] = resolved_verbosity

    # Resolution du provider : "nvidia" par defaut (aucune entree existante a modifier),
    # sinon la config du modele designe son provider dans PROVIDERS (base_url + variable
    # d'environnement de la cle). Un modele Groq et un modele NVIDIA passent par le meme
    # ChatOpenAI -- seule l'adresse et la cle changent, rien en aval de get_llm() ne le sait.
    provider = PROVIDERS[config.get("provider", "nvidia")]
    api_key = os.environ.get(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Variable d'environnement {provider['api_key_env']} absente -- requise pour le "
            f"modele {resolved_name!r} (provider {config.get('provider', 'nvidia')!r})."
        )
    kwargs = {}
    if extra_body:
        # Parametre explicite de ChatOpenAI (langchain-openai recent) -- le passer via
        # model_kwargs fonctionne aussi mais declenche un UserWarning et un deplacement
        # automatique vers ce champ ; autant etre direct.
        kwargs["extra_body"] = extra_body
    # default_temperature is None signale un modele qui REJETTE ce parametre (GPT-5.x
    # reasoning cote OpenAI, erreur 400 "temperature does not support X, only default (1) is
    # supported" -- verifie juillet 2026). C'est prioritaire sur l'argument `temperature` de
    # get_llm : les sites d'appel existants (state_space_node_v3.py,
    # precondition_node_with_retry.py) appellent tous get_llm(temperature=0, model=...) --
    # sans cette priorite, ce 0 explicite ecraserait le None de la config et ferait planter
    # l'appel pour ces modeles precis, silencieusement pour l'appelant.
    if default_temperature is None:
        resolved_temperature = None
    else:
        resolved_temperature = temperature if temperature is not None else default_temperature
    if resolved_temperature is not None:
        kwargs["temperature"] = resolved_temperature
    llm = ChatOpenAI(
        model=resolved_name,
        openai_api_key=api_key,
        base_url=provider["base_url"],
        max_tokens=max_tokens if max_tokens is not None else default_max_tokens,
        **kwargs,
    )
    system_prompt = config.get("reasoning_system_prompt")
    return _ReasoningSafeChat(llm, system_prompt) if system_prompt else llm


if __name__ == "__main__":
    # Smoke test : un modele par mecanisme de controle -- verifier que chaque endpoint accepte
    # son parametre AVANT tout run complet (une erreur ici est explicite, pas confinee).
    for key in ("llama-3.1-8b", "gpt-oss-20b", "nemotron-super-49b", "nemotron-3-nano-30b"):
        try:
            llm = get_llm(temperature=0, model=key)
            response = llm.invoke("Reponds uniquement par le mot: OK")
            print(f"[{key}] -> {response.content!r}")
        except Exception as e:
            print(f"[{key}] ERREUR: {e}")