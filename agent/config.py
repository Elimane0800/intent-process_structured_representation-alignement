"""Configuration centralisee des modeles utilises dans le pipeline -- une seule source de
verite, plus aucun nom de modele code en dur dans un node ou un script d'orchestration.
Structuree pour boucler sur plusieurs modeles (MODELS_TO_RUN), pas seulement en choisir un a la
fois."""

from typing import NotRequired, TypedDict


class ModelConfig(TypedDict):
    name: str            # identifiant exact attendu par l'API du provider (agent/models/base_llm.py)
    max_tokens: int
    temperature: float
    # Provider d'inference (cle de PROVIDERS ci-dessous). Absent = "nvidia" -- les deux
    # exposent une API compatible OpenAI, donc le meme ChatOpenAI sert partout : ajouter un
    # provider = une entree dans PROVIDERS + une variable d'environnement, zero changement
    # en aval de get_llm().
    provider: NotRequired[str]
    # --- Controle du reasoning : TROIS mecanismes distincts selon la famille de modele, tous
    # verifies contre les fiches NVIDIA/HF (juillet 2026). Jamais envoyes aux modeles qui ne
    # les exposent pas. Objectif commun : eviter que le budget de tokens parte en raisonnement
    # interne avant le JSON (hypothese documentee, precondition_observation.md section 11) et
    # que des blocs <think> polluent le content parse.
    # (1) famille gpt-oss / mistral-medium : parametre API reasoning_effort (low/medium/high,
    #     pas de "off" -- low MINIMISE seulement).
    reasoning_effort: NotRequired[str]
    # (2) famille llama-nemotron v1.x (ex. super-49b-v1.5) : system prompt "/no_think" --
    #     PAR DEFAUT (system prompt vide) le reasoning est ON avec blocs <think> dans le
    #     content, ce qui casserait json.loads ; ce champ est donc OBLIGATOIRE pour ces
    #     modeles dans ce pipeline. Injecte par base_llm via un wrapper (le pipeline n'envoie
    #     sinon aucun system prompt).
    reasoning_system_prompt: NotRequired[str]
    # (3) famille nemotron-3 (MoE hybride) : flag de chat template enable_thinking, transmis
    #     dans extra_body. Le reasoning sort dans reasoning_content (champ separe), pas en
    #     balises <think>. NB : mecanisme verifie via la doc RAG NVIDIA/vLLM -- a confirmer
    #     par smoke test sur l'endpoint integrate.api.nvidia.com avant run complet.
    extra_body: NotRequired[dict]


# Providers d'inference -- tous compatibles OpenAI, differencies uniquement par l'URL et la
# variable d'environnement portant la cle. "nvidia" est le defaut historique (aucune entree
# de MODELS existante n'a besoin d'etre modifiee).
PROVIDERS: dict[str, dict[str, str]] = {
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
}


# Catalogue complet -- repris de MODELS dans state_space_node_v3.py / precondition_node_with_retry.py,
# qui le dupliquaient chacun de leur cote. Cle courte -> config complete.
MODELS: dict[str, ModelConfig] = {
    "llama-3.1-8b": {"name": "meta/llama-3.1-8b-instruct", "max_tokens": 8192, "temperature": 0},
    "llama-3.3-70b": {"name": "meta/llama-3.3-70b-instruct", "max_tokens": 8192, "temperature": 0},
    "mistral-medium-3.5": {"name": "mistralai/mistral-medium-3.5-128b", "max_tokens": 8192, "temperature": 0,
                            "reasoning_effort": "low"},
    "mistral-nemotron": {"name": "mistralai/mistral-nemotron", "max_tokens": 8192, "temperature": 0},
    "gpt-oss-20b": {"name": "openai/gpt-oss-20b", "max_tokens": 8192, "temperature": 0,
                     "reasoning_effort": "low"},
    "gpt-oss-120b": {"name": "openai/gpt-oss-120b", "max_tokens": 8192, "temperature": 0,
                      "reasoning_effort": "low"},
    "mistral-large-3": {"name": "mistralai/mistral-large-3-675b-instruct-2512", "max_tokens": 8192, "temperature": 0},
    "kimi-k2.6": {"name": "moonshotai/kimi-k2.6", "max_tokens": 8192, "temperature": 0},
    "inkling": {"name": "thinkingmachines/inkling", "max_tokens": 8192, "temperature": 0},
    # Reasoning OFF via system prompt (mecanisme 2) -- greedy/temperature=0 est exactement la
    # recommandation NVIDIA pour le mode OFF de ce modele.
    "nemotron-super-49b": {"name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                            "max_tokens": 8192, "temperature": 0,
                            "reasoning_system_prompt": "/no_think"},
    # Reasoning OFF via flag de chat template (mecanisme 3).
    "nemotron-3-nano-30b": {"name": "nvidia/nemotron-3-nano-30b-a3b",
                             "max_tokens": 8192, "temperature": 0,
                             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "nemotron-3-super-120b": {"name": "nvidia/nemotron-3-super-120b-a12b",
                               "max_tokens": 8192, "temperature": 0,
                               "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    # Premier modele hors NVIDIA -- via Groq (API compatible OpenAI, cf. PROVIDERS). Qwen3.x
    # est une famille hybride reasoning ; Groq expose reasoning_effort pour la piloter --
    # "none" vise la desactivation (mode reponse directe, coherent avec le reste du run).
    # A CONFIRMER par smoke test : si l'endpoint rejette "none" pour ce modele, replier sur
    # "default" (le strip defensif des blocs <think> peut alors etre ajoute en le passant au
    # mecanisme 2 si necessaire). Necessite GROQ_API_KEY dans l'environnement/.env.
    "qwen3.6-27b": {"name": "qwen/qwen3.6-27b", "provider": "groq",
                     "max_tokens": 4096, "temperature": 0,
                     "reasoning_effort": "none"},
    # Via OpenRouter -- controle du reasoning par leur API unifiee ({"reasoning": ...} dans
    # le corps de requete), transmis tel quel via extra_body. enabled=False = mode reponse
    # directe, coherent avec le reste du run.
    "north-mini-code": {"name": "cohere/north-mini-code:free", "provider": "openrouter",
                         "max_tokens": 4096, "temperature": 0,
                         "extra_body": {"reasoning": {"enabled": False}}},
    "gemma-4-31b-it": {"name": "google/gemma-4-31b-it:free", "provider": "openrouter",
                         "max_tokens": 4096, "temperature": 0,
                         "extra_body": {"reasoning": {"enabled": False}}},
}

# Modele d'embedding -- config separee car les parametres pertinents sont differents
# (input_type, taille de lot), pas un ModelConfig standard. Decisions actees et testees dans la
# conversation, cf. state_matching_observations.md : tout en 'query' (pas 'passage'), lots de 8.
EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-1b-v2"
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_INPUT_TYPE = "query"

# Modele par defaut du node redacteur du rapport final -- role independant du modele "sous
# test" (state_space/precondition), cf. report_observations.md section 4. Peut etre surchargee
# par run, pas necessairement le meme modele que celui evalue.
DEFAULT_REPORT_MODEL_KEY = "llama-3.1-8b"

# Liste effectivement bouclee par les scripts d'orchestration (ex. run_job_application.py) --
# volontairement distincte de MODELS (le catalogue complet) : certains modeles du catalogue
# peuvent etre indisponibles ou instables cote fournisseur (cf. state_space_observation.md,
# section sur les modeles non deployes/instables). A etendre au fur et a mesure.
MODELS_TO_RUN: list[str] = [
    #"llama-3.1-8b",
    "mistral-nemotron",
    "gpt-oss-20b",
]


def get_model_config(key: str) -> ModelConfig:
    if key not in MODELS:
        raise KeyError(f"Modele inconnu: {key!r}. Disponibles: {sorted(MODELS)}")
    return MODELS[key]