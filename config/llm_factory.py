"""Provider-neutral LangChain chat-model factory for research and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from config.settings import settings

Tier = Literal["quick", "deep"]

_ALIASES = {
    "gemini": "google",
    "google_gemini": "google",
    "claude": "anthropic",
    "grok": "xai",
    "qwen": "dashscope",
    "alibaba": "dashscope",
    "glm": "zhipu",
    "kimi": "moonshot",
    "nim": "nvidia",
    "bedrock": "aws_bedrock",
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
}

_API_KEYS = {
    "openai": ("openai_api_key",),
    "google": ("google_api_key",),
    "anthropic": ("anthropic_api_key",),
    "xai": ("xai_api_key",),
    "deepseek": ("deepseek_api_key",),
    "dashscope": ("dashscope_api_key", "dashscope_cn_api_key"),
    "zhipu": ("zhipu_api_key", "zhipu_cn_api_key"),
    "minimax": ("minimax_api_key", "minimax_cn_api_key"),
    "mistral": ("mistral_api_key",),
    "moonshot": ("moonshot_api_key",),
    "groq": ("groq_api_key",),
    "nvidia": ("nvidia_api_key",),
    "openrouter": ("openrouter_api_key",),
    "azure_openai": ("azure_openai_api_key",),
    "openai_compatible": ("openai_compatible_api_key",),
}

_OPENAI_COMPATIBLE_BASE_URLS = {
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "dashscope": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "minimax": "https://api.minimax.io/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

_DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "xai": "grok-3-mini",
    "deepseek": "deepseek-chat",
    "dashscope": "qwen-plus",
    "zhipu": "glm-4.5-air",
    "minimax": "MiniMax-M2",
    "mistral": "mistral-small-latest",
    "moonshot": "kimi-k2-turbo-preview",
    "groq": "llama-3.3-70b-versatile",
    "nvidia": "meta/llama-3.3-70b-instruct",
    "openrouter": "openai/gpt-4.1-mini",
    "azure_openai": "",
    "aws_bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "ollama": "llama3.1",
    "openai_compatible": "",
}


@dataclass(frozen=True)
class LLMConfiguration:
    """Resolved provider, model, and credentials for a single thinking tier."""

    tier: Tier
    provider: str
    model: str
    api_key: str
    base_url: str | None = None


def _secret_for(provider: str) -> str:
    for name in _API_KEYS.get(provider, ()):
        value = getattr(settings, name, "")
        if value and not value.startswith("your_"):
            return value
    return ""


def _auto_provider() -> str:
    priority = (
        "anthropic", "openai", "google", "deepseek", "xai", "dashscope",
        "zhipu", "minimax", "mistral", "moonshot", "groq", "nvidia",
        "openrouter", "azure_openai", "openai_compatible",
    )
    for provider in priority:
        if provider == "azure_openai" and not settings.azure_openai_endpoint:
            continue
        if _secret_for(provider):
            return provider
    if settings.aws_default_region:
        return "aws_bedrock"
    raise RuntimeError(
        "No LLM provider is configured. Set a provider API key or explicitly "
        "select local Ollama with TRADEMAX_LLM_PROVIDER=ollama."
    )


def resolve_configuration(tier: Tier = "quick") -> LLMConfiguration:
    """Resolve provider/model settings, preferring tier-specific overrides."""
    if tier not in ("quick", "deep"):
        raise ValueError("tier must be 'quick' or 'deep'")

    tier_provider = getattr(settings, f"trademax_{tier}_think_provider").strip().lower()
    global_provider = settings.trademax_llm_provider.strip().lower()
    provider = _ALIASES.get(tier_provider or global_provider, tier_provider or global_provider)
    if not provider or provider == "auto":
        provider = _auto_provider()
    if provider not in _DEFAULT_MODELS:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    model = getattr(settings, f"trademax_{tier}_think_llm").strip()
    model = model or _DEFAULT_MODELS[provider]
    if not model:
        raise ValueError(f"A model name must be configured for provider {provider}")

    api_key = _secret_for(provider)
    if provider not in ("aws_bedrock", "ollama") and not api_key:
        raise RuntimeError(f"No API key configured for selected provider {provider}")
    if provider == "aws_bedrock" and not settings.aws_default_region:
        raise RuntimeError("AWS_DEFAULT_REGION is required for AWS Bedrock")

    base_url = None
    if provider == "ollama":
        base_url = settings.ollama_base_url.rstrip("/")
    elif provider == "openai_compatible":
        base_url = settings.trademax_llm_backend_url.rstrip("/")
        if not base_url:
            raise RuntimeError("TRADEMAX_LLM_BACKEND_URL is required for OpenAI-compatible servers")
    elif provider == "azure_openai":
        if not settings.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI")
    elif provider in _OPENAI_COMPATIBLE_BASE_URLS:
        base_url = _OPENAI_COMPATIBLE_BASE_URLS[provider]
        if provider == "dashscope" and _secret_for("dashscope_cn_api_key"):
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return LLMConfiguration(tier, provider, model, api_key, base_url)


def _load_class(module_name: str, class_name: str) -> type[Any]:
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Provider integration {module_name} is not installed. "
            "Install the matching optional dependency for the selected provider."
        ) from exc
    return getattr(module, class_name)


def create_llm(tier: Tier = "quick", temperature: float | None = None) -> Any:
    """Create a LangChain chat model with structured-output support.

    Credentials are read only from server-side environment-backed settings.
    No automatic provider fallback is performed because it could change model
    behavior or leak prompts to a different provider.
    """
    config = resolve_configuration(tier)
    temp = settings.trademax_temperature if temperature is None else temperature
    if not 0 <= temp <= 2:
        raise ValueError("temperature must be between 0 and 2")
    common = {
        "model": config.model,
        "temperature": temp,
        "max_retries": settings.trademax_llm_retry_budget,
    }

    if config.provider == "anthropic":
        model_class = _load_class("langchain_anthropic", "ChatAnthropic")
        return model_class(model=config.model, api_key=config.api_key, temperature=temp,
                           max_retries=settings.trademax_llm_retry_budget)
    if config.provider == "google":
        model_class = _load_class("langchain_google_genai", "ChatGoogleGenerativeAI")
        return model_class(model=config.model, google_api_key=config.api_key, temperature=temp,
                           max_retries=settings.trademax_llm_retry_budget)
    if config.provider == "groq":
        model_class = _load_class("langchain_groq", "ChatGroq")
        return model_class(model=config.model, api_key=config.api_key, temperature=temp,
                           max_retries=settings.trademax_llm_retry_budget)
    if config.provider == "mistral":
        model_class = _load_class("langchain_mistralai", "ChatMistralAI")
        return model_class(model=config.model, api_key=config.api_key, temperature=temp,
                           max_retries=settings.trademax_llm_retry_budget)
    if config.provider == "ollama":
        if config.base_url and config.base_url.rstrip("/").endswith("/v1"):
            model_class = _load_class("langchain_openai", "ChatOpenAI")
            return model_class(
                **common, api_key=config.api_key or "ollama", base_url=config.base_url
            )
        model_class = _load_class("langchain_ollama", "ChatOllama")
        return model_class(
            model=config.model, base_url=config.base_url, temperature=temp,
            num_predict=2048,
        )
    if config.provider == "aws_bedrock":
        model_class = _load_class("langchain_aws", "ChatBedrockConverse")
        return model_class(model=config.model, region_name=settings.aws_default_region,
                           temperature=temp, max_retries=settings.trademax_llm_retry_budget)
    if config.provider == "azure_openai":
        model_class = _load_class("langchain_openai", "AzureChatOpenAI")
        return model_class(
            azure_deployment=config.model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            api_key=config.api_key,
            temperature=temp,
            max_retries=settings.trademax_llm_retry_budget,
        )

    model_class = _load_class("langchain_openai", "ChatOpenAI")
    kwargs = {**common, "api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return model_class(**kwargs)


def get_quick_llm(temperature: float | None = None) -> Any:
    """Build the quick-tier model used for frequent analysis."""
    return create_llm("quick", temperature)


def get_deep_llm(temperature: float | None = None) -> Any:
    """Build the deep-tier model used for governance and final decisions."""
    return create_llm("deep", temperature)
