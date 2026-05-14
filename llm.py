"""Chat model factory.

Dispatches between the direct Anthropic API and the Grove proxy gateway
based on the ``LLM_PROVIDER`` env var (default: ``anthropic``).

The Grove proxy authenticates via an ``api-key`` header (rather than the
SDK's default ``x-api-key``) and uses a custom base URL; it also does
not support streaming responses, so we force ``streaming=False`` there
and enable it for the direct Anthropic path.
"""

import os

from langchain_anthropic import ChatAnthropic

GROVE_BASE_URL = "https://grove-gateway-prod.azure-api.net/grove-foundry-prod/anthropic"
GROVE_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_PROVIDER = "anthropic"


def create_chat_model(
    model: str = "claude-haiku-4-5",
    *,
    provider: str | None = None,
    **kwargs,
) -> ChatAnthropic:
    """Build a ChatAnthropic client for the configured provider.

    ``provider`` defaults to the ``LLM_PROVIDER`` env var (or
    ``"anthropic"``). Set ``LLM_PROVIDER=grove`` to route through the
    Grove proxy instead. Extra ``**kwargs`` are forwarded to
    ``ChatAnthropic`` (e.g. ``temperature``, ``max_tokens``, ``timeout``).
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider == "anthropic":
        return _create_anthropic(model, **kwargs)
    if provider == "grove":
        return _create_grove(model, **kwargs)
    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}; expected 'anthropic' or 'grove'."
    )


def _create_anthropic(model: str, **kwargs) -> ChatAnthropic:
    key = kwargs.pop("api_key", None) or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or pass api_key=..."
        )
    kwargs.setdefault("streaming", True)
    return ChatAnthropic(model=model, api_key=key, **kwargs)


def _create_grove(model: str, **kwargs) -> ChatAnthropic:
    key = kwargs.pop("api_key", None) or os.getenv("GROVE_API_KEY")
    if not key:
        raise ValueError(
            "GROVE_API_KEY is not set. Add it to your .env file or pass api_key=..."
        )
    kwargs.setdefault("streaming", False)
    return ChatAnthropic(
        model=model,
        base_url=kwargs.pop("base_url", GROVE_BASE_URL),
        # The Anthropic SDK requires an api_key value; the proxy ignores
        # the default x-api-key header and authenticates via the
        # `api-key` header injected below.
        api_key="unused-proxy-auth-via-default-headers",
        default_headers={
            "api-key": key,
            "anthropic-version": GROVE_ANTHROPIC_VERSION,
        },
        **kwargs,
    )
