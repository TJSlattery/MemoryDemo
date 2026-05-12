"""
Grove proxy utility for Anthropic models.

Routes Anthropic API requests through the company's Grove API gateway,
which expects the API key in an `api-key` header (rather than the SDK's
default `x-api-key`) and a custom base URL.
"""

import os

from langchain_anthropic import ChatAnthropic

GROVE_BASE_URL = "https://grove-gateway-prod.azure-api.net/grove-foundry-prod/anthropic"
GROVE_ANTHROPIC_VERSION = "2023-06-01"


def create_grove_chat_model(
    model: str = "claude-haiku-4-5",
    *,
    api_key: str | None = None,
    base_url: str = GROVE_BASE_URL,
    **kwargs,
) -> ChatAnthropic:
    """Build a ChatAnthropic client that talks to the Grove proxy.

    Args:
        model: Anthropic model identifier (e.g. "claude-haiku-4-5").
        api_key: Grove API key. Falls back to the GROVE_API_KEY env var.
        base_url: Proxy base URL. Defaults to the Grove production gateway.
        **kwargs: Additional keyword arguments forwarded to ChatAnthropic
            (e.g. temperature, max_tokens, timeout).

    Returns:
        A configured ChatAnthropic instance ready to use anywhere a
        LangChain chat model is expected.
    """
    key = api_key or os.getenv("GROVE_API_KEY")
    if not key:
        raise ValueError(
            "GROVE_API_KEY is not set. Add it to your .env file or pass api_key=..."
        )

    kwargs.setdefault("streaming", True)
    return ChatAnthropic(
        model=model,
        base_url=base_url,
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
