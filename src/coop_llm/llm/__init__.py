from .base import LLMClient, LLMResponse, RateLimiter, with_retries
from .mock import MockLLM


def make_client(cfg: dict, dry_run: bool = False) -> LLMClient:
    """依 models.yaml 的一個項目建立後端。

    provider: openai_compat | anthropic | mock
    dry_run=True 時一律使用 MockLLM,不碰網路。
    """
    name = cfg["name"]
    if dry_run or cfg.get("provider") == "mock":
        return MockLLM(policy=cfg.get("policy", "random"), name=name)
    provider = cfg["provider"]
    if provider == "openai_compat":
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient(cfg["model"], base_url=cfg.get("base_url"),
                                  api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"), name=name,
                                  extra_body=cfg.get("extra_body"), rpm=cfg.get("rpm"),
                                  max_concurrent=cfg.get("max_concurrent"),
                                  max_tokens_param=cfg.get("max_tokens_param", "max_tokens"),
                                  timeout=cfg.get("timeout", 180.0), attempts=cfg.get("attempts", 8))
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(cfg["model"], api_key_env=cfg.get("api_key_env", "ANTHROPIC_API_KEY"), name=name)
    raise ValueError(f"未知 provider {provider!r}")


__all__ = ["RateLimiter", "LLMClient", "LLMResponse", "MockLLM", "with_retries", "make_client"]
