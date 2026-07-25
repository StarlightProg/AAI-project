"""Backward-compatible runtime wrappers for the secure structured providers.

New code should construct supervisors through ``traceguard.supervisor.factory``.
"""

from __future__ import annotations

from traceguard.supervisor.llm import (
    GeminiSupervisor as GeminiProvider,
)
from traceguard.supervisor.llm import (
    GoogleGenAITransport,
    QwenSupervisor,
)
from traceguard.supervisor.llm import (
    OllamaSupervisor as OllamaProvider,
)


class GeminiSupervisor(QwenSupervisor):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        max_transport_retries: int = 2,
    ) -> None:
        super().__init__(
            provider=GeminiProvider(
                model=model,
                transport=GoogleGenAITransport(api_key),
                timeout=timeout,
                max_transport_retries=max_transport_retries,
            ),
            deterministic_enabled=False,
        )


class OllamaSupervisor(QwenSupervisor):
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        *,
        max_transport_retries: int = 2,
    ) -> None:
        super().__init__(
            provider=OllamaProvider(
                model=model,
                url=base_url,
                timeout=timeout,
                max_transport_retries=max_transport_retries,
            ),
            deterministic_enabled=False,
        )
