"""Configurable redaction for supervisor prompts and traces."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from traceguard.types import StrictModel

DEFAULT_SECRET_PATTERNS = [
    r"(?i)\bapi[_-]?key\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})",
    r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{12,}",
    r"(?i)\bpassword\s*[:=]\s*['\"]?[^'\"\s]+",
    r"(?i)\b(access|session|refresh)[_-]?token\s*[:=]\s*['\"]?[A-Za-z0-9_\-.=]{12,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    r"(?i)\bcookie\s*[:=]\s*[^;\n]+",
]


class RedactionConfig(StrictModel):
    enabled: bool = True
    replacement: str = "[REDACTED_SECRET]"
    extra_patterns: list[str] = Field(default_factory=list)

    @property
    def patterns(self) -> list[re.Pattern[str]]:
        flags = re.DOTALL
        return [
            re.compile(pattern, flags)
            for pattern in [*DEFAULT_SECRET_PATTERNS, *self.extra_patterns]
        ]


def redact_text(text: str, config: RedactionConfig | None = None) -> str:
    config = config or RedactionConfig()
    if not config.enabled:
        return text
    redacted = text
    for pattern in config.patterns:
        redacted = pattern.sub(config.replacement, redacted)
    return redacted


def redact_value(value: Any, config: RedactionConfig | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, config)
    if isinstance(value, Mapping):
        return {str(key): redact_value(item, config) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_value(item, config) for item in value]
    return value
