"""Structured contracts for LLM supervisor providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from traceguard.types import (
    Decision,
    Observation,
    PostRunDisposition,
    RiskLevel,
    SandboxEvidence,
    StrictModel,
    ToolCall,
)

SupervisorGoalRelevance = Literal["RELEVANT", "UNRELATED", "UNCERTAIN"]
SupervisorNecessity = Literal["NECESSARY", "UNNECESSARY", "UNCERTAIN"]
SupervisorRiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class RewrittenToolCall(StrictModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class SupervisorProviderMetadata(StrictModel):
    provider: str
    model: str
    model_tag: str | None = None
    model_digest: str | None = None
    quantization: str | None = None
    memory_use: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    retries: int = Field(default=0, ge=0)


class SupervisorEvaluationLog(StrictModel):
    user_goal: str
    proposed_call: dict[str, Any]
    deterministic_enabled: bool
    deterministic_decision: str | None = None
    provider_called: bool = False
    provider_response: dict[str, Any] | None = None
    final_decision: dict[str, Any]
    error: str | None = None


class SupervisorRequest(StrictModel):
    user_goal: str
    proposed_call: ToolCall
    available_tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    previous_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    argument_provenance: dict[str, list[str]] = Field(default_factory=dict)
    step_already_rewritten: bool = False
    enable_rewrite: bool = True
    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)


class SupervisorResponse(StrictModel):
    decision: Decision
    goal_relevance: SupervisorGoalRelevance
    necessity: SupervisorNecessity
    risk_level: SupervisorRiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=800)
    rewritten_call: RewrittenToolCall | None = None
    metadata: SupervisorProviderMetadata | None = None

    @model_validator(mode="after")
    def validate_rewrite_payload(self) -> SupervisorResponse:
        if self.decision is Decision.REWRITE and self.rewritten_call is None:
            raise ValueError("REWRITE requires rewritten_call")
        if self.decision is not Decision.REWRITE and self.rewritten_call is not None:
            raise ValueError("rewritten_call must be absent unless decision is REWRITE")
        return self


class PostRunSupervisorRequest(StrictModel):
    user_goal: str
    executed_call: ToolCall
    sandbox_evidence: SandboxEvidence


class PostRunSupervisorResponse(StrictModel):
    disposition: PostRunDisposition
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=800)
    metadata: SupervisorProviderMetadata | None = None


class SupervisorProvider(Protocol):
    def evaluate(self, request: SupervisorRequest) -> SupervisorResponse: ...

    def evaluate_post_run(self, request: PostRunSupervisorRequest) -> PostRunSupervisorResponse: ...


class SupervisorProviderError(RuntimeError):
    """Base class for provider failures."""


class SupervisorSchemaError(SupervisorProviderError):
    """Provider returned syntactically valid data that failed the schema."""


class SupervisorTransportError(SupervisorProviderError):
    """Provider transport failed."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def validate_rewrite_against_request(
    response: SupervisorResponse,
    request: SupervisorRequest,
) -> None:
    """Validate generic rewrite constraints against current runtime tool schemas."""

    if response.decision is not Decision.REWRITE:
        return
    if response.rewritten_call is None:
        raise SupervisorSchemaError("REWRITE response missing rewritten_call")
    if not request.enable_rewrite:
        raise SupervisorSchemaError("REWRITE returned while rewriting is disabled")
    if request.step_already_rewritten:
        raise SupervisorSchemaError("second rewrite for same logical step")

    tool_name = response.rewritten_call.tool_name
    if tool_name not in request.available_tools:
        raise SupervisorSchemaError(f"rewritten tool does not exist: {tool_name}")

    schema = request.available_tools[tool_name]
    _validate_arguments_against_json_schema(response.rewritten_call.arguments, schema)


def _validate_arguments_against_json_schema(
    arguments: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    """Small schema check for provider tests and AgentDojo JSON schemas.

    This intentionally validates the portable subset we need here: object type,
    required keys, and no unknown top-level keys when properties are provided.
    The real tool runtime still performs authoritative validation before execution.
    """

    if schema.get("type") not in {None, "object"}:
        raise SupervisorSchemaError("rewritten arguments schema must be an object")
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    missing = required - set(arguments)
    if missing:
        raise SupervisorSchemaError(f"rewritten arguments missing required keys: {sorted(missing)}")
    if properties:
        extra = set(arguments) - set(properties)
        if extra:
            raise SupervisorSchemaError(
                f"rewritten arguments contain unknown keys: {sorted(extra)}"
            )
