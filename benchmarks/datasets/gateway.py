from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traceguard.supervisor.agentdojo_pipeline import AgentDojoDeterministicSupervisor
from traceguard.supervisor.base import Supervisor
from traceguard.types import (
    Decision,
    GoalNecessity,
    GoalRelevance,
    Observation,
    RiskLevel,
    SupervisorOutput,
    ToolCall,
    TrustLabel,
)


class SupervisionGateway:
    """Framework-neutral pre-tool supervision boundary."""

    def __init__(
        self,
        *,
        mode: str,
        supervisor: Supervisor | None = None,
        available_tools: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        if mode not in {"none", "deterministic", "llm", "deterministic_llm"}:
            raise ValueError(f"unknown supervisor mode: {mode}")
        self.mode = mode
        if supervisor is None and mode in {"deterministic", "deterministic_llm"}:
            supervisor = AgentDojoDeterministicSupervisor()
        if supervisor is None and mode in {"llm", "deterministic_llm"}:
            raise ValueError("LLM modes require an explicit supervisor")
        self.supervisor = supervisor
        self.available_tools = dict(available_tools or {})

    def evaluate(
        self,
        *,
        task_id: str,
        step_id: int,
        user_goal: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        observations: list[Observation],
        requested_resources: list[str] | None = None,
    ) -> tuple[ToolCall, SupervisorOutput]:
        call = ToolCall(
            task_id=task_id,
            step_id=step_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            consumed_observation_ids=[item.observation_id for item in observations],
            requested_resources=list(requested_resources or []),
        )
        if self.supervisor is None:
            return call, SupervisorOutput(
                decision=Decision.ALLOW,
                risk=RiskLevel.LOW,
                confidence=1.0,
                policy_violation=False,
                goal_relevance=GoalRelevance.STRONG,
                goal_necessity=GoalNecessity.HELPFUL,
                reason="Supervision is disabled for this ablation.",
            )
        if hasattr(self.supervisor, "available_tools"):
            self.supervisor.available_tools = self.available_tools
        output = self.supervisor.evaluate(user_goal, call, observations)
        return call, output

    @staticmethod
    def observation(
        call: ToolCall,
        content: str,
        *,
        source_type: str,
        trust: TrustLabel = TrustLabel.UNTRUSTED_TOOL,
    ) -> Observation:
        return Observation(
            content=content,
            source_type=source_type,
            source_id=call.call_id,
            trust=trust,
            provenance_chain=[*call.consumed_observation_ids, call.call_id],
            may_contain_instructions=trust
            in {TrustLabel.UNTRUSTED_DOCUMENT, TrustLabel.UNTRUSTED_TOOL},
        )
