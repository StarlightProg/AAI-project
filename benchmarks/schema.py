from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from traceguard.types import GoalNecessity, GoalRelevance, ThreatModel


class ProposedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    consumed_observation_ids: list[str] = Field(default_factory=list)
    requested_resources: list[str] = Field(default_factory=list)
    expected_relevance: GoalRelevance | None = None
    expected_necessity: GoalNecessity | None = None
    unsafe_proposal: bool | None = None


class CallPredicate(BaseModel):
    """Typed description of a tool call that must or must not occur."""

    model_config = ConfigDict(extra="forbid")
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    match: Literal["exact", "subset"] = "subset"


class DatasetProvenance(BaseModel):
    """Stable identity for a case imported from an upstream benchmark."""

    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    native_case_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    upstream_split: str | None = None
    license: str | None = None


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    threat_model: ThreatModel
    split: Literal["dev", "test"] = "dev"
    initial_state: dict[str, Any] = Field(default_factory=dict)
    user_goal: str
    attacker_goal: str | None = None
    attack_source: str | None = None
    attack_family: str | None = None
    payload_group_id: str | None = None
    participant_group_id: str | None = None
    authorized_near_neighbor_id: str | None = None
    provenance: DatasetProvenance | None = None
    available_tools: list[str]
    prohibited_effects: list[str] = Field(default_factory=list)
    forbidden_calls: list[CallPredicate] = Field(default_factory=list)
    expected_state_diff: dict[str, Any] | None = None
    utility_checks: list[dict[str, Any]]
    security_checks: list[dict[str, Any]]
    proposed_calls: list[ProposedCall] = Field(default_factory=list)
    docker_applicable: bool = False

    @model_validator(mode="after")
    def require_checks(self) -> BenchmarkCase:
        if not self.utility_checks or not self.security_checks:
            raise ValueError("utility_checks and security_checks are required")
        for index, proposed in enumerate(self.proposed_calls):
            if (
                proposed.expected_relevance is None
                or proposed.expected_necessity is None
                or proposed.unsafe_proposal is None
            ):
                raise ValueError(f"proposed_calls[{index}] requires explicit metric labels")
        return self


def load_cases(path: Path, *, split: Literal["dev", "test", "all"] = "all") -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [BenchmarkCase.model_validate(item) for item in raw]
    if split == "all":
        return cases
    return [case for case in cases if case.split == split]


def default_cases_path() -> Path:
    return Path(__file__).resolve().parent / "cases" / "custom_cases.json"


def verify_frozen_cases(path: Path | None = None) -> dict[str, Any]:
    cases_path = (path or default_cases_path()).resolve()
    manifest_path = cases_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    if manifest.get("status") != "frozen":
        raise ValueError("benchmark manifest is not frozen")
    if actual_digest != manifest.get("cases_sha256"):
        raise ValueError("benchmark cases changed without a new frozen manifest")
    cases = load_cases(cases_path)
    actual_counts: dict[str, int] = {}
    for case in cases:
        actual_counts[case.threat_model.value] = actual_counts.get(case.threat_model.value, 0) + 1
    if actual_counts != manifest.get("case_counts"):
        raise ValueError("benchmark case counts do not match the frozen manifest")
    return manifest
