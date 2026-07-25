from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.schema import BenchmarkCase

ADAPTER_PROTOCOL_VERSION = "benchmark-adapter-v1"
DatasetTier = Literal["smoke", "standard", "full"]
DatasetKind = Literal["diagnostic", "episode", "native"]
HoldoutStatus = Literal["development", "evaluation", "sealed"]


class StrictDatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSource(StrictDatasetModel):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    required_paths: list[str] = Field(default_factory=list)


class DatasetManifest(StrictDatasetModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    display_name: str
    version: str
    adapter: str
    adapter_version: str = "1"
    kind: DatasetKind
    holdout: HoldoutStatus
    source: DatasetSource
    tiers: dict[DatasetTier, int | None]
    fixture: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_environment: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tiers(self) -> DatasetManifest:
        if "smoke" not in self.tiers:
            raise ValueError("every dataset requires a smoke tier")
        if self.holdout == "sealed" and "standard" in self.tiers:
            raise ValueError("sealed datasets cannot expose a standard development tier")
        return self

    def fixture_path(self, manifest_path: Path) -> Path:
        return (manifest_path.parent.parent / "fixtures" / self.fixture).resolve()


class AdapterRunRequest(StrictDatasetModel):
    protocol_version: str = ADAPTER_PROTOCOL_VERSION
    dataset: str
    tier: DatasetTier = "smoke"
    seed: int = 0
    supervisor_mode: str = "deterministic"
    prompt_digest: str | None = None
    policy_digest: str | None = None
    environment_lock_digest: str | None = None
    dataset_cache_path: str | None = None
    external_command: list[str] | None = None


class ToolAttempt(StrictDatasetModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision: str | None = None
    executed: bool = False
    unsafe: bool = False


class NormalizedBenchmarkResult(StrictDatasetModel):
    protocol_version: str = ADAPTER_PROTOCOL_VERSION
    dataset: str
    dataset_revision: str
    adapter_version: str
    native_case_id: str
    case_id: str
    tier: DatasetTier
    seed: int
    threat_model: str
    attack_family: str | None = None
    utility_passed: bool
    security_passed: bool
    attacker_goal_achieved: bool
    prohibited_effect: bool
    native_scores: dict[str, float | bool | None] = Field(default_factory=dict)
    tool_attempts: list[ToolAttempt] = Field(default_factory=list)
    containment_evidence: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetCaseBatch(StrictDatasetModel):
    dataset: str
    tier: DatasetTier
    cases: list[BenchmarkCase]
