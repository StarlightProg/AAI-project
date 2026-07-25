from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from benchmarks.datasets.models import (
    AdapterRunRequest,
    DatasetManifest,
    NormalizedBenchmarkResult,
)
from benchmarks.schema import BenchmarkCase


class BenchmarkAdapter(ABC):
    """Boundary between TraceGuard and a pinned upstream benchmark."""

    def __init__(self, manifest: DatasetManifest, manifest_path: Path) -> None:
        self.manifest = manifest
        self.manifest_path = manifest_path

    @abstractmethod
    def load_cases(self, request: AdapterRunRequest) -> list[BenchmarkCase]:
        """Load normalized cases without executing external effects."""

    @abstractmethod
    def run(
        self,
        request: AdapterRunRequest,
        cases: Sequence[BenchmarkCase] | None = None,
    ) -> list[NormalizedBenchmarkResult]:
        """Run or delegate cases and return normalized results."""

    def fixture_path(self) -> Path:
        return self.manifest.fixture_path(self.manifest_path)
