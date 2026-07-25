from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from benchmarks.datasets.models import (
    ADAPTER_PROTOCOL_VERSION,
    AdapterRunRequest,
    NormalizedBenchmarkResult,
)


def run_external_adapter(
    command: Sequence[str],
    request: AdapterRunRequest,
    *,
    cwd: Path,
    timeout: float = 3600,
) -> list[NormalizedBenchmarkResult]:
    """Run a dependency-isolated adapter over a strict JSON stdin/stdout protocol."""
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ValueError("external adapter command requires a non-empty argv")
    result = subprocess.run(
        list(command),
        cwd=cwd,
        input=request.model_dump_json() + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONNOUSERSITE": "1"},
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "external adapter failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("external adapter returned malformed JSON") from exc
    if not isinstance(payload, list):
        raise RuntimeError("external adapter must return a JSON list")
    parsed = [NormalizedBenchmarkResult.model_validate(item) for item in payload]
    if any(item.protocol_version != ADAPTER_PROTOCOL_VERSION for item in parsed):
        raise RuntimeError("external adapter protocol version mismatch")
    if any(item.dataset != request.dataset for item in parsed):
        raise RuntimeError("external adapter returned the wrong dataset")
    return parsed
