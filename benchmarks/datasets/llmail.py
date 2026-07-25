from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def normalize_payload(value: str) -> str:
    """Normalize only for grouping; raw attack text must remain unchanged."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def payload_family_ids(
    payloads: Iterable[str],
    *,
    threshold: float = 0.85,
    seed: int = 0,
) -> list[str]:
    """Group near duplicates with datasketch MinHashLSH."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise RuntimeError("install TraceGuard with the 'datasets' extra") from exc

    values = list(payloads)
    sketches = []
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    parents = list(range(len(values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for index, payload in enumerate(values):
        sketch = MinHash(num_perm=128, seed=seed)
        text = normalize_payload(payload)
        shingles = {text[pos : pos + 5] for pos in range(max(1, len(text) - 4))}
        for shingle in sorted(shingles):
            sketch.update(shingle.encode("utf-8"))
        for candidate in lsh.query(sketch):
            union(index, int(candidate))
        lsh.insert(str(index), sketch)
        sketches.append(sketch)

    roots = {root: position for position, root in enumerate(sorted({find(i) for i in parents}))}
    return [f"payload-family-{roots[find(index)]:06d}" for index in range(len(values))]


@dataclass(frozen=True)
class LLMailRecord:
    record_id: str
    participant_id: str
    payload_family_id: str
    scenario: str
    phase: str
    payload: str
    successful: bool


def validate_group_disjoint_splits(splits: dict[str, list[LLMailRecord]]) -> list[str]:
    errors: list[str] = []
    participant_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in splits.items():
        for record in records:
            participant_splits[record.participant_id].add(split)
            family_splits[record.payload_family_id].add(split)
    for participant, assigned in participant_splits.items():
        if len(assigned) > 1:
            errors.append(f"participant {participant} crosses splits: {sorted(assigned)}")
    for family, assigned in family_splits.items():
        if len(assigned) > 1:
            errors.append(f"payload family {family} crosses splits: {sorted(assigned)}")
    return errors


def stratified_selection(
    records: list[LLMailRecord],
    *,
    limit: int,
) -> list[LLMailRecord]:
    """Deterministic round-robin over scenario, phase, and outcome strata."""
    buckets: dict[tuple[str, str, bool], list[LLMailRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.record_id):
        buckets[(record.scenario, record.phase, record.successful)].append(record)
    selected: list[LLMailRecord] = []
    ordered_keys = sorted(buckets)
    index = 0
    while len(selected) < min(limit, len(records)):
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
        index += 1
    return selected


def record_from_mapping(item: dict[str, Any], family_id: str) -> LLMailRecord:
    return LLMailRecord(
        record_id=str(item.get("id") or item.get("submission_id")),
        participant_id=str(item.get("team_id") or item.get("participant_id")),
        payload_family_id=family_id,
        scenario=str(item.get("scenario")),
        phase=str(item.get("phase", "unknown")),
        payload=f"{item.get('subject', '')}\n{item.get('body', '')}".strip(),
        successful=bool(item.get("successful") or item.get("api_called")),
    )
