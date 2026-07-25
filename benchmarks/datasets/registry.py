from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.datasets.base import BenchmarkAdapter
from benchmarks.datasets.models import DatasetManifest


class DatasetRegistry:
    def __init__(self, manifests_root: Path, cache_root: Path) -> None:
        self.manifests_root = manifests_root.resolve()
        self.cache_root = cache_root.resolve()
        self._manifests: dict[str, tuple[DatasetManifest, Path]] = {}
        for path in sorted(self.manifests_root.glob("*.json")):
            manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
            if manifest.name in self._manifests:
                raise ValueError(f"duplicate dataset manifest: {manifest.name}")
            self._manifests[manifest.name] = (manifest, path)

    def names(self) -> list[str]:
        return sorted(self._manifests)

    def get(self, name: str) -> DatasetManifest:
        try:
            return self._manifests[name][0]
        except KeyError as exc:
            raise KeyError(f"unknown dataset: {name}") from exc

    def manifest_path(self, name: str) -> Path:
        self.get(name)
        return self._manifests[name][1]

    def adapter(self, name: str) -> BenchmarkAdapter:
        manifest = self.get(name)
        module_name, class_name = manifest.adapter.rsplit(":", 1)
        module = importlib.import_module(module_name)
        adapter_type: Any = getattr(module, class_name)
        adapter = adapter_type(manifest, self.manifest_path(name))
        if not isinstance(adapter, BenchmarkAdapter):
            raise TypeError(f"{manifest.adapter} is not a BenchmarkAdapter")
        return adapter

    def cache_path(self, name: str) -> Path:
        manifest = self.get(name)
        return self.cache_root / f"{manifest.name}-{manifest.source.revision[:12]}"

    def verify(self, name: str, *, fixture_only: bool = False) -> dict[str, Any]:
        manifest = self.get(name)
        fixture = manifest.fixture_path(self.manifest_path(name))
        if not fixture.is_file():
            raise ValueError(f"missing fixture: {fixture}")
        fixture_digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if fixture_digest != manifest.fixture_sha256:
            raise ValueError(f"fixture digest mismatch for {name}")
        result: dict[str, Any] = {
            "dataset": name,
            "fixture_verified": True,
            "fixture_sha256": fixture_digest,
            "revision": manifest.source.revision,
        }
        if fixture_only:
            return result

        target = self.cache_path(name)
        if not (target / ".git").is_dir():
            raise ValueError(f"dataset cache is missing: {target}")
        revision = _git(["rev-parse", "HEAD"], cwd=target).strip()
        if revision != manifest.source.revision:
            raise ValueError(f"dataset revision mismatch for {name}: {revision}")
        missing = [item for item in manifest.source.required_paths if not (target / item).exists()]
        if missing:
            raise ValueError(f"dataset cache is missing required paths: {missing}")
        result.update({"cache_verified": True, "cache_path": str(target)})
        return result

    def fetch(self, name: str) -> dict[str, Any]:
        manifest = self.get(name)
        target = self.cache_path(name)
        if target.exists():
            return self.verify(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.fetching")
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            _git(
                [
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    manifest.source.repository,
                    str(temporary),
                ]
            )
            _git(
                ["fetch", "--depth", "1", "origin", manifest.source.revision],
                cwd=temporary,
            )
            _git(["checkout", "--detach", manifest.source.revision], cwd=temporary)
            temporary.rename(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.verify(name)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": manifest.name,
                "display_name": manifest.display_name,
                "version": manifest.version,
                "kind": manifest.kind,
                "holdout": manifest.holdout,
                "revision": manifest.source.revision,
                "tiers": manifest.tiers,
                "cached": self.cache_path(manifest.name).is_dir(),
            }
            for manifest, _ in (self._manifests[name] for name in self.names())
        ]


def _git(arguments: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def default_dataset_registry(
    *,
    cache_root: Path | None = None,
) -> DatasetRegistry:
    root = Path(__file__).resolve().parent
    return DatasetRegistry(
        root / "manifests",
        cache_root or Path(".cache/traceguard-datasets"),
    )
