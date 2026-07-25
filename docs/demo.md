# TraceGuard Short Demo

The demo takes about three minutes and requires no credentials.

## 1. Show normal utility

```bash
python -m traceguard smoke
```

Point out that the calculator call is typed, approved, executed, and returned
with provenance.

## 2. Show the frozen comparison

```bash
python -m traceguard smoke-matrix --seed 0
```

Open the printed run directory, then compare `manifest_A0.json` with
`manifest_A7.json`. In `representative_traces.json`, show the proposed unsafe
call, the supervisor decision, and the absence of an exposed unsafe result.

## 3. Show real containment

```bash
python -m traceguard sandbox-check
TRACEGUARD_RUN_DOCKER_TESTS=1 python -m pytest tests/sandbox -q
```

Emphasize the pinned digest, ARM64 check, non-root execution, no network,
read-only root filesystem, bounded resources, and verified cleanup.

## 4. Close with the result

The frozen 168-episode run preserved 100% benign utility. The complete hybrid
had 0% attack success and 0% compromise on the custom benchmark; the
no-supervisor baseline had 40% attack success and 33.3% compromise. These
numbers describe the frozen custom benchmark and should not be generalized
beyond it without the AgentDojo/live-model comparison.
