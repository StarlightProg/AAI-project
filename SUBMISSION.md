# TraceGuard submission checklist

## 1. Credentials

- Rotate the Gemini key that was previously placed in `.env.example`.
- Keep the replacement only in an ignored `.env` or an exported shell variable.
- Confirm `git diff -- .env.example` shows only an empty placeholder.
- Never show `.env`, `env`, `printenv`, or shell history in the video.

## 2. Repository

```bash
source .venv/bin/activate
python -m pytest -q
ruff check .
ruff format --check .
python -m build
git status --short
```

Include source, tests, configs, benchmark manifests, documentation, `uv.lock`,
and the LaTeX report source. Do not include `.venv`, caches, or credentials.
Generated `artifacts/` are ignored; either submit the named final result directory
separately or regenerate it from the documented command.

## 3. Report

1. Replace all three red author placeholders in `report/main.tex`.
2. Compile the PDF using `report/README.md`.
3. Check that tables are legible and bibliography links work.
4. Submit both `report/main.tex` + `report/references.bib` and the compiled PDF.

## 4. Video

Follow `docs/demo.md`. The primary command is:

```bash
python -m traceguard demo --gemini
```

The command saves the exact sanitized traces shown in the recording. Close with
the scoped result: on the frozen 21-case custom benchmark, A7 had 0% attack
success and 100% benign utility; the result is not a universal robustness claim.

## 5. Final consistency check

- Report, README, and spoken video use the same benchmark size and metrics.
- Live Gemini output is presented as a live integration check unless a full,
  multi-seed live matrix has actually been completed.
- AgentDojo smoke evidence is not merged with the custom benchmark.
- The submitted commit contains no API keys, canaries, or raw unredacted traces.
