# Building the TraceGuard report

Update the three author placeholders in `main.tex`, then compile from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable, the equivalent manual sequence is:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The report intentionally distinguishes the 168-episode frozen custom benchmark
from the smaller live-provider and AgentDojo integration checks. Do not replace
those labels with a broader robustness claim.
