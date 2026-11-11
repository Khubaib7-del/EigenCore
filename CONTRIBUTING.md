# Contributing to EigenCore

Thanks for helping. EigenCore is a small, pre-alpha project, so this guide is short.

## Before you start

- Read the [review and plan](docs/review/FINAL_REPORT.md); section 8 lists the work in priority order.
- For anything larger than a small fix, open an issue first so we can agree on the approach.

## Setup

```bash
git clone https://github.com/Khubaib7-del/EigenCore.git
cd EigenCore
python -m pip install -e ".[dev,inference]"
pytest tests/ -v
ruff check src/ tests/
ruff format --check src/ tests/
```

CI runs on Ubuntu and Windows with Python 3.10 and 3.12. Ruff is pinned below 0.15 on purpose.

## Rules that keep the project honest

1. **Label claims.** Anything user-facing (README, docs, CLI or TUI output) is SHIPPED, REPORTED-ONLY or PLANNED. Don't describe a plan as a feature.
2. **No unmeasured numbers.** No "x faster", "speedup" or tok/s unless it was measured, with the machine, model, quant, context, threads and backend version recorded. Arithmetic ceilings are labelled "estimate".
3. **Correctness before speed.** New kernels or optimizations need a reference comparison (for kernels: gguf-py / ggml), a fallback, and a benchmark through `eigencore benchmark` before they can be enabled by default.
4. **Safety stays below the UI.** Tool permissions are enforced in the policy layer, never only in prompts or rendering.
5. **No binary files** (images, decks, models, archives) in pull requests unless a maintainer asks.

## Pull requests

- One concern per PR, with tests.
- Add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).
- Use [Conventional Commit](https://www.conventionalcommits.org/) prefixes (`feat:`, `fix:`, `docs:`, `test:`, `ci:`), as the history already does.
- Fill in the PR template, including which status label the change affects.

## Benchmarks

If your change claims a performance effect, attach the JSON from `eigencore benchmark --report run.json` (or `llama-bench -o json`) and name the hardware. Maintainers may ask for a run on the reference laptop before merging.

## Conduct

By participating you agree to the [code of conduct](CODE_OF_CONDUCT.md).
