# Contributing

```bash
git clone https://github.com/openlayer-ai/jevals && cd jevals
uv sync --extra dev
uv run pytest
uv run ruff check src tests && uv run ruff format src tests
```

Tests run against the mock backend and recorded wire shapes, so no API keys are needed.

## Adding an eval

1. Subclass `Eval` (or `NoulEval` / `ChoiceEval` / `ScoreEval` for one-question evals) in `src/jevals/agent/_evals.py`, `security/_evals.py`, or `quality/_evals.py`.
2. Keep `state()` small and deterministic. Describe options, don't label them.
3. Export it from the module's `__init__.py`; `jevals list` picks it up from there.
4. Add a test in `tests/test_evals.py` using `MockBackend(answers={...})`.

## What helps most

Calibration data. If you have labeled traces for any built-in eval, a `jevals calibrate` table for a real backend is worth more than a new eval.

## Reporting

Bugs and ideas go in GitHub issues. Security problems: see [SECURITY.md](SECURITY.md).
