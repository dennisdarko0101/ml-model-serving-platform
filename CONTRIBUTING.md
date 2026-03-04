# Contributing

## Development setup

```bash
# Clone and install dev dependencies
git clone <repo-url>
cd ml-model-serving-platform
pip install -e ".[dev]"

# Train sample models (used by tests)
make train-samples
```

## Workflow

1. Create a feature branch from `main`.
2. Make your changes with tests.
3. Run the full check suite:

```bash
make lint        # ruff linter
make typecheck   # mypy
make test        # pytest
```

4. Open a pull request.

## Code style

- **Formatter**: ruff format (line length 100)
- **Linter**: ruff check
- **Types**: use type hints; run mypy
- Python 3.11+ features are welcome (e.g. `X | Y` unions, `match` statements).

## Testing

- Unit tests go in `tests/unit/`.
- Integration tests in `tests/integration/`.
- Use real sklearn models where possible; mock PyTorch / ONNX.
- Aim for 80%+ coverage on new code.

## Commit messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
