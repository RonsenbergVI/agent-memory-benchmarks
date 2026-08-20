# Development commands. `make help` lists them.
# The benchmark itself is the `amb` CLI — see README.md; this file is only
# the local dev workflow, and deliberately does not wrap docker compose
# (integrations are run with their own compose files, by hand).

.DEFAULT_GOAL := help
.PHONY: help sync hooks lint format typecheck check test smoke report docs clean

help:  ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Install the workspace and dev dependencies
	uv sync

hooks:  ## Install the pre-commit git hook (one-off, per clone)
	uv run pre-commit install

lint:  ## Ruff lint with autofix
	uvx ruff check --fix .

format:  ## Ruff format
	uvx ruff format .

typecheck:  ## Static type check
	uvx ty check

check:  ## Everything pre-commit runs, on every file (run before committing)
	uv run pre-commit run --all-files

test:  ## Run the test suite
	uv run pytest -q

smoke:  ## Fastest end-to-end check: naive baseline, no API calls
	uv run amb run --system naive --dataset locomo --limit 1 --turns 40 --questions 5

report:  ## Aggregate run reports into a comparison table
	uv run amb report --latest

docs:  ## Regenerate the charts and the generated documents (RESULTS.md, README.md)
	uv run amb plot all
	uv run amb report --latest --output RESULTS.md --summary README.md

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache dist build
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
