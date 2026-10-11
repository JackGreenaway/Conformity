#!/usr/bin/env bash
# Shared documentation and release gates, executed from the repository root.
set -euo pipefail
TASK_PYTHON_VERSION=${TASK_PYTHON_VERSION:-3.12}
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release pytest --doctest-modules src tests
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release ruff check src/conformity tests
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release ruff check --select E4,E7,E9,F scripts
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release ruff format --check scripts
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release python scripts/build_api_reference.py --check
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release python scripts/check_docs.py
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release mkdocs build --strict
uv run --locked --python "$TASK_PYTHON_VERSION" --group docs --group release python scripts/check_site_links.py
