.PHONY: tests coverage check lint format-check format hooks

tests:
	uv run pytest -q

coverage:
	uv run coverage run -m pytest
	uv run coverage report

check: lint format-check tests

lint:
	uv run ruff check .

format-check:
	uv run black --check .

format:
	uv run black .

hooks:
	uv run pre-commit install
