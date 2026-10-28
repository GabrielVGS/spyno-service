.PHONY: sync lint test ci docker

sync:
	uv sync --frozen --extra dev

lint:
	uv run --frozen ruff check .

test:
	uv run --frozen pytest -q

ci: lint test

docker:
	docker build -t spyno-service:local .
