.PHONY: test lint build-web

test:
	pytest tests -v

lint:
	ruff check src apps tests

build-web:
	cd apps/web && npm run build
