SHELL=/bin/bash
LINT_PATHS=tf_quant_finance/

build:
	python -m build

pytest:
	python -m pytest --no-header -vv

pylint:
	pylint tf_quant_finance --output-format=colorized

type: 
	mypy ${LINT_PATHS}

lint:
	ruff check tf_quant_finance --output-format=full

lint-complete: lint pylint

format:
	isort ${LINT_PATHS}
	black ${LINT_PATHS}

check-codestyle:
	black ${LINT_PATHS} --check

commit-checks: format type lint

release: 
	build
	twine upload dist/*

test-release: 
	build
	twine upload dist/* -r testpypi

.PHONY: clean spelling doc lint format check-codestyle commit-checks pylint