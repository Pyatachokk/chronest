.PHONY: format lint build apidoc build_doc

format:
	poetry run isort .
	poetry run black .

lint:
	poetry run flake8 .

build:
	poetry build

build_doc:
	poetry run make -C docs html
	cp -a docs/_build/html/. docs/
