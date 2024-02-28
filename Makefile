.PHONY: format lint build apidoc build_doc test

format:
	poetry run isort .
	poetry run black .

lint:
	poetry run flake8 .

build:
	poetry build

build_doc:
	poetry run make -C docs html
	touch docs/_build/html/.nojekyll
	cp docs/_build/html/.nojekyll docs/
	cp -r docs/_build/html/* docs/

test:
	python -m pytest

publish_test:
	poetry build
	poetry publish --repository testpypi

