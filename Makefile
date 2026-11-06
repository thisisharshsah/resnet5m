.PHONY: install test lint params notebook clean

install:
	pip install -e ".[notebook,dev]"

test:
	pytest

lint:
	ruff check src tests

params:
	cifar-params configs/*.yaml --summary

notebook:
	python notebooks/build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/project1.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache .ipynb_checkpoints
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
