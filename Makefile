.PHONY: install test lint params notebook report clean

SOFFICE ?= soffice

install:
	pip install -e ".[notebook,dev]"

test:
	pytest

lint:
	ruff check src tests report

params:
	cifar-params configs/*.yaml --summary

notebook:
	python notebooks/build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/project1.ipynb

report:
	python report/make_figures.py
	cd report && npm ci --no-audit --no-fund && node build_report.js
	cd report && $(SOFFICE) --headless --convert-to pdf report.docx

clean:
	rm -rf .pytest_cache .ruff_cache .ipynb_checkpoints report/node_modules
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
