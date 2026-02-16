.PHONY: all ingest-preprocess analysis notebooks

all: ingest-preprocess analysis notebooks

ingest-preprocess:
	python scripts/01_ingest_and_preprocess.py --config configs/ingest_and_preprocess.yaml

analysis:
	python scripts/03_coupled_test.py --config configs/analysis.yaml

notebooks:
	python scripts/run_notebook.py notebooks/coupling_test.ipynb

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
