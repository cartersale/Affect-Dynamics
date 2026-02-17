.PHONY: all ingest-preprocess coupling hmm catrqa analysis clean

all: ingest-preprocess coupling hmm catrqa

ingest-preprocess:
	python scripts/01_ingest_and_preprocess.py --config configs/ingest_and_preprocess.yaml

coupling:
	python scripts/02_coupling_test.py --config configs/analysis.yaml

hmm:
	python scripts/03_fit_shared_hmm.py --config configs/analysis.yaml

catrqa:
	python scripts/04_catRQA.py --config configs/analysis.yaml

analysis: coupling hmm catrqa 

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
