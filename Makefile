# CellKG — build orchestration. Reproduces the graph and ML from data/raw/.
SHELL := /bin/zsh
PY := .venv/bin/python

.PHONY: setup organelles fetch build load validate query ml all clean

setup:                ## create venv (reuse base torch/pandas) and install deps
	python -m venv .venv --system-site-packages
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

organelles:           ## write curated organelle CSVs -> data/raw/organelles/
	$(PY) src/make_organelles.py

fetch:                ## download public sources -> data/raw/
	$(PY) src/fetch_sources.py

build:                ## ETL: raw sources -> neo4j-admin CSVs in data/import/
	$(PY) src/build_ensembl.py
	$(PY) src/build_hpa.py
	$(PY) src/build_go.py
	$(PY) src/build_string.py
	$(PY) src/emit_csvs.py

load:                 ## bulk-import into Neo4j, start server, apply DDL, verify counts
	zsh src/load.sh

validate:             ## characterization + interesting-network acceptance gates
	PYTHONPATH=src $(PY) src/validate.py

query:                ## run the 10 competency questions -> reports/
	PYTHONPATH=src $(PY) src/run_competency.py

ml:                   ## PyKEEN TransE + RotatE link prediction -> reports/
	PYTHONPATH=src $(PY) ml/run_pykeen.py

all: organelles fetch build load validate query ml

clean:                ## remove derived intermediates and import CSVs
	rm -rf data/interim/* data/import/*
