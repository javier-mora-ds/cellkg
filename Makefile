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

# ---- GNN track (additive; `all` is unchanged) --------------------------------
.PHONY: gnn-setup lock gnn-test gnn-graph gnn-splits gnn-kge gnn-baselines gnn-train gnn-ablations gnn-xperm \
        gnn-eval gnn-compare gnn gnn-fast all-gnn clean-gnn clean-gnn-all

GNNPY   := PYTHONPATH=src:ml $(PY)
GNN_CFG ?= ml/gnn/config/default.yaml
GNN_FLAGS ?=
GNN_JOBS ?= 4

gnn-setup:            ## declare and verify GNN dependencies in the existing venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import torch, torch_geometric, iterstrat, sklearn, matplotlib, yaml, pytest; print('torch', torch.__version__, 'pyg', torch_geometric.__version__)"

lock:                 ## refreeze requirements.lock after a deliberate dependency change
	$(PY) -m pip freeze > requirements.lock

gnn-test:             ## fast pytest smoke tests (no training)
	$(GNNPY) -m pytest tests -q -m "not slow"

gnn-graph:            ## data/import CSVs -> HeteroData edges + gene features (-> reports/gnn/graph/)
	$(GNNPY) -m gnn.graph --config $(GNN_CFG) $(GNN_FLAGS)

gnn-splits:           ## P1 = exact PyKEEN replay (Gate A; Neo4j once, then persisted); P2 = 5 stratified gene folds
	$(GNNPY) -m gnn.splits --config $(GNN_CFG) $(GNN_FLAGS)

gnn-kge:              ## re-train TransE + RotatE (identical config), save, score matrices, Gate B
	$(GNNPY) -m gnn.kge --config $(GNN_CFG) --jobs $(GNN_JOBS) $(GNN_FLAGS)

gnn-baselines:        ## popularity prior, neighbour vote / propagation, feature-only perceptron
	$(GNNPY) -m gnn.baselines --config $(GNN_CFG) --jobs $(GNN_JOBS) $(GNN_FLAGS)

gnn-train:            ## GraphSAGE: plain BCE and logit-adjusted BCE, both protocols, 5 seeds
	$(GNNPY) -m gnn.train --config $(GNN_CFG) --protocol all --loss bce --jobs $(GNN_JOBS) $(GNN_FLAGS)
	$(GNNPY) -m gnn.train --config $(GNN_CFG) --protocol all --loss logit_adjusted --jobs $(GNN_JOBS) $(GNN_FLAGS)

gnn-ablations:        ## class-balanced and asymmetric losses (1 seed); sanity check S4 (rho = 0, P1-full)
	$(GNNPY) -m gnn.train --config $(GNN_CFG) --protocol all --loss class_balanced --jobs $(GNN_JOBS) $(GNN_FLAGS)
	$(GNNPY) -m gnn.train --config $(GNN_CFG) --protocol all --loss asl --jobs $(GNN_JOBS) $(GNN_FLAGS)
	$(GNNPY) -m gnn.train --config $(GNN_CFG) --protocol p1 --regimes full --loss logit_adjusted --rho 0 --seeds 20260702 --jobs 1 $(GNN_FLAGS)

gnn-xperm:            ## topology-only control: perceptron and GraphSAGE (BCE) on row-permuted gene attributes; adds *_xperm* artifacts only
	$(GNNPY) -m gnn.train --config $(GNN_CFG) --protocol all --model mlp --loss bce --features perm --jobs $(GNN_JOBS) $(GNN_FLAGS)
	$(GNNPY) -m gnn.train --config $(GNN_CFG) --protocol all --model sage --loss bce --features perm --jobs $(GNN_JOBS) $(GNN_FLAGS)

gnn-eval:             ## the harness over every score matrix -> reports/gnn/metrics/
	$(GNNPY) -m gnn.evaluate --config $(GNN_CFG) $(GNN_FLAGS)

gnn-compare:          ## comparison tables + figures -> reports/gnn/
	$(GNNPY) -m gnn.compare --config $(GNN_CFG) $(GNN_FLAGS)

gnn: gnn-graph gnn-splits gnn-kge gnn-baselines gnn-train gnn-ablations gnn-eval gnn-compare

gnn-fast:             ## smoke run of the whole track on CPU into reports/gnn_fast (never overwrites reports/gnn)
	OUT=$${CELLKG_GNN_OUT:-$(CURDIR)/reports/gnn_fast}; mkdir -p $$OUT/export; \
	cp -n reports/gnn/export/*.parquet $$OUT/export/ 2>/dev/null || true; \
	CELLKG_GNN_OUT=$$OUT $(MAKE) gnn GNN_FLAGS="--fast --device cpu"

all-gnn: all gnn      ## the original chain followed by the GNN track

clean-gnn:            ## remove GNN artifacts but keep the persisted export and splits
	find reports/gnn -mindepth 1 -maxdepth 1 ! -name export ! -name splits -exec rm -rf {} +

clean-gnn-all:        ## remove every GNN artifact (a later gnn-splits needs Neo4j again for the exact replay)
	rm -rf reports/gnn .pytest_cache
