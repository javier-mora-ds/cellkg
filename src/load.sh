#!/usr/bin/env zsh
# Phase 3 — reproducible load: neo4j-admin bulk import, start server, apply DDL, verify.
set -e
ROOT=${0:A:h:h}                      # cellkg/ (this script lives in cellkg/src/)
NEO=$ROOT/server/neo4j
PY=$ROOT/.venv/bin/python
IMP=$ROOT/data/import
export JAVA_HOME=${JAVA_HOME:-$(/usr/libexec/java_home 2>/dev/null || true)}

source $ROOT/.env
$NEO/bin/neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD" >/dev/null 2>&1 || true
$NEO/bin/neo4j stop >/dev/null 2>&1 || true    # import needs the store offline

$NEO/bin/neo4j-admin database import full neo4j \
  --nodes=$IMP/gene_nodes.csv --nodes=$IMP/chromosome_nodes.csv --nodes=$IMP/cc_nodes.csv \
  --nodes=$IMP/bp_nodes.csv --nodes=$IMP/mf_nodes.csv --nodes=$IMP/protein_nodes.csv \
  --relationships=CONTAINS=$IMP/contains.csv --relationships=LOCATED_IN=$IMP/located_in.csv \
  --relationships=PARTICIPATES_IN=$IMP/participates_in.csv --relationships=HAS_FUNCTION=$IMP/has_function.csv \
  --relationships=ENCODES=$IMP/encodes.csv --relationships=INTERACTS_WITH=$IMP/interacts_with.csv \
  --relationships=ADJACENT_TO=$IMP/adjacent_to.csv --relationships=PART_OF=$IMP/part_of.csv \
  --relationships=SUBCLASS_OF=$IMP/subclass_of.csv \
  --overwrite-destination=true

$NEO/bin/neo4j start
PYTHONPATH=$ROOT/src $PY $ROOT/src/apply_ddl_verify.py
