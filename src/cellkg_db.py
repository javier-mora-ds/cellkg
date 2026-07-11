#!/usr/bin/env python3
"""Shared Neo4j connection helper for CellKG: loads .env and returns a ready driver."""
import time
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]


def env():
    e = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            e[k] = v
    return e


def driver(wait=True, timeout=120):
    e = env()
    d = GraphDatabase.driver(e["NEO4J_URI"], auth=(e["NEO4J_USER"], e["NEO4J_PASSWORD"]))
    if wait:
        deadline = time.time() + timeout
        while True:
            try:
                d.verify_connectivity()
                return d
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(2)
    return d


def database():
    return env().get("NEO4J_DB", "neo4j")
