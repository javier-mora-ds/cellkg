#!/usr/bin/env python3
"""Phase 1 — download the public sources into data/raw/ (idempotent).

Sources: Ensembl GRCh38 gene annotation (GTF), Human Protein Atlas (JSON), Gene
Ontology (go-basic.obo), and STRING v12.0 Homo sapiens (links + info + aliases). The
HPA archive is decompressed to proteinatlas.json. Files already present are skipped.
Run `make organelles` first so the curated organelle CSVs exist under
data/raw/organelles/.
"""
import gzip
import os
import shutil
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
REL = os.environ.get("ENSEMBL_RELEASE", "112")

SOURCES = {
    f"Homo_sapiens.GRCh38.{REL}.gtf.gz": f"https://ftp.ensembl.org/pub/release-{REL}/gtf/homo_sapiens/Homo_sapiens.GRCh38.{REL}.gtf.gz",
    "go-basic.obo": "http://purl.obolibrary.org/obo/go/go-basic.obo",
    "9606.protein.links.v12.0.txt.gz": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "9606.protein.info.v12.0.txt.gz": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
    "9606.protein.aliases.v12.0.txt.gz": "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
}
HPA_GZ_URL = "https://www.proteinatlas.org/download/proteinatlas.json.gz"


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (present): {dest.name}")
        return
    print(f"downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def main():
    for name, url in SOURCES.items():
        download(url, RAW / name)
    # HPA: download gz then decompress to proteinatlas.json
    pj = RAW / "proteinatlas.json"
    if not (pj.exists() and pj.stat().st_size > 0):
        gz = RAW / "proteinatlas.json.gz"
        download(HPA_GZ_URL, gz)
        print("decompressing proteinatlas.json.gz ...")
        with gzip.open(gz, "rb") as fi, pj.open("wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        print("skip (present): proteinatlas.json")
    print("OK fetch_sources")


if __name__ == "__main__":
    main()
