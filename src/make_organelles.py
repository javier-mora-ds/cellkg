#!/usr/bin/env python3
"""Generate the curated organelle inputs for CellKG into data/raw/organelles/.

Five CSVs are produced, encoding the ontology-grounded compartment layer:
  organelles.csv          22 anchors: code,name,description,function,go_id,go_label,confidence
  organelle_part_of.csv   containment tree (child,parent), CEL-rooted
  organelle_adjacent_to.csv  43 undirected adjacency pairs (a,b)
  hpa_crosswalk.csv       HPA subcellular term -> organelle code (exact HPA casing)
  biotype_groups.csv      raw Ensembl gene_biotype -> grouped Gene sub-label
"""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "organelles"
OUT.mkdir(parents=True, exist_ok=True)

# code, name, description, function, go_id, go_label, confidence
ORGANELLES = [
    (
        "CEL",
        "Cell",
        "The human cell; the root cellular anatomical entity.",
        "Basic structural and functional unit of life.",
        "GO:0110165",
        "cellular anatomical entity",
        "remapped",
    ),
    (
        "PMB",
        "Plasma Membrane",
        "A phospholipid bilayer surrounding the cell, embedded with proteins, glycoproteins and cholesterol.",
        "Regulates entry and exit of substances; structural support; cell signalling.",
        "GO:0005886",
        "plasma membrane",
        "high",
    ),
    (
        "CTP",
        "Cytoplasm",
        "A gel-like substance of water, salts and proteins filling the cell interior.",
        "Medium for metabolic reactions; suspends organelles.",
        "GO:0005737",
        "cytoplasm",
        "high",
    ),
    (
        "NUC",
        "Nucleus",
        "A membrane-bound organelle containing the cell's genetic material.",
        "Stores genetic information; regulates gene expression.",
        "GO:0005634",
        "nucleus",
        "high",
    ),
    (
        "NVL",
        "Nuclear Envelope",
        "A double membrane enclosing the nucleus, with nuclear pores for exchange.",
        "Separates nucleoplasm from cytoplasm; controls nucleocytoplasmic transport.",
        "GO:0005635",
        "nuclear envelope",
        "high",
    ),
    (
        "CRM",
        "Chromatin",
        "DNA and associated histone proteins that condense into chromosomes during division.",
        "Packages DNA; regulates transcription and replication.",
        "GO:0000785",
        "chromatin",
        "high",
    ),
    (
        "NCL",
        "Nucleolus",
        "A dense nuclear region where rRNA synthesis and ribosome assembly occur.",
        "Ribosomal RNA synthesis and ribosome subunit assembly.",
        "GO:0005730",
        "nucleolus",
        "high",
    ),
    (
        "MTC",
        "Mitochondria",
        "Double-membrane organelles with their own DNA, cristae and matrix.",
        "ATP production via oxidative phosphorylation.",
        "GO:0005739",
        "mitochondrion",
        "high",
    ),
    (
        "RBS",
        "Ribosomes",
        "Complexes of rRNA and protein, free in cytoplasm or bound to the ER.",
        "Protein synthesis (translation).",
        "GO:0005840",
        "ribosome",
        "high",
    ),
    (
        "ERT",
        "Endoplasmic Reticulum",
        "A network of membranous tubules and sacs.",
        "Synthesis, folding and transport of proteins and lipids.",
        "GO:0005783",
        "endoplasmic reticulum",
        "high",
    ),
    (
        "RER",
        "Rough ER",
        "Endoplasmic reticulum studded with ribosomes.",
        "Synthesis and modification of secretory and membrane proteins.",
        "GO:0005791",
        "rough endoplasmic reticulum",
        "high",
    ),
    (
        "SER",
        "Smooth ER",
        "Endoplasmic reticulum lacking ribosomes.",
        "Lipid synthesis, detoxification, calcium storage.",
        "GO:0005790",
        "smooth endoplasmic reticulum",
        "high",
    ),
    (
        "GLG",
        "Golgi Apparatus",
        "Stacks of flattened membranous cisternae.",
        "Modification, sorting and packaging of proteins and lipids.",
        "GO:0005794",
        "Golgi apparatus",
        "high",
    ),
    (
        "LYS",
        "Lysosomes",
        "Membrane-bound vesicles containing digestive enzymes.",
        "Intracellular digestion and recycling.",
        "GO:0005764",
        "lysosome",
        "high",
    ),
    (
        "PRX",
        "Peroxisomes",
        "Membrane-bound organelles with fatty-acid-metabolism and detox enzymes.",
        "Fatty acid oxidation; reactive oxygen species detoxification.",
        "GO:0005777",
        "peroxisome",
        "high",
    ),
    (
        "CTK",
        "Cytoskeleton",
        "A network of microfilaments, intermediate filaments and microtubules.",
        "Cell shape, mechanical support, intracellular transport and motility.",
        "GO:0005856",
        "cytoskeleton",
        "high",
    ),
    (
        "CTS",
        "Centrosomes and Centrioles",
        "The main microtubule-organising centre with a pair of centrioles.",
        "Microtubule organisation; spindle formation in mitosis.",
        "GO:0005813",
        "centrosome",
        "medium",
    ),
    (
        "VSC",
        "Vesicles and Vacuoles",
        "Membrane-bound sacs for transport and storage.",
        "Transport and storage of cellular materials.",
        "GO:0031982",
        "vesicle",
        "low",
    ),
    (
        "TRV",
        "Transport Vesicles",
        "Vesicles moving materials between organelles and the cell surface.",
        "Intracellular and surface-directed material transport.",
        "GO:0030133",
        "transport vesicle",
        "high",
    ),
    (
        "STV",
        "Storage Vacuoles",
        "Sacs storing nutrients, waste and other materials.",
        "Storage of nutrients, waste and other materials.",
        "GO:0000322",
        "storage vacuole",
        "medium",
    ),
    (
        "MCV",
        "Microvilli",
        "Microscopic membrane protrusions.",
        "Increase surface area for absorption.",
        "GO:0005902",
        "microvillus",
        "high",
    ),
    (
        "EMC",
        "Extracellular Matrix",
        "A network of proteins and carbohydrates outside the plasma membrane.",
        "Structural and biochemical support to surrounding cells.",
        "GO:0031012",
        "extracellular matrix",
        "high",
    ),
]

# containment tree (child -> parent), CEL-rooted
PART_OF = []
for child in [
    "PMB",
    "CTP",
    "NUC",
    "MTC",
    "RBS",
    "ERT",
    "GLG",
    "LYS",
    "PRX",
    "CTK",
    "CTS",
    "VSC",
    "MCV",
    "EMC",
]:
    PART_OF.append((child, "CEL"))
for child in ["NVL", "CRM", "NCL"]:
    PART_OF.append((child, "NUC"))
for child in ["RER", "SER"]:
    PART_OF.append((child, "ERT"))
for child in ["TRV", "STV"]:
    PART_OF.append((child, "VSC"))

# 43 undirected adjacency pairs
ADJACENT = [
    ("CEL", "CTK"),
    ("CEL", "CTP"),
    ("CEL", "CTS"),
    ("CEL", "EMC"),
    ("CEL", "GLG"),
    ("CEL", "LYS"),
    ("CEL", "MCV"),
    ("CEL", "MTC"),
    ("CEL", "NUC"),
    ("CEL", "PMB"),
    ("CEL", "PRX"),
    ("CEL", "RBS"),
    ("CEL", "RER"),
    ("CEL", "SER"),
    ("CEL", "STV"),
    ("CEL", "TRV"),
    ("CRM", "NUC"),
    ("CTK", "CTS"),
    ("CTK", "GLG"),
    ("CTK", "MTC"),
    ("CTK", "NUC"),
    ("CTK", "PMB"),
    ("CTP", "LYS"),
    ("CTP", "MTC"),
    ("CTP", "PMB"),
    ("CTP", "PRX"),
    ("CTP", "RBS"),
    ("EMC", "MCV"),
    ("EMC", "PMB"),
    ("GLG", "LYS"),
    ("GLG", "RER"),
    ("GLG", "SER"),
    ("GLG", "TRV"),
    ("LYS", "TRV"),
    ("MCV", "PMB"),
    ("NCL", "NUC"),
    ("NUC", "NVL"),
    ("NUC", "RBS"),
    ("PMB", "TRV"),
    ("RBS", "RER"),
    ("RER", "TRV"),
    ("SER", "TRV"),
    ("STV", "TRV"),
]

# HPA subcellular term (exact casing as in proteinatlas.json) -> organelle code
HPA_CROSSWALK = [
    ("Nucleoplasm", "NUC"),
    ("Cytosol", "CTP"),
    ("Vesicles", "TRV"),
    ("Plasma membrane", "PMB"),
    ("Mitochondria", "MTC"),
    ("Golgi apparatus", "GLG"),
    ("Nucleoli", "NCL"),
    ("Endoplasmic reticulum", "ERT"),
    ("Nuclear speckles", "NUC"),
    ("Nuclear bodies", "NUC"),
    ("Centrosome", "CTS"),
    ("Microtubules", "CTK"),
    ("Nucleoli fibrillar center", "NCL"),
    ("Cell Junctions", "PMB"),
    ("Nuclear membrane", "NVL"),
    ("Actin filaments", "CTK"),
    ("Nucleoli rim", "NCL"),
    ("Intermediate filaments", "CTK"),
    ("Centriolar satellite", "CTS"),
    ("Focal adhesion sites", "PMB"),
    ("Cytoplasmic bodies", "CTP"),
    ("Lipid droplets", "CTP"),
    ("Mitotic chromosome", "CRM"),
    ("Cytokinetic bridge", "CTK"),
    ("Peroxisomes", "PRX"),
    ("Midbody", "CTK"),
    ("Mitotic spindle", "CTK"),
    ("Lysosomes", "LYS"),
    ("Endosomes", "TRV"),
    ("Rods & Rings", "CTP"),
    ("Midbody ring", "CTK"),
    ("Microtubule ends", "CTK"),
    ("Aggresome", "CTP"),
    ("Kinetochore", "CRM"),
    ("Cleavage furrow", "PMB"),
]

# raw Ensembl gene_biotype -> grouped Gene sub-label
BIOTYPE_GROUPS = {
    "protein_coding": "ProteinCoding",
    "lncRNA": "LncRNA",
    "miRNA": "MiRNA",
    "snoRNA": "SnoRNA",
    "scaRNA": "SnoRNA",
    "snRNA": "SnRNA",
    "misc_RNA": "SmallRNA",
    "rRNA": "SmallRNA",
    "Mt_tRNA": "SmallRNA",
    "Mt_rRNA": "SmallRNA",
    "sRNA": "SmallRNA",
    "scRNA": "SmallRNA",
    "vault_RNA": "SmallRNA",
    "ribozyme": "SmallRNA",
    "processed_pseudogene": "Pseudogene",
    "unprocessed_pseudogene": "Pseudogene",
    "transcribed_unprocessed_pseudogene": "Pseudogene",
    "transcribed_processed_pseudogene": "Pseudogene",
    "rRNA_pseudogene": "Pseudogene",
    "transcribed_unitary_pseudogene": "Pseudogene",
    "IG_V_pseudogene": "Pseudogene",
    "unitary_pseudogene": "Pseudogene",
    "TR_V_pseudogene": "Pseudogene",
    "IG_C_pseudogene": "Pseudogene",
    "TR_J_pseudogene": "Pseudogene",
    "IG_J_pseudogene": "Pseudogene",
    "translated_processed_pseudogene": "Pseudogene",
    "IG_pseudogene": "Pseudogene",
    "IG_V_gene": "IGgene",
    "IG_D_gene": "IGgene",
    "IG_J_gene": "IGgene",
    "IG_C_gene": "IGgene",
    "TR_V_gene": "TRgene",
    "TR_J_gene": "TRgene",
    "TR_C_gene": "TRgene",
    "TR_D_gene": "TRgene",
    "TEC": "OtherRNA",
    "artifact": "OtherRNA",
}


def write_csv(name, header, rows):
    p = OUT / name
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p, len(rows)


def main():
    p1, n1 = write_csv(
        "organelles.csv",
        ["code", "name", "description", "function", "go_id", "go_label", "confidence"],
        ORGANELLES,
    )
    p2, n2 = write_csv("organelle_part_of.csv", ["child", "parent"], PART_OF)
    p3, n3 = write_csv("organelle_adjacent_to.csv", ["a", "b"], ADJACENT)
    p4, n4 = write_csv("hpa_crosswalk.csv", ["hpa_term", "code"], HPA_CROSSWALK)
    p5, n5 = write_csv(
        "biotype_groups.csv", ["biotype", "biotype_group"], sorted(BIOTYPE_GROUPS.items())
    )

    codes = {o[0] for o in ORGANELLES}
    assert n1 == 22, f"expected 22 organelles, got {n1}"
    assert n2 == 21, f"expected 21 part_of edges, got {n2}"
    assert n3 == 43, f"expected 43 adjacency pairs, got {n3}"
    assert n4 == 35, f"expected 35 HPA crosswalk rows, got {n4}"
    assert all(c in codes for _, c in HPA_CROSSWALK), "crosswalk maps to unknown organelle code"
    assert all(ch in codes and pa in codes for ch, pa in PART_OF), "part_of references unknown code"
    assert all(a in codes and b in codes for a, b in ADJACENT), "adjacency references unknown code"
    non_root = {ch for ch, _ in PART_OF}
    assert codes - non_root == {"CEL"}, "containment must be a single CEL-rooted tree"
    print(
        f"organelles.csv={n1}  part_of={n2}  adjacent_to={n3}  hpa_crosswalk={n4}  biotype_groups={n5}"
    )
    print("OK: curated organelle inputs written to", OUT)


if __name__ == "__main__":
    main()
