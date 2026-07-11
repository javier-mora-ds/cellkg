// CellKG — uniqueness constraints and indexes (Phase 3, Task 3.1)
CREATE CONSTRAINT gene_ensembl IF NOT EXISTS FOR (g:Gene) REQUIRE g.ensembl_id IS UNIQUE;
CREATE CONSTRAINT organelle_code IF NOT EXISTS FOR (o:Organelle) REQUIRE o.code IS UNIQUE;
CREATE CONSTRAINT chromosome_name IF NOT EXISTS FOR (c:Chromosome) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT cc_goid IF NOT EXISTS FOR (cc:CellularComponent) REQUIRE cc.go_id IS UNIQUE;
CREATE CONSTRAINT protein_uniprot IF NOT EXISTS FOR (p:Protein) REQUIRE p.uniprot_id IS UNIQUE;
CREATE CONSTRAINT bp_term IF NOT EXISTS FOR (b:BiologicalProcess) REQUIRE b.term IS UNIQUE;
CREATE CONSTRAINT mf_term IF NOT EXISTS FOR (m:MolecularFunction) REQUIRE m.term IS UNIQUE;
CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol);
CREATE INDEX gene_biotype IF NOT EXISTS FOR (g:Gene) ON (g.biotype);
