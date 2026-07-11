// CellKG — competency questions (Phase 5). Blocks are delimited by "// @@ | Qn | title".

// @@ | Q1 | Structural: RNA-biotype composition of the genes on chromosome 17
MATCH (:Chromosome {name: '17'})-[:CONTAINS]->(g:Gene)
RETURN g.biotype AS biotype, count(*) AS genes
ORDER BY genes DESC LIMIT 10;

// @@ | Q2 | Recursive hierarchy: cellular components under the nucleus (GO:0005634)
MATCH (x:CellularComponent)-[:SUBCLASS_OF|PART_OF*1..]->(:CellularComponent {go_id: 'GO:0005634'})
RETURN DISTINCT x.go_id AS go_id, x.name AS component
ORDER BY component;

// @@ | Q3 | RNA-aware: biotype composition of genes localized to the nucleolus (NCL)
MATCH (g:Gene)-[:LOCATED_IN]->(:Organelle {code: 'NCL'})
RETURN g.biotype AS biotype, count(*) AS genes
ORDER BY genes DESC;

// @@ | Q4 | Two-hop N-to-N: genes sharing >= 2 biological processes with TP53
MATCH (t:Gene {symbol: 'TP53'})-[:PARTICIPATES_IN]->(p:BiologicalProcess)<-[:PARTICIPATES_IN]-(g:Gene)
WHERE g <> t
WITH g, count(DISTINCT p) AS shared
WHERE shared >= 2
RETURN g.symbol AS gene, shared
ORDER BY shared DESC LIMIT 15;

// @@ | Q5 | Centrality: organelles ranked by number of localized genes (degree)
MATCH (o:Organelle)<-[:LOCATED_IN]-(g:Gene)
RETURN o.code AS code, o.name AS organelle, count(g) AS localized_genes
ORDER BY localized_genes DESC LIMIT 10;

// @@ | Q6 | Similarity: Jaccard over shared organelles between TP53 and other genes
MATCH (t:Gene {symbol: 'TP53'})-[:LOCATED_IN]->(o:Organelle)
WITH t, collect(DISTINCT o) AS t_orgs
MATCH (g:Gene)-[:LOCATED_IN]->(o2:Organelle)
WHERE g <> t
WITH g, t_orgs, collect(DISTINCT o2) AS g_orgs
WITH g,
     size([x IN g_orgs WHERE x IN t_orgs]) AS inter,
     size(t_orgs) + size([x IN g_orgs WHERE NOT x IN t_orgs]) AS uni
WHERE inter > 0
RETURN g.symbol AS gene, round(toFloat(inter) / uni, 3) AS jaccard
ORDER BY jaccard DESC, gene LIMIT 15;

// @@ | Q7 | Path: shortest path connecting TP53 and EGFR across interaction/function layers
MATCH p = shortestPath(
  (:Gene {symbol: 'TP53'})-[:INTERACTS_WITH|PARTICIPATES_IN|LOCATED_IN*..6]-(:Gene {symbol: 'EGFR'}))
RETURN length(p) AS hops,
       [n IN nodes(p) | coalesce(n.symbol, n.name, n.term)] AS path;

// @@ | Q8 | Context-aware: high-reliability (Enhanced/Supported) localizations in mitochondria
MATCH (g:Gene)-[r:LOCATED_IN]->(:Organelle {code: 'MTC'})
WHERE r.reliability IN ['Enhanced', 'Supported']
RETURN g.symbol AS gene, r.reliability AS reliability, r.hpa_term AS hpa_term
ORDER BY gene LIMIT 15;

// @@ | Q9 | Interaction-aware: highest-confidence STRING partners of TP53 and shared organelles
MATCH (t:Gene {symbol: 'TP53'})-[r:INTERACTS_WITH]-(g:Gene)
WHERE r.score >= 900
OPTIONAL MATCH (t)-[:LOCATED_IN]->(o:Organelle)<-[:LOCATED_IN]-(g)
RETURN g.symbol AS partner, r.score AS score, collect(DISTINCT o.code) AS shared_organelles
ORDER BY score DESC LIMIT 15;

// @@ | Q10 | Cross-layer: processes enriched among genes co-localized with TP53 in the nucleus
MATCH (t:Gene {symbol: 'TP53'})-[:LOCATED_IN]->(:Organelle {code: 'NUC'})<-[:LOCATED_IN]-(g:Gene)
WHERE g <> t
MATCH (g)-[:PARTICIPATES_IN]->(bp:BiologicalProcess)
RETURN bp.term AS process, count(DISTINCT g) AS genes
ORDER BY genes DESC LIMIT 15;
