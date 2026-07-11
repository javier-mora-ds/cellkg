// =====================================================================
// CellKG — presentation queries for the Neo4j Browser (http://localhost:7474)
// =====================================================================
// Run each block on its own: paste it into the Browser command bar and press
// Ctrl+Enter. Blocks that RETURN a path `p` or nodes draw a graph (use the
// "Graph" tab); blocks that RETURN scalars show a table (switch to "Table",
// and the bar-chart icon for numeric columns).
//
// Display tips for clean slides:
//   * After a graph renders, click each label chip above the result and set its
//     caption: Gene -> symbol, Organelle/CellularComponent -> name,
//     BiologicalProcess/MolecularFunction -> term.
//   * Use the camera/download icon on the result frame to export PNG/SVG.
//   * All queries are pre-sized to render legibly (3-52 nodes).
// =====================================================================


// ===== SLIDE 1 — The data model (property-graph meta-graph) ==========
// Shows every node label and relationship type. Opening "this is the schema" slide.
CALL db.schema.visualization();


// ===== SLIDE 2 — The cell-compartment ontology (22 organelles) =======
// Curated containment tree (PART_OF) + adjacency (ADJACENT_TO). Demonstrates a
// recursive hierarchy AND cycles in a single, small picture. (22 nodes)
MATCH p = (:Organelle)-[:PART_OF|ADJACENT_TO]->(:Organelle)
RETURN p;


// ===== SLIDE 3 — Ontology grounding in the Gene Ontology =============
// The Cellular Component is_a / part_of DAG (depth 8) that grounds the organelles
// in a public ontology. Demonstrates the "ontology-grounded" claim. (52 nodes)
MATCH p = (:CellularComponent)-[:SUBCLASS_OF|PART_OF]->(:CellularComponent)
RETURN p;


// ===== SLIDE 4 — One gene across every layer (integrative view) ======
// TP53 connected to its compartment (nucleus), biological processes, molecular
// functions, and its highest-confidence interaction partners. The single best
// picture of the graph's many-to-many, multi-layer integration. (~38 nodes)
MATCH p = (:Gene {symbol:'TP53'})-[:LOCATED_IN|PARTICIPATES_IN|HAS_FUNCTION]->()
RETURN p
UNION
MATCH p = (:Gene {symbol:'TP53'})-[r:INTERACTS_WITH]-(:Gene)
WHERE r.score >= 999
RETURN p;


// ===== SLIDE 5 — A functional module: p53 / DNA-damage network =======
// A dense, cyclic interaction cluster (STRING). Demonstrates edge density and the
// cycles that make the network "interesting". (14 nodes, ~60 edges)
WITH ['TP53','MDM2','MDM4','CDKN1A','ATM','ATR','CHEK2','BRCA1','BRCA2',
      'RAD51','CREBBP','EP300','RPA1','SIRT1'] AS module
MATCH p = (a:Gene)-[r:INTERACTS_WITH]-(b:Gene)
WHERE a.symbol IN module AND b.symbol IN module AND r.score >= 700
RETURN p;


// ===== SLIDE 6 — Two genes bridged by shared biology (N-to-N) ========
// BRCA1 and BRCA2 linked through the biological processes they share. Clean
// bipartite many-to-many bridge. (6 nodes)
MATCH p = (:Gene {symbol:'BRCA1'})-[:PARTICIPATES_IN]->(:BiologicalProcess)
          <-[:PARTICIPATES_IN]-(:Gene {symbol:'BRCA2'})
RETURN p;


// ===== SLIDE 7 — Cross-layer navigability (a path) ===================
// Shortest path from TP53 to the mitochondrion, hopping across interaction and
// localization layers: TP53 -> SCO2 -> Mitochondria (SCO2 is a p53-regulated
// mitochondrial assembly factor). Demonstrates that the graph is navigable.
MATCH p = shortestPath(
  (:Gene {symbol:'TP53'})-[:INTERACTS_WITH|LOCATED_IN|PARTICIPATES_IN*..6]-(:Organelle {code:'MTC'}))
RETURN p;


// ===== SLIDE 8 — The localization bridge (one compartment, many genes) =
// Genes that reside in the nucleolus. Shows the Gene -> Organelle bridge that
// makes the graph genuinely N-to-N. (nucleolus hub + 50 genes)
MATCH p = (g:Gene)-[:LOCATED_IN]->(:Organelle {code:'NCL'})
RETURN p LIMIT 50;


// ===== SLIDE 9 — Context-aware querying (evidence on the edges) ======
// Only Enhanced-reliability mitochondrial localizations. Demonstrates the
// "property graph with properties": filtering on edge evidence. (40 genes)
MATCH p = (g:Gene)-[r:LOCATED_IN]->(:Organelle {code:'MTC'})
WHERE r.reliability = 'Enhanced'
RETURN p LIMIT 40;


// ===== SLIDE 10 — Analytics: the organelle census (bar chart) ========
// Genes localized per compartment. Switch to Table, then the bar-chart view.
MATCH (o:Organelle)<-[:LOCATED_IN]-(g:Gene)
RETURN o.name AS organelle, count(g) AS genes
ORDER BY genes DESC;


// ===== SLIDE 11 — Scale (headline numbers) ===========================
// Node and relationship totals; and the label / type breakdowns for a stats slide.
MATCH (n) WITH count(n) AS total_nodes
MATCH ()-[r]->() RETURN total_nodes, count(r) AS total_relationships;

MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS n ORDER BY n DESC;


// ===== SLIDE 12 — RNA-awareness (biotype composition) ================
// The RNA-biotype spectrum of the modeled genes (Table / bar chart).
MATCH (g:Gene)
RETURN g.biotype AS biotype, count(*) AS genes
ORDER BY genes DESC LIMIT 10;


// ===== SLIDE 13 — Machine learning context (known vs predicted) ======
// Narrate with reports/ml_results.txt: RotatE ranks novel gene->organelle
// localizations. This query shows the KNOWN localizations of the panel genes so
// the audience can contrast them with the model's top-novel predictions.
MATCH (g:Gene)-[:LOCATED_IN]->(o:Organelle)
WHERE g.symbol IN ['TP53','BRCA1','EGFR','GAPDH','CANX','LMNA']
RETURN g.symbol AS gene, collect(o.code) AS known_organelles
ORDER BY gene;
