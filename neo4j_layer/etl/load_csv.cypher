// ================================================================
// Neuro-Neo4j — Bulk Data Ingestion (LOAD CSV)
// Author: basakanimesh14
//
// Prerequisites:
//   1. Run constraints.cypher first
//   2. Copy data/processed/*.csv to Neo4j import directory:
//      /var/lib/neo4j/import/data/
//   3. Execute via cypher-shell or Neo4j Browser
//
// Order of execution:
//   STEP 1 — Create Subject nodes
//   STEP 2 — Create BrainRegion nodes (AAL-116)
//   STEP 3 — Create CONNECTED_TO relationships (edges)
//   STEP 4 — Create PATIENT_OF relationships
// ================================================================

// ----------------------------------------------------------------
// STEP 1: Load Subject nodes from subjects.csv
// subjects.csv columns: PatientID, Diagnosis, Label, Age, ScannerSite
// ----------------------------------------------------------------
:auto
LOAD CSV WITH HEADERS FROM 'file:///data/subjects.csv' AS row
CALL {
    WITH row
    MERGE (s:Subject {patientId: row.PatientID})
    SET
        s.diagnosis   = row.Diagnosis,
        s.label       = toInteger(row.Label),
        s.age         = toInteger(row.Age),
        s.scannerSite = row.ScannerSite,
        s.createdAt   = datetime()
} IN TRANSACTIONS OF 100 ROWS;

// Verify
MATCH (s:Subject) RETURN count(s) AS subject_count;

// ----------------------------------------------------------------
// STEP 2: Create BrainRegion nodes (AAL-116)
// One shared set of region nodes — not per-subject duplicates.
// Atlas regions are identified by atlasId (0-based index).
//
// NOTE: In production, load from aal116_regions.csv.
//       Here we use inline UNWIND for 116 regions.
// ----------------------------------------------------------------

// Create a helper CSV-loadable representation
// Alternatively, load from file: 'file:///data/brain_regions.csv'
// brain_regions.csv columns: atlasId, anatomicalLabel, hemisphere

LOAD CSV WITH HEADERS FROM 'file:///data/brain_regions.csv' AS row
CALL {
    WITH row
    MERGE (r:BrainRegion {atlasId: toInteger(row.atlasId)})
    SET
        r.anatomicalLabel = row.anatomicalLabel,
        r.hemisphere      = row.hemisphere,
        r.atlasName       = 'AAL-116'
} IN TRANSACTIONS OF 50 ROWS;

// Verify
MATCH (r:BrainRegion) RETURN count(r) AS region_count;

// ----------------------------------------------------------------
// STEP 3: Load CONNECTED_TO relationships from edge_list.csv
// edge_list.csv columns: subject_id, source, target, weight, diagnosis
//
// Each row is an undirected edge in a subject-specific connectome.
// Weight = Pearson correlation coefficient (normalised).
// ----------------------------------------------------------------
:auto
LOAD CSV WITH HEADERS FROM 'file:///data/edge_list.csv' AS row
CALL {
    WITH row
    MATCH (s:Subject        {patientId:       row.subject_id})
    MATCH (src:BrainRegion  {anatomicalLabel: row.source})
    MATCH (tgt:BrainRegion  {anatomicalLabel: row.target})
    MERGE (src)-[e:CONNECTED_TO {subjectId: row.subject_id}]->(tgt)
    SET
        e.weight      = toFloat(row.weight),
        e.diagnosis   = row.diagnosis,
        e.subjectId   = row.subject_id
} IN TRANSACTIONS OF 500 ROWS;

// Verify
MATCH ()-[e:CONNECTED_TO]->() RETURN count(e) AS edge_count;

// ----------------------------------------------------------------
// STEP 4: Create PATIENT_OF relationships
// Links each Subject to ALL 116 BrainRegion nodes,
// enabling subject-specific subgraph queries.
// ----------------------------------------------------------------
:auto
MATCH (s:Subject)
MATCH (r:BrainRegion)
CALL {
    WITH s, r
    MERGE (s)-[:PATIENT_OF]->(r)
} IN TRANSACTIONS OF 1000 ROWS;

// Verify
MATCH (s:Subject)-[:PATIENT_OF]->(r:BrainRegion)
RETURN s.patientId, count(r) AS n_regions
LIMIT 5;

// ----------------------------------------------------------------
// Summary Query: Disease pathway exploration
// Returns top-10 most strongly connected inter-hemispheric edges
// in Diseased subjects.
// ----------------------------------------------------------------
MATCH (src:BrainRegion)-[e:CONNECTED_TO]->(tgt:BrainRegion)
WHERE e.diagnosis = 'Diseased'
  AND src.hemisphere <> tgt.hemisphere
RETURN
    src.anatomicalLabel AS region_a,
    tgt.anatomicalLabel AS region_b,
    avg(e.weight) AS mean_weight,
    count(e)      AS n_subjects
ORDER BY mean_weight DESC
LIMIT 10;
