// ================================================================
// Neuro-Neo4j — Neo4j Schema Constraints & Indexes
// Author: basakanimesh14
// Run this script ONCE after starting Neo4j to establish
// referential integrity and query performance.
// ================================================================
// Usage:
//   cypher-shell -u neo4j -p neuro_password -f constraints.cypher
// ================================================================

// ----------------------------------------------------------------
// 1. Subject node — uniqueness on PatientID
//    Each patient can only appear once in the graph.
// ----------------------------------------------------------------
CREATE CONSTRAINT subject_patient_id_unique IF NOT EXISTS
FOR (s:Subject)
REQUIRE s.patientId IS UNIQUE;

// ----------------------------------------------------------------
// 2. BrainRegion node — uniqueness on atlasId
//    Each AAL-116 region is a singleton node shared across subjects.
// ----------------------------------------------------------------
CREATE CONSTRAINT brain_region_atlas_id_unique IF NOT EXISTS
FOR (r:BrainRegion)
REQUIRE r.atlasId IS UNIQUE;

// ----------------------------------------------------------------
// 3. Index on Subject.diagnosis
//    Enables fast filtering by Healthy / Diseased label.
// ----------------------------------------------------------------
CREATE INDEX subject_diagnosis_idx IF NOT EXISTS
FOR (s:Subject)
ON (s.diagnosis);

// ----------------------------------------------------------------
// 4. Index on Subject.scannerSite
//    Supports domain-shift queries across scanner sites.
// ----------------------------------------------------------------
CREATE INDEX subject_scanner_site_idx IF NOT EXISTS
FOR (s:Subject)
ON (s.scannerSite);

// ----------------------------------------------------------------
// 5. Index on BrainRegion.anatomicalLabel
//    Supports lookup by AAL region name (e.g., "Hippocampus_L").
// ----------------------------------------------------------------
CREATE INDEX brain_region_label_idx IF NOT EXISTS
FOR (r:BrainRegion)
ON (r.anatomicalLabel);

// ----------------------------------------------------------------
// 6. Index on Subject.age
//    Supports age-stratified cohort queries.
// ----------------------------------------------------------------
CREATE INDEX subject_age_idx IF NOT EXISTS
FOR (s:Subject)
ON (s.age);

// ----------------------------------------------------------------
// Verification query — list all constraints and indexes
// ----------------------------------------------------------------
SHOW CONSTRAINTS;
SHOW INDEXES;
