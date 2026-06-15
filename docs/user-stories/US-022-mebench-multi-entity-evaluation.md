# US-022: MEBench Multi-Entity Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against MEBench so that I can measure how accurately the pipeline resolves entity-dense questions requiring integration of entity-centric information from heterogeneous document sources, using the Entity-Attributed F1 metric.

## Scope

Integrate MEBench (EMNLP 2025) into the APEX-RAG evaluation harness. MEBench contains 4,780 questions in three primary categories with eight distinct subtypes, designed specifically to expose how LLMs and RAG pipelines fail when questions involve multiple entities and fragmented information across diverse documents. Even GPT-4 achieves only 59% accuracy on this benchmark, making it a high-signal stress test for completeness and factual precision.

## Implementation Tasks

- Download the MEBench dataset from its arXiv supplementary materials or associated repository and parse all three primary categories and eight subtypes.
- Supply all documents associated with each question as raw evidence items to `run_pipeline()`.
- Implement the Entity-Attributed F1 (EA-F1) metric: for each entity in the answer, verify both its factual correctness and whether it is attributed to the correct source document.
- Measure entity-level recall: what fraction of gold entities from the reference answer appear in the pipeline's output.
- Measure entity-level precision: what fraction of pipeline-output entities are present in the gold answer.
- Evaluate attribution validity: for each correctly identified entity, verify the citation source matches the gold-document mapping.
- Stratify results by question category and subtype to identify which entity reasoning patterns the pipeline handles weakest.
- Aggregate results into an APEX-Eval AggregateReport with EA-F1 as the primary metric.

## Acceptance Criteria

- All 4,780 questions load and run without schema errors.
- EA-F1 is computed correctly: entity-level precision and recall combined with attribution verification.
- Entity-level recall and precision are reported separately from EA-F1.
- Attribution validity score (correct source for correct entity) is reported as a dedicated metric.
- Results are stratified by all three primary categories and all eight subtypes.
- APEX-Eval report includes an MEBench section with EA-F1 as the primary headline score.
- The pipeline achieves a documented EA-F1 score for comparison against the 59% GPT-4 baseline.
- Evaluation failures are counted per subtype to identify systematic weaknesses.

## Testing Expectations

- Unit tests verify EA-F1 computation against a fixture with known entity lists and source attributions.
- Unit tests verify entity recall and precision are computed independently before being combined.
- Unit tests verify attribution validity: correct entity with wrong source must not count as a valid attribution.
- Integration test runs 30 questions (across all three categories) end-to-end.
- Regression fixture captures expected EA-F1 on a 100-question stratified sample.

## Documentation Updates

- Document how to obtain the MEBench dataset and the entity-attribution annotation format.
- Document the EA-F1 metric definition, including how attribution is validated against citation source IDs.
- Document how per-subtype results reveal entity reasoning bottlenecks.

## Dependencies

- US-004 Evidence Fusion — cross-document entity deduplication and provenance preservation is central to EA-F1.
- US-009 Grounded Reasoning and Generation — citation links are the mechanism for entity attribution validation.
- US-011 APEX-Eval Framework — extends with EA-F1, entity recall/precision, and attribution validity metrics.
