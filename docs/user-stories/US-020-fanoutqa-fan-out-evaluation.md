# US-020: FanOutQA Fan-Out Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against FanOutQA so that I can measure whether the pipeline can synthesize information about large numbers of entities across multiple documents, testing the complex reasoning path under fan-out conditions.

## Scope

Integrate FanOutQA (Google DeepMind, ACL 2024) into the APEX-RAG evaluation harness. FanOutQA contains complex multi-hop questions that "fan out" to a large number of entities — each question requires finding information about many sub-entities and aggregating it into a single answer. The dataset uses English Wikipedia as the knowledge base and provides human-annotated decompositions showing how each complex question breaks into simpler sub-questions, enabling structured evaluation of the reasoning path.

## Implementation Tasks

- Load the FanOutQA dataset from the Google DeepMind GitHub repository (`google-deepmind/fanoutqa`) and parse the question, decomposition, and answer fields.
- Supply Wikipedia context paragraphs for each entity referenced in the question as raw evidence items to `run_pipeline()`.
- Activate the complex reasoning path by ensuring fan-out queries (multi-entity, multi-hop) route through `run_complex_reasoning()`.
- Evaluate answer correctness using exact match and F1 against gold answers.
- Evaluate reasoning decomposition quality: compare the pipeline's claim graph edges against the gold human-annotated decomposition steps.
- Measure claim completeness: what fraction of gold sub-question answers appear as supported claims in the pipeline's output.
- Test the three benchmark settings described in the paper (closed-book, oracle retrieval, and open retrieval).
- Aggregate results into an APEX-Eval AggregateReport with a fan-out reasoning section.

## Acceptance Criteria

- FanOutQA dataset loads and the decomposition annotations are parsed correctly.
- Complex reasoning path is activated for all fan-out queries (verified via `reasoning.used_complex_path`).
- Answer EM and F1 are computed against gold answers.
- Claim completeness (sub-question coverage) is computed against gold decomposition annotations.
- Results are computed for all three benchmark settings (closed-book, oracle, open).
- Inter-document dependency performance is reported as a dedicated score.
- APEX-Eval report includes a FanOutQA section with decomposition quality scores.
- Evaluation completes for the full validation set in a documented run time.

## Testing Expectations

- Unit tests verify claim graph edge matching against a known decomposition fixture.
- Unit tests verify that fan-out queries (multi-entity) activate `used_complex_path=True`.
- Integration test runs 20 representative fan-out questions end-to-end in oracle retrieval mode.
- Regression fixture captures expected answer F1 and claim completeness on a 50-question sample.

## Documentation Updates

- Document how to load FanOutQA and parse human-annotated decomposition annotations.
- Document how the claim graph maps to gold decomposition steps.
- Document the three benchmark settings and how to switch between them.

## Dependencies

- US-006 Complex Query Reasoning Path — fan-out questions are the primary stress test for claim decomposition and graph reasoning.
- US-004 Evidence Fusion — multi-entity evidence requires deduplication and provenance preservation across many sources.
- US-011 APEX-Eval Framework — extends with decomposition quality and inter-document dependency metrics.
