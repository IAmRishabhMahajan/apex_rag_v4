# US-023: MuDABench Multi-Document Analytical Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against MuDABench so that I can measure the pipeline's ability to perform quantitative analytical reasoning across large document collections, identifying whether the complex reasoning path can handle calculation-heavy, multi-document synthesis tasks.

## Scope

Integrate MuDABench into the APEX-RAG evaluation harness. MuDABench contains 332 analytical QA instances built from 80,000+ pages using distant supervision on financial databases. Each question requires extracting intermediate facts from numerous documents and synthesizing them through quantitative analysis — far beyond standard retrieval or span extraction. Published experiments show standard RAG significantly underperforms, with the best approach being a multi-agent workflow combining planning, extraction, and code generation.

## Implementation Tasks

- Download the MuDABench dataset from its arXiv supplementary materials or associated repository and parse all 332 analytical instances.
- For each question, supply all associated document pages as raw evidence items to `run_pipeline()`.
- Evaluate final answer accuracy: compare pipeline numeric or categorical answers against gold answers using normalized EM and F1.
- Evaluate intermediate-fact coverage: measure how many gold intermediate facts appear in the pipeline's evidence bundle or claim graph as an auxiliary diagnostic score.
- Activate the complex reasoning path for all analytical questions and verify `reasoning.used_complex_path=True`.
- Identify and flag questions where the pipeline defaults to a fallback answer (GroundingError caught) as systematic failures.
- Measure the confidence gap: compare `scored.scores.confidence` on analytical questions vs. the broader benchmark average to quantify how well the pipeline calibrates uncertainty on hard questions.
- Aggregate results into an APEX-Eval AggregateReport with answer accuracy and intermediate-fact coverage as the dual headline metrics.

## Acceptance Criteria

- All 332 analytical instances load and run without schema errors.
- Final answer accuracy (normalized EM and F1) is computed against gold answers.
- Intermediate-fact coverage is computed by matching gold facts against pipeline evidence items and claim graph nodes.
- Complex reasoning path is activated for all analytical questions (verified via `reasoning.used_complex_path`).
- Fallback answer rate (GroundingError catch count) is reported as a systematic failure metric.
- Confidence gap between analytical questions and baseline is computed and reported.
- APEX-Eval report includes a MuDABench section with both accuracy and intermediate-fact coverage scores.
- Bottleneck analysis: questions are stratified by whether failure occurs at retrieval, extraction, or reasoning.

## Testing Expectations

- Unit tests verify intermediate-fact coverage computation against a fixture with known gold facts and evidence items.
- Unit tests verify fallback answer detection (GroundingError catch and has_limitations=True).
- Unit tests verify confidence gap is computed correctly as the difference between two AggregateReport scores.
- Integration test runs 15 representative analytical questions end-to-end with complex reasoning enabled.
- Regression fixture captures expected answer accuracy and intermediate-fact coverage on a 50-question sample.

## Documentation Updates

- Document how to obtain the MuDABench dataset and the intermediate-fact annotation format.
- Document the dual-metric evaluation protocol (answer accuracy + intermediate-fact coverage).
- Document the confidence gap metric and how it reveals calibration weaknesses.
- Document recommended future enhancement: multi-agent planning + code generation workflow for quantitative tasks.

## Dependencies

- US-006 Complex Query Reasoning Path — analytical synthesis questions are the hardest test of claim decomposition and graph reasoning.
- US-007 Evidence Scoring and Gap Detection — 80k-page documents stress evidence coverage and gap detection at scale.
- US-008 Retrieval Repair Loop — high failure rate on analytical questions makes the repair loop central to performance.
- US-011 APEX-Eval Framework — extends with intermediate-fact coverage and confidence gap as diagnostic metrics.
