# US-013: HotpotQA Multi-Hop Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against the HotpotQA benchmark so that I can measure how well the pipeline handles multi-hop questions that require synthesizing evidence from multiple Wikipedia articles.

## Scope

Integrate HotpotQA into the APEX-RAG evaluation harness. HotpotQA contains ~113,000 questions across two settings — distractor (10 candidate paragraphs) and fullwiki (open Wikipedia retrieval) — and provides sentence-level supporting-fact annotations alongside answers, enabling evaluation of both answer correctness and retrieval explainability.

## Implementation Tasks

- Load the HotpotQA dataset from Hugging Face (`hotpotqa/hotpot_qa`) in both `distractor` and `fullwiki` configurations using the `datasets` library.
- Feed each question and its candidate context paragraphs into `run_pipeline()` as raw evidence items.
- Collect pipeline outputs and compare against ground-truth answers using Exact Match (EM) and token-level F1.
- Evaluate supporting-fact retrieval: compare evidence items selected by the pipeline against the gold `supporting_facts` annotations (EM and F1).
- Compute the Joint metric: the harmonic combination of answer F1 and supporting-fact F1 for every question.
- Separate results by question type (`bridge` vs `comparison`) and difficulty level (`easy`, `medium`, `hard`).
- Aggregate per-question results into an APEX-Eval AggregateReport and format it with `format_report()`.
- Persist raw results as JSON for regression tracking and offline analysis.

## Acceptance Criteria

- The harness loads both `distractor` and `fullwiki` configurations without manual data preparation.
- Answer EM and F1 scores are computed correctly and match reference implementations (tolerance ≤ 0.5%).
- Supporting-fact EM and F1 scores are computed correctly using sentence-level annotations.
- Joint scores are derived from per-question answer and supporting-fact F1 values.
- Results are stratified by question type and difficulty level.
- Evaluation completes for the full validation set (7,405 examples) in a documented run time.
- APEX-Eval report includes HotpotQA-specific sections alongside standard retrieval and claim metrics.
- Evaluation failures (grounding errors, empty bundles) are counted and reported, not silently skipped.

## Testing Expectations

- Unit tests verify EM and F1 calculation against known HotpotQA reference answers.
- Unit tests verify supporting-fact F1 using a fixture with known gold and predicted sentence lists.
- Integration test runs 10 representative questions end-to-end and asserts non-zero Joint F1.
- Regression fixture captures expected scores on a 100-question subset to detect regressions.

## Documentation Updates

- Document how to download and cache HotpotQA locally.
- Document the run command and expected output format.
- Document score interpretation: what answer EM vs Joint F1 reveals about the pipeline.

## Dependencies

- US-006 Complex Query Reasoning Path — bridge and comparison questions require multi-hop reasoning.
- US-007 Evidence Scoring and Gap Detection — supporting-fact coverage maps to evidence gap detection.
- US-011 APEX-Eval Framework — reuses retrieval, evidence, claim, and final metrics infrastructure.
