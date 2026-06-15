# US-019: TriviaQA Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against TriviaQA so that I can measure open-domain retrieval and reading comprehension on a large-scale dataset with multiple supporting documents per question, testing the pipeline's ability to find and use the most relevant evidence from a noisy candidate set.

## Scope

Integrate TriviaQA into the APEX-RAG evaluation harness. TriviaQA contains over 650,000 question-answer-evidence triples, built from 95,000+ trivia questions authored by enthusiasts, each paired with an average of six supporting documents from the web and Wikipedia. Its two variants — Reading Comprehension (RC) and Open-Domain (unfiltered) — test different retrieval regimes: span extraction from provided context vs. retrieval from a large corpus.

## Implementation Tasks

- Load TriviaQA from Hugging Face (`trivia_qa`) in both `rc` (reading comprehension) and `unfiltered` (open-domain) configurations.
- For the RC configuration, supply all provided evidence documents as raw evidence items to `run_pipeline()`.
- For the unfiltered configuration, treat the question alone as the query and supply only a sampled subset of evidence documents to simulate corpus-level retrieval.
- Evaluate answer accuracy using normalized Exact Match (EM) and F1 against the provided answer aliases (TriviaQA provides multiple acceptable answer forms per question).
- Implement answer normalization: lowercase, strip articles, punctuation, and extra whitespace before comparison.
- Measure retrieval precision: what fraction of the pipeline's evidence bundle overlaps with TriviaQA's gold evidence documents.
- Stratify results by evidence source (Wikipedia vs. web documents).
- Aggregate results into an APEX-Eval AggregateReport and emit JSON for regression tracking.

## Acceptance Criteria

- Both RC and unfiltered configurations load and run without schema errors.
- Answer EM and F1 are computed using normalized comparison against all answer aliases.
- Retrieval precision (evidence overlap) is computed and reported for the RC configuration.
- Results are stratified by Wikipedia vs. web evidence source type.
- A 10,000-question subset can be run as a fast evaluation in a documented run time.
- APEX-Eval report includes a TriviaQA section with RC and unfiltered variant breakdowns.
- Answer normalization is tested and verified against reference normalization logic.
- Evaluation failures (empty bundles, grounding errors) are counted and reported.

## Testing Expectations

- Unit tests verify answer normalization (lowercase, article stripping, punctuation removal) against a fixture.
- Unit tests verify alias-based EM/F1: any matching alias should count as correct.
- Integration test runs 50 RC questions and 50 unfiltered questions end-to-end.
- Regression fixture captures expected EM on a 500-question stratified sample.

## Documentation Updates

- Document how to download TriviaQA in both RC and unfiltered configurations.
- Document the answer normalization protocol and alias matching strategy.
- Document how RC vs. unfiltered results reveal open-domain retrieval vs. comprehension gaps.

## Dependencies

- US-002 Adaptive Retrieval Planning — unfiltered queries with no context require multi-strategy retrieval planning.
- US-007 Evidence Scoring and Gap Detection — noisy multi-document sets (6 docs/question) stress gap detection.
- US-011 APEX-Eval Framework — reuses retrieval and claim metrics; adds alias-based answer normalization.
