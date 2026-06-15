# US-014: MultiHop-RAG Benchmark Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against the MultiHop-RAG benchmark so that I can measure retrieval and reasoning accuracy on questions specifically designed to stress-test RAG pipelines on multi-document, multi-hop retrieval.

## Scope

Integrate MultiHop-RAG (Tang & Yang, COLM 2024) into the APEX-RAG evaluation harness. The dataset contains 2,556 queries with evidence distributed across 2–4 documents per query, covering four query types: inference, comparison, temporal, and null (no-answer). It is the only benchmark in this suite purpose-built specifically for RAG system evaluation, making it the primary regression target for retrieval quality.

## Implementation Tasks

- Load the MultiHop-RAG dataset from Hugging Face (`yixuantt/MultiHopRAG`) using both the `query` split and the `corpus` split.
- Construct evidence items from the corpus split and supply them to `run_pipeline()` alongside each question.
- Evaluate retrieval using the provided `evidence_list` field: compute MRR, Recall@K, and Precision@K against gold evidence documents.
- Evaluate question-answering accuracy by comparing pipeline answer text against the ground-truth `answer` field using EM and F1.
- Stratify results by `question_type` (inference, comparison, temporal, null) and report separately.
- For null queries, verify that the pipeline surfaces a `has_limitations=True` response rather than fabricating an answer.
- Compute retrieval metrics using document metadata (author, category, publication date) to test metadata-aware retrieval.
- Aggregate results into an APEX-Eval AggregateReport and emit JSON for regression tracking.

## Acceptance Criteria

- The harness loads the query and corpus splits from Hugging Face without manual preprocessing.
- Retrieval MRR, Recall@K, and Precision@K are computed against the gold evidence_list.
- QA accuracy (EM and F1) is computed for all non-null query types.
- Null queries result in responses with `has_limitations=True` in at least 80% of cases.
- Results are stratified and reported separately for each of the four question types.
- Evaluation covers the full 2,556-question set in a documented run time.
- Evaluation failures are counted and reported as a dedicated null-handling metric.
- APEX-Eval report sections are populated with MultiHop-RAG-specific retrieval and reasoning scores.

## Testing Expectations

- Unit tests verify MRR computation against a fixture with known gold and retrieved document lists.
- Unit tests verify null-query handling: pipeline must not assert all_claims_supported for unanswerable questions.
- Integration test runs 20 representative questions (5 per type) end-to-end and asserts non-zero Recall@5.
- Regression fixture records expected scores on a 50-question stratified sample to detect regressions.

## Documentation Updates

- Document how to download MultiHop-RAG from Hugging Face and the expected corpus schema.
- Document run command, expected scores, and how to reproduce reported results.
- Document how null-query detection is interpreted as a pipeline safety signal.

## Dependencies

- US-006 Complex Query Reasoning Path — inference and comparison queries require multi-hop claim reasoning.
- US-008 Retrieval Repair Loop — low-recall retrieval on distributed evidence triggers the repair loop.
- US-011 APEX-Eval Framework — reuses retrieval and claim metrics; adds null-query handling metric.
