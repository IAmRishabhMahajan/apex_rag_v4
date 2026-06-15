# US-021: T²-RAGBench Financial Tabular Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against T²-RAGBench so that I can measure pipeline performance on financial documents combining prose and tables, testing numerical reasoning and retrieval robustness under high-stakes domain conditions.

## Scope

Integrate T²-RAGBench (University of Hamburg) into the APEX-RAG evaluation harness. The benchmark contains 23,088 QA pairs from 7,318 real financial reports across three subsets: FinQA (8,281 pairs from 2,789 docs), ConvFinQA (3,458 pairs from 1,806 docs), and TAT-DQA (11,349 pairs from 2,723 docs). It evaluates both retrieval quality (MRR@3) and numerical answer accuracy (NM metric), making it the primary benchmark for testing the pipeline's high-risk financial query handling.

## Implementation Tasks

- Download T²-RAGBench data from the official benchmark website or associated repository and parse all three subsets (FinQA, ConvFinQA, TAT-DQA).
- Supply each document's text and table content as raw evidence items to `run_pipeline()`, enabling numerical reasoning queries.
- Activate high-risk mode (`high_risk=True`) for all financial queries to enforce stricter confidence thresholds and disclaimers.
- Compute the NM (Numerical Match) metric: evaluate whether pipeline-generated answers match gold numeric values within an acceptable tolerance.
- Compute MRR@3: measure how often the correct supporting document appears in the top-3 retrieved evidence items.
- Evaluate ConvFinQA conversational turns: pass prior-turn answers as context and measure consistency across conversation history.
- Compute a weighted average total score combining NM and MRR@3 across the three subsets per the benchmark protocol.
- Verify that high-risk disclaimers appear in financial answers and that `is_high_risk=True` on the verified answer.
- Aggregate results into an APEX-Eval AggregateReport with per-subset financial scores.

## Acceptance Criteria

- All three subsets (FinQA, ConvFinQA, TAT-DQA) load and parse without schema errors.
- NM score is computed with numeric normalization (rounding, unit handling) for each subset.
- MRR@3 is computed against gold supporting documents for each subset.
- Weighted average total score matches the benchmark's official scoring protocol.
- High-risk mode is activated for all financial queries and verified via `risk_assessment.category == FINANCIAL`.
- Financial answers include a non-empty disclaimer in `critique.disclaimer`.
- ConvFinQA conversational turns are evaluated in order with prior context supplied.
- APEX-Eval report includes per-subset scores (FinQA, ConvFinQA, TAT-DQA) and the weighted total.

## Testing Expectations

- Unit tests verify NM computation: numeric normalization, tolerance matching, and unit stripping.
- Unit tests verify MRR@3 computation against a known gold-document fixture.
- Unit tests verify high-risk mode activates the FINANCIAL risk category for financial query phrases.
- Integration test runs 30 questions (10 per subset) end-to-end with high-risk mode enabled.
- Regression fixture captures expected NM and MRR@3 on a 100-question sample per subset.

## Documentation Updates

- Document how to obtain the T²-RAGBench dataset and the format of its three subsets.
- Document the NM metric, numeric normalization rules, and tolerance thresholds.
- Document how high-risk mode affects scoring thresholds and answer disclaimers.

## Dependencies

- US-010 Risk Assessment, Critique, and Verification — financial queries activate the FINANCIAL risk category.
- US-006 Complex Query Reasoning Path — numerical multi-step reasoning in FinQA requires the complex path.
- US-011 APEX-Eval Framework — extends with NM and MRR@3 metrics and weighted multi-subset scoring.
