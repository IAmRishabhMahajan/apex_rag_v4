# US-015: RAGBench Large-Scale Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against RAGBench so that I can measure end-to-end pipeline performance across 100,000 examples spanning five industry domains using the TRACe explainable evaluation framework.

## Scope

Integrate RAGBench (Galileo AI, arXiv:2407.11005) into the APEX-RAG evaluation harness. RAGBench is the largest available RAG evaluation dataset at ~100,000 examples drawn from five industry domains: medical (PubMedQA), legal (CUAD), financial (FinQA), general knowledge (HotpotQA subset), and COVID-19 (CovidQA). The TRACe framework adds explainable adherence, relevance, and sentence-level support labels, enabling fine-grained diagnostics beyond simple accuracy.

## Implementation Tasks

- Load RAGBench from Hugging Face (`galileo-ai/ragbench`) and iterate over all 12 domain subsets.
- For each example, supply the provided context documents as raw evidence items to `run_pipeline()`.
- Compute TRACe adherence score: proportion of pipeline answer sentences with at least one supporting context sentence.
- Compute TRACe relevance score: whether retrieved evidence is topically relevant to the question.
- Compute sentence-level support rate by comparing `sentence_support_information` labels against pipeline claim statuses.
- Aggregate results by domain subset and expose domain-specific breakdowns in the APEX-Eval report.
- Compare APEX-RAG's adherence and claim-support scores against the published RoBERTa baseline and TruLens/RAGAS reference scores provided in the dataset.
- Persist per-example results as JSON; persist aggregate summary as a CSV for cross-benchmark comparison.

## Acceptance Criteria

- All 12 domain subsets load and run without schema errors.
- TRACe adherence and relevance scores are computed per example and averaged per domain.
- Sentence-level support rate aligns with APEX-Eval's `unsupported_claim_count` metric within ±5%.
- Domain-specific results surface in the APEX-Eval AggregateReport.
- A subset of 1,000 examples can be run as a fast smoke test in under 5 minutes.
- The harness produces a comparison table showing APEX-RAG scores against published baselines.
- Evaluation failures (validation blocks, grounding errors) are counted per domain and reported.
- All results are reproducible from a single command with a documented random seed.

## Testing Expectations

- Unit tests verify TRACe adherence computation against a fixture with known sentence-support labels.
- Unit tests verify domain-subset loading and schema parsing for at least 3 representative subsets.
- Integration test runs 50 examples from the medical and legal subsets end-to-end.
- Regression fixture captures expected adherence scores on a 200-example stratified sample.

## Documentation Updates

- Document how to download RAGBench and select individual domain subsets.
- Document the TRACe metrics and how they map to APEX-Eval concepts.
- Document the comparison table format and how to interpret domain-level performance gaps.

## Dependencies

- US-005 Validation Mesh — domain-specific validation thresholds may need domain-aware tuning.
- US-009 Grounded Reasoning and Generation — sentence-level support rate measures claim grounding.
- US-011 APEX-Eval Framework — TRACe adherence and relevance extend the existing metrics set.
