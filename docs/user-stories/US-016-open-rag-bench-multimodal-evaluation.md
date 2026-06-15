# US-016: Open RAG Bench Multimodal Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against Open RAG Bench so that I can measure retrieval and answer quality on PDF documents containing text, tables, and images, testing the pipeline's ability to handle multimodal enterprise-style sources.

## Scope

Integrate Open RAG Bench (Vectara) into the APEX-RAG evaluation harness. The benchmark comprises 1,000 arXiv papers (~400 positive documents and 600 hard negatives) with 3,045 QA pairs across four modality types: text-only (1,914), text-image (763), text-table (148), and text-table-image (220). Hard negative mining makes retrieval particularly challenging, closely approximating production-grade retrieval difficulty.

## Implementation Tasks

- Download the Open RAG Bench dataset from Hugging Face (`vectara/open_ragbench`) and parse the BEIR-format corpus, queries, qrels, and answers files.
- Map each query to its positive and hard-negative documents using the qrels relevance file.
- Supply positive and hard-negative documents as mixed evidence items to `run_pipeline()` to test retrieval discrimination.
- Evaluate retrieval precision by checking whether the pipeline surfaces positive documents over hard negatives.
- Evaluate answer quality by comparing pipeline text against reference answers for text-only and abstractive queries using ROUGE-L and F1.
- Stratify results by modality type (text-only, text-image, text-table, text-table-image) and document section type.
- Flag table- and image-referencing queries where the pipeline cannot access visual content as a known limitation.
- Aggregate results into an APEX-Eval AggregateReport and produce a modality-stratified breakdown.

## Acceptance Criteria

- The harness correctly loads all 3,045 QA pairs and the 1,000-document corpus from the BEIR format.
- Retrieval precision against hard negatives is computed and reported per query.
- Answer quality (ROUGE-L and F1) is computed for all text-answerable queries.
- Multimodal queries (image, table) are flagged and counted as a coverage limitation.
- Results are stratified by the four modality types (text-only, text-image, text-table, text-table-image).
- Hard-negative retrieval discrimination score (positive-ranked-above-negative rate) is reported.
- Evaluation covers all 3,045 QA pairs in a documented run time.
- APEX-Eval report includes an Open RAG Bench section with modality-specific scores.

## Testing Expectations

- Unit tests verify BEIR qrels parsing and positive/hard-negative document mapping.
- Unit tests verify ROUGE-L computation against a known reference fixture.
- Integration test runs 30 text-only questions end-to-end and asserts positive retrieval rate above 50%.
- Regression fixture captures expected precision scores on a 100-question sample.

## Documentation Updates

- Document the BEIR corpus format and how to load it using the `datasets` library.
- Document how table and image query types are handled (flagged vs. skipped).
- Document the hard-negative discrimination metric and how to interpret low scores.

## Dependencies

- US-007 Evidence Scoring and Gap Detection — hard negatives test the gap detector's ability to identify irrelevant evidence.
- US-008 Retrieval Repair Loop — hard-negative bundles are expected to trigger low-confidence scores and repair attempts.
- US-011 APEX-Eval Framework — extends with ROUGE-L answer quality and modality coverage metrics.
