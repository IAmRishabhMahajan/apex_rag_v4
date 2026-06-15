# US-017: EnterpriseRAG-Bench Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against EnterpriseRAG-Bench so that I can measure pipeline performance on realistic internal business documents including Slack threads, emails, meeting transcripts, and project tickets.

## Scope

Integrate EnterpriseRAG-Bench (Onyx) into the APEX-RAG evaluation harness. The benchmark contains 500+ questions across 10 difficulty categories and ~500,000 documents from nine enterprise document sources: Slack (275K), Gmail (120K), Linear (35K), Google Drive (25K), HubSpot (15K), Fireflies (10K), GitHub (8K), Jira (6K), and Confluence (5K). The question categories range from simple fact retrieval to multi-document conflict resolution and unavailable-information detection, mirroring real internal knowledge-base search scenarios.

## Implementation Tasks

- Download EnterpriseRAG-Bench from the latest GitHub release or Hugging Face and parse the question and document sets.
- Map each question to its relevant document subset by source type (Slack, Gmail, etc.) and supply as raw evidence items to `run_pipeline()`.
- Evaluate answer correctness using the `answer_evaluation` scripts from the repository for each of the 10 question categories.
- Test multi-document reasoning questions that require synthesizing information from two or more document sources of different types.
- Test conflict-resolution questions where documents contain contradictory information, verifying that the pipeline surfaces `conflict_count > 0` and appropriate limitations.
- Test unavailable-information questions where the correct answer is that no document contains the answer, verifying `has_limitations=True` responses.
- Stratify results by document source type (9 categories) and question difficulty category (10 categories).
- Aggregate results into an APEX-Eval AggregateReport with enterprise-specific breakdowns.

## Acceptance Criteria

- All 500+ questions load and run through the pipeline without schema errors.
- Per-question correctness scores are computed using the repository's `answer_evaluation` scripts.
- Multi-document questions are correctly identified and the complex reasoning path is activated.
- Conflict-resolution questions result in `conflict_count > 0` in the evidence bundle.
- Unavailable-information questions produce `has_limitations=True` in at least 75% of cases.
- Results are stratified by document source and question category.
- APEX-Eval report includes an enterprise-specific breakdown section.
- Evaluation completes for all 500+ questions in a documented run time.

## Testing Expectations

- Unit tests verify conflict detection triggers correctly when contradictory Slack and email evidence is supplied.
- Unit tests verify unavailable-information responses surface has_limitations without fabricating answers.
- Integration test runs 20 questions (2 per category) end-to-end using a 5-source document subset.
- Regression fixture captures correctness scores on a 50-question stratified sample.

## Documentation Updates

- Document how to download and set up the EnterpriseRAG-Bench dataset.
- Document how each of the 10 question categories maps to APEX-RAG pipeline stages.
- Document how conflict-resolution and unavailability detection results are interpreted.

## Dependencies

- US-004 Evidence Fusion — cross-source evidence fusion is central to multi-document enterprise queries.
- US-006 Complex Query Reasoning Path — multi-document synthesis questions activate the reasoning path.
- US-010 Risk Assessment, Critique, and Verification — conflict and unavailability require escalation safeguards.
- US-011 APEX-Eval Framework — reuses standard metrics; adds enterprise-category stratification.
