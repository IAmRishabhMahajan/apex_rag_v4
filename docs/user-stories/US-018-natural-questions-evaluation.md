# US-018: Natural Questions Evaluation

## User Story

As a system evaluator, I want to run APEX-RAG against the Natural Questions benchmark so that I can measure open-domain retrieval and answer accuracy on real Google search queries backed by Wikipedia, establishing a strong baseline for factual question answering.

## Scope

Integrate Google's Natural Questions (NQ) dataset into the APEX-RAG evaluation harness. NQ contains 300,000+ questions derived from real user Google searches, each paired with a Wikipedia article and two annotation tiers: long-answer candidates (paragraph-level) and short answers (span-level, including yes/no). NQ is the standard baseline for open-domain retrieval, used extensively in RAG research for measuring factual precision.

## Implementation Tasks

- Load the Natural Questions dataset from Hugging Face (`google-research-datasets/natural_questions`) using the validation split (7,830 examples) for evaluation.
- For each question, supply the corresponding Wikipedia article passage as the primary evidence item to `run_pipeline()`.
- Evaluate short-answer extraction: compare pipeline answer text against gold short answers using Exact Match (EM) and F1.
- Evaluate long-answer selection: check whether the pipeline's evidence bundle includes the gold long-answer paragraph.
- Handle yes/no questions as a special case: verify pipeline produces a definitive yes or no in the answer text rather than a hedged response.
- Count questions with no short answer (`short_answers = []`) and verify the pipeline surfaces `has_limitations=True` for those.
- Aggregate per-question results into an APEX-Eval AggregateReport and emit JSON for regression tracking.
- Produce a breakdown by answer type: short, long, yes/no, and no-answer.

## Acceptance Criteria

- The harness loads the NQ validation split without manual preprocessing.
- Short-answer EM and F1 scores are computed correctly against all annotator short answers (using max-annotator matching).
- Long-answer paragraph selection accuracy is computed and reported.
- Yes/no questions produce definitive single-word answers in at least 80% of cases.
- No-answer questions result in `has_limitations=True` in at least 80% of cases.
- Results are stratified by answer type (short, long, yes/no, no-answer).
- Evaluation covers the full validation set (7,830 questions) in a documented run time.
- APEX-Eval report includes an NQ-specific section with answer-type breakdowns.

## Testing Expectations

- Unit tests verify multi-annotator EM/F1 aggregation (max over annotators) against a known fixture.
- Unit tests verify yes/no answer classification from pipeline output text.
- Integration test runs 30 representative questions (split across answer types) end-to-end.
- Regression fixture captures expected short-answer EM on a 200-question sample.

## Documentation Updates

- Document how to load and cache the Natural Questions dataset locally.
- Document the multi-annotator evaluation protocol (max-over-annotators EM/F1).
- Document how answer-type breakdown results reveal specific pipeline weaknesses.

## Dependencies

- US-001 Query Intelligence — factual NQ queries should detect FACTUAL intent.
- US-009 Grounded Reasoning and Generation — short-answer extraction is the primary measure of generation grounding.
- US-011 APEX-Eval Framework — reuses claim and final metrics; adds short/long answer type evaluation.
